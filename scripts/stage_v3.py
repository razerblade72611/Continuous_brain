"""Gate v3 Stage A (screening) launcher: runs the agreed config sweep in three phases, N jobs at a time, applies the
agreed selection rule, prints a compact summary, and can stop the RunPod pod when done.

  python scripts/stage_v3.py A [--parallel 6] [--lives 25] [--seeds 0,1] [--stop-pod]

Plan (revised after the first Stage A, Claude-Grok amendments A-F on DAV-105; results go to results/dev3/stageA2):
  A1  L (AdamW, clip 1, KL every step): lr {1e-4,3e-4} x KL {0.3,1}   T: layer {7,14,21} (lr 1e-2, night 4)
      S-fitted: layer {7,14,21} (lr 100, top_t 64)
  A2  T at its best layer: lr {1e-2,3e-2} x night {2,4}   S at its best layer: lr {100,300} x top_t {64,256}
  A3  G on the better of T/S (its frozen config, names-only cue list): replay_cues {8,16}; S-random = S winner, s_fit false
Checkpoints after lives 1, 2, 3, 5, 10, 25. Primary measure: fact-specific forced choice among the attribute's pool and
the specific margin logprob(true) - mean logprob(distractors) (amendment A).
Selection (amendment C): within-life acquisition (mean of life-1 template at life 1 and life-10 template at life 10)
>= 6/10, within the competence band (mean ppl ratio <= 1.02, no seed > 1.05, both dev monitors, last checkpoint);
maximise the life-1 specific margin at the last checkpoint. Fallbacks, in order: acquires but none in band; in band but
none acquires; any. Ties: lower worst-monitor ppl ratio, then plan order. The fallback used is recorded.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dev3/stageA2"


def job_name(arm, cfg):
    return arm + "_" + "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))


def run_jobs(jobs, parallel, lives, seeds):
    """jobs: list of (arm, cfg dict). Runs those without a result file; returns {name: path}."""
    OUT.mkdir(parents=True, exist_ok=True)
    pending, running, paths = list(jobs), [], {}
    for arm, cfg in jobs:
        paths[job_name(arm, cfg)] = OUT / f"{job_name(arm, cfg)}.json"
    while pending or running:
        while pending and len(running) < parallel:
            arm, cfg = pending.pop(0)
            p = paths[job_name(arm, cfg)]
            if p.exists() and json.loads(p.read_text()).get("status") == "complete":
                continue
            if p.exists():
                p.rename(p.with_suffix(f".interrupted.{int(time.time())}.json"))   # kept on record, never overwritten
            cmd = [sys.executable, str(ROOT / "scripts/run_v3_dev.py"), "--arm", arm, "--seeds", seeds,
                   "--lives", str(lives), "--out", str(p.relative_to(ROOT)), "--set"] + [f"{k}={v}" for k, v in cfg.items()]
            log = open(p.with_suffix(".log"), "w")
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), p, log))
            print(time.strftime("%H:%M:%S"), "start", p.name, flush=True)
        for item in list(running):
            proc, p, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(time.strftime("%H:%M:%S"), "done ", p.name, "exit", proc.returncode, flush=True)
        time.sleep(5)
    return paths


def score(path):
    """Amendment C (Grok, DAV-105): a config qualifies only if WITHIN-LIFE acquisition (mean of life-1 template at
    life 1 and life-10 template at life 10) is >= 6/10; then it is ranked by the life-1 SPECIFIC margin at the last
    checkpoint, within the competence band. Margin = logprob(true) - logprob(strongest same-pool distractor) when
    recorded (margin_best; > 0 exactly when forced choice is right), else vs the distractor mean."""
    d = json.loads(path.read_text())
    if d.get("status") != "complete":
        return None
    base = d["base_ppl_loss"]
    acq, margin, fc, rec, ratios = [], [], [], [], {k: [] for k in base}
    for r in d["per_seed"]:
        cp = {c["after_life"]: c for c in r["checkpoints"]}
        last = r["checkpoints"][-1]
        a1 = sum(x["template"] for x in cp[1]["life1"])
        a10 = sum(x["template"] for x in cp[10]["life10"]) if 10 in cp else a1
        acq.append((a1 + a10) / 2)
        margin.append(sum(x.get("margin_best", x["margin"]) for x in last["life1"]) / len(last["life1"]))
        fc.append(sum(x["fc"] for x in last["life1"]))
        rec.append(sum(x["template"] for x in last["life1"]))
        for k in base:
            ratios[k].append(math.exp(last["ppl_loss"][k] - base[k]))
    band = all(sum(v) / len(v) <= 1.02 and max(v) <= 1.05 for v in ratios.values())
    mean = lambda v: sum(v) / len(v)
    return {"acquisition": mean(acq), "acquires": mean(acq) >= 6, "margin_last": mean(margin), "fc_last": mean(fc),
            "recall_last": mean(rec), "band": band,
            "ppl_ratio": {k: round(mean(v), 4) for k, v in ratios.items()},
            "ppl_ratio_max": {k: round(max(v), 4) for k, v in ratios.items()}, "config": d["config"]}


def select(paths):
    scored = {n: score(p) for n, p in paths.items()}
    scored = {n: s for n, s in scored.items() if s}
    for rule, ok in (("acquires+band", lambda s: s["acquires"] and s["band"]), ("acquires only (none in band)", lambda s: s["acquires"]),
                     ("band only (none acquires)", lambda s: s["band"]), ("none", lambda s: True)):
        c = {n: s for n, s in scored.items() if ok(s)}
        if c:
            best = max(c, key=lambda n: (c[n]["margin_last"], -max(c[n]["ppl_ratio"].values())))
            return best, rule, scored
    return None, "no complete runs", scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A"])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--lives", type=int, default=25)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--stop-pod", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    summary = {"stage": "A", "lives": a.lives, "seeds": a.seeds, "phases": {}}

    a1 = [("L", {"lr": lr, "kl_weight": kl, "night_passes": 4}) for lr in (1e-4, 3e-4) for kl in (0.3, 1.0)] + \
         [("T", {"layer": l, "lr": 1e-2, "night_passes": 4}) for l in (7, 14, 21)] + \
         [("S", {"layer": l, "lr": 100, "top_t": 64, "s_fit": "true", "night_passes": 4}) for l in (7, 14, 21)]   # L first: slowest
    p1 = run_jobs(a1, a.parallel, a.lives, a.seeds)
    winners = {}
    for arm in ("T", "S", "L"):
        best, rule, sc = select({n: p for n, p in p1.items() if n.startswith(arm + "_")})
        winners[arm] = (best, rule)
        summary["phases"].setdefault("A1", {})[arm] = {"best": best, "rule": rule, "scores": sc}
    tl = json.loads(p1[winners["T"][0]].read_text())["config"]["layer"] if winners["T"][0] else 14
    sl_ = json.loads(p1[winners["S"][0]].read_text())["config"]["layer"] if winners["S"][0] else 14
    a2 = [("T", {"layer": tl, "lr": lr, "night_passes": n}) for lr in (1e-2, 3e-2) for n in (2, 4)] + \
         [("S", {"layer": sl_, "lr": lr, "top_t": t, "s_fit": "true", "night_passes": 4}) for lr in (100, 300) for t in (64, 256)]
    p2 = run_jobs(a2, a.parallel, a.lives, a.seeds)
    final = {}
    for arm in ("T", "S"):
        pool = {n: p for n, p in {**p1, **p2}.items() if n.startswith(arm + "_")}
        best, rule, sc = select(pool)
        final[arm] = best
        summary["phases"].setdefault("A2", {})[arm] = {"best": best, "rule": rule, "scores": sc}
    final["L"] = winners["L"][0]
    # A3: G on the better of T/S, plus S-random ablation (S winner, s_fit false)
    sT, sS = (score({**p1, **p2}[final[x]]) if final[x] else None for x in ("T", "S"))
    key = lambda x: (x["acquires"], x["band"], x["margin_last"])
    base = "T" if (sT and (not sS or key(sT) >= key(sS))) else "S"
    bcfg = {k: v for k, v in json.loads({**p1, **p2}[final[base]].read_text())["config"].items()
            if k in ("layer", "lr", "night_passes", "top_t", "s_fit")}
    bcfg = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in bcfg.items()}
    a3 = [("G", {**bcfg, "g_base": base, "replay_cues": c, "g_gate": "true"}) for c in (8, 16)] + \
         [("G", {**bcfg, "g_base": base, "replay_cues": 16, "g_gate": "false"})]   # ungated ablation (design item 4)
    scfg = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in
            json.loads({**p1, **p2}[final["S"]].read_text())["config"].items()
            if k in ("layer", "lr", "night_passes", "top_t")} if final["S"] else None
    if scfg:
        a3.append(("S", {**scfg, "s_fit": "false"}))
    p3 = run_jobs(a3, a.parallel, a.lives, a.seeds)
    gbest, grule, gsc = select({n: p for n, p in p3.items() if n.startswith("G_")})
    final["G"] = gbest
    summary["phases"]["A3"] = {"G_base": base, "G": {"best": gbest, "rule": grule, "scores": gsc},
                               "S_random": {n: score(p) for n, p in p3.items() if n.startswith("S_")}}
    summary["frozen_for_stage_B"] = {arm: json.loads({**p1, **p2, **p3}[n].read_text())["config"] if n else None
                                     for arm, n in final.items()}
    summary["hours"] = round((time.time() - t0) / 3600, 2)
    (ROOT / "results/dev3/stageA2_summary.json").write_text(json.dumps(summary, indent=1))
    subprocess.run(["tar", "czf", str(ROOT / "results_stageA2.tar.gz"), "-C", str(ROOT), "results/dev3/stageA2"])
    compact = {arm: {"job": n, "rule": (summary["phases"].get("A2", {}).get(arm) or summary["phases"]["A1"].get(arm)
                                        or summary["phases"]["A3"].get(arm) or {}).get("rule")} for arm, n in final.items()}
    print("STAGE A DONE", json.dumps({"hours": summary["hours"], "G_base": base, "winners": compact,
                                      "scores": {n: {k: s[k] for k in ("acquisition", "margin_last", "fc_last", "recall_last", "band", "ppl_ratio")}
                                                 for ph in ("A1", "A2") for arm in summary["phases"].get(ph, {})
                                                 for n, s in summary["phases"][ph][arm]["scores"].items()}}, default=str))
    if a.stop_pod and os.environ.get("RUNPOD_POD_ID"):
        subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])


if __name__ == "__main__":
    main()
