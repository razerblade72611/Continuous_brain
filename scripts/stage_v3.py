"""Gate v3 Stage A (screening) launcher: runs the agreed config sweep in three phases, N jobs at a time, applies the
agreed selection rule, prints a compact summary, and can stop the RunPod pod when done.

  python scripts/stage_v3.py A  [--parallel 6] [--lives 25] [--seeds 0,1] [--stop-pod]
  python scripts/stage_v3.py A0 [--parallel 6] [--stop-pod]      # 5-life early-retention triage (A2-A0), no auto-advance

Runner repairs (Claude + Sol, 2026-10-25, after the A2 OOM / lost-pod incident; source baseline 95cad33):
  R1 a worker that exits non-zero or leaves no complete result stops the stage before any scoring (was: crash later)
  R2 a "complete" result is reused only if it matches this run: lives, seeds, config, backbone hash, and experiment
     source unchanged since the result's commit; otherwise it is set aside as .mismatch.<ts>.json (never overwritten)
  R3 a retried job's previous log is kept as .log.<ts> instead of being overwritten
  R4 --stop-pod stops the pod only if results, archive and checksum are all on the identified network-volume mount
     (CB_PERSIST_DIR, default /workspace) and the archive re-reads with the same checksum
  R5 preflight: refuse to run on uncommitted experiment source or a backbone whose hash is not the locked one;
     after the first worker failure no new workers are started

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


EXPERIMENT_SOURCES = ["being", "scripts/run_v3_dev.py"]   # what a reusable result must have been produced by
# Qwen3-0.6B backbone hash used by every Gate v3 result so far (first Stage A, smoke runs). A run whose backbone differs
# is stopped by the preflight check unless --accept-backbone <hash> is given explicitly (and recorded in the summary).
EXPECTED_BACKBONE = "c75424fb5370f0d514a05d7c640e6c2e1e4b504564e4d48cb2ddedb71609e0a0"
PERSIST_DIR = Path(os.environ.get("CB_PERSIST_DIR", "/workspace"))   # the pod's network-volume mount


def _norm(v):
    if isinstance(v, bool):
        return str(v).lower()
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v).lower()


def _source_unchanged_since(commit):
    """True if the experiment source is identical at `commit` and HEAD. Unknown commit -> False (needs full history)."""
    if not commit:
        return False
    r = subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--"] + EXPERIMENT_SOURCES, cwd=ROOT,
                       capture_output=True)
    return r.returncode == 0


def result_problem(p, arm, cfg, lives, seeds, backbone):
    """None if the result file is a complete, correctly bound result for this job; else a reason string."""
    try:
        d = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({type(e).__name__})"
    if d.get("status") != "complete":
        return f"status {d.get('status')}"
    if d.get("config", {}).get("arm") != arm:
        return f"arm {d.get('config', {}).get('arm')} != {arm}"
    if d.get("lives") != lives:
        return f"lives {d.get('lives')} != {lives}"
    got_seeds = sorted(r.get("probe_seed") for r in d.get("per_seed", []))
    if got_seeds != sorted(int(x) for x in seeds.split(",")):
        return f"seeds {got_seeds} != {seeds}"
    for k, v in cfg.items():
        if _norm(d.get("config", {}).get(k)) != _norm(v):
            return f"config {k}={d.get('config', {}).get(k)} != {v}"
    if d.get("backbone_sha256") != backbone:
        return f"backbone {str(d.get('backbone_sha256'))[:12]} != expected {backbone[:12]}"
    for r in d.get("per_seed", []):
        if r.get("backbone_start") != r.get("backbone_end"):
            return "backbone changed during the run"
    if not _source_unchanged_since(d.get("commit")):
        return f"experiment source changed since commit {str(d.get('commit'))[:7]}"
    return None


def run_jobs(jobs, parallel, lives, seeds, out_dir, backbone=EXPECTED_BACKBONE):
    """jobs: list of (arm, cfg dict). Runs those without a valid complete result.
    Returns ({name: path}, {name: failure reason}) -- callers must not score anything if failures is non-empty."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pending, running, paths, failures = list(jobs), [], {}, {}
    cfgs = {job_name(arm, cfg): (arm, cfg) for arm, cfg in jobs}
    for arm, cfg in jobs:
        paths[job_name(arm, cfg)] = out_dir / f"{job_name(arm, cfg)}.json"
    while pending or running:
        while pending and len(running) < parallel:
            arm, cfg = pending.pop(0)
            p = paths[job_name(arm, cfg)]
            if p.exists():
                why = result_problem(p, arm, cfg, lives, seeds, backbone)
                if why is None:
                    print(time.strftime("%H:%M:%S"), "reuse", p.name, flush=True)
                    continue
                tag = "interrupted" if why.startswith("status") else "mismatch"
                kept = p.with_suffix(f".{tag}.{int(time.time())}.json")
                p.rename(kept)                                   # kept on record, never overwritten
                print(time.strftime("%H:%M:%S"), "set aside", kept.name, "--", why, flush=True)
            lp = p.with_suffix(".log")
            if lp.exists():
                lp.rename(p.with_suffix(f".log.{int(time.time())}"))   # keep the previous attempt's log (R3)
            cmd = [sys.executable, str(ROOT / "scripts/run_v3_dev.py"), "--arm", arm, "--seeds", seeds,
                   "--lives", str(lives), "--out", str(p.relative_to(ROOT)), "--set"] + [f"{k}={v}" for k, v in cfg.items()]
            log = open(lp, "w")
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), p, log))
            print(time.strftime("%H:%M:%S"), "start", p.name, flush=True)
        for item in list(running):
            proc, p, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(time.strftime("%H:%M:%S"), "done ", p.name, "exit", proc.returncode, flush=True)
                name = p.stem
                if proc.returncode != 0:
                    failures[name] = f"exit {proc.returncode} (see {p.with_suffix('.log').name})"
                elif not p.exists():
                    failures[name] = "exited 0 but wrote no result"
                else:
                    why = result_problem(p, *cfgs[name], lives, seeds, backbone)
                    if why:
                        failures[name] = why
        if failures and pending:   # fail fast: do not launch more work into a broken environment (e.g. OOM)
            for arm, cfg in pending:
                failures[job_name(arm, cfg)] = "not started (stage stopping after an earlier failure)"
            pending = []
            print(time.strftime("%H:%M:%S"), "a job failed -- no new jobs will be started", flush=True)
        if pending or running:
            time.sleep(5)
    return paths, failures


def abort_on_failures(failures, phase, summary_path, summary, stop_pod, bundle_dir):
    if not failures:
        return
    summary["failed"] = {"phase": phase, "jobs": failures}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    print(f"STAGE STOPPED in {phase}: {len(failures)} job(s) failed -- nothing was scored.", flush=True)
    for n, why in failures.items():
        print("  FAILED", n, "--", why, flush=True)
    finish(bundle_dir, summary_path, stop_pod)
    sys.exit(2)


def _mount_of(path):
    """(mount point, source, fs type) of the filesystem holding `path`, from /proc/self/mountinfo."""
    path = Path(path).resolve()
    best = ("", "", "")
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mnt = left.split()[4]
            fstype, source = right.split()[:2]
            if (str(path) == mnt or str(path).startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best[0]):
                best = (mnt, source, fstype)
    except OSError:
        pass
    return best


def persistent_ok(paths, persist_dir=None):
    """True only if every path lies under persist_dir and persist_dir is its own network-filesystem mount."""
    persist_dir = Path(persist_dir or PERSIST_DIR).resolve()
    mnt, source, fstype = _mount_of(persist_dir)
    network = fstype.startswith("fuse") or fstype in ("nfs", "nfs4", "ceph", "cifs", "smb3")
    is_mount = mnt == str(persist_dir)
    inside = all(str(Path(x).resolve()).startswith(str(persist_dir) + "/") for x in paths)
    same = all(_mount_of(x)[0] == mnt for x in paths)
    return (network and is_mount and inside and same), {"persist_dir": str(persist_dir), "mount": mnt, "source": source,
                                                       "fstype": fstype}


def finish(bundle_dir, summary_path, stop_pod):
    """Bundle + checksum the results. --stop-pod is honoured only if the results, the archive and its checksum are all
    on the pod's identified network-volume mount (PERSIST_DIR) and the archive re-reads with the same checksum (R4)."""
    import hashlib
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tar = ROOT / f"results_{bundle_dir.name}.tar.gz"
    rel = [str(bundle_dir.relative_to(ROOT))] + ([str(summary_path.relative_to(ROOT))] if summary_path.exists() else [])
    subprocess.run(["tar", "czf", str(tar), "-C", str(ROOT)] + rel, check=True)
    with open(tar, "rb") as fh:
        os.fsync(fh.fileno())
    digest = hashlib.sha256(tar.read_bytes()).hexdigest()
    side = tar.parent / (tar.name + ".sha256")
    side.write_text(f"{digest}  {tar.name}\n")
    reread = hashlib.sha256(Path(tar).read_bytes()).hexdigest()
    listed = subprocess.run(["tar", "tzf", str(tar)], capture_output=True, text=True)
    persisted, where = persistent_ok([bundle_dir, tar, side] + ([summary_path] if summary_path.exists() else []))
    ok = listed.returncode == 0 and reread == digest and persisted
    print(f"BUNDLE {tar} sha256={digest} files={len(listed.stdout.splitlines())} reread_ok={reread == digest} "
          f"persistent={persisted} {json.dumps(where)}", flush=True)
    if stop_pod and os.environ.get("RUNPOD_POD_ID"):
        if ok:
            subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])
        else:
            print("NOT stopping the pod: results/archive not verified on the network volume. Copy them off first.",
                  flush=True)
    return ok


def score(path):
    """Amendment C (Grok, DAV-105): a config qualifies only if WITHIN-LIFE acquisition (mean of life-1 template at
    life 1 and life-10 template at life 10) is >= 6/10; then it is ranked by the life-1 SPECIFIC margin at the last
    checkpoint, within the competence band. Margin = logprob(true) - logprob(strongest same-pool distractor) when
    recorded (margin_best; > 0 exactly when forced choice is right), else vs the distractor mean."""
    if not path.exists():
        return None
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


OUT_A0 = ROOT / "results/dev3/stageA2_A0"

A1_CONFIGS = [("L", {"lr": lr, "kl_weight": kl, "night_passes": 4}) for lr in (1e-4, 3e-4) for kl in (0.3, 1.0)] + \
             [("T", {"layer": l, "lr": 1e-2, "night_passes": 4}) for l in (7, 14, 21)] + \
             [("S", {"layer": l, "lr": 100, "top_t": 64, "s_fit": "true", "night_passes": 4}) for l in (7, 14, 21)]


def a0_metrics(path):
    """A2-A0 measures (frozen 2026-10-25, Claude + Sol). Acquisition = life-1 template recall measured at checkpoint 1
    (NOT the A2 selector's (a1+a10)/2 score; no a10=a1 fallback). Retention = life-1 forced choice at checkpoint 5.
    Competence = both monitors at checkpoint 5: mean ppl ratio <= 1.02 and no seed > 1.05."""
    d = json.loads(path.read_text())
    base, seeds, problems = d["base_ppl_loss"], {}, []
    ratios = {k: [] for k in base}
    for r in d["per_seed"]:
        cp = {c["after_life"]: c for c in r["checkpoints"]}
        if not {1, 2, 3, 5} <= set(cp):
            problems.append(f"seed {r.get('probe_seed')}: checkpoints {sorted(cp)}")
            continue
        if "life1" not in r.get("wiped", {}):
            problems.append(f"seed {r.get('probe_seed')}: no wiped baseline")
        if r.get("backbone_start") != r.get("backbone_end"):
            problems.append(f"seed {r.get('probe_seed')}: backbone changed")
        acq = sum(bool(x["template"]) for x in cp[1]["life1"])
        fc = {L: sum(bool(x.get("fc")) for x in cp[L]["life1"]) for L in (1, 2, 3, 5)}
        mb = [x.get("margin_best") for x in cp[5]["life1"]]
        vals = [v for v in mb if v is not None] + [cp[5]["ppl_loss"][k] for k in base]
        if any(not math.isfinite(v) for v in vals):
            problems.append(f"seed {r.get('probe_seed')}: non-finite metric")
        for k in base:
            ratios[k].append(math.exp(cp[5]["ppl_loss"][k] - base[k]))
        seeds[r["probe_seed"]] = {"acq_l1": acq, "fc": fc,
                                  "margin_best_l5": (sum(v for v in mb if v is not None) / max(1, len([v for v in mb if v is not None]))),
                                  "wiped_template_l1": sum(bool(x["template"]) for x in r.get("wiped", {}).get("life1", []))}
    band = bool(seeds) and all(sum(v) / len(v) <= 1.02 and max(v) <= 1.05 for v in ratios.values() if v)
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return {"seeds": seeds, "problems": problems, "band": band,
            "acq_mean": mean([v["acq_l1"] for v in seeds.values()]),
            "ret_mean": mean([v["fc"][5] for v in seeds.values()]),
            "ret_min_seed": min([v["fc"][5] for v in seeds.values()], default=0),
            "ret_max_seed": max([v["fc"][5] for v in seeds.values()], default=0),
            "ppl_ratio": {k: round(mean(v), 4) for k, v in ratios.items()},
            "ppl_ratio_max": {k: round(max(v), 4) if v else None for k, v in ratios.items()}}


def a0_verdict(metrics, expected_seeds):
    """Frozen decision rule. GO: some config acquires (>=6/10 mean at life 1), retains >=3/10 mean and >=2/10 in each seed
    at life 5, and is in the competence band. STOP: at least one config acquires; every acquiring config that is in the
    band is <=1/10 in each seed at life 5; and no other in-band config reaches >=2/10 mean retention.
    INCONCLUSIVE: anything else, including missing/invalid data. GO authorises A2 Phase 1 at 25 lives only."""
    invalid = {n: m["problems"] or ["seed coverage"] for n, m in metrics.items()
               if m["problems"] or sorted(m["seeds"]) != sorted(expected_seeds)}
    if invalid or len(metrics) != len(A1_CONFIGS):
        return "INCONCLUSIVE", {"reason": "validity", "invalid": invalid, "configs": len(metrics)}
    go = [n for n, m in metrics.items() if m["acq_mean"] >= 6 and m["ret_mean"] >= 3 and m["ret_min_seed"] >= 2 and m["band"]]
    if go:
        return "GO", {"qualifying": go}
    acquirers = [n for n, m in metrics.items() if m["acq_mean"] >= 6]
    competent_acq = [n for n in acquirers if metrics[n]["band"]]
    others_retain = [n for n, m in metrics.items() if n not in competent_acq and m["band"] and m["ret_mean"] >= 2]
    # clarified before any A0 data (2026-10-25): STOP needs at least one acquiring config INSIDE the band; an
    # acquirer that retains but breaks competence is not grounds to stop the search -> INCONCLUSIVE
    if competent_acq and all(metrics[n]["ret_max_seed"] <= 1 for n in competent_acq) and not others_retain:
        return "STOP", {"acquirers": acquirers, "competent_acquirers": competent_acq}
    return "INCONCLUSIVE", {"acquirers": acquirers, "competent_acquirers": competent_acq, "others_retaining": others_retain}


def stage_a0(a):
    if a.lives != 25 or a.seeds != "0,1":
        sys.exit("A0 is frozen at 5 lives and seeds 0,1; do not pass --lives/--seeds")
    lives, seeds = 5, "0,1"
    t0 = time.time()
    spath = ROOT / "results/dev3/stageA2_A0_summary.json"
    summary = {"stage": "A0", "protocol": "A2-A0 early-retention triage, frozen 2026-10-25 (Claude + Sol)",
               "lives": lives, "seeds": seeds, "backbone": a.backbone, "backbone_accepted_explicitly": bool(a.accept_backbone)}
    paths, fails = run_jobs(A1_CONFIGS, a.parallel, lives, seeds, OUT_A0, a.backbone)
    abort_on_failures(fails, "A0", spath, summary, a.stop_pod, OUT_A0)
    metrics = {n: a0_metrics(p) for n, p in paths.items()}
    verdict, why = a0_verdict(metrics, [0, 1])
    summary.update({"verdict": verdict, "why": why, "metrics": metrics, "hours": round((time.time() - t0) / 3600, 2)})
    spath.write_text(json.dumps(summary, indent=1, default=str))
    print("A0 VERDICT", verdict, json.dumps(why, default=str), flush=True)
    for n, m in metrics.items():
        print(f"  {n:58s} acq {m['acq_mean']:.1f}  ret@5 {m['ret_mean']:.1f} (min {m['ret_min_seed']})  band {m['band']}  "
              f"ppl {m['ppl_ratio']}", flush=True)
    finish(OUT_A0, spath, a.stop_pod)


def check_clean_source():
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + EXPERIMENT_SOURCES +
                           ["scripts/stage_v3.py"], cwd=ROOT, capture_output=True, text=True)
    if dirty.returncode != 0 or dirty.stdout.strip():
        sys.exit("refusing to run: experiment source has uncommitted changes or this is not a git checkout\n"
                 + dirty.stdout + dirty.stderr)


def preflight_backbone(accept=None):
    """Load the backbone once (CPU is fine) and compare its hash with the locked one before any GPU work is queued."""
    code = ("import sys; sys.path.insert(0, %r); from being.v3.core import load_qwen, backbone_hash; "
            "m, _ = load_qwen(device='cpu'); print(backbone_hash(m))" % str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    got = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
    want = accept or EXPECTED_BACKBONE
    if got != want:
        sys.exit(f"refusing to run: backbone hash {got} != expected {want}\n{r.stderr[-800:]}")
    print("BACKBONE OK", got, "(explicitly accepted)" if accept else "", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A", "A0"])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--lives", type=int, default=25)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--stop-pod", action="store_true")
    ap.add_argument("--accept-backbone", help="run with this backbone hash instead of EXPECTED_BACKBONE (recorded)")
    a = ap.parse_args()
    check_clean_source()
    a.backbone = preflight_backbone(a.accept_backbone)
    if a.stage == "A0":
        return stage_a0(a)
    t0 = time.time()
    summary = {"stage": "A", "lives": a.lives, "seeds": a.seeds, "phases": {}, "backbone": a.backbone,
               "backbone_accepted_explicitly": bool(a.accept_backbone)}
    spath = ROOT / "results/dev3/stageA2_summary.json"

    a1 = A1_CONFIGS   # L first: slowest
    p1, f1 = run_jobs(a1, a.parallel, a.lives, a.seeds, OUT, a.backbone)
    abort_on_failures(f1, "A1", spath, summary, a.stop_pod, OUT)
    winners = {}
    for arm in ("T", "S", "L"):
        best, rule, sc = select({n: p for n, p in p1.items() if n.startswith(arm + "_")})
        winners[arm] = (best, rule)
        summary["phases"].setdefault("A1", {})[arm] = {"best": best, "rule": rule, "scores": sc}
    tl = json.loads(p1[winners["T"][0]].read_text())["config"]["layer"] if winners["T"][0] else 14
    sl_ = json.loads(p1[winners["S"][0]].read_text())["config"]["layer"] if winners["S"][0] else 14
    a2 = [("T", {"layer": tl, "lr": lr, "night_passes": n}) for lr in (1e-2, 3e-2) for n in (2, 4)] + \
         [("S", {"layer": sl_, "lr": lr, "top_t": t, "s_fit": "true", "night_passes": 4}) for lr in (100, 300) for t in (64, 256)]
    p2, f2 = run_jobs(a2, a.parallel, a.lives, a.seeds, OUT, a.backbone)
    abort_on_failures(f2, "A2", spath, summary, a.stop_pod, OUT)
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
    p3, f3 = run_jobs(a3, a.parallel, a.lives, a.seeds, OUT, a.backbone)
    abort_on_failures(f3, "A3", spath, summary, a.stop_pod, OUT)
    gbest, grule, gsc = select({n: p for n, p in p3.items() if n.startswith("G_")})
    final["G"] = gbest
    summary["phases"]["A3"] = {"G_base": base, "G": {"best": gbest, "rule": grule, "scores": gsc},
                               "S_random": {n: score(p) for n, p in p3.items() if n.startswith("S_")}}
    summary["frozen_for_stage_B"] = {arm: json.loads({**p1, **p2, **p3}[n].read_text())["config"] if n else None
                                     for arm, n in final.items()}
    summary["hours"] = round((time.time() - t0) / 3600, 2)
    spath.write_text(json.dumps(summary, indent=1))
    compact = {arm: {"job": n, "rule": (summary["phases"].get("A2", {}).get(arm) or summary["phases"]["A1"].get(arm)
                                        or summary["phases"]["A3"].get(arm) or {}).get("rule")} for arm, n in final.items()}
    print("STAGE A DONE", json.dumps({"hours": summary["hours"], "G_base": base, "winners": compact,
                                      "scores": {n: {k: s[k] for k in ("acquisition", "margin_last", "fc_last", "recall_last", "band", "ppl_ratio")}
                                                 for ph in ("A1", "A2") for arm in summary["phases"].get(ph, {})
                                                 for n, s in summary["phases"][ph][arm]["scores"].items()}}, default=str))
    finish(OUT, spath, a.stop_pod)


if __name__ == "__main__":
    main()
