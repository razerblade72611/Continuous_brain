#!/usr/bin/env python3
"""CB2 r3-H sealed acceptance (exploration lane): the preregistered, one-shot test of candidate r3h-B on 90001-90040.

Preregistration: evidence/explore/cb2/r3h-sealed-prereg.json (committed before any sealed world is opened; signed
off by Sol and approved by David on DAV-96).  This script only executes it:

  1. integrity   the candidate restores to the declared state hash; its installed weights and event are the declared
                 ones (no retraining, no checkpoint selection, no tuning after results);
  2. evaluate    cb2_evaluate.py evaluate, unchanged protocol, on the preregistered seeds, paired with the routine,
                 with per-life delegation telemetry;
  3. continuity  save -> restore keeps the state hash; a consolidation re-installing the same weights keeps every
                 non-skill channel, the tick, the life and the weights hash and chains to the declared event; the
                 preregistered replay world, lived again by that restored + consolidated copy, gives identical
                 actions; delegation provenance covers 100% of executed primitive actions;
  4. verdict     the four gates, with raw request counts beside the ratio.

Any crash, missing world or deviation is written to the report as such.  Nothing is rerun silently.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import git_rev, vision  # noqa: E402
from cb2_evaluate import load_individual, parse_seeds  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def integrity(pre: dict, home: Path, v) -> dict:
    ind, meta = load_individual(home, v)                       # refuses if the snapshot != its recorded hash
    c = pre["candidate"]
    wfile = home / "weights" / f"{c['weights_sha256']}.pt"
    out = {"state_hash": meta["state_hash"], "weights_sha256": meta.get("weights_sha256"),
           "event_sha": meta.get("event_sha"), "weights_file_sha256": _sha(wfile) if wfile.exists() else None,
           "consolidations": meta.get("consolidations")}
    out["ok"] = (out["state_hash"] == c["state_hash"] and out["weights_sha256"] == c["weights_sha256"]
                 and out["event_sha"] == c["event_sha"] and out["weights_file_sha256"] == c["weights_sha256"])
    return out


def continuity(pre: dict, home: Path, v, eval_rep: dict, telemetry: Path) -> dict:
    from continuous_brain_cb2.consolidation import install
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual import codec
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_individual.skills import ACTIONS
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    c = pre["candidate"]
    out = {}
    ind, meta = load_individual(home, v)
    new = lambda: Individual(IndividualConfig(lifelong=True, skills="cb2"), ind.vision,   # its own organ
                             LanguageOrgan.from_config({"backend": "dev"}))
    # (a) save -> restore keeps the declared state hash
    saved = copy.deepcopy(ind.save())
    back = new()
    back.restore(copy.deepcopy(saved))
    back.life = ind.life
    out["save_restore_state_hash"] = back.state_hash()
    out["save_restore_ok"] = back.state_hash() == c["state_hash"]
    # (b) consolidation round-trip: re-install the same weights into a fresh individual
    weights = (home / "weights" / f"{c['weights_sha256']}.pt").read_bytes()
    before = {n: np.array(back.rt.store.read(n)) for n in back.rt.store.names() if not n.startswith("skills.")}
    dst = new()
    ev = install(back, dst, weights, {"note": "r3h sealed continuity check (same weights)"})
    changed = [n for n, val in before.items() if not np.array_equal(np.array(dst.rt.store.read(n)), val)]
    lineage = dst.read_state("skills")["lineage"]
    out["consolidation"] = {"kind": ev["kind"], "weights_sha256": ev["weights_sha256"],
                            "parent_state_hash": ev["parent_state_hash"], "prev_event_sha": ev["prev_event_sha"],
                            "non_skill_channels": len(before), "non_skill_channels_changed": changed,
                            "tick_kept": dst.rt.tick == back.rt.tick, "life_kept": dst.life == back.life,
                            "lineage_events": len(lineage)}
    out["consolidation_ok"] = (ev["kind"] == "install" and ev["weights_sha256"] == c["weights_sha256"]
                               and ev["parent_state_hash"] == c["state_hash"] and ev["prev_event_sha"] == c["event_sha"]
                               and not changed and dst.rt.tick == back.rt.tick and dst.life == back.life)
    # (c) the preregistered replay world, lived again by the restored + consolidated copy: identical actions
    seeds = parse_seeds(pre["seeds"])
    rs = int(pre["replay_seed"])
    idx = seeds.index(rs)
    re_saved = copy.deepcopy(dst.save())
    test = new()
    test.restore(re_saved)
    test.life = int(meta["life"]) + 1 + idx                    # as the evaluation numbered that world's life
    taken = []

    def on_step(n, action, *_):
        taken.append(int(action))
    tmp = Path(tempfile.mkdtemp(prefix="r3h_replay"))
    from continuous_brain_cb2.archive import StreamWriter
    w = StreamWriter(tmp, "replay", "PRACTICE", "copy-of-replay", {})
    record_life(test, W1Env(W1Config()), rs, w, steps=int(pre["steps"]), who="SELF", lived=False, on_step=on_step)
    w.close()
    shutil.rmtree(tmp, ignore_errors=True)
    orig = [json.loads(l)["taken"] for l in open(telemetry / f"delegation-{rs}.jsonl")]
    out["replay"] = {"seed": rs, "steps_eval": len(orig), "steps_replay": len(taken),
                     "first_difference": next((i for i, (x, y) in enumerate(zip(orig, taken)) if x != y), None)}
    out["replay_ok"] = orig == taken
    # (d) delegation provenance covers 100% of executed primitive actions, in every world
    per, bad = {}, []
    for e in eval_rep["episodes"]["individual"]:
        f = telemetry / f"delegation-{e['seed']}.jsonl"
        rows = [json.loads(l) for l in open(f)] if f.exists() else []
        okrows = sum(1 for r in rows if r.get("provenance") == "scripted-executor-organ" and ACTIONS[r["action"]] == r["taken"])
        per[e["seed"]] = {"steps": e["steps"], "logged": len(rows), "with_provenance": okrows}
        if not (len(rows) == e["steps"] == okrows):
            bad.append(e["seed"])
    out["delegation"] = {"worlds": len(per), "steps": sum(p["steps"] for p in per.values()),
                         "logged_with_provenance": sum(p["with_provenance"] for p in per.values()), "worlds_failing": bad}
    out["delegation_ok"] = not bad and len(per) == len(seeds)
    out["ok"] = out["save_restore_ok"] and out["consolidation_ok"] and out["replay_ok"] and out["delegation_ok"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", default=str(REPO / "evidence/explore/cb2/r3h-sealed-prereg.json"))
    ap.add_argument("--candidate", required=True, help="the candidate's individuals directory (holds cb2-ind-0001)")
    ap.add_argument("--work", required=True, help="new directory for the evaluation archive and telemetry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rehearsal-seeds", default=None,
                    help="REHEARSAL ONLY: run the machinery on development seeds instead of the sealed ones")
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    if a.rehearsal_seeds:
        pre = {**pre, "seeds": a.rehearsal_seeds, "replay_seed": int(parse_seeds(a.rehearsal_seeds)[0]),
               "label": pre["label"] + "-rehearsal", "REHEARSAL": True}
    work = Path(a.work)
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty: the sealed test runs once, into a new directory")
    work.mkdir(parents=True, exist_ok=True)
    home = Path(a.candidate) / pre["candidate"]["individual"]
    rep = {"format": "cb2-r3h-sealed/1", "status": "ENGINEERING_ONLY", "prereg": pre,
           "prereg_sha256": _sha(Path(a.prereg)), **git_rev(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "deviations": []}
    out = Path(a.out)

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    v = vision()
    try:
        rep["integrity"] = integrity(pre, home, v)
        save()
        if not rep["integrity"]["ok"]:
            rep["verdict"] = "NOT RUN: the candidate is not the preregistered artifact"
            save()
            print(json.dumps(rep["integrity"], indent=1))
            return 1
        tel = work / "telemetry"
        ev_out = work / f"eval-{pre['label']}.json"
        cmd = [sys.executable, str(REPO / "scripts/explore/cb2_evaluate.py"), "evaluate", "--individuals", str(a.candidate),
               "--individual", pre["candidate"]["individual"], "--label", pre["label"], "--seeds", pre["seeds"],
               "--steps", str(pre["steps"]), "--archive", str(work / "archive"), "--out", str(ev_out),
               "--telemetry", str(tel)]
        rep["evaluate_cmd"] = cmd
        save()
        p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
        (work / "evaluate.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        if p.returncode != 0:
            rep["deviations"].append(f"evaluation exited {p.returncode} (see evaluate.log); NOT rerun")
            rep["verdict"] = "INCOMPLETE"
            save()
            return 1
        ev = json.loads(ev_out.read_text())
        s, rt = ev["summary"]["individual"], ev["summary"]["routine"]
        seeds = parse_seeds(pre["seeds"])
        if len(ev["episodes"]["individual"]) != len(seeds) or len(ev["episodes"]["routine"]) != len(seeds):
            rep["deviations"].append("missing worlds in the evaluation")
        ratio = {m: s[m] / rt[m] for m in ("crafter_score", "survival_steps", "achievements_per_episode")}
        ob_c, ob_r = s["obligations_on_time"], rt["obligations_on_time"]
        req_ratio = (ob_c["rate"] / ob_r["rate"]) if ob_c["rate"] is not None and ob_r["rate"] else None
        rep["evaluation"] = {"candidate": s, "routine": rt, "ratio_to_routine": {k: round(x, 4) for k, x in ratio.items()},
                             "requests": {"candidate_on_time": ob_c["fulfilled"], "candidate_eligible": ob_c["fulfilled"] + ob_c["failed"],
                                          "routine_on_time": ob_r["fulfilled"], "routine_eligible": ob_r["fulfilled"] + ob_r["failed"],
                                          "candidate_rate": ob_c["rate"], "routine_rate": ob_r["rate"],
                                          "ratio": None if req_ratio is None else round(req_ratio, 4)},
                             "paired_difference": ev["paired_difference"],
                             "candidate_outlived_routine": sum(x["steps"] > y["steps"] for x, y in
                                                               zip(ev["episodes"]["individual"], ev["episodes"]["routine"]))}
        save()
        rep["continuity"] = continuity(pre, home, v, ev, tel)
        g = pre["gates"]
        gates = {"score": ratio["crafter_score"] >= g["score_ratio_min"],
                 "survival": ratio["survival_steps"] >= g["survival_ratio_min"],
                 "requests": req_ratio is not None and req_ratio >= g["requests_ratio_min"],
                 "continuity": bool(rep["continuity"]["ok"])}
        rep["gates"] = gates
        rep["verdict"] = ("PASS" if all(gates.values()) and not rep["deviations"] else "FAIL") + \
                         (" (REHEARSAL on development seeds: not the acceptance test)" if pre.get("REHEARSAL") else "")
    except Exception:
        rep["deviations"].append("crash: " + traceback.format_exc())
        rep["verdict"] = "INCOMPLETE (crash, reported; not rerun)"
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(json.dumps({k: rep.get(k) for k in ("verdict", "gates", "deviations")}, indent=1))
    print(json.dumps(rep.get("evaluation", {}), indent=1))
    return 0 if rep["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
