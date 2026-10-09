#!/usr/bin/env python3
"""CB2-S2 sealed test (exploration lane): executes evidence/explore/cb2/s2-prereg.json's sealed part once.

  python scripts/explore/cb2_sealed_s2.py --candidate <dir: copy of r3h-B with the selected S2 selector installed> \\
      --comparator <frozen r3h-B dir> --selection <s2 run>/selection.json --work <new dir>

The S2 candidate runs through cb2_sealed_r3h.py (integrity, the unchanged evaluation paired with the routine,
continuity); frozen r3h-B through cb2_evaluate.py on the same worlds at the same time.  Verdicts by the prereg:
S2 complete (beyond the routine), S2-progress (beyond r3h-B), or no improvement.  Reported, never rerun.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def lower95(d: np.ndarray) -> float:
    rng = np.random.default_rng(20261009)
    return float(np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), 5))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=str(REPO / "evidence/explore/cb2/s2-prereg.json"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/s2-sealed-report.json"))
    ap.add_argument("--rehearsal-seeds", default=None)
    ap.add_argument("--comparator-name", default="r3h-B", help="the frozen comparator (r3h-B for S2/S2b, S2b for S2c)")
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    W = Path(a.work)
    if W.exists():
        raise SystemExit(f"{W} exists: the sealed test runs once, into a new directory")
    W.mkdir(parents=True)
    sel = json.loads(Path(a.selection).read_text())
    if sel.get("selected") is None:
        raise SystemExit("no eligible checkpoint: by the preregistration the sealed set is not opened")
    meta = json.loads((Path(a.candidate) / "cb2-ind-0001" / "individual.json").read_text())
    seeds = a.rehearsal_seeds or pre["sealed"]["seeds"]
    cpre = {"candidate": {"individual": "cb2-ind-0001", "state_hash": meta["state_hash"],
                          "weights_sha256": meta["weights_sha256"], "event_sha": meta["event_sha"]},
            "seeds": pre["sealed"]["seeds"], "steps": 10000, "label": "s2-sealed",
            "replay_seed": int(pre["sealed"]["seeds"].split("-")[0]),
            "gates": {"score_ratio_min": 1.0, "survival_ratio_min": 0.0, "requests_ratio_min": 0.85}}
    pf = W / "candidate-prereg.json"
    pf.write_text(json.dumps(cpre, indent=1) + "\n")
    c_cmd = [sys.executable, str(REPO / "scripts/explore/cb2_sealed_r3h.py"), "--prereg", str(pf), "--candidate",
             a.candidate, "--work", str(W / "candidate"), "--out", str(W / "candidate-report.json")]
    if a.rehearsal_seeds:
        c_cmd += ["--rehearsal-seeds", a.rehearsal_seeds]
    r_cmd = [sys.executable, str(REPO / "scripts/explore/cb2_evaluate.py"), "evaluate", "--individuals", a.comparator,
             "--label", "s2-sealed-r3hB", "--seeds", seeds, "--archive", str(W / "comparator_archive"),
             "--out", str(W / "eval-r3hB.json")]
    rep = {"format": "cb2-s2-sealed/1", "status": "ENGINEERING_ONLY", "prereg": pre, "selection": sel,
           "candidate": cpre["candidate"], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "deviations": [], "REHEARSAL": bool(a.rehearsal_seeds)}
    with open(W / "candidate.log", "w") as lc, open(W / "comparator.log", "w") as lr:
        pc = subprocess.Popen(c_cmd, cwd=REPO, stdout=lc, stderr=subprocess.STDOUT)
        pr = subprocess.Popen(r_cmd, cwd=REPO, stdout=lr, stderr=subprocess.STDOUT)
        rc_c, rc_r = pc.wait(), pr.wait()
    cand = json.loads((W / "candidate-report.json").read_text()) if (W / "candidate-report.json").exists() else {}
    rep["candidate_report"] = cand
    if "evaluation" not in cand:
        rep["deviations"].append(f"candidate run incomplete (exit {rc_c}); NOT rerun")
    if rc_r != 0 or not (W / "eval-r3hB.json").exists():
        rep["deviations"].append(f"comparator evaluation exited {rc_r}; NOT rerun")
    if rep["deviations"]:
        rep["verdict"] = "INCOMPLETE"
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(rep["verdict"], rep["deviations"])
        return 1
    ce = json.loads((W / "candidate" / f"eval-{cand['prereg']['label']}.json").read_text())
    be = json.loads((W / "eval-r3hB.json").read_text())
    ci = {e["seed"]: e for e in ce["episodes"]["individual"]}
    ri = {e["seed"]: e for e in ce["episodes"]["routine"]}
    bi = {e["seed"]: e for e in be["episodes"]["individual"]}
    common = sorted(ci.keys() & ri.keys() & bi.keys())
    d_rt = np.array([ci[s]["steps"] - ri[s]["steps"] for s in common], float)
    d_b = np.array([ci[s]["steps"] - bi[s]["steps"] for s in common], float)
    s_c, s_b = ce["summary"]["individual"]["crafter_score"], be["summary"]["individual"]["crafter_score"]
    g = cand.get("gates", {})
    rep["paired"] = {"worlds": len(common),
                     "vs_routine": {"survival_mean": float(d_rt.mean()), "survival_lower95": lower95(d_rt)},
                     "vs_r3hB": {"survival_mean": float(d_b.mean()), "survival_lower95": lower95(d_b),
                                 "score_s2": s_c, "score_r3hB": s_b},
                     "r3hB_summary": be["summary"]["individual"]}
    common_ok = g.get("requests", False) and g.get("continuity", False) and not cand.get("deviations")
    complete = rep["paired"]["vs_routine"]["survival_lower95"] > 0 and g.get("score", False) and common_ok
    progress = rep["paired"]["vs_r3hB"]["survival_lower95"] > 0 and s_c >= s_b and common_ok
    rep["comparator_name"] = a.comparator_name                # keys stay "vs_r3hB" for compatibility: the comparator
    rep["verdict"] = ("S2 COMPLETE (beyond the routine)" if complete else f"S2-PROGRESS (beyond {a.comparator_name}, not the routine)"
                      if progress else "NO IMPROVEMENT") + (" [REHEARSAL]" if a.rehearsal_seeds else "")
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "gates": g, "paired": {k: rep["paired"][k] for k in ("vs_routine", "vs_r3hB")},
                      "ratio": cand["evaluation"]["ratio_to_routine"], "requests": cand["evaluation"]["requests"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
