#!/usr/bin/env python3
"""Vision organ v2 sealed test (exploration lane): executes evidence/explore/cb2/v2-sealed-prereg.json once.

  python scripts/explore/cb2_sealed_v2.py --candidate <dir holding the v2 copy> --comparator <frozen r3h-B dir> \\
      --work <new dir>

The candidate (frozen r3h-B with vision v2) runs through cb2_sealed_r3h.py (integrity, the unchanged evaluation,
continuity); the comparator (frozen r3h-B with v1) through cb2_evaluate.py on the same worlds, at the same time.
Then the organ is checked (the v2 weights and the recorded swap event), and non-inferiority vs v1 is computed by a
seeded paired bootstrap.  Any crash or deviation is reported, never rerun silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default=str(REPO / "evidence/explore/cb2/v2-sealed-prereg.json"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--comparator", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/v2-sealed-report.json"))
    ap.add_argument("--rehearsal-seeds", default=None)
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    W = Path(a.work)
    if W.exists():
        raise SystemExit(f"{W} exists: the sealed test runs once, into a new directory")
    W.mkdir(parents=True)
    seeds = a.rehearsal_seeds or pre["seeds"]
    # the candidate's own sealed run, through the r3-H executor (same gates, same continuity)
    r3h_pre = {"candidate": {k: pre["candidate"][k] for k in ("individual", "state_hash", "weights_sha256", "event_sha")},
               "seeds": pre["seeds"], "steps": pre["steps"], "label": pre["label"], "replay_seed": pre["replay_seed"],
               "gates": pre["gates"]}
    pf = W / "candidate-prereg.json"
    pf.write_text(json.dumps(r3h_pre, indent=1) + "\n")
    cand_cmd = [sys.executable, str(REPO / "scripts/explore/cb2_sealed_r3h.py"), "--prereg", str(pf),
                "--candidate", a.candidate, "--work", str(W / "candidate"), "--out", str(W / "candidate-report.json")]
    if a.rehearsal_seeds:
        cand_cmd += ["--rehearsal-seeds", a.rehearsal_seeds]
    comp_cmd = [sys.executable, str(REPO / "scripts/explore/cb2_evaluate.py"), "evaluate", "--individuals", a.comparator,
                "--label", pre["label"] + "-v1", "--seeds", seeds, "--steps", str(pre["steps"]),
                "--archive", str(W / "comparator_archive"), "--out", str(W / "eval-v1.json")]
    rep = {"format": "cb2-v2-sealed/1", "status": "ENGINEERING_ONLY", "prereg": pre,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deviations": [],
           "REHEARSAL": bool(a.rehearsal_seeds)}
    with open(W / "candidate.log", "w") as lc, open(W / "comparator.log", "w") as lv:
        pc = subprocess.Popen(cand_cmd, cwd=REPO, stdout=lc, stderr=subprocess.STDOUT)
        pv = subprocess.Popen(comp_cmd, cwd=REPO, stdout=lv, stderr=subprocess.STDOUT)
        rc_c, rc_v = pc.wait(), pv.wait()
    cand = json.loads((W / "candidate-report.json").read_text()) if (W / "candidate-report.json").exists() else {}
    rep["candidate_report"] = cand
    if rc_v != 0 or not (W / "eval-v1.json").exists():
        rep["deviations"].append(f"comparator evaluation exited {rc_v}; NOT rerun")
    if "evaluation" not in cand:
        rep["deviations"].append(f"candidate run incomplete (exit {rc_c}); NOT rerun")
    if rep["deviations"]:
        rep["verdict"] = "INCOMPLETE"
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(rep["verdict"], rep["deviations"])
        return 1
    # the organ it lived with
    home = Path(a.candidate) / pre["candidate"]["individual"]
    meta = json.loads((home / "individual.json").read_text())
    organ_ok = (meta.get("organs", {}).get("vision", {}).get("weights_sha256") == pre["candidate"]["vision_weights_sha256"]
                and meta.get("organ_event_sha") == pre["candidate"]["organ_event_sha"])
    v_eps = json.loads((W / "eval-v1.json").read_text())
    c_eps = json.loads((W / "candidate" / f"eval-{cand['prereg']['label']}.json").read_text())
    ci = {e["seed"]: e for e in c_eps["episodes"]["individual"]}
    vi = {e["seed"]: e for e in v_eps["episodes"]["individual"]}
    common = sorted(ci.keys() & vi.keys())
    d = np.array([ci[s]["steps"] - vi[s]["steps"] for s in common], float)
    rng = np.random.default_rng(20261009)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lb = float(np.percentile(boots, 5))
    sc_c, sc_v = c_eps["summary"]["individual"]["crafter_score"], v_eps["summary"]["individual"]["crafter_score"]
    ni = pre["non_inferiority_vs_v1"]
    rep["v2_vs_v1"] = {"worlds": len(common), "survival_mean_diff": float(d.mean()), "survival_lower95": lb,
                       "score_v2": sc_c, "score_v1": sc_v, "score_ratio_v2_over_v1": sc_c / sc_v,
                       "v1_summary": v_eps["summary"]["individual"], "routine_same_worlds_v1_run": v_eps["summary"]["routine"]}
    gates = dict(cand.get("gates", {}))
    gates["organ_is_v2"] = organ_ok
    gates["non_inferior_survival"] = lb > ni["survival_paired_lower95_min"]
    gates["non_inferior_score"] = sc_c / sc_v >= ni["score_ratio_vs_v1_min"]
    rep["gates"] = gates
    rep["superior_survival_reported"] = lb > 0
    rep["verdict"] = ("PASS" if all(gates.values()) and not cand.get("deviations") else "FAIL") + \
                     (" (REHEARSAL: not the sealed test)" if a.rehearsal_seeds else "")
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "gates": gates, "v2_vs_v1": {k: rep["v2_vs_v1"][k] for k in
                      ("survival_mean_diff", "survival_lower95", "score_ratio_v2_over_v1")},
                      "candidate_ratio": cand["evaluation"]["ratio_to_routine"],
                      "requests": cand["evaluation"]["requests"]}, indent=1))
    return 0 if rep["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
