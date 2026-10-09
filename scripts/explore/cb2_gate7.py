#!/usr/bin/env python3
"""Gate 7 (DAV-90): the real organ swap and its acceptance, on a copy of the frozen r3h-B candidate.

  python scripts/explore/cb2_gate7.py --candidate <candidates/r3h-B> --work <new dir> \\
      --reference-telemetry <r3h dev delegation dir> --reference-eval <eval-r3h-B.json>

Steps (criteria posted on DAV-90 before running):
  1. copy the candidate; swap vision v1 -> v2 (cb2_organ_swap.py); verify
  2. closed loop with v2 on the development worlds 75001-75040 (cb2_evaluate.py, unchanged protocol)
  3. roll back v2 -> v1; verify
  4. live world 75001 again after the rollback and compare every action with the never-swapped r3h-B's
  5. report: identity (state hash + identity digest unchanged by both events, chain verified), rollback exactness,
     capability (v2 vs v1 tile accuracy on v1's test set; closed-loop ratios vs the routine; paired vs r3h-B-v1)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(args, log):
    p = subprocess.run([PY, *args], cwd=REPO, text=True, capture_output=True)
    Path(log).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"STOP: {' '.join(args[:2])} failed (see {log})")
    return p.stdout


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--reference-telemetry", required=True)
    ap.add_argument("--reference-eval", required=True)
    ap.add_argument("--seeds", default="75001-75040")
    ap.add_argument("--replay-seed", type=int, default=75001)
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/gate7-report.json"))
    a = ap.parse_args()
    W = Path(a.work)
    if W.exists():
        raise SystemExit(f"{W} exists: Gate 7 runs into a new directory")
    W.mkdir(parents=True)
    C = W / "c"
    C.mkdir()
    shutil.copytree(Path(a.candidate) / "cb2-ind-0001", C / "cb2-ind-0001")
    v1 = "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999"
    sw = "scripts/explore/cb2_organ_swap.py"
    swap = json.loads(run([sw, "swap", "--individuals", str(C), "--weights", "runs/explore/vision/tile_decoder_v2.pt",
                           "--card", "continuous_brain_organs/cards/tile_decoder_v2.json",
                           "--note", "Gate 7: vision tile_decoder_v1 -> tile_decoder_v2 on a copy of frozen r3h-B"], W / "swap.log"))
    ver1 = json.loads(run([sw, "verify", "--individuals", str(C)], W / "verify1.log"))
    run(["scripts/explore/cb2_evaluate.py", "evaluate", "--individuals", str(C), "--label", "g7-v2", "--seeds", a.seeds,
         "--archive", str(W / "archive"), "--out", str(W / "eval-g7-v2.json"), "--telemetry", str(W / "tel_v2")], W / "eval_v2.log")
    rb = json.loads(run([sw, "rollback", "--individuals", str(C), "--to", v1, "--note", "Gate 7 rollback v2 -> v1"], W / "rollback.log"))
    ver2 = json.loads(run([sw, "verify", "--individuals", str(C)], W / "verify2.log"))
    run(["scripts/explore/cb2_evaluate.py", "evaluate", "--individuals", str(C), "--label", "g7-rollback",
         "--seeds", str(a.replay_seed), "--archive", str(W / "archive"), "--out", str(W / "eval-g7-rollback.json"),
         "--telemetry", str(W / "tel_rb")], W / "eval_rb.log")
    ref = [json.loads(l)["taken"] for l in open(Path(a.reference_telemetry) / f"delegation-{a.replay_seed}.jsonl")]
    got = [json.loads(l)["taken"] for l in open(W / "tel_rb" / f"delegation-{a.replay_seed}.jsonl")]
    e2 = json.load(open(W / "eval-g7-v2.json"))
    e1 = json.load(open(a.reference_eval))
    s2, rt = e2["summary"]["individual"], e2["summary"]["routine"]
    s1 = e1["summary"]["individual"]
    d = [x["steps"] - y["steps"] for x, y in zip(e2["episodes"]["individual"], e1["episodes"]["individual"])]
    du = [x["n_unlocked"] - y["n_unlocked"] for x, y in zip(e2["episodes"]["individual"], e1["episodes"]["individual"])]
    import numpy as np
    v1e = json.load(open(REPO / "evidence/explore/vision/tile_decoder_v1_eval.json"))
    v2e = json.load(open(REPO / "evidence/explore/vision/tile_decoder_v2_eval.json"))
    acc = {k: {"v1": {h: v1e[k][h]["accuracy"] for h in ("material", "object", "both")},
               "v2": {h: v2e[k][h]["accuracy"] for h in ("material", "object", "both")}} for k in ("overall", "night")}
    ratio = {m: round(s2[m] / rt[m], 4) for m in ("crafter_score", "survival_steps", "achievements_per_episode")}
    crit = {
        "2_identity": swap["parent_state_hash"] == swap["state_hash"] == rb["state_hash"]
                      and swap["identity_digest"] == rb["identity_digest"] and ver1["log_matches_individual"]
                      and ver2["log_matches_individual"] and ver2["vision_weights_sha256"] == v1,
        "3_rollback_exact": ref == got,
        "4a_vision_not_worse": all(acc[k]["v2"][h] >= acc[k]["v1"][h] for k in acc for h in ("material", "object")),
        "4b_closed_loop": ratio["crafter_score"] >= 0.9 and ratio["survival_steps"] >= 0.9,
    }
    rep = {"format": "cb2-gate7/1", "status": "ENGINEERING_ONLY", "work": str(W), "swap": swap, "rollback": rb,
           "verify_after_swap": ver1, "verify_after_rollback": ver2,
           "replay": {"seed": a.replay_seed, "reference_steps": len(ref), "steps": len(got),
                      "first_difference": next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)},
           "vision_test_accuracy": acc,
           "closed_loop_v2": {"candidate": s2, "routine": rt, "ratio_to_routine": ratio},
           "paired_v2_vs_v1": {"steps": {"mean": float(np.mean(d)), "se": float(np.std(d, ddof=1) / np.sqrt(len(d)))},
                               "n_unlocked": {"mean": float(np.mean(du)), "se": float(np.std(du, ddof=1) / np.sqrt(len(du)))},
                               "v1": {k: s1[k] for k in ("crafter_score", "survival_steps", "achievements_per_episode")}},
           "criteria": crit, "verdict": "PASS" if all(crit.values()) else "FAIL",
           "note": "criterion 1 (tests) is run separately: tests/test_cb2_organ_swap.py"}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"criteria": crit, "verdict": rep["verdict"], "ratio": ratio, "replay": rep["replay"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
