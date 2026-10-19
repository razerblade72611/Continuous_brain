#!/usr/bin/env python3
"""Frozen TRAIN learning gate for Executive Composition Production v1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Frozen before seeing production result.
MIN_NORMAL_PAIR_FULL = 0.85
MIN_INVERT_PAIR_FULL = 0.85
MIN_FOUR_WAY_FULL = 0.80
MIN_NORMAL_PAIR_SIDE = 0.98
MIN_INVERT_PAIR_SIDE = 0.98
MIN_FOUR_WAY_SIDE = 0.95
MIN_FLIP_CONSISTENCY = 0.98
MIN_MEMORY_SWAP_DONOR = 0.98
MAX_EXEC_OFF_INVERT_SIDE = 0.10
MAX_MEMORY_ZERO_INVERT_SIDE = 0.10
MIN_EXEC_OFF_NORMAL_SIDE = 0.95
MAX_NONE_DIFF = 1e-6
MIN_FACTOR_INVERT_SIDE = 0.90
MIN_FACTOR_INVERT_FULL = 0.60

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    p = json.loads(Path(a.input).read_text())
    if p.get("format") != "executive_composition_final_train_causal_v1":
        raise RuntimeError("unexpected evaluator format")
    if p.get("split") != "train_only":
        raise RuntimeError("not TRAIN-only")
    if p.get("old_dev_touched") or p.get("old_test_touched"):
        raise RuntimeError("held-out contamination")

    s = p["summary"]
    checks = {
        "normal_pair_full":
            s["normal"]["normal"]["pair_full"] >= MIN_NORMAL_PAIR_FULL,
        "invert_pair_full":
            s["normal"]["invert"]["pair_full"] >= MIN_INVERT_PAIR_FULL,
        "four_way_full":
            s["normal"]["four_way_full"] >= MIN_FOUR_WAY_FULL,
        "normal_pair_side":
            s["normal"]["normal"]["pair_side"] >= MIN_NORMAL_PAIR_SIDE,
        "invert_pair_side":
            s["normal"]["invert"]["pair_side"] >= MIN_INVERT_PAIR_SIDE,
        "four_way_side":
            s["normal"]["four_way_side"] >= MIN_FOUR_WAY_SIDE,
        "instruction_flip_consistency":
            s["instruction_flip_consistency"] >= MIN_FLIP_CONSISTENCY,
        "memory_swap_normal_donor":
            s["memory_swapped_donor_target"]["normal"]["pair_side"]
            >= MIN_MEMORY_SWAP_DONOR,
        "memory_swap_invert_donor":
            s["memory_swapped_donor_target"]["invert"]["pair_side"]
            >= MIN_MEMORY_SWAP_DONOR,
        "executive_off_normal_preserved":
            s["executive_off"]["normal"]["pair_side"]
            >= MIN_EXEC_OFF_NORMAL_SIDE,
        "executive_off_invert_fails":
            s["executive_off"]["invert"]["pair_side"]
            <= MAX_EXEC_OFF_INVERT_SIDE,
        "memory_zero_invert_fails":
            s["memory_zero"]["invert"]["pair_side"]
            <= MAX_MEMORY_ZERO_INVERT_SIDE,
        "instruction_none_exact_noop":
            s["none_exact_noop_max_diff"] <= MAX_NONE_DIFF,
        "instruction_has_nonzero_effect":
            s["instruction_logit_separation_mean"] > 0.0,
    }

    factor_failures = []
    for field, levels in p["factor_summary"].items():
        for value, r in levels.items():
            ok = (
                r["invert_pair_side"] >= MIN_FACTOR_INVERT_SIDE
                and r["invert_pair_full"] >= MIN_FACTOR_INVERT_FULL
            )
            if not ok:
                factor_failures.append({
                    "factor": field,
                    "value": value,
                    **r,
                })
    checks["factor_robustness"] = len(factor_failures) == 0

    passed = all(checks.values())

    result = {
        "format": "executive_composition_train_learning_decision_v1",
        "pass": passed,
        "checks": checks,
        "factor_failures": factor_failures,
        "summary": s,
        "old_dev_touched": False,
        "old_test_touched": False,
    }
    Path(a.output).write_text(json.dumps(result, indent=2, sort_keys=True))

    print("EXECUTIVE_COMPOSITION_TRAIN_LEARNING_DECISION_V1")
    print(
        f"normal_pair_full={s['normal']['normal']['pair_full']:.4f} "
        f"invert_pair_full={s['normal']['invert']['pair_full']:.4f} "
        f"four_way_full={s['normal']['four_way_full']:.4f}"
    )
    print(
        f"normal_pair_side={s['normal']['normal']['pair_side']:.4f} "
        f"invert_pair_side={s['normal']['invert']['pair_side']:.4f} "
        f"four_way_side={s['normal']['four_way_side']:.4f}"
    )
    print(
        f"exec_off_invert_side="
        f"{s['executive_off']['invert']['pair_side']:.4f} "
        f"memory_zero_invert_side="
        f"{s['memory_zero']['invert']['pair_side']:.4f}"
    )
    print(
        f"memory_swap_donor_normal="
        f"{s['memory_swapped_donor_target']['normal']['pair_side']:.4f} "
        f"memory_swap_donor_invert="
        f"{s['memory_swapped_donor_target']['invert']['pair_side']:.4f}"
    )
    print(
        f"flip_consistency={s['instruction_flip_consistency']:.4f} "
        f"none_diff={s['none_exact_noop_max_diff']:.3g}"
    )
    print()
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:", len(factor_failures))
    print("OLD_DEV_TOUCHED: false")
    print("OLD_TEST_TOUCHED: false")
    print("EXECUTIVE_TRAIN_LEARNING_GATE:", "PASS" if passed else "FAIL")
    print("EXECUTIVE_COMPOSITION_TRAIN_LEARNING_DECISION_COMPLETE")
    sys.exit(0 if passed else 4)

if __name__ == "__main__":
    main()
