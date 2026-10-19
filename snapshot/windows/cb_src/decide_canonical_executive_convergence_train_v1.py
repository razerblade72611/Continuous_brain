#!/usr/bin/env python3
"""Frozen TRAIN convergence gate for the canonical recall executive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULES = ("invert_lr", "vertical", "vertical_reverse")

# Frozen before convergence continuation is run.
MIN_RULE_PAIR_FULL = 0.95
MIN_SIX_WAY_FULL = 0.90
MIN_DONOR_PAIR = 0.95
MAX_EXEC_OFF_PAIR = 0.05
MAX_CANONICAL_ZERO_PAIR = 0.05
MAX_NONE_DIFF = 1e-6
MIN_FACTOR_RULE = 0.85
MIN_FACTOR_SIX = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.input).read_text())
    if p.get("format") != "canonical_executive_convergence_final_train_v1":
        raise RuntimeError("unexpected convergence evaluator format")
    if p.get("split") != "train_only":
        raise RuntimeError("convergence gate is not TRAIN-only")
    if p.get("fresh_test_touched") is not False:
        raise RuntimeError("TEST contamination")

    s = p["summary"]
    checks = {}

    for rule in RULES:
        checks[f"{rule}_pair_full"] = (
            s["normal"][rule] >= MIN_RULE_PAIR_FULL
        )
        checks[f"{rule}_donor"] = (
            s["canonical_memory_donor"][rule] >= MIN_DONOR_PAIR
        )
        checks[f"{rule}_executive_off_fails"] = (
            s["executive_off"][rule] <= MAX_EXEC_OFF_PAIR
        )
        checks[f"{rule}_canonical_zero_fails"] = (
            s["canonical_memory_zero"][rule]
            <= MAX_CANONICAL_ZERO_PAIR
        )

    checks["six_way_full"] = (
        s["normal"]["six_way"] >= MIN_SIX_WAY_FULL
    )
    checks["none_exact_noop"] = (
        s["none_max_diff"] <= MAX_NONE_DIFF
    )

    factor_failures = []
    for field, levels in p["factor_summary"].items():
        for value, record in levels.items():
            ok = (
                record["six_way_full"] >= MIN_FACTOR_SIX
                and all(
                    record[f"{rule}_pair_full"]
                    >= MIN_FACTOR_RULE
                    for rule in RULES
                )
            )
            if not ok:
                factor_failures.append({
                    "factor": field,
                    "value": value,
                    **record,
                })

    checks["factor_robustness"] = len(factor_failures) == 0
    passed = all(checks.values())

    result = {
        "format": "canonical_executive_convergence_train_decision_v1",
        "pass": passed,
        "checks": checks,
        "factor_failures": factor_failures,
        "summary": s,
        "fresh_test_touched": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("CANONICAL_EXECUTIVE_CONVERGENCE_TRAIN_DECISION_V1")
    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={s['normal'][rule]:.4f} "
            f"donor={s['canonical_memory_donor'][rule]:.4f} "
            f"off={s['executive_off'][rule]:.4f} "
            f"zero={s['canonical_memory_zero'][rule]:.4f}"
        )
    print(
        f"six_way={s['normal']['six_way']:.4f} "
        f"none_diff={s['none_max_diff']:.3g}"
    )
    print()
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:", len(factor_failures))
    print("FRESH_TEST_TOUCHED: false")
    print("CANONICAL_CONVERGENCE_GATE:", "PASS" if passed else "FAIL")
    print("CANONICAL_EXECUTIVE_CONVERGENCE_TRAIN_DECISION_COMPLETE")

    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
