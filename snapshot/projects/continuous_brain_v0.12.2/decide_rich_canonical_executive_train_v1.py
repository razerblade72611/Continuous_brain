#!/usr/bin/env python3
"""Frozen TRAIN admission gate for the rich canonical executive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULES = ("invert_lr", "vertical", "vertical_reverse")

MIN_RULE_PAIR = 0.95
MIN_DONOR_PAIR = 0.95
MAX_EXEC_OFF = 0.05
MAX_ZERO = 0.05
MIN_SIX_WAY = 0.90
MAX_NONE_DIFF = 1e-6
MIN_FACTOR_RULE = 0.85
MIN_FACTOR_SIX = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.input).read_text())
    if p.get("format") != "rich_canonical_executive_final_train_v1":
        raise RuntimeError("unexpected evaluator format")
    if p.get("split") != "train_only":
        raise RuntimeError("gate must be TRAIN-only")
    if p.get("v3_dev_touched") or p.get("v3_test_touched"):
        raise RuntimeError("held-out contamination")

    s = p["summary"]
    checks = {}

    for rule in RULES:
        checks[f"{rule}_pair_full"] = (
            s["normal"][rule] >= MIN_RULE_PAIR
        )
        checks[f"{rule}_donor"] = (
            s["canonical_donor"][rule] >= MIN_DONOR_PAIR
        )
        checks[f"{rule}_executive_off_fails"] = (
            s["executive_off"][rule] <= MAX_EXEC_OFF
        )
        checks[f"{rule}_canonical_zero_fails"] = (
            s["canonical_zero"][rule] <= MAX_ZERO
        )

    checks["six_way_full"] = (
        s["normal"]["six_way"] >= MIN_SIX_WAY
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
        "format": "rich_canonical_executive_train_decision_v1",
        "pass": passed,
        "checks": checks,
        "factor_failures": factor_failures,
        "summary": s,
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("RICH_CANONICAL_EXECUTIVE_TRAIN_DECISION_V1")
    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={s['normal'][rule]:.4f} "
            f"donor={s['canonical_donor'][rule]:.4f} "
            f"off={s['executive_off'][rule]:.4f} "
            f"zero={s['canonical_zero'][rule]:.4f}"
        )
    print(
        f"six_way={s['normal']['six_way']:.4f} "
        f"none_diff={s['none_max_diff']:.3g}"
    )
    print()
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:", len(factor_failures))
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("RICH_CANONICAL_TRAIN_GATE:", "PASS" if passed else "FAIL")
    print("RICH_CANONICAL_EXECUTIVE_TRAIN_DECISION_COMPLETE")

    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
