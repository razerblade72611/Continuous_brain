#!/usr/bin/env python3
"""Pre-registered fresh-DEV admission for novel-rule executive v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULES = ("invert_lr", "vertical", "vertical_reverse")

MIN_RULE_PAIR = 0.80
MIN_SIX = 0.70
MAX_OFF_PAIR = 0.15
MAX_ZERO_PAIR = 0.15
MIN_DONOR = 0.90
MAX_NONE_DIFF = 1e-6
MIN_FACTOR_RULE = 0.60
MIN_FACTOR_SIX = 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.input).read_text())

    if (
        p.get("format")
        != "executive_novel_rule_fresh_dev_v2"
    ):
        raise RuntimeError("invalid fresh DEV format")
    if p.get("split") != "fresh_dev":
        raise RuntimeError("not fresh DEV")
    if p.get("fresh_test_touched") is not False:
        raise RuntimeError("fresh TEST contamination")

    s = p["summary"]
    checks = {}

    for rule in RULES:
        checks[f"{rule}_pair_full"] = (
            s["normal"][rule] >= MIN_RULE_PAIR
        )
        checks[f"{rule}_executive_off_fails"] = (
            s["executive_off"][rule] <= MAX_OFF_PAIR
        )
        checks[f"{rule}_memory_zero_fails"] = (
            s["memory_zero"][rule] <= MAX_ZERO_PAIR
        )
        checks[f"{rule}_donor_memory"] = (
            s["memory_swapped_donor_target"][rule]
            >= MIN_DONOR
        )

    checks["six_way_full"] = (
        s["normal"]["six_way"] >= MIN_SIX
    )
    checks["none_exact_noop"] = (
        s["none_max_diff"] <= MAX_NONE_DIFF
    )

    failures = []

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
                failures.append({
                    "factor": field,
                    "value": value,
                    **record,
                })

    checks["factor_robustness"] = len(failures) == 0
    passed = all(checks.values())

    result = {
        "format":
            "executive_novel_rule_fresh_dev_decision_v2",
        "pass": passed,
        "checks": checks,
        "factor_failures": failures,
        "summary": s,
        "fresh_test_touched": False,
    }

    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("EXECUTIVE_NOVEL_RULE_FRESH_DEV_DECISION_V2")

    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={s['normal'][rule]:.4f} "
            f"off={s['executive_off'][rule]:.4f} "
            f"zero={s['memory_zero'][rule]:.4f} "
            f"donor="
            f"{s['memory_swapped_donor_target'][rule]:.4f}"
        )

    print(
        f"six_way_full={s['normal']['six_way']:.4f} "
        f"none_diff={s['none_max_diff']:.3g}"
    )
    print()

    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("factor_failure_count:", len(failures))
    print("FRESH_TEST_TOUCHED: false")
    print(
        "NOVEL_RULE_FRESH_DEV_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "EXECUTIVE_NOVEL_RULE_FRESH_DEV_DECISION_COMPLETE"
    )

    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
