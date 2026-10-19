#!/usr/bin/env python3
"""
Report factor-level LEFT/RIGHT asymmetry from the existing train-only
working-memory epoch audit.

No model is loaded. No DEV/TEST condition is touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FACTOR_ORDER = (
    "corridor_length",
    "cue_steps",
    "move_step",
    "goal_x_offset",
    "goal_y_offset",
    "start_x",
    "start_y",
    "distractor_pattern",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=(
            "state/working_memory_m1/train_epoch1_v1/"
            "training_audit.json"
        ),
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text())
    slices = payload["factor_slices"]

    print("WORKING_MEMORY_TRAIN_FACTOR_SLICE_REPORT_V1")
    print()

    all_rows = []

    for factor in FACTOR_ORDER:
        print("=" * 88)
        print(factor)
        print("=" * 88)
        print(
            f"{'value':>12} {'n':>5} "
            f"{'L+':>8} {'R+':>8} {'both+':>8} "
            f"{'gap':>8} {'Lmean':>10} {'Rmean':>10}"
        )

        factor_rows = []
        for value, s in slices[factor].items():
            l = float(s["left_margin_positive_fraction"])
            r = float(s["right_margin_positive_fraction"])
            both = float(s["both_margins_positive_fraction"])
            gap = r - l
            lm = float(s["left_margin_mean"])
            rm = float(s["right_margin_mean"])
            n = int(s["n"])

            row = {
                "factor": factor,
                "value": value,
                "n": n,
                "left": l,
                "right": r,
                "both": both,
                "gap": gap,
                "left_mean": lm,
                "right_mean": rm,
            }
            factor_rows.append(row)
            all_rows.append(row)

        # Sort levels naturally-ish by their string form for stable output.
        for row in sorted(factor_rows, key=lambda x: x["value"]):
            print(
                f"{row['value']:>12} {row['n']:5d} "
                f"{row['left']:8.4f} {row['right']:8.4f} "
                f"{row['both']:8.4f} {row['gap']:+8.4f} "
                f"{row['left_mean']:+10.4f} "
                f"{row['right_mean']:+10.4f}"
            )
        print()

    print("=" * 88)
    print("LARGEST RIGHT-MINUS-LEFT POSITIVE-FRACTION GAPS")
    print("=" * 88)
    for row in sorted(
        all_rows,
        key=lambda x: (x["gap"], -x["left"]),
        reverse=True,
    )[:20]:
        print(
            f"{row['factor']:>20}={row['value']:<10} "
            f"n={row['n']:3d} "
            f"L+={row['left']:.4f} "
            f"R+={row['right']:.4f} "
            f"both+={row['both']:.4f} "
            f"gap={row['gap']:+.4f} "
            f"Lmean={row['left_mean']:+.4f} "
            f"Rmean={row['right_mean']:+.4f}"
        )

    print()
    print("=" * 88)
    print("LOWEST LEFT POSITIVE-FRACTION LEVELS")
    print("=" * 88)
    for row in sorted(
        all_rows,
        key=lambda x: (x["left"], -x["gap"]),
    )[:20]:
        print(
            f"{row['factor']:>20}={row['value']:<10} "
            f"n={row['n']:3d} "
            f"L+={row['left']:.4f} "
            f"R+={row['right']:.4f} "
            f"both+={row['both']:.4f} "
            f"gap={row['gap']:+.4f}"
        )

    print()
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_TRAIN_FACTOR_SLICE_REPORT_COMPLETE")


if __name__ == "__main__":
    main()
