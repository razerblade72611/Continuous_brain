#!/usr/bin/env python3
"""
Corrected train-only temporal diagnosis for Working Memory M1.

Unlike v1, this groups by REALIZED hidden delay from the production training
log itself:

    realized_hidden_delay = decision_step - cue_steps

The generator's behavioral_signature() is retained only as a declared
conservative signature and is audited against realized timing.

No model is loaded. DEV/TEST are never touched.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from continuous_brain.varied_cued_detour_world import (
    TRAIN_CONDITION_RANGE,
    generate_task_spec,
)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def rankdata(values):
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[pairs[k][0]] = rank
        i = j
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def summarize(rows):
    lpos = [float(r["left_decision_margin"] > 0.0) for r in rows]
    rpos = [float(r["right_decision_margin"] > 0.0) for r in rows]
    both = [
        float(
            r["left_decision_margin"] > 0.0
            and r["right_decision_margin"] > 0.0
        )
        for r in rows
    ]
    return {
        "n": len(rows),
        "left_positive": mean(lpos),
        "right_positive": mean(rpos),
        "both_positive": mean(both),
        "gap": mean(rpos) - mean(lpos),
        "left_mean": mean(
            [float(r["left_decision_margin"]) for r in rows]
        ),
        "right_mean": mean(
            [float(r["right_decision_margin"]) for r in rows]
        ),
    }


def print_group(groups):
    print(
        f"{'delay':>7} {'n':>5} "
        f"{'L+':>8} {'R+':>8} {'both+':>8} "
        f"{'gap':>8} {'Lmean':>10} {'Rmean':>10}"
    )
    for delay, rows in sorted(groups.items()):
        s = summarize(rows)
        print(
            f"{delay:7d} {s['n']:5d} "
            f"{s['left_positive']:8.4f} "
            f"{s['right_positive']:8.4f} "
            f"{s['both_positive']:8.4f} "
            f"{s['gap']:+8.4f} "
            f"{s['left_mean']:+10.4f} "
            f"{s['right_mean']:+10.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        default=(
            "state/working_memory_m1/train_epoch1_v1/"
            "training.jsonl"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "state/working_memory_m1/train_epoch1_v1/"
            "training_temporal_diagnosis_realized_v2.json"
        ),
    )
    args = ap.parse_args()

    rows = []
    for lineno, line in enumerate(
        Path(args.jsonl).read_text().splitlines(), 1
    ):
        if not line.strip():
            continue
        r = json.loads(line)
        cid = int(r["condition_id"])
        if cid not in TRAIN_CONDITION_RANGE:
            raise AssertionError(
                f"non-train condition at line {lineno}: {cid}"
            )

        spec = generate_task_spec(cid)
        cue_steps = int(spec.cue_steps)
        decision_step = int(r["decision_step"])
        realized = decision_step - cue_steps
        if realized < 0:
            raise AssertionError(
                f"negative realized hidden delay at condition {cid}"
            )

        declared = int(spec.behavioral_signature()[1])

        r = dict(r)
        r["_cue_steps"] = cue_steps
        r["_decision_step"] = decision_step
        r["_realized_hidden_delay"] = realized
        r["_declared_hidden_delay"] = declared
        r["_declared_minus_realized"] = declared - realized
        rows.append(r)

    if len(rows) != 1600:
        raise AssertionError(
            f"expected 1600 train rows, got {len(rows)}"
        )
    if {int(r["condition_id"]) for r in rows} != set(
        TRAIN_CONDITION_RANGE
    ):
        raise AssertionError("train condition coverage mismatch")

    mismatches = [
        r for r in rows
        if r["_realized_hidden_delay"] != r["_declared_hidden_delay"]
    ]
    delta_counts = collections.Counter(
        int(r["_declared_minus_realized"])
        for r in mismatches
    )

    by_delay = collections.defaultdict(list)
    by_delay_cue = collections.defaultdict(list)
    for r in rows:
        by_delay[r["_realized_hidden_delay"]].append(r)
        by_delay_cue[
            (r["_realized_hidden_delay"], r["_cue_steps"])
        ].append(r)

    delays = [float(r["_realized_hidden_delay"]) for r in rows]
    left_margin = [
        float(r["left_decision_margin"]) for r in rows
    ]
    right_margin = [
        float(r["right_decision_margin"]) for r in rows
    ]
    left_ok = [
        float(r["left_decision_margin"] > 0.0) for r in rows
    ]
    right_ok = [
        float(r["right_decision_margin"] > 0.0) for r in rows
    ]

    correlations = {
        "realized_delay_vs_left_margin_pearson":
            pearson(delays, left_margin),
        "realized_delay_vs_right_margin_pearson":
            pearson(delays, right_margin),
        "realized_delay_vs_left_margin_spearman":
            spearman(delays, left_margin),
        "realized_delay_vs_right_margin_spearman":
            spearman(delays, right_margin),
        "realized_delay_vs_left_positive_spearman":
            spearman(delays, left_ok),
        "realized_delay_vs_right_positive_spearman":
            spearman(delays, right_ok),
    }

    # Also report bin-level Spearman explicitly, avoiding future ambiguity.
    bin_delays = []
    bin_l = []
    bin_r = []
    for delay, group in sorted(by_delay.items()):
        s = summarize(group)
        bin_delays.append(float(delay))
        bin_l.append(float(s["left_positive"]))
        bin_r.append(float(s["right_positive"]))

    binned_correlations = {
        "delay_bin_vs_left_positive_spearman":
            spearman(bin_delays, bin_l),
        "delay_bin_vs_right_positive_spearman":
            spearman(bin_delays, bin_r),
    }

    print("WORKING_MEMORY_TRAIN_TEMPORAL_DIAGNOSIS_REALIZED_V2")
    print("records:", len(rows))
    print("train_condition_coverage_exact: true")
    print(
        "declared_vs_realized_delay_mismatches:",
        f"{len(mismatches)}/{len(rows)}",
    )
    print(
        "declared_minus_realized_delta_counts:",
        dict(sorted(delta_counts.items())),
    )
    print()
    print("=" * 94)
    print("REALIZED HIDDEN DELAY = LOGGED decision_step - cue_steps")
    print("=" * 94)
    print_group(by_delay)

    print()
    print("PER_CONDITION_CORRELATIONS")
    for k, v in correlations.items():
        print(f"{k}: {v:+.6f}")

    print()
    print("BINNED_CORRELATIONS")
    for k, v in binned_correlations.items():
        print(f"{k}: {v:+.6f}")

    print()
    print("=" * 94)
    print("REALIZED HIDDEN DELAY x CUE STEPS")
    print("=" * 94)
    print(
        f"{'delay':>6} {'cue':>5} {'n':>5} "
        f"{'L+':>8} {'R+':>8} {'both+':>8} {'gap':>8}"
    )
    for (delay, cue), group in sorted(by_delay_cue.items()):
        s = summarize(group)
        print(
            f"{delay:6d} {cue:5d} {s['n']:5d} "
            f"{s['left_positive']:8.4f} "
            f"{s['right_positive']:8.4f} "
            f"{s['both_positive']:8.4f} "
            f"{s['gap']:+8.4f}"
        )

    result = {
        "format": "working_memory_train_temporal_diagnosis_realized_v2",
        "records": len(rows),
        "train_condition_coverage_exact": True,
        "delay_definition": "logged decision_step - generated cue_steps",
        "declared_vs_realized_delay_mismatches": len(mismatches),
        "declared_minus_realized_delta_counts":
            dict(sorted(delta_counts.items())),
        "correlations": correlations,
        "binned_correlations": binned_correlations,
        "by_realized_hidden_delay": {
            str(k): summarize(v)
            for k, v in sorted(by_delay.items())
        },
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print(
        "WORKING_MEMORY_TRAIN_TEMPORAL_DIAGNOSIS_REALIZED_COMPLETE"
    )


if __name__ == "__main__":
    main()
