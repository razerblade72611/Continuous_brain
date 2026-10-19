#!/usr/bin/env python3
"""
Train-only temporal diagnosis for Working Memory M1.

Reads the completed 1600-update training JSONL and re-groups delayed-decision
margins by the generator's actual temporal quantities:
  * predecision teacher steps
  * cue-hidden delay steps
  * total teacher path steps
  * cue-hidden-delay x cue_steps interaction

No model is loaded. DEV and TEST are never touched.
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


def summarize(rows):
    return {
        "n": len(rows),
        "left_positive": mean(
            [float(r["left_decision_margin"] > 0.0) for r in rows]
        ),
        "right_positive": mean(
            [float(r["right_decision_margin"] > 0.0) for r in rows]
        ),
        "both_positive": mean(
            [
                float(
                    r["left_decision_margin"] > 0.0
                    and r["right_decision_margin"] > 0.0
                )
                for r in rows
            ]
        ),
        "left_mean": mean(
            [float(r["left_decision_margin"]) for r in rows]
        ),
        "right_mean": mean(
            [float(r["right_decision_margin"]) for r in rows]
        ),
        "gap": (
            mean([float(r["right_decision_margin"] > 0.0) for r in rows])
            - mean([float(r["left_decision_margin"] > 0.0) for r in rows])
        ),
    }


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
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


def print_group(title, groups):
    print("=" * 94)
    print(title)
    print("=" * 94)
    print(
        f"{'value':>12} {'n':>5} "
        f"{'L+':>8} {'R+':>8} {'both+':>8} "
        f"{'gap':>8} {'Lmean':>10} {'Rmean':>10}"
    )
    for value, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        s = summarize(rows)
        print(
            f"{str(value):>12} {s['n']:5d} "
            f"{s['left_positive']:8.4f} "
            f"{s['right_positive']:8.4f} "
            f"{s['both_positive']:8.4f} "
            f"{s['gap']:+8.4f} "
            f"{s['left_mean']:+10.4f} "
            f"{s['right_mean']:+10.4f}"
        )
    print()


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
            "training_temporal_diagnosis.json"
        ),
    )
    args = ap.parse_args()

    rows = []
    for lineno, line in enumerate(Path(args.jsonl).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        cid = int(r["condition_id"])
        if cid not in TRAIN_CONDITION_RANGE:
            raise AssertionError(
                f"non-train condition at line {lineno}: {cid}"
            )

        spec = generate_task_spec(cid)
        behavioral = tuple(spec.behavioral_signature())
        # behavioral_signature:
        # (predecision_steps, hidden_delay_steps,
        #  lateral_steps, post_vertical_steps, total_steps)
        r = dict(r)
        r["_predecision_steps"] = int(behavioral[0])
        r["_hidden_delay_steps"] = int(behavioral[1])
        r["_lateral_steps"] = int(behavioral[2])
        r["_post_vertical_steps"] = int(behavioral[3])
        r["_total_teacher_steps"] = int(behavioral[4])
        r["_cue_steps"] = int(spec.cue_steps)
        r["_corridor_length"] = float(spec.corridor_length)
        r["_move_step"] = float(spec.move_step)
        rows.append(r)

    if len(rows) != 1600:
        raise AssertionError(
            f"expected 1600 train rows, got {len(rows)}"
        )
    ids = {int(r["condition_id"]) for r in rows}
    if ids != set(TRAIN_CONDITION_RANGE):
        raise AssertionError("train condition coverage mismatch")

    by_hidden = collections.defaultdict(list)
    by_pre = collections.defaultdict(list)
    by_total = collections.defaultdict(list)
    by_hidden_cue = collections.defaultdict(list)

    for r in rows:
        by_hidden[r["_hidden_delay_steps"]].append(r)
        by_pre[r["_predecision_steps"]].append(r)
        by_total[r["_total_teacher_steps"]].append(r)
        by_hidden_cue[
            (r["_hidden_delay_steps"], r["_cue_steps"])
        ].append(r)

    hidden = [r["_hidden_delay_steps"] for r in rows]
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
        "hidden_delay_vs_left_margin_pearson":
            pearson(hidden, left_margin),
        "hidden_delay_vs_right_margin_pearson":
            pearson(hidden, right_margin),
        "hidden_delay_vs_left_margin_spearman":
            spearman(hidden, left_margin),
        "hidden_delay_vs_right_margin_spearman":
            spearman(hidden, right_margin),
        "hidden_delay_vs_left_positive_spearman":
            spearman(hidden, left_ok),
        "hidden_delay_vs_right_positive_spearman":
            spearman(hidden, right_ok),
    }

    print("WORKING_MEMORY_TRAIN_TEMPORAL_DIAGNOSIS_V1")
    print("records:", len(rows))
    print("train_condition_coverage_exact: true")
    print()

    print_group("HIDDEN DELAY STEPS", by_hidden)
    print_group("PREDECISION STEPS", by_pre)
    print_group("TOTAL TEACHER PATH STEPS", by_total)

    print("=" * 94)
    print("CORRELATIONS WITH HIDDEN DELAY")
    print("=" * 94)
    for key, value in correlations.items():
        print(f"{key}: {value:+.6f}")

    print()
    print("=" * 94)
    print("HIDDEN DELAY x CUE-VISIBLE STEPS")
    print("=" * 94)
    print(
        f"{'delay':>6} {'cue':>5} {'n':>5} "
        f"{'L+':>8} {'R+':>8} {'both+':>8} {'gap':>8}"
    )
    for (delay, cue_steps), group in sorted(by_hidden_cue.items()):
        s = summarize(group)
        print(
            f"{delay:6d} {cue_steps:5d} {s['n']:5d} "
            f"{s['left_positive']:8.4f} "
            f"{s['right_positive']:8.4f} "
            f"{s['both_positive']:8.4f} "
            f"{s['gap']:+8.4f}"
        )

    output = {
        "format": "working_memory_train_temporal_diagnosis_v1",
        "records": len(rows),
        "train_condition_coverage_exact": True,
        "correlations": correlations,
        "by_hidden_delay": {
            str(k): summarize(v)
            for k, v in sorted(by_hidden.items())
        },
        "by_predecision_steps": {
            str(k): summarize(v)
            for k, v in sorted(by_pre.items())
        },
        "by_total_teacher_steps": {
            str(k): summarize(v)
            for k, v in sorted(by_total.items())
        },
        "by_hidden_delay_x_cue_steps": {
            f"{k[0]}|{k[1]}": summarize(v)
            for k, v in sorted(by_hidden_cue.items())
        },
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))

    print()
    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_TRAIN_TEMPORAL_DIAGNOSIS_COMPLETE")


if __name__ == "__main__":
    main()
