#!/usr/bin/env python3
"""
Audit the completed 1600-update working-memory train epoch without touching DEV.

This is descriptive/training-integrity analysis only.  It verifies the claimed
without-replacement schedule and quantifies LEFT/RIGHT delayed-decision margin
symmetry over all 1600 train conditions.
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


FACTOR_FIELDS = (
    "corridor_length",
    "cue_steps",
    "move_step",
    "goal_x_offset",
    "goal_y_offset",
    "start_x",
    "start_y",
    "distractor_pattern",
)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows):
    return {
        "n": len(rows),
        "left_margin_positive_fraction": mean(
            [float(r["left_decision_margin"] > 0.0) for r in rows]
        ),
        "right_margin_positive_fraction": mean(
            [float(r["right_decision_margin"] > 0.0) for r in rows]
        ),
        "both_margins_positive_fraction": mean(
            [
                float(
                    r["left_decision_margin"] > 0.0
                    and r["right_decision_margin"] > 0.0
                )
                for r in rows
            ]
        ),
        "left_margin_mean": mean(
            [r["left_decision_margin"] for r in rows]
        ),
        "right_margin_mean": mean(
            [r["right_decision_margin"] for r in rows]
        ),
        "decision_loss_mean": mean(
            [r["decision_loss"] for r in rows]
        ),
        "loss_mean": mean([r["loss"] for r in rows]),
        "memory_max_mean": mean(
            [
                max(r["left_memory_max"], r["right_memory_max"])
                for r in rows
            ]
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        default="state/working_memory_m1/train_epoch1_v1/training.jsonl",
    )
    ap.add_argument(
        "--output",
        default="state/working_memory_m1/train_epoch1_v1/training_audit.json",
    )
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise RuntimeError(
                f"invalid JSON at line {lineno}: {exc}"
            ) from exc
        rows.append(row)

    expected_n = len(TRAIN_CONDITION_RANGE)
    if len(rows) != expected_n:
        raise AssertionError(
            f"expected exactly {expected_n} training records, got {len(rows)}"
        )

    updates = [int(r["update"]) for r in rows]
    expected_updates = list(range(1, expected_n + 1))
    if updates != expected_updates:
        raise AssertionError("training updates are not exactly 1..1600")

    condition_ids = [int(r["condition_id"]) for r in rows]
    if len(set(condition_ids)) != expected_n:
        dupes = [
            cid
            for cid, n in collections.Counter(condition_ids).items()
            if n != 1
        ]
        raise AssertionError(
            f"condition IDs not unique; duplicate/missing candidates={dupes[:20]}"
        )

    if set(condition_ids) != set(TRAIN_CONDITION_RANGE):
        missing = sorted(set(TRAIN_CONDITION_RANGE) - set(condition_ids))
        extra = sorted(set(condition_ids) - set(TRAIN_CONDITION_RANGE))
        raise AssertionError(
            f"train condition coverage mismatch missing={missing[:20]} "
            f"extra={extra[:20]}"
        )

    required_numeric = (
        "loss",
        "decision_loss",
        "left_decision_margin",
        "right_decision_margin",
        "candidate_grad_max",
        "write_gate_grad_max",
        "workspace_grad_max",
        "left_memory_max",
        "right_memory_max",
    )
    for row in rows:
        for key in required_numeric:
            x = float(row[key])
            if not math.isfinite(x):
                raise AssertionError(
                    f"non-finite {key} at update {row['update']}"
                )
        if float(row["candidate_grad_max"]) <= 0.0:
            raise AssertionError(
                f"candidate gradient missing at update {row['update']}"
            )
        if int(row["update"]) > 1 and float(
            row["write_gate_grad_max"]
        ) <= 0.0:
            raise AssertionError(
                f"write-gate gradient missing at update {row['update']}"
            )
        if row.get("workspace_exact") is not True:
            raise AssertionError(
                f"workspace_exact false at update {row['update']}"
            )

    required_base_checks = set(range(50, 1601, 50))
    observed_base_checks = {
        int(r["update"])
        for r in rows
        if r.get("full_base_verified") is True
    }
    if not required_base_checks.issubset(observed_base_checks):
        missing = sorted(required_base_checks - observed_base_checks)
        raise AssertionError(
            f"missing full base verification updates: {missing}"
        )

    overall = summarize(rows)

    quartiles = {}
    for i, (a, b) in enumerate(
        ((1, 400), (401, 800), (801, 1200), (1201, 1600)),
        1,
    ):
        qrows = [
            r for r in rows if a <= int(r["update"]) <= b
        ]
        quartiles[f"q{i}_{a}_{b}"] = summarize(qrows)

    factor_slices = {}
    for field in FACTOR_FIELDS:
        groups = collections.defaultdict(list)
        for row in rows:
            spec = generate_task_spec(int(row["condition_id"]))
            groups[getattr(spec, field)].append(row)
        factor_slices[field] = {
            str(value): summarize(group)
            for value, group in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }

    result = {
        "format": "working_memory_training_epoch_audit_v1",
        "jsonl": str(path),
        "records": len(rows),
        "condition_coverage_exact": True,
        "updates_exact": True,
        "workspace_exact_every_update": True,
        "required_full_base_checks_present": True,
        "candidate_grad_positive_every_update": True,
        "write_gate_grad_positive_after_update1": True,
        "overall": overall,
        "quartiles": quartiles,
        "factor_slices": factor_slices,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("WORKING_MEMORY_TRAINING_EPOCH_AUDIT_V1")
    print("records:", len(rows))
    print("condition_coverage_exact: true")
    print("updates_exact: true")
    print("workspace_exact_every_update: true")
    print("required_full_base_checks_present: true")
    print("candidate_grad_positive_every_update: true")
    print("write_gate_grad_positive_after_update1: true")
    print(
        "overall_left_margin_positive:",
        f"{overall['left_margin_positive_fraction']:.4f}",
    )
    print(
        "overall_right_margin_positive:",
        f"{overall['right_margin_positive_fraction']:.4f}",
    )
    print(
        "overall_both_margins_positive:",
        f"{overall['both_margins_positive_fraction']:.4f}",
    )
    print(
        "overall_left_margin_mean:",
        f"{overall['left_margin_mean']:.6f}",
    )
    print(
        "overall_right_margin_mean:",
        f"{overall['right_margin_mean']:.6f}",
    )
    for name, stats in quartiles.items():
        print(
            name,
            f"L+={stats['left_margin_positive_fraction']:.4f}",
            f"R+={stats['right_margin_positive_fraction']:.4f}",
            f"both+={stats['both_margins_positive_fraction']:.4f}",
            f"Lmean={stats['left_margin_mean']:+.4f}",
            f"Rmean={stats['right_margin_mean']:+.4f}",
        )
    print("wrote:", out)
    print("WORKING_MEMORY_TRAINING_EPOCH_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
