#!/usr/bin/env python3
"""
Audit the VariedCuedDetourWorld task family BEFORE any model training.

This is an admission gate for the *environment generator*, not for a neural
mechanism.  It answers the exact question that the old corridor experiment
failed to ask:

    Do nominal condition IDs correspond to many genuinely different delayed
    decision problems, or do they collapse to a handful of effective cases?

Checks:
  * no condition_id reuse/wrap;
  * left/right members share all non-cue variables;
  * pre-junction teacher actions are identical;
  * after the cue is hidden, paired observations and RGB renders are exactly
    identical until the first cue-dependent decision;
  * first cue-dependent actions are opposite LEFT/RIGHT commitments;
  * both teacher trajectories solve their member;
  * exact structural-signature count;
  * conservative translation-invariant behavioral-signature count;
  * hidden-sensory-trace hash count;
  * factor-level coverage;
  * duplicate concentration.

No neural model is loaded and no optimizer step occurs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    DEV_CONDITION_RANGE,
    FAMILY_SIZE,
    LEFT,
    RIGHT,
    TEST_CONDITION_RANGE,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
    generate_task_spec,
    split_name,
)


def hash_update_array(h, arr):
    a = np.asarray(arr)
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())


def run_pair(condition_id, image_size):
    left = VariedCuedDetourWorld(
        condition_id,
        cue="left",
    )
    right = VariedCuedDetourWorld(
        condition_id,
        cue="right",
    )

    if left.spec != right.spec:
        raise AssertionError("pair members have different non-cue specs")

    obs_l = left._obs()
    obs_r = right._obs()
    rgb_l = left.render_rgb(image_size)
    rgb_r = right.render_rgb(image_size)

    # Initial non-cue state must be exact.
    if not np.array_equal(obs_l[[0, 1, 4, 5]], obs_r[[0, 1, 4, 5]]):
        raise AssertionError("pair initial non-cue observation mismatch")
    if np.array_equal(obs_l, obs_r):
        raise AssertionError("initial cue failed to alter cue-bearing observation")
    if np.array_equal(rgb_l, rgb_r):
        raise AssertionError("initial visual cue failed to differ")

    hidden_trace = hashlib.sha256()
    hidden_equal_steps = 0
    predecision_steps = 0

    while True:
        # Once the cue window is over, the pair must be sensor-identical
        # until the first cue-dependent action.
        if left.steps >= left.cue_steps:
            obs_l = left._obs()
            obs_r = right._obs()
            rgb_l = left.render_rgb(image_size)
            rgb_r = right.render_rgb(image_size)

            if not np.array_equal(obs_l, obs_r):
                raise AssertionError(
                    f"hidden observation cue leak at condition {condition_id}, "
                    f"step {left.steps}"
                )
            if not np.array_equal(rgb_l, rgb_r):
                raise AssertionError(
                    f"hidden render cue leak at condition {condition_id}, "
                    f"step {left.steps}"
                )

            hash_update_array(hidden_trace, obs_l)
            hash_update_array(hidden_trace, rgb_l)
            hidden_equal_steps += 1

        action_l = int(left.teacher_action())
        action_r = int(right.teacher_action())

        lateral_l = action_l in (LEFT, RIGHT)
        lateral_r = action_r in (LEFT, RIGHT)

        if lateral_l or lateral_r:
            if not (lateral_l and lateral_r):
                raise AssertionError(
                    "only one member reached the decision boundary"
                )
            if action_l != LEFT or action_r != RIGHT:
                raise AssertionError(
                    f"first cue-dependent actions were not LEFT/RIGHT: "
                    f"{ACTION_NAMES[action_l]} / {ACTION_NAMES[action_r]}"
                )
            break

        if action_l != action_r:
            raise AssertionError(
                f"pre-junction action mismatch: "
                f"{ACTION_NAMES[action_l]} vs {ACTION_NAMES[action_r]}"
            )

        obs_l, _, done_l, _ = left.step(action_l)
        obs_r, _, done_r, _ = right.step(action_r)
        if done_l or done_r:
            raise AssertionError("pair terminated before cue-dependent decision")
        predecision_steps += 1

        if predecision_steps > 100:
            raise RuntimeError("predecision loop exceeded safety bound")

    if hidden_equal_steps < 3:
        raise AssertionError(
            f"condition {condition_id} had only {hidden_equal_steps} "
            "cue-hidden identical steps"
        )

    first_decision_left = action_l
    first_decision_right = action_r

    def finish(env):
        action_hist = collections.Counter()
        steps_before = env.steps
        info = {"reached_correct": False}
        safety = 0
        while not env.done:
            action = int(env.teacher_action())
            action_hist[ACTION_NAMES[action]] += 1
            _, _, _, info = env.step(action)
            safety += 1
            if safety > 120:
                raise RuntimeError("teacher rollout exceeded safety bound")
        if not info.get("reached_correct", False):
            raise AssertionError(
                f"teacher failed condition={condition_id} cue={env.cue}"
            )
        return {
            "continuation_steps": int(env.steps - steps_before),
            "total_steps": int(env.steps),
            "action_hist": dict(sorted(action_hist.items())),
        }

    left_finish = finish(left)
    right_finish = finish(right)

    # The family is deliberately mirror-symmetric in relative geometry.
    # The absolute action names differ by cue, but path lengths should match.
    if left_finish["total_steps"] != right_finish["total_steps"]:
        raise AssertionError(
            f"paired teacher path length asymmetry at {condition_id}: "
            f"{left_finish['total_steps']} vs {right_finish['total_steps']}"
        )

    actual_behavioral_signature = (
        int(predecision_steps),
        int(hidden_equal_steps),
        int(left_finish["total_steps"]),
        int(right_finish["total_steps"]),
    )

    return {
        "condition_id": int(condition_id),
        "split": split_name(condition_id),
        "spec_sha256": left.spec.sha256(),
        "structural_signature": list(left.spec.structural_signature()),
        "declared_behavioral_signature": list(
            left.spec.behavioral_signature()
        ),
        "actual_behavioral_signature": list(
            actual_behavioral_signature
        ),
        "hidden_trace_sha256": hidden_trace.hexdigest(),
        "hidden_equal_steps": int(hidden_equal_steps),
        "predecision_steps": int(predecision_steps),
        "first_decision_left": ACTION_NAMES[first_decision_left],
        "first_decision_right": ACTION_NAMES[first_decision_right],
        "left_total_steps": left_finish["total_steps"],
        "right_total_steps": right_finish["total_steps"],
        "left_action_hist": left_finish["action_hist"],
        "right_action_hist": right_finish["action_hist"],
        "spec": left.spec.to_dict(),
    }


def factor_coverage(rows):
    fields = (
        "start_x",
        "start_y",
        "corridor_length",
        "goal_x_offset",
        "goal_y_offset",
        "cue_steps",
        "move_step",
        "distractor_pattern",
    )
    out = {}
    for field in fields:
        counts = collections.Counter(
            row["spec"][field] for row in rows
        )
        out[field] = {
            "levels": len(counts),
            "counts": {
                str(k): int(v)
                for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))
            },
        }
    return out


def most_common_signature(rows, field, limit=10):
    counts = collections.Counter(
        tuple(row[field]) if isinstance(row[field], list) else row[field]
        for row in rows
    )
    return [
        {"signature": list(sig) if isinstance(sig, tuple) else sig, "count": n}
        for sig, n in counts.most_common(limit)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--conditions", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument(
        "--min-structural-fraction",
        type=float,
        default=1.0,
        help="required unique structural signatures / requested conditions",
    )
    ap.add_argument(
        "--min-behavioral-fraction",
        type=float,
        default=0.70,
        help="required unique conservative behavioral signatures / conditions",
    )
    ap.add_argument(
        "--min-hidden-trace-fraction",
        type=float,
        default=0.90,
        help="required unique cue-hidden sensory trace hashes / conditions",
    )
    ap.add_argument(
        "--output",
        default="state/working_memory_m1/task_family_audit.json",
    )
    args = ap.parse_args()

    if args.conditions < 1:
        raise ValueError("--conditions must be >= 1")
    stop = args.start_id + args.conditions
    if args.start_id < 0 or stop > FAMILY_SIZE:
        raise ValueError(
            f"requested IDs [{args.start_id},{stop}) exceed "
            f"finite family [0,{FAMILY_SIZE})"
        )

    ids = list(range(args.start_id, stop))
    rows = []

    print("VARIED_CUED_DETOUR_FAMILY_AUDIT")
    print("family_size:", FAMILY_SIZE)
    print(
        "frozen_splits:",
        f"train={TRAIN_CONDITION_RANGE.start}:{TRAIN_CONDITION_RANGE.stop}",
        f"dev={DEV_CONDITION_RANGE.start}:{DEV_CONDITION_RANGE.stop}",
        f"test={TEST_CONDITION_RANGE.start}:{TEST_CONDITION_RANGE.stop}",
    )
    print("audit_condition_ids:", f"{ids[0]}..{ids[-1]}")
    print("conditions:", len(ids))
    print("image_size:", args.image_size)

    for i, condition_id in enumerate(ids, 1):
        rows.append(run_pair(condition_id, args.image_size))
        if i == 1 or i % 32 == 0 or i == len(ids):
            print(f"audited={i}/{len(ids)}")

    structural = {
        tuple(row["structural_signature"]) for row in rows
    }
    declared_behavioral = {
        tuple(row["declared_behavioral_signature"])
        for row in rows
    }
    actual_behavioral = {
        tuple(row["actual_behavioral_signature"])
        for row in rows
    }
    hidden_traces = {
        row["hidden_trace_sha256"] for row in rows
    }
    spec_hashes = {
        row["spec_sha256"] for row in rows
    }

    n = len(rows)
    structural_fraction = len(structural) / n
    declared_behavioral_fraction = len(declared_behavioral) / n
    actual_behavioral_fraction = len(actual_behavioral) / n
    hidden_trace_fraction = len(hidden_traces) / n
    spec_hash_fraction = len(spec_hashes) / n

    coverage = factor_coverage(rows)
    split_counts = collections.Counter(row["split"] for row in rows)

    manifest_digest = hashlib.sha256()
    for row in rows:
        manifest_digest.update(
            json.dumps(
                row["spec"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    manifest_sha256 = manifest_digest.hexdigest()

    checks = {
        "all_spec_hashes_unique": len(spec_hashes) == n,
        "structural_fraction_ok":
            structural_fraction >= args.min_structural_fraction,
        "declared_behavioral_fraction_ok":
            declared_behavioral_fraction >= args.min_behavioral_fraction,
        "actual_behavioral_fraction_ok":
            actual_behavioral_fraction >= args.min_behavioral_fraction,
        "hidden_trace_fraction_ok":
            hidden_trace_fraction >= args.min_hidden_trace_fraction,
        "pair_integrity_all_passed": True,
        "teacher_pair_success_all_passed": True,
    }
    passed = all(checks.values())

    result = {
        "format": "varied_cued_detour_family_audit_v1",
        "family_size": FAMILY_SIZE,
        "splits": {
            "train": [
                TRAIN_CONDITION_RANGE.start,
                TRAIN_CONDITION_RANGE.stop,
            ],
            "dev": [
                DEV_CONDITION_RANGE.start,
                DEV_CONDITION_RANGE.stop,
            ],
            "test": [
                TEST_CONDITION_RANGE.start,
                TEST_CONDITION_RANGE.stop,
            ],
        },
        "audit_start_id": args.start_id,
        "conditions": n,
        "image_size": args.image_size,
        "manifest_sha256": manifest_sha256,
        "unique_counts": {
            "spec_hashes": len(spec_hashes),
            "structural_signatures": len(structural),
            "declared_behavioral_signatures": len(declared_behavioral),
            "actual_behavioral_signatures": len(actual_behavioral),
            "hidden_trace_hashes": len(hidden_traces),
        },
        "unique_fractions": {
            "spec_hashes": spec_hash_fraction,
            "structural_signatures": structural_fraction,
            "declared_behavioral_signatures":
                declared_behavioral_fraction,
            "actual_behavioral_signatures":
                actual_behavioral_fraction,
            "hidden_trace_hashes": hidden_trace_fraction,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "factor_coverage": coverage,
        "top_declared_behavioral_duplicates":
            most_common_signature(
                rows, "declared_behavioral_signature"
            ),
        "top_actual_behavioral_duplicates":
            most_common_signature(
                rows, "actual_behavioral_signature"
            ),
        "checks": checks,
        "passed": passed,
        "rows": rows,
    }

    print()
    print("manifest_sha256:", manifest_sha256)
    print("spec_hashes_unique:", f"{len(spec_hashes)}/{n}")
    print(
        "structural_signatures_unique:",
        f"{len(structural)}/{n}",
        f"fraction={structural_fraction:.4f}",
    )
    print(
        "declared_behavioral_signatures_unique:",
        f"{len(declared_behavioral)}/{n}",
        f"fraction={declared_behavioral_fraction:.4f}",
    )
    print(
        "actual_behavioral_signatures_unique:",
        f"{len(actual_behavioral)}/{n}",
        f"fraction={actual_behavioral_fraction:.4f}",
    )
    print(
        "hidden_trace_hashes_unique:",
        f"{len(hidden_traces)}/{n}",
        f"fraction={hidden_trace_fraction:.4f}",
    )

    print()
    print("factor_levels:")
    for field, info in coverage.items():
        print(f"  {field}: {info['levels']}")

    print()
    for name, ok in checks.items():
        print(f"{name}: {str(ok).lower()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("wrote:", out)

    if not passed:
        print("TASK_FAMILY_ADMISSION: FAIL")
        raise SystemExit(2)

    print("TASK_FAMILY_ADMISSION: PASS")
    print("VARIED_CUED_DETOUR_FAMILY_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
