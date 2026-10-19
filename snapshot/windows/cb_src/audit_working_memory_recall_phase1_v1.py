#!/usr/bin/env python3
"""
Train-only matched audit for the 1600-update recall-arm comparison.

Reads:
- M1.5 / linear production JSONL
- M2 / gated production JSONL
- optional existing M1 writer-only JSONL for side-sign baseline only

No model inference. No DEV. No TEST.
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


def read_jsonl(path):
    rows = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(
                f"{path}: invalid JSON at line {lineno}: {exc}"
            ) from exc
    return rows


def validate_arm(rows, arm):
    n = len(TRAIN_CONDITION_RANGE)
    if len(rows) != n:
        raise AssertionError(f"{arm}: expected {n} rows, got {len(rows)}")

    updates = [int(r["update"]) for r in rows]
    if updates != list(range(1, n + 1)):
        raise AssertionError(f"{arm}: updates are not exactly 1..{n}")

    ids = [int(r["condition_id"]) for r in rows]
    if len(set(ids)) != n or set(ids) != set(TRAIN_CONDITION_RANGE):
        raise AssertionError(f"{arm}: train condition coverage mismatch")

    if any(r.get("arm") != arm for r in rows):
        raise AssertionError(f"{arm}: arm label mismatch")

    for r in rows:
        u = int(r["update"])
        if r.get("workspace_exact") is not True:
            raise AssertionError(f"{arm}: workspace_exact false update {u}")
        if float(r["candidate_grad_max"]) <= 0.0:
            raise AssertionError(f"{arm}: candidate grad missing update {u}")
        if u > 1 and float(r["write_gate_grad_max"]) <= 0.0:
            raise AssertionError(f"{arm}: writer gate grad missing update {u}")
        if u > 1 and float(r["recall_projection_grad_max"]) <= 0.0:
            raise AssertionError(f"{arm}: projection grad missing update {u}")
        if arm == "gated" and u >= 3:
            if float(r["recall_gate_weight_grad_max"]) <= 0.0:
                raise AssertionError(f"{arm}: gate weight grad missing update {u}")
            if float(r["recall_gate_bias_grad_max"]) <= 0.0:
                raise AssertionError(f"{arm}: gate bias grad missing update {u}")

    required_verify = set(range(50, n + 1, 50))
    observed_verify = {
        int(r["update"])
        for r in rows
        if r.get("full_base_verified") is True
    }
    if not required_verify.issubset(observed_verify):
        raise AssertionError(
            f"{arm}: missing base verifications "
            f"{sorted(required_verify - observed_verify)}"
        )

    return ids


def realized_delay(row):
    spec = generate_task_spec(int(row["condition_id"]))
    return int(row["decision_step"]) - int(spec.cue_steps)


def summarize_arm(rows):
    n = len(rows)
    lf = [bool(r["left_decision"]["full_correct"]) for r in rows]
    rf = [bool(r["right_decision"]["full_correct"]) for r in rows]
    ll = [bool(r["left_decision"]["lateral_top_choice"]) for r in rows]
    rl = [bool(r["right_decision"]["lateral_top_choice"]) for r in rows]
    ls = [bool(r["left_decision"]["side_correct"]) for r in rows]
    rs = [bool(r["right_decision"]["side_correct"]) for r in rows]

    return {
        "n": n,
        "left_full": mean([float(x) for x in lf]),
        "right_full": mean([float(x) for x in rf]),
        "pair_full": mean([float(a and b) for a, b in zip(lf, rf)]),
        "left_lateral_top": mean([float(x) for x in ll]),
        "right_lateral_top": mean([float(x) for x in rl]),
        "pair_lateral_top": mean([float(a and b) for a, b in zip(ll, rl)]),
        "left_side": mean([float(x) for x in ls]),
        "right_side": mean([float(x) for x in rs]),
        "pair_side": mean([float(a and b) for a, b in zip(ls, rs)]),
        "left_margin_mean": mean([
            float(r["left_decision"]["correct_side_margin"]) for r in rows
        ]),
        "right_margin_mean": mean([
            float(r["right_decision"]["correct_side_margin"]) for r in rows
        ]),
        "loss_mean": mean([float(r["loss"]) for r in rows]),
        "memory_max_mean": mean([
            max(float(r["left_memory_max"]), float(r["right_memory_max"]))
            for r in rows
        ]),
        "recall_max_mean": mean([
            max(float(r["left_recall_max"]), float(r["right_recall_max"]))
            for r in rows
        ]),
    }


def summarize_m1_side(rows):
    return {
        "n": len(rows),
        "left_side": mean([
            float(float(r["left_decision_margin"]) > 0.0) for r in rows
        ]),
        "right_side": mean([
            float(float(r["right_decision_margin"]) > 0.0) for r in rows
        ]),
        "pair_side": mean([
            float(
                float(r["left_decision_margin"]) > 0.0
                and float(r["right_decision_margin"]) > 0.0
            )
            for r in rows
        ]),
        "left_margin_mean": mean([
            float(r["left_decision_margin"]) for r in rows
        ]),
        "right_margin_mean": mean([
            float(r["right_decision_margin"]) for r in rows
        ]),
    }


def by_delay(rows, *, old_m1=False):
    groups = collections.defaultdict(list)
    for r in rows:
        if old_m1:
            spec = generate_task_spec(int(r["condition_id"]))
            d = int(r["decision_step"]) - int(spec.cue_steps)
        else:
            d = realized_delay(r)
        groups[d].append(r)

    if old_m1:
        return {
            str(k): summarize_m1_side(v)
            for k, v in sorted(groups.items())
        }
    return {
        str(k): summarize_arm(v)
        for k, v in sorted(groups.items())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", required=True)
    ap.add_argument("--gated", required=True)
    ap.add_argument("--m1-jsonl", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    linear = read_jsonl(args.linear)
    gated = read_jsonl(args.gated)

    linear_ids = validate_arm(linear, "linear")
    gated_ids = validate_arm(gated, "gated")

    if linear_ids != gated_ids:
        raise AssertionError("linear/gated condition order differs")

    for a, b in zip(linear, gated):
        if int(a["decision_step"]) != int(b["decision_step"]):
            raise AssertionError(
                f"decision timing differs update {a['update']}"
            )

    result = {
        "format": "working_memory_recall_phase1_audit_v1",
        "records_each": len(linear),
        "condition_order_identical": True,
        "decision_timing_identical": True,
        "train_coverage_exact": True,
        "linear": {
            "overall": summarize_arm(linear),
            "by_realized_delay": by_delay(linear),
        },
        "gated": {
            "overall": summarize_arm(gated),
            "by_realized_delay": by_delay(gated),
        },
        "dev_touched": False,
        "test_touched": False,
    }

    if args.m1_jsonl:
        m1 = read_jsonl(args.m1_jsonl)
        if len(m1) != len(TRAIN_CONDITION_RANGE):
            raise AssertionError(
                f"M1 baseline expected 1600 rows, got {len(m1)}"
            )
        m1_ids = [int(r["condition_id"]) for r in m1]
        if set(m1_ids) != set(TRAIN_CONDITION_RANGE):
            raise AssertionError("M1 baseline train coverage mismatch")
        result["m1_side_baseline"] = {
            "overall": summarize_m1_side(m1),
            "by_realized_delay": by_delay(m1, old_m1=True),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("WORKING_MEMORY_RECALL_PHASE1_AUDIT_V1")
    print("records_each:", len(linear))
    print("condition_order_identical: true")
    print("decision_timing_identical: true")
    print("train_coverage_exact: true")

    for name, rows in (("linear", linear), ("gated", gated)):
        s = summarize_arm(rows)
        print(
            f"{name}: "
            f"pair_full={s['pair_full']:.4f} "
            f"Lfull={s['left_full']:.4f} "
            f"Rfull={s['right_full']:.4f} "
            f"pair_lat={s['pair_lateral_top']:.4f} "
            f"Llat={s['left_lateral_top']:.4f} "
            f"Rlat={s['right_lateral_top']:.4f} "
            f"pair_side={s['pair_side']:.4f} "
            f"Lside={s['left_side']:.4f} "
            f"Rside={s['right_side']:.4f}"
        )

    if "m1_side_baseline" in result:
        s = result["m1_side_baseline"]["overall"]
        print(
            "m1_side_baseline: "
            f"pair_side={s['pair_side']:.4f} "
            f"Lside={s['left_side']:.4f} "
            f"Rside={s['right_side']:.4f}"
        )

    print()
    print("REALIZED_DELAY_SIDE_AND_FULL")
    delays = sorted(
        set(int(k) for k in result["linear"]["by_realized_delay"])
        | set(int(k) for k in result["gated"]["by_realized_delay"])
    )
    print(
        f"{'delay':>5} | "
        f"{'linear pairF/L/R':>22} | "
        f"{'linear pairS/L/R':>22} | "
        f"{'gated pairF/L/R':>22} | "
        f"{'gated pairS/L/R':>22}"
    )
    for d in delays:
        l = result["linear"]["by_realized_delay"].get(str(d))
        g = result["gated"]["by_realized_delay"].get(str(d))
        def fmt(s, key):
            if s is None:
                return "      -"
            if key == "full":
                return (
                    f"{s['pair_full']:.2f}/"
                    f"{s['left_full']:.2f}/"
                    f"{s['right_full']:.2f}"
                )
            return (
                f"{s['pair_side']:.2f}/"
                f"{s['left_side']:.2f}/"
                f"{s['right_side']:.2f}"
            )
        print(
            f"{d:5d} | "
            f"{fmt(l,'full'):>22} | "
            f"{fmt(l,'side'):>22} | "
            f"{fmt(g,'full'):>22} | "
            f"{fmt(g,'side'):>22}"
        )

    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_RECALL_PHASE1_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
