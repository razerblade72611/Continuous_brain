#!/usr/bin/env python3
"""Compare matched linear/gated production-smoke logs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MATCH_THROUGH_UPDATE = 3
RTOL = 3e-5
ATOL = 1e-9


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", required=True)
    ap.add_argument("--gated", required=True)
    args = ap.parse_args()

    linear = read_jsonl(args.linear)
    gated = read_jsonl(args.gated)

    if len(linear) != len(gated):
        raise AssertionError(
            f"log lengths differ {len(linear)} != {len(gated)}"
        )
    if not linear:
        raise AssertionError("empty smoke logs")

    print("WORKING_MEMORY_RECALL_SMOKE_COMPARISON_V1")
    print("records_each:", len(linear))

    for a, b in zip(linear, gated):
        if int(a["update"]) != int(b["update"]):
            raise AssertionError("update schedule mismatch")
        if int(a["condition_id"]) != int(b["condition_id"]):
            raise AssertionError(
                f"condition mismatch update {a['update']}: "
                f"{a['condition_id']} != {b['condition_id']}"
            )
        if int(a["decision_step"]) != int(b["decision_step"]):
            raise AssertionError("decision timing mismatch")

    print("condition_schedule_identical: true")
    print("decision_timing_identical: true")

    fields = (
        "loss",
        "decision_loss",
        "left_memory_max",
        "right_memory_max",
        "left_recall_max",
        "right_recall_max",
        "candidate_grad_max",
        "write_gate_grad_max",
        "recall_projection_grad_max",
    )
    for i in range(min(MATCH_THROUGH_UPDATE, len(linear))):
        a = linear[i]
        b = gated[i]
        for field in fields:
            if not close(a[field], b[field]):
                raise AssertionError(
                    f"pre-gate match failed update={i+1} "
                    f"field={field}: {a[field]} != {b[field]}"
                )
    print("neutral_paths_match_through_update3: true")

    def aggregate(rows):
        n = len(rows)
        return {
            "left_full": sum(
                bool(r["left_decision"]["full_correct"]) for r in rows
            ) / n,
            "right_full": sum(
                bool(r["right_decision"]["full_correct"]) for r in rows
            ) / n,
            "left_lateral_top": sum(
                bool(r["left_decision"]["lateral_top_choice"]) for r in rows
            ) / n,
            "right_lateral_top": sum(
                bool(r["right_decision"]["lateral_top_choice"]) for r in rows
            ) / n,
            "left_side": sum(
                bool(r["left_decision"]["side_correct"]) for r in rows
            ) / n,
            "right_side": sum(
                bool(r["right_decision"]["side_correct"]) for r in rows
            ) / n,
        }

    for name, rows in (("linear", linear), ("gated", gated)):
        s = aggregate(rows)
        print(
            f"{name}: "
            f"Lfull={s['left_full']:.3f} "
            f"Rfull={s['right_full']:.3f} "
            f"Llat={s['left_lateral_top']:.3f} "
            f"Rlat={s['right_lateral_top']:.3f} "
            f"Lside={s['left_side']:.3f} "
            f"Rside={s['right_side']:.3f}"
        )

    print("WORKING_MEMORY_RECALL_SMOKE_COMPARISON_OK")


if __name__ == "__main__":
    main()
