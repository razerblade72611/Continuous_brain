#!/usr/bin/env python3
"""
Post-hoc TRAIN-only analysis of the completed final recall causal evaluation.

Uses only existing JSON/JSONL artifacts:
  - final_train_causal_v1/{linear,gated}.json
  - phase1 training.jsonl for each arm

No model is loaded. DEV and TEST are never touched.

Analyses
--------
1) Hidden-reset route factorial:
   Z  = hidden reset + both memory routes zero
   H  = hidden reset + historical memory route only
   D  = hidden reset + direct recall only
   HD = hidden reset + both memory routes

   Reports whether HD succeeds when:
     - both H and D succeed
     - only H succeeds
     - only D succeeds
     - neither succeeds

   This distinguishes redundancy/rescue from "joint-only" synergy.

2) Fixed-final-checkpoint residual training-order effect:
   maps each condition to the update at which it appeared during the one-epoch
   training schedule, then scores the FINAL checkpoint by early/middle/late
   training thirds and computes Spearman correlations with visitation position.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


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


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def read_json(path):
    return json.loads(Path(path).read_text())


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


def pair_metric(row, mode, metric):
    l = row["modes"][mode]["left"]
    r = row["modes"][mode]["right"]

    if metric == "full":
        return bool(l["full_correct"] and r["full_correct"])
    if metric == "side":
        return bool(l["side_correct"] and r["side_correct"])
    if metric == "lateral":
        return bool(
            l["lateral_top_choice"] and r["lateral_top_choice"]
        )
    raise ValueError(metric)


def member_margin(row, mode):
    return 0.5 * (
        float(row["modes"][mode]["left"]["correct_side_margin"])
        + float(row["modes"][mode]["right"]["correct_side_margin"])
    )


def route_factorial(rows, metric):
    z = "hidden_reset_all_zero"
    h = "hidden_reset_historical_only"
    d = "hidden_reset_direct_only"
    hd = "hidden_reset_both_memory_paths"

    categories = {
        "both_isolated_correct": [],
        "historical_only_correct": [],
        "direct_only_correct": [],
        "neither_isolated_correct": [],
    }

    joint_only_success = 0
    joint_only_failure = 0
    rescue_from_historical = 0
    rescue_from_direct = 0
    interference_when_any_isolated_correct = 0

    interaction_margins = []

    for row in rows:
        hb = pair_metric(row, h, metric)
        db = pair_metric(row, d, metric)
        hdb = pair_metric(row, hd, metric)

        if hb and db:
            key = "both_isolated_correct"
        elif hb:
            key = "historical_only_correct"
        elif db:
            key = "direct_only_correct"
        else:
            key = "neither_isolated_correct"

        categories[key].append(hdb)

        if not hb and not db:
            if hdb:
                joint_only_success += 1
            else:
                joint_only_failure += 1
        elif hb and not db and hdb:
            rescue_from_historical += 1
        elif db and not hb and hdb:
            rescue_from_direct += 1

        if (hb or db) and not hdb:
            interference_when_any_isolated_correct += 1

        # Additive interaction on average correct-side margin:
        # HD - H - D + Z.  Positive means the combined routes create more
        # margin than the sum of their isolated contributions at zero hidden.
        interaction_margins.append(
            member_margin(row, hd)
            - member_margin(row, h)
            - member_margin(row, d)
            + member_margin(row, z)
        )

    cat_summary = {}
    for key, vals in categories.items():
        cat_summary[key] = {
            "n": len(vals),
            "joint_success": sum(bool(x) for x in vals),
            "joint_success_fraction": mean([float(x) for x in vals]),
        }

    return {
        "metric": metric,
        "categories": cat_summary,
        "joint_only_success": joint_only_success,
        "joint_only_failure": joint_only_failure,
        "rescue_from_historical": rescue_from_historical,
        "rescue_from_direct": rescue_from_direct,
        "interference_when_any_isolated_correct":
            interference_when_any_isolated_correct,
        "mean_additive_margin_interaction": mean(interaction_margins),
        "positive_additive_margin_interaction_fraction": mean(
            [float(x > 0.0) for x in interaction_margins]
        ),
    }


def final_order_effect(causal_rows, training_rows):
    if len(causal_rows) != 1600 or len(training_rows) != 1600:
        raise AssertionError("expected 1600 causal and 1600 training rows")

    update_for_condition = {}
    for row in training_rows:
        cid = int(row["condition_id"])
        upd = int(row["update"])
        if cid in update_for_condition:
            raise AssertionError(f"duplicate condition in training log: {cid}")
        update_for_condition[cid] = upd

    if len(update_for_condition) != 1600:
        raise AssertionError("training condition coverage is not 1600 unique")

    enriched = []
    for row in causal_rows:
        cid = int(row["condition_id"])
        if cid not in update_for_condition:
            raise AssertionError(f"causal condition absent from training log: {cid}")
        upd = update_for_condition[cid]
        normal = row["modes"]["normal"]
        enriched.append({
            "condition_id": cid,
            "update": upd,
            "pair_full": float(
                normal["left"]["full_correct"]
                and normal["right"]["full_correct"]
            ),
            "left_full": float(normal["left"]["full_correct"]),
            "right_full": float(normal["right"]["full_correct"]),
            "pair_side": float(
                normal["left"]["side_correct"]
                and normal["right"]["side_correct"]
            ),
            "pair_margin": 0.5 * (
                float(normal["left"]["correct_side_margin"])
                + float(normal["right"]["correct_side_margin"])
            ),
        })

    enriched.sort(key=lambda r: r["update"])

    thirds = {
        "early_1_533": [r for r in enriched if 1 <= r["update"] <= 533],
        "middle_534_1066": [
            r for r in enriched if 534 <= r["update"] <= 1066
        ],
        "late_1067_1600": [
            r for r in enriched if 1067 <= r["update"] <= 1600
        ],
    }

    def summarize(group):
        return {
            "n": len(group),
            "pair_full": mean([r["pair_full"] for r in group]),
            "left_full": mean([r["left_full"] for r in group]),
            "right_full": mean([r["right_full"] for r in group]),
            "pair_side": mean([r["pair_side"] for r in group]),
            "pair_margin_mean": mean([r["pair_margin"] for r in group]),
        }

    updates = [float(r["update"]) for r in enriched]
    return {
        "thirds": {k: summarize(v) for k, v in thirds.items()},
        "spearman_update_vs_pair_full":
            spearman(updates, [r["pair_full"] for r in enriched]),
        "spearman_update_vs_left_full":
            spearman(updates, [r["left_full"] for r in enriched]),
        "spearman_update_vs_right_full":
            spearman(updates, [r["right_full"] for r in enriched]),
        "spearman_update_vs_pair_margin":
            spearman(updates, [r["pair_margin"] for r in enriched]),
        "late_minus_early_pair_full": (
            summarize(thirds["late_1067_1600"])["pair_full"]
            - summarize(thirds["early_1_533"])["pair_full"]
        ),
        "late_minus_early_left_full": (
            summarize(thirds["late_1067_1600"])["left_full"]
            - summarize(thirds["early_1_533"])["left_full"]
        ),
        "late_minus_early_right_full": (
            summarize(thirds["late_1067_1600"])["right_full"]
            - summarize(thirds["early_1_533"])["right_full"]
        ),
    }


def analyze_arm(name, causal_path, training_path):
    causal = read_json(causal_path)
    if causal.get("split") != "train":
        raise RuntimeError(f"{name}: causal file is not TRAIN")
    if causal.get("dev_touched") or causal.get("test_touched"):
        raise RuntimeError(f"{name}: held-out contamination flag")
    rows = causal["per_condition"]

    training = read_jsonl(training_path)

    return {
        "arm": name,
        "route_factorial_full": route_factorial(rows, "full"),
        "route_factorial_side": route_factorial(rows, "side"),
        "route_factorial_lateral": route_factorial(rows, "lateral"),
        "final_order_effect": final_order_effect(rows, training),
    }


def print_factorial(name, result, metric_key):
    f = result[metric_key]
    print(f"{name} {f['metric'].upper()} ROUTE FACTORIAL")
    for key, s in f["categories"].items():
        print(
            f"  {key}: n={s['n']} "
            f"joint_success={s['joint_success']} "
            f"fraction={s['joint_success_fraction']:.4f}"
        )
    print(
        f"  joint_only_success={f['joint_only_success']} "
        f"joint_only_failure={f['joint_only_failure']}"
    )
    print(
        f"  interference_when_any_isolated_correct="
        f"{f['interference_when_any_isolated_correct']}"
    )
    print(
        f"  mean_additive_margin_interaction="
        f"{f['mean_additive_margin_interaction']:+.6f}"
    )
    print(
        f"  positive_additive_margin_interaction_fraction="
        f"{f['positive_additive_margin_interaction_fraction']:.4f}"
    )


def print_order(name, o):
    print(f"{name} FIXED-FINAL TRAINING-ORDER CHECK")
    for key, s in o["thirds"].items():
        print(
            f"  {key}: n={s['n']} "
            f"pairF={s['pair_full']:.4f} "
            f"Lfull={s['left_full']:.4f} "
            f"Rfull={s['right_full']:.4f} "
            f"pairS={s['pair_side']:.4f} "
            f"margin={s['pair_margin_mean']:+.4f}"
        )
    print(
        f"  rho(update,pair_full)="
        f"{o['spearman_update_vs_pair_full']:+.6f}"
    )
    print(
        f"  rho(update,left_full)="
        f"{o['spearman_update_vs_left_full']:+.6f}"
    )
    print(
        f"  rho(update,right_full)="
        f"{o['spearman_update_vs_right_full']:+.6f}"
    )
    print(
        f"  rho(update,pair_margin)="
        f"{o['spearman_update_vs_pair_margin']:+.6f}"
    )
    print(
        f"  late_minus_early_pair_full="
        f"{o['late_minus_early_pair_full']:+.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--linear-causal",
        default=(
            "state/working_memory_recall_phase1_v1/"
            "final_train_causal_v1/linear.json"
        ),
    )
    ap.add_argument(
        "--gated-causal",
        default=(
            "state/working_memory_recall_phase1_v1/"
            "final_train_causal_v1/gated.json"
        ),
    )
    ap.add_argument(
        "--linear-training",
        default=(
            "state/working_memory_recall_phase1_v1/"
            "linear/training.jsonl"
        ),
    )
    ap.add_argument(
        "--gated-training",
        default=(
            "state/working_memory_recall_phase1_v1/"
            "gated/training.jsonl"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "state/working_memory_recall_phase1_v1/"
            "final_train_causal_v1/posthoc_route_order_analysis_v1.json"
        ),
    )
    args = ap.parse_args()

    linear = analyze_arm(
        "linear",
        args.linear_causal,
        args.linear_training,
    )
    gated = analyze_arm(
        "gated",
        args.gated_causal,
        args.gated_training,
    )

    result = {
        "format": "working_memory_recall_posthoc_route_order_v1",
        "linear": linear,
        "gated": gated,
        "train_only": True,
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("WORKING_MEMORY_RECALL_POSTHOC_ROUTE_ORDER_V1")
    print()
    for name, arm in (("linear", linear), ("gated", gated)):
        print_factorial(name, arm, "route_factorial_full")
        print_factorial(name, arm, "route_factorial_side")
        print_order(name, arm["final_order_effect"])
        print()

    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_RECALL_POSTHOC_ROUTE_ORDER_COMPLETE")


if __name__ == "__main__":
    main()
