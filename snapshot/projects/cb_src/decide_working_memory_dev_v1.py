#!/usr/bin/env python3
"""
Pre-registered DEV admission decision for Working Memory M1.

This script is intentionally model-free.  It re-derives metrics from the
per-condition DEV JSON produced by evaluate_working_memory_dev_v1.py.

Thresholds were frozen before the DEV evaluator was run.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path


EXPECTED_FORMAT = "working_memory_dev_causal_eval_v1"
EXPECTED_DEV_IDS = set(range(1600, 1980))
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 94173

# Pre-registered admission thresholds.
MIN_NORMAL_PAIR_FULL = 0.70
MIN_NORMAL_PAIR_FULL_WILSON_LOWER = 0.65
MIN_MEMBER_FULL = 0.80
MAX_OVERALL_LEFT_RIGHT_GAP = 0.10

MIN_MEMORY_ADDED_PAIR_GAIN = 0.10
MAX_MEMORY_GAIN_P_ONE_SIDED = 0.01

MIN_HIDDEN_RESET_OWN_PAIR_SIDE = 0.65
MIN_HIDDEN_RESET_SWAP_DONOR_PAIR = 0.65

MIN_SWAP_POSITIVE_FRACTION = 0.70
MAX_SWAP_SIGN_P_ONE_SIDED = 0.001

MIN_FACTOR_PAIR_FULL = 0.50
MAX_FACTOR_DROP_FROM_OVERALL = 0.25
MAX_FACTOR_LEFT_RIGHT_GAP = 0.25

BOTH_RESET_MAX_LOGIT_DIFF = 1e-6


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


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n + z * z / (4.0 * n * n)
        )
        / den
    )
    return (max(0.0, center - half), min(1.0, center + half))


def binom_upper_tail(k, n, p=0.5):
    if n == 0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(n, i)
            * (p ** i)
            * ((1.0 - p) ** (n - i))
            for i in range(k, n + 1)
        ),
    )


def bootstrap_mean_ci(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    append = means.append
    for _ in range(draws):
        append(
            sum(values[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    lo = means[int(0.025 * (draws - 1))]
    hi = means[int(0.975 * (draws - 1))]
    return (lo, hi)


def mode_stats(rows, mode):
    n = len(rows)
    left = [
        bool(r["modes"][mode]["left"]["full_correct"])
        for r in rows
    ]
    right = [
        bool(r["modes"][mode]["right"]["full_correct"])
        for r in rows
    ]
    pair = [a and b for a, b in zip(left, right)]

    left_side = [
        bool(r["modes"][mode]["left"]["side_correct"])
        for r in rows
    ]
    right_side = [
        bool(r["modes"][mode]["right"]["side_correct"])
        for r in rows
    ]
    pair_side = [
        a and b for a, b in zip(left_side, right_side)
    ]

    return {
        "n": n,
        "left_full": mean([float(x) for x in left]),
        "right_full": mean([float(x) for x in right]),
        "pair_full": mean([float(x) for x in pair]),
        "left_side": mean([float(x) for x in left_side]),
        "right_side": mean([float(x) for x in right_side]),
        "pair_side": mean([float(x) for x in pair_side]),
        "left_full_bits": left,
        "right_full_bits": right,
        "pair_full_bits": pair,
    }


def paired_gain_test(a_bits, b_bits):
    # a is hypothesized better than b.
    a_only = sum(a and not b for a, b in zip(a_bits, b_bits))
    b_only = sum(b and not a for a, b in zip(a_bits, b_bits))
    discordant = a_only + b_only
    p = binom_upper_tail(a_only, discordant, 0.5)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_one_sided": p,
    }


def factor_slices(rows, overall_pair):
    result = {}
    failures = []

    for field in FACTOR_FIELDS:
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r["spec"][field]].append(r)

        field_result = {}
        for value, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
            normal = mode_stats(group, "normal")
            hidden_own = mode_stats(group, "hidden_reset_own")
            swap_effect = mean(
                [float(r["swap"]["condition_effect"]) for r in group]
            )

            min_allowed = max(
                MIN_FACTOR_PAIR_FULL,
                overall_pair - MAX_FACTOR_DROP_FROM_OVERALL,
            )
            pair_ok = normal["pair_full"] >= min_allowed
            gap = abs(normal["left_full"] - normal["right_full"])
            gap_ok = gap <= MAX_FACTOR_LEFT_RIGHT_GAP

            entry = {
                "n": len(group),
                "normal_pair_full": normal["pair_full"],
                "normal_left_full": normal["left_full"],
                "normal_right_full": normal["right_full"],
                "normal_left_right_gap": gap,
                "minimum_allowed_pair_full": min_allowed,
                "hidden_reset_own_pair_side": hidden_own["pair_side"],
                "mean_swap_effect": swap_effect,
                "pair_threshold_pass": pair_ok,
                "left_right_gap_pass": gap_ok,
                "pass": pair_ok and gap_ok,
            }
            field_result[str(value)] = entry

            if not entry["pass"]:
                failures.append(
                    {
                        "factor": field,
                        "value": value,
                        "entry": entry,
                    }
                )

        result[field] = field_result

    return result, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="state/working_memory_m1/dev_eval_v1.json",
    )
    ap.add_argument(
        "--output",
        default="state/working_memory_m1/dev_decision_v1.json",
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text())
    if payload.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("DEV evaluator format mismatch")
    if payload.get("split") != "dev":
        raise RuntimeError("input is not DEV split")
    if payload.get("test_conditions_touched") is not False:
        raise RuntimeError("TEST contamination flag is not false")

    rows = payload["per_condition"]
    ids = [int(r["condition_id"]) for r in rows]
    if len(rows) != 380:
        raise AssertionError(f"expected 380 DEV conditions, got {len(rows)}")
    if len(set(ids)) != 380 or set(ids) != EXPECTED_DEV_IDS:
        raise AssertionError("DEV condition IDs missing, duplicated, or contaminated")

    normal = mode_stats(rows, "normal")
    disabled = mode_stats(rows, "memory_disabled_all")
    zero_after = mode_stats(rows, "zero_after_hidden")
    zero_decision = mode_stats(rows, "zero_at_decision")
    hidden_own = mode_stats(rows, "hidden_reset_own")
    both_reset = mode_stats(rows, "both_reset")

    pair_k = sum(normal["pair_full_bits"])
    pair_lo, pair_hi = wilson(pair_k, len(rows))

    memory_gain = normal["pair_full"] - disabled["pair_full"]
    memory_gain_test = paired_gain_test(
        normal["pair_full_bits"],
        disabled["pair_full_bits"],
    )

    swap_effects = [
        float(r["swap"]["condition_effect"])
        for r in rows
    ]
    swap_mean = mean(swap_effects)
    swap_ci = bootstrap_mean_ci(swap_effects)
    swap_positive = sum(x > 0.0 for x in swap_effects)
    swap_positive_fraction = swap_positive / len(rows)
    swap_sign_p = binom_upper_tail(
        swap_positive,
        len([x for x in swap_effects if x != 0.0]),
        0.5,
    )

    hreset_swap_donor_pair = mean(
        [
            float(
                r["hidden_reset_memory_code"][
                    "swapped_pair_donor_consistent"
                ]
            )
            for r in rows
        ]
    )

    hreset_own_pair = mean(
        [
            float(
                r["hidden_reset_memory_code"][
                    "own_pair_side_correct"
                ]
            )
            for r in rows
        ]
    )

    both_reset_max_diff = max(
        float(r["modes"]["both_reset"]["pair_logits_max_abs_diff"])
        for r in rows
    )

    factor_results, factor_failures = factor_slices(
        rows, normal["pair_full"]
    )

    checks = {
        "normal_pair_full":
            normal["pair_full"] >= MIN_NORMAL_PAIR_FULL,
        "normal_pair_wilson_lower":
            pair_lo >= MIN_NORMAL_PAIR_FULL_WILSON_LOWER,
        "left_member_full":
            normal["left_full"] >= MIN_MEMBER_FULL,
        "right_member_full":
            normal["right_full"] >= MIN_MEMBER_FULL,
        "overall_left_right_gap":
            abs(normal["left_full"] - normal["right_full"])
            <= MAX_OVERALL_LEFT_RIGHT_GAP,
        "memory_added_pair_gain":
            memory_gain >= MIN_MEMORY_ADDED_PAIR_GAIN,
        "memory_added_p":
            memory_gain_test["p_one_sided"]
            < MAX_MEMORY_GAIN_P_ONE_SIDED,
        "hidden_reset_own_memory_pair_side":
            hreset_own_pair >= MIN_HIDDEN_RESET_OWN_PAIR_SIDE,
        "hidden_reset_swap_donor_pair":
            hreset_swap_donor_pair
            >= MIN_HIDDEN_RESET_SWAP_DONOR_PAIR,
        "swap_mean_positive":
            swap_mean > 0.0,
        "swap_bootstrap_lower_positive":
            swap_ci[0] > 0.0,
        "swap_positive_fraction":
            swap_positive_fraction >= MIN_SWAP_POSITIVE_FRACTION,
        "swap_sign_p":
            swap_sign_p < MAX_SWAP_SIGN_P_ONE_SIDED,
        "factor_slices":
            len(factor_failures) == 0,
        "both_reset_identical":
            both_reset_max_diff <= BOTH_RESET_MAX_LOGIT_DIFF,
        "both_reset_pair_side_zero":
            both_reset["pair_side"] == 0.0,
    }

    passed = all(checks.values())

    result = {
        "format": "working_memory_dev_admission_decision_v1",
        "source": args.input,
        "thresholds": {
            "min_normal_pair_full": MIN_NORMAL_PAIR_FULL,
            "min_normal_pair_full_wilson_lower":
                MIN_NORMAL_PAIR_FULL_WILSON_LOWER,
            "min_member_full": MIN_MEMBER_FULL,
            "max_overall_left_right_gap":
                MAX_OVERALL_LEFT_RIGHT_GAP,
            "min_memory_added_pair_gain":
                MIN_MEMORY_ADDED_PAIR_GAIN,
            "max_memory_gain_p_one_sided":
                MAX_MEMORY_GAIN_P_ONE_SIDED,
            "min_hidden_reset_own_pair_side":
                MIN_HIDDEN_RESET_OWN_PAIR_SIDE,
            "min_hidden_reset_swap_donor_pair":
                MIN_HIDDEN_RESET_SWAP_DONOR_PAIR,
            "min_swap_positive_fraction":
                MIN_SWAP_POSITIVE_FRACTION,
            "max_swap_sign_p_one_sided":
                MAX_SWAP_SIGN_P_ONE_SIDED,
            "min_factor_pair_full":
                MIN_FACTOR_PAIR_FULL,
            "max_factor_drop_from_overall":
                MAX_FACTOR_DROP_FROM_OVERALL,
            "max_factor_left_right_gap":
                MAX_FACTOR_LEFT_RIGHT_GAP,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "normal": {
            "pair_full": normal["pair_full"],
            "pair_full_wilson95": [pair_lo, pair_hi],
            "left_full": normal["left_full"],
            "right_full": normal["right_full"],
            "left_right_gap": abs(
                normal["left_full"] - normal["right_full"]
            ),
            "pair_side": normal["pair_side"],
        },
        "memory_disabled_all": {
            "pair_full": disabled["pair_full"],
            "pair_side": disabled["pair_side"],
        },
        "zero_after_hidden": {
            "pair_full": zero_after["pair_full"],
            "pair_side": zero_after["pair_side"],
        },
        "zero_at_decision": {
            "pair_full": zero_decision["pair_full"],
            "pair_side": zero_decision["pair_side"],
        },
        "memory_added_value": {
            "pair_full_gain_vs_disabled": memory_gain,
            **memory_gain_test,
        },
        "hidden_reset_memory": {
            "own_memory_pair_side": hreset_own_pair,
            "swapped_memory_donor_pair_side":
                hreset_swap_donor_pair,
        },
        "swap_causal_effect": {
            "mean": swap_mean,
            "bootstrap95": list(swap_ci),
            "positive_conditions": swap_positive,
            "positive_fraction": swap_positive_fraction,
            "sign_p_one_sided": swap_sign_p,
        },
        "both_reset": {
            "pair_side": both_reset["pair_side"],
            "max_pair_logits_abs_diff": both_reset_max_diff,
        },
        "factor_slices": factor_results,
        "factor_failures": factor_failures,
        "checks": checks,
        "passed": passed,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("WORKING_MEMORY_DEV_ADMISSION_DECISION_V1")
    print(
        f"normal pair_full={normal['pair_full']:.4f} "
        f"Wilson95=[{pair_lo:.4f},{pair_hi:.4f}] "
        f"Lfull={normal['left_full']:.4f} "
        f"Rfull={normal['right_full']:.4f}"
    )
    print(
        f"disabled pair_full={disabled['pair_full']:.4f} "
        f"gain={memory_gain:+.4f} "
        f"discordant={memory_gain_test['discordant']} "
        f"p_one_sided={memory_gain_test['p_one_sided']:.8g}"
    )
    print(
        f"hidden_reset own-memory pair_side={hreset_own_pair:.4f} "
        f"swap-donor pair_side={hreset_swap_donor_pair:.4f}"
    )
    print(
        f"swap mean={swap_mean:+.6f} "
        f"CI95=[{swap_ci[0]:+.6f},{swap_ci[1]:+.6f}] "
        f"positive={swap_positive}/{len(rows)} "
        f"p_one_sided={swap_sign_p:.8g}"
    )
    print(
        f"factor_failures={len(factor_failures)} "
        f"both_reset_pair_side={both_reset['pair_side']:.4f} "
        f"both_reset_max_diff={both_reset_max_diff:.3g}"
    )
    for name, ok in checks.items():
        print(f"{name}: {str(ok).lower()}")
    print("wrote:", out)
    print(
        "WORKING_MEMORY_DEV_ADMISSION:",
        "PASS" if passed else "FAIL",
    )


if __name__ == "__main__":
    main()
