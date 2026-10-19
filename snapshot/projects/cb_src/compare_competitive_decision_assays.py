#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def metric_value(row, metric):
    if metric == "closed_loop_success":
        return float(row["closed_loop_success"])
    if metric == "junction_side":
        return float(row["junction"]["side_choice_correct"])
    if metric == "junction_full":
        return float(row["junction"]["full_action_correct"])
    raise KeyError(metric)


def seed_map(result, length_key):
    rows = result["lengths"][length_key]["per_seed"]
    return {int(r["seed"]): r for r in rows}


def paired_bootstrap(a, b, metric, draws, rng):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise RuntimeError("no common seeds")

    diffs = [
        metric_value(a[s], metric) - metric_value(b[s], metric)
        for s in seeds
    ]
    observed = sum(diffs) / len(diffs)

    boots = []
    n = len(diffs)
    for _ in range(draws):
        boots.append(
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        )

    return {
        "n_pairs": n,
        "difference": observed,
        "ci95_low": percentile(boots, 0.025),
        "ci95_high": percentile(boots, 0.975),
    }


def exact_mcnemar(a, b, metric):
    seeds = sorted(set(a) & set(b))
    a_only = 0
    b_only = 0

    for s in seeds:
        av = bool(metric_value(a[s], metric))
        bv = bool(metric_value(b[s], metric))
        if av and not bv:
            a_only += 1
        elif bv and not av:
            b_only += 1

    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        one_tail = sum(
            math.comb(n, i) * (0.5 ** n)
            for i in range(k + 1)
        )
        p = min(1.0, 2.0 * one_tail)

    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant_pairs": n,
        "two_sided_exact_p": p,
    }


def pct(x):
    return f"{100.0*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--independent", required=True)
    ap.add_argument("--uniform", required=True)
    ap.add_argument("--bootstrap-draws", type=int, default=50000)
    ap.add_argument("--bootstrap-seed", type=int, default=73021)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results = {
        "base": load(args.base),
        "shared": load(args.shared),
        "independent": load(args.independent),
        "uniform": load(args.uniform),
    }

    reference_lengths = results["base"]["corridor_lengths"]
    for name, result in results.items():
        if result["corridor_lengths"] != reference_lengths:
            raise RuntimeError(f"length mismatch for {name}")
        if result["seed_base"] != results["base"]["seed_base"]:
            raise RuntimeError(f"seed-base mismatch for {name}")
        if (
            result["episodes_per_length"]
            != results["base"]["episodes_per_length"]
        ):
            raise RuntimeError(f"episode-count mismatch for {name}")

    rng = random.Random(args.bootstrap_seed)
    report = {
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "lengths": {},
    }

    print(
        f"{'length':>7} {'condition':>12} "
        f"{'closed':>8} {'j-full':>8} {'j-side':>8} "
        f"{'margin':>10}"
    )
    print("-" * 70)

    comparisons = (
        ("shared_minus_uniform", "shared", "uniform"),
        ("independent_minus_uniform", "independent", "uniform"),
        ("shared_minus_base", "shared", "base"),
        ("independent_minus_base", "independent", "base"),
        ("independent_minus_shared", "independent", "shared"),
    )
    metrics = (
        "closed_loop_success",
        "junction_side",
        "junction_full",
    )

    for lv in reference_lengths:
        key = f"{float(lv):.3f}"
        report["lengths"][key] = {
            "conditions": {},
            "paired": {},
        }
        maps = {}

        for condition in ("base", "uniform", "shared", "independent"):
            s = results[condition]["lengths"][key]
            maps[condition] = seed_map(results[condition], key)

            report["lengths"][key]["conditions"][condition] = {
                "closed_loop_success_rate":
                    s["closed_loop_success_rate"],
                "junction_full_action_accuracy":
                    s["junction_full_action_accuracy"],
                "junction_side_choice_accuracy":
                    s["junction_side_choice_accuracy"],
                "junction_mean_correct_minus_wrong_side_logit":
                    s["junction_mean_correct_minus_wrong_side_logit"],
            }

            print(
                f"{float(lv):7.3f} "
                f"{condition:>12} "
                f"{pct(s['closed_loop_success_rate']):>8} "
                f"{pct(s['junction_full_action_accuracy']):>8} "
                f"{pct(s['junction_side_choice_accuracy']):>8} "
                f"{s['junction_mean_correct_minus_wrong_side_logit']:10.4f}"
            )

        print()
        for label, a_name, b_name in comparisons:
            entry = {}
            for metric in metrics:
                boot = paired_bootstrap(
                    maps[a_name],
                    maps[b_name],
                    metric,
                    args.bootstrap_draws,
                    rng,
                )
                exact = exact_mcnemar(
                    maps[a_name],
                    maps[b_name],
                    metric,
                )
                entry[metric] = {
                    "bootstrap": boot,
                    "mcnemar": exact,
                }

            report["lengths"][key]["paired"][label] = entry

            for metric in metrics:
                b = entry[metric]["bootstrap"]
                m = entry[metric]["mcnemar"]
                print(
                    f"  {key} {label:<26} {metric:<19} "
                    f"Δ={b['difference']:+.3f} "
                    f"CI95=[{b['ci95_low']:+.3f},"
                    f"{b['ci95_high']:+.3f}] "
                    f"discordant={m['discordant_pairs']:2d} "
                    f"p={m['two_sided_exact_p']:.6g}"
                )
        print()

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, sort_keys=True))
        print("wrote:", p)

    print("PAIRED_DECISION_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
