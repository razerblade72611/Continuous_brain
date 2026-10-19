#!/usr/bin/env python3
import argparse
import collections
import math
from continuous_brain.cued_detour_world import CuedDetourWorld


def exact_binomial_two_sided(k, n, p=0.5):
    # Exact two-sided test by summing outcomes no more likely than observed.
    def prob(x):
        return math.comb(n, x) * (p ** x) * ((1.0 - p) ** (n - x))

    p_obs = prob(k)
    return min(
        1.0,
        sum(prob(x) for x in range(n + 1) if prob(x) <= p_obs + 1e-18),
    )


def cue_for_seed(seed, corridor_length):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=float(corridor_length),
        max_episode_steps=120,
    )
    env.reset()
    return str(env.cue)


def longest_run(cues):
    if not cues:
        return None, 0, None, None

    best_cue = cues[0]
    best_len = 1
    best_start = 0
    cur_cue = cues[0]
    cur_len = 1
    cur_start = 0

    for i in range(1, len(cues)):
        if cues[i] == cur_cue:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_cue = cur_cue
                best_len = cur_len
                best_start = cur_start
            cur_cue = cues[i]
            cur_len = 1
            cur_start = i

    if cur_len > best_len:
        best_cue = cur_cue
        best_len = cur_len
        best_start = cur_start

    return best_cue, best_len, best_start, best_start + best_len - 1


def report_slice(label, cues, first_update):
    counts = collections.Counter(cues)
    n = len(cues)
    left = counts.get("left", 0)
    right = counts.get("right", 0)
    print(
        f"{label:<14} updates={first_update:03d}-"
        f"{first_update+n-1:03d} "
        f"left={left:3d} ({left/n:.3f}) "
        f"right={right:3d} ({right/n:.3f}) "
        f"delta={right-left:+d}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, default=82000)
    ap.add_argument("--updates", type=int, default=250)
    ap.add_argument("--corridor-length", type=float, default=0.3)
    args = ap.parse_args()

    if args.updates < 1:
        raise ValueError("--updates must be >= 1")

    seeds = [
        args.seed_base + update
        for update in range(1, args.updates + 1)
    ]
    cues = [
        cue_for_seed(seed, args.corridor_length)
        for seed in seeds
    ]

    unexpected = sorted(set(cues) - {"left", "right"})
    if unexpected:
        raise RuntimeError(f"unexpected cue values: {unexpected}")

    counts = collections.Counter(cues)
    left = counts["left"]
    right = counts["right"]

    print("TRAINING_CUE_BALANCE_AUDIT")
    print("seed_base:", args.seed_base)
    print("updates:", args.updates)
    print("first_seed:", seeds[0])
    print("last_seed:", seeds[-1])
    print("corridor_length:", args.corridor_length)
    print()
    print(f"left:  {left}/{args.updates} = {left/args.updates:.4f}")
    print(f"right: {right}/{args.updates} = {right/args.updates:.4f}")
    print(f"right_minus_left: {right-left:+d}")
    print(
        "two_sided_exact_binomial_p:",
        f"{exact_binomial_two_sided(right, args.updates):.12g}",
    )

    cue, run_len, start_i, end_i = longest_run(cues)
    print()
    print("longest_same_cue_run:", run_len)
    print("longest_run_cue:", cue)
    print("longest_run_updates:",
          f"{start_i+1}-{end_i+1}")
    print("longest_run_seeds:",
          f"{seeds[start_i]}-{seeds[end_i]}")

    print()
    # 5 equal blocks of 50 for temporal balance.
    block = 50
    for start in range(0, args.updates, block):
        chunk = cues[start:start + block]
        report_slice(
            f"block_{start//block+1}",
            chunk,
            start + 1,
        )

    print()
    # Early/middle/late summaries to catch curriculum drift.
    thirds = [
        ("early", 0, args.updates // 3),
        ("middle", args.updates // 3, 2 * args.updates // 3),
        ("late", 2 * args.updates // 3, args.updates),
    ]
    for label, a, b in thirds:
        report_slice(label, cues[a:b], a + 1)

    print()
    print("cue_sequence:")
    print("".join("L" if c == "left" else "R" for c in cues))
    print()
    print("TRAINING_CUE_BALANCE_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
