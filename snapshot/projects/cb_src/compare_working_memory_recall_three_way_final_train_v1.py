#!/usr/bin/env python3
"""Compare fixed-final TRAIN causal results for M1.5, matched control, and M2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def summary(payload):
    o = payload["overall"]
    n = o["modes"]["normal"]
    off = o["modes"]["direct_recall_off"]
    add = o["direct_recall_added_value"]
    swap = o["direct_swap"]
    code = o["direct_only_memory_code"]
    return {
        "pair_full": n["pair_full"],
        "left_full": n["left_full"],
        "right_full": n["right_full"],
        "pair_lateral": n["pair_lateral_top"],
        "pair_side": n["pair_side"],
        "off_pair_full": off["pair_full"],
        "gain_full": add["pair_full_gain_vs_off"],
        "swap_mean": swap["mean_condition_effect"],
        "swap_positive": swap["positive_fraction"],
        "direct_own_side": code["own_pair_side_correct"],
        "direct_swap_donor": code["swapped_pair_donor_consistent"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--gated", required=True)
    args = ap.parse_args()

    payloads = {
        "m1_5_linear": load(args.linear),
        "capacity_ungated": load(args.control),
        "m2_gated": load(args.gated),
    }

    print("WORKING_MEMORY_RECALL_THREE_WAY_FINAL_TRAIN_COMPARISON_V1")

    ss = {}
    for name, payload in payloads.items():
        if payload.get("dev_touched") or payload.get("test_touched"):
            raise RuntimeError(f"{name}: held-out contamination flag")
        s = summary(payload)
        ss[name] = s
        print(
            f"{name}: "
            f"pairF={s['pair_full']:.4f} "
            f"Lfull={s['left_full']:.4f} "
            f"Rfull={s['right_full']:.4f} "
            f"pairLat={s['pair_lateral']:.4f} "
            f"pairSide={s['pair_side']:.4f} "
            f"offPairF={s['off_pair_full']:.4f} "
            f"gainF={s['gain_full']:+.4f} "
            f"swapMean={s['swap_mean']:+.4f} "
            f"swapPos={s['swap_positive']:.4f} "
            f"directOwnSide={s['direct_own_side']:.4f}"
        )

    c = ss["capacity_ungated"]
    g = ss["m2_gated"]
    l = ss["m1_5_linear"]

    print(
        "capacity_minus_linear:",
        f"pair_full={c['pair_full']-l['pair_full']:+.4f}",
        f"left_full={c['left_full']-l['left_full']:+.4f}",
        f"right_full={c['right_full']-l['right_full']:+.4f}",
        f"direct_own_side={c['direct_own_side']-l['direct_own_side']:+.4f}",
    )
    print(
        "gated_minus_capacity:",
        f"pair_full={g['pair_full']-c['pair_full']:+.4f}",
        f"left_full={g['left_full']-c['left_full']:+.4f}",
        f"right_full={g['right_full']-c['right_full']:+.4f}",
        f"direct_own_side={g['direct_own_side']-c['direct_own_side']:+.4f}",
    )

    print()
    print("THREE_WAY_BY_REALIZED_DELAY")
    delays = sorted(
        set(int(k) for p in payloads.values()
            for k in p["by_realized_hidden_delay"])
    )
    print(
        f"{'delay':>5} | "
        f"{'linear F/S':>13} | "
        f"{'control F/S':>13} | "
        f"{'gated F/S':>13}"
    )
    for d in delays:
        vals = []
        for key in ("m1_5_linear", "capacity_ungated", "m2_gated"):
            x = payloads[key]["by_realized_hidden_delay"][str(d)]
            vals.append(
                f"{x['modes']['normal']['pair_full']:.3f}/"
                f"{x['modes']['normal']['pair_side']:.3f}"
            )
        print(
            f"{d:5d} | {vals[0]:>13} | {vals[1]:>13} | {vals[2]:>13}"
        )

    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_RECALL_THREE_WAY_FINAL_TRAIN_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
