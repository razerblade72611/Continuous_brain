#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--linear",required=True); ap.add_argument("--gated",required=True); args=ap.parse_args()
    L=load(args.linear); G=load(args.gated)
    if L["split"]!="train" or G["split"]!="train" or L["dev_touched"] or G["dev_touched"] or L["test_touched"] or G["test_touched"]:
        raise RuntimeError("split contamination")
    if [r["condition_id"] for r in L["per_condition"]] != [r["condition_id"] for r in G["per_condition"]]:
        raise AssertionError("condition order differs")
    print("WORKING_MEMORY_RECALL_FINAL_TRAIN_COMPARISON_V1")
    for name,p in (("linear",L),("gated",G)):
        o=p["overall"]; n=o["modes"]["normal"]; off=o["modes"]["direct_recall_off"]; add=o["direct_recall_added_value"]; sw=o["direct_swap"]; c=o["direct_only_memory_code"]
        print(f"{name}: pairF={n['pair_full']:.4f} Lfull={n['left_full']:.4f} Rfull={n['right_full']:.4f} pairLat={n['pair_lateral_top']:.4f} pairSide={n['pair_side']:.4f} offPairF={off['pair_full']:.4f} gainF={add['pair_full_gain_vs_off']:+.4f} swapMean={sw['mean_condition_effect']:+.4f} swapPos={sw['positive_fraction']:.4f} directOwnSide={c['own_pair_side_correct']:.4f} directSwapDonor={c['swapped_pair_donor_consistent']:.4f}")
    ln=L["overall"]["modes"]["normal"]; gn=G["overall"]["modes"]["normal"]
    print(f"gated_minus_linear: pair_full={gn['pair_full']-ln['pair_full']:+.4f} left_full={gn['left_full']-ln['left_full']:+.4f} right_full={gn['right_full']-ln['right_full']:+.4f} pair_lateral={gn['pair_lateral_top']-ln['pair_lateral_top']:+.4f} pair_side={gn['pair_side']-ln['pair_side']:+.4f}")
    print("DEV_TOUCHED: false"); print("TEST_TOUCHED: false"); print("WORKING_MEMORY_RECALL_FINAL_TRAIN_COMPARISON_COMPLETE")
if __name__=="__main__": main()
