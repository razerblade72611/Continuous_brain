#!/usr/bin/env python3
"""Pre-registered DEV admission decision for winning capacity-ungated memory."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path

# Frozen before DEV access.
MIN_PAIR_FULL=.80
MIN_WILSON_LOWER=.75
MIN_MEMBER_FULL=.85
MAX_MEMBER_GAP=.10
MIN_PAIR_SIDE=.95
MIN_RECALL_GAIN=.50
MAX_GAIN_P=1e-6
MIN_SAME_HIDDEN_DIRECT_SIDE=.95
MIN_HRESET_DIRECT=.75
MIN_HRESET_SWAP=.75
MIN_SWAP_POSITIVE=.90
MAX_SWAP_SIGN_P=1e-6
MIN_FACTOR_PAIR_FULL=.60
MAX_FACTOR_DROP=.20
MAX_FACTOR_MEMBER_GAP=.20
MAX_ZERO_DIFF=1e-6

MIN_DEV_CUE_READER=.95
MIN_DEV_CUE_FACTOR_SLICE=.90
MIN_DEV_POSTWRITE=.95
MIN_DEV_DUAL=.90
MAX_DEV_MEMORY_ZERO=.60
MAX_DEV_FORCED_NORMAL=.60
MIN_DEV_COUNTERFACTUAL=.95
MIN_DEV_SWAP_DONOR=.95
MAX_DEV_LINEAR_CONTROL=.60
MIN_DEV_FACTORIZED=.95

FACTORS=("corridor_length","goal_x_offset","goal_y_offset","cue_steps","move_step","start_x","start_y","distractor_pattern")

def mean(xs): return sum(xs)/len(xs)
def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)

def mode(rows,name):
    lf=[r["modes"][name]["left"]["full_correct"] for r in rows]
    rf=[r["modes"][name]["right"]["full_correct"] for r in rows]
    ls=[r["modes"][name]["left"]["side_correct"] for r in rows]
    rs=[r["modes"][name]["right"]["side_correct"] for r in rows]
    return {
        "left_full":mean([float(x) for x in lf]),"right_full":mean([float(x) for x in rf]),
        "pair_full":mean([float(a and b) for a,b in zip(lf,rf)]),
        "pair_side":mean([float(a and b) for a,b in zip(ls,rs)]),
    }

def factor_checks(rows,overall):
    failures=[]; report={}
    for f in FACTORS:
        report[f]={}
        for v in sorted({r["factors"][f] for r in rows}):
            g=[r for r in rows if r["factors"][f]==v]; s=mode(g,"normal")
            allowed=max(MIN_FACTOR_PAIR_FULL,overall-MAX_FACTOR_DROP)
            gap=abs(s["left_full"]-s["right_full"])
            ok=s["pair_full"]>=allowed and gap<=MAX_FACTOR_MEMBER_GAP
            report[f][str(v)]={"n":len(g),"pair_full":s["pair_full"],"min_allowed":allowed,"member_gap":gap,"pass":ok}
            if not ok: failures.append({"factor":f,"value":v,**report[f][str(v)]})
    return report,failures

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--causal",required=True); ap.add_argument("--representation",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args()
    c=json.loads(Path(a.causal).read_text()); r=json.loads(Path(a.representation).read_text())
    if c.get("format")!="working_memory_capacity_dev_causal_v2" or c.get("split")!="dev" or c.get("test_touched") is not False:
        raise RuntimeError("causal DEV artifact invalid")
    if r.get("format")!="working_memory_capacity_dev_representation_v1" or r.get("split")!="dev" or r.get("reader_fit_split")!="train" or r.get("test_touched") is not False:
        raise RuntimeError("representation DEV artifact invalid")
    rows=c["per_condition"]
    if len(rows)!=380 or {x["condition_id"] for x in rows}!=set(range(1600,1980)):
        raise AssertionError("DEV IDs invalid")
    n=mode(rows,"normal"); off=mode(rows,"direct_recall_off"); shd=mode(rows,"same_hidden_direct_only")
    pair_k=round(n["pair_full"]*len(rows)); lo,hi=wilson(pair_k,len(rows))
    o=c["overall"]; sw=o["direct_swap"]; hr=o["hidden_reset_direct"]; gain=o["direct_recall_added_value"]
    z=max(x["modes"]["hidden_reset_all_zero"]["pair_logits_max_abs_diff"] for x in rows)
    factors,fails=factor_checks(rows,n["pair_full"])
    rs=r["summary"]
    checks={
        "pair_full":n["pair_full"]>=MIN_PAIR_FULL,
        "pair_wilson_lower":lo>=MIN_WILSON_LOWER,
        "left_full":n["left_full"]>=MIN_MEMBER_FULL,
        "right_full":n["right_full"]>=MIN_MEMBER_FULL,
        "member_gap":abs(n["left_full"]-n["right_full"])<=MAX_MEMBER_GAP,
        "pair_side":n["pair_side"]>=MIN_PAIR_SIDE,
        "recall_gain":gain["pair_full_gain_vs_off"]>=MIN_RECALL_GAIN,
        "recall_gain_p":gain["p_one_sided"]<MAX_GAIN_P,
        "same_hidden_direct_side":shd["pair_side"]>=MIN_SAME_HIDDEN_DIRECT_SIDE,
        "hidden_reset_direct":hr["own_pair_side"]>=MIN_HRESET_DIRECT,
        "hidden_reset_swap":hr["swapped_donor_pair"]>=MIN_HRESET_SWAP,
        "swap_mean":sw["mean"]>0,
        "swap_ci_lower":sw["bootstrap95"][0]>0,
        "swap_positive":sw["positive_fraction"]>=MIN_SWAP_POSITIVE,
        "swap_sign_p":sw["sign_p_one_sided"]<MAX_SWAP_SIGN_P,
        "all_zero_identity":z<=MAX_ZERO_DIFF,
        "factor_slices":len(fails)==0,
        "dev_cue_reader":rs["cue_accuracy"]>=MIN_DEV_CUE_READER,
        "dev_cue_factor_slices":rs["minimum_cue_factor_slice_accuracy"]>=MIN_DEV_CUE_FACTOR_SLICE,
        "dev_postwrite":rs["joint_accuracy"]>=MIN_DEV_POSTWRITE,
        "dev_postwrite_normal":rs["joint_normal"]>=MIN_DEV_POSTWRITE,
        "dev_postwrite_invert":rs["joint_invert"]>=MIN_DEV_POSTWRITE,
        "dev_dual":rs["joint_dual_both"]>=MIN_DEV_DUAL,
        "dev_memory_zero":rs["joint_memory_zero"]<=MAX_DEV_MEMORY_ZERO,
        "dev_forced_normal":rs["joint_forced_normal"]<=MAX_DEV_FORCED_NORMAL,
        "dev_instruction_flip":rs["joint_instruction_flip"]>=MIN_DEV_COUNTERFACTUAL,
        "dev_swap_donor":rs["joint_swap_donor"]>=MIN_DEV_SWAP_DONOR,
        "dev_linear_control":rs["linear_joint_control"]<=MAX_DEV_LINEAR_CONTROL,
        "dev_factorized":rs["factorized_postwrite_accuracy"]>=MIN_DEV_FACTORIZED,
    }
    passed=all(checks.values())
    result={"format":"working_memory_capacity_dev_admission_decision_v2","pass":passed,
            "checks":checks,"normal":n,"wilson95":[lo,hi],"recall_off":off,
            "same_hidden_direct":shd,"recall_gain":gain,"swap":sw,
            "hidden_reset_direct":hr,"all_zero_max_diff":z,
            "representation":rs,"factor_results":factors,"factor_failures":fails,
            "test_touched":False}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    print("WORKING_MEMORY_CAPACITY_DEV_ADMISSION_DECISION_V2")
    print(f"normal_pair_full={n['pair_full']:.4f} Wilson95=[{lo:.4f},{hi:.4f}] L={n['left_full']:.4f} R={n['right_full']:.4f} pair_side={n['pair_side']:.4f}")
    print(f"recall_off_pair_full={off['pair_full']:.4f} gain={gain['pair_full_gain_vs_off']:+.4f}")
    print(f"same_hidden_direct_pair_side={shd['pair_side']:.4f}")
    print(f"hidden_reset_direct={hr['own_pair_side']:.4f} swapped_donor={hr['swapped_donor_pair']:.4f}")
    print(f"swap_mean={sw['mean']:+.6f} positive={sw['positive_fraction']:.4f} CIlo={sw['bootstrap95'][0]:+.6f}")
    print(f"DEV cue_reader={rs['cue_accuracy']:.4f} min_factor_slice={rs['minimum_cue_factor_slice_accuracy']:.4f}")
    print(f"DEV postwrite={rs['joint_accuracy']:.4f} normal={rs['joint_normal']:.4f} invert={rs['joint_invert']:.4f} dual={rs['joint_dual_both']:.4f}")
    print(f"controls memory_zero={rs['joint_memory_zero']:.4f} forced_normal={rs['joint_forced_normal']:.4f} linear={rs['linear_joint_control']:.4f}")
    print()
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:",len(fails))
    print("TEST_TOUCHED: false")
    print("DEV_ADMISSION:", "PASS" if passed else "FAIL")
    print("WORKING_MEMORY_CAPACITY_DEV_ADMISSION_DECISION_COMPLETE")
    sys.exit(0 if passed else 4)

if __name__=="__main__": main()
