#!/usr/bin/env python3
"""Frozen TRAIN gate for contextual canonical executive."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
RULES=("invert_lr","vertical","vertical_reverse")
MIN_PAIR=.95; MIN_DONOR=.95; MAX_OFF=.05; MAX_ZERO=.05; MIN_SIX=.90; MAX_NONE=1e-6; MIN_FACTOR_PAIR=.85; MIN_FACTOR_DONOR=.85; MIN_FACTOR_SIX=.75
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    p=json.loads(Path(a.input).read_text())
    if p.get("format")!="contextual_canonical_executive_final_train_v1" or p.get("split")!="train_only" or p.get("v3_dev_touched") or p.get("v3_test_touched"): raise RuntimeError("invalid TRAIN artifact")
    s=p["summary"]; checks={}
    for rule in RULES:
        checks[f"{rule}_pair_full"]=s["normal"][rule]>=MIN_PAIR; checks[f"{rule}_donor"]=s["canonical_donor"][rule]>=MIN_DONOR
        checks[f"{rule}_executive_off_fails"]=s["executive_off"][rule]<=MAX_OFF; checks[f"{rule}_canonical_zero_fails"]=s["canonical_zero"][rule]<=MAX_ZERO
    checks["six_way_full"]=s["normal"]["six_way"]>=MIN_SIX; checks["none_exact_noop"]=s["none_max_diff"]<=MAX_NONE
    fails=[]
    for field,levels in p["factor_summary"].items():
        for value,r in levels.items():
            ok=r["six_way_full"]>=MIN_FACTOR_SIX and all(r[f"{rule}_pair_full"]>=MIN_FACTOR_PAIR and r[f"{rule}_donor"]>=MIN_FACTOR_DONOR for rule in RULES)
            if not ok: fails.append({"factor":field,"value":value,**r})
    checks["factor_robustness"]=len(fails)==0; passed=all(checks.values())
    result={"format":"contextual_canonical_executive_train_decision_v1","pass":passed,"checks":checks,"factor_failures":fails,"summary":s,"v3_dev_touched":False,"v3_test_touched":False}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    print("CONTEXTUAL_CANONICAL_EXECUTIVE_TRAIN_DECISION_V1")
    for rule in RULES: print(f"{rule}: pair={s['normal'][rule]:.4f} donor={s['canonical_donor'][rule]:.4f} off={s['executive_off'][rule]:.4f} zero={s['canonical_zero'][rule]:.4f}")
    print(f"six_way={s['normal']['six_way']:.4f} none_diff={s['none_max_diff']:.3g}")
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:",len(fails)); print("V3_DEV_TOUCHED: false"); print("V3_TEST_TOUCHED: false")
    print("CONTEXTUAL_CANONICAL_TRAIN_GATE:","PASS" if passed else "FAIL")
    sys.exit(0 if passed else 4)
if __name__=="__main__": main()
