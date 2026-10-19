#!/usr/bin/env python3
"""Pre-registered v3 fresh-DEV admission for canonical recall executive."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
RULES=("invert_lr","vertical","vertical_reverse")
MIN_RULE=.80; MIN_SIX=.70; MAX_OFF=.15; MAX_ZERO=.15; MIN_DONOR=.90; MAX_NONE=1e-6; MIN_FACTOR_RULE=.60; MIN_FACTOR_SIX=.45
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    p=json.loads(Path(a.input).read_text())
    if p.get("format")!="canonical_recall_executive_fresh_dev_v1" or p.get("split")!="fresh_dev" or p.get("fresh_test_touched") is not False: raise RuntimeError("invalid fresh DEV artifact")
    s=p["summary"]; checks={}
    for rule in RULES:
        checks[f"{rule}_pair_full"]=s["normal"][rule]>=MIN_RULE
        checks[f"{rule}_executive_off_fails"]=s["executive_off"][rule]<=MAX_OFF
        checks[f"{rule}_canonical_zero_fails"]=s["canonical_memory_zero"][rule]<=MAX_ZERO
        checks[f"{rule}_canonical_donor"]=s["canonical_memory_donor"][rule]>=MIN_DONOR
    checks["six_way_full"]=s["normal"]["six_way"]>=MIN_SIX; checks["none_exact_noop"]=s["none_max_diff"]<=MAX_NONE
    fails=[]
    for field,levels in p["factor_summary"].items():
        for value,r in levels.items():
            ok=r["six_way_full"]>=MIN_FACTOR_SIX and all(r[f"{rule}_pair_full"]>=MIN_FACTOR_RULE for rule in RULES)
            if not ok: fails.append({"factor":field,"value":value,**r})
    checks["factor_robustness"]=len(fails)==0; passed=all(checks.values())
    result={"format":"canonical_recall_executive_fresh_dev_decision_v1","pass":passed,"checks":checks,"factor_failures":fails,"summary":s,"fresh_test_touched":False}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    print("CANONICAL_RECALL_EXECUTIVE_FRESH_DEV_DECISION_V1")
    for rule in RULES: print(f"{rule}: pair={s['normal'][rule]:.4f} off={s['executive_off'][rule]:.4f} zero={s['canonical_memory_zero'][rule]:.4f} donor={s['canonical_memory_donor'][rule]:.4f}")
    print(f"six_way_full={s['normal']['six_way']:.4f} none_diff={s['none_max_diff']:.3g}")
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:",len(fails)); print("FRESH_TEST_TOUCHED: false"); print("CANONICAL_RECALL_FRESH_DEV_ADMISSION:","PASS" if passed else "FAIL")
    sys.exit(0 if passed else 4)
if __name__=="__main__": main()
