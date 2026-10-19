#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
RULES=("invert_lr","vertical","vertical_reverse")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    p=json.loads(Path(a.input).read_text())
    if p.get("format")!="option_intent_executive_final_train_v1" or p.get("split")!="train_only" or p.get("v3_dev_touched") or p.get("v3_test_touched"):
        raise RuntimeError("invalid TRAIN artifact")
    s=p["summary"]; checks={}
    for r in RULES:
        checks[r+"_pair"]=s["normal"][r]>=.95
        checks[r+"_donor"]=s["donor"][r]>=.95
        checks[r+"_off"]=s["off"][r]<=.05
        checks[r+"_zero"]=s["zero"][r]<=.05
    checks["six_way"]=s["normal"]["six_way"]>=.90
    checks["none"]=s["none_max_diff"]<=1e-6
    checks["pre_response_action_preservation"]=s["pre_response_action_preservation"]>=.99
    checks["pre_gate_off_rate"]=s["pre_gate_off_rate"]>=.99
    checks["response_gate_on_rate"]=s["response_gate_on_rate"]>=.99
    fails=[]
    for field,levels in p["factor_summary"].items():
        for value,x in levels.items():
            ok=(x["six_way_full"]>=.75 and x["pre_response_action_preservation"]>=.95
                and x["pre_gate_off_rate"]>=.95 and x["response_gate_on_rate"]>=.95
                and all(x[r+"_pair_full"]>=.85 and x[r+"_donor"]>=.85 for r in RULES))
            if not ok: fails.append({"factor":field,"value":value,**x})
    checks["factors"]=not fails; passed=all(checks.values())
    Path(a.output).write_text(json.dumps({"format":"option_intent_train_decision_v1","pass":passed,"checks":checks,"factor_failures":fails,"summary":s},indent=2,sort_keys=True))
    print("OPTION_INTENT_EXECUTIVE_TRAIN_DECISION_V1")
    for r in RULES: print(f"{r}: pair={s['normal'][r]:.4f} donor={s['donor'][r]:.4f} off={s['off'][r]:.4f} zero={s['zero'][r]:.4f}")
    print(f"six_way={s['normal']['six_way']:.4f} preKeep={s['pre_response_action_preservation']:.4f} gateOff={s['pre_gate_off_rate']:.4f} gateOn={s['response_gate_on_rate']:.4f} none_diff={s['none_max_diff']:.3g}")
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:",len(fails))
    print("OPTION_INTENT_TRAIN_GATE:","PASS" if passed else "FAIL")
    sys.exit(0 if passed else 4)
if __name__=="__main__": main()
