#!/usr/bin/env python3
"""Fixed-final TRAIN causal evaluation for Action Intent Cortex."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from continuous_brain.action_intent_integrity_v1 import (select_device, load_executive, provenance, assert_exact, assert_files_unchanged)
from continuous_brain.action_intent_gate_v1 import validate_report
import torch

from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.action_intent_executive_v1 import ActionIntentExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import ACTION_NAMES,TRAIN_CONDITION_RANGE,RULE_NONE,CanonicalExecutiveRuleWorld,generate_canonical_rule_spec,target_action

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
EXPECTED_FORMAT="action_intent_executive_training_v1"
RULES=("invert_lr","vertical","vertical_reverse")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()
def clone(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def digest(st):
    h=hashlib.sha256()
    for k in sorted(st):
        t=st[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()
def mean(x):return sum(x)/len(x) if x else float("nan")
def load(path,device):
    return load_executive(path, device, minimum_update=1600)

@torch.no_grad()
def snap(model,cfg,device,cid,cue):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule="invert_lr"); obs=env.reset(); h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device)
    while not env.done:
        im=_image_tensor(env,cfg.image_size,device); ctx,_=_context_tensor(obs,device,cfg)
        if env.at_response_step():
            hist,recall,_,ws,base_logits=model.components(im,ctx,mem,prev,h)
            return {"recall":recall.detach(),"base_logits":base_logits.detach()}
        teacher=int(env.teacher_action()); o=model.admitted.step(im,ctx,mem,prev,h,use_memory=True,use_recall=True); obs,_,_,_=env.step(teacher); h=o["hidden"]; mem=o["memory_next"]; prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
    raise RuntimeError("no response")
def cls(logits,target):
    if not torch.isfinite(logits).all().item(): raise ValueError("nonfinite action logits")
    p=int(torch.argmax(logits).item());return {"pred":p,"pred_name":ACTION_NAMES[p],"correct":bool(p==target)}
def eval_condition(model,cfg,device,cid):
    L=snap(model,cfg,device,cid,"left");R=snap(model,cfg,device,cid,"right"); snaps={"left":L,"right":R}; row={"condition_id":int(cid),"members":{}}
    for cue,s in snaps.items():
        dc="right" if cue=="left" else "left"; donor=snaps[dc]["recall"]; zero=torch.zeros_like(s["recall"]); row["members"][cue]={}
        for rid,rule in enumerate(RULES,1):
            rt=torch.tensor([rid],dtype=torch.long,device=device); target=target_action(cue,rule); dtarget=target_action(dc,rule)
            normal,_,_=model.compose(s["recall"],rt,s["base_logits"]); donorlog,_,_=model.compose(donor,rt,s["base_logits"]); zerolog,_,_=model.compose(zero,rt,s["base_logits"])
            none,_,_=model.compose(s["recall"],torch.tensor([RULE_NONE],dtype=torch.long,device=device),s["base_logits"])
            row["members"][cue][rule]={"normal":cls(normal[0].detach().cpu(),target),"donor":cls(donorlog[0].detach().cpu(),dtarget),"zero":cls(zerolog[0].detach().cpu(),target),"off":cls(s["base_logits"][0].detach().cpu(),target),"none_diff":float((none-s["base_logits"]).abs().max().cpu())}
    sp=generate_canonical_rule_spec(cid);row["factors"]={"corridor_length":sp.corridor_length,"goal_x_offset":sp.goal_x_offset,"goal_y_offset":sp.goal_y_offset,"cue_steps":sp.cue_steps,"move_step":sp.move_step};return row
def pair(r,rule,mode):return bool(r["members"]["left"][rule][mode]["correct"] and r["members"]["right"][rule][mode]["correct"])
def six(r,mode):return bool(all(r["members"][c][rule][mode]["correct"] for c in ("left","right") for rule in RULES))
def summary(rows):
    s={}
    for mode in ("normal","donor","zero","off"):
        s[mode]={rule:mean([float(pair(r,rule,mode)) for r in rows]) for rule in RULES};s[mode]["six_way"]=mean([float(six(r,mode)) for r in rows])
    s["none_max_diff"]=max(r["members"][c][rule]["none_diff"] for r in rows for c in ("left","right") for rule in RULES);return s
def factors(rows):
    out={}
    for field in ("corridor_length","goal_x_offset","goal_y_offset","cue_steps","move_step"):
        out[field]={}
        for v in sorted({r["factors"][field] for r in rows}):
            g=[r for r in rows if r["factors"][field]==v];s=summary(g); rec={"n":len(g),"six_way_full":s["normal"]["six_way"]}
            for rule in RULES:rec[f"{rule}_pair_full"]=s["normal"][rule];rec[f"{rule}_donor"]=s["donor"][rule]
            out[field][str(v)]=rec
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device", choices=("directml", "cpu"), default="directml");ap.add_argument("--checkpoint",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    device=select_device(a.device);model,cfg,payload=load(Path(a.checkpoint),device);aref=clone(model.admitted);bref=clone(model.base);eref=clone(model.executive)
    identity=provenance(a.checkpoint,payload,device)
    print("ACTION_INTENT_EXECUTIVE_FINAL_TRAIN_V1");print("device:",device);print("checkpoint_global_update:",payload["global_update"]);print("v3_dev_touched: false");print("v3_test_touched: false")
    rows=[]
    for i,cid in enumerate(TRAIN_CONDITION_RANGE,1):
        rows.append(eval_condition(model,cfg,device,int(cid)))
        if i==1 or i%100==0 or i==1600:
            s=summary(rows);print(f"evaluated={i}/1600 invert={s['normal']['invert_lr']:.4f} vertical={s['normal']['vertical']:.4f} vreverse={s['normal']['vertical_reverse']:.4f} six={s['normal']['six_way']:.4f}")
    assert_exact(aref, model.admitted, "admitted after evaluation")
    assert_exact(bref, model.base, "base after evaluation")
    assert_exact(eref, model.executive, "executive after evaluation")
    assert_files_unchanged(identity, a.checkpoint)
    s=summary(rows)
    res={"format":"action_intent_executive_final_train_v1_1", "split":"train_only",
         "summary":s, "factor_summary":factors(rows), "per_condition":rows,
         "v3_dev_touched":False, "v3_test_touched":False, "provenance":identity,
         "preservation":{"admitted_bit_exact":True, "base_bit_exact":True,
                         "executive_bit_exact":True, "checkpoint_files_unchanged":True},
         "scope":"teacher_prefix_response_mapping_only"}
    validate_report(res)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False))
    for rule in RULES:print(f"{rule}: pair={s['normal'][rule]:.4f} donor={s['donor'][rule]:.4f} off={s['off'][rule]:.4f} zero={s['zero'][rule]:.4f}")
    print(f"six_way={s['normal']['six_way']:.4f} none_diff={s['none_max_diff']:.9g}");print("admitted_brain_bit_exact: true");print("historical_base_bit_exact: true");print("V3_DEV_TOUCHED: false");print("V3_TEST_TOUCHED: false");print("ACTION_INTENT_EXECUTIVE_FINAL_TRAIN_COMPLETE")
if __name__=="__main__":main()
