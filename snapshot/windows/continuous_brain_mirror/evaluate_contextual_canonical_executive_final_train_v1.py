#!/usr/bin/env python3
"""Fixed-final TRAIN causal evaluation for contextual canonical executive."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch, torch_directml

from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.contextual_canonical_executive_v1 import ContextualCanonicalExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import (
    ACTION_NAMES,TRAIN_CONDITION_RANGE,RULE_NONE,CanonicalExecutiveRuleWorld,
    generate_canonical_rule_spec,target_action,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
EXPECTED_FORMAT="contextual_canonical_executive_training_v1"
RULES=("invert_lr","vertical","vertical_reverse")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""): h.update(c)
    return h.hexdigest()
def clone(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        t=state[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")


def load(path,device):
    ep=torch.load(path,map_location="cpu",weights_only=True)
    if ep.get("format")!=EXPECTED_FORMAT or int(ep.get("global_update",-1))<1600: raise RuntimeError("checkpoint metadata mismatch")
    if ep.get("source_admitted_checkpoint_sha256")!=EXPECTED_ADMITTED_SHA256: raise RuntimeError("provenance mismatch")
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256: raise RuntimeError("admitted SHA mismatch")
    ap=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(ap["memory"],strict=True); admitted.recall.load_state_dict(ap["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for p in admitted.parameters(): p.requires_grad_(False)
    model=ContextualCanonicalExecutiveInfant.grow_from_admitted(admitted)
    model.executive.load_state_dict(ep["executive"],strict=True); model=model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,cfg,ep


@torch.no_grad()
def snapshot(model,cfg,device,cid,cue):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule="invert_lr"); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device)
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        if env.at_response_step(): return {"image":image.detach(),"context":context.detach(),"hidden":h.detach(),"memory":mem.detach(),"prev":prev.detach()}
        teacher=int(env.teacher_action()); out=model.admitted.step(image,context,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher); h=out["hidden"]; mem=out["memory_next"]; prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
    raise RuntimeError("no response")


@torch.no_grad()
def compose(model,snap,memory_for_recall,rule_id,mode):
    hist=model.admitted.historical_step(snap["image"],snap["context"],snap["memory"],snap["prev"],snap["hidden"])
    if mode=="zero":
        recall=torch.zeros_like(hist["historical_workspace"])
    else:
        recall,_=model.admitted.recall(hist["hidden"],memory_for_recall)
    if mode=="off":
        delta=torch.zeros_like(hist["historical_workspace"])
    else:
        delta,_=model.executive(hist["historical_workspace"],recall,torch.tensor([rule_id],dtype=torch.long,device=recall.device))
    logits=model.base.motor(hist["historical_workspace"]+recall+delta)[0].detach().float().cpu()
    return logits


def classify(logits,target):
    p=int(torch.argmax(logits).item()); return {"pred":p,"pred_name":ACTION_NAMES[p],"correct":bool(p==target)}


def eval_condition(model,cfg,device,cid):
    L=snapshot(model,cfg,device,cid,"left"); R=snapshot(model,cfg,device,cid,"right")
    for k in ("image","context","prev"):
        if not torch.equal(L[k].cpu(),R[k].cpu()): raise AssertionError(f"pair {k} mismatch")
    snaps={"left":L,"right":R}; row={"condition_id":int(cid),"members":{}}
    for cue,snap in snaps.items():
        donor_cue="right" if cue=="left" else "left"; donor=snaps[donor_cue]["memory"]; own=snap["memory"]; row["members"][cue]={}
        for rid,rule in enumerate(RULES,1):
            target=target_action(cue,rule); donor_target=target_action(donor_cue,rule)
            normal=compose(model,snap,own,rid,"normal"); off=compose(model,snap,own,rid,"off")
            zero=compose(model,snap,own,rid,"zero"); donor_logits=compose(model,snap,donor,rid,"normal")
            none=compose(model,snap,own,RULE_NONE,"normal"); admitted=compose(model,snap,own,rid,"off")
            row["members"][cue][rule]={
                "normal":classify(normal,target),"executive_off":classify(off,target),
                "canonical_zero":classify(zero,target),"canonical_donor":classify(donor_logits,donor_target),
                "none":classify(none,target),"none_diff":float((none-admitted).abs().max()),
            }
    s=generate_canonical_rule_spec(cid)
    row["factors"]={"corridor_length":s.corridor_length,"goal_x_offset":s.goal_x_offset,"goal_y_offset":s.goal_y_offset,"cue_steps":s.cue_steps,"move_step":s.move_step}
    return row


def pair(r,rule,mode): return bool(r["members"]["left"][rule][mode]["correct"] and r["members"]["right"][rule][mode]["correct"])
def six(r,mode): return bool(all(r["members"][c][rule][mode]["correct"] for c in ("left","right") for rule in RULES))
def summarize(rows):
    out={}
    for mode in ("normal","executive_off","canonical_zero","canonical_donor","none"):
        out[mode]={rule:mean([float(pair(r,rule,mode)) for r in rows]) for rule in RULES}; out[mode]["six_way"]=mean([float(six(r,mode)) for r in rows])
    out["none_max_diff"]=max(r["members"][c][rule]["none_diff"] for r in rows for c in ("left","right") for rule in RULES)
    return out
def factor_summary(rows):
    out={}
    for field in ("corridor_length","goal_x_offset","goal_y_offset","cue_steps","move_step"):
        out[field]={}
        for v in sorted({r["factors"][field] for r in rows}):
            g=[r for r in rows if r["factors"][field]==v]; s=summarize(g); rec={"n":len(g),"six_way_full":s["normal"]["six_way"]}
            for rule in RULES: rec[f"{rule}_pair_full"]=s["normal"][rule]; rec[f"{rule}_donor"]=s["canonical_donor"][rule]
            out[field][str(v)]=rec
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    device=torch_directml.device(); model,cfg,payload=load(Path(a.checkpoint),device); aref=clone(model.admitted); bref=clone(model.base)
    print("CONTEXTUAL_CANONICAL_EXECUTIVE_FINAL_TRAIN_V1"); print("device:",device); print("checkpoint_global_update:",payload["global_update"])
    print("train_range: 0:1600"); print("v3_dev_touched: false"); print("v3_test_touched: false")
    rows=[]
    for i,cid in enumerate(TRAIN_CONDITION_RANGE,1):
        rows.append(eval_condition(model,cfg,device,int(cid)))
        if i==1 or i%100==0 or i==1600:
            s=summarize(rows); print(f"evaluated={i}/1600 invert={s['normal']['invert_lr']:.4f} vertical={s['normal']['vertical']:.4f} vreverse={s['normal']['vertical_reverse']:.4f} six={s['normal']['six_way']:.4f}")
    for k,v in aref.items():
        if not torch.equal(v,model.admitted.state_dict()[k].detach().cpu()): raise AssertionError(f"admitted changed {k}")
    for k,v in bref.items():
        if not torch.equal(v,model.base.state_dict()[k].detach().cpu()): raise AssertionError(f"base changed {k}")
    s=summarize(rows); fs=factor_summary(rows)
    result={"format":"contextual_canonical_executive_final_train_v1","split":"train_only","checkpoint_global_update":int(payload["global_update"]),
            "summary":s,"factor_summary":fs,"per_condition":rows,"v3_dev_touched":False,"v3_test_touched":False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    print()
    for rule in RULES: print(f"{rule}: pair={s['normal'][rule]:.4f} donor={s['canonical_donor'][rule]:.4f} off={s['executive_off'][rule]:.4f} zero={s['canonical_zero'][rule]:.4f}")
    print(f"six_way={s['normal']['six_way']:.4f} none_diff={s['none_max_diff']:.9g}")
    print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true"); print("V3_DEV_TOUCHED: false"); print("V3_TEST_TOUCHED: false")
    print("CONTEXTUAL_CANONICAL_EXECUTIVE_FINAL_TRAIN_COMPLETE")

if __name__=="__main__": main()
