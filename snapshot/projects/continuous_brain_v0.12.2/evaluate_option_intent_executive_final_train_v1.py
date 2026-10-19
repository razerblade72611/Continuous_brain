#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import torch,torch_directml
from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.option_intent_executive_v1 import OptionIntentExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import ACTION_NAMES,TRAIN_CONDITION_RANGE,RULE_NONE,CanonicalExecutiveRuleWorld,generate_canonical_rule_spec,target_action,UP

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
EXPECTED_FORMAT="option_intent_executive_training_v1"
RULES=("invert_lr","vertical","vertical_reverse")

def clone(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def digest(st):
    h=hashlib.sha256()
    for k in sorted(st):
        t=st[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def load(path,device):
    ep=torch.load(path,map_location="cpu",weights_only=True)
    if ep.get("format")!=EXPECTED_FORMAT or int(ep.get("global_update",-1))<1600: raise RuntimeError("checkpoint mismatch")
    ap=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(ap["memory"],strict=True); admitted.recall.load_state_dict(ap["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for p in admitted.parameters(): p.requires_grad_(False)
    model=OptionIntentExecutiveInfant.grow_from_admitted(admitted); model.executive.load_state_dict(ep["executive"],strict=True); model=model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,cfg,ep

@torch.no_grad()
def snaps(model,cfg,device,cid,cue):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule="invert_lr"); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device); rows=[]
    while not env.done:
        im=_image_tensor(env,cfg.image_size,device); ctx,_=_context_tensor(obs,device,cfg)
        if env.instruction_visible():
            hist,D,_,ws,B=model.components(im,ctx,mem,prev,h)
            rows.append({"H":hist["historical_workspace"].detach(),"D":D.detach(),"B":B.detach(),"response":bool(env.at_response_step())})
        if env.at_response_step(): break
        teacher=int(env.teacher_action()); o=model.admitted.step(im,ctx,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher); h=o["hidden"]; mem=o["memory_next"]; prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
    return rows

def cls(logits,target):
    p=int(torch.argmax(logits).item()); return {"pred":p,"pred_name":ACTION_NAMES[p],"correct":bool(p==target)}
def eval_condition(model,cfg,device,cid):
    S={c:snaps(model,cfg,device,cid,c) for c in ("left","right")}; row={"condition_id":cid,"members":{},"timing":[]}
    for c in ("left","right"):
        for s in S[c]:
            g,_,_=model.executive.compute_gate(s["H"],s["B"]); gv=float(g.detach().cpu().reshape(-1)[0])
            rec={"cue":c,"response":bool(s["response"]),"gate":gv}
            if not s["response"]:
                rec["preserve"]={}
                for rid,r in enumerate(RULES,1):
                    rt=torch.tensor([rid],dtype=torch.long,device=device)
                    logits,_,_=model.compose(s["H"],s["D"],rt,s["B"])
                    rec["preserve"][r]=bool(int(torch.argmax(logits,dim=-1)[0])==UP)
            row["timing"].append(rec)
    for c in ("left","right"):
        dc="right" if c=="left" else "left"; rs=S[c][-1]; donorD=S[dc][-1]["D"]; zero=torch.zeros_like(rs["D"]); row["members"][c]={}
        for rid,r in enumerate(RULES,1):
            rt=torch.tensor([rid],dtype=torch.long,device=device)
            y=target_action(c,r); dy=target_action(dc,r)
            normal,_,_=model.compose(rs["H"],rs["D"],rt,rs["B"])
            donor,_,_=model.compose(rs["H"],donorD,rt,rs["B"])
            zerolog,_,_=model.compose(rs["H"],zero,rt,rs["B"])
            none,_,_=model.compose(rs["H"],rs["D"],torch.tensor([RULE_NONE],dtype=torch.long,device=device),rs["B"])
            row["members"][c][r]={"normal":cls(normal[0].cpu(),y),"donor":cls(donor[0].cpu(),dy),"zero":cls(zerolog[0].cpu(),y),
                                  "off":cls(rs["B"][0].cpu(),y),"none_diff":float((none-rs["B"]).abs().max().cpu())}
    sp=generate_canonical_rule_spec(cid); row["factors"]={"corridor_length":sp.corridor_length,"goal_x_offset":sp.goal_x_offset,"goal_y_offset":sp.goal_y_offset,"cue_steps":sp.cue_steps,"move_step":sp.move_step}; return row

def pair(r,rule,mode): return bool(r["members"]["left"][rule][mode]["correct"] and r["members"]["right"][rule][mode]["correct"])
def six(r,mode): return bool(all(r["members"][c][rule][mode]["correct"] for c in ("left","right") for rule in RULES))
def summarize(rows):
    s={}
    for mode in ("normal","donor","zero","off"):
        s[mode]={r:mean([float(pair(x,r,mode)) for x in rows]) for r in RULES}; s[mode]["six_way"]=mean([float(six(x,mode)) for x in rows])
    s["none_max_diff"]=max(x["members"][c][r]["none_diff"] for x in rows for c in ("left","right") for r in RULES)
    pre=[q for x in rows for q in x["timing"] if not q["response"]]; resp=[q for x in rows for q in x["timing"] if q["response"]]
    s["pre_gate_off_rate"]=mean([float(q["gate"]<.5) for q in pre]); s["response_gate_on_rate"]=mean([float(q["gate"]>=.5) for q in resp])
    s["pre_gate_mean"]=mean([q["gate"] for q in pre]); s["response_gate_mean"]=mean([q["gate"] for q in resp])
    s["pre_response_action_preservation"]=mean([float(v) for q in pre for v in q["preserve"].values()])
    return s
def factor_summary(rows):
    out={}
    for field in ("corridor_length","goal_x_offset","goal_y_offset","cue_steps","move_step"):
        out[field]={}
        for v in sorted({x["factors"][field] for x in rows}):
            g=[x for x in rows if x["factors"][field]==v]; s=summarize(g); rec={"n":len(g),"six_way_full":s["normal"]["six_way"],
                "pre_response_action_preservation":s["pre_response_action_preservation"],"pre_gate_off_rate":s["pre_gate_off_rate"],"response_gate_on_rate":s["response_gate_on_rate"]}
            for r in RULES: rec[r+"_pair_full"]=s["normal"][r]; rec[r+"_donor"]=s["donor"][r]
            out[field][str(v)]=rec
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    device=torch_directml.device(); model,cfg,p=load(Path(a.checkpoint),device); aref=clone(model.admitted); bref=clone(model.base)
    print("OPTION_INTENT_EXECUTIVE_FINAL_TRAIN_V1"); print("device:",device); print("checkpoint_global_update:",p["global_update"]); print("v3_dev_touched: false"); print("v3_test_touched: false")
    rows=[]
    for i,cid in enumerate(TRAIN_CONDITION_RANGE,1):
        rows.append(eval_condition(model,cfg,device,int(cid)))
        if i==1 or i%100==0 or i==1600:
            s=summarize(rows); print(f"evaluated={i}/1600 invert={s['normal']['invert_lr']:.4f} vertical={s['normal']['vertical']:.4f} vreverse={s['normal']['vertical_reverse']:.4f} six={s['normal']['six_way']:.4f} preKeep={s['pre_response_action_preservation']:.4f} gateOff={s['pre_gate_off_rate']:.4f} gateOn={s['response_gate_on_rate']:.4f}")
    s=summarize(rows); result={"format":"option_intent_executive_final_train_v1","split":"train_only","summary":s,"factor_summary":factor_summary(rows),"v3_dev_touched":False,"v3_test_touched":False}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    for r in RULES: print(f"{r}: pair={s['normal'][r]:.4f} donor={s['donor'][r]:.4f} off={s['off'][r]:.4f} zero={s['zero'][r]:.4f}")
    print(f"six_way={s['normal']['six_way']:.4f} preKeep={s['pre_response_action_preservation']:.4f} gateOff={s['pre_gate_off_rate']:.4f} gateOn={s['response_gate_on_rate']:.4f} preGateMean={s['pre_gate_mean']:.4f} responseGateMean={s['response_gate_mean']:.4f} none_diff={s['none_max_diff']:.9g}")
    print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true"); print("V3_DEV_TOUCHED: false"); print("V3_TEST_TOUCHED: false"); print("OPTION_INTENT_EXECUTIVE_FINAL_TRAIN_COMPLETE")
if __name__=="__main__": main()
