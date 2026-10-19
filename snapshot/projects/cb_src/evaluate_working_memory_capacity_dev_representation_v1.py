#!/usr/bin/env python3
"""
DEV representation/generalization evaluator.

New readers are trained ONLY on TRAIN memories from the frozen winning writer.
They are then evaluated on DEV memories with zero DEV fitting.
TEST is never accessed.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch, torch_directml
from torch import nn

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,_image_tensor,_load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    LEFT,RIGHT,TRAIN_CONDITION_RANGE,DEV_CONDITION_RANGE,
    VariedCuedDetourWorld,generate_task_spec,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_FORMAT="working_memory_recall_capacity_control_training_v1"
SEEDS=(101,202,303)
EPOCHS_LIN=500; EPOCHS_MLP=700
LR_LIN=2e-2; LR_MLP=1e-2; WD=1e-4

def clone_state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def build(path,device):
    p=torch.load(path,map_location="cpu",weights_only=True)
    if p.get("format")!=EXPECTED_FORMAT or p.get("arm")!="capacity_ungated" or int(p.get("update",-1))!=1600:
        raise RuntimeError("checkpoint metadata mismatch")
    base,cfg,_,sha=_load_base_checkpoint(BASE_CHECKPOINT)
    if sha!=EXPECTED_BASE_SHA256 or digest(clone_state(base))!=EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base mismatch")
    m=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    m.memory.load_state_dict(p["memory"],strict=True); m.recall.load_state_dict(p["recall"],strict=True)
    m=m.to(device); m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m,cfg

@torch.no_grad()
def get_memory(model,cfg,device,cid,cue):
    env=VariedCuedDetourWorld(cid,cue=cue,max_episode_steps=160); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device)
    prev=torch.zeros(1,dtype=torch.long,device=device)
    while not env.done:
        teacher=int(env.teacher_action())
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        if teacher in (LEFT,RIGHT): return mem.detach().float().cpu()[0]
        out=model.step(image,context,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher); prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
        h=out["hidden"]; mem=out["memory_next"]
    raise RuntimeError("no decision")

def collect(model,cfg,device,ids,label):
    rows=[]; total=len(ids)
    for i,cid in enumerate(ids,1):
        spec=generate_task_spec(int(cid))
        for cue,y in (("left",0),("right",1)):
            rows.append({
                "condition_id":int(cid),"split":label,"cue_label":y,
                "corridor_length":float(spec.corridor_length),
                "cue_steps":int(spec.cue_steps),
                "memory":get_memory(model,cfg,device,int(cid),cue),
            })
        if i==1 or i%200==0 or i==total: print(f"collected_{label}={i}/{total}")
    return rows

def stats(train):
    x=torch.stack([r["memory"] for r in train]); mu=x.mean(0,keepdim=True)
    sd=x.std(0,unbiased=False,keepdim=True); sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
    return mu,sd

class Cue(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Linear(64,2)
    def forward(self,x): return self.net(x)
class Joint(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(65,64),nn.Tanh(),nn.Linear(64,2))
    def forward(self,x): return self.net(x)
class JointLinear(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Linear(65,2)
    def forward(self,x): return self.net(x)

def fit(model,x,y,epochs,lr):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WD); loss=nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); z=loss(model(x),y); z.backward(); opt.step()
    model.eval(); return model

def acc(p,y): return float((p==y).float().mean())

def cue_tensors(rows,mu,sd):
    x=(torch.stack([r["memory"] for r in rows])-mu)/sd
    y=torch.tensor([r["cue_label"] for r in rows],dtype=torch.long)
    return x,y

def expand(rows):
    ex=[]
    for r in rows:
        for ins in (0,1):
            cue=int(r["cue_label"]); response=cue if ins==0 else 1-cue
            ex.append({**r,"instruction":ins,"response":response})
    return ex

def joint_tensors(rows,mu,sd):
    mem=(torch.stack([r["memory"] for r in rows])-mu)/sd
    ins=torch.tensor([float(r["instruction"]) for r in rows]).unsqueeze(1)
    x=torch.cat([mem,ins],1); y=torch.tensor([r["response"] for r in rows],dtype=torch.long)
    return x,y

def joint_metrics(model,dev_ex,mu,sd):
    x,y=joint_tensors(dev_ex,mu,sd)
    with torch.no_grad(): p=model(x).argmax(1)
    normal=torch.tensor([r["instruction"]==0 for r in dev_ex],dtype=torch.bool); invert=~normal
    xz=x.clone(); xz[:,:64]=0
    xn=x.clone(); xn[:,64]=0
    xf=x.clone(); xf[:,64]=1-xf[:,64]; yf=1-y
    with torch.no_grad():
        pz=model(xz).argmax(1); pn=model(xn).argmax(1); pf=model(xf).argmax(1)
    dual=[]
    for i in range(0,len(dev_ex),2):
        dual.append(bool(p[i]==y[i] and p[i+1]==y[i+1]))
    # donor swap within condition/cue while retaining instruction
    mem={(r["condition_id"],r["cue_label"]):r["memory"] for r in dev_ex}
    swapped=[]; donor=[]
    for r in dev_ex:
        dc=1-int(r["cue_label"]); dr=dc if r["instruction"]==0 else 1-dc
        swapped.append({**r,"memory":mem[(r["condition_id"],dc)],"response":dr}); donor.append(dr)
    xs,_=joint_tensors(swapped,mu,sd)
    with torch.no_grad(): ps=model(xs).argmax(1)
    return {
        "accuracy":acc(p,y),"normal":acc(p[normal],y[normal]),"invert":acc(p[invert],y[invert]),
        "dual_both":sum(dual)/len(dual),"memory_zero":acc(pz,y),
        "forced_normal":acc(pn,y),"instruction_flip":acc(pf,yf),
        "swap_donor":acc(ps,torch.tensor(donor,dtype=torch.long)),
    }

def factor_cue_slices(dev_rows,pred):
    out={}
    for field in ("corridor_length","cue_steps"):
        out[field]={}
        vals=sorted({r[field] for r in dev_rows})
        for v in vals:
            idx=[i for i,r in enumerate(dev_rows) if r[field]==v]
            y=torch.tensor([dev_rows[i]["cue_label"] for i in idx],dtype=torch.long)
            pp=pred[torch.tensor(idx,dtype=torch.long)]
            out[field][str(v)]={"n":len(idx)//2,"accuracy":acc(pp,y)}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); device=torch_directml.device(); model,cfg=build(Path(a.checkpoint),device)
    print("WORKING_MEMORY_CAPACITY_DEV_REPRESENTATION_V1")
    print("brain_device:",device); print("reader_device: cpu")
    print("reader_training_split: TRAIN_ONLY"); print("reader_evaluation_split: DEV_ONLY"); print("test_touched: false")
    train=collect(model,cfg,device,TRAIN_CONDITION_RANGE,"train")
    dev=collect(model,cfg,device,DEV_CONDITION_RANGE,"dev")
    mu,sd=stats(train); tx,ty=cue_tensors(train,mu,sd); vx,vy=cue_tensors(dev,mu,sd)
    train_ex=expand(train); dev_ex=expand(dev); jx,jy=joint_tensors(train_ex,mu,sd)
    runs=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        cue=fit(Cue(),tx,ty,EPOCHS_LIN,LR_LIN)
        with torch.no_grad(): cue_pred=cue(vx).argmax(1)
        torch.manual_seed(seed)
        joint=fit(Joint(),jx,jy,EPOCHS_MLP,LR_MLP)
        jm=joint_metrics(joint,dev_ex,mu,sd)
        torch.manual_seed(seed)
        jl=fit(JointLinear(),jx,jy,EPOCHS_LIN,LR_LIN)
        with torch.no_grad(): lin_acc=acc(jl(joint_tensors(dev_ex,mu,sd)[0]).argmax(1),joint_tensors(dev_ex,mu,sd)[1])
        runs.append({
            "seed":seed,"cue_accuracy":acc(cue_pred,vy),
            "cue_factor_slices":factor_cue_slices(dev,cue_pred),
            "joint":jm,"linear_joint_control":lin_acc,
            "factorized_postwrite_accuracy":acc(cue_pred,vy),
        })
    def avg(path):
        vals=[]
        for r in runs:
            x=r
            for k in path: x=x[k]
            vals.append(float(x))
        return sum(vals)/len(vals)
    summary={
        "cue_accuracy":avg(("cue_accuracy",)),
        "joint_accuracy":avg(("joint","accuracy")),
        "joint_normal":avg(("joint","normal")),
        "joint_invert":avg(("joint","invert")),
        "joint_dual_both":avg(("joint","dual_both")),
        "joint_memory_zero":avg(("joint","memory_zero")),
        "joint_forced_normal":avg(("joint","forced_normal")),
        "joint_instruction_flip":avg(("joint","instruction_flip")),
        "joint_swap_donor":avg(("joint","swap_donor")),
        "linear_joint_control":avg(("linear_joint_control",)),
        "factorized_postwrite_accuracy":avg(("factorized_postwrite_accuracy",)),
    }
    # minimum cue accuracy over every reported factor slice and seed
    all_slice=[float(s["accuracy"]) for r in runs for fld in r["cue_factor_slices"].values() for s in fld.values()]
    summary["minimum_cue_factor_slice_accuracy"]=min(all_slice)
    result={"format":"working_memory_capacity_dev_representation_v1","split":"dev",
            "reader_fit_split":"train","summary":summary,"runs":runs,"test_touched":False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    print()
    for k,v in summary.items(): print(f"{k}: {v:.4f}")
    print("historical_base_bit_exact: true")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_CAPACITY_DEV_REPRESENTATION_COMPLETE")

if __name__=="__main__": main()
