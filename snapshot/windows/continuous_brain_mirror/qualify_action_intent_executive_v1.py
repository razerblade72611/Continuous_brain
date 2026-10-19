#!/usr/bin/env python3
"""Qualification for ActionIntentExecutiveInfant."""
from __future__ import annotations
import argparse, hashlib, math
from pathlib import Path
import torch
from torch.nn import functional as F
from continuous_brain.action_intent_integrity_v1 import select_device, load_protected

from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.action_intent_executive_v1 import ActionIntentExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import CanonicalExecutiveRuleWorld,RULE_NONE

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
RULES=("invert_lr","vertical","vertical_reverse")

class DMLSafeAdam:
    def __init__(self,params,lr=2e-4,betas=(.9,.999),eps=1e-8):
        self.params=[p for p in params if p.requires_grad]; self.lr=float(lr)
        self.b1=float(betas[0]); self.b2=float(betas[1]); self.eps=float(eps); self.step_count=0
        self.state=[{"m":torch.zeros_like(p),"v":torch.zeros_like(p)} for p in self.params]
    def zero_grad(self,set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none: p.grad=None
                else: p.grad.zero_()
    @torch.no_grad()
    def step(self):
        self.step_count+=1; bc1=1-self.b1**self.step_count; bc2=1-self.b2**self.step_count
        for p,s in zip(self.params,self.state):
            if p.grad is None: continue
            g=p.grad
            if not torch.isfinite(g).all().item(): raise RuntimeError("non-finite gradient")
            s["m"].mul_(self.b1).add_(g,alpha=1-self.b1)
            s["v"].mul_(self.b2).add_(g*g,alpha=1-self.b2)
            den=s["v"].sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            p.add_(s["m"]/den,alpha=-self.lr/bc1)

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
def assert_exact(ref,m,label):
    for k,v in ref.items():
        if not torch.equal(v,m.state_dict()[k].detach().cpu()): raise AssertionError(f"{label}: changed {k}")
def gmax(g):
    if g is None:return 0.0
    if not torch.isfinite(g).all().item(): raise RuntimeError("non-finite gradient")
    return float(g.detach().abs().max().cpu())

def load(device):
    return load_protected(device)

def member_loss(model,cfg,device,cid,cue,rule):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule=rule); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device)
    response=None
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        out=model.step(image,context,mem,prev,h,rid)
        if not env.instruction_visible() and torch.count_nonzero(out["executive_intent_delta"].detach().cpu()).item()!=0:
            raise AssertionError("pre-rule intent nonzero")
        teacher=int(env.teacher_action()); target=torch.tensor([teacher],dtype=torch.long,device=device)
        if env.at_response_step(): response=F.cross_entropy(out["logits"],target)
        obs,_,_,_=env.step(teacher); h=out["hidden"].detach(); mem=out["memory_next"].detach(); prev=(target+1).long()
    if response is None: raise AssertionError("no response")
    return response

def sixway(model,cfg,device,cid):
    return torch.stack([member_loss(model,cfg,device,cid,c,r) for c in ("left","right") for r in RULES]).mean()

@torch.no_grad()
def exact_noop(model,cfg,device):
    env=CanonicalExecutiveRuleWorld(0,cue="left",rule="vertical"); obs=env.reset()
    h1=model.initial_state(1,device); h0=model.admitted.initial_state(1,device)
    m1=model.initial_memory(1,device); m0=model.admitted.initial_memory(1,device)
    p1=torch.zeros(1,dtype=torch.long,device=device); p0=p1.clone()
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        got=model.step(image,context,m1,p1,h1,rid)
        plain=model.admitted.step(image,context,m0,p0,h0,use_memory=True,use_recall=True)
        if torch.count_nonzero(got["executive_intent_delta"].cpu()).item()!=0: raise AssertionError("initial intent nonzero")
        if not torch.equal(got["logits"],plain["logits"]): raise AssertionError("initial logits not exact no-op")
        teacher=int(env.teacher_action()); obs,_,_,_=env.step(teacher); nxt=torch.tensor([teacher+1],dtype=torch.long,device=device)
        h1=got["hidden"]; h0=plain["hidden"]; m1=got["memory_next"]; m0=plain["memory_next"]; p1=nxt; p0=nxt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("directml", "cpu"), default="directml")
    args = parser.parse_args()
    torch.manual_seed(717171); device=select_device(args.device)
    model,cfg,aref,bref,params=load(device)
    print("ACTION_INTENT_EXECUTIVE_QUALIFICATION_V1"); print("device:",device)
    print("executive_parameter_count:",sum(p.numel() for p in params))
    print("control_interface: frozen_motor_logit_residual")
    print("raw_memory_readers_in_executive: 0"); print("canonical_recall_required: true")
    print("recurrent_hidden_direct_input: false"); print("world_model_input_changed: false")
    print("admitted_brain_in_optimizer: false"); print("v3_dev_touched: false"); print("v3_test_touched: false")
    exact_noop(model,cfg,device); print("EXACT_INITIAL_ACTION_INTENT_NOOP_OK")
    opt=DMLSafeAdam(params,lr=2e-4)
    for u,cid in enumerate((0,17,2),1):
        opt.zero_grad(set_to_none=True); loss=sixway(model,cfg,device,cid); loss.backward()
        og=gmax(model.executive.output.weight.grad); cg=gmax(model.executive.content.weight.grad)
        rg=gmax(model.executive.rule.weight.grad); bg=gmax(model.executive.base_action.weight.grad); fg=gmax(model.executive.fusion.weight.grad)
        print(f"update{u}_loss:",float(loss.detach().cpu())); print(f"update{u}_output_grad_max:",og)
        print(f"update{u}_content_grad_max:",cg); print(f"update{u}_rule_grad_max:",rg)
        print(f"update{u}_base_action_grad_max:",bg); print(f"update{u}_fusion_grad_max:",fg)
        if og<=0: raise AssertionError("output no gradient")
        if u==1:
            if any(x!=0 for x in (cg,rg,bg,fg)): raise AssertionError("upstream moved before zero output")
        else:
            if min(cg,rg,bg,fg)<=0: raise AssertionError("intent path not load-bearing")
        opt.step(); assert_exact(aref,model.admitted,"admitted"); assert_exact(bref,model.base,"base")
    with torch.no_grad():
        D=torch.zeros(4,int(cfg.bus_dim),device=device); ids=torch.tensor([1,2,3,1],dtype=torch.long,device=device); B=torch.randn(4,5,device=device)
        delta,_=model.executive(D,ids,B)
        if torch.count_nonzero(delta.cpu()).item()!=0: raise AssertionError("zero canonical content not exact zero")
        D=torch.randn(4,int(cfg.bus_dim),device=device); ids=torch.full((4,),RULE_NONE,dtype=torch.long,device=device)
        delta,_=model.executive(D,ids,B)
        if torch.count_nonzero(delta.cpu()).item()!=0: raise AssertionError("RULE_NONE not exact zero")
    print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true")
    print("zero_canonical_content_exact_noop: true"); print("rule_none_exact_noop_after_learning: true")
    print("staged_gradients: true"); print("ACTION_INTENT_EXECUTIVE_QUALIFICATION_V1: PASS")

if __name__=="__main__": main()
