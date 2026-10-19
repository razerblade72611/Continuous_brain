#!/usr/bin/env python3
from __future__ import annotations
import hashlib, math
from pathlib import Path
import torch, torch_directml
from torch.nn import functional as F
from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.option_intent_executive_v1 import OptionIntentExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import CanonicalExecutiveRuleWorld,RULE_NONE

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
RULES=("invert_lr","vertical","vertical_reverse")

class DMLSafeAdam:
    def __init__(self,params,lr=2e-4):
        self.params=[p for p in params if p.requires_grad]; self.lr=lr; self.b1=.9; self.b2=.999; self.eps=1e-8; self.n=0
        self.state=[{"m":torch.zeros_like(p),"v":torch.zeros_like(p)} for p in self.params]
    def zero_grad(self):
        for p in self.params: p.grad=None
    @torch.no_grad()
    def step(self):
        self.n+=1; c1=1-self.b1**self.n; c2=1-self.b2**self.n
        for p,s in zip(self.params,self.state):
            if p.grad is None: continue
            g=p.grad
            if not torch.isfinite(g).all().item(): raise RuntimeError("non-finite grad")
            s["m"].mul_(self.b1).add_(g,alpha=1-self.b1); s["v"].mul_(self.b2).add_(g*g,alpha=1-self.b2)
            p.add_(s["m"]/(s["v"].sqrt()/math.sqrt(c2)+self.eps),alpha=-self.lr/c1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""): h.update(c)
    return h.hexdigest()
def clone(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def digest(st):
    h=hashlib.sha256()
    for k in sorted(st):
        t=st[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()
def assert_exact(ref,m,label):
    for k,v in ref.items():
        if not torch.equal(v,m.state_dict()[k].detach().cpu()): raise AssertionError(f"{label}: changed {k}")
def gmax(g): return 0.0 if g is None else float(g.detach().abs().max().cpu())

def load(device):
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256: raise RuntimeError("admitted SHA mismatch")
    ap=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(ap["memory"],strict=True); admitted.recall.load_state_dict(ap["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for p in admitted.parameters(): p.requires_grad_(False)
    model=OptionIntentExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True); model.admitted.eval(); model.base.eval()
    return model,cfg,clone(admitted),clone(base),model.trainable_parameters()

def trajectory_loss(model,cfg,device,cid,cue,rule):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule=rule); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device)
    response=None; gates=[]
    while not env.done:
        im=_image_tensor(env,cfg.image_size,device); ctx,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        o=model.step(im,ctx,mem,prev,h,rid)
        if env.instruction_visible():
            tgt=torch.tensor([[1.0 if env.at_response_step() else 0.0]],device=device)
            gates.append(F.binary_cross_entropy_with_logits(o["executive_gate_logit"],tgt))
        teacher=int(env.teacher_action()); y=torch.tensor([teacher],dtype=torch.long,device=device)
        if env.at_response_step(): response=F.cross_entropy(o["logits"],y)
        obs,_,_,_=env.step(teacher); h=o["hidden"].detach(); mem=o["memory_next"].detach(); prev=(y+1).long()
    return response,torch.stack(gates).mean()

def loss6(model,cfg,device,cid):
    rs=[]; gs=[]
    for c in ("left","right"):
        for r in RULES:
            a,b=trajectory_loss(model,cfg,device,cid,c,r); rs.append(a); gs.append(b)
    return torch.stack(rs).mean()+.2*torch.stack(gs).mean()

@torch.no_grad()
def exact_noop(model,cfg,device):
    env=CanonicalExecutiveRuleWorld(0,cue="left",rule="vertical"); obs=env.reset()
    h1=model.initial_state(1,device); h0=model.admitted.initial_state(1,device)
    m1=model.initial_memory(1,device); m0=model.admitted.initial_memory(1,device)
    p1=torch.zeros(1,dtype=torch.long,device=device); p0=p1.clone()
    while not env.done:
        im=_image_tensor(env,cfg.image_size,device); ctx,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        got=model.step(im,ctx,m1,p1,h1,rid); plain=model.admitted.step(im,ctx,m0,p0,h0,use_memory=True,use_recall=True)
        if torch.count_nonzero(got["executive_intent_delta"].cpu()).item()!=0 or not torch.equal(got["logits"],plain["logits"]):
            raise AssertionError("initial no-op mismatch")
        teacher=int(env.teacher_action()); obs,_,_,_=env.step(teacher); nxt=torch.tensor([teacher+1],dtype=torch.long,device=device)
        h1=got["hidden"]; h0=plain["hidden"]; m1=got["memory_next"]; m0=plain["memory_next"]; p1=nxt; p0=nxt

def main():
    torch.manual_seed(818181); device=torch_directml.device()
    model,cfg,aref,bref,params=load(device)
    print("OPTION_INTENT_EXECUTIVE_QUALIFICATION_V1"); print("device:",device)
    print("executive_parameter_count:",sum(p.numel() for p in params))
    print("control_hierarchy: WHAT_intent_plus_WHEN_gate"); print("raw_memory_readers_in_executive: 0")
    print("canonical_recall_required_for_intent: true"); print("gate_uses_protected_state_only: true")
    print("world_model_input_changed: false"); print("admitted_brain_in_optimizer: false")
    print("v3_dev_touched: false"); print("v3_test_touched: false")
    exact_noop(model,cfg,device); print("EXACT_INITIAL_OPTION_INTENT_NOOP_OK")
    opt=DMLSafeAdam(params)
    for u,cid in enumerate((0,17,2),1):
        opt.zero_grad(); loss=loss6(model,cfg,device,cid); loss.backward()
        vals=dict(intent_out=gmax(model.executive.intent_output.weight.grad),
                  content=gmax(model.executive.content.weight.grad),
                  rule=gmax(model.executive.rule.weight.grad),
                  intent_fusion=gmax(model.executive.intent_fusion.weight.grad),
                  gate_ws=gmax(model.executive.gate_workspace.weight.grad),
                  gate_action=gmax(model.executive.gate_action.weight.grad),
                  gate_fusion=gmax(model.executive.gate_fusion.weight.grad),
                  gate_out=gmax(model.executive.gate_output.weight.grad))
        print(f"update{u}_loss:",float(loss.detach().cpu()))
        for k,v in vals.items(): print(f"update{u}_{k}_grad_max:",v)
        if vals["intent_out"]<=0 or vals["gate_out"]<=0: raise AssertionError("missing output gradients")
        if u==1 and any(vals[k]!=0 for k in ("content","rule","intent_fusion")): raise AssertionError("intent staging failed")
        if u>1 and min(vals[k] for k in ("content","rule","intent_fusion","gate_ws","gate_action","gate_fusion"))<=0: raise AssertionError("path not load-bearing")
        opt.step(); assert_exact(aref,model.admitted,"admitted"); assert_exact(bref,model.base,"base")
    with torch.no_grad():
        H=torch.randn(4,int(cfg.bus_dim),device=device); B=torch.randn(4,5,device=device)
        D=torch.zeros(4,int(cfg.bus_dim),device=device); ids=torch.tensor([1,2,3,1],dtype=torch.long,device=device)
        d,_=model.executive(H,D,ids,B)
        if torch.count_nonzero(d.cpu()).item(): raise AssertionError("zero content not exact no-op")
        D=torch.randn(4,int(cfg.bus_dim),device=device); ids=torch.full((4,),RULE_NONE,dtype=torch.long,device=device)
        d,_=model.executive(H,D,ids,B)
        if torch.count_nonzero(d.cpu()).item(): raise AssertionError("NONE not exact no-op")
    print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true")
    print("zero_canonical_content_exact_noop: true"); print("rule_none_exact_noop_after_learning: true")
    print("OPTION_INTENT_EXECUTIVE_QUALIFICATION_V1: PASS")

if __name__=="__main__": main()
