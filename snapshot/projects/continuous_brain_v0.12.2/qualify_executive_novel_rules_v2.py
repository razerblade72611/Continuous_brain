#!/usr/bin/env python3
"""Neural qualification for the fresh novel-rule executive v2."""
from __future__ import annotations
import hashlib, math
from pathlib import Path
import torch, torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor, _image_tensor, _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.executive_novel_rules_v2 import NovelRuleExecutiveInfant
from continuous_brain.executive_novel_rule_world_v2 import (
    ExecutiveNovelRuleWorld, RULE_NONE,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
RULES=("invert_lr","vertical","vertical_reverse")


class DMLSafeAdam:
    def __init__(self,params,lr=2e-4,betas=(.9,.999),eps=1e-8):
        self.params=[p for p in params if p.requires_grad]; self.lr=float(lr)
        self.b1=float(betas[0]); self.b2=float(betas[1]); self.eps=float(eps)
        self.step_count=0
        self.state=[{"m":torch.zeros_like(p),"v":torch.zeros_like(p)} for p in self.params]
    def zero_grad(self,set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none: p.grad=None
                else: p.grad.zero_()
    @torch.no_grad()
    def step(self):
        self.step_count+=1
        bc1=1-self.b1**self.step_count; bc2=1-self.b2**self.step_count
        for p,s in zip(self.params,self.state):
            if p.grad is None: continue
            g=p.grad
            if not torch.isfinite(g).all().item(): raise RuntimeError("non-finite grad")
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
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def assert_exact(ref,m,label):
    for k,v in ref.items():
        got=m.state_dict()[k].detach().cpu()
        if not torch.equal(v,got): raise AssertionError(f"{label}: changed {k}")

def gmax(g):
    return 0.0 if g is None else float(g.detach().abs().max().cpu())

def load(device):
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")
    p=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST:
        raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(p["memory"],strict=True)
    admitted.recall.load_state_dict(p["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for q in admitted.parameters(): q.requires_grad_(False)
    model=NovelRuleExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True); model.admitted.eval(); model.base.eval()
    return model,cfg,clone(admitted),clone(base),model.trainable_parameters()


def run_member(model,cfg,device,cid,cue,rule):
    env=ExecutiveNovelRuleWorld(cid,cue=cue,rule=rule)
    obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device)
    prev=torch.zeros(1,dtype=torch.long,device=device)
    response_loss=None; residual_max=0.0
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device)
        context,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        out=model.step(image,context,mem,prev,h,rid)
        if not env.instruction_visible():
            if torch.count_nonzero(out["executive_residual"].detach().cpu()).item()!=0:
                raise AssertionError("pre-instruction residual nonzero")
        teacher=int(env.teacher_action())
        target=torch.tensor([teacher],dtype=torch.long,device=device)
        if env.at_response_step():
            response_loss=F.cross_entropy(out["logits"],target)
        residual_max=max(residual_max,float(out["executive_residual"].detach().abs().max().cpu()))
        obs,_,_,_=env.step(teacher)
        h=out["hidden"].detach(); mem=out["memory_next"].detach(); prev=(target+1).long()
    if response_loss is None: raise AssertionError("no response loss")
    return response_loss,residual_max


def sixway(model,cfg,device,cid):
    losses=[]; rmax=0
    for cue in ("left","right"):
        for rule in RULES:
            loss,r=run_member(model,cfg,device,cid,cue,rule)
            losses.append(loss); rmax=max(rmax,r)
    return torch.stack(losses).mean(),rmax


def main():
    torch.manual_seed(24681); device=torch_directml.device()
    model,cfg,aref,bref,params=load(device)
    print("EXECUTIVE_NOVEL_RULE_QUALIFICATION_V2")
    print("device:",device)
    print("executive_parameter_count:",sum(p.numel() for p in params))
    print("members_per_update: 6")
    print("admitted_brain_in_optimizer: false")
    print("fresh_dev_touched: false")
    print("fresh_test_touched: false")

    # Exact zero-init no-op at active rules before learning.
    with torch.no_grad():
        mem=torch.randn(6,int(cfg.memory_dim),device=device)
        for rid in (1,2,3):
            ids=torch.full((6,),rid,dtype=torch.long,device=device)
            residual,_=model.executive(mem,ids)
            if torch.count_nonzero(residual.detach().cpu()).item()!=0:
                raise AssertionError("initial residual not exact zero")
    print("EXACT_INITIAL_EXECUTIVE_NOOP_OK")

    opt=DMLSafeAdam(params,lr=2e-4)
    for update,cid in enumerate((0,17,2),1):
        opt.zero_grad(set_to_none=True)
        loss,rmax=sixway(model,cfg,device,cid); loss.backward()
        og=gmax(model.executive.output.weight.grad)
        mg=gmax(model.executive.memory_features.weight.grad)
        rg=gmax(model.executive.rule.weight.grad)
        print(f"update{update}_loss:",float(loss.detach().cpu()))
        print(f"update{update}_residual_max:",rmax)
        print(f"update{update}_output_grad_max:",og)
        print(f"update{update}_memory_grad_max:",mg)
        print(f"update{update}_rule_grad_max:",rg)
        if og<=0: raise AssertionError("missing output gradient")
        if update==1:
            if mg!=0 or rg!=0: raise AssertionError("upstream learned before output moved")
        else:
            if mg<=0 or rg<=0: raise AssertionError("memory/rule path not load-bearing")
        opt.step()
        assert_exact(aref,model.admitted,"admitted")
        assert_exact(bref,model.base,"base")

    with torch.no_grad():
        mem=torch.randn(4,int(cfg.memory_dim),device=device)
        ids=torch.full((4,),RULE_NONE,dtype=torch.long,device=device)
        residual,_=model.executive(mem,ids)
        if torch.count_nonzero(residual.detach().cpu()).item()!=0:
            raise AssertionError("NONE not exact no-op after learning")

    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("rule_none_exact_noop_after_learning: true")
    print("staged_gradients: true")
    print("EXECUTIVE_NOVEL_RULE_QUALIFICATION_V2: PASS")

if __name__=="__main__": main()
