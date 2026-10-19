#!/usr/bin/env python3
"""
Production training for second-generation novel-rule executive.

Protected:
- historical 14k brain;
- DEV-admitted working-memory writer/reader;
- frozen motor/world model.

Trainable:
- new NovelRuleExecutive only.

Every update uses one fresh structural TRAIN condition and all six:
    2 cues × 3 late rules.

Fresh DEV/TEST are not accessed during training.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path
import torch, torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,_image_tensor,_load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.executive_novel_rules_v2 import NovelRuleExecutiveInfant
from continuous_brain.executive_novel_rule_world_v2 import (
    TRAIN_CONDITION_RANGE, ExecutiveNovelRuleWorld,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
CHECKPOINT_FORMAT="executive_novel_rules_training_v2"
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
        self.step_count+=1; bc1=1-self.b1**self.step_count; bc2=1-self.b2**self.step_count
        for p,s in zip(self.params,self.state):
            if p.grad is None: continue
            g=p.grad
            if not torch.isfinite(g).all().item(): raise RuntimeError("non-finite grad")
            s["m"].mul_(self.b1).add_(g,alpha=1-self.b1)
            s["v"].mul_(self.b2).add_(g*g,alpha=1-self.b2)
            den=s["v"].sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            p.add_(s["m"]/den,alpha=-self.lr/bc1)
    def portable_state(self):
        return {
            "step_count":self.step_count,"lr":self.lr,
            "beta1":self.b1,"beta2":self.b2,"eps":self.eps,
            "state":[{"exp_avg":x["m"].detach().cpu(),"exp_avg_sq":x["v"].detach().cpu()} for x in self.state],
        }
    def load_portable_state(self, payload):
        self.step_count=int(payload["step_count"])
        if len(payload["state"]) != len(self.state):
            raise RuntimeError("optimizer parameter count mismatch")
        for dst,src in zip(self.state,payload["state"]):
            dst["m"].copy_(src["exp_avg"].to(dst["m"].device))
            dst["v"].copy_(src["exp_avg_sq"].to(dst["v"].device))


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
        if not torch.equal(v,m.state_dict()[k].detach().cpu()):
            raise AssertionError(f"{label}: changed {k}")

def schedule(seed):
    ids=list(TRAIN_CONDITION_RANGE); rng=random.Random(seed); rng.shuffle(ids); return ids

def load(device):
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256: raise RuntimeError("admitted SHA mismatch")
    p=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(p["memory"],strict=True); admitted.recall.load_state_dict(p["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for q in admitted.parameters(): q.requires_grad_(False)
    model=NovelRuleExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True); model.admitted.eval(); model.base.eval()
    return model,cfg,clone(admitted),clone(base),model.trainable_parameters()

def run_member(model,cfg,device,cid,cue,rule):
    env=ExecutiveNovelRuleWorld(cid,cue=cue,rule=rule); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device)
    prev=torch.zeros(1,dtype=torch.long,device=device)
    post=[]; response=None; pred=None; rmax=0.0
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        rid=torch.tensor([env.instruction_id()],dtype=torch.long,device=device)
        out=model.step(image,context,mem,prev,h,rid)
        if not env.instruction_visible():
            if torch.count_nonzero(out["executive_residual"].detach().cpu()).item()!=0:
                raise AssertionError("pre-rule residual nonzero")
        teacher=int(env.teacher_action()); target=torch.tensor([teacher],dtype=torch.long,device=device)
        if env.instruction_visible():
            post.append(F.cross_entropy(out["logits"],target))
        if env.at_response_step():
            response=F.cross_entropy(out["logits"],target)
            pred=int(torch.argmax(out["logits"],dim=-1)[0].item())
        rmax=max(rmax,float(out["executive_residual"].detach().abs().max().cpu()))
        obs,_,_,_=env.step(teacher)
        h=out["hidden"].detach(); mem=out["memory_next"].detach(); prev=(target+1).long()
    if response is None: raise AssertionError("no response")
    return .5*torch.stack(post).mean()+.5*response, bool(pred==teacher), rmax

def sixway(model,cfg,device,cid):
    losses=[]; correct={}; rmax=0
    for cue in ("left","right"):
        for rule in RULES:
            loss,ok,r=run_member(model,cfg,device,cid,cue,rule)
            losses.append(loss); correct[(cue,rule)]=ok; rmax=max(rmax,r)
    return torch.stack(losses).mean(),correct,rmax

def save(path,model,opt,update,args):
    torch.save({
        "format":CHECKPOINT_FORMAT,"update":int(update),
        "source_admitted_checkpoint_sha256":EXPECTED_ADMITTED_SHA256,
        "historical_base_sha256":EXPECTED_BASE_SHA256,
        "historical_base_digest":EXPECTED_BASE_DIGEST,
        "schedule_seed":args.schedule_seed,"init_seed":args.init_seed,"lr":args.lr,
        "executive":{k:v.detach().cpu() for k,v in model.executive.state_dict().items()},
        "optimizer":opt.portable_state(),
    },path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state-dir",default="state/executive_novel_rules_v2")
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--schedule-seed",type=int,default=731927)
    ap.add_argument("--init-seed",type=int,default=24682)
    ap.add_argument("--resume",default=None)
    a=ap.parse_args()
    state=Path(a.state_dir); state.mkdir(parents=True,exist_ok=True)
    log=state/"training.jsonl"

    torch.manual_seed(a.init_seed); device=torch_directml.device()
    model,cfg,aref,bref,params=load(device); opt=DMLSafeAdam(params,lr=a.lr); ids=schedule(a.schedule_seed)

    start_update=1
    log_mode="w"
    if a.resume:
        rp=Path(a.resume)
        payload=torch.load(rp,map_location="cpu",weights_only=True)
        if payload.get("format")!=CHECKPOINT_FORMAT:
            raise RuntimeError("resume checkpoint format mismatch")
        if payload.get("source_admitted_checkpoint_sha256")!=EXPECTED_ADMITTED_SHA256:
            raise RuntimeError("resume admitted SHA mismatch")
        if int(payload.get("schedule_seed",-1))!=a.schedule_seed or int(payload.get("init_seed",-1))!=a.init_seed:
            raise RuntimeError("resume seed mismatch")
        model.executive.load_state_dict(payload["executive"],strict=True)
        opt.load_portable_state(payload["optimizer"])
        completed=int(payload["update"])
        start_update=completed+1
        if not log.exists():
            raise RuntimeError("resume requested but training.jsonl missing")
        lines=[x for x in log.read_text().splitlines() if x.strip()]
        kept=[]
        for line in lines:
            row=json.loads(line)
            if int(row["update"])<=completed:
                kept.append(line)
        if len(kept)!=completed:
            raise RuntimeError("training log does not match checkpoint update")
        log.write_text("\n".join(kept)+("\n" if kept else ""))
        log_mode="a"
    print("EXECUTIVE_NOVEL_RULE_PRODUCTION_TRAINING_V2")
    print("device:",device); print("updates:",len(ids)); print("members_per_update: 6"); print("start_update:",start_update)
    print("fresh_train_range: 0:1600"); print("fresh_dev_touched: false"); print("fresh_test_touched: false")
    started=time.time()
    with log.open(log_mode) as f:
        for u,cid in enumerate(ids,1):
            if u < start_update:
                continue
            opt.zero_grad(set_to_none=True); loss,corr,rmax=sixway(model,cfg,device,cid); loss.backward(); opt.step()
            if u==1 or u%25==0 or u==len(ids):
                pair={rule:int(corr[("left",rule)] and corr[("right",rule)]) for rule in RULES}
                six=int(all(corr.values()))
                print(f"update={u:04d}/1600 condition={cid:04d} loss={float(loss.detach().cpu()):.6f} invert={pair['invert_lr']} vertical={pair['vertical']} vreverse={pair['vertical_reverse']} six={six} resid={rmax:.5f}")
            rec={"update":u,"condition_id":cid,"loss":float(loss.detach().cpu()),"six_way_full":bool(all(corr.values())),
                 **{f"{r}_pair_full":bool(corr[("left",r)] and corr[("right",r)]) for r in RULES}}
            f.write(json.dumps(rec)+"\n"); f.flush()
            if u%100==0 or u==len(ids):
                assert_exact(aref,model.admitted,"admitted"); assert_exact(bref,model.base,"base")
                ck=state/f"executive_novel_rules_u{u:06d}.portable.pt"; save(ck,model,opt,u,a)
                (state/"latest.portable.pt").write_bytes(ck.read_bytes()); print("saved:",ck)
    assert_exact(aref,model.admitted,"admitted final"); assert_exact(bref,model.base,"base final")
    print("elapsed_seconds:",round(time.time()-started,3))
    print("portable_cpu_checkpoint: true")
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("FRESH_DEV_TOUCHED: false")
    print("FRESH_TEST_TOUCHED: false")
    print("EXECUTIVE_NOVEL_RULE_PRODUCTION_TRAINING_COMPLETE")

if __name__=="__main__": main()
