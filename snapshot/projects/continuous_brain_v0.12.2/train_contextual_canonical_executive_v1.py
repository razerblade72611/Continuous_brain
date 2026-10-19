#!/usr/bin/env python3
"""
TRAIN-only contextual canonical executive training.

Each update uses one v3 TRAIN structural condition.

Own-content objective:
  all 2 cues × 3 rules, with post-instruction action preservation plus response.

Coherent-donor objective:
  at the response step, hold recipient historical workspace fixed, substitute
  opposite-cue canonical recall, and require the donor-cue response under the
  same rule.

Total loss:
  0.5 * mean(own six member losses)
+ 0.5 * mean(six coherent-donor response losses)

No v3 DEV/TEST access.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,_image_tensor,_load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.contextual_canonical_executive_v1 import (
    ContextualCanonicalExecutiveInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    TRAIN_CONDITION_RANGE, CanonicalExecutiveRuleWorld, target_action, UP,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
FORMAT="contextual_canonical_executive_training_v1"
RULES=("invert_lr","vertical","vertical_reverse")
EPOCH_SIZE=1600
SCHEDULE_SEED=1031441
INIT_SEED=616162


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
    def portable_state(self):
        return {"step_count":self.step_count,"lr":self.lr,"beta1":self.b1,"beta2":self.b2,"eps":self.eps,
                "state":[{"exp_avg":s["m"].detach().cpu(),"exp_avg_sq":s["v"].detach().cpu()} for s in self.state]}
    def load_portable_state(self,p):
        self.step_count=int(p["step_count"]); self.b1=float(p.get("beta1",.9)); self.b2=float(p.get("beta2",.999)); self.eps=float(p.get("eps",1e-8))
        if len(p["state"])!=len(self.state): raise RuntimeError("optimizer state mismatch")
        for d,s in zip(self.state,p["state"]):
            d["m"].copy_(s["exp_avg"].to(d["m"].device)); d["v"].copy_(s["exp_avg_sq"].to(d["v"].device))


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
def schedule(epoch):
    ids=list(TRAIN_CONDITION_RANGE); random.Random(SCHEDULE_SEED+1000003*int(epoch)).shuffle(ids); return ids


def load(device):
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256: raise RuntimeError("admitted SHA mismatch")
    ap=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("base mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(ap["memory"],strict=True); admitted.recall.load_state_dict(ap["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for p in admitted.parameters(): p.requires_grad_(False)
    model=ContextualCanonicalExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True); model.admitted.eval(); model.base.eval()
    return model,cfg,clone(admitted),clone(base),model.trainable_parameters()


@torch.no_grad()
def collect_post_snapshots(model,cfg,device,cid,cue):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule="invert_lr"); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device)
    snaps=[]
    while not env.done:
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        if env.instruction_visible():
            snaps.append({"image":image.detach(),"context":context.detach(),"hidden":h.detach(),"memory":mem.detach(),
                          "prev":prev.detach(),"response":bool(env.at_response_step())})
        if env.at_response_step(): break
        teacher=int(env.teacher_action())
        out=model.admitted.step(image,context,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher); h=out["hidden"]; mem=out["memory_next"]
        prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
    if not snaps or not snaps[-1]["response"]: raise AssertionError("missing response snapshot")
    return snaps


def response_components(model,snap,memory_for_recall):
    hist=model.admitted.historical_step(snap["image"],snap["context"],snap["memory"],snap["prev"],snap["hidden"])
    recall,_=model.admitted.recall(hist["hidden"],memory_for_recall)
    return hist,recall


def update_loss(model,cfg,device,cid):
    snaps={cue:collect_post_snapshots(model,cfg,device,cid,cue) for cue in ("left","right")}
    own_losses=[]; donor_losses=[]; metrics={}; delta_max=0.0

    for cue in ("left","right"):
        donor_cue="right" if cue=="left" else "left"
        donor_memory=snaps[donor_cue][-1]["memory"]

        for rid,rule in enumerate(RULES,1):
            rule_id=torch.tensor([rid],dtype=torch.long,device=device)
            step_losses=[]; response_pred=None; response_target=None

            for snap in snaps[cue]:
                hist,recall=response_components(model,snap,snap["memory"])
                logits,delta,_,_,_=model.compose(hist,recall,rule_id)
                teacher=target_action(cue,rule) if snap["response"] else UP
                target=torch.tensor([teacher],dtype=torch.long,device=device)
                step_losses.append(F.cross_entropy(logits,target))
                delta_max=max(delta_max,float(delta.detach().abs().max().cpu()))
                if snap["response"]:
                    response_pred=int(torch.argmax(logits,dim=-1)[0].item()); response_target=teacher

            response_loss=step_losses[-1]
            own_losses.append(.5*torch.stack(step_losses).mean()+.5*response_loss)
            metrics[(cue,rule)]=bool(response_pred==response_target)

            rsnap=snaps[cue][-1]
            hist,donor_recall=response_components(model,rsnap,donor_memory)
            donor_logits,_,_,_,_=model.compose(hist,donor_recall,rule_id)
            donor_target=target_action(donor_cue,rule)
            donor_losses.append(F.cross_entropy(donor_logits,torch.tensor([donor_target],dtype=torch.long,device=device)))

    own=torch.stack(own_losses).mean(); donor=torch.stack(donor_losses).mean()
    return .5*own+.5*donor,own,donor,metrics,delta_max


def save(path,model,opt,update,lr):
    torch.save({"format":FORMAT,"global_update":int(update),"update":int(update),
                "source_admitted_checkpoint_sha256":EXPECTED_ADMITTED_SHA256,
                "historical_base_sha256":EXPECTED_BASE_SHA256,"historical_base_digest":EXPECTED_BASE_DIGEST,
                "schedule_seed":SCHEDULE_SEED,"init_seed":INIT_SEED,"current_lr":float(lr),
                "executive":{k:v.detach().cpu() for k,v in model.executive.state_dict().items()},
                "optimizer":opt.portable_state()},path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state-dir",default="state/contextual_canonical_executive_v1")
    ap.add_argument("--target-update",type=int,required=True); ap.add_argument("--lr",type=float,required=True); a=ap.parse_args()
    if a.target_update%EPOCH_SIZE: raise ValueError("target must end on full epoch")
    state=Path(a.state_dir); state.mkdir(parents=True,exist_ok=True); latest=state/"latest.portable.pt"; log=state/"training.jsonl"
    torch.manual_seed(INIT_SEED); device=torch_directml.device(); model,cfg,aref,bref,params=load(device); opt=DMLSafeAdam(params,lr=a.lr)
    completed=0
    if latest.is_file():
        p=torch.load(latest,map_location="cpu",weights_only=True)
        if p.get("format")!=FORMAT: raise RuntimeError("latest format mismatch")
        model.executive.load_state_dict(p["executive"],strict=True); opt.load_portable_state(p["optimizer"]); completed=int(p["global_update"])
    if completed>a.target_update: raise RuntimeError("latest exceeds target")
    if completed==a.target_update: print("already_at_target_update:",completed); return
    opt.lr=float(a.lr)
    print("CONTEXTUAL_CANONICAL_EXECUTIVE_TRAINING_V1"); print("device:",device)
    print("executive_parameter_count:",sum(p.numel() for p in params)); print("start_global_update:",completed+1)
    print("target_global_update:",a.target_update); print("stage_lr:",a.lr)
    print("own_loss_weight: 0.5"); print("coherent_donor_loss_weight: 0.5")
    print("v3_train_only: true"); print("v3_dev_touched: false"); print("v3_test_touched: false")
    started=time.time()
    with log.open("a" if log.exists() else "w") as f:
        for u in range(completed+1,a.target_update+1):
            z=u-1; epoch=z//EPOCH_SIZE; within=z%EPOCH_SIZE; cid=schedule(epoch)[within]
            opt.zero_grad(set_to_none=True); loss,own,donor,corr,dmax=update_loss(model,cfg,device,cid); loss.backward(); opt.step()
            rec={"global_update":u,"epoch_index":epoch,"condition_id":int(cid),"lr":float(opt.lr),
                 "loss":float(loss.detach().cpu()),"own_loss":float(own.detach().cpu()),"donor_loss":float(donor.detach().cpu()),
                 "six_way_full":bool(all(corr.values()))}
            for r in RULES: rec[f"{r}_pair_full"]=bool(corr[("left",r)] and corr[("right",r)])
            f.write(json.dumps(rec)+"\n"); f.flush()
            if u==completed+1 or u%25==0 or u==a.target_update:
                pairs={r:int(corr[("left",r)] and corr[("right",r)]) for r in RULES}
                print(f"update={u:05d}/{a.target_update:05d} epoch={epoch+1} condition={cid:04d} loss={rec['loss']:.6f} own={rec['own_loss']:.6f} donor={rec['donor_loss']:.6f} invert={pairs['invert_lr']} vertical={pairs['vertical']} vreverse={pairs['vertical_reverse']} six={int(all(corr.values()))} delta={dmax:.5f}")
            if u%100==0 or u==a.target_update:
                assert_exact(aref,model.admitted,"admitted"); assert_exact(bref,model.base,"base")
                ck=state/f"contextual_canonical_u{u:06d}.portable.pt"; save(ck,model,opt,u,opt.lr); latest.write_bytes(ck.read_bytes()); print("saved:",ck)
    assert_exact(aref,model.admitted,"admitted final"); assert_exact(bref,model.base,"base final")
    print("elapsed_seconds:",round(time.time()-started,3)); print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true")
    print("V3_DEV_TOUCHED: false"); print("V3_TEST_TOUCHED: false"); print("CONTEXTUAL_CANONICAL_EXECUTIVE_TRAINING_COMPLETE")

if __name__=="__main__": main()
