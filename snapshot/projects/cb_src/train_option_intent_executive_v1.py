#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,random,time
from pathlib import Path
import torch,torch_directml
from torch.nn import functional as F
from continuous_brain.evaluate_competitive_infant import _context_tensor,_image_tensor,_load_base_checkpoint
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from continuous_brain.option_intent_executive_v1 import OptionIntentExecutiveInfant
from continuous_brain.canonical_executive_rule_world_v3 import TRAIN_CONDITION_RANGE,CanonicalExecutiveRuleWorld,target_action

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT="state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256="f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
FORMAT="option_intent_executive_training_v1"
RULES=("invert_lr","vertical","vertical_reverse")
EPOCH_SIZE=1600; SCHEDULE_SEED=1314551; INIT_SEED=818182

class DMLSafeAdam:
    def __init__(self,params,lr=2e-4):
        self.params=[p for p in params if p.requires_grad]; self.lr=float(lr); self.b1=.9; self.b2=.999; self.eps=1e-8; self.n=0
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
    def portable_state(self):
        return {"step_count":self.n,"lr":self.lr,"state":[{"exp_avg":s["m"].detach().cpu(),"exp_avg_sq":s["v"].detach().cpu()} for s in self.state]}
    def load_portable_state(self,p):
        self.n=int(p["step_count"])
        for d,s in zip(self.state,p["state"]):
            d["m"].copy_(s["exp_avg"].to(d["m"].device)); d["v"].copy_(s["exp_avg_sq"].to(d["v"].device))

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
def schedule(epoch):
    ids=list(TRAIN_CONDITION_RANGE); random.Random(SCHEDULE_SEED+1000003*epoch).shuffle(ids); return ids

def load(device):
    ap=torch.load(ADMITTED_CHECKPOINT,map_location="cpu",weights_only=True)
    base,cfg,_,bsha=_load_base_checkpoint(BASE_CHECKPOINT)
    if sha(ADMITTED_CHECKPOINT)!=EXPECTED_ADMITTED_SHA256 or bsha!=EXPECTED_BASE_SHA256 or digest(clone(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("provenance mismatch")
    admitted=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    admitted.memory.load_state_dict(ap["memory"],strict=True); admitted.recall.load_state_dict(ap["recall"],strict=True)
    admitted=admitted.to(device); admitted.eval()
    for p in admitted.parameters(): p.requires_grad_(False)
    model=OptionIntentExecutiveInfant.grow_from_admitted(admitted).to(device); model.train(True); model.admitted.eval(); model.base.eval()
    return model,cfg,clone(admitted),clone(base),model.trainable_parameters()

@torch.no_grad()
def snapshots(model,cfg,device,cid,cue):
    env=CanonicalExecutiveRuleWorld(cid,cue=cue,rule="invert_lr"); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device); prev=torch.zeros(1,dtype=torch.long,device=device); rows=[]
    while not env.done:
        im=_image_tensor(env,cfg.image_size,device); ctx,_=_context_tensor(obs,device,cfg)
        if env.instruction_visible():
            hist,recall,_,ws,base_logits=model.components(im,ctx,mem,prev,h)
            rows.append({"H":hist["historical_workspace"].detach(),"D":recall.detach(),"B":base_logits.detach(),"response":bool(env.at_response_step())})
        if env.at_response_step(): break
        teacher=int(env.teacher_action()); o=model.admitted.step(im,ctx,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher); h=o["hidden"]; mem=o["memory_next"]; prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
    return rows

def update_loss(model,cfg,device,cid):
    S={c:snapshots(model,cfg,device,cid,c) for c in ("left","right")}
    own=[]; donor=[]; gate=[]; corr={}; dmax=0.0
    for c in ("left","right"):
        for s in S[c]:
            _,gl,_=model.executive.compute_gate(s["H"],s["B"])
            tgt=torch.tensor([[1.0 if s["response"] else 0.0]],device=device)
            gate.append(F.binary_cross_entropy_with_logits(gl,tgt))
    for c in ("left","right"):
        dc="right" if c=="left" else "left"; rs=S[c][-1]; donorD=S[dc][-1]["D"]
        for rid,r in enumerate(RULES,1):
            rt=torch.tensor([rid],dtype=torch.long,device=device)
            logits,delta,_=model.compose(rs["H"],rs["D"],rt,rs["B"])
            y=target_action(c,r); own.append(F.cross_entropy(logits,torch.tensor([y],dtype=torch.long,device=device)))
            corr[(c,r)]=int(torch.argmax(logits,dim=-1)[0])==y; dmax=max(dmax,float(delta.detach().abs().max().cpu()))
            dlogits,_,_=model.compose(rs["H"],donorD,rt,rs["B"])
            dy=target_action(dc,r); donor.append(F.cross_entropy(dlogits,torch.tensor([dy],dtype=torch.long,device=device)))
    own_l=torch.stack(own).mean(); donor_l=torch.stack(donor).mean(); gate_l=torch.stack(gate).mean()
    return .55*own_l+.25*donor_l+.20*gate_l,own_l,donor_l,gate_l,corr,dmax

def save(path,model,opt,u):
    torch.save({"format":FORMAT,"global_update":u,"source_admitted_checkpoint_sha256":EXPECTED_ADMITTED_SHA256,
                "historical_base_sha256":EXPECTED_BASE_SHA256,"historical_base_digest":EXPECTED_BASE_DIGEST,
                "executive":{k:v.detach().cpu() for k,v in model.executive.state_dict().items()},"optimizer":opt.portable_state()},path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state-dir",default="state/option_intent_executive_v1"); ap.add_argument("--target-update",type=int,required=True); ap.add_argument("--lr",type=float,required=True); a=ap.parse_args()
    state=Path(a.state_dir); state.mkdir(parents=True,exist_ok=True); latest=state/"latest.portable.pt"; log=state/"training.jsonl"
    torch.manual_seed(INIT_SEED); device=torch_directml.device(); model,cfg,aref,bref,params=load(device); opt=DMLSafeAdam(params,a.lr); completed=0
    if latest.exists():
        p=torch.load(latest,map_location="cpu",weights_only=True); model.executive.load_state_dict(p["executive"],strict=True); opt.load_portable_state(p["optimizer"]); completed=int(p["global_update"])
    opt.lr=float(a.lr)
    print("OPTION_INTENT_EXECUTIVE_TRAINING_V1"); print("device:",device); print("executive_parameter_count:",sum(p.numel() for p in params))
    print("start_global_update:",completed+1); print("target_global_update:",a.target_update); print("stage_lr:",a.lr)
    print("own_response_weight: 0.55"); print("donor_response_weight: 0.25"); print("timing_gate_weight: 0.20")
    print("v3_dev_touched: false"); print("v3_test_touched: false")
    started=time.time()
    with log.open("a" if log.exists() else "w") as f:
        for u in range(completed+1,a.target_update+1):
            z=u-1; epoch=z//EPOCH_SIZE; cid=schedule(epoch)[z%EPOCH_SIZE]
            opt.zero_grad(); loss,own,donor,gate,corr,dmax=update_loss(model,cfg,device,cid); loss.backward(); opt.step()
            if u%25==0 or u==completed+1 or u==a.target_update:
                pairs={r:int(corr[("left",r)] and corr[("right",r)]) for r in RULES}
                print(f"update={u:05d}/{a.target_update:05d} epoch={epoch+1} condition={cid:04d} loss={float(loss):.6f} own={float(own):.6f} donor={float(donor):.6f} gate={float(gate):.6f} invert={pairs['invert_lr']} vertical={pairs['vertical']} vreverse={pairs['vertical_reverse']} six={int(all(corr.values()))} intent={dmax:.5f}")
            f.write(json.dumps({"global_update":u,"condition_id":cid,"loss":float(loss.detach().cpu())})+"\n"); f.flush()
            if u%100==0 or u==a.target_update:
                assert_exact(aref,model.admitted,"admitted"); assert_exact(bref,model.base,"base")
                ck=state/f"option_intent_u{u:06d}.portable.pt"; save(ck,model,opt,u); latest.write_bytes(ck.read_bytes()); print("saved:",ck)
    print("elapsed_seconds:",round(time.time()-started,3)); print("admitted_brain_bit_exact: true"); print("historical_base_bit_exact: true")
    print("V3_DEV_TOUCHED: false"); print("V3_TEST_TOUCHED: false"); print("OPTION_INTENT_EXECUTIVE_TRAINING_COMPLETE")

if __name__=="__main__": main()
