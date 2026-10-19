#!/usr/bin/env python3
"""
Frozen capacity-ungated Working Memory DEV causal evaluator v2.

Uses ONLY frozen DEV IDs 1600..1979.
Never accesses TEST IDs 1980..2359.
No training or optimizer step occurs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
import torch, torch_directml

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor, _image_tensor, _load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES, DEV_CONDITION_RANGE, LEFT, RIGHT,
    VariedCuedDetourWorld, generate_task_spec,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_FORMAT="working_memory_recall_capacity_control_training_v1"
EXPECTED_UPDATE=1600
BOOTSTRAP_DRAWS=20000
BOOTSTRAP_SEED=67117

def clone_state(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""): h.update(c)
    return h.hexdigest()

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def binom_upper(k,n,p=.5):
    if n<=0: return 1.0
    return min(1.0,sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1)))

def bootstrap_ci(vals):
    rng=random.Random(BOOTSTRAP_SEED); n=len(vals); ms=[]
    for _ in range(BOOTSTRAP_DRAWS):
        ms.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    ms.sort()
    return [ms[int(.025*(BOOTSTRAP_DRAWS-1))],
            ms[int(.975*(BOOTSTRAP_DRAWS-1))]]

def build(path,device):
    p=torch.load(path,map_location="cpu",weights_only=True)
    if p.get("format")!=EXPECTED_FORMAT: raise RuntimeError("checkpoint format mismatch")
    if p.get("arm")!="capacity_ungated": raise RuntimeError("checkpoint arm mismatch")
    if int(p.get("update",-1))!=EXPECTED_UPDATE: raise RuntimeError("checkpoint update mismatch")
    if p.get("base_checkpoint_sha256")!=EXPECTED_BASE_SHA256: raise RuntimeError("checkpoint base SHA mismatch")
    if p.get("base_digest")!=EXPECTED_BASE_DIGEST: raise RuntimeError("checkpoint base digest mismatch")
    if p.get("train_condition_range")!=[0,1600]: raise RuntimeError("train range metadata mismatch")
    base,cfg,_,sha=_load_base_checkpoint(BASE_CHECKPOINT)
    if sha!=EXPECTED_BASE_SHA256: raise RuntimeError("loaded base SHA mismatch")
    if digest(clone_state(base))!=EXPECTED_BASE_DIGEST: raise RuntimeError("loaded base digest mismatch")
    m=CapacityMatchedRecallWorkingMemoryInfant.grow_from(base,cfg)
    m.memory.load_state_dict(p["memory"],strict=True)
    m.recall.load_state_dict(p["recall"],strict=True)
    m=m.to(device); m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m,cfg,p

@torch.no_grad()
def snapshot(model,cfg,device,cid,cue):
    env=VariedCuedDetourWorld(cid,cue=cue,max_episode_steps=160); obs=env.reset()
    h=model.initial_state(1,device); mem=model.initial_memory(1,device)
    prev=torch.zeros(1,dtype=torch.long,device=device)
    while not env.done:
        teacher=int(env.teacher_action())
        image=_image_tensor(env,cfg.image_size,device)
        context,_=_context_tensor(obs,device,cfg)
        if teacher in (LEFT,RIGHT):
            if env.steps < env.cue_steps: raise AssertionError("decision while cue visible")
            return dict(
                teacher=teacher, decision_step=int(env.steps),
                delay=int(env.steps-env.cue_steps),
                image=image.detach().clone(), context=context.detach().clone(),
                prev=prev.detach().clone(), hidden=h.detach().clone(),
                memory=mem.detach().clone(),
            )
        out=model.step(image,context,mem,prev,h,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher)
        prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
        h=out["hidden"]; mem=out["memory_next"]
    raise RuntimeError("no delayed decision")

def assert_pair(L,R):
    for k in ("image","context","prev"):
        if not torch.equal(L[k].cpu(),R[k].cpu()): raise AssertionError(f"pair {k} mismatch")
    if L["decision_step"]!=R["decision_step"] or L["delay"]!=R["delay"]:
        raise AssertionError("pair timing mismatch")

@torch.no_grad()
def forward(model,snap,h,hist_mem,rec_mem,recall_on):
    x=model.historical_step(snap["image"],snap["context"],hist_mem,snap["prev"],h)
    if recall_on:
        residual,_=model.recall(x["hidden"],rec_mem)
    else:
        residual=torch.zeros_like(x["historical_workspace"])
    ws=x["historical_workspace"]+residual
    return model.base.motor(ws)[0].detach().float().cpu()

def cls(logits,teacher):
    full=int(torch.argmax(logits)); d=float(logits[RIGHT]-logits[LEFT])
    side=RIGHT if d>0 else LEFT
    return dict(
        full_pred=full, full_pred_name=ACTION_NAMES[full],
        full_correct=bool(full==teacher),
        lateral_top=bool(full in (LEFT,RIGHT)),
        side_correct=bool(side==teacher),
        right_minus_left=d,
        correct_side_margin=float(d if teacher==RIGHT else -d),
    )

def eval_condition(model,cfg,device,cid):
    L=snapshot(model,cfg,device,cid,"left"); R=snapshot(model,cfg,device,cid,"right")
    assert_pair(L,R)
    Zm=torch.zeros_like(L["memory"]); Zh=model.initial_state(1,device)
    modes={}
    def pm(name,lh,lhi,lrec,rh,rhi,rrec,ron):
        ll=forward(model,L,lh,lhi,lrec,ron); rr=forward(model,R,rh,rhi,rrec,ron)
        modes[name]={
            "left":cls(ll,LEFT),"right":cls(rr,RIGHT),
            "pair_logits_max_abs_diff":float((ll-rr).abs().max()),
        }
    pm("normal",L["hidden"],L["memory"],L["memory"],
       R["hidden"],R["memory"],R["memory"],True)
    pm("direct_recall_off",L["hidden"],L["memory"],Zm,
       R["hidden"],R["memory"],Zm,False)
    pm("same_hidden_direct_only",L["hidden"],Zm,L["memory"],
       R["hidden"],Zm,R["memory"],True)
    pm("same_hidden_historical_only",L["hidden"],L["memory"],Zm,
       R["hidden"],R["memory"],Zm,False)
    pm("direct_recall_swapped",L["hidden"],L["memory"],R["memory"],
       R["hidden"],R["memory"],L["memory"],True)
    pm("hidden_reset_direct_only",Zh,Zm,L["memory"],Zh,Zm,R["memory"],True)
    pm("hidden_reset_direct_swapped",Zh,Zm,R["memory"],Zh,Zm,L["memory"],True)
    pm("hidden_reset_all_zero",Zh,Zm,Zm,Zh,Zm,Zm,False)
    if modes["hidden_reset_all_zero"]["pair_logits_max_abs_diff"]>1e-6:
        raise AssertionError("all-zero pair diverged")

    ndL=modes["normal"]["left"]["right_minus_left"]
    ndR=modes["normal"]["right"]["right_minus_left"]
    swL=modes["direct_recall_swapped"]["left"]["right_minus_left"]
    swR=modes["direct_recall_swapped"]["right"]["right_minus_left"]
    doL=modes["hidden_reset_direct_only"]["left"]["right_minus_left"]
    doR=modes["hidden_reset_direct_only"]["right"]["right_minus_left"]
    dsL=modes["hidden_reset_direct_swapped"]["left"]["right_minus_left"]
    dsR=modes["hidden_reset_direct_swapped"]["right"]["right_minus_left"]
    spec=generate_task_spec(cid)
    factors={
        "corridor_length":float(spec.corridor_length),
        "goal_x_offset":float(spec.goal_x_offset),
        "goal_y_offset":float(spec.goal_y_offset),
        "cue_steps":int(spec.cue_steps),
        "move_step":float(spec.move_step),
        "start_x":float(spec.start_x),
        "start_y":float(spec.start_y),
        "distractor_pattern":int(spec.distractor_pattern),
    }
    return {
        "condition_id":int(cid),"realized_hidden_delay":int(L["delay"]),
        "factors":factors,"modes":modes,
        "direct_swap":{
            "condition_effect":float(.5*((swL-ndL)+(ndR-swR))),
        },
        "direct_only_memory_code":{
            "own_pair_side_correct":bool(doL<0 and doR>0),
            "swapped_pair_donor_consistent":bool(dsL>0 and dsR<0),
        },
    }

def mode_stats(rows,mode):
    lf=[r["modes"][mode]["left"]["full_correct"] for r in rows]
    rf=[r["modes"][mode]["right"]["full_correct"] for r in rows]
    ls=[r["modes"][mode]["left"]["side_correct"] for r in rows]
    rs=[r["modes"][mode]["right"]["side_correct"] for r in rows]
    pf=[a and b for a,b in zip(lf,rf)]; ps=[a and b for a,b in zip(ls,rs)]
    return {
        "n":len(rows),"left_full":mean([float(x) for x in lf]),
        "right_full":mean([float(x) for x in rf]),
        "pair_full":mean([float(x) for x in pf]),
        "left_side":mean([float(x) for x in ls]),
        "right_side":mean([float(x) for x in rs]),
        "pair_side":mean([float(x) for x in ps]),
        "_pair_full_bits":pf,
    }

def summarize(rows):
    modes={k:mode_stats(rows,k) for k in rows[0]["modes"]}
    vals=[float(r["direct_swap"]["condition_effect"]) for r in rows]
    pos=sum(v>0 for v in vals); nz=sum(v!=0 for v in vals)
    normal=modes["normal"]; off=modes["direct_recall_off"]
    n_only=sum(a and not b for a,b in zip(normal["_pair_full_bits"],off["_pair_full_bits"]))
    o_only=sum(b and not a for a,b in zip(normal["_pair_full_bits"],off["_pair_full_bits"]))
    disc=n_only+o_only
    clean={k:{kk:vv for kk,vv in v.items() if not kk.startswith("_")} for k,v in modes.items()}
    return {
        "modes":clean,
        "direct_recall_added_value":{
            "pair_full_gain_vs_off":normal["pair_full"]-off["pair_full"],
            "normal_only_pair_full":n_only,"off_only_pair_full":o_only,
            "discordant":disc,"p_one_sided":binom_upper(n_only,disc),
        },
        "direct_swap":{
            "mean":mean(vals),"bootstrap95":bootstrap_ci(vals),
            "positive_fraction":pos/len(rows),
            "positive_conditions":pos,"sign_p_one_sided":binom_upper(pos,nz),
        },
        "hidden_reset_direct":{
            "own_pair_side":mean([float(r["direct_only_memory_code"]["own_pair_side_correct"]) for r in rows]),
            "swapped_donor_pair":mean([float(r["direct_only_memory_code"]["swapped_pair_donor_consistent"]) for r in rows]),
        },
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    device=torch_directml.device()
    model,cfg,payload=build(Path(a.checkpoint),device)
    base_ref=clone_state(model.base)
    print("WORKING_MEMORY_CAPACITY_DEV_CAUSAL_V2")
    print("device:",device); print("checkpoint:",a.checkpoint)
    print("checkpoint_sha256:",file_sha(a.checkpoint))
    print("checkpoint_update:",payload["update"])
    print("dev_condition_range: 1600:1980")
    print("test_touched: false")
    rows=[]; total=len(DEV_CONDITION_RANGE)
    for i,cid in enumerate(DEV_CONDITION_RANGE,1):
        rows.append(eval_condition(model,cfg,device,int(cid)))
        if i==1 or i%50==0 or i==total:
            s=mode_stats(rows,"normal")
            print(f"evaluated={i}/{total} pairF={s['pair_full']:.4f} L={s['left_full']:.4f} R={s['right_full']:.4f} pairS={s['pair_side']:.4f}")
    if digest(clone_state(model.base))!=EXPECTED_BASE_DIGEST:
        raise AssertionError("historical base changed")
    for k,v in base_ref.items():
        if not torch.equal(v,model.base.state_dict()[k].detach().cpu()):
            raise AssertionError(f"base tensor changed {k}")
    overall=summarize(rows)
    result={
        "format":"working_memory_capacity_dev_causal_v2",
        "split":"dev","condition_range":[1600,1980],"conditions":len(rows),
        "checkpoint":a.checkpoint,"checkpoint_sha256":file_sha(a.checkpoint),
        "overall":overall,"per_condition":rows,
        "test_touched":False,
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    n=overall["modes"]["normal"]; d=overall["direct_recall_added_value"]
    sw=overall["direct_swap"]; hr=overall["hidden_reset_direct"]
    print()
    print(f"normal: pairF={n['pair_full']:.4f} L={n['left_full']:.4f} R={n['right_full']:.4f} pairS={n['pair_side']:.4f}")
    print(f"direct_recall_off_pairF={overall['modes']['direct_recall_off']['pair_full']:.4f}")
    print(f"same_hidden_direct_pairS={overall['modes']['same_hidden_direct_only']['pair_side']:.4f}")
    print(f"recall_gain={d['pair_full_gain_vs_off']:+.4f} p={d['p_one_sided']:.8g}")
    print(f"swap_mean={sw['mean']:+.6f} CI95=[{sw['bootstrap95'][0]:+.6f},{sw['bootstrap95'][1]:+.6f}] positive={sw['positive_fraction']:.4f} p={sw['sign_p_one_sided']:.8g}")
    print(f"hidden_reset_direct_own={hr['own_pair_side']:.4f} swapped_donor={hr['swapped_donor_pair']:.4f}")
    print("historical_base_bit_exact: true")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_CAPACITY_DEV_CAUSAL_COMPLETE")

if __name__=="__main__": main()
