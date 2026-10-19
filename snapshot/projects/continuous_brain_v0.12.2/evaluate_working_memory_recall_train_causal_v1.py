#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import torch, torch_directml

from continuous_brain.evaluate_competitive_infant import _context_tensor, _image_tensor, _load_base_checkpoint
from continuous_brain.varied_cued_detour_world import ACTION_NAMES, LEFT, RIGHT, TRAIN_CONDITION_RANGE, VariedCuedDetourWorld
from continuous_brain.working_memory_recall_arms_v2 import RecallBridgeWorkingMemoryInfant

BASE_CHECKPOINT="state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256="086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST="8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_FORMAT="working_memory_recall_training_v2"
EXPECTED_UPDATE=1600
BOOTSTRAP_DRAWS=20000
BOOTSTRAP_SEED=55127

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024), b""): h.update(c)
    return h.hexdigest()

def clone_state(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def state_digest(state):
    h=hashlib.sha256()
    for k in sorted(state):
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def binom_upper_tail(k,n,p=.5):
    if n<=0: return 1.0
    return min(1.0,sum(math.comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(k,n+1)))

def bootstrap_mean_ci(values):
    rng=random.Random(BOOTSTRAP_SEED); n=len(values); ms=[]
    for _ in range(BOOTSTRAP_DRAWS):
        ms.append(sum(values[rng.randrange(n)] for _ in range(n))/n)
    ms.sort()
    return ms[int(.025*(BOOTSTRAP_DRAWS-1))], ms[int(.975*(BOOTSTRAP_DRAWS-1))]

def build(checkpoint,arm,device):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload.get("format")!=EXPECTED_FORMAT: raise RuntimeError("checkpoint format mismatch")
    if payload.get("arm")!=arm: raise RuntimeError("checkpoint arm mismatch")
    if int(payload.get("update",-1))!=EXPECTED_UPDATE: raise RuntimeError("checkpoint update mismatch")
    if payload.get("base_checkpoint_sha256")!=EXPECTED_BASE_SHA256: raise RuntimeError("checkpoint base SHA mismatch")
    if payload.get("base_digest")!=EXPECTED_BASE_DIGEST: raise RuntimeError("checkpoint base digest mismatch")
    if payload.get("train_condition_range") != [0,1600]: raise RuntimeError("checkpoint train range mismatch")
    base,cfg,_,sha=_load_base_checkpoint(BASE_CHECKPOINT)
    if sha!=EXPECTED_BASE_SHA256: raise RuntimeError("loaded base SHA mismatch")
    digest=state_digest(clone_state(base))
    if digest!=EXPECTED_BASE_DIGEST: raise RuntimeError("loaded base digest mismatch")
    model=RecallBridgeWorkingMemoryInfant.grow_from(base,cfg,arm=arm)
    model.memory.load_state_dict(payload["memory"],strict=True)
    model.recall.load_state_dict(payload["recall"],strict=True)
    model=model.to(device); model.eval(); model.base.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,cfg,payload,digest

@torch.no_grad()
def snapshot(model,cfg,device,cid,cue):
    env=VariedCuedDetourWorld(cid,cue=cue,max_episode_steps=160); obs=env.reset()
    hidden=model.initial_state(1,device); memory=model.initial_memory(1,device)
    prev=torch.zeros(1,dtype=torch.long,device=device)
    while not env.done:
        teacher=int(env.teacher_action())
        image=_image_tensor(env,cfg.image_size,device); context,_=_context_tensor(obs,device,cfg)
        if teacher in (LEFT,RIGHT):
            if env.steps < env.cue_steps: raise AssertionError("decision while cue visible")
            return dict(condition_id=int(cid), cue=cue, teacher=teacher, decision_step=int(env.steps),
                        realized_hidden_delay=int(env.steps-env.cue_steps), image=image.detach().clone(),
                        context=context.detach().clone(), prev_action=prev.detach().clone(),
                        hidden=hidden.detach().clone(), memory=memory.detach().clone())
        out=model.step(image,context,memory,prev,hidden,use_memory=True,use_recall=True)
        obs,_,_,_=env.step(teacher)
        prev=torch.tensor([teacher+1],dtype=torch.long,device=device)
        hidden=out["hidden"]; memory=out["memory_next"]
    raise RuntimeError("episode ended before decision")

def assert_pair(left,right):
    for f in ("image","context","prev_action"):
        if not torch.equal(left[f].detach().cpu(),right[f].detach().cpu()): raise AssertionError(f"pair {f} differs")
    if left["decision_step"]!=right["decision_step"] or left["realized_hidden_delay"]!=right["realized_hidden_delay"]:
        raise AssertionError("pair timing differs")

@torch.no_grad()
def forward_i(model,snap,hidden,hist_mem,rec_mem,enable_recall=True):
    hist=model.historical_step(snap["image"],snap["context"],hist_mem,snap["prev_action"],hidden)
    if enable_recall:
        residual,aux=model.recall(hist["hidden"],rec_mem)
    else:
        residual=torch.zeros_like(hist["historical_workspace"]); aux={"recall_multiplier":None}
    workspace=hist["historical_workspace"]+residual
    logits=model.base.motor(workspace)[0].detach().float().cpu()
    return logits, float(torch.linalg.vector_norm(residual.detach().float().cpu()).item()), (
        None if aux.get("recall_multiplier") is None else float(aux["recall_multiplier"].detach().mean().cpu())
    )

def classify(result,teacher):
    logits,res_l2,mult=result
    full=int(torch.argmax(logits).item()); ll=float(logits[LEFT]); rl=float(logits[RIGHT]); d=rl-ll
    side=RIGHT if d>0 else LEFT
    return dict(full_pred=full,full_pred_name=ACTION_NAMES[full],full_correct=bool(full==teacher),
                lateral_top_choice=bool(full in (LEFT,RIGHT)),side_pred=int(side),side_pred_name=ACTION_NAMES[side],
                side_correct=bool(side==teacher),right_minus_left=float(d),
                correct_side_margin=float(d if teacher==RIGHT else -d),
                recall_residual_l2=res_l2,recall_multiplier_mean=mult)

def eval_condition(model,cfg,device,cid):
    L=snapshot(model,cfg,device,cid,"left"); R=snapshot(model,cfg,device,cid,"right"); assert_pair(L,R)
    Zm=torch.zeros_like(L["memory"]); Zh=model.initial_state(1,device); modes={}
    def pm(name,lh,lm_hist,lm_rec,rh,rm_hist,rm_rec,enable=True):
        lo=forward_i(model,L,lh,lm_hist,lm_rec,enable); ro=forward_i(model,R,rh,rm_hist,rm_rec,enable)
        modes[name]={"left":classify(lo,LEFT),"right":classify(ro,RIGHT),
                     "pair_logits_max_abs_diff":float((lo[0]-ro[0]).abs().max().item())}
    pm("normal",L["hidden"],L["memory"],L["memory"],R["hidden"],R["memory"],R["memory"])
    pm("direct_recall_off",L["hidden"],L["memory"],Zm,R["hidden"],R["memory"],Zm,False)
    pm("direct_recall_swapped",L["hidden"],L["memory"],R["memory"],R["hidden"],R["memory"],L["memory"])
    pm("total_memory_swapped",L["hidden"],R["memory"],R["memory"],R["hidden"],L["memory"],L["memory"])
    pm("hidden_reset_historical_only",Zh,L["memory"],Zm,Zh,R["memory"],Zm,False)
    pm("hidden_reset_direct_only",Zh,Zm,L["memory"],Zh,Zm,R["memory"])
    pm("hidden_reset_direct_swapped",Zh,Zm,R["memory"],Zh,Zm,L["memory"])
    pm("hidden_reset_both_memory_paths",Zh,L["memory"],L["memory"],Zh,R["memory"],R["memory"])
    pm("hidden_reset_all_zero",Zh,Zm,Zm,Zh,Zm,Zm,False)
    if modes["hidden_reset_all_zero"]["pair_logits_max_abs_diff"]>1e-6: raise AssertionError("all-zero pair diverged")
    ndL=modes["normal"]["left"]["right_minus_left"]; ndR=modes["normal"]["right"]["right_minus_left"]
    sdL=modes["direct_recall_swapped"]["left"]["right_minus_left"]; sdR=modes["direct_recall_swapped"]["right"]["right_minus_left"]
    doL=modes["hidden_reset_direct_only"]["left"]["right_minus_left"]; doR=modes["hidden_reset_direct_only"]["right"]["right_minus_left"]
    dsL=modes["hidden_reset_direct_swapped"]["left"]["right_minus_left"]; dsR=modes["hidden_reset_direct_swapped"]["right"]["right_minus_left"]
    eff=.5*((sdL-ndL)+(ndR-sdR))
    return {"condition_id":int(cid),"realized_hidden_delay":int(L["realized_hidden_delay"]),
            "memory_pair_l2_distance":float(torch.linalg.vector_norm((L["memory"]-R["memory"]).detach().float().cpu()).item()),
            "modes":modes,
            "direct_swap":{"condition_effect":float(eff),"positive":bool(eff>0)},
            "direct_only_memory_code":{"own_pair_side_correct":bool(doL<0 and doR>0),
                                       "swapped_pair_donor_consistent":bool(dsL>0 and dsR<0)}}

def summarize_mode(rows,mode):
    LF=[bool(r["modes"][mode]["left"]["full_correct"]) for r in rows]
    RF=[bool(r["modes"][mode]["right"]["full_correct"]) for r in rows]
    LL=[bool(r["modes"][mode]["left"]["lateral_top_choice"]) for r in rows]
    RL=[bool(r["modes"][mode]["right"]["lateral_top_choice"]) for r in rows]
    LS=[bool(r["modes"][mode]["left"]["side_correct"]) for r in rows]
    RS=[bool(r["modes"][mode]["right"]["side_correct"]) for r in rows]
    PF=[a and b for a,b in zip(LF,RF)]; PL=[a and b for a,b in zip(LL,RL)]; PS=[a and b for a,b in zip(LS,RS)]
    return {"n":len(rows),"left_full":mean([float(x) for x in LF]),"right_full":mean([float(x) for x in RF]),
            "pair_full":mean([float(x) for x in PF]),"left_lateral_top":mean([float(x) for x in LL]),
            "right_lateral_top":mean([float(x) for x in RL]),"pair_lateral_top":mean([float(x) for x in PL]),
            "left_side":mean([float(x) for x in LS]),"right_side":mean([float(x) for x in RS]),
            "pair_side":mean([float(x) for x in PS]),"_pair_full_bits":PF}

def summarize(rows):
    modes={k:summarize_mode(rows,k) for k in rows[0]["modes"]}
    vals=[float(r["direct_swap"]["condition_effect"]) for r in rows]
    pos=sum(v>0 for v in vals); nz=sum(v!=0 for v in vals); ci=bootstrap_mean_ci(vals)
    normal=modes["normal"]; off=modes["direct_recall_off"]
    n_only=sum(a and not b for a,b in zip(normal["_pair_full_bits"],off["_pair_full_bits"]))
    o_only=sum(b and not a for a,b in zip(normal["_pair_full_bits"],off["_pair_full_bits"]))
    disc=n_only+o_only
    clean_modes={k:{kk:vv for kk,vv in v.items() if not kk.startswith("_")} for k,v in modes.items()}
    return {"modes":clean_modes,
            "direct_recall_added_value":{"pair_full_gain_vs_off":normal["pair_full"]-off["pair_full"],
                                         "pair_side_gain_vs_off":normal["pair_side"]-off["pair_side"],
                                         "normal_only_pair_full":n_only,"off_only_pair_full":o_only,
                                         "discordant_pair_full":disc,"p_one_sided_pair_full":binom_upper_tail(n_only,disc)},
            "direct_swap":{"mean_condition_effect":mean(vals),"bootstrap95":list(ci),
                           "positive_conditions":pos,"positive_fraction":pos/len(rows),
                           "sign_p_one_sided":binom_upper_tail(pos,nz)},
            "direct_only_memory_code":{"own_pair_side_correct":mean([float(r["direct_only_memory_code"]["own_pair_side_correct"]) for r in rows]),
                                       "swapped_pair_donor_consistent":mean([float(r["direct_only_memory_code"]["swapped_pair_donor_consistent"]) for r in rows])},
            "memory_pair_l2_mean":mean([float(r["memory_pair_l2_distance"]) for r in rows])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",required=True,choices=("linear","gated"))
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    device=torch_directml.device(); model,cfg,payload,digest=build(Path(args.checkpoint),args.arm,device); base_ref=clone_state(model.base)
    print("WORKING_MEMORY_RECALL_TRAIN_CAUSAL_EVAL_V1"); print("arm:",args.arm); print("device:",device)
    print("checkpoint:",args.checkpoint); print("checkpoint_sha256:",sha256_file(args.checkpoint)); print("checkpoint_update:",payload["update"])
    print("base_digest:",digest); print("train_condition_range: 0:1600"); print("dev_touched: false"); print("test_touched: false")
    rows=[]; total=len(TRAIN_CONDITION_RANGE)
    for i,cid in enumerate(TRAIN_CONDITION_RANGE,1):
        rows.append(eval_condition(model,cfg,device,int(cid)))
        if i==1 or i%100==0 or i==total:
            s=summarize_mode(rows,"normal")
            print(f"evaluated={i}/{total} pair_full={s['pair_full']:.4f} Lfull={s['left_full']:.4f} Rfull={s['right_full']:.4f} pair_side={s['pair_side']:.4f}")
    if state_digest(clone_state(model.base))!=EXPECTED_BASE_DIGEST: raise AssertionError("historical base changed")
    for k,v in base_ref.items():
        if not torch.equal(v,model.base.state_dict()[k].detach().cpu()): raise AssertionError(f"base tensor changed {k}")
    overall=summarize(rows); groups=collections.defaultdict(list)
    for r in rows: groups[int(r["realized_hidden_delay"])].append(r)
    by_delay={str(d):summarize(g) for d,g in sorted(groups.items())}
    result={"format":"working_memory_recall_train_causal_eval_v1","arm":args.arm,"checkpoint":args.checkpoint,
            "checkpoint_sha256":sha256_file(args.checkpoint),"checkpoint_update":int(payload["update"]),
            "base_checkpoint_sha256":payload["base_checkpoint_sha256"],"base_digest":EXPECTED_BASE_DIGEST,
            "split":"train","condition_range":[0,1600],"conditions":len(rows),"dev_touched":False,"test_touched":False,
            "overall":overall,"by_realized_hidden_delay":by_delay,"per_condition":rows}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True))
    n=overall["modes"]["normal"]; off=overall["modes"]["direct_recall_off"]; add=overall["direct_recall_added_value"]; sw=overall["direct_swap"]; code=overall["direct_only_memory_code"]
    print()
    print(f"normal: pair_full={n['pair_full']:.4f} Lfull={n['left_full']:.4f} Rfull={n['right_full']:.4f} pair_lat={n['pair_lateral_top']:.4f} pair_side={n['pair_side']:.4f} Lside={n['left_side']:.4f} Rside={n['right_side']:.4f}")
    print(f"direct_recall_off: pair_full={off['pair_full']:.4f} pair_side={off['pair_side']:.4f}")
    print(f"direct_recall_added_value: pair_full_gain={add['pair_full_gain_vs_off']:+.4f} pair_side_gain={add['pair_side_gain_vs_off']:+.4f} p_full={add['p_one_sided_pair_full']:.8g}")
    print(f"direct_swap: mean={sw['mean_condition_effect']:+.6f} CI95=[{sw['bootstrap95'][0]:+.6f},{sw['bootstrap95'][1]:+.6f}] positive={sw['positive_conditions']}/{len(rows)} p={sw['sign_p_one_sided']:.8g}")
    print(f"direct_only_memory_code: own_pair_side={code['own_pair_side_correct']:.4f} swapped_donor_pair={code['swapped_pair_donor_consistent']:.4f}")
    print()
    print("REALIZED_DELAY_CAUSAL_SUMMARY")
    print(f"{'delay':>5} {'normalF':>8} {'normalS':>8} {'offF':>8} {'swap+':>8} {'directOwnS':>11} {'swapDonor':>10}")
    for d,s in sorted(((int(k),v) for k,v in by_delay.items())):
        print(f"{d:5d} {s['modes']['normal']['pair_full']:8.4f} {s['modes']['normal']['pair_side']:8.4f} {s['modes']['direct_recall_off']['pair_full']:8.4f} {s['direct_swap']['positive_fraction']:8.4f} {s['direct_only_memory_code']['own_pair_side_correct']:11.4f} {s['direct_only_memory_code']['swapped_pair_donor_consistent']:10.4f}")
    print("historical_base_bit_exact: true"); print("wrote:",args.output); print("TRAIN_ONLY: true"); print("DEV_TOUCHED: false"); print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_RECALL_TRAIN_CAUSAL_EVAL_COMPLETE")

if __name__=="__main__": main()
