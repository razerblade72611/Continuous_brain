#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import torch
from torch.nn import functional as F

from continuous_brain.additive_goal_options_v1 import (
    GOAL_DECODER_PATH, AdditiveGoalOptionBank,
    load_goal_options, save_goal_options, sha256,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT, MEMORY_CHECKPOINT,
    assert_pair_identical, build_substrate,
    collect_decision_snapshot, load_goal_decoder,
)
from continuous_brain.varied_cued_detour_world import (
    VariedCuedDetourWorld, LEFT, RIGHT,
)
from continuous_brain.evaluate_competitive_infant import (
    _context_tensor, _image_tensor,
)
from continuous_brain.planning_v103 import (
    _active_option_should_terminate, _option_policy_action,
)

TRAIN_IDS=range(0,1024)
DEV_IDS=range(1728,1856)
MAX_STEPS=160
DIRECT_MIN_COMMIT=3
DIRECT_TERMINATE_THRESHOLD=.50
SKILL_MIN_COMMIT=1
SKILL_TERMINATE_THRESHOLD=.50

EPOCHS=100
BATCH_SIZE=512
LR=1e-3
WEIGHT_DECAY=1e-5
SEED=22001

MIN_GOAL_ACC=.95
MIN_OWN_COMMIT=.95
MIN_PAIR_COMMIT=.90
MIN_DONOR_COMMIT=.95
MIN_FLIP_COMMIT=.95
MIN_TERM_RATE=.90
MIN_TERM_WITHIN2=.85
MIN_ADVANTAGE=.25
MAX_BASELINE=.70

def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0

def balanced_accuracy(y,p):
    out=[]
    for c in (0,1):
        idx=[i for i,v in enumerate(y) if int(v)==c]
        if idx: out.append(sum(int(int(p[i])==c) for i in idx)/len(idx))
    return mean(out)

@torch.no_grad()
def root_from_env(base,cfg,env,obs):
    image=_image_tensor(env,cfg.image_size,"cpu")
    _,ctx=_context_tensor(obs,"cpu",cfg)
    spatial=base.target_spatial_bus(image)
    return base.planning.encode_state(spatial,ctx)

@torch.no_grad()
def collect_dataset(base,cfg):
    lat=[]; goals=[]; actions=[]; terms=[]
    episodes=0
    for n,cid in enumerate(TRAIN_IDS,1):
        for cue in ("left","right"):
            env=VariedCuedDetourWorld(
                cid,cue=cue,max_episode_steps=MAX_STEPS,
                image_size_hint=int(cfg.image_size)
            )
            obs=env.reset()
            goal=0 if cue=="left" else 1
            expected=LEFT if cue=="left" else RIGHT
            started=False; lateral=0
            while not env.done:
                root=root_from_env(base,cfg,env,obs)
                teacher=int(env.teacher_action())
                if not started:
                    if teacher not in (LEFT,RIGHT):
                        obs,_,done,_=env.step(teacher)
                        if done: raise RuntimeError("ended before delayed decision")
                        continue
                    started=True
                if teacher in (LEFT,RIGHT):
                    if teacher!=expected: raise AssertionError("teacher side mismatch")
                    lat.append(root.detach().cpu()); goals.append(goal)
                    actions.append(teacher); terms.append(0.0); lateral+=1
                    obs,_,_,_=env.step(teacher)
                else:
                    lat.append(root.detach().cpu()); goals.append(goal)
                    actions.append(-1); terms.append(1.0)
                    break
            if lateral<1: raise AssertionError("no lateral samples")
            episodes+=1
        if n==1 or n%128==0:
            print(f"skill_dataset_conditions={n}/{len(TRAIN_IDS)}")
    return {
        "latent":torch.cat(lat).float(),
        "goal":torch.tensor(goals,dtype=torch.long),
        "action":torch.tensor(actions,dtype=torch.long),
        "terminate":torch.tensor(terms,dtype=torch.float32),
        "episodes":episodes,
    }

def train_bank(cfg,data):
    torch.manual_seed(SEED); random.seed(SEED)
    bank=AdditiveGoalOptionBank(int(cfg.planning_dim),int(cfg.action_dim))
    opt=torch.optim.AdamW(bank.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    x,g,a,t=data["latent"],data["goal"],data["action"],data["terminate"]
    pos=float((t>.5).sum()); neg=float((t<=.5).sum())
    pos_weight=torch.tensor([max(1.0,neg/max(1.0,pos))])
    gen=torch.Generator().manual_seed(SEED)
    n=x.shape[0]; metrics={}
    for epoch in range(1,EPOCHS+1):
        bank.train()
        order=torch.randperm(n,generator=gen)
        total=0.0; seen=0
        for start in range(0,n,BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]
            xb,gb,ab,tb=x[idx],g[idx],a[idx],t[idx]
            opt.zero_grad(set_to_none=True)
            al=bank.action_logits(xb,gb)
            tl=bank.termination_logit(xb,gb)
            mask=ab>=0
            pl=F.cross_entropy(al[mask],ab[mask])
            tr=F.binary_cross_entropy_with_logits(tl,tb,pos_weight=pos_weight)
            loss=pl+tr
            loss.backward(); opt.step()
            total+=float(loss.item())*len(idx); seen+=len(idx)
        bank.eval()
        with torch.no_grad():
            m=a>=0
            pa=bank.action_logits(x[m],g[m]).argmax(-1)
            pacc=float((pa==a[m]).float().mean())
            tp=(torch.sigmoid(bank.termination_logit(x,g))>=.5).long()
            tbal=balanced_accuracy(t.tolist(),tp.tolist())
        metrics={
            "epoch":epoch,
            "loss":total/max(1,seen),
            "policy_accuracy":pacc,
            "termination_balanced_accuracy":tbal,
            "termination_pos_weight":float(pos_weight.item()),
        }
        if epoch==1 or epoch%10==0 or epoch==EPOCHS:
            print(f"goal_option_epoch={epoch:03d}/{EPOCHS} loss={metrics['loss']:.6f} policy={pacc:.4f} term_bal={tbal:.4f}")
    return bank,metrics

@torch.no_grad()
def teacher_lateral_steps(cfg,cid,cue,prefix):
    env=VariedCuedDetourWorld(
        cid,cue=cue,max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size)
    )
    obs=env.reset()
    for a in prefix:
        obs,_,done,_=env.step(int(a))
        if done: raise RuntimeError("prefix ended episode")
    expected=LEFT if cue=="left" else RIGHT
    n=0
    while not env.done and int(env.teacher_action()) in (LEFT,RIGHT):
        a=int(env.teacher_action())
        if a!=expected: raise AssertionError("teacher side mismatch")
        obs,_,_,_=env.step(a); n+=1
    return n

@torch.no_grad()
def run_skill(base,cfg,bank,cid,actual_cue,prefix,goal_side):
    env=VariedCuedDetourWorld(
        cid,cue=actual_cue,max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size)
    )
    obs=env.reset()
    for a in prefix:
        obs,_,done,_=env.step(int(a))
        if done: raise RuntimeError("prefix ended episode")
    actual_target=LEFT if actual_cue=="left" else RIGHT
    goal_target=LEFT if goal_side==0 else RIGHT

    skill=True; skill_age=0; skill_steps=0; skill_terminated=False
    current_option=None; option_age=0
    first_lateral=None; final={"reached_correct":False}; reward=0.0

    while not env.done:
        root=root_from_env(base,cfg,env,obs)
        if skill:
            gt=torch.tensor([goal_side],dtype=torch.long)
            pterm=float(torch.sigmoid(bank.termination_logit(root,gt))[0])
            if skill_age>=SKILL_MIN_COMMIT and pterm>=SKILL_TERMINATE_THRESHOLD:
                skill=False; skill_terminated=True
                sel=base.options.selector_logits(root)[0]
                current_option=int(torch.argmax(sel)); option_age=0
                continue
            action=int(torch.argmax(bank.action_logits(root,gt),-1)[0])
            skill_age+=1; skill_steps+=1
        else:
            if _active_option_should_terminate(
                base,root,current_option,option_age,
                DIRECT_MIN_COMMIT,DIRECT_TERMINATE_THRESHOLD
            ):
                sel=base.options.selector_logits(root)[0]
                current_option=int(torch.argmax(sel)); option_age=0
            action=int(_option_policy_action(base,root,current_option))
            option_age+=1
        if first_lateral is None and action in (LEFT,RIGHT):
            first_lateral=action
        obs,r,_,final=env.step(action); reward+=float(r)

    return {
        "first_lateral":first_lateral,
        "commitment_follows_goal":first_lateral==goal_target,
        "commitment_correct":first_lateral==actual_target,
        "completed_correct":bool(final.get("reached_correct",False)),
        "skill_steps":skill_steps,
        "skill_terminated":skill_terminated,
        "reward":reward,
    }

@torch.no_grad()
def run_baseline(base,cfg,cid,cue,prefix):
    env=VariedCuedDetourWorld(
        cid,cue=cue,max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size)
    )
    obs=env.reset()
    for a in prefix:
        obs,_,done,_=env.step(int(a))
        if done: raise RuntimeError("prefix ended episode")
    current=None; age=0; first=None; final={"reached_correct":False}
    while not env.done:
        root=root_from_env(base,cfg,env,obs)
        if _active_option_should_terminate(
            base,root,current,age,DIRECT_MIN_COMMIT,DIRECT_TERMINATE_THRESHOLD
        ):
            current=int(torch.argmax(base.options.selector_logits(root)[0])); age=0
        a=int(_option_policy_action(base,root,current)); age+=1
        if first is None and a in (LEFT,RIGHT): first=a
        obs,_,_,final=env.step(a)
    target=LEFT if cue=="left" else RIGHT
    return {"commitment_correct":first==target,"completed_correct":bool(final.get("reached_correct",False))}

def decode_goal(decoder,recall):
    return int(torch.argmax(decoder(recall),-1)[0])

@torch.no_grad()
def evaluate(base,cfg,admitted,decoder,bank):
    own=[]; donor=[]; flip=[]; baseline=[]; pair=[]; goals=[]; term=[]; within2=[]; owncomp=[]; basecomp=[]; details=[]
    for n,cid in enumerate(DEV_IDS,1):
        L=collect_decision_snapshot(base,cfg,admitted,cid,"left")
        R=collect_decision_snapshot(base,cfg,admitted,cid,"right")
        assert_pair_identical(L,R)
        pairmap={"left":L,"right":R}; pair_ok=[]
        for cue in ("left","right"):
            s=pairmap[cue]; d=pairmap["right" if cue=="left" else "left"]
            og=decode_goal(decoder,s["recall"]); dg=decode_goal(decoder,d["recall"])
            expected=0 if cue=="left" else 1
            goals.append(float(og==expected))
            teacher_steps=teacher_lateral_steps(cfg,cid,cue,s["prefix_actions"])
            ro=run_skill(base,cfg,bank,cid,cue,s["prefix_actions"],og)
            rd=run_skill(base,cfg,bank,cid,cue,s["prefix_actions"],dg)
            rf=run_skill(base,cfg,bank,cid,cue,s["prefix_actions"],1-og)
            rb=run_baseline(base,cfg,cid,cue,s["prefix_actions"])
            own.append(float(ro["commitment_correct"]))
            donor.append(float(rd["commitment_follows_goal"]))
            flip.append(float(rf["commitment_follows_goal"]))
            baseline.append(float(rb["commitment_correct"]))
            term.append(float(ro["skill_terminated"]))
            within2.append(float(abs(ro["skill_steps"]-teacher_steps)<=2))
            owncomp.append(float(ro["completed_correct"])); basecomp.append(float(rb["completed_correct"]))
            pair_ok.append(bool(ro["commitment_correct"]))
            details.append({
                "condition_id":cid,"cue":cue,"own_goal":og,"donor_goal":dg,
                "teacher_lateral_steps":teacher_steps,"own":ro,"donor":rd,"flip":rf,"baseline":rb
            })
        pair.append(float(all(pair_ok)))
        if n==1 or n%16==0: print(f"goal_option_dev_conditions={n}/{len(DEV_IDS)}")
    summary={
        "goal_accuracy":mean(goals),
        "own_commitment_correct":mean(own),
        "paired_both_sides_commitment":mean(pair),
        "donor_commitment_follows_donor_goal":mean(donor),
        "explicit_goal_flip_follows_flipped_goal":mean(flip),
        "baseline_commitment_correct":mean(baseline),
        "commitment_advantage":mean(own)-mean(baseline),
        "skill_termination_rate":mean(term),
        "termination_within_2_teacher_steps":mean(within2),
        "own_completion_correct":mean(owncomp),
        "baseline_completion_correct":mean(basecomp),
        "completion_advantage":mean(owncomp)-mean(basecomp),
    }
    return summary,details

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state-dir",default="state/additive_goal_options_v1")
    args=ap.parse_args()
    state=Path(args.state_dir); state.mkdir(parents=True,exist_ok=True)
    result_path=state/"result.json"; ckpt=state/"goal_options.pt"

    base_before=sha256(BASE_CHECKPOINT); mem_before=sha256(MEMORY_CHECKPOINT)
    goal_sha=sha256(GOAL_DECODER_PATH)

    print("ADDITIVE_GOAL_OPTIONS_V1")
    print("device: cpu")
    print("historical_option_count_preserved: 6")
    print("new_goal_option_count: 2")
    print("historical_option_training_performed: false")
    print("memory_training_performed: false")
    print("planner_training_performed: false")
    print("train_conditions: 0-1023")
    print("fresh_dev_conditions: 1728-1855")
    print("test_touched: false")

    base,cfg,admitted=build_substrate("cpu")
    decoder,_=load_goal_decoder(GOAL_DECODER_PATH,"cpu")
    old={k:v.detach().cpu().clone() for k,v in base.options.state_dict().items()}

    data=collect_dataset(base,cfg)
    dataset_meta={
        "episodes":data["episodes"],
        "samples":int(data["latent"].shape[0]),
        "policy_samples":int((data["action"]>=0).sum()),
        "termination_positive":int((data["terminate"]>.5).sum()),
    }
    print("dataset:",dataset_meta)

    bank,train_metrics=train_bank(cfg,data)
    save_goal_options(
        ckpt,bank,goal_sha,
        {"train_start":0,"train_end":1023,"seed":SEED,"epochs":EPOCHS,"training_metrics":train_metrics}
    )
    bank,_=load_goal_options(ckpt,goal_sha,"cpu")
    dev,details=evaluate(base,cfg,admitted,decoder,bank)

    for k,v in old.items():
        if not torch.equal(v,base.options.state_dict()[k].detach().cpu()):
            raise RuntimeError(f"historical option tensor changed: {k}")
    if sha256(BASE_CHECKPOINT)!=base_before: raise RuntimeError("base checkpoint changed")
    if sha256(MEMORY_CHECKPOINT)!=mem_before: raise RuntimeError("memory checkpoint changed")

    checks={
        "fresh_dev_goal_accuracy":dev["goal_accuracy"]>=MIN_GOAL_ACC,
        "own_commitment":dev["own_commitment_correct"]>=MIN_OWN_COMMIT,
        "paired_both_sides_commitment":dev["paired_both_sides_commitment"]>=MIN_PAIR_COMMIT,
        "donor_commitment":dev["donor_commitment_follows_donor_goal"]>=MIN_DONOR_COMMIT,
        "goal_flip_commitment":dev["explicit_goal_flip_follows_flipped_goal"]>=MIN_FLIP_COMMIT,
        "skill_termination":dev["skill_termination_rate"]>=MIN_TERM_RATE,
        "termination_timing":dev["termination_within_2_teacher_steps"]>=MIN_TERM_WITHIN2,
        "commitment_advantage":dev["commitment_advantage"]>=MIN_ADVANTAGE,
        "baseline_not_solving_memory_task":dev["baseline_commitment_correct"]<=MAX_BASELINE,
    }
    passed=all(checks.values())
    result={
        "format":"continuous_brain.additive_goal_options_v1",
        "development_gate_passed":passed,
        "training":{"new_goal_options":True,"historical_options":False,"memory":False,"base":False,"planner":False},
        "split":{"train":[0,1023],"fresh_dev":[1728,1855],"test_touched":False},
        "train_dataset":dataset_meta,
        "train_metrics":train_metrics,
        "dev":dev,"checks":checks,"details":details,
        "protected_unchanged":True,
        "historical_six_options_bit_exact":True,
        "v019_remains_closed":True,
        "detour_planner_repair_closed":True,
    }
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True))

    print("\\nTRAIN")
    for k,v in train_metrics.items(): print(f"{k}: {v}")
    print("\\nFRESH_DEV")
    for k,v in dev.items(): print(f"{k}: {v}")
    print("\\nGATE")
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("PROTECTED_UNCHANGED: true")
    print("HISTORICAL_SIX_OPTIONS_BIT_EXACT: true")
    print("TEST_TOUCHED: false")
    print("V019_REMAINS_CLOSED: true")
    print("DETOUR_PLANNER_REPAIR_CLOSED: true")
    print("ADDITIVE_GOAL_OPTIONS_DEV_GATE:", "PASS" if passed else "FAIL")
    print("ADDITIVE_GOAL_OPTIONS_RUN_COMPLETE")
    print("result:",result_path)
    print("goal_options:",ckpt)
    raise SystemExit(0 if passed else 4)

if __name__=="__main__":
    main()
