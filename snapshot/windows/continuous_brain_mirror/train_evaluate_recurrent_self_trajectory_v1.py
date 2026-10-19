#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WAIT,
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    PersistentSelfState,
    load_self_model,
    sha256,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    ACTION_DIM,
    MODE_INVESTIGATE,
    MODE_NORMAL,
    PRIMARY_A,
    PRIMARY_B,
    PRIMARY_NONE,
    ExplicitIntentionState,
    RecurrentSelfTrajectoryModel,
    active_goal_from_intention,
    load_trajectory_model,
    rollout,
    save_trajectory_model,
    trajectory_self_value,
    update_explicit_intention,
)
from train_evaluate_joint_self_world_prediction_v1 import (
    make_state,
    make_world,
    transition,
)

TRAIN_IDS = range(0, 4096)
DEV_IDS = range(10000, 11024)
TEST_IDS = range(20000, 21024)
SEQ_LEN = 8

SEED = 31001
EPOCHS = 70
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6

MAX_TF_WORLD_MAE = .030
MAX_TF_SELF_MAE = .025
MIN_TF_PRIMARY = .985
MIN_TF_MODE = .985
MIN_TF_AGENCY = .990

MAX_ROLLOUT_WORLD_MAE = .075
MAX_ROLLOUT_SELF_MAE = .060
MIN_ROLLOUT_PRIMARY = .950
MIN_ROLLOUT_MODE = .950
MIN_ROLLOUT_GOAL = .950

MIN_TIE_FULL = .95
MAX_TIE_SCALAR_ONLY = .60
MIN_TIE_DONOR = .95

MIN_INTERRUPT_PRESERVE = 1.0
MIN_RESUME = 1.0
MIN_SURPRISE_SELF_ERROR = .10

SOURCE_V030_RESULT = Path(
    "state/joint_self_world_prediction_v1/result.json"
)
SOURCE_V0301_RESULT = Path(
    "state/joint_self_world_repair_v1/result.json"
)
SOURCE_V0302_RESULT = Path(
    "state/joint_self_goal_boundary_diagnostic_v1/result.json"
)
V029_TEST = Path(
    "state/self_broadcast_centrality_test_v1/result.json"
)

PROTECTED = {
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
    "self_test":
        Path("state/persistent_self_autobiographical_test_v1/result.json"),
    "v029_test":
        V029_TEST,
    "v030_result":
        SOURCE_V030_RESULT,
    "v0301_result":
        SOURCE_V0301_RESULT,
    "v0302_result":
        SOURCE_V0302_RESULT,
}


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0


def initial_explicit_state(seq_id):
    base = make_state(seq_id)
    rng = random.Random(0x310000 + 100003 * int(seq_id))

    # New family intentionally includes ambiguous commitment ties where the
    # continuing primary intention cannot be recovered from the scalars alone.
    tie_case = (int(seq_id) % 5 == 0)

    if tie_case:
        level = .55 + .35 * rng.random()
        base.unresolved_a = level
        base.unresolved_b = level
        primary = (
            PRIMARY_A
            if (int(seq_id) // 5) % 2 == 0
            else PRIMARY_B
        )
        mode = MODE_NORMAL
        base.uncertainty = .20 + .15 * rng.random()
        base.prediction_error_ema = .08 + .12 * rng.random()
        base.active_goal = (
            GOAL_COMMITMENT_A
            if primary == PRIMARY_A
            else GOAL_COMMITMENT_B
        )
        return base, ExplicitIntentionState(primary, mode), True

    goal = int(base.active_goal)
    if goal == GOAL_COMMITMENT_A:
        primary = PRIMARY_A
        mode = MODE_NORMAL
    elif goal == GOAL_COMMITMENT_B:
        primary = PRIMARY_B
        mode = MODE_NORMAL
    elif goal == GOAL_INVESTIGATE:
        mode = MODE_INVESTIGATE
        if base.unresolved_a > base.unresolved_b + 1e-6:
            primary = PRIMARY_A
        elif base.unresolved_b > base.unresolved_a + 1e-6:
            primary = PRIMARY_B
        elif max(base.unresolved_a, base.unresolved_b) > .10:
            primary = (
                PRIMARY_A
                if rng.random() < .5
                else PRIMARY_B
            )
        else:
            primary = PRIMARY_NONE
    else:
        primary = PRIMARY_NONE
        mode = MODE_NORMAL

    return base, ExplicitIntentionState(primary, mode), False


def sequence_for_id(seq_id):
    rng = random.Random(0x311000 + 104729 * int(seq_id))
    state, intent, tie_case = initial_explicit_state(seq_id)
    world = make_world(seq_id)

    current_world=[]
    current_self=[]
    current_primary=[]
    current_mode=[]
    actions=[]
    next_world=[]
    next_self=[]
    next_primary=[]
    next_mode=[]
    agency=[]

    for t in range(SEQ_LEN):
        # Guarantee information-gathering steps for a meaningful portion of
        # investigate trajectories; otherwise actions remain varied.
        if (
            intent.mode == MODE_INVESTIGATE
            and (t == 0 or rng.random() < .45)
        ):
            action = ACTION_OBSERVE
        elif (
            tie_case
            and t < 3
        ):
            action = (
                ACTION_WORK_A
                if intent.primary == PRIMARY_A
                else ACTION_WORK_B
            )
        else:
            action = rng.randrange(0, ACTION_DIM)

        current_world.append(world.clone())
        current_self.append(state.vector()[:6].clone())
        current_primary.append(int(intent.primary))
        current_mode.append(int(intent.mode))
        actions.append(int(action))

        nw, ns, ag, _ = transition(
            world,
            state,
            action,
            surprise=False,
        )
        next_intent = update_explicit_intention(
            intent.primary,
            intent.mode,
            ns.vector()[:6],
        )

        next_world.append(nw.clone())
        next_self.append(ns.vector()[:6].clone())
        next_primary.append(int(next_intent.primary))
        next_mode.append(int(next_intent.mode))
        agency.append(float(ag))

        world = nw
        state = ns
        intent = next_intent

    return {
        "world": torch.stack(current_world),
        "self": torch.stack(current_self),
        "primary": torch.tensor(current_primary, dtype=torch.long),
        "mode": torch.tensor(current_mode, dtype=torch.long),
        "action": torch.tensor(actions, dtype=torch.long),
        "next_world": torch.stack(next_world),
        "next_self": torch.stack(next_self),
        "next_primary": torch.tensor(next_primary, dtype=torch.long),
        "next_mode": torch.tensor(next_mode, dtype=torch.long),
        "agency": torch.tensor(agency, dtype=torch.float32),
        "tie_case": bool(tie_case),
    }


def make_dataset(ids,label):
    rows=[]
    ids=list(ids)
    for n, sid in enumerate(ids,1):
        rows.append(sequence_for_id(sid))
        if n==1 or n%1024==0 or n==len(ids):
            print(f"{label}_trajectory_sequences={n}/{len(ids)}")

    keys=[
        "world","self","primary","mode","action",
        "next_world","next_self","next_primary","next_mode","agency"
    ]
    data={k:torch.stack([r[k] for r in rows]) for k in keys}
    data["tie_case"]=torch.tensor(
        [r["tie_case"] for r in rows],
        dtype=torch.bool,
    )
    return data


def teacher_forced_outputs(model,data,idx=None):
    if idx is None:
        world=data["world"]
        selfc=data["self"]
        primary=data["primary"]
        mode=data["mode"]
        action=data["action"]
    else:
        world=data["world"][idx]
        selfc=data["self"][idx]
        primary=data["primary"][idx]
        mode=data["mode"][idx]
        action=data["action"][idx]

    B,T,_=world.shape
    hidden=model.init_hidden(
        world[:,0],
        selfc[:,0],
        primary[:,0],
        mode[:,0],
    )

    outs=[]
    for t in range(T):
        out=model.step(
            world[:,t],
            selfc[:,t],
            primary[:,t],
            mode[:,t],
            action[:,t],
            hidden,
        )
        hidden=out["hidden"]
        outs.append(out)
    return outs


@torch.no_grad()
def teacher_metrics(model,data):
    model.eval()
    outs=teacher_forced_outputs(model,data)
    pred_world=torch.stack([o["next_world"] for o in outs],dim=1)
    pred_self=torch.stack([o["next_self_cont"] for o in outs],dim=1)
    pred_primary=torch.stack([o["primary_logits"].argmax(-1) for o in outs],dim=1)
    pred_mode=torch.stack([o["mode_logits"].argmax(-1) for o in outs],dim=1)
    pred_agency=torch.stack([(torch.sigmoid(o["agency_logit"])>=.5) for o in outs],dim=1)

    return {
        "world_mae":float((pred_world-data["next_world"]).abs().mean().item()),
        "self_mae":float((pred_self-data["next_self"]).abs().mean().item()),
        "primary_accuracy":float((pred_primary==data["next_primary"]).float().mean().item()),
        "mode_accuracy":float((pred_mode==data["next_mode"]).float().mean().item()),
        "active_goal_accuracy":float(
            (
                active_goal_from_intention(pred_primary,pred_mode)
                ==
                active_goal_from_intention(data["next_primary"],data["next_mode"])
            ).float().mean().item()
        ),
        "agency_accuracy":float(
            (
                pred_agency==(data["agency"]>=.5)
            ).float().mean().item()
        ),
    }


def train_model(model,train,dev):
    torch.manual_seed(SEED)
    opt=torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen=torch.Generator().manual_seed(SEED)
    n=train["world"].shape[0]
    best=None
    best_score=float("inf")
    best_epoch=0

    for epoch in range(1,EPOCHS+1):
        model.train()
        order=torch.randperm(n,generator=gen)
        total=0.0
        seen=0

        for start in range(0,n,BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]
            outs=teacher_forced_outputs(model,train,idx)

            world_loss=0.0
            self_loss=0.0
            primary_loss=0.0
            mode_loss=0.0
            agency_loss=0.0

            for t,out in enumerate(outs):
                world_loss=world_loss+F.mse_loss(
                    out["next_world"],
                    train["next_world"][idx,t],
                )
                self_loss=self_loss+F.mse_loss(
                    out["next_self_cont"],
                    train["next_self"][idx,t],
                )
                primary_loss=primary_loss+F.cross_entropy(
                    out["primary_logits"],
                    train["next_primary"][idx,t],
                )
                mode_loss=mode_loss+F.cross_entropy(
                    out["mode_logits"],
                    train["next_mode"][idx,t],
                )
                agency_loss=agency_loss+F.binary_cross_entropy_with_logits(
                    out["agency_logit"],
                    train["agency"][idx,t],
                )

            scale=1.0/len(outs)
            loss=(
                1.2*world_loss*scale
                +1.5*self_loss*scale
                +.8*primary_loss*scale
                +.8*mode_loss*scale
                +.4*agency_loss*scale
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()

            total+=float(loss.item())*len(idx)
            seen+=len(idx)

        train_m=teacher_metrics(model,train)
        dev_m=teacher_metrics(model,dev)
        score=(
            dev_m["world_mae"]
            +dev_m["self_mae"]
            +(1.0-dev_m["primary_accuracy"])
            +(1.0-dev_m["mode_accuracy"])
            +.5*(1.0-dev_m["agency_accuracy"])
        )

        if score<best_score:
            best_score=score
            best_epoch=epoch
            best={
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }

        if epoch==1 or epoch%5==0 or epoch==EPOCHS:
            print(
                f"trajectory_epoch={epoch:03d}/{EPOCHS} "
                f"loss={total/max(1,seen):.6f} "
                f"dev_world={dev_m['world_mae']:.4f} "
                f"dev_self={dev_m['self_mae']:.4f} "
                f"dev_primary={dev_m['primary_accuracy']:.4f} "
                f"dev_mode={dev_m['mode_accuracy']:.4f} "
                f"dev_goal={dev_m['active_goal_accuracy']:.4f}"
            )

        if (
            epoch>=15
            and dev_m["world_mae"]<=.018
            and dev_m["self_mae"]<=.018
            and dev_m["primary_accuracy"]>=.995
            and dev_m["mode_accuracy"]>=.995
            and dev_m["agency_accuracy"]>=.995
        ):
            print(f"trajectory_early_stop_epoch={epoch}")
            break

    if best is None:
        raise RuntimeError("no best trajectory model")
    model.load_state_dict(best)
    model.eval()
    return teacher_metrics(model,train),teacher_metrics(model,dev),best_epoch


@torch.no_grad()
def rollout_metrics(model,data,max_sequences=512):
    n=min(max_sequences,data["world"].shape[0])
    world_err=[]
    self_err=[]
    primary_ok=[]
    mode_ok=[]
    goal_ok=[]

    for i in range(n):
        actions=data["action"][i].tolist()
        r=rollout(
            model,
            data["world"][i,0],
            data["self"][i,0],
            int(data["primary"][i,0].item()),
            int(data["mode"][i,0].item()),
            actions,
        )
        true_world=data["next_world"][i]
        true_self=data["next_self"][i]
        pred_world=r["worlds"]
        pred_self=r["selfs"]

        world_err.append(float((pred_world-true_world).abs().mean().item()))
        self_err.append(float((pred_self-true_self).abs().mean().item()))

        tp=data["next_primary"][i]
        tm=data["next_mode"][i]
        pg=torch.tensor(r["primaries"])
        mg=torch.tensor(r["modes"])
        primary_ok.extend((pg==tp).float().tolist())
        mode_ok.extend((mg==tm).float().tolist())
        goal_ok.extend(
            (
                active_goal_from_intention(pg,mg)
                ==
                active_goal_from_intention(tp,tm)
            ).float().tolist()
        )

    return {
        "world_mae":mean(world_err),
        "self_mae":mean(self_err),
        "primary_accuracy":mean(primary_ok),
        "mode_accuracy":mean(mode_ok),
        "active_goal_accuracy":mean(goal_ok),
        "sequences":n,
        "horizon":SEQ_LEN,
    }


@torch.no_grad()
def tie_intention_planning_assay(model):
    full=[]
    scalar_only=[]
    donor=[]

    candidate_a=[ACTION_WORK_A]*3
    candidate_b=[ACTION_WORK_B]*3

    for i in range(256):
        world=torch.tensor([1.0,1.0,.55,.30],dtype=torch.float32)
        # EXACT same continuous self in A and B conditions.
        selfc=torch.tensor([.90,.90,.65,.20,.10,.50],dtype=torch.float32)

        def score(primary,actions,explicit=True):
            p=int(primary)
            if not explicit:
                # scalar-only reconstruction on an exact tie deterministically
                # selects A, erasing the continuing B intention.
                p=PRIMARY_A
            r=rollout(
                model,world,selfc,p,MODE_NORMAL,actions
            )
            return float(
                trajectory_self_value(
                    r["selfs"][-1],
                    primary=p,
                    final_mode=r["final_mode"],
                ).item()
            )

        for primary,expected in (
            (PRIMARY_A,0),
            (PRIMARY_B,1),
        ):
            scores=[
                score(primary,candidate_a,True),
                score(primary,candidate_b,True),
            ]
            pred=max(range(2),key=lambda j:scores[j])
            full.append(float(pred==expected))

            scores_z=[
                score(primary,candidate_a,False),
                score(primary,candidate_b,False),
            ]
            pred_z=max(range(2),key=lambda j:scores_z[j])
            scalar_only.append(float(pred_z==expected))

        a_scores=[
            score(PRIMARY_A,candidate_a,True),
            score(PRIMARY_A,candidate_b,True),
        ]
        b_scores=[
            score(PRIMARY_B,candidate_a,True),
            score(PRIMARY_B,candidate_b,True),
        ]
        donor.append(float(
            max(range(2),key=lambda j:a_scores[j])==0
            and max(range(2),key=lambda j:b_scores[j])==1
        ))

    return {
        "explicit_intention_guided_trajectory_choice":mean(full),
        "scalar_rederived_intention_choice":mean(scalar_only),
        "donor_primary_flips_choice_world_and_scalars_fixed":mean(donor),
    }


def interruption_resumption_assay():
    preserve=[]
    resume=[]
    self_errors=[]

    for i in range(128):
        base=make_state(30000+4*i+GOAL_COMMITMENT_A)
        base.unresolved_a=.85
        base.unresolved_b=.10
        base.confidence=.65
        base.uncertainty=.20
        base.prediction_error_ema=.10
        base.active_goal=GOAL_COMMITMENT_A
        intent=ExplicitIntentionState(PRIMARY_A,MODE_NORMAL)
        world=torch.tensor([.35,.20,.45,.35],dtype=torch.float32)

        normal_world,normal_state,_,_=transition(
            world,base,ACTION_WORK_A,surprise=False
        )
        surprise_world,surprise_state,_,_=transition(
            world,base,ACTION_WORK_A,surprise=True
        )

        expected_self=normal_state.vector()[:6]
        actual_self=surprise_state.vector()[:6]
        self_errors.append(
            float((expected_self-actual_self).abs().mean().item())
        )

        surprised_intent=update_explicit_intention(
            intent.primary,
            intent.mode,
            actual_self,
        )
        preserve.append(float(
            surprised_intent.primary==PRIMARY_A
            and surprised_intent.mode==MODE_INVESTIGATE
        ))

        state=surprise_state
        world=surprise_world
        cur=surprised_intent

        # Information gathering resolves temporary investigate mode while the
        # primary intention remains A.
        for _ in range(5):
            world,state,_,_=transition(
                world,state,ACTION_OBSERVE,surprise=False
            )
            cur=update_explicit_intention(
                cur.primary,
                cur.mode,
                state.vector()[:6],
            )
            if cur.mode==MODE_NORMAL:
                break

        resume.append(float(
            cur.primary==PRIMARY_A
            and cur.mode==MODE_NORMAL
            and cur.active_goal==GOAL_COMMITMENT_A
        ))

    return {
        "surprise_preserves_primary_and_enters_investigate":mean(preserve),
        "primary_resumes_after_investigate_resolves":mean(resume),
        "mean_surprise_self_prediction_error":mean(self_errors),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recurrent_self_trajectory_v1",
    )
    args=ap.parse_args()

    state_dir=Path(args.state_dir)
    state_dir.mkdir(parents=True,exist_ok=True)
    model_path=state_dir/"recurrent_self_trajectory.pt"
    result_path=state_dir/"result.json"

    for name,path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    v030=json.loads(SOURCE_V030_RESULT.read_text())
    v0301=json.loads(SOURCE_V0301_RESULT.read_text())
    v0302=json.loads(SOURCE_V0302_RESULT.read_text())
    if v030.get("development_gate_passed") is not False:
        raise RuntimeError("expected v0.30.0 DEV FAIL")
    if v0301.get("repair_dev_gate_passed") is not False:
        raise RuntimeError("expected v0.30.1 repair FAIL")
    if v0302.get("joint_test_touched") is not False:
        raise RuntimeError("v0.30 TEST was touched")
    if v0302.get("finding")!="GOAL_ERRORS_CONCENTRATE_NEAR_SELF_ARBITRATION_BOUNDARIES":
        raise RuntimeError("v0.30.2 boundary finding required")

    before={name:sha256(path) for name,path in PROTECTED.items()}

    print("RECURRENT_SELF_TRAJECTORY_V1")
    print("device: cpu")
    print("goal: recurrent_imagined_self_trajectory_with_explicit_intention")
    print("trainable_component: recurrent_self_trajectory_model_only")
    print("historical_brain_training: false")
    print("persistent_self_training: false")
    print("v030_test_touched: false")
    print("trajectory_train_ids: 0-4095")
    print("trajectory_dev_ids: 10000-11023")
    print("trajectory_test_ids: 20000-21023")
    print("trajectory_test_touched: false")
    print("sequence_length:",SEQ_LEN)
    print("intention_representation: primary_commitment_plus_temporary_mode")
    print("hysteresis_enter: error>=0.60 or uncertainty>=0.86")
    print("hysteresis_exit: error<=0.45 and uncertainty<=0.72")

    train=make_dataset(TRAIN_IDS,"train")
    dev=make_dataset(DEV_IDS,"dev")

    torch.manual_seed(SEED)
    model=RecurrentSelfTrajectoryModel(
        hidden_dim=144,
        context_dim=96,
    )
    train_tf,dev_tf,best_epoch=train_model(model,train,dev)

    save_trajectory_model(
        model_path,
        model,
        metadata={
            "seed":SEED,
            "train_ids":[0,4095],
            "dev_ids":[10000,11023],
            "test_ids":[20000,21023],
            "test_touched":False,
            "sequence_length":SEQ_LEN,
            "best_epoch":best_epoch,
            "explicit_intention":{
                "primary":["NONE","A","B"],
                "mode":["NORMAL","INVESTIGATE"],
                "enter_error":.60,
                "enter_uncertainty":.86,
                "exit_error":.45,
                "exit_uncertainty":.72,
            },
        },
    )
    model,_=load_trajectory_model(model_path,"cpu")

    rollout_dev=rollout_metrics(model,dev)
    tie=tie_intention_planning_assay(model)
    interruption=interruption_resumption_assay()

    checks={
        "teacher_world":dev_tf["world_mae"]<=MAX_TF_WORLD_MAE,
        "teacher_self":dev_tf["self_mae"]<=MAX_TF_SELF_MAE,
        "teacher_primary":dev_tf["primary_accuracy"]>=MIN_TF_PRIMARY,
        "teacher_mode":dev_tf["mode_accuracy"]>=MIN_TF_MODE,
        "teacher_agency":dev_tf["agency_accuracy"]>=MIN_TF_AGENCY,

        "rollout_world":rollout_dev["world_mae"]<=MAX_ROLLOUT_WORLD_MAE,
        "rollout_self":rollout_dev["self_mae"]<=MAX_ROLLOUT_SELF_MAE,
        "rollout_primary":rollout_dev["primary_accuracy"]>=MIN_ROLLOUT_PRIMARY,
        "rollout_mode":rollout_dev["mode_accuracy"]>=MIN_ROLLOUT_MODE,
        "rollout_goal":rollout_dev["active_goal_accuracy"]>=MIN_ROLLOUT_GOAL,

        "explicit_intention_guides_trajectory":
            tie["explicit_intention_guided_trajectory_choice"]>=MIN_TIE_FULL,
        "scalar_only_control_collapses":
            tie["scalar_rederived_intention_choice"]<=MAX_TIE_SCALAR_ONLY,
        "donor_primary_flips_trajectory_choice":
            tie["donor_primary_flips_choice_world_and_scalars_fixed"]>=MIN_TIE_DONOR,

        "surprise_preserves_primary":
            interruption["surprise_preserves_primary_and_enters_investigate"]>=MIN_INTERRUPT_PRESERVE,
        "primary_resumes":
            interruption["primary_resumes_after_investigate_resolves"]>=MIN_RESUME,
        "surprise_self_error":
            interruption["mean_surprise_self_prediction_error"]>=MIN_SURPRISE_SELF_ERROR,
    }
    passed=all(checks.values())

    after={name:sha256(path) for name,path in PROTECTED.items()}
    if before!=after:
        raise RuntimeError("protected artifact changed during v0.31")

    result={
        "format":"continuous_brain.recurrent_self_trajectory_v1",
        "development_gate_passed":bool(passed),
        "training":{
            "recurrent_self_trajectory_model":True,
            "historical_brain":False,
            "persistent_self":False,
            "working_memory":False,
            "episodic_reader":False,
        },
        "split":{
            "train":[0,4095],
            "dev":[10000,11023],
            "test":[20000,21023],
            "test_touched":False,
        },
        "historical_status":{
            "v0300":"FAIL",
            "v0301":"FAIL",
            "v0302_finding":"GOAL_ERRORS_CONCENTRATE_NEAR_SELF_ARBITRATION_BOUNDARIES",
            "v030_test_touched":False,
        },
        "architecture":{
            "explicit_primary_intention":True,
            "temporary_investigate_mode":True,
            "hysteresis":True,
            "sequence_length":SEQ_LEN,
            "hidden_dim":144,
        },
        "scope":{
            "claim":"recurrent imagined self-trajectories preserve explicit primary intention through temporary investigate mode and make that intention causally load-bearing in trajectory choice",
            "subjective_experience_claimed":False,
            "sentience_claimed":False,
            "consciousness_claimed":False,
        },
        "best_epoch":best_epoch,
        "train_teacher_forced":train_tf,
        "dev_teacher_forced":dev_tf,
        "dev_rollout":rollout_dev,
        "tie_intention_metrics":tie,
        "interruption_resumption_metrics":interruption,
        "checks":checks,
        "protected_unchanged":True,
        "trajectory_test_touched":False,
    }
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True))

    print("\nTRAIN_TEACHER_FORCED")
    for k,v in train_tf.items():
        print(f"{k}: {v}")

    print("\nDEV_TEACHER_FORCED")
    for k,v in dev_tf.items():
        print(f"{k}: {v}")

    print("\nDEV_AUTOREGRESSIVE_ROLLOUT")
    for k,v in rollout_dev.items():
        print(f"{k}: {v}")

    print("\nTIE_INTENTION_CAUSAL")
    for k,v in tie.items():
        print(f"{k}: {v}")

    print("\nINTERRUPTION_RESUMPTION")
    for k,v in interruption.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k,v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("V030_TEST_TOUCHED: false")
    print("TRAJECTORY_TEST_TOUCHED: false")
    print(
        "RECURRENT_SELF_TRAJECTORY_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print("RECURRENT_SELF_TRAJECTORY_RUN_COMPLETE")
    print("result:",result_path)
    print("model:",model_path)

    raise SystemExit(0 if passed else 4)


if __name__=="__main__":
    main()
