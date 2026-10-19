#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.dynamic_route_world_v1 import (
    ACTION_NAMES,
    NODE_NAMES,
    START,
    LEFT_MID,
    RIGHT_MID,
    OPEN,
    KNOWN_BLOCK,
    SURPRISE_BLOCK,
    MODE_NAMES,
    APPROACH_LEFT,
    APPROACH_RIGHT,
    DIRECT,
    SWITCH,
    CROSS_FINISH,
    DynamicRouteWorld,
    generate_spec,
    model_transition,
)
from continuous_brain.consequence_replanning_v1 import (
    RouteConsequenceModel,
    action_vector,
    goal_vector,
    load_consequence_model,
    plan_route,
    predict_transition,
    save_consequence_model,
    state_mismatch,
    state_vector,
)
from continuous_brain.episodic_goal_skill_integration_v1 import (
    EPISODIC_READER_PATH,
    EXPECTED_EPISODIC_READER_SHA,
    GOAL_DECODER_PATH,
    decode_goal,
    select_canonical_recall,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    build_substrate,
    collect_decision_snapshot,
    load_goal_decoder,
    sha256,
)

TRAIN_ROUTE_IDS = range(0, 2048)
DEV_ROUTE_IDS = range(2048, 2304)
TEST_ROUTE_IDS = range(2304, 2560)

SEED = 25001
EPOCHS = 40
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Frozen DEV gate.
MIN_MODEL_NODE_ACC = .995
MIN_MODEL_BLOCK_ACC = .99
MAX_MODEL_COST_MAE = .04
MIN_MODEL_SUCCESS_ACC = .995
MIN_MODEL_VALID_ACC = .995

MIN_MEMORY_GOAL_ACC = .99
MIN_OVERALL_REPLAN_SUCCESS = .95
MIN_OPEN_SUCCESS = .95
MIN_KNOWN_SUCCESS = .95
MIN_SURPRISE_SUCCESS = .95
MAX_SURPRISE_PLAN_ONCE = .10
MIN_SURPRISE_REPLAN_TRIGGER = .95
MIN_SURPRISE_MISMATCH = .95
MIN_QUERY_PLAN_FLIP = .95
MIN_KNOWN_MODEL_OVERRIDE = .95
MIN_SURPRISE_ADVANTAGE = .80


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0


def cue_value(cue):
    return 0 if cue=="left" else 1


def cue_name(v):
    return "left" if int(v)==0 else "right"


def canonical_states_for_spec(spec):
    known = bool(spec.mode == KNOWN_BLOCK)
    return [
        {"node": START, "blocked": False, "known_target_blocked": known},
        {"node": LEFT_MID, "blocked": False, "known_target_blocked": known},
        {"node": LEFT_MID, "blocked": True, "known_target_blocked": known},
        {"node": RIGHT_MID, "blocked": False, "known_target_blocked": known},
        {"node": RIGHT_MID, "blocked": True, "known_target_blocked": known},
    ]


def build_transition_dataset(ids):
    sv=[]; gv=[]; av=[]
    next_node=[]; next_block=[]; cost=[]; success=[]; valid=[]

    for n,cid in enumerate(ids,1):
        spec=generate_spec(cid)
        for goal in (0,1):
            for state in canonical_states_for_spec(spec):
                for action in range(len(ACTION_NAMES)):
                    target=model_transition(state,goal,action,spec)
                    sv.append(state_vector(state,spec))
                    gv.append(goal_vector(goal))
                    av.append(action_vector(action))
                    next_node.append(int(target["next_node"]))
                    next_block.append(float(target["next_blocked"]))
                    cost.append(float(target["cost"]))
                    success.append(float(target["success"]))
                    valid.append(float(target["valid"]))
        if n==1 or n%256==0 or n==len(ids):
            print(f"transition_dataset_conditions={n}/{len(ids)}")

    return {
        "state":torch.stack(sv).float(),
        "goal":torch.stack(gv).float(),
        "action":torch.stack(av).float(),
        "next_node":torch.tensor(next_node,dtype=torch.long),
        "next_blocked":torch.tensor(next_block,dtype=torch.float32),
        "cost":torch.tensor(cost,dtype=torch.float32),
        "success":torch.tensor(success,dtype=torch.float32),
        "valid":torch.tensor(valid,dtype=torch.float32),
    }


def model_metrics(model,data):
    model.eval()
    with torch.no_grad():
        out=model(data["state"],data["goal"],data["action"])
        node=float(
            (out["next_node_logits"].argmax(-1)==data["next_node"])
            .float().mean().item()
        )
        block=float(
            (
                (torch.sigmoid(out["next_blocked_logit"])>=.5).long()
                ==data["next_blocked"].long()
            ).float().mean().item()
        )
        cmae=float((out["cost"]-data["cost"]).abs().mean().item())
        succ=float(
            (
                (torch.sigmoid(out["success_logit"])>=.5).long()
                ==data["success"].long()
            ).float().mean().item()
        )
        val=float(
            (
                (torch.sigmoid(out["valid_logit"])>=.5).long()
                ==data["valid"].long()
            ).float().mean().item()
        )
    return {
        "next_node_accuracy":node,
        "next_blocked_accuracy":block,
        "cost_mae":cmae,
        "success_accuracy":succ,
        "valid_accuracy":val,
    }


def train_model(train):
    torch.manual_seed(SEED)
    random.seed(SEED)

    model=RouteConsequenceModel(hidden_dim=128)
    opt=torch.optim.AdamW(
        model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY
    )
    gen=torch.Generator().manual_seed(SEED)
    n=train["state"].shape[0]
    metrics={}

    for epoch in range(1,EPOCHS+1):
        model.train()
        order=torch.randperm(n,generator=gen)
        total=0.0; seen=0

        for start in range(0,n,BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]
            out=model(
                train["state"][idx],
                train["goal"][idx],
                train["action"][idx],
            )
            loss_node=F.cross_entropy(
                out["next_node_logits"],train["next_node"][idx]
            )
            loss_block=F.binary_cross_entropy_with_logits(
                out["next_blocked_logit"],train["next_blocked"][idx]
            )
            loss_cost=F.smooth_l1_loss(
                out["cost"],train["cost"][idx]
            )
            loss_success=F.binary_cross_entropy_with_logits(
                out["success_logit"],train["success"][idx]
            )
            loss_valid=F.binary_cross_entropy_with_logits(
                out["valid_logit"],train["valid"][idx]
            )
            loss=(
                loss_node+loss_block+loss_cost+
                loss_success+.5*loss_valid
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total+=float(loss.item())*len(idx); seen+=len(idx)

        metrics=model_metrics(model,train)
        metrics["epoch"]=epoch
        metrics["loss"]=total/max(1,seen)

        if epoch==1 or epoch%5==0 or epoch==EPOCHS:
            print(
                f"consequence_epoch={epoch:03d}/{EPOCHS} "
                f"loss={metrics['loss']:.6f} "
                f"node={metrics['next_node_accuracy']:.4f} "
                f"block={metrics['next_blocked_accuracy']:.4f} "
                f"cost_mae={metrics['cost_mae']:.4f} "
                f"success={metrics['success_accuracy']:.4f} "
                f"valid={metrics['valid_accuracy']:.4f}"
            )

        if (
            epoch>=10
            and metrics["next_node_accuracy"]>=.999
            and metrics["next_blocked_accuracy"]>=.999
            and metrics["cost_mae"]<=.02
            and metrics["success_accuracy"]>=.999
            and metrics["valid_accuracy"]>=.999
        ):
            print(f"consequence_early_stop_epoch={epoch}")
            break

    model.eval()
    return model,metrics


def execute_replanning(model,condition_id,goal_side):
    env=DynamicRouteWorld(condition_id,goal_side)
    state=env.reset()

    plan=plan_route(model,state,env.spec,goal_side)
    queue=list(plan["actions"])
    preds=list(plan["predictions"])

    initial_plan=list(queue)
    initial_predicted_success=float(plan["predicted_success_probability"])
    mismatch_count=0
    replan_count=0
    traces=[]

    while not env.done:
        if not queue:
            plan=plan_route(model,state,env.spec,goal_side)
            queue=list(plan["actions"])
            preds=list(plan["predictions"])
            replan_count+=1

        action=int(queue.pop(0))
        pred=preds.pop(0)
        actual,reward,done,info=env.step(action)
        mismatch=state_mismatch(pred["next_state"],actual)
        if mismatch:
            mismatch_count+=1
            if not done:
                plan=plan_route(model,actual,env.spec,goal_side)
                queue=list(plan["actions"])
                preds=list(plan["predictions"])
                replan_count+=1

        traces.append({
            "action":action,
            "action_name":ACTION_NAMES[action],
            "predicted_next":pred["next_state"],
            "actual_next":dict(actual),
            "mismatch":bool(mismatch),
            "reward":float(reward),
        })
        state=actual

    return {
        "success":bool(env.success),
        "mode":MODE_NAMES[env.spec.mode],
        "total_cost":float(env.total_cost),
        "steps":int(env.steps),
        "mismatch_count":int(mismatch_count),
        "replan_count":int(replan_count),
        "initial_plan":initial_plan,
        "initial_predicted_success":initial_predicted_success,
        "trace":traces,
    }


def execute_plan_once(model,condition_id,goal_side):
    env=DynamicRouteWorld(condition_id,goal_side)
    state=env.reset()
    plan=plan_route(model,state,env.spec,goal_side)
    actions=list(plan["actions"])

    for action in actions:
        if env.done:
            break
        env.step(action)

    # A stale plan that ended before the real episode ended counts as failure.
    return {
        "success":bool(env.success),
        "mode":MODE_NAMES[env.spec.mode],
        "total_cost":float(env.total_cost),
        "steps":int(env.steps),
        "initial_plan":actions,
    }


def execute_reactive(condition_id,goal_side):
    env=DynamicRouteWorld(condition_id,goal_side)
    env.reset()
    approach=APPROACH_LEFT if int(goal_side)==0 else APPROACH_RIGHT
    env.step(approach)
    if not env.done:
        env.step(DIRECT)
    return {
        "success":bool(env.success),
        "mode":MODE_NAMES[env.spec.mode],
        "total_cost":float(env.total_cost),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state-dir",default="state/consequence_replanning_v1")
    args=ap.parse_args()
    state_dir=Path(args.state_dir); state_dir.mkdir(parents=True,exist_ok=True)
    ckpt=state_dir/"consequence_model.pt"
    result_path=state_dir/"result.json"

    protected_paths={
        "base":Path(BASE_CHECKPOINT),
        "memory":Path(MEMORY_CHECKPOINT),
        "goal_decoder":Path(GOAL_DECODER_PATH),
        "episodic_reader":Path(EPISODIC_READER_PATH),
    }
    before={k:sha256(v) for k,v in protected_paths.items()}

    print("CONSEQUENCE_REPLANNING_V1")
    print("device: cpu")
    print("new_task_family: dynamic_route_graph_v1")
    print("historical_planner_training: false")
    print("historical_world_model_training: false")
    print("memory_training: false")
    print("episodic_reader_training: false")
    print("goal_decoder_training: false")
    print("trainable_component: route_consequence_model_only")
    print("route_train_ids: 0-2047")
    print("route_dev_ids: 2048-2303")
    print("route_test_ids: 2304-2559")
    print("route_test_touched: false")
    print("surprise_transition_in_training: false")

    train=build_transition_dataset(TRAIN_ROUTE_IDS)
    model,train_metrics=train_model(train)
    save_consequence_model(
        ckpt,model,{
            "seed":SEED,
            "route_train":[0,2047],
            "route_dev":[2048,2303],
            "route_test":[2304,2559],
            "surprise_transition_in_training":False,
        }
    )
    model,_=load_consequence_model(ckpt,"cpu")

    # Fresh DEV transition fidelity uses the ordinary model dynamics.  Surprise
    # changes are deliberately excluded from this prediction dataset.
    dev_transition=build_transition_dataset(DEV_ROUTE_IDS)
    dev_model=model_metrics(model,dev_transition)

    base,cfg,admitted=build_substrate("cpu")
    decoder,_=load_goal_decoder(GOAL_DECODER_PATH,"cpu")
    reader,_=load_reader(EPISODIC_READER_PATH,"cpu")

    recall_cache={}
    def recall(cid,cue):
        key=(int(cid),str(cue))
        if key not in recall_cache:
            recall_cache[key]=collect_decision_snapshot(
                base,cfg,admitted,int(cid),str(cue),device="cpu"
            )["recall"].clone()
        return recall_cache[key]

    memory_goal_ok=[]
    replan_rows=[]
    plan_once_rows=[]
    reactive_rows=[]
    query_flip=[]
    known_override=[]

    by_mode={name:[] for name in MODE_NAMES}
    planonce_by_mode={name:[] for name in MODE_NAMES}
    reactive_by_mode={name:[] for name in MODE_NAMES}
    surprise_replan_trigger=[]
    surprise_mismatch=[]

    # 256 fresh route conditions, two late queries each.
    for n,cid in enumerate(DEV_ROUTE_IDS,1):
        event1=(cid*17+11)%1600
        event2=(cid*37+101)%1600
        cue1="left" if cid%2==0 else "right"
        cue2="right" if cue1=="left" else "left"
        s0=recall(event1,cue1)
        s1=recall(event2,cue2)

        query_results={}
        for query,expected_cue in (
            (QUERY_FIRST,cue1),
            (QUERY_SECOND,cue2),
        ):
            selected=select_canonical_recall(reader,s0,s1,query)
            goal,_=decode_goal(decoder,selected["selected_recall"])
            expected_goal=cue_value(expected_cue)
            memory_goal_ok.append(float(goal==expected_goal))

            r=execute_replanning(model,cid,goal)
            p=execute_plan_once(model,cid,goal)
            h=execute_reactive(cid,goal)

            replan_rows.append(r); plan_once_rows.append(p); reactive_rows.append(h)
            by_mode[r["mode"]].append(float(r["success"]))
            planonce_by_mode[p["mode"]].append(float(p["success"]))
            reactive_by_mode[h["mode"]].append(float(h["success"]))

            if r["mode"]=="surprise_block":
                surprise_replan_trigger.append(float(r["replan_count"]>=1))
                surprise_mismatch.append(float(r["mismatch_count"]>=1))

            if r["mode"]=="known_block":
                target_approach=APPROACH_LEFT if goal==0 else APPROACH_RIGHT
                first=int(r["initial_plan"][0])
                known_override.append(float(first!=target_approach))

            query_results[query]={
                "goal":goal,
                "first_plan_action":int(r["initial_plan"][0]),
            }

        query_flip.append(float(
            query_results[QUERY_FIRST]["goal"]!=query_results[QUERY_SECOND]["goal"]
            and query_results[QUERY_FIRST]["first_plan_action"]
                !=query_results[QUERY_SECOND]["first_plan_action"]
        ))

        if n==1 or n%32==0 or n==len(DEV_ROUTE_IDS):
            print(f"route_dev_conditions={n}/{len(DEV_ROUTE_IDS)}")

    overall=mean(float(r["success"]) for r in replan_rows)
    plan_once_overall=mean(float(r["success"]) for r in plan_once_rows)
    reactive_overall=mean(float(r["success"]) for r in reactive_rows)

    metrics={
        "memory_selected_goal_accuracy":mean(memory_goal_ok),
        "replanning_success":overall,
        "plan_once_success":plan_once_overall,
        "reactive_success":reactive_overall,
        "open_replanning_success":mean(by_mode["open"]),
        "known_block_replanning_success":mean(by_mode["known_block"]),
        "surprise_block_replanning_success":mean(by_mode["surprise_block"]),
        "open_plan_once_success":mean(planonce_by_mode["open"]),
        "known_block_plan_once_success":mean(planonce_by_mode["known_block"]),
        "surprise_block_plan_once_success":mean(planonce_by_mode["surprise_block"]),
        "surprise_block_reactive_success":mean(reactive_by_mode["surprise_block"]),
        "surprise_replan_trigger_rate":mean(surprise_replan_trigger),
        "surprise_mismatch_detection_rate":mean(surprise_mismatch),
        "query_flip_changes_goal_and_initial_plan":mean(query_flip),
        "known_block_model_override_rate":mean(known_override),
        "surprise_replanning_advantage":
            mean(by_mode["surprise_block"])-mean(planonce_by_mode["surprise_block"]),
        "mean_replans":mean(r["replan_count"] for r in replan_rows),
        "mean_mismatches":mean(r["mismatch_count"] for r in replan_rows),
    }

    checks={
        "model_next_node":dev_model["next_node_accuracy"]>=MIN_MODEL_NODE_ACC,
        "model_next_blocked":dev_model["next_blocked_accuracy"]>=MIN_MODEL_BLOCK_ACC,
        "model_cost":dev_model["cost_mae"]<=MAX_MODEL_COST_MAE,
        "model_success":dev_model["success_accuracy"]>=MIN_MODEL_SUCCESS_ACC,
        "model_valid":dev_model["valid_accuracy"]>=MIN_MODEL_VALID_ACC,
        "memory_selected_goal":metrics["memory_selected_goal_accuracy"]>=MIN_MEMORY_GOAL_ACC,
        "overall_replanning_success":metrics["replanning_success"]>=MIN_OVERALL_REPLAN_SUCCESS,
        "open_success":metrics["open_replanning_success"]>=MIN_OPEN_SUCCESS,
        "known_block_success":metrics["known_block_replanning_success"]>=MIN_KNOWN_SUCCESS,
        "surprise_success":metrics["surprise_block_replanning_success"]>=MIN_SURPRISE_SUCCESS,
        "surprise_plan_once_fails":metrics["surprise_block_plan_once_success"]<=MAX_SURPRISE_PLAN_ONCE,
        "surprise_replan_trigger":metrics["surprise_replan_trigger_rate"]>=MIN_SURPRISE_REPLAN_TRIGGER,
        "surprise_mismatch_detection":metrics["surprise_mismatch_detection_rate"]>=MIN_SURPRISE_MISMATCH,
        "query_flip_plan":metrics["query_flip_changes_goal_and_initial_plan"]>=MIN_QUERY_PLAN_FLIP,
        "known_block_consequence_override":metrics["known_block_model_override_rate"]>=MIN_KNOWN_MODEL_OVERRIDE,
        "surprise_replanning_advantage":metrics["surprise_replanning_advantage"]>=MIN_SURPRISE_ADVANTAGE,
    }
    passed=all(checks.values())

    after={k:sha256(v) for k,v in protected_paths.items()}
    if before!=after:
        raise RuntimeError("protected cognitive artifact changed")

    result={
        "format":"continuous_brain.consequence_replanning_v1",
        "development_gate_passed":bool(passed),
        "scope":{
            "claim":"episodic-goal-conditioned consequence planning with prediction-error-triggered replanning in a new symbolic route family",
            "historical_planner_requalified":False,
            "3d_planning_claimed":False,
            "general_intelligence_claimed":False,
        },
        "training":{
            "route_consequence_model":True,
            "historical_planner":False,
            "historical_world_model":False,
            "memory":False,
            "episodic_reader":False,
            "goal_decoder":False,
        },
        "split":{
            "route_train":[0,2047],
            "route_dev":[2048,2303],
            "route_test":[2304,2559],
            "route_test_touched":False,
        },
        "train_metrics":train_metrics,
        "dev_model_metrics":dev_model,
        "behavior_metrics":metrics,
        "checks":checks,
        "protected_unchanged":True,
        "v023_milestone_unchanged":True,
        "v024_integration_unchanged":True,
    }
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True))

    print("\nTRAIN_MODEL")
    for k,v in train_metrics.items(): print(f"{k}: {v}")
    print("\nDEV_MODEL")
    for k,v in dev_model.items(): print(f"{k}: {v}")
    print("\nDEV_BEHAVIOR")
    for k,v in metrics.items(): print(f"{k}: {v}")
    print("\nGATE")
    for k,v in checks.items(): print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("V023_MILESTONE_UNCHANGED: true")
    print("V024_INTEGRATION_UNCHANGED: true")
    print("ROUTE_TEST_TOUCHED: false")
    print("CONSEQUENCE_REPLANNING_DEV_GATE:", "PASS" if passed else "FAIL")
    print("CONSEQUENCE_REPLANNING_RUN_COMPLETE")
    print("result:",result_path)
    print("consequence_model:",ckpt)

    raise SystemExit(0 if passed else 4)


if __name__=="__main__":
    main()
