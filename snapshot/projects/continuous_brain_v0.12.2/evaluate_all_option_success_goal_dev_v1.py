#!/usr/bin/env python3
"""
v0.20.2 development assay: all-option success-goal planning.

NO TRAINING. CPU only.

Fresh relative-to-development seed window:
    14001..14040

This is a DEVELOPMENT gate, not a global held-out generalization claim.

Frozen gate:
- causal all-option goal-swap rate >= .15
- on changed same-state decisions, success-goal chosen plan has >= predicted
  success than neutral plan in >= .80 of swaps
- success-goal autonomous completion >= neutral all-option completion
- success-goal completion >= direct learned-option baseline - .10
- success-goal collisions <= neutral + 5
- success-goal collisions <= direct baseline + 10
- protected artifacts unchanged
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    evaluate_hierarchical_options,
)
from continuous_brain.environment import DetourWorld
from continuous_brain.all_option_success_goal_v1 import (
    NEUTRAL_GOAL,
    SUCCESS_GOAL,
    all_option_plan,
)

CURRENT=Path("state/child_v113_dml_smoke/infant_training.pt")
MEMORY=Path("state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt")
INTENT=Path("state/option_intent_executive_v1/latest.portable.pt")
CLOSED_V019=Path("state/recurrent_option_progress_v1/latest.portable.pt")

EXPECTED={
 "current":"086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
 "memory":"f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
 "intent":"041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
 "v019":"237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
}

SEED_START=14001
EPISODES=40
STEPS=120
OPTION_DEPTH=2
BEAM_WIDTH=12
MAX_OPTION_STEPS=12
DISCOUNT=.95
MIN_COMMIT=3
TERMINATE_THRESHOLD=.50

MIN_SWAP=.15
MIN_DIRECTION=.80
DIRECT_COMPLETION_TOL=.10
NEUTRAL_COLLISION_TOL=5.0
DIRECT_COLLISION_TOL=10.0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()

def protected():
    return {
      "current":sha(CURRENT),
      "memory":sha(MEMORY),
      "intent":sha(INTENT),
      "v019":sha(CLOSED_V019),
    }

def require_provenance():
    paths={"current":CURRENT,"memory":MEMORY,"intent":INTENT,"v019":CLOSED_V019}
    for name,exp in EXPECTED.items():
        p=paths[name]
        if not p.is_file(): raise FileNotFoundError(p)
        got=sha(p)
        if got!=exp: raise RuntimeError(f"{name} SHA mismatch expected={exp} actual={got}")

def mean(xs):
    xs=list(xs);return sum(xs)/len(xs) if xs else 0.0

@torch.no_grad()
def run_episode(model,seed,active_goal,counterfactual_goal):
    env=DetourWorld(seed=int(seed),max_episode_steps=STEPS)
    obs=env.reset()
    hidden=model.nucleus.initial_state(1,"cpu")
    memory=torch.zeros(1,model.cfg.memory_dim,dtype=torch.float32)
    previous_action=torch.zeros(1,dtype=torch.long)

    current_option=None
    option_age=0
    total_reward=0.0
    completed=False
    steps_used=STEPS
    trace=[]

    for step in range(1,STEPS+1):
        image=_env_image_tensor(env,model.cfg.image_size,"cpu")
        context,context7=_env_context(obs,"cpu")

        semantic=model.vision(image)
        spatial_online=model.spatial(image)
        fused=model.fuse_vision(semantic,spatial_online)
        nuc=model.nucleus(fused,context,memory,previous_action,hidden)
        hidden=nuc["hidden"]

        stable_spatial=model.target_spatial_bus(image)
        root=model.planning.encode_state(stable_spatial,context7)

        boundary=_active_option_should_terminate(
            model,root,current_option,option_age,MIN_COMMIT,TERMINATE_THRESHOLD
        )

        if boundary:
            active=all_option_plan(
                model,root,active_goal,
                option_depth=OPTION_DEPTH,
                beam_width=BEAM_WIDTH,
                max_option_steps=MAX_OPTION_STEPS,
                discount=DISCOUNT,
                min_commit=MIN_COMMIT,
                terminate_threshold=TERMINATE_THRESHOLD,
            )
            cf=all_option_plan(
                model,root,counterfactual_goal,
                option_depth=OPTION_DEPTH,
                beam_width=BEAM_WIDTH,
                max_option_steps=MAX_OPTION_STEPS,
                discount=DISCOUNT,
                min_commit=MIN_COMMIT,
                terminate_threshold=TERMINATE_THRESHOLD,
            )
            current_option=int(active["chosen_option"])
            option_age=0
            trace.append({
                "step":step,
                "active_goal":active_goal.name,
                "active":active,
                "counterfactual_goal":counterfactual_goal.name,
                "counterfactual":cf,
                "changed":bool(active["chosen_option"]!=cf["chosen_option"]),
            })

        action=_option_policy_action(model,root,int(current_option))
        option_age+=1
        obs,reward,done,info=env.step(action)
        total_reward+=float(reward)
        previous_action=torch.tensor([action+1],dtype=torch.long)

        if bool(info.get("reached")):
            completed=True
            steps_used=step
        if done: break

    return {
        "seed":int(seed),
        "goal":active_goal.name,
        "completed":bool(completed),
        "steps":int(steps_used),
        "collisions":int(env.collisions),
        "reward":float(total_reward),
        "trace":trace,
    }

def summarize(rows):
    return {
        "episodes":len(rows),
        "completion_rate":mean(float(r["completed"]) for r in rows),
        "mean_collisions":mean(r["collisions"] for r in rows),
        "mean_reward":mean(r["reward"] for r in rows),
        "mean_steps_penalized":mean(r["steps"] if r["completed"] else STEPS+1 for r in rows),
        "mean_boundaries":mean(len(r["trace"]) for r in rows),
    }

def intervention(rows):
    opportunities=0;changes=0;directional=0
    examples=[]
    for row in rows:
        for e in row["trace"]:
            opportunities+=1
            if not e["changed"]:continue
            changes+=1
            if e["active_goal"]=="success":
                success=e["active"]; neutral=e["counterfactual"]
            else:
                success=e["counterfactual"]; neutral=e["active"]
            ssucc=float(success["score_components"]["success"])
            nsucc=float(neutral["score_components"]["success"])
            ok=ssucc>=nsucc
            directional+=int(ok)
            if len(examples)<100:
                examples.append({
                    "step":e["step"],
                    "success_option":success["chosen_option"],
                    "neutral_option":neutral["chosen_option"],
                    "success_predicted_success":ssucc,
                    "neutral_predicted_success":nsucc,
                    "directional_success":ok,
                })
    return {
        "opportunities":opportunities,
        "changes":changes,
        "swap_rate":changes/opportunities if opportunities else 0.0,
        "directional_success_fraction":directional/changes if changes else 0.0,
        "examples":examples,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="state/all_option_success_goal_v1/result.json")
    args=ap.parse_args()

    require_provenance(); before=protected()
    print("ALL_OPTION_SUCCESS_GOAL_DEV_V1")
    print("device: cpu")
    print("training_performed: false")
    print("candidate_filter: all_options")
    print("active_goal_term: predicted_success_only")
    print("collision_goal_weight: 0")
    print("seed_start:",SEED_START)
    print("episodes:",EPISODES)
    print("global_training_holdout_claimed: false")

    baseline=evaluate_hierarchical_options(
        CURRENT,
        episodes=EPISODES,
        steps=STEPS,
        seed_start=SEED_START,
        device="cpu",
        output=None,
        option_depth=2,
        selector_topk=2,
        beam_width=8,
        max_option_steps=12,
        discount=.95,
        min_commit=3,
        terminate_threshold=.50,
    )
    if baseline.get("checkpoint_sha256")!=EXPECTED["current"]:
        raise RuntimeError("baseline provenance mismatch")
    if baseline.get("experiment",{}).get("training_performed_during_evaluation") is not False:
        raise RuntimeError("baseline unexpectedly trained")

    model,saved=load_checkpoint(CURRENT,"cpu")
    model.eval()

    success_rows=[];neutral_rows=[]
    for i in range(EPISODES):
        seed=SEED_START+i
        success_rows.append(run_episode(model,seed,SUCCESS_GOAL,NEUTRAL_GOAL))
        neutral_rows.append(run_episode(model,seed,NEUTRAL_GOAL,SUCCESS_GOAL))
        print(f"paired_dev_episodes={i+1}/{EPISODES}")

    success=summarize(success_rows)
    neutral=summarize(neutral_rows)
    causal=intervention(success_rows+neutral_rows)
    direct=baseline["option_direct_diagnostic"]

    checks={
        "goal_swap_rate": causal["swap_rate"]>=MIN_SWAP,
        "directional_success": causal["directional_success_fraction"]>=MIN_DIRECTION,
        "success_completion_ge_neutral": success["completion_rate"]>=neutral["completion_rate"],
        "success_completion_near_direct": success["completion_rate"]>=float(direct["completion_rate"])-DIRECT_COMPLETION_TOL,
        "success_collisions_vs_neutral": success["mean_collisions"]<=neutral["mean_collisions"]+NEUTRAL_COLLISION_TOL,
        "success_collisions_vs_direct": success["mean_collisions"]<=float(direct["mean_collisions"])+DIRECT_COLLISION_TOL,
    }
    passed=all(checks.values())

    after=protected()
    if before!=after: raise RuntimeError("protected artifact changed")

    result={
        "format":"continuous_brain.all_option_success_goal_dev_v1",
        "development_gate_passed":passed,
        "training_performed":False,
        "device":"cpu",
        "seed_protocol":{
            "seed_start":SEED_START,
            "seed_end":SEED_START+EPISODES-1,
            "episodes":EPISODES,
            "fresh_relative_to_v0201_diagnostic":True,
            "global_training_holdout_claimed":False,
        },
        "thresholds":{
            "min_goal_swap_rate":MIN_SWAP,
            "min_directional_success":MIN_DIRECTION,
            "direct_completion_tolerance":DIRECT_COMPLETION_TOL,
            "neutral_collision_tolerance":NEUTRAL_COLLISION_TOL,
            "direct_collision_tolerance":DIRECT_COLLISION_TOL,
        },
        "checks":checks,
        "baseline":{
            "reactive":baseline["reactive"],
            "direct":direct,
            "historical_hierarchical":baseline["hierarchical_option_planning"],
        },
        "success_goal":success,
        "neutral_all_option":neutral,
        "causal_goal_intervention":causal,
        "protected_unchanged":True,
        "v019_remains_closed":True,
        "memory_preserved":True,
        "v018_intent_preserved":True,
        "episodes_data":{
            "success":success_rows,
            "neutral":neutral_rows,
        },
    }

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True))

    print("\nBASELINE")
    print("direct completion/collisions:",direct["completion_rate"],direct["mean_collisions"])
    print("hierarchical completion/collisions:",baseline["hierarchical_option_planning"]["completion_rate"],baseline["hierarchical_option_planning"]["mean_collisions"])
    print("\nALL_OPTION")
    print("success completion/collisions:",success["completion_rate"],success["mean_collisions"])
    print("neutral completion/collisions:",neutral["completion_rate"],neutral["mean_collisions"])
    print("goal swap rate:",causal["swap_rate"])
    print("directional success:",causal["directional_success_fraction"])
    print()
    for k,v in checks.items():print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("PROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print("ALL_OPTION_SUCCESS_GOAL_DEV_GATE:", "PASS" if passed else "FAIL")
    print("ALL_OPTION_SUCCESS_GOAL_DEV_COMPLETE")
    raise SystemExit(0 if passed else 4)

if __name__=="__main__":main()
