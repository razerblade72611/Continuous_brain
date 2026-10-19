#!/usr/bin/env python3
"""
v0.20.4 spent-window A/B diagnostic.

Compares:
- direct learned options;
- historical capped hierarchical planner;
- continuation-consistent hierarchical planner.

Seeds 14001..14040 are already spent development evidence.
This run makes no new admission/generalization claim.
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
from continuous_brain.continuation_consistent_planning_v1 import (
    continuation_consistent_option_choice,
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
SELECTOR_TOPK=2
BEAM_WIDTH=8
CHUNK_STEPS=12
DISCOUNT=.95
MIN_COMMIT=3
TERMINATE_THRESHOLD=.50


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
def run_episode(model,seed):
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
    high_level=0
    selector_overrides=0
    option_switches=0
    illegal_cap_handoffs=0
    selected_first_macro_over_chunk=0
    selected_first_macro_unterminated_to_horizon=0
    traces=[]

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
            plan=continuation_consistent_option_choice(
                model,
                root,
                remaining_steps=STEPS-step+1,
                option_depth=OPTION_DEPTH,
                selector_topk=SELECTOR_TOPK,
                beam_width=BEAM_WIDTH,
                computation_chunk_steps=CHUNK_STEPS,
                discount=DISCOUNT,
                min_commit=MIN_COMMIT,
                terminate_threshold=TERMINATE_THRESHOLD,
            )
            previous=current_option
            current_option=int(plan["chosen_option"])
            option_age=0
            high_level+=1
            selector_overrides+=int(plan["selector_overridden"])
            option_switches+=int(previous is not None and int(previous)!=current_option)
            illegal_cap_handoffs+=int(plan["illegal_cap_handoffs"])

            first=plan["macro_diagnostics"][0]
            selected_first_macro_over_chunk+=int(first["primitive_steps"]>CHUNK_STEPS)
            selected_first_macro_unterminated_to_horizon+=int(
                first["primitive_steps"]>CHUNK_STEPS
                and not first["terminated"]
                and first["ended_by_episode_horizon"]
            )
            traces.append({
                "step":step,
                "plan":plan,
            })

        action=_option_policy_action(model,root,int(current_option))
        option_age+=1
        obs,reward,done,info=env.step(action)
        total_reward+=float(reward)
        previous_action=torch.tensor([action+1],dtype=torch.long)

        if bool(info.get("reached")):
            completed=True;steps_used=step
        if done:break

    return {
        "seed":int(seed),
        "completed":bool(completed),
        "steps":int(steps_used),
        "collisions":int(env.collisions),
        "reward":float(total_reward),
        "high_level_decisions":int(high_level),
        "selector_overrides":int(selector_overrides),
        "option_switches":int(option_switches),
        "illegal_cap_handoffs":int(illegal_cap_handoffs),
        "selected_first_macro_over_chunk":int(selected_first_macro_over_chunk),
        "selected_first_macro_unterminated_to_horizon":
            int(selected_first_macro_unterminated_to_horizon),
        "trace":traces,
    }


def summary(rows):
    return {
        "episodes":len(rows),
        "completion_rate":mean(float(r["completed"]) for r in rows),
        "mean_collisions":mean(r["collisions"] for r in rows),
        "mean_reward":mean(r["reward"] for r in rows),
        "mean_steps_penalized":mean(r["steps"] if r["completed"] else STEPS+1 for r in rows),
        "mean_high_level_decisions":mean(r["high_level_decisions"] for r in rows),
        "mean_selector_overrides":mean(r["selector_overrides"] for r in rows),
        "mean_option_switches":mean(r["option_switches"] for r in rows),
        "illegal_cap_handoffs":sum(r["illegal_cap_handoffs"] for r in rows),
        "selected_first_macro_over_chunk":
            sum(r["selected_first_macro_over_chunk"] for r in rows),
        "selected_first_macro_unterminated_to_horizon":
            sum(r["selected_first_macro_unterminated_to_horizon"] for r in rows),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="state/continuation_consistent_planning_v1/result.json")
    args=ap.parse_args()

    require_provenance();before=protected()

    print("CONTINUATION_CONSISTENT_PLANNING_V1")
    print("device: cpu")
    print("training_performed: false")
    print("spent_development_window: 14001-14040")
    print("new_development_window_consumed: false")
    print("computation_chunk_steps:",CHUNK_STEPS)
    print("chunk_boundary_allows_option_switch: false")

    baseline=evaluate_hierarchical_options(
        CURRENT,
        episodes=EPISODES,
        steps=STEPS,
        seed_start=SEED_START,
        device="cpu",
        output=None,
        option_depth=OPTION_DEPTH,
        selector_topk=SELECTOR_TOPK,
        beam_width=BEAM_WIDTH,
        max_option_steps=CHUNK_STEPS,
        discount=DISCOUNT,
        min_commit=MIN_COMMIT,
        terminate_threshold=TERMINATE_THRESHOLD,
    )
    if baseline.get("checkpoint_sha256")!=EXPECTED["current"]:
        raise RuntimeError("baseline provenance mismatch")

    model,_=load_checkpoint(CURRENT,"cpu")
    model.eval()

    rows=[]
    for i in range(EPISODES):
        rows.append(run_episode(model,SEED_START+i))
        print(f"continuation_consistent_episodes={i+1}/{EPISODES}")

    repaired=summary(rows)

    after=protected()
    if before!=after:raise RuntimeError("protected artifact changed")

    historical=baseline["hierarchical_option_planning"]
    direct=baseline["option_direct_diagnostic"]

    descriptive={
        "completion_delta_vs_historical":
            repaired["completion_rate"]-float(historical["completion_rate"]),
        "collision_delta_vs_historical":
            repaired["mean_collisions"]-float(historical["mean_collisions"]),
        "completion_delta_vs_direct":
            repaired["completion_rate"]-float(direct["completion_rate"]),
        "collision_delta_vs_direct":
            repaired["mean_collisions"]-float(direct["mean_collisions"]),
    }

    result={
        "format":"continuous_brain.continuation_consistent_planning_v1",
        "software_status":"PASS",
        "scientific_admission_decided":False,
        "training_performed":False,
        "device":"cpu",
        "seed_window":{
            "start":SEED_START,
            "end":SEED_START+EPISODES-1,
            "reused_spent_development":True,
            "new_development_window_consumed":False,
        },
        "contract":{
            "computation_chunk_steps":CHUNK_STEPS,
            "chunk_boundary_allows_option_switch":False,
            "branch_only_after_predicted_termination":True,
            "remaining_episode_horizon_enforced":True,
        },
        "baseline":{
            "direct":direct,
            "historical_hierarchical":historical,
        },
        "continuation_consistent":repaired,
        "descriptive_comparison":descriptive,
        "episodes":rows,
        "protected_unchanged":True,
        "v019_remains_closed":True,
        "memory_preserved":True,
        "v018_intent_preserved":True,
    }

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True))

    print("\nBASELINE")
    print("direct completion/collisions:",direct["completion_rate"],direct["mean_collisions"])
    print("historical hierarchical completion/collisions:",historical["completion_rate"],historical["mean_collisions"])
    print("\nCONTINUATION_CONSISTENT")
    print("completion/collisions:",repaired["completion_rate"],repaired["mean_collisions"])
    print("illegal_cap_handoffs:",repaired["illegal_cap_handoffs"])
    print("selected_first_macro_over_chunk:",repaired["selected_first_macro_over_chunk"])
    print("selected_first_macro_unterminated_to_horizon:",repaired["selected_first_macro_unterminated_to_horizon"])
    print("completion_delta_vs_historical:",descriptive["completion_delta_vs_historical"])
    print("collision_delta_vs_historical:",descriptive["collision_delta_vs_historical"])
    print("completion_delta_vs_direct:",descriptive["completion_delta_vs_direct"])
    print("collision_delta_vs_direct:",descriptive["collision_delta_vs_direct"])

    print("\nPROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print("SCIENTIFIC_ADMISSION_DECIDED: false")
    print("CONTINUATION_CONSISTENT_PLANNING_RUN: PASS")
    print("result:",out)

if __name__=="__main__":main()
