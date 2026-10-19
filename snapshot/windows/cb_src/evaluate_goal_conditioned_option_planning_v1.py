#!/usr/bin/env python3
"""
Evaluate v0.20.0 goal-conditioned autonomous option planning.

Software PASS means the read-only causal assay completed with valid provenance.
It is NOT itself a scientific admission threshold.

The assay uses CPU because v0.19.2 established that the historical planner is
behaviorally requalified on CPU but not DirectML.

Evaluation seeds 13001..13020 are new to the historical P4b reports, but this
script does NOT claim they were held out from all historical training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v103 import evaluate_hierarchical_options
from continuous_brain.goal_conditioned_option_planning_v1 import (
    COMPLETION_GOAL,
    SAFETY_GOAL,
    run_goal_episode,
    summarize,
)

CURRENT = Path("state/child_v113_dml_smoke/infant_training.pt")
MEMORY = Path(
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
)
INTENT = Path("state/option_intent_executive_v1/latest.portable.pt")
CLOSED_V019 = Path("state/recurrent_option_progress_v1/latest.portable.pt")

EXPECTED = {
    "current":
        "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
    "memory":
        "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "intent":
        "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "v019":
        "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
}

SEED_START = 13001
EPISODES = 20
STEPS = 120

PLANNER_ARGS = dict(
    option_depth=2,
    selector_topk=2,
    beam_width=8,
    max_option_steps=12,
    discount=.95,
    min_commit=3,
    terminate_threshold=.50,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8*1024*1024), b""):
            h.update(c)
    return h.hexdigest()


def protected_hashes():
    return {
        "current": sha(CURRENT),
        "memory": sha(MEMORY),
        "intent": sha(INTENT),
        "v019": sha(CLOSED_V019),
    }


def require_provenance():
    for name, expected in EXPECTED.items():
        path = {
            "current": CURRENT,
            "memory": MEMORY,
            "intent": INTENT,
            "v019": CLOSED_V019,
        }[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha(path)
        if got != expected:
            raise RuntimeError(
                f"{name} SHA mismatch expected={expected} actual={got}"
            )


def directional_summary(rows):
    swaps = []
    for row in rows:
        for event in row["high_level_trace"]:
            if not event["goal_swap_changed_option"]:
                continue

            active = event["active_plan"]
            cf = event["counterfactual_plan"]

            # Normalize names so comparisons are always completion vs safety.
            if event["active_goal"] == "completion":
                completion = active
                safety = cf
            else:
                completion = cf
                safety = active

            cc = completion["score_components"]
            sc = safety["score_components"]

            swaps.append({
                "completion_success_minus_safety":
                    cc["success"] - sc["success"],
                "completion_collision_minus_safety":
                    cc["collision"] - sc["collision"],
                "completion_goal_value_minus_safety":
                    cc["goal_value"] - sc["goal_value"],
                "completion_reward_minus_safety":
                    cc["reward"] - sc["reward"],
                "completion_option":
                    completion["chosen_option"],
                "safety_option":
                    safety["chosen_option"],
            })

    def frac(pred):
        return (
            sum(1 for x in swaps if pred(x)) / len(swaps)
            if swaps else 0.0
        )

    return {
        "swap_events": len(swaps),
        "completion_has_ge_success_fraction":
            frac(lambda x: x["completion_success_minus_safety"] >= 0),
        "safety_has_le_collision_fraction":
            frac(lambda x: x["completion_collision_minus_safety"] >= 0),
        "mean_completion_success_minus_safety": (
            sum(x["completion_success_minus_safety"] for x in swaps)
            / len(swaps) if swaps else 0.0
        ),
        "mean_completion_collision_minus_safety": (
            sum(x["completion_collision_minus_safety"] for x in swaps)
            / len(swaps) if swaps else 0.0
        ),
        "examples": swaps[:100],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="state/goal_conditioned_option_planning_v1/result.json",
    )
    args = ap.parse_args()

    require_provenance()
    before = protected_hashes()

    print("GOAL_CONDITIONED_OPTION_PLANNING_V1")
    print("device: cpu")
    print("training_performed: false")
    print("autonomous_environment_control: true")
    print("teacher_driven_trajectories: false")
    print("v019_remains_closed: true")
    print("memory_checkpoint_preserved: true")
    print("v018_intent_preserved: true")
    print("seed_start:", SEED_START)
    print("episodes:", EPISODES)
    print("held_out_from_historical_reports: true")
    print("held_out_from_all_training_claimed: false")

    print("\n=== Historical planner baseline on same evaluation seeds ===")
    baseline = evaluate_hierarchical_options(
        CURRENT,
        episodes=EPISODES,
        steps=STEPS,
        seed_start=SEED_START,
        device="cpu",
        output=None,
        **PLANNER_ARGS,
    )
    if baseline.get("experiment", {}).get(
        "training_performed_during_evaluation"
    ) is not False:
        raise RuntimeError("baseline evaluator training attestation invalid")
    if baseline.get("checkpoint_sha256") != EXPECTED["current"]:
        raise RuntimeError("baseline checkpoint provenance mismatch")

    model, saved = load_checkpoint(CURRENT, "cpu")
    model.eval()
    metadata = dict(saved.get("metadata") or {})
    if metadata.get("option_cortex_version") != "latent-temporal-options-v1":
        raise RuntimeError("trained temporal options unavailable")

    completion_rows = []
    safety_rows = []

    for i in range(EPISODES):
        seed = SEED_START + i

        completion_rows.append(
            run_goal_episode(
                model,
                seed,
                COMPLETION_GOAL,
                counterfactual_goal=SAFETY_GOAL,
                max_steps=STEPS,
                device="cpu",
                **PLANNER_ARGS,
            )
        )
        safety_rows.append(
            run_goal_episode(
                model,
                seed,
                SAFETY_GOAL,
                counterfactual_goal=COMPLETION_GOAL,
                max_steps=STEPS,
                device="cpu",
                **PLANNER_ARGS,
            )
        )
        print(f"goal_conditioned_paired_episodes={i+1}/{EPISODES}")

    completion = summarize(completion_rows, STEPS)
    safety = summarize(safety_rows, STEPS)
    directional = directional_summary(
        completion_rows + safety_rows
    )

    pooled_opportunities = (
        completion["total_goal_swap_opportunities"]
        + safety["total_goal_swap_opportunities"]
    )
    pooled_changes = (
        completion["total_goal_swap_changed_options"]
        + safety["total_goal_swap_changed_options"]
    )
    pooled_change_rate = (
        pooled_changes / pooled_opportunities
        if pooled_opportunities else 0.0
    )

    after = protected_hashes()
    if before != after:
        raise RuntimeError("protected file changed during read-only evaluation")

    result = {
        "format": "continuous_brain.goal_conditioned_option_planning_v1",
        "software_status": "PASS",
        "scientific_admission_decided": False,
        "training_performed": False,
        "device": "cpu",
        "autonomous_environment_control": True,
        "teacher_driven_trajectories": False,
        "checkpoint_sha256": EXPECTED["current"],
        "memory_checkpoint_sha256": EXPECTED["memory"],
        "v018_intent_checkpoint_sha256": EXPECTED["intent"],
        "v019_closed_checkpoint_sha256": EXPECTED["v019"],
        "protected_unchanged": before == after,
        "seed_protocol": {
            "seed_start": SEED_START,
            "seed_end": SEED_START + EPISODES - 1,
            "episodes": EPISODES,
            "steps": STEPS,
            "historical_report_overlap": False,
            "all_training_holdout_claimed": False,
        },
        "goal_profiles": {
            "completion": COMPLETION_GOAL.as_dict(),
            "safety": SAFETY_GOAL.as_dict(),
        },
        "baseline_historical_planner": {
            "reactive": baseline["reactive"],
            "option_direct": baseline["option_direct_diagnostic"],
            "hierarchical": baseline["hierarchical_option_planning"],
            "historical_p4b_gate_on_these_new_seeds":
                baseline["p4b_passed"],
        },
        "completion_goal": completion,
        "safety_goal": safety,
        "causal_goal_swap": {
            "opportunities": pooled_opportunities,
            "changed_options": pooled_changes,
            "pooled_change_rate": pooled_change_rate,
            "directional": directional,
        },
        "episodes": {
            "completion": completion_rows,
            "safety": safety_rows,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nBASELINE")
    print(
        "reactive completion/collisions:",
        baseline["reactive"]["completion_rate"],
        baseline["reactive"]["mean_collisions"],
    )
    print(
        "direct completion/collisions:",
        baseline["option_direct_diagnostic"]["completion_rate"],
        baseline["option_direct_diagnostic"]["mean_collisions"],
    )
    print(
        "hierarchical completion/collisions:",
        baseline["hierarchical_option_planning"]["completion_rate"],
        baseline["hierarchical_option_planning"]["mean_collisions"],
    )

    print("\nGOAL_CONDITIONED")
    print(
        "completion completion/collisions:",
        completion["completion_rate"],
        completion["mean_collisions"],
    )
    print(
        "safety completion/collisions:",
        safety["completion_rate"],
        safety["mean_collisions"],
    )
    print(
        "goal_swap opportunities/changes/rate:",
        pooled_opportunities,
        pooled_changes,
        round(pooled_change_rate, 6),
    )
    print(
        "completion>=safety predicted success on swaps:",
        directional["completion_has_ge_success_fraction"],
    )
    print(
        "safety<=completion predicted collision on swaps:",
        directional["safety_has_le_collision_fraction"],
    )

    print("\nPROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("AUTONOMOUS_ENVIRONMENT_CONTROL: true")
    print("TEACHER_DRIVEN_TRAJECTORIES: false")
    print("V019_REMAINS_CLOSED: true")
    print("SCIENTIFIC_ADMISSION_DECIDED: false")
    print("GOAL_CONDITIONED_OPTION_PLANNING_RUN: PASS")
    print("result:", output)


if __name__ == "__main__":
    main()
