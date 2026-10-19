#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    AutobiographicalTrajectoryMemory,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.endogenous_goal_genesis_v1 import (
    PRIMARY_A,
    PRIMARY_B,
    EndogenousGoalGenerator,
    GoalLifecycle,
    load_goal_lifecycle,
    save_goal_lifecycle,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from continuous_brain.self_development_hysteresis_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
)
from continuous_brain.self_development_loop_v1 import (
    DevelopmentalState,
    reliability_from_memory,
)
from evaluate_endogenous_goal_genesis_v1 import (
    AUTOBIO_MODEL,
    DEV_IDS,
    DONOR_IDS,
    MIN_CAPABILITY_GAIN,
    MIN_DISCREPANCY_REDUCTION,
    MIN_DONOR_FLIP,
    MIN_ERASED_NEUTRAL,
    MIN_GOAL_CAUSAL_PLAN,
    MIN_GOAL_COMPLETION,
    MIN_GOAL_GENESIS,
    MIN_GOAL_PERSISTENCE,
    MIN_RELOAD,
    MAX_LATE_DISCREPANCY_RATIO,
    SOURCE_V034_TEST,
    TRAJECTORY_MODEL,
    actual_fragile_primary,
    build_bootstrap_memory,
    discrepancy_scalar,
    donor_goal_assay,
    execute_development_episode,
    goal_plan_causality_assay,
    initial_capabilities,
    mean,
    persistence_assay,
    reliability_for_primary,
    stable_primary,
)

SOURCE_V0350 = Path(
    "state/endogenous_goal_genesis_v1/result.json"
)
SOURCE_V0351 = Path(
    "state/endogenous_goal_completion_diagnostic_v1/result.json"
)

MAX_GOAL_EPISODES = 16

PROTECTED = {
    "v034_test": SOURCE_V034_TEST,
    "v0350_result": SOURCE_V0350,
    "v0351_result": SOURCE_V0351,
    "autobio_model": AUTOBIO_MODEL,
    "trajectory_model": TRAJECTORY_MODEL,
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "persistent_self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "persistent_self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
}


def run_subject_horizon16(
    trajectory_model,
    autobiographical_model,
    generator,
    subject_id: int,
):
    memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )
    lifecycle = GoalLifecycle()

    initial_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    goal = generator.generate(
        initial_rel,
        generation=lifecycle.generation,
        cycle=0,
    )
    if goal is None:
        raise RuntimeError(
            "normal developmental autobiography unexpectedly produced no goal"
        )
    lifecycle.begin(goal)

    generated_primary = int(goal.primary)
    evaluator_fragile = actual_fragile_primary(subject_id)
    evaluator_stable = stable_primary(subject_id)

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )
    initial_target_capability = float(
        state.capability_for(generated_primary)
    )

    bootstrap_target_discrepancy = mean(
        discrepancy_scalar(r)
        for r in memory.records
        if int(r.primary) == generated_primary
    )

    active_before_interrupt = copy.deepcopy(
        lifecycle.active_goal
    )
    interrupt_record, _ = execute_development_episode(
        trajectory_model,
        subject_id,
        -1,
        evaluator_stable,
        STRATEGY_DIRECT,
        state,
    )
    memory.append(interrupt_record)
    active_after_interrupt = copy.deepcopy(
        lifecycle.active_goal
    )
    goal_persisted = (
        active_before_interrupt
        == active_after_interrupt
    )

    target_discrepancies = []
    target_reliability_trace = []
    completion_cycle = None
    completion_reliability = None
    target_episodes_executed = 0

    for cycle in range(MAX_GOAL_EPISODES):
        if lifecycle.active_goal is None:
            break

        current_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        primary = lifecycle.active_goal.primary
        current_target_rel = reliability_for_primary(
            current_rel,
            primary,
        )
        target_reliability_trace.append(
            current_target_rel
        )

        if lifecycle.active_goal.target_met(
            current_target_rel
        ):
            completion_cycle = cycle
            completion_reliability = current_target_rel
            lifecycle.complete_active(cycle=cycle)
            break

        record, metrics = execute_development_episode(
            trajectory_model,
            subject_id,
            cycle,
            primary,
            STRATEGY_SCAFFOLD,
            state,
        )
        memory.append(record)
        target_discrepancies.append(
            float(metrics["discrepancy"])
        )
        target_episodes_executed += 1
        state.cycle += 1

    # One decision/read after the final allowed target episode.
    if lifecycle.active_goal is not None:
        current_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        primary = lifecycle.active_goal.primary
        current_target_rel = reliability_for_primary(
            current_rel,
            primary,
        )
        target_reliability_trace.append(
            current_target_rel
        )
        if lifecycle.active_goal.target_met(
            current_target_rel
        ):
            completion_cycle = MAX_GOAL_EPISODES
            completion_reliability = current_target_rel
            lifecycle.complete_active(
                cycle=MAX_GOAL_EPISODES
            )

    completed = len(lifecycle.completed_goals) >= 1

    final_target_capability = float(
        state.capability_for(generated_primary)
    )
    capability_gain = (
        final_target_capability
        - initial_target_capability
    )

    if target_discrepancies:
        late_discrepancy = mean(
            target_discrepancies[-2:]
        )
    else:
        late_discrepancy = bootstrap_target_discrepancy

    discrepancy_ratio = (
        late_discrepancy
        / max(bootstrap_target_discrepancy, 1e-6)
    )

    next_goal = None
    next_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    if lifecycle.active_goal is None:
        next_goal = generator.generate(
            next_rel,
            generation=lifecycle.generation,
            cycle=state.cycle,
        )

    return {
        "subject_id": int(subject_id),
        "memory": memory,
        "lifecycle": lifecycle,
        "generated_primary": generated_primary,
        "evaluator_fragile_primary": evaluator_fragile,
        "goal_matches_fragile":
            generated_primary == evaluator_fragile,
        "goal_persisted_through_interrupt":
            bool(goal_persisted),
        "completed": bool(completed),
        "completion_cycle": completion_cycle,
        "completion_reliability":
            completion_reliability,
        "target_episodes_executed":
            target_episodes_executed,
        "target_reliability_trace":
            target_reliability_trace,
        "capability_gain": capability_gain,
        "bootstrap_target_discrepancy":
            bootstrap_target_discrepancy,
        "late_target_discrepancy":
            late_discrepancy,
        "late_to_bootstrap_discrepancy_ratio":
            discrepancy_ratio,
        "next_goal_primary":
            (
                int(next_goal.primary)
                if next_goal is not None
                else None
            ),
    }


def lifecycle_persistence_assay(
    autobiographical_model,
    subject_result,
    state_dir,
):
    path = state_dir / "goal_lifecycle_probe.pt"
    lifecycle = subject_result["lifecycle"]
    memory = subject_result["memory"]

    before_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )

    save_goal_lifecycle(
        path,
        lifecycle,
        metadata={
            "reliability": before_rel.tolist(),
        },
    )
    loaded, payload = load_goal_lifecycle(path)

    metadata_rel = torch.tensor(
        payload["metadata"]["reliability"],
        dtype=before_rel.dtype,
    )

    return {
        "goal_lifecycle_exact_after_reload":
            float(loaded == lifecycle),
        "goal_context_reliability_exact_after_reload":
            float(
                torch.equal(
                    before_rel,
                    metadata_rel,
                )
            ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/endogenous_goal_horizon_repair_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    v034 = json.loads(SOURCE_V034_TEST.read_text())
    v0350 = json.loads(SOURCE_V0350.read_text())
    v0351 = json.loads(SOURCE_V0351.read_text())

    if v034.get("test_admission_passed") is not True:
        raise RuntimeError("v0.34.1 TEST PASS required")
    if v0350.get("development_gate_passed") is not False:
        raise RuntimeError("v0.35.0 formal DEV FAIL required")

    failed = sorted(
        k for k, v in v0350["checks"].items()
        if not bool(v)
    )
    if failed != [
        "self_generated_goal_completes_from_self_estimate"
    ]:
        raise RuntimeError(
            f"unexpected v0.35.0 failed gates: {failed}"
        )

    if v0351.get("finding") != (
        "EIGHT_EPISODE_HORIZON_TOO_SHORT_FOR_EXISTING_"
        "SELF_ESTIMATE_COMPLETION_CRITERION"
    ):
        raise RuntimeError(
            "v0.35.1 did not authorize the 16-episode horizon repair"
        )

    c16 = (
        v0351["metrics"]
        ["completion_rate_by_episode"]
        ["16"]
    )
    if float(c16) < MIN_GOAL_COMPLETION:
        raise RuntimeError(
            "diagnostic 16-episode completion did not clear the original gate"
        )

    if v0350.get("v035_test_touched") is not False:
        raise RuntimeError("v0.35 TEST already touched")

    before = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }

    print("ENDOGENOUS_GOAL_HORIZON_REPAIR_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.35")
    print("source_v0350_formal_status: FAIL")
    print(
        "source_v0351_finding:",
        v0351["finding"],
    )
    print("repair_scope: developmental_horizon_only")
    print("goal_generator_changed: false")
    print("goal_completion_threshold_changed: false")
    print("goal_target_delta: +0.12")
    print("old_max_target_episodes: 8")
    print("new_max_target_episodes: 16")
    print(
        "horizon_source:",
        "first preregistered diagnostic checkpoint clearing original 0.80 completion gate",
    )
    print("development_window_reused: 90000-90127")
    print("new_development_window_consumed: false")
    print("test_subjects: 100000-100127")
    print("test_touched: false")

    trajectory_model, _ = load_trajectory_model(
        TRAJECTORY_MODEL,
        "cpu",
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    autobiographical_model, _ = load_neutral_model(
        AUTOBIO_MODEL,
        "cpu",
    )
    for p in autobiographical_model.parameters():
        p.requires_grad_(False)
    autobiographical_model.eval()

    generator = EndogenousGoalGenerator()

    subjects = []
    ids = list(DEV_IDS)
    for n, sid in enumerate(ids, 1):
        subjects.append(
            run_subject_horizon16(
                trajectory_model,
                autobiographical_model,
                generator,
                int(sid),
            )
        )
        if n == 1 or n % 32 == 0 or n == len(ids):
            print(
                f"horizon_repair_subjects={n}/128"
            )

    donor = donor_goal_assay(
        trajectory_model,
        autobiographical_model,
        generator,
    )
    goal_plan = goal_plan_causality_assay(
        trajectory_model,
        autobiographical_model,
        generator,
    )
    persistence = lifecycle_persistence_assay(
        autobiographical_model,
        subjects[0],
        state_dir,
    )

    goal_match_rate = mean(
        float(s["goal_matches_fragile"])
        for s in subjects
    )
    interruption_persistence_rate = mean(
        float(
            s["goal_persisted_through_interrupt"]
        )
        for s in subjects
    )
    completion_rate = mean(
        float(s["completed"])
        for s in subjects
    )
    capability_gain_mean = mean(
        s["capability_gain"]
        for s in subjects
    )
    discrepancy_reduction_rate = mean(
        float(
            s["late_to_bootstrap_discrepancy_ratio"]
            <= MAX_LATE_DISCREPANCY_RATIO
        )
        for s in subjects
    )
    discrepancy_ratio_mean = mean(
        s["late_to_bootstrap_discrepancy_ratio"]
        for s in subjects
    )
    episodes_mean = mean(
        s["target_episodes_executed"]
        for s in subjects
    )
    next_goal_generated_rate = mean(
        float(s["next_goal_primary"] is not None)
        for s in subjects
    )

    metrics = {
        "self_generated_goal_matches_autobiographical_fragility":
            goal_match_rate,
        "active_goal_persists_through_unrelated_interrupt":
            interruption_persistence_rate,
        "self_generated_goal_completion_rate":
            completion_rate,
        "mean_target_episodes_executed":
            episodes_mean,
        "mean_target_capability_gain":
            capability_gain_mean,
        "discrepancy_reduction_rate":
            discrepancy_reduction_rate,
        "mean_late_to_bootstrap_discrepancy_ratio":
            discrepancy_ratio_mean,
        "post_completion_next_goal_generated_rate":
            next_goal_generated_rate,
    }

    checks = {
        "endogenous_goal_genesis":
            goal_match_rate >= MIN_GOAL_GENESIS,
        "donor_autobiography_changes_generated_goal":
            donor[
                "donor_history_flips_self_generated_goal"
            ] >= MIN_DONOR_FLIP,
        "erased_autobiography_has_no_directional_goal":
            donor[
                "erased_history_generates_no_developmental_goal"
            ] >= MIN_ERASED_NEUTRAL,
        "self_generated_goal_persists":
            interruption_persistence_rate
            >= MIN_GOAL_PERSISTENCE,
        "self_generated_goal_controls_plan":
            goal_plan[
                "self_generated_goal_controls_target_plan"
            ] >= MIN_GOAL_CAUSAL_PLAN,
        "self_generated_goal_completes_from_self_estimate":
            completion_rate >= MIN_GOAL_COMPLETION,
        "self_generated_goal_develops_capability":
            capability_gain_mean >= MIN_CAPABILITY_GAIN,
        "self_generated_goal_reduces_discrepancy":
            discrepancy_reduction_rate
            >= MIN_DISCREPANCY_REDUCTION,
        "goal_lifecycle_persists":
            persistence[
                "goal_lifecycle_exact_after_reload"
            ] >= MIN_RELOAD,
        "goal_context_persists":
            persistence[
                "goal_context_reliability_exact_after_reload"
            ] >= MIN_RELOAD,
    }

    passed = all(checks.values())

    after = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during v0.35.2 repair"
        )

    result = {
        "format":
            "continuous_brain.endogenous_goal_horizon_repair_v1",
        "family": "v0.35",
        "repair_dev_gate_passed": bool(passed),
        "training_performed": False,
        "historical_status": {
            "v0350": "DEV_FAIL",
            "v0351":
                "EIGHT_EPISODE_HORIZON_TOO_SHORT_FOR_EXISTING_SELF_ESTIMATE_COMPLETION_CRITERION",
        },
        "repair": {
            "scope":
                "developmental_horizon_only",
            "goal_generator_changed":
                False,
            "goal_completion_threshold_changed":
                False,
            "goal_target_delta":
                0.12,
            "old_max_target_episodes":
                8,
            "new_max_target_episodes":
                16,
            "horizon_source":
                "first preregistered diagnostic checkpoint clearing original 0.80 completion gate",
        },
        "split": {
            "reused_dev": [90000, 90127],
            "test": [100000, 100127],
            "test_touched": False,
        },
        "metrics": metrics,
        "donor_metrics": donor,
        "goal_plan_metrics": goal_plan,
        "persistence_metrics": persistence,
        "checks": checks,
        "protected_unchanged": True,
        "v0341_test_pass_preserved": True,
        "v035_test_touched": False,
        "scope": {
            "functional_endogenous_goal_genesis_claimed_if_pass":
                True,
            "open_ended_goal_space_claimed":
                False,
            "subjective_experience_claimed":
                False,
            "sentience_claimed":
                False,
            "consciousness_claimed":
                False,
        },
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nHORIZON_REPAIR")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nDONOR_AUTOBIOGRAPHY")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nGOAL_PLAN_CAUSALITY")
    for k, v in goal_plan.items():
        print(f"{k}: {v}")

    print("\nPERSISTENCE")
    for k, v in persistence.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("TRAINING_PERFORMED: false")
    print("GOAL_GENERATOR_CHANGED: false")
    print("GOAL_COMPLETION_THRESHOLD_CHANGED: false")
    print("PROTECTED_UNCHANGED: true")
    print("V0341_TEST_PASS_PRESERVED: true")
    print("V035_TEST_TOUCHED: false")
    print(
        "ENDOGENOUS_GOAL_HORIZON_REPAIR_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "ENDOGENOUS_GOAL_HORIZON_REPAIR_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
