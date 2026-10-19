#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    AutobiographicalTrajectoryMemory,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.endogenous_goal_genesis_v1 import (
    EndogenousGoalGenerator,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from continuous_brain.self_development_hysteresis_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
)
from continuous_brain.self_development_loop_v1 import (
    reliability_from_memory,
)
from evaluate_endogenous_goal_genesis_v1 import (
    AUTOBIO_MODEL,
    MAX_LATE_DISCREPANCY_RATIO,
    MIN_CAPABILITY_GAIN,
    MIN_DISCREPANCY_REDUCTION,
    MIN_DONOR_FLIP,
    MIN_ERASED_NEUTRAL,
    MIN_GOAL_CAUSAL_PLAN,
    MIN_GOAL_COMPLETION,
    MIN_GOAL_GENESIS,
    MIN_GOAL_PERSISTENCE,
    MIN_RELOAD,
    SOURCE_V034_TEST,
    TRAJECTORY_MODEL,
    build_bootstrap_memory,
    mean,
    reliability_for_primary,
)
from evaluate_endogenous_goal_horizon_repair_v1 import (
    lifecycle_persistence_assay,
    run_subject_horizon16,
)

SOURCE_V0352 = Path(
    "state/endogenous_goal_horizon_repair_v1/result.json"
)
SOURCE_V0350 = Path(
    "state/endogenous_goal_genesis_v1/result.json"
)
SOURCE_V0351 = Path(
    "state/endogenous_goal_completion_diagnostic_v1/result.json"
)

TEST_IDS = range(100000, 100128)

PROTECTED = {
    "v034_test": SOURCE_V034_TEST,
    "v0350_result": SOURCE_V0350,
    "v0351_result": SOURCE_V0351,
    "v0352_result": SOURCE_V0352,
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


def frozen_test_donor_goal_assay(
    trajectory_model,
    autobiographical_model,
    generator,
):
    """
    Held-out donor-autobiography intervention using only TEST subjects.

    Neighboring even/odd TEST subjects have opposite autobiographical
    fragility. Swapping only the history must flip the endogenous target.
    Erased autobiography must produce no directional goal.
    """
    flips = []
    erased_none = []

    empty = AutobiographicalTrajectoryMemory()
    empty_rel = reliability_from_memory(
        autobiographical_model,
        empty,
    )
    empty_goal = generator.generate(
        empty_rel,
        generation=0,
        cycle=0,
    )

    ids = list(TEST_IDS)
    for i in range(0, len(ids), 2):
        even_memory = build_bootstrap_memory(
            trajectory_model,
            ids[i],
        )
        odd_memory = build_bootstrap_memory(
            trajectory_model,
            ids[i + 1],
        )

        even_rel = reliability_from_memory(
            autobiographical_model,
            even_memory,
        )
        odd_rel = reliability_from_memory(
            autobiographical_model,
            odd_memory,
        )

        even_goal = generator.generate(
            even_rel,
            generation=0,
            cycle=0,
        )
        odd_goal = generator.generate(
            odd_rel,
            generation=0,
            cycle=0,
        )

        flips.append(
            float(
                even_goal is not None
                and odd_goal is not None
                and even_goal.primary
                != odd_goal.primary
            )
        )
        erased_none.append(
            float(empty_goal is None)
        )

    return {
        "donor_history_flips_self_generated_goal":
            mean(flips),
        "erased_history_generates_no_developmental_goal":
            mean(erased_none),
    }


def frozen_test_goal_plan_assay(
    trajectory_model,
    autobiographical_model,
    generator,
):
    from continuous_brain.persistent_self_autobiographical_v1 import (
        ACTION_WORK_A,
        ACTION_WORK_B,
    )
    from evaluate_self_development_loop_v1 import plan

    correct = []

    for sid in TEST_IDS:
        memory = build_bootstrap_memory(
            trajectory_model,
            int(sid),
        )
        rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        goal = generator.generate(
            rel,
            generation=0,
            cycle=0,
        )
        if goal is None:
            correct.append(0.0)
            continue

        actions = plan(
            goal.primary,
            STRATEGY_SCAFFOLD,
        )
        expected_work = (
            ACTION_WORK_A
            if int(goal.primary) == 1
            else ACTION_WORK_B
        )
        work_actions = [
            a
            for a in actions
            if a in (
                ACTION_WORK_A,
                ACTION_WORK_B,
            )
        ]
        correct.append(
            float(
                len(work_actions) > 0
                and all(
                    a == expected_work
                    for a in work_actions
                )
            )
        )

    return {
        "self_generated_goal_controls_target_plan":
            mean(correct),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/endogenous_goal_genesis_test_v1",
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
    v0352 = json.loads(SOURCE_V0352.read_text())

    if v034.get("test_admission_passed") is not True:
        raise RuntimeError("v0.34.1 frozen TEST PASS required")
    if v0350.get("development_gate_passed") is not False:
        raise RuntimeError("v0.35.0 historical DEV FAIL required")
    if v0352.get("repair_dev_gate_passed") is not True:
        raise RuntimeError("v0.35.2 DEV PASS required")
    if v0352.get("v035_test_touched") is not False:
        raise RuntimeError("v0.35 TEST was already touched")

    if v0352["repair"]["goal_generator_changed"] is not False:
        raise RuntimeError("goal generator was changed")
    if v0352["repair"]["goal_completion_threshold_changed"] is not False:
        raise RuntimeError("completion threshold was changed")
    if float(v0352["repair"]["goal_target_delta"]) != 0.12:
        raise RuntimeError("unexpected goal target delta")
    if int(v0352["repair"]["new_max_target_episodes"]) != 16:
        raise RuntimeError("unexpected frozen horizon")

    before = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }

    print("ENDOGENOUS_GOAL_GENESIS_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.35")
    print("source_v0341_test_status: PASS")
    print("source_v0350_dev_status: FAIL")
    print(
        "source_v0351_diagnostic:",
        v0351["finding"],
    )
    print("source_v0352_dev_status: PASS")
    print("goal_generator_changed_from_v0352: false")
    print("goal_completion_threshold_changed_from_v0352: false")
    print("goal_target_delta: +0.12")
    print("max_target_episodes: 16")
    print("test_subjects: 100000-100127")
    print("test_opened_in_this_run: true")
    print("test_spent_after_run: true")
    print("test_tuning_after_run: forbidden")

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
    ids = list(TEST_IDS)
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
                f"goal_genesis_test_subjects={n}/128"
            )

    donor = frozen_test_donor_goal_assay(
        trajectory_model,
        autobiographical_model,
        generator,
    )
    goal_plan = frozen_test_goal_plan_assay(
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
            "protected artifact changed during frozen v0.35 TEST"
        )

    result = {
        "format":
            "continuous_brain.endogenous_goal_genesis_test_v1",
        "family": "v0.35",
        "test_admission_passed": bool(passed),
        "training_performed": False,
        "historical_status": {
            "v0341": "FROZEN_TEST_PASS",
            "v0350": "DEV_FAIL",
            "v0351":
                "EIGHT_EPISODE_HORIZON_TOO_SHORT_FOR_EXISTING_SELF_ESTIMATE_COMPLETION_CRITERION",
            "v0352": "DEV_PASS",
        },
        "frozen_mechanism": {
            "goal_generator_changed": False,
            "goal_completion_threshold_changed": False,
            "goal_target_delta": 0.12,
            "max_target_episodes": 16,
            "external_primary_goal_supplied_to_generator":
                False,
            "hidden_capability_supplied_to_generator":
                False,
            "subject_id_supplied_to_generator":
                False,
            "reward_label_supplied_to_generator":
                False,
        },
        "split": {
            "test": [100000, 100127],
            "test_opened_in_this_run": True,
            "test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "metrics": metrics,
        "donor_metrics": donor,
        "goal_plan_metrics": goal_plan,
        "persistence_metrics": persistence,
        "checks": checks,
        "protected_unchanged": True,
        "source_models_unchanged": True,
        "v0341_test_pass_preserved": True,
        "v035_test_spent": True,
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

    print("\nTEST_GOAL_GENESIS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTEST_DONOR_AUTOBIOGRAPHY")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nTEST_GOAL_PLAN_CAUSALITY")
    for k, v in goal_plan.items():
        print(f"{k}: {v}")

    print("\nTEST_PERSISTENCE")
    for k, v in persistence.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("TRAINING_PERFORMED: false")
    print("GOAL_GENERATOR_CHANGED: false")
    print("GOAL_COMPLETION_THRESHOLD_CHANGED: false")
    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_MODELS_UNCHANGED: true")
    print("V0341_TEST_PASS_PRESERVED: true")
    print("V035_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "ENDOGENOUS_GOAL_GENESIS_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "ENDOGENOUS_GOAL_GENESIS_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
