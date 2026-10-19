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
from continuous_brain.recurrent_self_trajectory_v1 import (
    load_trajectory_model,
)
from continuous_brain.self_development_loop_v1 import (
    reliability_from_memory,
)
from evaluate_endogenous_goal_genesis_v1 import (
    build_bootstrap_memory,
    mean,
)
from evaluate_recursive_autobiographical_curriculum_v1 import (
    AUTOBIO_MODEL,
    TRAJECTORY_MODEL,
    MAX_FIRST_GOAL_EPISODES,
    SECOND_GOAL_PROBE_EPISODES,
    MIN_FIRST_GOAL_GENESIS,
    MIN_FIRST_GOAL_COMPLETION,
    MIN_SECOND_GOAL_GENERATION,
    MIN_CURRICULUM_ADVANCE,
    MIN_FROZEN_HISTORY_REPEATS,
    MIN_UPDATED_VS_FROZEN_DIVERGENCE,
    MIN_DONOR_FIRST_GOAL_FLIP,
    MIN_ERASED_NEUTRAL,
    MIN_SECOND_GOAL_PLAN_CAUSALITY,
    MIN_SECOND_GOAL_CAPABILITY_EFFECT,
    MIN_PERSISTENCE,
    run_recursive_subject,
    persistence_assay,
)

SOURCE_V0360 = Path(
    "state/recursive_autobiographical_curriculum_v1/result.json"
)
SOURCE_V0353 = Path(
    "state/endogenous_goal_genesis_test_v1/result.json"
)

TEST_IDS = range(120000, 120128)

PROTECTED = {
    "v0360_result": SOURCE_V0360,
    "v0353_result": SOURCE_V0353,
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


def frozen_test_donor_assay(
    trajectory_model,
    autobiographical_model,
    generator,
):
    """
    Held-out donor/neutral assay using only v0.36 TEST histories.

    Neighboring even/odd subject IDs have opposite autobiographical fragility.
    Swapping only autobiography must flip the first self-authored goal.
    Empty autobiography must generate no directional goal.
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
                and int(even_goal.primary)
                != int(odd_goal.primary)
            )
        )
        erased_none.append(
            float(empty_goal is None)
        )

    return {
        "donor_history_flips_first_goal":
            mean(flips),
        "erased_history_generates_no_goal":
            mean(erased_none),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recursive_autobiographical_curriculum_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    v0360 = json.loads(
        SOURCE_V0360.read_text()
    )
    v0353 = json.loads(
        SOURCE_V0353.read_text()
    )

    if v0360.get(
        "development_gate_passed"
    ) is not True:
        raise RuntimeError(
            "v0.36.0 DEV PASS required"
        )
    if v0360.get(
        "v036_test_touched"
    ) is not False:
        raise RuntimeError(
            "v0.36 TEST was already touched"
        )
    if v0353.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.35.3 frozen TEST PASS required"
        )
    if (
        v0353.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.35 TEST must remain closed/spent"
        )

    arch = v0360.get("architecture", {})
    if int(
        arch.get("first_goal_horizon", -1)
    ) != MAX_FIRST_GOAL_EPISODES:
        raise RuntimeError(
            "v0.36 first-goal horizon mismatch"
        )
    if int(
        arch.get("second_goal_probe_episodes", -1)
    ) != SECOND_GOAL_PROBE_EPISODES:
        raise RuntimeError(
            "v0.36 second-goal probe mismatch"
        )
    if arch.get(
        "frozen_bootstrap_counterfactual"
    ) is not True:
        raise RuntimeError(
            "v0.36 frozen-history counterfactual required"
        )

    before = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }

    print("RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.36")
    print("source_v0353_test_status: PASS")
    print("source_v0360_dev_status: PASS")
    print(
        "curriculum_source:",
        "updated autobiographical self-estimate",
    )
    print("external_second_goal_supplied: false")
    print("frozen_history_counterfactual_enabled: true")
    print(
        "first_goal_horizon:",
        MAX_FIRST_GOAL_EPISODES,
    )
    print(
        "second_goal_probe_episodes:",
        SECOND_GOAL_PROBE_EPISODES,
    )
    print(
        "curriculum_advancement_definition:",
        "different primary OR same primary with target >= previous target + 0.05",
    )
    print("test_subjects: 120000-120127")
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

    rows = []
    ids = list(TEST_IDS)
    for n, sid in enumerate(ids, 1):
        rows.append(
            run_recursive_subject(
                trajectory_model,
                autobiographical_model,
                generator,
                int(sid),
            )
        )
        if (
            n == 1
            or n % 32 == 0
            or n == len(ids)
        ):
            print(
                f"recursive_curriculum_test_subjects={n}/128"
            )

    donor = frozen_test_donor_assay(
        trajectory_model,
        autobiographical_model,
        generator,
    )

    completers = [
        r for r in rows
        if r.get("first_completed", False)
    ]
    if not completers:
        raise RuntimeError(
            "no first-goal completers in frozen TEST"
        )

    first_genesis_rate = mean(
        float(
            r.get("first_goal_generated", False)
            and r.get(
                "first_goal_matches_autobiographical_fragility",
                False,
            )
        )
        for r in rows
    )
    first_completion_rate = mean(
        float(r.get("first_completed", False))
        for r in rows
    )

    second_generation_conditional = mean(
        float(r["second_goal_generated"])
        for r in completers
    )
    advance_conditional = mean(
        float(r["curriculum_advanced"])
        for r in completers
    )
    frozen_repeat_conditional = mean(
        float(
            r["frozen_history_repeats_first_goal"]
        )
        for r in completers
    )
    divergence_conditional = mean(
        float(
            r[
                "updated_vs_frozen_second_goal_diverges"
            ]
        )
        for r in completers
    )
    second_plan_conditional = mean(
        float(r["second_goal_controls_plan"])
        for r in completers
    )
    second_capability_effect = mean(
        float(r["second_goal_changes_capability"])
        for r in completers
    )
    second_capability_gain_mean = mean(
        r["second_goal_capability_gain"]
        for r in completers
    )

    persistence = persistence_assay(
        completers[0],
        state_dir,
    )

    metrics = {
        "first_goal_genesis_rate":
            first_genesis_rate,
        "first_goal_completion_rate":
            first_completion_rate,
        "second_goal_generation_given_first_completion":
            second_generation_conditional,
        "curriculum_advancement_given_first_completion":
            advance_conditional,
        "frozen_bootstrap_history_repeats_first_goal":
            frozen_repeat_conditional,
        "updated_vs_frozen_second_goal_divergence":
            divergence_conditional,
        "second_goal_controls_plan":
            second_plan_conditional,
        "second_goal_changes_target_capability":
            second_capability_effect,
        "mean_second_goal_capability_gain":
            second_capability_gain_mean,
    }

    checks = {
        "first_goal_endogenous":
            first_genesis_rate
            >= MIN_FIRST_GOAL_GENESIS,
        "first_goal_completes":
            first_completion_rate
            >= MIN_FIRST_GOAL_COMPLETION,
        "updated_autobiography_authors_second_goal":
            second_generation_conditional
            >= MIN_SECOND_GOAL_GENERATION,
        "updated_autobiography_advances_curriculum":
            advance_conditional
            >= MIN_CURRICULUM_ADVANCE,
        "frozen_autobiography_repeats_old_goal":
            frozen_repeat_conditional
            >= MIN_FROZEN_HISTORY_REPEATS,
        "updated_history_causes_new_curriculum":
            divergence_conditional
            >= MIN_UPDATED_VS_FROZEN_DIVERGENCE,
        "donor_history_changes_first_goal":
            donor[
                "donor_history_flips_first_goal"
            ] >= MIN_DONOR_FIRST_GOAL_FLIP,
        "erased_history_has_no_directional_goal":
            donor[
                "erased_history_generates_no_goal"
            ] >= MIN_ERASED_NEUTRAL,
        "second_goal_controls_behavior":
            second_plan_conditional
            >= MIN_SECOND_GOAL_PLAN_CAUSALITY,
        "second_goal_changes_capability":
            second_capability_effect
            >= MIN_SECOND_GOAL_CAPABILITY_EFFECT,
        "curriculum_persists":
            persistence[
                "curriculum_snapshot_exact_after_reload"
            ] >= MIN_PERSISTENCE,
    }

    passed = all(checks.values())

    after = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during frozen v0.36 TEST"
        )

    result = {
        "format":
            "continuous_brain.recursive_autobiographical_curriculum_test_v1",
        "family": "v0.36",
        "test_admission_passed": bool(passed),
        "training_performed": False,
        "historical_status": {
            "v0353": "FROZEN_TEST_PASS",
            "v0360": "DEV_PASS",
        },
        "frozen_mechanism": {
            "external_second_goal_supplied":
                False,
            "goal_source":
                "updated frozen autobiographical self-estimate",
            "first_goal_horizon":
                MAX_FIRST_GOAL_EPISODES,
            "second_goal_probe_episodes":
                SECOND_GOAL_PROBE_EPISODES,
            "frozen_bootstrap_counterfactual":
                True,
            "curriculum_advancement_definition":
                "different primary OR same primary with target >= previous target + 0.05",
            "thresholds_changed_from_v0360":
                False,
            "neural_models_changed":
                False,
        },
        "split": {
            "test": [120000, 120127],
            "test_opened_in_this_run":
                True,
            "test_spent":
                True,
            "test_tuning_after_run_forbidden":
                True,
        },
        "metrics": metrics,
        "donor_metrics": donor,
        "persistence_metrics": persistence,
        "checks": checks,
        "protected_unchanged": True,
        "source_models_unchanged": True,
        "v0353_test_pass_preserved": True,
        "v036_test_spent": True,
        "scope": {
            "functional_recursive_self_authored_curriculum_claimed_if_pass":
                True,
            "open_ended_semantic_goal_space_claimed":
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

    print("\nTEST_CURRICULUM")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTEST_DONOR_AND_NEUTRAL")
    for k, v in donor.items():
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
    print("THRESHOLDS_CHANGED_FROM_V0360: false")
    print("NEURAL_MODELS_CHANGED: false")
    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_MODELS_UNCHANGED: true")
    print("V0353_TEST_PASS_PRESERVED: true")
    print("V036_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
