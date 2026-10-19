#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    GoalLifecycle,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recursive_autobiographical_curriculum_v1 import (
    CurriculumSnapshot,
    goal_is_advanced,
    load_curriculum_snapshot,
    same_goal_semantics,
    save_curriculum_snapshot,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    load_trajectory_model,
)
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
    TRAJECTORY_MODEL,
    actual_fragile_primary,
    build_bootstrap_memory,
    discrepancy_scalar,
    execute_development_episode,
    initial_capabilities,
    mean,
    reliability_for_primary,
    stable_primary,
)
from evaluate_self_development_loop_v1 import plan

SOURCE_V0353 = Path(
    "state/endogenous_goal_genesis_test_v1/result.json"
)

DEV_IDS = range(110000, 110128)
TEST_IDS = range(120000, 120128)
DONOR_IDS = range(111000, 111128)

MAX_FIRST_GOAL_EPISODES = 16
SECOND_GOAL_PROBE_EPISODES = 4

MIN_FIRST_GOAL_GENESIS = .95
MIN_FIRST_GOAL_COMPLETION = .80
MIN_SECOND_GOAL_GENERATION = .95
MIN_CURRICULUM_ADVANCE = .95
MIN_FROZEN_HISTORY_REPEATS = .95
MIN_UPDATED_VS_FROZEN_DIVERGENCE = .95
MIN_DONOR_FIRST_GOAL_FLIP = .95
MIN_ERASED_NEUTRAL = 1.0
MIN_SECOND_GOAL_PLAN_CAUSALITY = 1.0
MIN_SECOND_GOAL_CAPABILITY_EFFECT = .80
MIN_PERSISTENCE = 1.0

PROTECTED = {
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


def complete_first_goal(
    trajectory_model,
    autobiographical_model,
    generator,
    subject_id: int,
):
    bootstrap_memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )
    memory = copy.deepcopy(bootstrap_memory)
    bootstrap_rel = reliability_from_memory(
        autobiographical_model,
        bootstrap_memory,
    )

    lifecycle = GoalLifecycle()
    first_goal = generator.generate(
        bootstrap_rel,
        generation=0,
        cycle=0,
    )
    if first_goal is None:
        return None
    lifecycle.begin(first_goal)

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )

    # Preserve the admitted interruption semantics from v0.35.
    interrupt_record, _ = execute_development_episode(
        trajectory_model,
        subject_id,
        -1,
        stable_primary(subject_id),
        STRATEGY_DIRECT,
        state,
    )
    memory.append(interrupt_record)

    first_target_discrepancies = []

    for cycle in range(MAX_FIRST_GOAL_EPISODES):
        if lifecycle.active_goal is None:
            break

        current_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        current_target_rel = reliability_for_primary(
            current_rel,
            lifecycle.active_goal.primary,
        )

        if lifecycle.active_goal.target_met(
            current_target_rel
        ):
            lifecycle.complete_active(cycle)
            break

        record, metrics = execute_development_episode(
            trajectory_model,
            subject_id,
            cycle,
            lifecycle.active_goal.primary,
            STRATEGY_SCAFFOLD,
            state,
        )
        memory.append(record)
        first_target_discrepancies.append(
            float(metrics["discrepancy"])
        )
        state.cycle += 1

    if lifecycle.active_goal is not None:
        current_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        current_target_rel = reliability_for_primary(
            current_rel,
            lifecycle.active_goal.primary,
        )
        if lifecycle.active_goal.target_met(
            current_target_rel
        ):
            lifecycle.complete_active(
                MAX_FIRST_GOAL_EPISODES
            )

    first_completed = (
        len(lifecycle.completed_goals) >= 1
    )

    return {
        "bootstrap_memory":
            bootstrap_memory,
        "memory":
            memory,
        "bootstrap_rel":
            bootstrap_rel,
        "lifecycle":
            lifecycle,
        "first_goal":
            first_goal,
        "first_completed":
            first_completed,
        "state":
            state,
        "first_target_discrepancies":
            first_target_discrepancies,
    }


def run_recursive_subject(
    trajectory_model,
    autobiographical_model,
    generator,
    subject_id: int,
):
    first = complete_first_goal(
        trajectory_model,
        autobiographical_model,
        generator,
        subject_id,
    )
    if first is None:
        return {
            "subject_id": int(subject_id),
            "first_goal_generated": False,
            "first_completed": False,
        }

    first_goal = first["first_goal"]
    first_completed = bool(first["first_completed"])
    memory = first["memory"]
    bootstrap_memory = first["bootstrap_memory"]
    state = first["state"]
    lifecycle = first["lifecycle"]

    second_goal = None
    frozen_second_goal = None
    advanced = False
    frozen_repeats = False
    updated_vs_frozen_diverges = False
    second_plan_correct = False
    second_capability_gain = 0.0
    second_discrepancies = []

    if first_completed:
        updated_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        second_goal = generator.generate(
            updated_rel,
            generation=lifecycle.generation,
            cycle=state.cycle,
        )

        frozen_rel = reliability_from_memory(
            autobiographical_model,
            bootstrap_memory,
        )
        frozen_second_goal = generator.generate(
            frozen_rel,
            generation=lifecycle.generation,
            cycle=state.cycle,
        )

        advanced = goal_is_advanced(
            first_goal,
            second_goal,
        )
        frozen_repeats = same_goal_semantics(
            first_goal,
            frozen_second_goal,
        )
        updated_vs_frozen_diverges = (
            not same_goal_semantics(
                second_goal,
                frozen_second_goal,
            )
        )

        if second_goal is not None:
            actions = plan(
                second_goal.primary,
                STRATEGY_SCAFFOLD,
            )
            from continuous_brain.persistent_self_autobiographical_v1 import (
                ACTION_WORK_A,
                ACTION_WORK_B,
            )
            expected_work = (
                ACTION_WORK_A
                if int(second_goal.primary) == 1
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
            second_plan_correct = (
                len(work_actions) > 0
                and all(
                    a == expected_work
                    for a in work_actions
                )
            )

            initial_second_capability = float(
                state.capability_for(
                    second_goal.primary
                )
            )

            for j in range(
                SECOND_GOAL_PROBE_EPISODES
            ):
                record, metrics = execute_development_episode(
                    trajectory_model,
                    subject_id,
                    100 + j,
                    second_goal.primary,
                    STRATEGY_SCAFFOLD,
                    state,
                )
                memory.append(record)
                second_discrepancies.append(
                    float(metrics["discrepancy"])
                )
                state.cycle += 1

            second_capability_gain = (
                float(
                    state.capability_for(
                        second_goal.primary
                    )
                )
                - initial_second_capability
            )

    return {
        "subject_id":
            int(subject_id),
        "first_goal_generated":
            True,
        "first_goal_primary":
            int(first_goal.primary),
        "first_goal_matches_autobiographical_fragility":
            int(first_goal.primary)
            == actual_fragile_primary(subject_id),
        "first_completed":
            first_completed,
        "second_goal_generated":
            second_goal is not None,
        "second_goal_primary":
            (
                int(second_goal.primary)
                if second_goal is not None
                else None
            ),
        "second_goal_target":
            (
                float(second_goal.target_reliability)
                if second_goal is not None
                else None
            ),
        "first_goal_target":
            float(first_goal.target_reliability),
        "curriculum_advanced":
            bool(advanced),
        "frozen_history_repeats_first_goal":
            bool(frozen_repeats),
        "updated_vs_frozen_second_goal_diverges":
            bool(updated_vs_frozen_diverges),
        "second_goal_controls_plan":
            bool(second_plan_correct),
        "second_goal_capability_gain":
            float(second_capability_gain),
        "second_goal_changes_capability":
            bool(second_capability_gain > 1e-9),
        "second_goal_discrepancy_mean":
            (
                mean(second_discrepancies)
                if second_discrepancies
                else None
            ),
        "first_goal":
            first_goal,
        "second_goal":
            second_goal,
        "frozen_second_goal":
            frozen_second_goal,
    }


def donor_assay(
    trajectory_model,
    autobiographical_model,
    generator,
):
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

    ids = list(DONOR_IDS)
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


def persistence_assay(
    subject_result,
    state_dir,
):
    path = state_dir / "curriculum_snapshot.pt"

    snapshot = CurriculumSnapshot(
        first_goal=subject_result["first_goal"],
        second_goal=subject_result["second_goal"],
        frozen_history_second_goal=subject_result[
            "frozen_second_goal"
        ],
    )
    save_curriculum_snapshot(
        path,
        snapshot,
    )
    loaded, _ = load_curriculum_snapshot(path)

    return {
        "curriculum_snapshot_exact_after_reload":
            float(loaded == snapshot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recursive_autobiographical_curriculum_v1",
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

    v0353 = json.loads(
        SOURCE_V0353.read_text()
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
            "v0.35 TEST must be closed/spent"
        )

    before = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }

    print("RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.36")
    print("source_v0353_test_status: PASS")
    print(
        "curriculum_source:",
        "updated autobiographical self-estimate",
    )
    print(
        "external_second_goal_supplied: false"
    )
    print(
        "frozen_history_counterfactual_enabled: true"
    )
    print("dev_subjects: 110000-110127")
    print("test_subjects: 120000-120127")
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

    rows = []
    ids = list(DEV_IDS)
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
                f"recursive_curriculum_subjects={n}/128"
            )

    donor = donor_assay(
        trajectory_model,
        autobiographical_model,
        generator,
    )

    completers = [
        r for r in rows
        if r.get("first_completed", False)
    ]

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
            donor["donor_history_flips_first_goal"]
            >= MIN_DONOR_FIRST_GOAL_FLIP,
        "erased_history_has_no_directional_goal":
            donor["erased_history_generates_no_goal"]
            >= MIN_ERASED_NEUTRAL,
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
            "protected artifact changed during v0.36 DEV"
        )

    result = {
        "format":
            "continuous_brain.recursive_autobiographical_curriculum_v1",
        "family": "v0.36",
        "development_gate_passed": bool(passed),
        "training_performed": False,
        "source_v0353_test_status": "PASS",
        "architecture": {
            "external_second_goal_supplied":
                False,
            "goal_source":
                "updated frozen autobiographical self-estimate",
            "first_goal_horizon":
                16,
            "second_goal_probe_episodes":
                SECOND_GOAL_PROBE_EPISODES,
            "frozen_bootstrap_counterfactual":
                True,
            "curriculum_advancement_definition":
                "different primary OR same primary with target >= previous target + 0.05",
        },
        "split": {
            "dev": [110000, 110127],
            "test": [120000, 120127],
            "test_touched": False,
            "donor_probe": [111000, 111127],
        },
        "metrics": metrics,
        "donor_metrics": donor,
        "persistence_metrics": persistence,
        "checks": checks,
        "protected_unchanged": True,
        "v0353_test_pass_preserved": True,
        "v036_test_touched": False,
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

    print("\nCURRICULUM")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nDONOR_AND_NEUTRAL")
    for k, v in donor.items():
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
    print("PROTECTED_UNCHANGED: true")
    print("V0353_TEST_PASS_PRESERVED: true")
    print("V036_TEST_TOUCHED: false")
    print(
        "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
