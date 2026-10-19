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
from evaluate_self_development_hysteresis_v1 import (
    AUTOBIO_MODEL,
    TRAJECTORY_MODEL,
    build_bootstrap_memory,
    discrepancy_scalar,
    execute_development_episode,
    initial_capabilities,
    mean,
    primary_index,
    reliability_for_primary,
)

SOURCE_V034_TEST = Path(
    "state/event_consistent_self_development_test_v1/result.json"
)
AUTOBIO_TEST = Path(
    "state/autobiographical_self_discrepancy_neutral_test_v1/result.json"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

DEV_IDS = range(90000, 90128)
TEST_IDS = range(100000, 100128)
DONOR_IDS = range(91000, 91128)

MAX_GOAL_EPISODES = 8

MIN_GOAL_GENESIS = .95
MIN_DONOR_FLIP = .95
MIN_ERASED_NEUTRAL = 1.0
MIN_GOAL_PERSISTENCE = 1.0
MIN_GOAL_CAUSAL_PLAN = 1.0
MIN_GOAL_COMPLETION = .80
MIN_CAPABILITY_GAIN = .35
MAX_LATE_DISCREPANCY_RATIO = .75
MIN_DISCREPANCY_REDUCTION = .80
MIN_RELOAD = 1.0

PROTECTED = {
    "v034_test": SOURCE_V034_TEST,
    "autobio_model": AUTOBIO_MODEL,
    "autobio_test": AUTOBIO_TEST,
    "trajectory_model": TRAJECTORY_MODEL,
    "trajectory_test": TRAJECTORY_TEST,
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


def actual_fragile_primary(subject_id: int) -> int:
    return PRIMARY_A if int(subject_id) % 2 == 0 else PRIMARY_B


def stable_primary(subject_id: int) -> int:
    return (
        PRIMARY_B
        if actual_fragile_primary(subject_id) == PRIMARY_A
        else PRIMARY_A
    )


def create_goal_from_memory(
    generator,
    autobiographical_model,
    memory,
    lifecycle,
    cycle,
):
    rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    goal = generator.generate(
        rel,
        generation=lifecycle.generation,
        cycle=cycle,
    )
    if goal is not None:
        lifecycle.begin(goal)
    return goal, rel


def run_subject(
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

    goal, initial_rel = create_goal_from_memory(
        generator,
        autobiographical_model,
        memory,
        lifecycle,
        cycle=0,
    )

    if goal is None:
        raise RuntimeError(
            "normal developmental autobiography unexpectedly produced no goal"
        )

    generated_primary = int(goal.primary)
    evaluator_fragile = actual_fragile_primary(
        subject_id
    )
    evaluator_stable = stable_primary(subject_id)

    cap_a, cap_b = initial_capabilities(
        subject_id
    )
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )

    initial_target_capability = state.capability_for(
        generated_primary
    )

    bootstrap_target_discrepancy = mean(
        discrepancy_scalar(r)
        for r in memory.records
        if int(r.primary) == generated_primary
    )

    # Interruption control: an unrelated stable/direct episode occurs while the
    # self-generated goal is active. The active goal object must survive intact.
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
            completion_reliability = (
                current_target_rel
            )
            lifecycle.complete_active(
                cycle=cycle
            )
            break

        record, metrics = (
            execute_development_episode(
                trajectory_model,
                subject_id,
                cycle,
                primary,
                STRATEGY_SCAFFOLD,
                state,
            )
        )
        memory.append(record)
        target_discrepancies.append(
            metrics["discrepancy"]
        )
        state.cycle += 1

    # One final decision event is allowed after the last appended episode.
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

    completed = len(
        lifecycle.completed_goals
    ) >= 1

    final_target_capability = state.capability_for(
        generated_primary
    )
    capability_gain = (
        final_target_capability
        - initial_target_capability
    )

    late_discrepancy = mean(
        target_discrepancies[-2:]
    )
    discrepancy_ratio = (
        late_discrepancy
        / max(
            bootstrap_target_discrepancy,
            1e-6,
        )
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
        "subject_id":
            int(subject_id),
        "memory":
            memory,
        "lifecycle":
            lifecycle,
        "generated_primary":
            generated_primary,
        "evaluator_fragile_primary":
            evaluator_fragile,
        "initial_reliability":
            initial_rel,
        "goal_matches_fragile":
            generated_primary
            == evaluator_fragile,
        "goal_persisted_through_interrupt":
            bool(goal_persisted),
        "completed":
            bool(completed),
        "completion_cycle":
            completion_cycle,
        "completion_reliability":
            completion_reliability,
        "target_reliability_trace":
            target_reliability_trace,
        "capability_gain":
            capability_gain,
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


def donor_goal_assay(
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
        even = build_bootstrap_memory(
            trajectory_model,
            ids[i],
        )
        odd = build_bootstrap_memory(
            trajectory_model,
            ids[i + 1],
        )

        even_rel = reliability_from_memory(
            autobiographical_model,
            even,
        )
        odd_rel = reliability_from_memory(
            autobiographical_model,
            odd,
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
            float(
                empty_goal is None
            )
        )

    return {
        "donor_history_flips_self_generated_goal":
            mean(flips),
        "erased_history_generates_no_developmental_goal":
            mean(erased_none),
    }


def goal_plan_causality_assay(
    trajectory_model,
    autobiographical_model,
    generator,
):
    """
    Same autobiography -> generated target -> plan identity.

    The self-generated primary must determine which work action appears in the
    developmental plan. A donor-history swap must therefore change both goal
    identity and the target work action without an external task assignment.
    """
    from continuous_brain.persistent_self_autobiographical_v1 import (
        ACTION_WORK_A,
        ACTION_WORK_B,
    )
    from evaluate_self_development_loop_v1 import plan

    correct = []
    ids = list(DONOR_IDS)

    for sid in ids:
        memory = build_bootstrap_memory(
            trajectory_model,
            sid,
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
            if goal.primary == PRIMARY_A
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


def persistence_assay(
    generator,
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
            "reliability":
                before_rel.tolist(),
        },
    )
    loaded, payload = load_goal_lifecycle(
        path
    )

    lifecycle_exact = (
        loaded == lifecycle
    )

    metadata_rel = torch.tensor(
        payload["metadata"]["reliability"],
        dtype=before_rel.dtype,
    )

    return {
        "goal_lifecycle_exact_after_reload":
            float(lifecycle_exact),
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
        default="state/endogenous_goal_genesis_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    v034 = json.loads(
        SOURCE_V034_TEST.read_text()
    )
    if v034.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.34.1 frozen TEST PASS required"
        )
    if (
        v034.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.34 TEST must be spent/closed"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("ENDOGENOUS_GOAL_GENESIS_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.35")
    print("source_v0341_test_status: PASS")
    print("external_primary_goal_supplied_to_generator: false")
    print("hidden_capability_supplied_to_generator: false")
    print("subject_id_supplied_to_generator: false")
    print("reward_label_supplied_to_generator: false")
    print("goal_source: frozen_autobiographical_self_estimate")
    print("goal_target_change: +0.12 reliability")
    print("dev_subjects: 90000-90127")
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
            run_subject(
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
                f"goal_genesis_dev_subjects={n}/128"
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
    persistence = persistence_assay(
        generator,
        autobiographical_model,
        subjects[0],
        state_dir,
    )

    goal_match_rate = mean(
        float(s["goal_matches_fragile"])
        for s in subjects
    )
    persistence_rate = mean(
        float(
            s[
                "goal_persisted_through_interrupt"
            ]
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
            s[
                "late_to_bootstrap_discrepancy_ratio"
            ]
            <= MAX_LATE_DISCREPANCY_RATIO
        )
        for s in subjects
    )
    discrepancy_ratio_mean = mean(
        s[
            "late_to_bootstrap_discrepancy_ratio"
        ]
        for s in subjects
    )
    next_goal_generated_rate = mean(
        float(
            s["next_goal_primary"]
            is not None
        )
        for s in subjects
    )

    metrics = {
        "self_generated_goal_matches_autobiographical_fragility":
            goal_match_rate,
        "active_goal_persists_through_unrelated_interrupt":
            persistence_rate,
        "self_generated_goal_completion_rate":
            completion_rate,
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
            goal_match_rate
            >= MIN_GOAL_GENESIS,
        "donor_autobiography_changes_generated_goal":
            donor[
                "donor_history_flips_self_generated_goal"
            ] >= MIN_DONOR_FLIP,
        "erased_autobiography_has_no_directional_goal":
            donor[
                "erased_history_generates_no_developmental_goal"
            ] >= MIN_ERASED_NEUTRAL,
        "self_generated_goal_persists":
            persistence_rate
            >= MIN_GOAL_PERSISTENCE,
        "self_generated_goal_controls_plan":
            goal_plan[
                "self_generated_goal_controls_target_plan"
            ] >= MIN_GOAL_CAUSAL_PLAN,
        "self_generated_goal_completes_from_self_estimate":
            completion_rate
            >= MIN_GOAL_COMPLETION,
        "self_generated_goal_develops_capability":
            capability_gain_mean
            >= MIN_CAPABILITY_GAIN,
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
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during v0.35 DEV"
        )

    result = {
        "format":
            "continuous_brain.endogenous_goal_genesis_v1",
        "family":
            "v0.35",
        "development_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "source_v0341_test_status":
            "PASS",
        "architecture": {
            "external_primary_goal_supplied_to_generator":
                False,
            "hidden_capability_supplied_to_generator":
                False,
            "subject_id_supplied_to_generator":
                False,
            "reward_label_supplied_to_generator":
                False,
            "goal_source":
                "frozen autobiographical reliability",
            "goal_choice":
                "lower-reliability commitment",
            "neutral_tie":
                "no directional developmental goal",
            "goal_target_delta":
                0.12,
            "active_goal_persistent":
                True,
            "completion_source":
                "updated autobiographical self-estimate",
        },
        "split": {
            "dev": [90000, 90127],
            "test": [100000, 100127],
            "test_touched": False,
            "donor_probe": [91000, 91127],
        },
        "metrics":
            metrics,
        "donor_metrics":
            donor,
        "goal_plan_metrics":
            goal_plan,
        "persistence_metrics":
            persistence,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "v0341_test_pass_preserved":
            True,
        "v035_test_touched":
            False,
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

    print("\nGOAL_GENESIS")
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
    print("PROTECTED_UNCHANGED: true")
    print("V0341_TEST_PASS_PRESERVED: true")
    print("V035_TEST_TOUCHED: false")
    print(
        "ENDOGENOUS_GOAL_GENESIS_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "ENDOGENOUS_GOAL_GENESIS_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
