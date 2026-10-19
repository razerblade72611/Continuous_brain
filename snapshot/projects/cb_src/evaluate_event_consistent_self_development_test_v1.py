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
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from continuous_brain.self_development_hysteresis_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
    HistoryRelativeDevelopmentalController,
)
from continuous_brain.self_development_loop_v1 import (
    DevelopmentalState,
    reliability_from_memory,
)
from evaluate_self_development_hysteresis_v1 import (
    AUTOBIO_MODEL,
    TRAJECTORY_MODEL,
    build_bootstrap_memory,
    execute_development_episode,
    mean,
    persistence_assay,
    reliability_for_primary,
    run_subject_hysteresis,
)
from diagnose_self_development_event_order_v1 import trace_subject
from evaluate_event_consistent_self_development_v1 import (
    MAX_LATE_DISCREPANCY_RATIO,
    MIN_BEHAVIOR_SWITCH,
    MIN_BEHAVIOR_TO_HISTORY,
    MIN_CAPABILITY_GAIN,
    MIN_DISCREPANCY_REDUCTION_RATE,
    MIN_DONOR_FLIP,
    MIN_EVENT_ALIGNED_UPDATE_RATE,
    MIN_EVENT_CONSISTENT_FULL_LOOP,
    MIN_FROZEN_HISTORY_BLOCK,
    MIN_HISTORY_GUIDED_STRATEGY,
    MIN_PERSISTENCE,
    MIN_PROFILE_IDENTIFICATION,
    MIN_STABLE_DIRECT,
    REQUIRED_EVENT_ALIGNMENT,
    event_consistent_full_loop,
)

TEST_IDS = range(80000, 80128)

SOURCE_V0340 = Path(
    "state/event_consistent_self_development_v1/result.json"
)
SOURCE_V0335 = Path(
    "state/self_development_event_aligned_test_v1/result.json"
)
AUTOBIO_TEST = Path(
    "state/autobiographical_self_discrepancy_neutral_test_v1/result.json"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

PROTECTED = {
    "v0340_result": SOURCE_V0340,
    "v0335_result": SOURCE_V0335,
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


def frozen_test_donor_assay(
    trajectory_model,
    autobiographical_model,
    controller,
):
    """
    Donor-history intervention entirely inside the frozen TEST population.

    By parity, neighboring even/odd TEST subjects have opposite fragile
    commitments. For a fixed commitment, replacing only autobiography with the
    paired stable donor history must change the initial developmental strategy.
    """
    fragile_scaffold = []
    donor_flip = []
    erased_direct = []

    empty = AutobiographicalTrajectoryMemory()
    empty_rel = reliability_from_memory(
        autobiographical_model,
        empty,
    )

    ids = list(TEST_IDS)
    for i in range(0, len(ids), 2):
        even_id = ids[i]
        odd_id = ids[i + 1]

        even_memory = build_bootstrap_memory(
            trajectory_model,
            even_id,
        )
        odd_memory = build_bootstrap_memory(
            trajectory_model,
            odd_id,
        )

        for primary in (1, 2):
            fragile_memory = (
                even_memory
                if primary == 1
                else odd_memory
            )
            stable_donor_memory = (
                odd_memory
                if primary == 1
                else even_memory
            )

            fragile_rel = reliability_from_memory(
                autobiographical_model,
                fragile_memory,
            )
            donor_rel = reliability_from_memory(
                autobiographical_model,
                stable_donor_memory,
            )

            fragile_phase = controller.initialize(
                reliability_for_primary(
                    fragile_rel,
                    primary,
                )
            )
            donor_phase = controller.initialize(
                reliability_for_primary(
                    donor_rel,
                    primary,
                )
            )
            empty_phase = controller.initialize(
                reliability_for_primary(
                    empty_rel,
                    primary,
                )
            )

            f_strategy = controller.strategy(
                fragile_phase
            )
            d_strategy = controller.strategy(
                donor_phase
            )
            e_strategy = controller.strategy(
                empty_phase
            )

            fragile_scaffold.append(
                float(
                    f_strategy == STRATEGY_SCAFFOLD
                )
            )
            donor_flip.append(
                float(
                    f_strategy == STRATEGY_SCAFFOLD
                    and d_strategy == STRATEGY_DIRECT
                )
            )
            erased_direct.append(
                float(
                    e_strategy == STRATEGY_DIRECT
                )
            )

    return {
        "fragile_history_selects_scaffold":
            mean(fragile_scaffold),
        "donor_history_flips_strategy":
            mean(donor_flip),
        "erased_history_selects_neutral_direct":
            mean(erased_direct),
    }


def frozen_behavior_to_history_assay(
    trajectory_model,
):
    """
    Frozen causal intervention using the TEST subject IDs.

    For each subject, compare DIRECT vs SCAFFOLD at matched low capability and
    matched exogenous draws. Strategy must change the next autobiographical
    discrepancy. This uses no fitting and changes no source artifact.
    """
    effects = []

    for i, sid in enumerate(TEST_IDS):
        primary = 1 if i % 2 == 0 else 2
        cap = .20 + .05 * ((i % 7) / 6.0)

        if primary == 1:
            direct_state = DevelopmentalState(
                cap, .92, 0
            )
            scaffold_state = DevelopmentalState(
                cap, .92, 0
            )
        else:
            direct_state = DevelopmentalState(
                .92, cap, 0
            )
            scaffold_state = DevelopmentalState(
                .92, cap, 0
            )

        shared_u = [.40, .40, .40]

        direct_record, _ = execute_development_episode(
            trajectory_model,
            int(sid),
            0,
            primary,
            STRATEGY_DIRECT,
            direct_state,
            forced_uniforms=shared_u,
        )
        scaffold_record, _ = execute_development_episode(
            trajectory_model,
            int(sid),
            0,
            primary,
            STRATEGY_SCAFFOLD,
            scaffold_state,
            forced_uniforms=shared_u,
        )

        direct_d = float(
            direct_record.feature()[-2].item()
        )
        scaffold_d = float(
            scaffold_record.feature()[-2].item()
        )

        effects.append(
            float(
                direct_d >= scaffold_d + .05
            )
        )

    return {
        "strategy_changes_next_autobiographical_discrepancy":
            mean(effects),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/event_consistent_self_development_test_v1",
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

    v0340 = json.loads(
        SOURCE_V0340.read_text()
    )
    v0335 = json.loads(
        SOURCE_V0335.read_text()
    )

    if v0340.get("development_gate_passed") is not True:
        raise RuntimeError(
            "v0.34.0 DEV PASS required"
        )
    if v0340.get("v034_test_touched") is not False:
        raise RuntimeError(
            "v0.34 TEST was already touched"
        )
    if v0335.get("test_admission_passed") is not False:
        raise RuntimeError(
            "v0.33.5 frozen TEST FAIL must remain preserved"
        )
    if (
        v0335.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.33 TEST must remain spent"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.34")
    print("source_v0340_dev_status: PASS")
    print("source_v0335_test_status: FAIL")
    print("v0335_test_failure_preserved: true")
    print("v033_test_reopened_or_retuned: false")
    print("controller_changed_from_v0340: false")
    print("thresholds_changed_from_v0340: false")
    print("neural_models_changed: false")
    print("development_dynamics_changed: false")
    print("full_loop_definition_changed_from_v0340: false")
    print("v034_test_subjects: 80000-80127")
    print("v034_test_opened_in_this_run: true")
    print("v034_test_spent_after_run: true")
    print("test_tuning_after_run: forbidden")

    trajectory_model, _ = (
        load_trajectory_model(
            TRAJECTORY_MODEL,
            "cpu",
        )
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    autobiographical_model, _ = (
        load_neutral_model(
            AUTOBIO_MODEL,
            "cpu",
        )
    )
    for p in autobiographical_model.parameters():
        p.requires_grad_(False)
    autobiographical_model.eval()

    controller = (
        HistoryRelativeDevelopmentalController()
    )

    subjects = []
    frozen_controls = []
    traces = []
    composites = []

    ids = list(TEST_IDS)
    for n, sid in enumerate(ids, 1):
        s = run_subject_hysteresis(
            trajectory_model,
            autobiographical_model,
            controller,
            int(sid),
            freeze_decision_history=False,
        )
        f = run_subject_hysteresis(
            trajectory_model,
            autobiographical_model,
            controller,
            int(sid),
            freeze_decision_history=True,
        )
        t = trace_subject(
            trajectory_model,
            autobiographical_model,
            controller,
            int(sid),
        )

        subjects.append(s)
        frozen_controls.append(f)
        traces.append(t)
        composites.append(
            event_consistent_full_loop(
                s,
                t,
                f,
            )
        )

        if (
            n == 1
            or n % 32 == 0
            or n == len(ids)
        ):
            print(
                f"v034_test_subjects={n}/128"
            )

    donor = frozen_test_donor_assay(
        trajectory_model,
        autobiographical_model,
        controller,
    )
    behavior_history = (
        frozen_behavior_to_history_assay(
            trajectory_model,
        )
    )
    persistence = persistence_assay(
        autobiographical_model,
        controller,
        subjects[0],
        state_dir,
    )

    profile_identification = mean(
        float(
            s["initial_fragile_reliability"]
            < s["initial_stable_reliability"]
        )
        for s in subjects
    )
    initial_scaffold = mean(
        float(
            s["fragile_strategies"][0]
            == STRATEGY_SCAFFOLD
        )
        for s in subjects
    )
    stable_direct = mean(
        float(
            all(
                st == STRATEGY_DIRECT
                for st in s[
                    "stable_strategies"
                ]
            )
        )
        for s in subjects
    )
    switch_rate = mean(
        float(
            s["switched_to_direct"]
        )
        for s in subjects
    )
    capability_gain_mean = mean(
        s["fragile_capability_gain"]
        for s in subjects
    )
    discrepancy_ratio_mean = mean(
        s[
            "late_to_bootstrap_discrepancy_ratio"
        ]
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
    frozen_block_rate = mean(
        float(
            c["frozen_history_block"]
        )
        for c in composites
    )

    predecision_cross_rate = mean(
        float(
            t["ever_predecision_crossed"]
        )
        for t in traces
    )
    graduation_rate = mean(
        float(
            t["ever_graduated"]
        )
        for t in traces
    )
    exact_alignment_rate = mean(
        float(
            t["exact_event_alignment"]
        )
        for t in traces
    )
    graduated_without_cross = mean(
        float(
            t[
                "graduated_without_predecision_cross"
            ]
        )
        for t in traces
    )
    cross_without_graduation = mean(
        float(
            t[
                "predecision_cross_without_graduation"
            ]
        )
        for t in traces
    )

    full_loop_rate = mean(
        float(
            c["passed"]
        )
        for c in composites
    )

    closed_loop = {
        "initial_fragile_profile_identification":
            profile_identification,
        "initial_fragile_scaffold_rate":
            initial_scaffold,
        "stable_commitment_always_direct_rate":
            stable_direct,
        "fragile_behavior_switch_to_direct_rate":
            switch_rate,
        "mean_fragile_capability_gain":
            capability_gain_mean,
        "mean_late_to_bootstrap_discrepancy_ratio":
            discrepancy_ratio_mean,
        "discrepancy_reduction_rate":
            discrepancy_reduction_rate,
        "frozen_history_blocks_graduation_rate":
            frozen_block_rate,
        "event_consistent_full_loop_rate":
            full_loop_rate,
    }

    event_metrics = {
        "predecision_cross_rate":
            predecision_cross_rate,
        "graduation_rate":
            graduation_rate,
        "exact_event_alignment_rate":
            exact_alignment_rate,
        "graduated_without_predecision_cross_rate":
            graduated_without_cross,
        "predecision_cross_without_graduation_rate":
            cross_without_graduation,
    }

    component_rates = {}
    keys = [
        "initial_fragility",
        "initial_scaffold",
        "stable_direct",
        "event_cross",
        "exact_alignment",
        "switched",
        "capability_gain",
        "reduced_discrepancy",
        "frozen_history_block",
    ]
    for key in keys:
        component_rates[key] = mean(
            float(c[key])
            for c in composites
        )

    checks = {
        "initial_autobiography_identifies_fragility":
            profile_identification
            >= MIN_PROFILE_IDENTIFICATION,
        "autobiography_guides_initial_behavior":
            initial_scaffold
            >= MIN_HISTORY_GUIDED_STRATEGY,
        "stable_history_guides_direct_behavior":
            stable_direct
            >= MIN_STABLE_DIRECT,
        "donor_history_changes_behavior":
            donor[
                "donor_history_flips_strategy"
            ] >= MIN_DONOR_FLIP,
        "erased_history_removes_fragility_response":
            donor[
                "erased_history_selects_neutral_direct"
            ] >= MIN_DONOR_FLIP,
        "behavior_changes_next_autobiography":
            behavior_history[
                "strategy_changes_next_autobiographical_discrepancy"
            ] >= MIN_BEHAVIOR_TO_HISTORY,
        "new_autobiography_updates_self_estimate_at_decision":
            predecision_cross_rate
            >= MIN_EVENT_ALIGNED_UPDATE_RATE,
        "self_estimate_event_exactly_causes_graduation":
            (
                exact_alignment_rate
                >= REQUIRED_EVENT_ALIGNMENT
                and graduated_without_cross == 0.0
                and cross_without_graduation == 0.0
                and abs(
                    predecision_cross_rate
                    - graduation_rate
                ) <= 1e-12
            ),
        "self_estimate_changes_later_behavior":
            switch_rate
            >= MIN_BEHAVIOR_SWITCH,
        "frozen_autobiography_blocks_behavior_change":
            frozen_block_rate
            >= MIN_FROZEN_HISTORY_BLOCK,
        "behavior_develops_capability":
            capability_gain_mean
            >= MIN_CAPABILITY_GAIN,
        "development_reduces_discrepancy":
            discrepancy_reduction_rate
            >= MIN_DISCREPANCY_REDUCTION_RATE,
        "event_consistent_full_feedback_loop":
            full_loop_rate
            >= MIN_EVENT_CONSISTENT_FULL_LOOP,
        "capability_persists":
            persistence[
                "capability_exact_after_reload"
            ] >= MIN_PERSISTENCE,
        "autobiography_persists":
            persistence[
                "autobiographical_reliability_exact_after_reload"
            ] >= MIN_PERSISTENCE,
        "behavior_policy_persists":
            persistence[
                "developmental_strategy_exact_after_reload"
            ] >= MIN_PERSISTENCE,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during frozen v0.34 TEST"
        )

    result = {
        "format":
            "continuous_brain.event_consistent_self_development_test_v1",
        "family":
            "v0.34",
        "test_admission_passed":
            bool(passed),
        "training_performed":
            False,
        "historical_status": {
            "v0335":
                "FROZEN_TEST_FAIL",
            "v0335_test_spent":
                True,
            "v0340":
                "DEV_PASS",
        },
        "frozen_mechanism": {
            "controller_changed":
                False,
            "thresholds_changed":
                False,
            "neural_models_changed":
                False,
            "development_dynamics_changed":
                False,
            "full_loop_definition_changed":
                False,
            "self_estimate_gain_threshold":
                0.12,
        },
        "split": {
            "test": [80000, 80127],
            "test_opened_in_this_run":
                True,
            "test_spent":
                True,
            "test_tuning_after_run_forbidden":
                True,
        },
        "closed_loop_metrics":
            closed_loop,
        "event_aligned_metrics":
            event_metrics,
        "per_subject_component_rates":
            component_rates,
        "donor_history_metrics":
            donor,
        "behavior_to_history_metrics":
            behavior_history,
        "persistence_metrics":
            persistence,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "source_models_unchanged":
            True,
        "v0335_frozen_test_failure_preserved":
            True,
        "v033_test_reopened":
            False,
        "v034_test_spent":
            True,
        "scope": {
            "functional_self_development_loop_claimed_if_pass":
                True,
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

    print("\nV034_TEST_CLOSED_LOOP")
    for k, v in closed_loop.items():
        print(f"{k}: {v}")

    print("\nV034_TEST_EVENT_ALIGNED")
    for k, v in event_metrics.items():
        print(f"{k}: {v}")

    print("\nV034_TEST_COMPONENT_RATES")
    for k, v in component_rates.items():
        print(f"{k}: {v}")

    print("\nV034_TEST_DONOR_HISTORY")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nV034_TEST_BEHAVIOR_TO_HISTORY")
    for k, v in behavior_history.items():
        print(f"{k}: {v}")

    print("\nV034_TEST_PERSISTENCE")
    for k, v in persistence.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("TRAINING_PERFORMED: false")
    print(
        "V0335_FROZEN_TEST_FAILURE_PRESERVED: true"
    )
    print("V033_TEST_REOPENED: false")
    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_MODELS_UNCHANGED: true")
    print("V034_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
