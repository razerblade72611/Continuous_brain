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
    initial_capabilities,
    mean,
    persistence_assay,
    reliability_for_primary,
    run_subject_hysteresis,
)
from diagnose_self_development_event_order_v1 import trace_subject

DEV_IDS = range(70000, 70128)
TEST_IDS = range(80000, 80128)
DONOR_IDS = range(71000, 71128)
BEHAVIOR_PROBE_IDS = range(72000, 72128)

SOURCE_V0334 = Path(
    "state/self_development_event_aligned_v1/result.json"
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

MIN_PROFILE_IDENTIFICATION = .95
MIN_HISTORY_GUIDED_STRATEGY = .95
MIN_DONOR_FLIP = .95
MIN_BEHAVIOR_TO_HISTORY = .90
MIN_EVENT_ALIGNED_UPDATE_RATE = .80
MIN_BEHAVIOR_SWITCH = .75
MIN_CAPABILITY_GAIN = .35
MIN_STABLE_DIRECT = .95
MAX_LATE_DISCREPANCY_RATIO = .75
MIN_DISCREPANCY_REDUCTION_RATE = .80
MIN_EVENT_CONSISTENT_FULL_LOOP = .70
MIN_FROZEN_HISTORY_BLOCK = .95
MIN_PERSISTENCE = 1.0
REQUIRED_EVENT_ALIGNMENT = 1.0

PROTECTED = {
    "v0334_result": SOURCE_V0334,
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


def fragile_primary(subject_id: int) -> int:
    return 1 if int(subject_id) % 2 == 0 else 2


def donor_assay(
    trajectory_model,
    autobiographical_model,
    controller,
):
    fragile_scaffold = []
    donor_flip = []
    erased_direct = []

    empty = AutobiographicalTrajectoryMemory()
    empty_rel = reliability_from_memory(
        autobiographical_model,
        empty,
    )

    ids = list(DONOR_IDS)
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
                even_memory if primary == 1 else odd_memory
            )
            stable_memory = (
                odd_memory if primary == 1 else even_memory
            )

            fragile_rel = reliability_from_memory(
                autobiographical_model,
                fragile_memory,
            )
            stable_rel = reliability_from_memory(
                autobiographical_model,
                stable_memory,
            )

            fragile_phase = controller.initialize(
                reliability_for_primary(
                    fragile_rel,
                    primary,
                )
            )
            donor_phase = controller.initialize(
                reliability_for_primary(
                    stable_rel,
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


def fresh_behavior_to_history_assay(
    trajectory_model,
):
    effects = []

    for i, sid in enumerate(BEHAVIOR_PROBE_IDS):
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


def event_consistent_full_loop(
    subject,
    trace,
    frozen_control,
):
    """
    Corrected per-subject composite.

    Deliberately does NOT use the historical final reliability gain.
    The self-estimate condition is measured at the actual controller decision
    event, exactly as admitted in v0.33.4.
    """
    initial_fragility = (
        subject["initial_fragile_reliability"]
        < subject["initial_stable_reliability"]
    )
    initial_scaffold = (
        len(subject["fragile_strategies"]) > 0
        and subject["fragile_strategies"][0]
        == STRATEGY_SCAFFOLD
    )
    stable_direct = all(
        st == STRATEGY_DIRECT
        for st in subject["stable_strategies"]
    )
    event_cross = bool(
        trace["ever_predecision_crossed"]
    )
    exact_alignment = bool(
        trace["exact_event_alignment"]
        and not trace[
            "graduated_without_predecision_cross"
        ]
        and not trace[
            "predecision_cross_without_graduation"
        ]
    )
    switched = bool(
        subject["switched_to_direct"]
    )
    capability = (
        subject["fragile_capability_gain"]
        >= MIN_CAPABILITY_GAIN
    )
    reduced_discrepancy = (
        subject[
            "late_to_bootstrap_discrepancy_ratio"
        ]
        <= MAX_LATE_DISCREPANCY_RATIO
    )
    frozen_block = (
        not frozen_control["switched_to_direct"]
        and all(
            st == STRATEGY_SCAFFOLD
            for st in frozen_control[
                "fragile_strategies"
            ]
        )
    )

    passed = all(
        [
            initial_fragility,
            initial_scaffold,
            stable_direct,
            event_cross,
            exact_alignment,
            switched,
            capability,
            reduced_discrepancy,
            frozen_block,
        ]
    )

    return {
        "passed": bool(passed),
        "initial_fragility": bool(initial_fragility),
        "initial_scaffold": bool(initial_scaffold),
        "stable_direct": bool(stable_direct),
        "event_cross": bool(event_cross),
        "exact_alignment": bool(exact_alignment),
        "switched": bool(switched),
        "capability_gain": bool(capability),
        "reduced_discrepancy": bool(reduced_discrepancy),
        "frozen_history_block": bool(frozen_block),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/event_consistent_self_development_v1",
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

    v0334 = json.loads(
        SOURCE_V0334.read_text()
    )
    v0335 = json.loads(
        SOURCE_V0335.read_text()
    )

    if v0334.get("repair_dev_gate_passed") is not True:
        raise RuntimeError(
            "v0.33.4 DEV PASS required"
        )
    if v0335.get("test_admission_passed") is not False:
        raise RuntimeError(
            "v0.33.5 frozen TEST FAIL required"
        )
    if (
        v0335.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.33 TEST must be spent/closed"
        )

    failed = sorted(
        k
        for k, v in v0335["checks"].items()
        if not bool(v)
    )
    if failed != ["full_feedback_loop"]:
        raise RuntimeError(
            f"unexpected v0.33.5 failures: {failed}"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("EVENT_CONSISTENT_SELF_DEVELOPMENT_V1")
    print("device: cpu")
    print("training_performed: false")
    print("family: v0.34")
    print("source_v0334_dev_status: PASS")
    print("source_v0335_test_status: FAIL")
    print("source_v0335_only_failure: full_feedback_loop")
    print(
        "v0335_test_reopened_or_retuned: false"
    )
    print(
        "v034_change:",
        "per_subject_full_loop_uses_event_aligned_self_estimate_and_frozen_history_control",
    )
    print(
        "controller_changed_from_v0334: false"
    )
    print(
        "thresholds_changed_from_v0334: false"
    )
    print(
        "neural_models_changed: false"
    )
    print(
        "development_dynamics_changed: false"
    )
    print("v034_dev_subjects: 70000-70127")
    print("v034_test_subjects: 80000-80127")
    print("v034_test_touched: false")
    print("donor_probe_subjects: 71000-71127")
    print(
        "behavior_probe_subjects: 72000-72127"
    )

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

    ids = list(DEV_IDS)
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
                s, t, f
            )
        )

        if (
            n == 1
            or n % 32 == 0
            or n == len(ids)
        ):
            print(
                f"v034_dev_subjects={n}/128"
            )

    donor = donor_assay(
        trajectory_model,
        autobiographical_model,
        controller,
    )
    behavior_history = (
        fresh_behavior_to_history_assay(
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
        float(s["switched_to_direct"])
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
        float(t["ever_predecision_crossed"])
        for t in traces
    )
    graduation_rate = mean(
        float(t["ever_graduated"])
        for t in traces
    )
    exact_alignment_rate = mean(
        float(t["exact_event_alignment"])
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
        float(c["passed"])
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
            "protected artifact changed during v0.34 DEV"
        )

    result = {
        "format":
            "continuous_brain.event_consistent_self_development_v1",
        "family":
            "v0.34",
        "development_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "historical_status": {
            "v0334":
                "DEV_PASS",
            "v0335":
                "FROZEN_TEST_FAIL",
            "v0335_test_spent":
                True,
        },
        "architectural_correction": {
            "stale_v033_composite":
                "used final reliability gain inside full_loop after standalone gate had moved to event-aligned semantics",
            "v033_test_reopened":
                False,
            "v033_test_retuned":
                False,
            "controller_changed":
                False,
            "thresholds_changed":
                False,
            "neural_models_changed":
                False,
            "development_dynamics_changed":
                False,
            "new_family_fresh_split":
                True,
            "full_loop_definition":
                "per-subject conjunction of initial autobiographical fragility, initial scaffold, stable direct, decision-event self-estimate crossing, exact crossing/graduation alignment, later direct behavior, capability gain, discrepancy reduction, and frozen-history blocking",
        },
        "split": {
            "dev": [70000, 70127],
            "test": [80000, 80127],
            "test_touched": False,
            "donor_probe": [71000, 71127],
            "behavior_probe": [72000, 72127],
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
        "v0335_frozen_test_failure_preserved":
            True,
        "v034_test_touched":
            False,
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

    print("\nV034_CLOSED_LOOP")
    for k, v in closed_loop.items():
        print(f"{k}: {v}")

    print("\nV034_EVENT_ALIGNED")
    for k, v in event_metrics.items():
        print(f"{k}: {v}")

    print("\nPER_SUBJECT_COMPONENT_RATES")
    for k, v in component_rates.items():
        print(f"{k}: {v}")

    print("\nDONOR_HISTORY")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nBEHAVIOR_TO_HISTORY")
    for k, v in behavior_history.items():
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
    print(
        "V0335_FROZEN_TEST_FAILURE_PRESERVED: true"
    )
    print("V033_TEST_REOPENED: false")
    print("PROTECTED_UNCHANGED: true")
    print("V034_TEST_TOUCHED: false")
    print(
        "EVENT_CONSISTENT_SELF_DEVELOPMENT_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "EVENT_CONSISTENT_SELF_DEVELOPMENT_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
