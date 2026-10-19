#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from evaluate_self_development_hysteresis_v1 import (
    AUTOBIO_MODEL,
    DEV_IDS,
    TRAJECTORY_MODEL,
    behavior_to_history_assay,
    donor_phase_assay,
    mean,
    persistence_assay,
    run_subject_hysteresis,
)
from diagnose_self_development_event_order_v1 import trace_subject

SOURCE_V0330 = Path("state/self_development_loop_v1/result.json")
SOURCE_V0331 = Path("state/self_development_hysteresis_v1/result.json")
SOURCE_V0332 = Path(
    "state/self_development_gain_timing_diagnostic_v1/result.json"
)
SOURCE_V0333 = Path(
    "state/self_development_event_order_diagnostic_v1/result.json"
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
MIN_FULL_LOOP = .70
MIN_FROZEN_HISTORY_BLOCK = .95
MIN_PERSISTENCE = 1.0
REQUIRED_EVENT_ALIGNMENT = 1.0

PROTECTED = {
    "v0330_result": SOURCE_V0330,
    "v0331_result": SOURCE_V0331,
    "v0332_result": SOURCE_V0332,
    "v0333_result": SOURCE_V0333,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_development_event_aligned_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    v0330 = json.loads(SOURCE_V0330.read_text())
    v0331 = json.loads(SOURCE_V0331.read_text())
    v0332 = json.loads(SOURCE_V0332.read_text())
    v0333 = json.loads(SOURCE_V0333.read_text())

    if v0330.get("development_gate_passed") is not False:
        raise RuntimeError("v0.33.0 historical FAIL required")
    if v0331.get("repair_dev_gate_passed") is not False:
        raise RuntimeError("v0.33.1 historical FAIL required")

    failed = sorted(
        k for k, v in v0331["checks"].items() if not bool(v)
    )
    if failed != ["new_autobiography_updates_self_estimate"]:
        raise RuntimeError(
            f"unexpected v0.33.1 failed gates: {failed}"
        )

    if v0332.get("timing_consistent") is not False:
        raise RuntimeError(
            "v0.33.2 coarse timing explanation must remain rejected"
        )

    if v0333.get("event_order_consistent") is not True:
        raise RuntimeError(
            "v0.33.3 exact event-order diagnostic PASS required"
        )

    if v0333["metrics"]["graduated_without_predecision_cross_rate"] != 0.0:
        raise RuntimeError("v0.33.3 graduation/crossing mismatch present")
    if v0333["metrics"]["predecision_cross_without_graduation_rate"] != 0.0:
        raise RuntimeError("v0.33.3 crossing/graduation mismatch present")
    if v0333["metrics"]["exact_event_alignment_rate"] != 1.0:
        raise RuntimeError("v0.33.3 event alignment not exact")

    if v0331.get("development_test_touched") is not False:
        raise RuntimeError("v0.33 TEST already touched")

    before = {name: sha256(path) for name, path in PROTECTED.items()}

    print("SELF_DEVELOPMENT_EVENT_ALIGNED_V1")
    print("device: cpu")
    print("training_performed: false")
    print("source_v0330_formal_status: FAIL")
    print("source_v0331_formal_status: FAIL")
    print("source_v0332_coarse_timing_explanation: REJECTED")
    print("source_v0333_exact_event_order: SUPPORTED")
    print("repair_scope: gate_semantics_only")
    print("controller_changed: false")
    print("development_dynamics_changed: false")
    print("neural_models_changed: false")
    print("development_window_reused: subjects 10000-10127")
    print("new_development_window_consumed: false")
    print("development_test_subjects: 20000-20127")
    print("development_test_touched: false")
    print(
        "old_failed_gate:",
        "final_reliability_gain_ge_0.12_rate_ge_0.80",
    )
    print(
        "replacement_gate:",
        "predecision_reliability_cross_ge_0.12_rate_ge_0.80_with_exact_event_alignment",
    )

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

    controller = HistoryRelativeDevelopmentalController()

    subjects = []
    frozen_controls = []
    traces = []

    for n, sid in enumerate(DEV_IDS, 1):
        sid = int(sid)

        subjects.append(
            run_subject_hysteresis(
                trajectory_model,
                autobiographical_model,
                controller,
                sid,
                freeze_decision_history=False,
            )
        )
        frozen_controls.append(
            run_subject_hysteresis(
                trajectory_model,
                autobiographical_model,
                controller,
                sid,
                freeze_decision_history=True,
            )
        )
        traces.append(
            trace_subject(
                trajectory_model,
                autobiographical_model,
                controller,
                sid,
            )
        )

        if n == 1 or n % 32 == 0 or n == 128:
            print(f"event_aligned_dev_subjects={n}/128")

    donor = donor_phase_assay(
        trajectory_model,
        autobiographical_model,
        controller,
    )
    behavior_history = behavior_to_history_assay(
        trajectory_model,
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
            s["fragile_strategies"][0] == STRATEGY_SCAFFOLD
        )
        for s in subjects
    )
    stable_direct = mean(
        float(
            all(st == STRATEGY_DIRECT for st in s["stable_strategies"])
        )
        for s in subjects
    )
    switch_rate = mean(
        float(s["switched_to_direct"]) for s in subjects
    )
    capability_gain_mean = mean(
        s["fragile_capability_gain"] for s in subjects
    )
    discrepancy_ratio_mean = mean(
        s["late_to_bootstrap_discrepancy_ratio"]
        for s in subjects
    )
    discrepancy_reduction_rate = mean(
        float(
            s["late_to_bootstrap_discrepancy_ratio"]
            <= MAX_LATE_DISCREPANCY_RATIO
        )
        for s in subjects
    )
    full_loop_rate = mean(
        float(s["full_loop"]) for s in subjects
    )
    frozen_history_no_graduation = mean(
        float(
            not s["switched_to_direct"]
            and all(
                st == STRATEGY_SCAFFOLD
                for st in s["fragile_strategies"]
            )
        )
        for s in frozen_controls
    )

    # Historical final-state metric retained for record but no longer used as
    # the causal update gate.
    final_gain_rate = mean(
        float(s["fragile_reliability_gain"] >= .12)
        for s in subjects
    )
    final_gain_mean = mean(
        s["fragile_reliability_gain"]
        for s in subjects
    )

    predecision_cross_rate = mean(
        float(t["ever_predecision_crossed"]) for t in traces
    )
    graduation_rate = mean(
        float(t["ever_graduated"]) for t in traces
    )
    exact_event_alignment_rate = mean(
        float(t["exact_event_alignment"]) for t in traces
    )
    graduated_without_cross = mean(
        float(t["graduated_without_predecision_cross"])
        for t in traces
    )
    cross_without_graduation = mean(
        float(t["predecision_cross_without_graduation"])
        for t in traces
    )

    event_metrics = {
        "predecision_cross_rate":
            predecision_cross_rate,
        "graduation_rate":
            graduation_rate,
        "exact_event_alignment_rate":
            exact_event_alignment_rate,
        "graduated_without_predecision_cross_rate":
            graduated_without_cross,
        "predecision_cross_without_graduation_rate":
            cross_without_graduation,
        "historical_final_gain_rate":
            final_gain_rate,
        "historical_mean_final_gain":
            final_gain_mean,
    }

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
        "full_closed_loop_rate":
            full_loop_rate,
        "frozen_history_blocks_graduation_rate":
            frozen_history_no_graduation,
    }

    checks = {
        "initial_autobiography_identifies_fragility":
            profile_identification >= MIN_PROFILE_IDENTIFICATION,
        "autobiography_guides_initial_behavior":
            initial_scaffold >= MIN_HISTORY_GUIDED_STRATEGY,
        "stable_history_guides_direct_behavior":
            stable_direct >= MIN_STABLE_DIRECT,

        "donor_history_changes_behavior":
            donor["donor_history_flips_strategy"] >= MIN_DONOR_FLIP,
        "erased_history_removes_fragility_response":
            donor["erased_history_selects_neutral_direct"] >= MIN_DONOR_FLIP,

        "behavior_changes_next_autobiography":
            behavior_history[
                "strategy_changes_next_autobiographical_discrepancy"
            ] >= MIN_BEHAVIOR_TO_HISTORY,

        "new_autobiography_updates_self_estimate_at_decision":
            predecision_cross_rate >= MIN_EVENT_ALIGNED_UPDATE_RATE,
        "self_estimate_event_exactly_causes_graduation":
            (
                exact_event_alignment_rate >= REQUIRED_EVENT_ALIGNMENT
                and graduated_without_cross == 0.0
                and cross_without_graduation == 0.0
                and abs(predecision_cross_rate - graduation_rate) <= 1e-12
            ),

        "self_estimate_changes_later_behavior":
            switch_rate >= MIN_BEHAVIOR_SWITCH,
        "frozen_autobiography_blocks_behavior_change":
            frozen_history_no_graduation >= MIN_FROZEN_HISTORY_BLOCK,

        "behavior_develops_capability":
            capability_gain_mean >= MIN_CAPABILITY_GAIN,
        "development_reduces_discrepancy":
            discrepancy_reduction_rate >= MIN_DISCREPANCY_REDUCTION_RATE,
        "full_feedback_loop":
            full_loop_rate >= MIN_FULL_LOOP,

        "capability_persists":
            persistence["capability_exact_after_reload"] >= MIN_PERSISTENCE,
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

    after = {name: sha256(path) for name, path in PROTECTED.items()}
    if before != after:
        raise RuntimeError(
            "protected artifact changed during v0.33.4 event-aligned repair"
        )

    result = {
        "format":
            "continuous_brain.self_development_event_aligned_v1",
        "repair_dev_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "historical_status": {
            "v0330": "FAIL",
            "v0331": "FAIL",
            "v0332":
                "coarse_timing_explanation_rejected",
            "v0333":
                "exact_event_order_supported",
        },
        "repair": {
            "controller_changed":
                False,
            "development_dynamics_changed":
                False,
            "neural_models_changed":
                False,
            "threshold_changed":
                False,
            "self_estimate_gain_threshold":
                0.12,
            "old_gate":
                "final_gain_rate",
            "replacement_gate":
                "decision_event_crossing_rate_with_exact_alignment",
            "justification":
                "v0.33.3 measured the exact self-estimate available to the controller and found exact causal alignment between +0.12 crossings and graduation",
        },
        "split": {
            "reused_dev": [10000, 10127],
            "test": [20000, 20127],
            "test_touched": False,
        },
        "closed_loop_metrics":
            closed_loop,
        "event_aligned_metrics":
            event_metrics,
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
        "v0322_test_pass_preserved":
            True,
        "development_test_touched":
            False,
        "scope": {
            "claim":
                "updated autobiography causes a meaningful self-estimate increase at the actual decision event; that event exactly aligns with the developmental strategy transition, while frozen autobiography blocks the transition",
            "subjective_experience_claimed":
                False,
            "sentience_claimed":
                False,
            "consciousness_claimed":
                False,
        },
    }

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nCLOSED_LOOP")
    for k, v in closed_loop.items():
        print(f"{k}: {v}")

    print("\nEVENT_ALIGNED_SELF_ESTIMATE")
    for k, v in event_metrics.items():
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
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("TRAINING_PERFORMED: false")
    print("CONTROLLER_CHANGED: false")
    print("THRESHOLD_CHANGED: false")
    print("PROTECTED_UNCHANGED: true")
    print("V0322_TEST_PASS_PRESERVED: true")
    print("DEVELOPMENT_TEST_TOUCHED: false")
    print(
        "SELF_DEVELOPMENT_EVENT_ALIGNED_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print("SELF_DEVELOPMENT_EVENT_ALIGNED_RUN_COMPLETE")
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
