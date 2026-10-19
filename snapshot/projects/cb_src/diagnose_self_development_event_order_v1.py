#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from continuous_brain.self_development_hysteresis_v1 import (
    MEANINGFUL_SELF_ESTIMATE_GAIN,
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
    DEV_IDS,
    SOURCE_V033_RESULT,
    TRAJECTORY_MODEL,
    build_bootstrap_memory,
    execute_development_episode,
    fragile_primary,
    initial_capabilities,
    reliability_for_primary,
)

SOURCE_V0331_RESULT = Path(
    "state/self_development_hysteresis_v1/result.json"
)
SOURCE_V0332_RESULT = Path(
    "state/self_development_gain_timing_diagnostic_v1/result.json"
)
AUTOBIO_TEST = Path(
    "state/autobiographical_self_discrepancy_neutral_test_v1/result.json"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

DEVELOPMENT_CYCLES = 8

PROTECTED = {
    "v0330_result": SOURCE_V033_RESULT,
    "v0331_result": SOURCE_V0331_RESULT,
    "v0332_result": SOURCE_V0332_RESULT,
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
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def trace_subject(
    trajectory_model,
    autobiographical_model,
    controller,
    subject_id: int,
):
    memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )

    initial_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    phases = {
        1: controller.initialize(float(initial_rel[0].item())),
        2: controller.initialize(float(initial_rel[1].item())),
    }

    fp = fragile_primary(subject_id)
    baseline = reliability_for_primary(initial_rel, fp)

    fragile_decisions = []
    every_append = []
    decision_index = 0
    append_index = 0

    for cycle in range(DEVELOPMENT_CYCLES):
        order = (
            [1, 2]
            if (cycle + subject_id) % 2 == 0
            else [2, 1]
        )

        for primary in order:
            decision_rel = reliability_from_memory(
                autobiographical_model,
                memory,
            )
            current = reliability_for_primary(
                decision_rel,
                primary,
            )

            phase = phases[primary]
            strategy_before_update = controller.strategy(phase)
            graduated_before = bool(phase.graduated)

            phases[primary] = controller.update(
                phase,
                current,
            )

            strategy_after_update = controller.strategy(
                phases[primary]
            )
            graduated_after = bool(
                phases[primary].graduated
            )
            graduation_event = (
                strategy_before_update == STRATEGY_SCAFFOLD
                and strategy_after_update == STRATEGY_DIRECT
            )

            if primary == fp:
                fragile_current = reliability_for_primary(
                    decision_rel,
                    fp,
                )
                fragile_decisions.append(
                    {
                        "decision_index": decision_index,
                        "cycle": cycle,
                        "primary": int(primary),
                        "predecision_reliability":
                            fragile_current,
                        "predecision_gain":
                            fragile_current - baseline,
                        "threshold_crossed_at_decision":
                            bool(
                                fragile_current
                                >= baseline
                                + MEANINGFUL_SELF_ESTIMATE_GAIN
                            ),
                        "strategy_before_update":
                            int(strategy_before_update),
                        "strategy_after_update":
                            int(strategy_after_update),
                        "graduated_before":
                            graduated_before,
                        "graduated_after":
                            graduated_after,
                        "graduation_event":
                            bool(graduation_event),
                    }
                )
                decision_index += 1

            record, _ = execute_development_episode(
                trajectory_model,
                subject_id,
                cycle,
                primary,
                strategy_after_update,
                state,
            )
            memory.append(record)

            post_rel = reliability_from_memory(
                autobiographical_model,
                memory,
            )
            post_fragile = reliability_for_primary(
                post_rel,
                fp,
            )
            every_append.append(
                {
                    "append_index": append_index,
                    "cycle": cycle,
                    "episode_primary": int(primary),
                    "post_append_fragile_reliability":
                        post_fragile,
                    "post_append_gain":
                        post_fragile - baseline,
                    "threshold_crossed_post_append":
                        bool(
                            post_fragile
                            >= baseline
                            + MEANINGFUL_SELF_ESTIMATE_GAIN
                        ),
                }
            )
            append_index += 1

        state.cycle += 1

    final_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    final_r = reliability_for_primary(
        final_rel,
        fp,
    )
    final_gain = final_r - baseline

    pre_cross_events = [
        e for e in fragile_decisions
        if e["threshold_crossed_at_decision"]
    ]
    graduation_events = [
        e for e in fragile_decisions
        if e["graduation_event"]
    ]
    post_cross_events = [
        e for e in every_append
        if e["threshold_crossed_post_append"]
    ]

    first_pre_cross = (
        pre_cross_events[0]["decision_index"]
        if pre_cross_events else None
    )
    first_graduation = (
        graduation_events[0]["decision_index"]
        if graduation_events else None
    )

    ever_predecision_crossed = bool(pre_cross_events)
    ever_graduated = bool(graduation_events)
    ever_postappend_crossed = bool(post_cross_events)

    # A post-append crossing can occur after the final fragile decision.
    last_fragile_decision_append_boundary = None
    # The final fragile decision itself happens immediately before one append.
    # Its corresponding append position is recoverable by scanning actual order.
    fragile_append_indices = [
        e["append_index"]
        for e in every_append
        if e["episode_primary"] == fp
    ]
    if fragile_append_indices:
        last_fragile_decision_append_boundary = fragile_append_indices[-1]

    post_cross_after_last_fragile_decision = False
    if last_fragile_decision_append_boundary is not None:
        post_cross_after_last_fragile_decision = any(
            e["append_index"] >= last_fragile_decision_append_boundary
            and e["threshold_crossed_post_append"]
            for e in every_append
        )

    exact_event_alignment = (
        ever_predecision_crossed == ever_graduated
        and (
            (not ever_predecision_crossed)
            or first_pre_cross == first_graduation
        )
    )

    return {
        "subject_id": int(subject_id),
        "fragile_primary": int(fp),
        "baseline_reliability": baseline,
        "final_reliability": final_r,
        "final_gain": final_gain,
        "peak_predecision_gain":
            max(
                [e["predecision_gain"] for e in fragile_decisions],
                default=0.0,
            ),
        "peak_postappend_gain":
            max(
                [e["post_append_gain"] for e in every_append],
                default=0.0,
            ),
        "ever_predecision_crossed":
            ever_predecision_crossed,
        "ever_graduated":
            ever_graduated,
        "ever_postappend_crossed":
            ever_postappend_crossed,
        "first_predecision_cross_index":
            first_pre_cross,
        "first_graduation_index":
            first_graduation,
        "exact_event_alignment":
            exact_event_alignment,
        "graduated_without_predecision_cross":
            bool(
                ever_graduated
                and not ever_predecision_crossed
            ),
        "predecision_cross_without_graduation":
            bool(
                ever_predecision_crossed
                and not ever_graduated
            ),
        "postappend_cross_without_predecision_cross":
            bool(
                ever_postappend_crossed
                and not ever_predecision_crossed
            ),
        "post_cross_after_last_fragile_decision":
            bool(post_cross_after_last_fragile_decision),
        "final_above_threshold":
            bool(
                final_gain
                >= MEANINGFUL_SELF_ESTIMATE_GAIN
            ),
        "fragile_decisions":
            fragile_decisions,
        "every_append":
            every_append,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_development_event_order_diagnostic_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    v0331 = json.loads(SOURCE_V0331_RESULT.read_text())
    if v0331.get("repair_dev_gate_passed") is not False:
        raise RuntimeError("v0.33.1 formal DEV FAIL required")

    failed = sorted(
        k
        for k, v in v0331["checks"].items()
        if not bool(v)
    )
    if failed != ["new_autobiography_updates_self_estimate"]:
        raise RuntimeError(
            f"unexpected v0.33.1 failed gates: {failed}"
        )

    v0332 = json.loads(SOURCE_V0332_RESULT.read_text())
    if v0332.get("timing_consistent") is not False:
        raise RuntimeError(
            "v0.33.2 must have rejected coarse timing explanation"
        )

    if v0331.get("development_test_touched") is not False:
        raise RuntimeError("v0.33 TEST already touched")

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_DEVELOPMENT_EVENT_ORDER_DIAGNOSTIC_V1")
    print("training_performed: false")
    print("read_only_diagnostic: true")
    print("source_v0331_formal_status: FAIL")
    print("source_v0332_timing_consistent: false")
    print("diagnostic_resolution: every_controller_decision_and_every_append")
    print("development_window_reused: 10000-10127")
    print("new_development_window_consumed: false")
    print("development_test_subjects: 20000-20127")
    print("development_test_touched: false")
    print(
        "meaningful_self_estimate_gain:",
        MEANINGFUL_SELF_ESTIMATE_GAIN,
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

    rows = []
    for n, sid in enumerate(DEV_IDS, 1):
        rows.append(
            trace_subject(
                trajectory_model,
                autobiographical_model,
                controller,
                int(sid),
            )
        )
        if n == 1 or n % 32 == 0 or n == 128:
            print(f"event_trace_subjects={n}/128")

    metrics = {
        "predecision_cross_rate":
            mean(float(r["ever_predecision_crossed"]) for r in rows),
        "graduation_rate":
            mean(float(r["ever_graduated"]) for r in rows),
        "postappend_cross_rate":
            mean(float(r["ever_postappend_crossed"]) for r in rows),
        "final_above_rate":
            mean(float(r["final_above_threshold"]) for r in rows),
        "exact_event_alignment_rate":
            mean(float(r["exact_event_alignment"]) for r in rows),
        "graduated_without_predecision_cross_rate":
            mean(
                float(r["graduated_without_predecision_cross"])
                for r in rows
            ),
        "predecision_cross_without_graduation_rate":
            mean(
                float(r["predecision_cross_without_graduation"])
                for r in rows
            ),
        "postappend_cross_without_predecision_cross_rate":
            mean(
                float(r["postappend_cross_without_predecision_cross"])
                for r in rows
            ),
        "post_cross_after_last_fragile_decision_rate":
            mean(
                float(r["post_cross_after_last_fragile_decision"])
                for r in rows
            ),
        "mean_peak_predecision_gain":
            mean(r["peak_predecision_gain"] for r in rows),
        "mean_peak_postappend_gain":
            mean(r["peak_postappend_gain"] for r in rows),
        "mean_final_gain":
            mean(r["final_gain"] for r in rows),
    }

    # Event-order explanation is supported only if the controller transition is
    # exactly aligned with the self-estimate seen at the decision point.
    event_order_consistent = (
        abs(
            metrics["predecision_cross_rate"]
            - metrics["graduation_rate"]
        ) <= 1e-12
        and metrics[
            "graduated_without_predecision_cross_rate"
        ] == 0.0
        and metrics[
            "predecision_cross_without_graduation_rate"
        ] == 0.0
        and metrics[
            "exact_event_alignment_rate"
        ] == 1.0
    )

    if event_order_consistent:
        finding = (
            "GRADUATION_IS_EXACTLY_ALIGNED_WITH_AUTOBIOGRAPHICAL_"
            "SELF_ESTIMATE_AT_THE_CONTROLLER_DECISION_EVENT"
        )
    else:
        finding = (
            "CONTROLLER_GRADUATION_NOT_FULLY_EXPLAINED_BY_"
            "DECISION_TIME_AUTOBIOGRAPHICAL_SELF_ESTIMATE"
        )

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during read-only event diagnostic"
        )

    result = {
        "format":
            "continuous_brain.self_development_event_order_diagnostic_v1",
        "training_performed":
            False,
        "read_only_diagnostic":
            True,
        "source_v0331_formal_status":
            "FAIL",
        "source_v0332_timing_consistent":
            False,
        "split": {
            "reused_dev": [10000, 10127],
            "test": [20000, 20127],
            "test_touched": False,
        },
        "threshold":
            MEANINGFUL_SELF_ESTIMATE_GAIN,
        "metrics":
            metrics,
        "event_order_consistent":
            bool(event_order_consistent),
        "finding":
            finding,
        "subjects":
            rows,
        "protected_unchanged":
            True,
        "development_test_touched":
            False,
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nEVENT_ORDER")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(
        "event_order_consistent:",
        event_order_consistent,
    )
    print("finding:", finding)
    print("PROTECTED_UNCHANGED: true")
    print("DEVELOPMENT_TEST_TOUCHED: false")
    print(
        "SELF_DEVELOPMENT_EVENT_ORDER_DIAGNOSTIC_COMPLETE"
    )
    print("result:", result_path)


if __name__ == "__main__":
    main()
