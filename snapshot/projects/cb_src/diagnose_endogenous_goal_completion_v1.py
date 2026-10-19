#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    DevelopmentalState,
    reliability_from_memory,
)
from evaluate_endogenous_goal_genesis_v1 import (
    AUTOBIO_MODEL,
    DEV_IDS,
    PROTECTED as V035_PROTECTED,
    SOURCE_V034_TEST,
    TRAJECTORY_MODEL,
    actual_fragile_primary,
    discrepancy_scalar,
    execute_development_episode,
    initial_capabilities,
    mean,
    reliability_for_primary,
    stable_primary,
    build_bootstrap_memory,
)

SOURCE_V035_RESULT = Path(
    "state/endogenous_goal_genesis_v1/result.json"
)

CHECKPOINTS = (8, 16, 24, 32)
MAX_EPISODES = max(CHECKPOINTS)

PROTECTED = dict(V035_PROTECTED)
PROTECTED["v035_result"] = SOURCE_V035_RESULT


def trace_subject(
    trajectory_model,
    autobiographical_model,
    generator,
    subject_id: int,
):
    memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )

    initial_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    goal = generator.generate(
        initial_rel,
        generation=0,
        cycle=0,
    )
    if goal is None:
        raise RuntimeError(
            "normal DEV autobiography unexpectedly produced no goal"
        )

    primary = int(goal.primary)
    baseline = float(goal.baseline_reliability)
    threshold = float(goal.target_reliability)

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )
    initial_capability = float(
        state.capability_for(primary)
    )

    bootstrap_discrepancy = mean(
        discrepancy_scalar(r)
        for r in memory.records
        if int(r.primary) == primary
    )

    # Exact v0.35.0 interruption semantics.
    interrupt_record, _ = execute_development_episode(
        trajectory_model,
        subject_id,
        -1,
        stable_primary(subject_id),
        STRATEGY_DIRECT,
        state,
    )
    memory.append(interrupt_record)

    after_interrupt_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    after_interrupt_target = reliability_for_primary(
        after_interrupt_rel,
        primary,
    )
    interrupt_shift = after_interrupt_target - baseline

    reliability_trace = []
    discrepancy_trace = []
    capability_trace = []
    first_cross = None

    # Decision immediately after interruption, matching original control flow.
    current_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    current_target = reliability_for_primary(
        current_rel,
        primary,
    )
    reliability_trace.append(current_target)
    capability_trace.append(
        float(state.capability_for(primary))
    )

    if current_target >= threshold:
        first_cross = 0

    for episode in range(MAX_EPISODES):
        if first_cross is not None:
            # Continue the diagnostic anyway so we can observe post-cross drift.
            pass

        record, metrics = execute_development_episode(
            trajectory_model,
            subject_id,
            episode,
            primary,
            STRATEGY_SCAFFOLD,
            state,
        )
        memory.append(record)
        state.cycle += 1

        discrepancy_trace.append(
            float(metrics["discrepancy"])
        )

        current_rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        current_target = reliability_for_primary(
            current_rel,
            primary,
        )
        reliability_trace.append(current_target)
        capability_trace.append(
            float(state.capability_for(primary))
        )

        if (
            first_cross is None
            and current_target >= threshold
        ):
            first_cross = episode + 1

    gains = [
        float(r - baseline)
        for r in reliability_trace
    ]

    completion_by = {
        str(h):
            bool(
                first_cross is not None
                and first_cross <= h
            )
        for h in CHECKPOINTS
    }

    discrepancy_ratio_by = {}
    for h in CHECKPOINTS:
        vals = discrepancy_trace[:h]
        late = mean(vals[-2:]) if vals else 0.0
        discrepancy_ratio_by[str(h)] = (
            late / max(bootstrap_discrepancy, 1e-6)
        )

    capability_gain_by = {
        str(h):
            float(
                capability_trace[h]
                - initial_capability
            )
        for h in CHECKPOINTS
    }

    return {
        "subject_id": int(subject_id),
        "generated_primary": primary,
        "actual_fragile_primary":
            actual_fragile_primary(subject_id),
        "baseline_reliability": baseline,
        "target_reliability": threshold,
        "after_interrupt_reliability":
            after_interrupt_target,
        "interrupt_shift": interrupt_shift,
        "first_cross_episode": first_cross,
        "completion_by": completion_by,
        "reliability_trace": reliability_trace,
        "gain_trace": gains,
        "peak_gain_8":
            max(gains[:9]),
        "final_gain_8":
            gains[8],
        "peak_gain_32":
            max(gains),
        "final_gain_32":
            gains[32],
        "bootstrap_discrepancy":
            bootstrap_discrepancy,
        "discrepancy_ratio_by":
            discrepancy_ratio_by,
        "capability_gain_by":
            capability_gain_by,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/endogenous_goal_completion_diagnostic_v1",
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
    if v034.get("test_admission_passed") is not True:
        raise RuntimeError("v0.34.1 PASS required")

    v035 = json.loads(SOURCE_V035_RESULT.read_text())
    if v035.get("development_gate_passed") is not False:
        raise RuntimeError(
            "v0.35.0 formal DEV FAIL required"
        )

    failed = sorted(
        k for k, v in v035["checks"].items()
        if not bool(v)
    )
    if failed != [
        "self_generated_goal_completes_from_self_estimate"
    ]:
        raise RuntimeError(
            f"unexpected v0.35.0 failed gates: {failed}"
        )

    if v035.get("v035_test_touched") is not False:
        raise RuntimeError(
            "v0.35 TEST must remain sealed"
        )

    before = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }

    print("ENDOGENOUS_GOAL_COMPLETION_DIAGNOSTIC_V1")
    print("training_performed: false")
    print("read_only_diagnostic: true")
    print("source_v0350_formal_status: FAIL")
    print(
        "source_only_failure:",
        "self_generated_goal_completes_from_self_estimate",
    )
    print("development_window_reused: 90000-90127")
    print("new_development_window_consumed: false")
    print("test_subjects: 100000-100127")
    print("test_touched: false")
    print("diagnostic_max_target_episodes: 32")
    print("completion_checkpoints: 8,16,24,32")

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
            trace_subject(
                trajectory_model,
                autobiographical_model,
                generator,
                int(sid),
            )
        )
        if n == 1 or n % 32 == 0 or n == len(ids):
            print(
                f"completion_diagnostic_subjects={n}/128"
            )

    completion_rates = {
        str(h):
            mean(
                float(r["completion_by"][str(h)])
                for r in rows
            )
        for h in CHECKPOINTS
    }

    mean_interrupt_shift = mean(
        r["interrupt_shift"] for r in rows
    )

    mean_peak_gain_8 = mean(
        r["peak_gain_8"] for r in rows
    )
    mean_final_gain_8 = mean(
        r["final_gain_8"] for r in rows
    )
    mean_peak_gain_32 = mean(
        r["peak_gain_32"] for r in rows
    )
    mean_final_gain_32 = mean(
        r["final_gain_32"] for r in rows
    )

    never_cross_32 = mean(
        float(r["first_cross_episode"] is None)
        for r in rows
    )

    discrepancy_rates = {
        str(h):
            mean(
                float(
                    r["discrepancy_ratio_by"][str(h)]
                    <= .75
                )
                for r in rows
            )
        for h in CHECKPOINTS
    }

    capability_means = {
        str(h):
            mean(
                r["capability_gain_by"][str(h)]
                for r in rows
            )
        for h in CHECKPOINTS
    }

    # Among subjects that do NOT complete by 8, determine whether objective
    # developmental change nevertheless occurred.
    failures8 = [
        r for r in rows
        if not r["completion_by"]["8"]
    ]
    failures8_discrepancy_success = mean(
        float(
            r["discrepancy_ratio_by"]["8"] <= .75
        )
        for r in failures8
    )
    failures8_capability_mean = mean(
        r["capability_gain_by"]["8"]
        for r in failures8
    )

    metrics = {
        "completion_rate_by_episode":
            completion_rates,
        "mean_interrupt_reliability_shift":
            mean_interrupt_shift,
        "mean_peak_gain_8":
            mean_peak_gain_8,
        "mean_final_gain_8":
            mean_final_gain_8,
        "mean_peak_gain_32":
            mean_peak_gain_32,
        "mean_final_gain_32":
            mean_final_gain_32,
        "never_cross_by_32_rate":
            never_cross_32,
        "discrepancy_reduction_rate_by_episode":
            discrepancy_rates,
        "mean_capability_gain_by_episode":
            capability_means,
        "eight_episode_noncompleters":
            len(failures8),
        "noncompleters8_discrepancy_reduction_rate":
            failures8_discrepancy_success,
        "noncompleters8_mean_capability_gain":
            failures8_capability_mean,
    }

    c8 = completion_rates["8"]
    c16 = completion_rates["16"]
    c32 = completion_rates["32"]

    if c16 >= .80:
        finding = (
            "EIGHT_EPISODE_HORIZON_TOO_SHORT_FOR_EXISTING_"
            "SELF_ESTIMATE_COMPLETION_CRITERION"
        )
    elif c32 >= .80:
        finding = (
            "SELF_ESTIMATE_COMPLETION_REQUIRES_LONGER_"
            "DEVELOPMENTAL_HORIZON"
        )
    elif (
        discrepancy_rates["8"] >= .80
        and failures8_discrepancy_success >= .80
        and failures8_capability_mean >= .35
    ):
        finding = (
            "AUTOBIOGRAPHICAL_RELIABILITY_TARGET_MISALIGNED_"
            "WITH_OBSERVED_DEVELOPMENTAL_CHANGE"
        )
    else:
        finding = (
            "DEVELOPMENTAL_CHANGE_INSUFFICIENT_TO_EXPLAIN_"
            "COMPLETION_FAILURE"
        )

    after = {
        name: sha256(Path(path))
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during diagnostic"
        )

    result = {
        "format":
            "continuous_brain.endogenous_goal_completion_diagnostic_v1",
        "training_performed":
            False,
        "read_only_diagnostic":
            True,
        "source_v0350_formal_status":
            "FAIL",
        "source_only_failure":
            "self_generated_goal_completes_from_self_estimate",
        "split": {
            "reused_dev": [90000, 90127],
            "test": [100000, 100127],
            "test_touched": False,
        },
        "metrics":
            metrics,
        "finding":
            finding,
        "subjects":
            rows,
        "protected_unchanged":
            True,
        "v035_test_touched":
            False,
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nCOMPLETION_DYNAMICS")
    for h in CHECKPOINTS:
        print(
            f"completion_rate_by_{h}:",
            completion_rates[str(h)],
        )
    print(
        "mean_interrupt_reliability_shift:",
        mean_interrupt_shift,
    )
    print("mean_peak_gain_8:", mean_peak_gain_8)
    print("mean_final_gain_8:", mean_final_gain_8)
    print("mean_peak_gain_32:", mean_peak_gain_32)
    print("mean_final_gain_32:", mean_final_gain_32)
    print("never_cross_by_32_rate:", never_cross_32)

    print("\nDEVELOPMENTAL_CHANGE")
    for h in CHECKPOINTS:
        print(
            f"discrepancy_reduction_rate_by_{h}:",
            discrepancy_rates[str(h)],
        )
        print(
            f"mean_capability_gain_by_{h}:",
            capability_means[str(h)],
        )
    print(
        "eight_episode_noncompleters:",
        len(failures8),
    )
    print(
        "noncompleters8_discrepancy_reduction_rate:",
        failures8_discrepancy_success,
    )
    print(
        "noncompleters8_mean_capability_gain:",
        failures8_capability_mean,
    )

    print("\nFINDING")
    print("finding:", finding)
    print("PROTECTED_UNCHANGED: true")
    print("V035_TEST_TOUCHED: false")
    print(
        "ENDOGENOUS_GOAL_COMPLETION_DIAGNOSTIC_COMPLETE"
    )
    print("result:", result_path)


if __name__ == "__main__":
    main()
