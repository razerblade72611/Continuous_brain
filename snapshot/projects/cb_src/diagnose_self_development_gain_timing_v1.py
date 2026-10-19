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
    HistoryRelativeDevelopmentalController,
    MEANINGFUL_SELF_ESTIMATE_GAIN,
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
)
from evaluate_self_development_hysteresis_v1 import (
    AUTOBIO_MODEL,
    DEV_IDS,
    SOURCE_V033_RESULT,
    TRAJECTORY_MODEL,
    run_subject_hysteresis,
)

SOURCE_V0331_RESULT = Path(
    "state/self_development_hysteresis_v1/result.json"
)

PROTECTED = {
    "v0330_result": SOURCE_V033_RESULT,
    "v0331_result": SOURCE_V0331_RESULT,
    "autobio_model": AUTOBIO_MODEL,
    "trajectory_model": TRAJECTORY_MODEL,
    "v032_test":
        Path("state/autobiographical_self_discrepancy_neutral_test_v1/result.json"),
    "v031_test":
        Path("state/recurrent_self_trajectory_test_v1/result.json"),
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_development_gain_timing_diagnostic_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    source = json.loads(SOURCE_V0331_RESULT.read_text())
    if source.get("repair_dev_gate_passed") is not False:
        raise RuntimeError("v0.33.1 formal DEV FAIL required")

    failed = sorted(
        k
        for k, v in source["checks"].items()
        if not bool(v)
    )
    if failed != ["new_autobiography_updates_self_estimate"]:
        raise RuntimeError(
            f"unexpected v0.33.1 failed gates: {failed}"
        )

    if source.get("development_test_touched") is not False:
        raise RuntimeError("v0.33 TEST already touched")

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_DEVELOPMENT_GAIN_TIMING_DIAGNOSTIC_V1")
    print("training_performed: false")
    print("read_only_diagnostic: true")
    print("source_v0331_formal_status: FAIL")
    print("source_only_failure: new_autobiography_updates_self_estimate")
    print("development_window_reused: 10000-10127")
    print("new_development_window_consumed: false")
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
        s = run_subject_hysteresis(
            trajectory_model,
            autobiographical_model,
            controller,
            int(sid),
            freeze_decision_history=False,
        )

        baseline = float(s["initial_fragile_reliability"])
        trace = [float(x) for x in s["reliability_trace"]]
        gains = [x - baseline for x in trace]
        peak_gain = max(gains)
        final_gain = gains[-1]

        first_cross_cycle = None
        for cycle_idx, gain in enumerate(gains):
            if gain >= MEANINGFUL_SELF_ESTIMATE_GAIN:
                first_cross_cycle = cycle_idx
                break

        strategies = list(s["fragile_strategies"])
        first_direct_episode = None
        for episode_idx, strategy in enumerate(strategies):
            if int(strategy) == STRATEGY_DIRECT:
                first_direct_episode = episode_idx
                break

        ever_crossed = first_cross_cycle is not None
        final_above = final_gain >= MEANINGFUL_SELF_ESTIMATE_GAIN
        graduated = bool(s["switched_to_direct"])

        rows.append(
            {
                "subject_id": int(sid),
                "baseline_reliability": baseline,
                "peak_gain": peak_gain,
                "final_gain": final_gain,
                "ever_crossed_gain_threshold": ever_crossed,
                "final_above_gain_threshold": final_above,
                "crossed_then_ended_below":
                    bool(ever_crossed and not final_above),
                "graduated_to_direct": graduated,
                "first_cross_cycle": first_cross_cycle,
                "first_direct_episode": first_direct_episode,
                "reliability_trace": trace,
                "fragile_strategies": strategies,
            }
        )

        if n == 1 or n % 32 == 0 or n == 128:
            print(f"diagnostic_subjects={n}/128")

    ever_rate = mean(
        float(r["ever_crossed_gain_threshold"])
        for r in rows
    )
    final_rate = mean(
        float(r["final_above_gain_threshold"])
        for r in rows
    )
    graduated_rate = mean(
        float(r["graduated_to_direct"])
        for r in rows
    )
    crossed_then_below_rate = mean(
        float(r["crossed_then_ended_below"])
        for r in rows
    )

    peak_gain_mean = mean(r["peak_gain"] for r in rows)
    final_gain_mean = mean(r["final_gain"] for r in rows)
    post_peak_drift_mean = mean(
        r["final_gain"] - r["peak_gain"]
        for r in rows
    )

    crossed_and_graduated = mean(
        float(
            r["ever_crossed_gain_threshold"]
            and r["graduated_to_direct"]
        )
        for r in rows
    )
    graduated_without_crossing = mean(
        float(
            r["graduated_to_direct"]
            and not r["ever_crossed_gain_threshold"]
        )
        for r in rows
    )
    crossed_without_graduating = mean(
        float(
            r["ever_crossed_gain_threshold"]
            and not r["graduated_to_direct"]
        )
        for r in rows
    )

    # If the timing interpretation is right, graduation should match the
    # occurrence of a meaningful self-estimate increase, while the final-only
    # metric can be lower because later DIRECT experience updates autobiography.
    timing_consistent = (
        abs(ever_rate - graduated_rate) <= 1e-12
        and graduated_without_crossing == 0.0
        and crossed_without_graduating == 0.0
        and ever_rate > final_rate
    )

    finding = (
        "MEANINGFUL_SELF_ESTIMATE_GAIN_IS_TRANSIENT_AND_CAUSALLY_PRECEDES_GRADUATION"
        if timing_consistent
        else
        "FINAL_GAIN_FAILURE_NOT_EXPLAINED_BY_POST_GRADUATION_DRIFT"
    )

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during read-only diagnostic"
        )

    result = {
        "format":
            "continuous_brain.self_development_gain_timing_diagnostic_v1",
        "training_performed": False,
        "read_only_diagnostic": True,
        "source_v0331_formal_status": "FAIL",
        "source_only_failure":
            "new_autobiography_updates_self_estimate",
        "split": {
            "reused_dev": [10000, 10127],
            "test": [20000, 20127],
            "test_touched": False,
        },
        "threshold":
            MEANINGFUL_SELF_ESTIMATE_GAIN,
        "metrics": {
            "ever_crossed_gain_threshold_rate":
                ever_rate,
            "final_above_gain_threshold_rate":
                final_rate,
            "graduated_to_direct_rate":
                graduated_rate,
            "crossed_then_ended_below_rate":
                crossed_then_below_rate,
            "mean_peak_gain":
                peak_gain_mean,
            "mean_final_gain":
                final_gain_mean,
            "mean_post_peak_drift":
                post_peak_drift_mean,
            "crossed_and_graduated_rate":
                crossed_and_graduated,
            "graduated_without_crossing_rate":
                graduated_without_crossing,
            "crossed_without_graduating_rate":
                crossed_without_graduating,
        },
        "timing_consistent":
            bool(timing_consistent),
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
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTIMING")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")
    print("timing_consistent:", timing_consistent)
    print("finding:", finding)
    print("PROTECTED_UNCHANGED: true")
    print("DEVELOPMENT_TEST_TOUCHED: false")
    print("SELF_DEVELOPMENT_GAIN_TIMING_DIAGNOSTIC_COMPLETE")
    print("result:", result_path)


if __name__ == "__main__":
    main()
