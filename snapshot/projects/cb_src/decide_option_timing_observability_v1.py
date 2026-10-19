#!/usr/bin/env python3
"""Interpret the frozen TRAIN-only timing observability probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PASS_BOUNDARY = 0.95
PASS_PRE_OFF = 0.95
PASS_RESPONSE_ON = 0.95
SAFE_CUE_GAP = 0.05


def observable(r):
    return (
        r["mean_boundary_accuracy"] >= PASS_BOUNDARY
        and r["mean_pre_off_rate_all"] >= PASS_PRE_OFF
        and r["mean_response_on_rate"] >= PASS_RESPONSE_ON
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.input).read_text())
    if p.get("format") != "option_timing_observability_probe_v1":
        raise RuntimeError("unexpected probe format")
    if p.get("v3_dev_touched") or p.get("v3_test_touched"):
        raise RuntimeError("held-out contamination")

    c = p["candidates"]
    workspace = c["workspace_plus_logits"]["mlp"]
    hidden_in = c["hidden_in"]["mlp"]
    hidden_out = c["hidden_out"]["mlp"]
    raw_obs = c["raw_obs"]["mlp"]
    rich = c["hidden_out_workspace_logits"]["mlp"]

    diagnosis = []
    recommendation = None

    if observable(workspace):
        diagnosis.append("CURRENT_GATE_INPUT_IS_OBSERVABLE")
        recommendation = (
            "The v0.18 gate inputs contain enough timing information. "
            "Keep the interface and redesign gate optimization/supervision."
        )
    elif observable(hidden_out) and hidden_out["mean_cue_pair_gap"] <= SAFE_CUE_GAP:
        diagnosis.append("POST_CORE_HIDDEN_EXPOSES_TIMING")
        recommendation = (
            "Use protected post-core hidden as input to the scalar WHEN gate. "
            "Keep WHAT isolated on canonical recall+rule and enforce cue-invariant timing."
        )
    elif observable(hidden_in) and hidden_in["mean_cue_pair_gap"] <= SAFE_CUE_GAP:
        diagnosis.append("RECURRENT_HIDDEN_IN_EXPOSES_TIMING")
        recommendation = (
            "Use protected recurrent hidden-in as scalar WHEN-gate context, "
            "with cue-invariance checks."
        )
    elif observable(raw_obs):
        diagnosis.append("RAW_STATE_HAS_TIMING_BUT_PROTECTED_ENCODER_DROPS_IT")
        recommendation = (
            "Repair state/progress representation before further executive training."
        )
    elif observable(rich):
        diagnosis.append("TIMING_REQUIRES_COMBINED_PROTECTED_STATE")
        recommendation = (
            "Use the combined protected hidden/workspace/action state for the scalar gate."
        )
    else:
        diagnosis.append("STATIC_TIMING_NOT_OBSERVABLE")
        recommendation = (
            "Stop static gate variants. Introduce a recurrent option-state/progress "
            "controller that learns temporal phase across steps."
        )

    result = {
        "format": "option_timing_observability_decision_v1",
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "thresholds": {
            "boundary": PASS_BOUNDARY,
            "pre_off": PASS_PRE_OFF,
            "response_on": PASS_RESPONSE_ON,
            "safe_cue_gap": SAFE_CUE_GAP,
        },
        "actual_v018_gate": p["actual_v018_gate"],
        "workspace_plus_logits_mlp": workspace,
        "hidden_in_mlp": hidden_in,
        "hidden_out_mlp": hidden_out,
        "raw_obs_mlp": raw_obs,
        "hidden_out_workspace_logits_mlp": rich,
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("OPTION_TIMING_OBSERVABILITY_DECISION_V1")
    for item in diagnosis:
        print("diagnosis:", item)
    print("recommendation:", recommendation)
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("OPTION_TIMING_OBSERVABILITY_DECISION_COMPLETE")


if __name__ == "__main__":
    main()
