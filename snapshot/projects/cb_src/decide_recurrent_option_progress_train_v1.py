#!/usr/bin/env python3
"""Frozen TRAIN gate for v0.19 recurrent option-progress controller."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULES = ("invert_lr", "vertical", "vertical_reverse")

MIN_PAIR = .95
MIN_DONOR = .95
MAX_OFF = .05
MAX_ZERO = .05
MIN_SIX = .90
MIN_PRE_KEEP = .99
MIN_GATE_OFF = .99
MIN_GATE_ON = .99
MAX_PRE_GATE_ABS = .10
MIN_RESPONSE_GATE_MEAN = .90
MAX_CUE_GAP_MEAN = .05
MAX_CUE_GAP_P99 = .15
MAX_NONE = 1e-6

MIN_FACTOR_PAIR = .85
MIN_FACTOR_DONOR = .85
MIN_FACTOR_SIX = .75
MIN_FACTOR_PRE_KEEP = .95
MIN_FACTOR_GATE_OFF = .95
MIN_FACTOR_GATE_ON = .95
MAX_FACTOR_CUE_GAP = .10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.input).read_text())
    if (
        p.get("format")
        != "recurrent_option_progress_final_train_v1"
        or p.get("split") != "train_only"
        or p.get("v3_dev_touched")
        or p.get("v3_test_touched")
    ):
        raise RuntimeError("invalid TRAIN artifact")

    s = p["summary"]
    checks = {}

    for rule in RULES:
        checks[f"{rule}_pair"] = (
            s["normal"][rule] >= MIN_PAIR
        )
        checks[f"{rule}_donor"] = (
            s["donor"][rule] >= MIN_DONOR
        )
        checks[f"{rule}_off"] = (
            s["off"][rule] <= MAX_OFF
        )
        checks[f"{rule}_zero"] = (
            s["zero"][rule] <= MAX_ZERO
        )

    checks["six_way"] = (
        s["normal"]["six_way"] >= MIN_SIX
    )
    checks["pre_response_action_preservation"] = (
        s["pre_response_action_preservation"]
        >= MIN_PRE_KEEP
    )
    checks["pre_gate_off_rate"] = (
        s["pre_gate_off_rate"] >= MIN_GATE_OFF
    )
    checks["response_gate_on_rate"] = (
        s["response_gate_on_rate"] >= MIN_GATE_ON
    )
    checks["pre_gate_abs_mean"] = (
        s["pre_gate_abs_mean"] <= MAX_PRE_GATE_ABS
    )
    checks["response_gate_mean"] = (
        s["response_gate_mean"]
        >= MIN_RESPONSE_GATE_MEAN
    )
    checks["cue_gate_gap_mean"] = (
        s["cue_gate_gap_mean"]
        <= MAX_CUE_GAP_MEAN
    )
    checks["cue_gate_gap_p99"] = (
        s["cue_gate_gap_p99"]
        <= MAX_CUE_GAP_P99
    )
    checks["none_exact_noop"] = (
        s["none_max_diff"] <= MAX_NONE
    )

    failures = []
    for field, levels in p["factor_summary"].items():
        for value, x in levels.items():
            ok = (
                x["six_way_full"] >= MIN_FACTOR_SIX
                and x["pre_response_action_preservation"]
                    >= MIN_FACTOR_PRE_KEEP
                and x["pre_gate_off_rate"]
                    >= MIN_FACTOR_GATE_OFF
                and x["response_gate_on_rate"]
                    >= MIN_FACTOR_GATE_ON
                and x["cue_gate_gap_mean"]
                    <= MAX_FACTOR_CUE_GAP
                and all(
                    x[f"{rule}_pair_full"]
                        >= MIN_FACTOR_PAIR
                    and x[f"{rule}_donor"]
                        >= MIN_FACTOR_DONOR
                    for rule in RULES
                )
            )
            if not ok:
                failures.append({
                    "factor": field,
                    "value": value,
                    **x,
                })

    checks["factor_robustness"] = not failures
    passed = all(checks.values())

    result = {
        "format":
            "recurrent_option_progress_train_decision_v1",
        "pass": passed,
        "checks": checks,
        "factor_failures": failures,
        "summary": s,
        "intent_gain": p["intent_gain"],
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("RECURRENT_OPTION_PROGRESS_TRAIN_DECISION_V1")
    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={s['normal'][rule]:.4f} "
            f"donor={s['donor'][rule]:.4f} "
            f"off={s['off'][rule]:.4f} "
            f"zero={s['zero'][rule]:.4f}"
        )
    print(
        f"six_way={s['normal']['six_way']:.4f} "
        f"preKeep={s['pre_response_action_preservation']:.4f} "
        f"gateOff={s['pre_gate_off_rate']:.4f} "
        f"gateOn={s['response_gate_on_rate']:.4f} "
        f"preGateAbs={s['pre_gate_abs_mean']:.4f} "
        f"responseGateMean={s['response_gate_mean']:.4f} "
        f"cueGap={s['cue_gate_gap_mean']:.4f} "
        f"cueGapP99={s['cue_gate_gap_p99']:.4f} "
        f"none_diff={s['none_max_diff']:.3g}"
    )
    print("intent_gain:", p["intent_gain"])
    print()
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("factor_failure_count:", len(failures))
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print(
        "RECURRENT_OPTION_PROGRESS_TRAIN_GATE:",
        "PASS" if passed else "FAIL",
    )
    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
