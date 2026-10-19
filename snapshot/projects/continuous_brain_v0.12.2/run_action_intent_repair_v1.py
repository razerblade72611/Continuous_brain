#!/usr/bin/env python3
"""Resume the bounded v0.17 schedule, then evaluate actual action control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from continuous_brain.action_intent_integrity_v1 import sha, validate_payload


def run_script(name, arguments, logfile, *, allowed=(0,)):
    command = [sys.executable, "-u", name, *map(str, arguments)]
    print("RUN:", " ".join(command), flush=True)
    with Path(logfile).open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if status not in allowed:
        raise RuntimeError(f"{name} failed with exit {status}; stopped without advancing training")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("directml", "cpu"), default="directml")
    parser.add_argument("--audit-only", action="store_true", help="Evaluate existing state; no optimizer steps")
    args = parser.parse_args()
    state = Path("state/action_intent_executive_v1")
    output = state / "control_repair_v1_1"
    output.mkdir(parents=True, exist_ok=True)
    latest = state / "latest.portable.pt"
    current = 0
    if latest.is_file():
        current = validate_payload(torch.load(latest, map_location="cpu", weights_only=True))
    elif args.audit_only:
        raise FileNotFoundError("No v0.17 executive checkpoint exists yet; omit --audit-only for its bounded training run")
    trained = False
    device = ["--device", args.device]
    if not args.audit_only:
        run_script("qualify_action_intent_executive_v1.py", device, output / "qualification.log")
    mapping = None
    mapping_path = None
    mapping_pass = False
    stages = [(1600, .0002), (3200, .0001), (4800, .00005)]
    if args.audit_only:
        stages = [(current, None)]
    for target, lr in stages:
        if current > target:
            continue
        if current < target:
            run_script("train_action_intent_executive_v1.py",
                       ["--state-dir", state, "--target-update", target, "--lr", lr, *device],
                       output / f"train_u{target}.log")
            trained = True
            current = validate_payload(torch.load(latest, map_location="cpu", weights_only=True))
            if current != target:
                raise RuntimeError("trainer did not reach the requested checkpoint")
        if current not in (1600, 3200, 4800):
            break  # Partial checkpoint is useful only for the sampled diagnostic.
        mapping_path = output / f"mapping_u{current}.json"
        decision_path = output / f"mapping_decision_u{current}.json"
        run_script("evaluate_action_intent_executive_final_train_v1.py",
                   ["--checkpoint", latest, "--output", mapping_path, *device],
                   output / f"mapping_u{current}.log")
        status = run_script("decide_action_intent_executive_train_v1.py",
                            ["--input", mapping_path, "--output", decision_path],
                            output / f"decision_u{current}.log", allowed=(0, 4))
        mapping = json.loads(decision_path.read_text())
        mapping_pass = status == 0
        if mapping_pass or args.audit_only:
            break
    conditions = 1600 if mapping_pass and not args.audit_only else 64
    control_path = output / f"control_u{current}_n{conditions}.json"
    run_script("evaluate_action_intent_control_v1.py",
               ["--checkpoint", latest, "--output", control_path, "--conditions", conditions, *device],
               output / f"control_u{current}_n{conditions}.log")
    control = json.loads(control_path.read_text())
    if mapping is not None and mapping["provenance"] != control["provenance"]:
        raise RuntimeError("mapping and control evaluations used different checkpoints or source")
    current_sha = sha(latest)
    if current_sha != control["provenance"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed after evaluation")
    ready = bool(mapping_pass and control["control_contract_passed"])
    if ready:
        status = "READY_FOR_FRESH_V4_GENERALIZATION"
    elif mapping_pass:
        status = "CONTROL_INTEGRATION_REQUIRED" if conditions == 1600 else "FULL_CONTROL_EVALUATION_PENDING"
    elif mapping is None:
        status = "PARTIAL_CHECKPOINT_DIAGNOSTIC_ONLY"
    else:
        status = "TRAIN_MAPPING_FAILED"
    report = {"format": "action_intent_repair_result_v1_1", "status": status,
              "global_update": current, "checkpoint_sha256": current_sha,
              "mapping_pass": mapping_pass, "mapping_report": str(mapping_path) if mapping_path else None,
              "control_report": str(control_path), "control_summary": control["summary"],
              "ready_for_fresh_v4": ready, "generalization_established": False,
              "v3_dev_touched": False, "v3_test_touched": False,
              "checkpoint_training_modified": trained}
    result = output / "result.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print("REPAIR_RESULT:", status)
    print("RESULT_FILE:", result)
    print("No fresh-family generalization or consciousness claim is made by this run.")
    return 0 if ready or args.audit_only else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("RUN_STOPPED:", str(exc), file=sys.stderr)
        sys.exit(2)
