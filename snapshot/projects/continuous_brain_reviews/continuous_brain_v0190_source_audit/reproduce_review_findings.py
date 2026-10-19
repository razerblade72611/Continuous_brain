"""Bounded synthetic checks of the supplied decision script and shell runner.

No torch, trained models, GPU, or user checkpoint is used. The runner subprocess
gets a disposable PROJECT_ROOT and a stub Python executable that simulates an
evaluation crash. This is software-failure evidence, not a capability test.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
DECIDER = SOURCE / "decide_recurrent_option_progress_train_v1.py"
RUNNER = SOURCE / "run_recurrent_option_progress_v1.sh"
RULES = ("invert_lr", "vertical", "vertical_reverse")


def synthetic_report(none_diff=0.0):
    summary = {
        mode: {**{rule: value for rule in RULES}, "six_way": value}
        for mode, value in (("normal", 1.0), ("donor", 1.0),
                            ("zero", 0.0), ("off", 0.0))
    }
    summary.update({
        "pre_response_action_preservation": 1.0,
        "pre_gate_off_rate": 1.0,
        "response_gate_on_rate": 1.0,
        "pre_gate_abs_mean": 0.0,
        "response_gate_mean": 1.0,
        "cue_gate_gap_mean": 0.0,
        "cue_gate_gap_p99": 0.0,
        "none_max_diff": none_diff,
    })
    return {
        "format": "recurrent_option_progress_final_train_v1",
        "split": "train_only", "summary": summary,
        "factor_summary": {}, "intent_gain": 1.0,
        "v3_dev_touched": False, "v3_test_touched": False,
    }


def check_decider(none_diff):
    with tempfile.TemporaryDirectory(prefix="cb19_decider_") as td:
        root = Path(td)
        src, dst = root / "input.json", root / "decision.json"
        src.write_text(json.dumps(synthetic_report(none_diff)))
        proc = subprocess.run(
            [sys.executable, str(DECIDER), "--input", str(src),
             "--output", str(dst)], capture_output=True, text=True, timeout=10)
        decision = json.loads(dst.read_text())
        return {"exit_code": proc.returncode, "accepted": decision["pass"],
                "factor_failures": len(decision["factor_failures"]),
                "nonzero_none_difference": none_diff,
                "raw_records_supplied": 0}


def check_runner(stale_report):
    with tempfile.TemporaryDirectory(prefix="cb19_runner_") as td:
        root = Path(td)
        state = root / "state/recurrent_option_progress_v1"
        state.mkdir(parents=True)
        # A marker, not a torch checkpoint. Only the stub below reads it.
        (state / "latest.portable.pt").write_text("1600")
        if stale_report:
            (state / "final_train_u1600.json").write_text(
                json.dumps(synthetic_report()))
        bin_dir = root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "python"
        stub.write_text("#!" + sys.executable + "\n" + f'''
import os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path("state/recurrent_option_progress_v1")
if args[0] == "-":
    print(Path(args[1]).read_text())
elif args[0].startswith("qualify_"):
    print("RECURRENT_OPTION_PROGRESS_QUALIFICATION_V1: PASS")
    print("frozen_v018_what_bit_exact: true")
elif args[0].startswith("train_"):
    target = args[args.index("--target-update") + 1]
    (state / "latest.portable.pt").write_text(target)
    with (state / "SIMULATED_TRAINING_REQUESTS.txt").open("a") as f:
        f.write(target + "\\n")
    print("RECURRENT_OPTION_PROGRESS_TRAINING_COMPLETE")
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("frozen_v018_what_bit_exact: true")
elif args[0].startswith("evaluate_"):
    print("INJECTED_EVALUATOR_CRASH", flush=True)
    sys.exit(23)
elif args[0].startswith("decide_"):
    os.execv({sys.executable!r}, [{sys.executable!r}, {str(DECIDER)!r}] + args[1:])
else:
    raise RuntimeError(args)
''')
        stub.chmod(0o755)
        env = dict(os.environ, PROJECT_ROOT=str(root),
                   PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        proc = subprocess.run(["bash", str(RUNNER)], env=env,
                              capture_output=True, text=True, timeout=15)
        output = proc.stdout + proc.stderr
        requests = state / "SIMULATED_TRAINING_REQUESTS.txt"
        return {
            "stale_report_supplied": stale_report,
            "exit_code": proc.returncode,
            "evaluator_crash_executed": "INJECTED_EVALUATOR_CRASH" in output,
            "runner_reported_pass": "RECURRENT_OPTION_PROGRESS_RUN: PASS" in output,
            "runner_reported_max_epochs": "FAIL_AFTER_MAX_EPOCHS" in output,
            "simulated_extra_training_targets": requests.read_text().splitlines()
                if requests.exists() else [],
        }


def main():
    results = {
        "scope": "synthetic software checks only; no model training or inference",
        "empty_evidence_accepted": check_decider(0.0),
        "nonzero_none_difference_accepted": check_decider(5e-7),
        "evaluator_crash_without_stale_report": check_runner(False),
        "evaluator_crash_with_stale_report": check_runner(True),
    }
    assert results["empty_evidence_accepted"]["accepted"]
    assert results["nonzero_none_difference_accepted"]["accepted"]
    assert results["evaluator_crash_without_stale_report"][
        "simulated_extra_training_targets"] == ["3200", "4800"]
    assert results["evaluator_crash_with_stale_report"]["runner_reported_pass"]
    (ROOT / "reproduced_findings.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
