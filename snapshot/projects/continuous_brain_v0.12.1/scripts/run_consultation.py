"""Run integration smoke, legacy-child evaluation, or the separate reference lab.

Works directly from an extracted archive; no editable installation is necessary.
The evaluator never trains or writes the source individual's checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SETTINGS = dict(option_depth=2, selector_topk=2, beam_width=8, max_option_steps=12,
                discount=.95, min_commit=3, terminate_threshold=.5)


def smoke(args):
    import torch
    from continuous_brain.infant import InfantBrain, InfantConfig
    from continuous_brain.parented_train import save_checkpoint
    from continuous_brain.planning_v103 import evaluate_hierarchical_options
    from continuous_brain.device_v111 import resolve_device
    from continuous_brain.evidence_v114 import sha256, write_json
    from continuous_brain.generalization_gate_v110 import (
        _run_generalized_episode, evaluate_generalization_gate, source_fingerprint, runtime_info)
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    device = resolve_device(args.device)
    started = time.perf_counter()
    model = InfantBrain(InfantConfig.for_profile("proof")).to(device).eval()
    rows = []
    for kind in ("randomized_detour", "two_barrier"):
        paired = [_run_generalized_episode(model, kind, 101, mode=mode, steps=12,
                    device=device, **SETTINGS)
                  for mode in ("reactive", "option_direct", "hierarchical")]
        assert all(r["initial_observation"] == paired[0]["initial_observation"]
                   and r["layout"] == paired[0]["layout"] for r in paired)
        rows.extend(paired)
    with tempfile.TemporaryDirectory(prefix="continuous_brain_smoke_") as tmp:
        checkpoint, report, output = (Path(tmp) / n for n in ("untrained.pt", "unqualified.json", "result.json"))
        # This is an untrained serialization fixture; this metadata declares the
        # module schema expected by the old evaluator, not learned capability.
        save_checkpoint(checkpoint, model, updates=0, parent_id="smoke-fixture",
            dataset_digest="none", metadata={"smoke_only": True,
                "option_cortex_version": "latent-temporal-options-v1"})
        before = sha256(checkpoint)
        unqualified = evaluate_hierarchical_options(checkpoint, episodes=1, steps=8,
            seed_start=101, device=device, **SETTINGS)
        write_json(report, unqualified)
        try:
            evaluate_generalization_gate(checkpoint, replication_report=report,
                episodes=1, steps=8, seed_start=101, device=device, output=output)
        except RuntimeError:
            rejected = True
        else:
            raise AssertionError("An unqualified checkpoint was accepted")
        proc = subprocess.run([sys.executable, "-m", "continuous_brain.parented_cli",
            "evaluate-generalization-gate", "--checkpoint", str(checkpoint),
            "--replication-report", str(report), "--episodes", "1", "--steps", "8",
            "--seed-start", "101", "--device", args.device, "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode != 0 and "RuntimeError" in proc.stderr, proc.stderr
        assert not output.exists()
        assert sha256(checkpoint) == before
    result = dict(passed=True, real_pytorch=True, models_or_simulators_mocked=False,
        model="proof-profile InfantBrain; random untrained weights",
        parameters=sum(p.numel() for p in model.parameters()), runtime=runtime_info(device),
        source_fingerprint=source_fingerprint(), training_performed=False,
        qualification_result=False, unqualified_report_rejected=rejected,
        cli_rejection_verified=True, checkpoint_unchanged=True,
        elapsed_seconds=time.perf_counter()-started, episodes=rows)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    return 0


def evaluate(args):
    import torch
    from continuous_brain.device_v111 import resolve_device
    from continuous_brain.evidence_v114 import sha256, write_json
    from continuous_brain.lifelong.legacy import _evaluate
    from continuous_brain.generalization_gate_v110 import (
        evaluate_generalization_gate, verify_replication_prerequisite,
        source_fingerprint, runtime_info)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    expected_names = {"replication.json", "randomized_detour.json", "two_barrier.json", "sequence_summary.json"}
    if checkpoint.parent == out and checkpoint.name in expected_names:
        raise ValueError("Output directory would overwrite the input checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    before = sha256(checkpoint)
    replication_path = out / "replication.json"
    # Existing fixed-layout protocol, with its own source/runtime-bound cache.
    _evaluate(checkpoint, replication_path, device)
    try:
        verify_replication_prerequisite(replication_path, checkpoint)
    except RuntimeError as error:
        write_json(out / "sequence_summary.json", dict(
            status="stopped_at_replication", reason=str(error), checkpoint_sha256=before,
            recommendation="Diagnose the exact checkpoint and fixed-layout traces before training or generalization."))
        print(str(error), file=sys.stderr)
        return 2
    results = {}
    for kind, reserved in (("randomized_detour", 61001), ("two_barrier", 71001)):
        seed = reserved if args.role == "confirmation" else 41001
        output = out / (kind + ".json")
        cached = json.loads(output.read_text()) if output.is_file() else {}
        exp = cached.get("experiment", {})
        matches = (cached.get("checkpoint_sha256") == before
            and cached.get("source_fingerprint") == source_fingerprint()
            and cached.get("runtime") == runtime_info(device)
            and cached.get("kind") == kind and cached.get("evaluation_role") == args.role
            and exp.get("seed_start") == seed and exp.get("paired_episodes") == 100
            and exp.get("steps_per_episode") == 160
            and all(exp.get(k) == v for k, v in SETTINGS.items()))
        if matches:
            print("Reusing matching completed result:", output.name, flush=True)
            result = cached
        else:
            result = evaluate_generalization_gate(checkpoint, replication_report=replication_path,
                kind=kind, episodes=100, steps=160, seed_start=seed,
                evaluation_role=args.role, device=device, output=output, **SETTINGS)
        results[kind] = {k: result[k] for k in ("benchmark_threshold_passed",
            "completion_evidence_status", "confirmation_supported", "diagnosis")}
    if sha256(checkpoint) != before:
        raise RuntimeError("Checkpoint changed during evaluation")
    write_json(out / "sequence_summary.json", dict(status="completed", role=args.role,
        checkpoint_sha256=before, checkpoint_unchanged=True, training_performed=False, results=results))
    print("Reports saved in", out)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="Real untrained-model execution; no qualification claim")
    s.add_argument("--device", default="cpu"); s.add_argument("--threads", type=int, default=2)
    s.add_argument("--output", default="results/consultation_smoke.json"); s.set_defaults(func=smoke)
    s = sub.add_parser("evaluate", help="Replication first, then both paired generalization diagnostics")
    s.add_argument("--checkpoint", required=True); s.add_argument("--output-dir", required=True)
    s.add_argument("--device", default="cpu"); s.add_argument("--threads", type=int, default=2)
    s.add_argument("--role", choices=["diagnostic", "confirmation"], default="diagnostic")
    s.set_defaults(func=evaluate)
    s = sub.add_parser("reference", help="Separate NumPy-only mock-grid baseline experiment")
    s.set_defaults(func=lambda args: subprocess.call([sys.executable,
        str(ROOT / "consultation/reference_lab/experiment_foraging_options.py")]))
    args = p.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Stopped. Completed evaluation reports remain; the unfinished evaluation restarts on rerun.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
