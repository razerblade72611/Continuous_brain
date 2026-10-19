"""Local v0.12.1 entry point: DirectML is explicit; CPU fallback is never silent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time
import webbrowser

import torch

from .device_v111 import resolve_device, device_report
from .lifelong import BUILD
from .lifelong.model import DevelopmentBrain, BrainConfig, model_payload, load_model, observation_tensors, finite_step
from .lifelong.world import ResourceWorld, feedback_integrity_audit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = Path.home() / "ContinuousBrain" / "individual_v120"
DEFAULT_STARTER = ROOT / "assets" / "development_starter.pt"


def print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)


def preflight(spec):
    import copy
    from .lifelong.learning import demonstrations, train_block
    device = resolve_device(spec)
    report = device_report(device, compute_test=True, matrix_size=32)
    report["requested"] = str(spec)
    model = DevelopmentBrain(BrainConfig(width=32, hidden=48)).to(device)
    obs = ResourceWorld(600003).observe()
    import numpy as np
    args = observation_tensors(obs, np.zeros(8, np.float32), np.zeros(4, np.float32), device)
    out = model(*args, model.initial(1, device))
    prediction = model.predict(out["hidden"], torch.eye(5)[1:2].to(device))
    loss = out["logits"].square().mean() + out["q"].square().mean() + prediction["reward"].square().mean()
    opt = torch.optim.Adam(model.parameters(), lr=.0001, foreach=False)
    finite_step(loss, model, opt)
    reference = copy.deepcopy(model)
    teaching = demonstrations(episodes=4, horizon=16)
    lived = copy.deepcopy(teaching)
    for trace in lived:
        for row in trace:
            row["teacher"] = -1
    train_block(model, teaching, lived, updates=2, batch_size=2, sequence=4,
                rng=random.Random(600005), reference=reference)
    report["model_forward_backward_optimizer"] = True
    report["demonstration_and_lived_replay_routes"] = True
    report["build"] = BUILD
    report["feedback_integrity"] = feedback_integrity_audit()
    return report


def bootstrap(args):
    from .lifelong.learning import demonstrations, correction_demonstrations, train_block, evaluate
    from .lifelong.runtime import save_model_file, runtime_signature
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Bootstrap output exists; choose another --output to preserve it")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = DevelopmentBrain().to(device)
    print("Generating labelled demonstration episodes; no external teacher is required.", flush=True)
    data = demonstrations(episodes=args.episodes)
    start = time.monotonic()
    phase, phase_budget = "initial", args.updates
    def progress(update, loss):
        if update == 1 or update % 50 == 0 or update == phase_budget:
            print(f"bootstrap_phase={phase} updates={update}/{phase_budget} loss={loss:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    result, _ = train_block(model, data, [], updates=args.updates, batch_size=args.batch,
                            sequence=args.sequence, lr=.0005, rng=rng, progress=progress)
    rounds = []
    for index in range(args.correction_rounds):
        phase, phase_budget = f"correction_{index+1}", args.correction_updates
        print(f"Collecting training-only corrective feedback, round {index+1}/{args.correction_rounds}.", flush=True)
        correction = correction_demonstrations(model, episodes=64, seed=1500001 + 1000 * index)
        data.extend(correction)
        print("Learning from the combined demonstration and correction data.", flush=True)
        r, _ = train_block(model, data, [], updates=args.correction_updates, batch_size=args.batch,
                           sequence=args.sequence, lr=.0003, rng=rng, progress=progress)
        rounds.append(r)
    print("Measuring starter behavior with the teacher removed.", flush=True)
    evaluation = evaluate(model)
    payload = {"build": BUILD, "kind": "development_starter", **model_payload(model),
               "training": result, "evaluation": evaluation, "seed": args.seed,
               "demonstration_episodes": args.episodes, "runtime": runtime_signature(device),
               "correction_rounds": rounds,
               "total_training_updates": args.updates + args.correction_rounds * args.correction_updates,
               "provenance": "authored shortest-path demonstrations in ResourceWorld; no v0.11 weights inherited"}
    save_model_file(output, payload)
    report = {k: v for k, v in payload.items() if k != "model"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_json({"starter": str(output), "model_sha256": payload["model_sha256"],
                "training": result, "evaluation": evaluation["summary"]})


def run(args):
    from .lifelong.runtime import DevelopmentRuntime, RuntimeConfig
    from .lifelong.dashboard import Dashboard
    report = preflight(args.device)
    print_json(report)
    cfg = None
    if args.config:
        cfg = RuntimeConfig(**json.loads(Path(args.config).read_text()))
    stop = {"requested": False}
    def request_stop(_signum, _frame):
        stop["requested"] = True
        print("Pause requested. Finishing the current atomic step/learning block and saving.", flush=True)
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    runner = None
    dash = None
    try:
        runner = DevelopmentRuntime(args.state_dir, resolve_device(args.device), cfg=cfg,
                                     starter=args.starter, seed=args.seed, name=args.name)
        if not args.no_dashboard:
            dash = Dashboard(runner.summary(), args.port)
            print("Dashboard: " + dash.url, flush=True)
            try:
                webbrowser.open(dash.url)
            except Exception:
                pass
        print(f"Running {args.steps} ADDITIONAL lived steps. Ctrl+C saves and pauses.", flush=True)
        result = runner.run(args.steps, on_tick=dash.update if dash else None,
                            stop_requested=lambda: stop["requested"])
        runner.store.finish_save(result)
        print_json({k: result[k] for k in ("identity", "steps", "episodes", "resources",
                                          "learning_blocks", "promotions", "active_evaluation",
                                          "forecaster", "status")})
        report_path = Path(args.state_dir).expanduser() / "latest_session.json"
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Session report: " + str(report_path.resolve()), flush=True)
        print("Saved. Relaunch the same command to continue this individual.", flush=True)
    finally:
        if dash:
            dash.close()
        if runner:
            runner.close()
        signal.signal(signal.SIGINT, previous_handler)


def talk(args):
    from .lifelong.storage import read_summary, read_events
    state = read_summary(args.state_dir)
    q = " ".join(args.question).lower().strip()
    if any(word in q for word in ("name", "who", "identity")):
        print(f"Name: {state['identity']['name']}. Individual: {state['identity']['id']}.")
    elif any(word in q for word in ("remember", "memory", "learn")):
        print_json(read_events(args.state_dir, 5))
    elif any(word in q for word in ("why", "action", "doing")):
        print_json(state["last_decision"])
    elif any(word in q for word in ("state", "feel", "need", "energy", "status")):
        print_json({"body": state["body"], "feedback": [state["positive"], state["negative"]],
                    "note": "Measured state report; subjective feelings are not established"})
    else:
        print("This local interface reports name, state, memory, and the last action. It does not understand unrestricted language.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuous Brain v0.12.1 local development")
    parser.add_argument("--threads", type=int, default=2, help="CPU worker threads (default 2)")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Check the actual device and a full model training step")
    p.add_argument("--device", default="dml")
    p.set_defaults(func=lambda a: print_json(preflight(a.device)))
    p = commands.add_parser("run", help="Start/resume a local individual")
    p.add_argument("--device", default="dml")
    p.add_argument("--state-dir", default=str(DEFAULT_STATE))
    p.add_argument("--starter", default=str(DEFAULT_STARTER))
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None, help="New individuals only; otherwise saved seed is used")
    p.add_argument("--name", default="Child")
    p.add_argument("--config", help="JSON protocol, fixed at the creation of an individual")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run)
    p = commands.add_parser("bootstrap", help="Train a new starter from explicit demonstrations")
    p.add_argument("--device", default="dml")
    p.add_argument("--output", default=str(DEFAULT_STARTER))
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sequence", type=int, default=12)
    p.add_argument("--correction-rounds", type=int, default=2)
    p.add_argument("--correction-updates", type=int, default=800)
    p.add_argument("--seed", type=int, default=120)
    p.set_defaults(func=bootstrap)
    p = commands.add_parser("status", help="Read the last committed status")
    p.add_argument("--state-dir", default=str(DEFAULT_STATE))
    def status(a):
        from .lifelong.storage import read_summary
        print_json(read_summary(a.state_dir))
    p.set_defaults(func=status)
    p = commands.add_parser("talk", help="Grounded state/memory queries; not a general language model")
    p.add_argument("--state-dir", default=str(DEFAULT_STATE))
    p.add_argument("question", nargs="+", default=["status"])
    p.set_defaults(func=talk)
    p = commands.add_parser("audit", help="Check environment feedback integrity")
    p.set_defaults(func=lambda a: print_json(feedback_integrity_audit()))
    p = commands.add_parser("snapshot", help="Create a consistent backup of a stopped individual")
    p.add_argument("--state-dir", default=str(DEFAULT_STATE))
    p.add_argument("--destination", required=True)
    def snapshot(a):
        from .lifelong.storage import LifetimeStore
        with LifetimeStore(a.state_dir) as store:
            if store.load() is None:
                raise ValueError("No initialized individual to snapshot")
            store.snapshot(a.destination)
        print("Snapshot saved to " + a.destination)
    p.set_defaults(func=snapshot)
    p = commands.add_parser("legacy", help="Evaluate or cross gates on your unchanged v0.11 child")
    p.add_argument("--device", default="dml")
    p.add_argument("--checkpoint")
    p.add_argument("--micro")
    p.add_argument("--low")
    p.add_argument("--trace-seed", type=int, help="Trace one spent seed (12001--12100) instead of the 100-episode suite")
    p.add_argument("--output-dir", default=str(Path.home() / "ContinuousBrain" / "legacy_diagnostics"))
    def legacy(a):
        from .lifelong.legacy import run_legacy_diagnostics
        print_json(run_legacy_diagnostics(a))
    p.set_defaults(func=legacy)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    torch.set_num_threads(args.threads)
    try:
        args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
