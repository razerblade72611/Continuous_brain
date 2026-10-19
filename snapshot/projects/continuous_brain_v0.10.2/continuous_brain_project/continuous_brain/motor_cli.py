"""Train/evaluate a small motor without loading or modifying any executive."""
from __future__ import annotations

import argparse
import json
import signal

import torch

from .motor import MOTOR_CONTRACT, validated_motor, check_qualification
from .motor_lab import MotorLabLock, MotorTrainer, atomic_save, evaluate_motor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--state-dir", default="state/motor_dev_v062")
    parser.add_argument("--updates", type=int, default=2000, help="additional supervised motor updates")
    parser.add_argument("--cases", type=int, default=256, help="training curriculum cases; fixed on resume")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes-per-skill", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_skill, args.torch_threads) < 1 or args.cases < 8 or args.seed < 0:
        parser.error("Positive counts/threads, at least 8 cases, and a nonnegative seed required")
    torch.set_num_threads(args.torch_threads)
    with MotorLabLock(args.state_dir, create=args.command == "train") as lab:
        checkpoint = lab.path / "motor_training.pt"
        if args.command == "train":
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
            print("Motor lab: teacher-labelled development observations; executive is not loaded.", flush=True)
            trainer = MotorTrainer(args.seed, args.cases, saved)
            print(f"motor_parameters={trainer.model.parameter_count} demonstrations={len(trainer.data['actions'])} "
                  f"resumed_updates={trainer.updates}", flush=True)
            running = True
            def stop(*_):
                nonlocal running
                running = False
            handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                for _ in range(args.updates):
                    if not running:
                        break
                    trainer.update()
                    if trainer.updates % 250 == 0:
                        atomic_save(trainer.payload(), checkpoint)
                        print(f"motor_updates={trainer.updates} imitation_loss={trainer.last_loss:.6f}", flush=True)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                atomic_save(trainer.payload(), checkpoint)
            print(json.dumps({"motor_updates": trainer.updates, "loss": trainer.last_loss,
                              "qualification": "required before attachment", "executive_modified": False}, indent=2))
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("kind") != "motor_lab_v1":
                raise ValueError("Not a motor-lab checkpoint")
            model = validated_motor(saved)
            report = evaluate_motor(model, saved["seed"], args.episodes_per_skill)
            (lab.path / "motor_evaluation.json").write_text(json.dumps(report, indent=2))
            if report["passed"]:
                candidate = {"contract": MOTOR_CONTRACT, "model": saved["model"], "digest": saved["digest"],
                             "updates": saved["updates"], "qualification": report,
                             "origin": "fresh_motor_teacher_trained_in_development"}
                check_qualification(candidate)
                atomic_save(candidate, lab.path / "qualified_motor.pt")
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                parser.exit(2, "Motor did not qualify. Preserve state; inspect the report before attachment.\n")


if __name__ == "__main__":
    main()
