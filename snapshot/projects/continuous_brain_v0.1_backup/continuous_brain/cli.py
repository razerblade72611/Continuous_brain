from __future__ import annotations

import argparse

from .agent import ContinuousAgent
from .config import BrainConfig
from .environment import ForagingWorld


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a continuously operating neural agent.")
    p.add_argument("--state-dir", default="state/continuous_brain")
    p.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C")
    p.add_argument("--brain-hz", type=float, default=10.0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--fast", action="store_true", help="run without real-time sleeping")
    p.add_argument("--ollama-model", default=None, help="optional local Ollama model name")
    p.add_argument("--planner-every", type=int, default=50, help="slow planner interval in brain ticks")
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = BrainConfig(brain_hz=args.brain_hz, seed=args.seed)
    env = ForagingWorld(seed=args.seed)
    agent = ContinuousAgent(
        env,
        cfg,
        state_dir=args.state_dir,
        device=args.device,
        ollama_model=args.ollama_model,
        planner_every=args.planner_every,
    )
    print(
        f"Continuous Brain v0.1 | device={agent.device} | {cfg.brain_hz:g} Hz | "
        "Ctrl+C saves state and exits."
    )
    agent.run(max_steps=args.steps, realtime=not args.fast)


if __name__ == "__main__":
    main()
