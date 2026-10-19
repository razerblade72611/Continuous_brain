from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .agent import Agent
from .executor import ActionExecutor
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .planner import HybridPlanner, LLMPlanner
from .recovery import RecoveryManager
from .roam import FreeRoamAgent, RoamPlanner
from .thor_adapter import ThorAdapter, ThorConfig


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="AI2-THOR Phase 9+ agent")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--goal", help="Run goal-directed task mode.")
    mode.add_argument("--roam", action="store_true", help="Run continuous free-roam exploration mode.")

    parser.add_argument("--scene", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--llm", action="store_true", help="Use the configured OpenAI-compatible local LLM.")
    parser.add_argument("--vision", action="store_true", help="In roam mode, send camera frames to a multimodal LLM.")
    parser.add_argument("--interact", action="store_true", help="In roam mode, allow pickup/open/close/place actions.")
    parser.add_argument("--roam-decisions", type=int, default=None, help="Maximum LLM decisions in roam mode.")
    parser.add_argument("--vision-every", type=int, default=None, help="Send a camera frame every N roam decisions.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override primitive Unity action budget.")
    parser.add_argument("--step-delay", type=float, default=None, help="Seconds to pause after every primitive Unity action.")
    parser.add_argument("--headless", action="store_true", help="Use AI2-THOR CloudRendering (Linux/Vulkan only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_cfg = cfg.get("agent", {})
    nav_cfg = cfg.get("navigation", {})
    roam_cfg = cfg.get("roam", {})
    sim_cfg = cfg.get("simulation", {})

    scene = args.scene or cfg.get("scene", "FloorPlan1")
    thor = ThorAdapter(
        ThorConfig(
            scene=scene,
            grid_size=float(cfg.get("grid_size", 0.25)),
            rotate_step_degrees=float(cfg.get("rotate_step_degrees", 90)),
            visibility_distance=float(cfg.get("visibility_distance", 1.5)),
            interaction_distance=float(nav_cfg.get("interaction_distance", 1.25)),
            width=int(cfg.get("image_width", 640)),
            height=int(cfg.get("image_height", 480)),
            field_of_view=int(cfg.get("field_of_view", 90)),
            headless=args.headless,
            step_delay_seconds=float(
                args.step_delay if args.step_delay is not None else sim_cfg.get("step_delay_seconds", 0.0)
            ),
        )
    )

    llm_cfg = cfg.get("llm", {})
    client = None
    if args.llm or bool(llm_cfg.get("enabled", False)):
        client = SyncLLMClient(
            base_url=os.getenv("AGENT_LLM_BASE_URL", llm_cfg.get("base_url", "http://127.0.0.1:8080/v1")),
            model=os.getenv("AGENT_LLM_MODEL", llm_cfg.get("model", "local-model")),
            timeout=float(llm_cfg.get("timeout_seconds", 120)),
        )

    log_dir = Path(agent_cfg.get("log_dir", "runs"))
    memory = EpisodicMemory(path=str(log_dir / "memory.jsonl"), max_records=2000)
    recovery = RecoveryManager(max_retries_per_action=int(agent_cfg.get("max_retries_per_action", 2)))
    max_steps = int(args.max_steps if args.max_steps is not None else agent_cfg.get("max_steps", 120))
    executor = ActionExecutor(
        thor=thor,
        memory=memory,
        recovery=recovery,
        max_steps=max_steps,
        logger_path=str(log_dir / "actions.jsonl"),
    )

    thor.start()
    try:
        if args.roam:
            roam_planner = RoamPlanner(
                client=client,
                interaction_distance=thor.config.interaction_distance,
                allow_interactions=bool(args.interact or roam_cfg.get("allow_interactions", False)),
            )
            roam_agent = FreeRoamAgent(
                thor=thor,
                planner=roam_planner,
                executor=executor,
                memory=memory,
                recovery=recovery,
                max_decisions=int(
                    args.roam_decisions
                    if args.roam_decisions is not None
                    else roam_cfg.get("max_decisions", 100)
                ),
                vision_every=int(
                    args.vision_every
                    if args.vision_every is not None
                    else roam_cfg.get("vision_every", 1)
                ),
                use_vision=bool(args.vision or roam_cfg.get("use_vision", False)),
                print_decisions=bool(roam_cfg.get("print_decisions", True)),
            )
            if client is None:
                print("warning: --roam is running without --llm; deterministic exploration fallback will be used.")
            result = roam_agent.run()
            print(f"roam_decisions={result.decisions}")
            print(f"steps={result.primitive_steps}")
            print(f"reason={result.reason}")
            return 0

        llm = LLMPlanner(client) if client is not None else None
        planner = HybridPlanner(llm=llm, interaction_distance=thor.config.interaction_distance)
        agent = Agent(
            thor=thor,
            planner=planner,
            executor=executor,
            memory=memory,
            recovery=recovery,
            max_replans=int(agent_cfg.get("max_replans", 30)),
        )
        result = agent.run(str(args.goal))
        print(f"success={result.success}")
        print(f"goal={result.goal}")
        print(f"reason={result.reason}")
        print(f"steps={result.steps}")
        return 0 if result.success else 2
    finally:
        thor.close()


if __name__ == "__main__":
    raise SystemExit(main())
