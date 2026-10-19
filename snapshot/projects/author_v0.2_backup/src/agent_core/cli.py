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
from .thor_adapter import ThorAdapter, ThorConfig


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="AI2-THOR Phase 9+ agent")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Use AI2-THOR CloudRendering (Linux/Vulkan only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_cfg = cfg.get("agent", {})
    nav_cfg = cfg.get("navigation", {})

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
        )
    )

    llm = None
    llm_cfg = cfg.get("llm", {})
    if args.llm or bool(llm_cfg.get("enabled", False)):
        client = SyncLLMClient(
            base_url=os.getenv("AGENT_LLM_BASE_URL", llm_cfg.get("base_url", "http://127.0.0.1:8080/v1")),
            model=os.getenv("AGENT_LLM_MODEL", llm_cfg.get("model", "local-model")),
            timeout=float(llm_cfg.get("timeout_seconds", 120)),
        )
        llm = LLMPlanner(client)

    planner = HybridPlanner(llm=llm, interaction_distance=thor.config.interaction_distance)
    log_dir = Path(agent_cfg.get("log_dir", "runs"))
    memory = EpisodicMemory(path=str(log_dir / "memory.jsonl"), max_records=1000)
    recovery = RecoveryManager(max_retries_per_action=int(agent_cfg.get("max_retries_per_action", 2)))
    executor = ActionExecutor(
        thor=thor,
        memory=memory,
        recovery=recovery,
        max_steps=int(agent_cfg.get("max_steps", 120)),
        logger_path=str(log_dir / "actions.jsonl"),
    )
    agent = Agent(
        thor=thor,
        planner=planner,
        executor=executor,
        memory=memory,
        recovery=recovery,
        max_replans=int(agent_cfg.get("max_replans", 30)),
    )

    thor.start()
    try:
        result = agent.run(args.goal)
        print(f"success={result.success}")
        print(f"goal={result.goal}")
        print(f"reason={result.reason}")
        print(f"steps={result.steps}")
        return 0 if result.success else 2
    finally:
        thor.close()


if __name__ == "__main__":
    raise SystemExit(main())
