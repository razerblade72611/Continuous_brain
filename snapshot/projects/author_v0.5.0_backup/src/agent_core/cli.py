from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .agent import Agent
from .executor import ActionExecutor
from .grounding import SemanticGrounder
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .planner import HybridPlanner, LLMPlanner
from .recovery import RecoveryManager
from .roam import EmbodiedLoop, EmbodiedPlanner, FreeRoamAgent
from .session import InteractiveEmbodiedSession
from .social import SpeechBus
from .social_session import SocialEmbodiedSession
from .thor_adapter import SharedThorWorld, ThorAdapter, ThorConfig


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def build_llm_client(args, cfg: dict) -> SyncLLMClient | None:
    llm_cfg = cfg.get("llm", {})
    if not (args.llm or bool(llm_cfg.get("enabled", False))):
        return None
    return SyncLLMClient(
        base_url=os.getenv("AGENT_LLM_BASE_URL", llm_cfg.get("base_url", "http://127.0.0.1:8080/v1")),
        model=os.getenv("AGENT_LLM_MODEL", llm_cfg.get("model", "local-model")),
        timeout=float(llm_cfg.get("timeout_seconds", 120)),
    )


def make_thor_config(args, cfg: dict) -> ThorConfig:
    nav_cfg = cfg.get("navigation", {})
    sim_cfg = cfg.get("simulation", {})
    return ThorConfig(
        scene=args.scene or cfg.get("scene", "FloorPlan1"),
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
        render_instance_segmentation=bool(sim_cfg.get("render_instance_segmentation", True)),
    )


def embodied_options(args, cfg: dict):
    embodied_cfg = cfg.get("embodied", {})
    roam_cfg = cfg.get("roam", {})
    allow_interactions = bool(embodied_cfg.get("allow_interactions", True))
    if args.interact:
        allow_interactions = True
    if args.no_interact:
        allow_interactions = False
    use_vision = bool(embodied_cfg.get("use_vision", True))
    if args.vision:
        use_vision = True
    if args.no_vision:
        use_vision = False
    vision_every = int(
        args.vision_every if args.vision_every is not None else embodied_cfg.get("vision_every", roam_cfg.get("vision_every", 1))
    )
    max_decisions = int(
        args.roam_decisions if args.roam_decisions is not None else embodied_cfg.get("max_decisions", roam_cfg.get("max_decisions", 500))
    )
    return embodied_cfg, allow_interactions, use_vision, vision_every, max_decisions


def make_loop(
    *, thor, client, cfg: dict, args, log_dir: Path, agent_label: str | None = None, social_enabled: bool = False
) -> EmbodiedLoop:
    agent_cfg = cfg.get("agent", {})
    embodied_cfg, allow_interactions, use_vision, vision_every, _ = embodied_options(args, cfg)
    memory = EpisodicMemory(path=str(log_dir / "memory.jsonl"), max_records=4000)
    recovery = RecoveryManager(max_retries_per_action=int(agent_cfg.get("max_retries_per_action", 2)))
    max_steps = int(args.max_steps if args.max_steps is not None else agent_cfg.get("max_steps", 3000))
    executor = ActionExecutor(
        thor=thor,
        memory=memory,
        recovery=recovery,
        max_steps=max_steps,
        logger_path=str(log_dir / "actions.jsonl"),
    )
    grounder = SemanticGrounder(interaction_distance=thor.config.interaction_distance)
    planner = EmbodiedPlanner(
        client=client,
        grounder=grounder,
        allow_interactions=allow_interactions,
        max_burst_actions=int(embodied_cfg.get("max_burst_actions", 6)),
        social_enabled=social_enabled,
        agent_label=agent_label,
    )
    return EmbodiedLoop(
        thor=thor,
        planner=planner,
        executor=executor,
        memory=memory,
        recovery=recovery,
        use_vision=use_vision,
        vision_every=vision_every,
        print_decisions=bool(embodied_cfg.get("print_decisions", True)),
        llm_controls_pace=bool(embodied_cfg.get("llm_controls_pace", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AI2-THOR embodied agent")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--goal", help="Run the legacy one-shot goal-directed task mode.")
    mode.add_argument("--roam", action="store_true", help="Run neutral autonomous embodied mode.")
    mode.add_argument("--session", action="store_true", help="Run a persistent autonomous/task interactive session.")

    parser.add_argument("--agents", type=int, default=1, help="Number of embodied agents. v0.5 social mode supports 1 or 2.")
    parser.add_argument("--speech-range", type=float, default=None, help="Hearing radius in meters for explicit Speak actions.")
    parser.add_argument("--initial-task", default=None, help="Optional task to start a single-agent --session with.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--llm", action="store_true", help="Use the configured OpenAI-compatible local LLM/VLM.")
    parser.add_argument("--vision", action="store_true", help="Send each AI2-THOR viewport to that agent's multimodal context.")
    parser.add_argument("--no-vision", action="store_true", help="Disable camera input even if config enables it.")
    parser.add_argument("--interact", action="store_true", help="Enable pickup/open/close/place capabilities.")
    parser.add_argument("--no-interact", action="store_true", help="Disable object interaction capabilities.")
    parser.add_argument("--roam-decisions", type=int, default=None, help="Maximum decisions (single) or social rounds (two agents).")
    parser.add_argument("--vision-every", type=int, default=None, help="Send a camera frame every N decisions per agent.")
    parser.add_argument("--max-steps", type=int, default=None, help="Primitive Unity action budget per embodied agent.")
    parser.add_argument("--step-delay", type=float, default=None, help="Base seconds to pause after each primitive Unity action.")
    parser.add_argument("--headless", action="store_true", help="Use AI2-THOR CloudRendering (Linux/Vulkan only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    embodied_cfg = cfg.get("embodied", {})
    if embodied_cfg.get("metadata_to_llm", False):
        raise ValueError("Neutral embodied mode requires embodied.metadata_to_llm: false")
    if args.agents not in {1, 2}:
        raise ValueError("v0.5.0 supports --agents 1 or --agents 2.")
    if args.agents == 2 and not args.session:
        raise ValueError("Two-agent social mode currently requires --session.")

    client = build_llm_client(args, cfg)
    thor_config = make_thor_config(args, cfg)
    _, _, _, _, max_decisions = embodied_options(args, cfg)
    base_log_dir = Path(cfg.get("agent", {}).get("log_dir", "runs"))

    if args.agents == 2:
        if client is None:
            raise RuntimeError("Two-agent social session requires --llm or llm.enabled: true.")
        social_cfg = cfg.get("social", {})
        labels_list = list(social_cfg.get("labels", ["A", "B"]))
        if len(labels_list) < 2:
            labels_list = ["A", "B"]
        labels = {0: str(labels_list[0]), 1: str(labels_list[1])}
        speech_range = float(args.speech_range if args.speech_range is not None else social_cfg.get("speech_range_m", 4.0))

        shared = SharedThorWorld(thor_config, agent_count=2)
        shared.start()
        try:
            loops = {}
            for agent_id in (0, 1):
                adapter = shared.agent(agent_id)
                adapter.start()
                label = labels[agent_id]
                loops[agent_id] = make_loop(
                    thor=adapter,
                    client=client,
                    cfg=cfg,
                    args=args,
                    log_dir=base_log_dir / f"agent_{label.lower()}",
                    agent_label=label,
                    social_enabled=True,
                )
            bus = SpeechBus(labels=labels, speech_range_m=speech_range)
            session = SocialEmbodiedSession(
                loops=loops,
                labels=labels,
                speech_bus=bus,
                max_rounds=max_decisions,
            )
            return session.run()
        finally:
            shared.close()

    # Single-agent path remains v0.4.1-compatible.
    thor = ThorAdapter(thor_config)
    thor.start()
    try:
        if args.roam or args.session:
            if client is None:
                raise RuntimeError("Neutral embodied roam/session requires --llm or llm.enabled: true.")
            loop = make_loop(thor=thor, client=client, cfg=cfg, args=args, log_dir=base_log_dir)
            if args.session:
                session = InteractiveEmbodiedSession(loop=loop, max_decisions=max_decisions, initial_task=args.initial_task)
                return session.run()
            result = FreeRoamAgent(loop=loop, max_decisions=max_decisions).run()
            print(f"roam_decisions={result.decisions}")
            print(f"steps={result.primitive_steps}")
            print(f"reason={result.reason}")
            return 0

        agent_cfg = cfg.get("agent", {})
        memory = EpisodicMemory(path=str(base_log_dir / "memory.jsonl"), max_records=4000)
        recovery = RecoveryManager(max_retries_per_action=int(agent_cfg.get("max_retries_per_action", 2)))
        max_steps = int(args.max_steps if args.max_steps is not None else agent_cfg.get("max_steps", 3000))
        executor = ActionExecutor(
            thor=thor,
            memory=memory,
            recovery=recovery,
            max_steps=max_steps,
            logger_path=str(base_log_dir / "actions.jsonl"),
        )
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
