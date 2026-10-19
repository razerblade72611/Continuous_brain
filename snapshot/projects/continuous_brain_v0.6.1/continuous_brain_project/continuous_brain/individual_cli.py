"""CLI for v0.6; the old continuous-brain entry point stays compatible."""
from __future__ import annotations
import json
import torch
from .cli import build_parser as legacy_parser, _build_environment, _try_open_browser
from .cognition import CognitiveConfig
from .development import DevelopmentRoom
from .individual import IndividualRuntime
from .vision import build_vision_backbone
from .individual_visualizer import individual_visualizer


def build_parser():
    p = legacy_parser()
    p.description = "Continuous Brain v0.6.1 — Persistent Individual Core (research foundation)"
    # Reuse the THOR adapter's argument schema, but do not advertise or silently
    # accept legacy learning/planner flags that this new runtime does not use.
    unused = {"steps", "additional_steps", "ollama_model", "planner_every",
              "bootstrap_checkpoint", "vision_every", "curiosity_weight", "homeostasis_weight",
              "replay_checkpoint_items", "stop_on_mastery"}
    for action in list(p._actions):
        if action.dest in unused:
            p.set_defaults(**{action.dest: action.default})
            p._actions.remove(action)
            for group in p._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for option in action.option_strings:
                p._option_string_actions.pop(option, None)
    for action in p._actions:
        if action.dest == "environment":
            action.choices = ("development", "thor")
    p.set_defaults(environment="development", state_dir="state/individual_v061",
                   vision_backbone="pixel-stats", curriculum_level=2, goal_signal="internal-map",
                   scene="FloorPlan1Discovery_physics")
    p.add_argument("--profile", choices=("tiny", "compact", "standard"), default="standard")
    p.add_argument("--decisions", type=int, default=0, help="additional executive decisions; 0 runs until Ctrl+C")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--import-v053-state", help="NEW individuals only: stopped THOR state (including v0.4.3); copied and preserved")
    p.add_argument("--individual-name", default="Continuous Brain")
    p.add_argument("--evaluate", action="store_true", help="no learning, teacher forcing, exploration or persistent writes")
    p.add_argument("--consolidate-updates", type=int, default=0, help="replay-only updates, then save and exit")
    p.add_argument("--torch-threads", type=int, default=4, help="CPU worker threads")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if args.steps or args.additional_steps or args.stop_on_mastery or args.bootstrap_checkpoint or args.ollama_model:
        p.error("v0.6 uses --decisions and --import-v053-state; legacy step/mastery/Ollama flags are unsupported")
    if args.environment == "thor" and args.curriculum_level != 2:
        p.error("The v0.6 THOR integration is scoped to the retained L2 motor interface")
    if args.evaluate and args.consolidate_updates:
        p.error("Evaluation cannot consolidate/train")
    if min(args.checkpoint_every, args.log_every, args.torch_threads, args.brain_hz) <= 0:
        p.error("Frequencies and thread counts must be positive")
    if min(args.decisions, args.consolidate_updates) < 0:
        p.error("Decision/update counts must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env = None
    runtime = None
    visual = None
    try:
        env = (DevelopmentRoom(args.seed, args.max_episode_steps) if args.environment == "development"
               else _build_environment(args))
        vision = build_vision_backbone(args.vision_backbone, device=args.vision_device,
                                      model_id=args.vision_model_id)
        cfg = CognitiveConfig.profile(args.profile, 6 + vision.output_dim)
        visual = individual_visualizer(args.visual_host, args.visual_port) if args.visual else None
        runtime = IndividualRuntime(env, args.state_dir, cfg, vision, device=device, seed=args.seed,
                                    import_v053=args.import_v053_state, evaluate=args.evaluate,
                                    visualizer=visual, name=args.individual_name)
        print(f"Continuous Brain 0.6.1 | {runtime.net.parameter_count:,} trainable parameters | "
              f"{device} | individual={runtime.self_model.individual_id}", flush=True)
        print(f"motor={runtime.status()['motor_source']} | "
              f"mode={'executive evaluation; motor unchanged (not saved)' if args.evaluate else 'development/training'}", flush=True)
        if visual:
            url = visual.start()
            runtime._dashboard()
            print(f"Dashboard: {url}", flush=True)
            if args.open_browser:
                _try_open_browser(url)
        if args.consolidate_updates:
            for _ in range(args.consolidate_updates):
                if runtime.learner.train() is None:
                    raise ValueError("Not enough contiguous replay to consolidate; run more decisions first")
            runtime.save()
        else:
            runtime.run(args.decisions, args.checkpoint_every, not args.fast, args.brain_hz, args.log_every)
        print(json.dumps(runtime.status(), indent=2))
    finally:
        if runtime:
            runtime.close()
        elif env:
            env.close()
        if visual:
            visual.close()


if __name__ == "__main__":
    main()
