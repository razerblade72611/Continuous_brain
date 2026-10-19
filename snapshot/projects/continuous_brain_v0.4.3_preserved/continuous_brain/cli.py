from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from .agent import ContinuousAgent
from .config import BrainConfig
from .environment import ForagingWorld
from .visualizer import LiveVisualizer
from .vision import build_vision_backbone


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a continuously operating neural agent.")
    p.add_argument("--state-dir", default="state/continuous_brain")
    p.add_argument("--steps", type=int, default=0, help="absolute total step to stop at; 0 means Ctrl+C")
    p.add_argument("--additional-steps", type=int, default=0, help="run this many additional ticks from the loaded checkpoint")
    p.add_argument("--brain-hz", type=float, default=10.0)
    p.add_argument("--device", default="auto", help="brain device: auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--fast", action="store_true", help="run without real-time sleeping")
    p.add_argument("--ollama-model", default=None, help="optional local pretrained Ollama planner")
    p.add_argument("--planner-every", type=int, default=50, help="slow planner interval in brain ticks")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bootstrap-checkpoint", default=None, help="import compatible neural weights into a fresh state branch")

    p.add_argument(
        "--environment", default="foraging", choices=("foraging", "thor"),
        help="2-D validation world or 3-D AI2-THOR embodiment",
    )
    p.add_argument("--scene", default="FloorPlan1_physics", help="AI2-THOR scene name")
    p.add_argument("--thor-executable", default=None, help="path to a local/custom AI2-THOR executable")
    p.add_argument("--thor-width", type=int, default=320)
    p.add_argument("--thor-height", type=int, default=240)
    p.add_argument("--thor-fov", type=float, default=90.0)
    p.add_argument("--grid-size", type=float, default=0.25)
    p.add_argument("--rotate-degrees", type=float, default=30.0)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--curriculum-level", type=int, default=1, choices=range(10))
    p.add_argument("--goal-signal", default="teacher-nav", choices=("teacher-nav", "sensor-only"))
    p.add_argument("--goal-dropout", type=float, default=0.0, help="probability [0,1] of masking point-goal slots each observation")
    p.add_argument("--no-random-start", action="store_true")
    p.add_argument("--safety-distance", type=float, default=0.20)
    p.add_argument("--action-noise", type=float, default=0.0, help="randomize movement/turn intent with this probability")
    p.add_argument("--orient-success-degrees", type=float, default=15.0)
    p.add_argument("--stagnation-steps", type=int, default=60, help="terminate/reset an episode after this many no-progress steps")
    p.add_argument("--stagnation-epsilon", type=float, default=0.5, help="minimum angular improvement in degrees that resets L0 stagnation")
    p.add_argument("--graduation-window", type=int, default=100, help="rolling episode window for curriculum graduation statistics")
    p.add_argument("--rehearsal-fraction", type=float, default=0.15, help="fraction of L1+ episodes used to rehearse earlier skills")
    p.add_argument("--level-epsilon-start", type=float, default=0.35, help="exploration floor when a new curriculum level is unlocked")
    p.add_argument("--level-epsilon-end", type=float, default=0.08, help="late curriculum-specific exploration floor")
    p.add_argument("--level-epsilon-decay-steps", type=int, default=5000, help="steps over which the new-level exploration floor decays")
    p.add_argument("--nav-angular-scale", type=float, default=0.003, help="L1 reward per degree of angular progress")
    p.add_argument("--nav-distance-scale", type=float, default=0.80, help="L1 reward per meter of waypoint-distance progress")
    p.add_argument("--nav-forward-bonus", type=float, default=0.02, help="L1 bonus for a successful forward movement")
    p.add_argument("--nav-blocked-penalty", type=float, default=0.08, help="L1 penalty for SafetyStop/blocked forward movement")
    p.add_argument("--nav-wrong-way-penalty", type=float, default=0.04, help="L1 penalty for advancing away from the active waypoint")
    p.add_argument("--nav-initial-path-steps", type=int, default=2, help="initial L1 teacher-path length in reachable-grid cells")
    p.add_argument("--nav-min-path-steps", type=int, default=2, help="minimum L1 teacher-path length; v0.4.3 defaults to two actual moves")
    p.add_argument("--nav-max-path-steps", type=int, default=8, help="maximum L1 path length as competence improves")
    p.add_argument("--nav-success-radius", type=float, default=0.12, help="meters from the final L1 point required for success")
    p.add_argument("--nav-min-successful-translations", type=int, default=1, help="successful MoveAhead transitions required before L1 may succeed")
    p.add_argument("--nav-alignment-gate-degrees", type=float, default=15.0, help="mask ADVANCE until teacher waypoint bearing is within this angle")
    p.add_argument("--nav-idle-aligned-penalty", type=float, default=0.02, help="extra L1 penalty for idling while aligned to the waypoint")
    p.add_argument("--nav-edge-failure-threshold", type=int, default=2, help="aligned failed traversals before a reachable-grid edge is persistently blocked")
    p.add_argument("--nav-graduation-blocked-rate", type=float, default=0.10, help="maximum blocked/(blocked+successful) move rate for L1 graduation")
    p.add_argument("--depth-safety-training", action="store_true", help="use the old depth pre-stop during simulator training; default lets AI2-THOR physics decide collisions")
    p.add_argument("--replay-checkpoint-items", type=int, default=8000, help="bounded replay transitions persisted across restarts")

    p.add_argument(
        "--vision-backbone",
        default="none",
        choices=("none", "pixel-stats", "dinov3-vits16"),
        help="visual feature provider; DINOv3 is frozen/pretrained",
    )
    p.add_argument("--vision-device", default="auto", help="device for pretrained vision model")
    p.add_argument("--vision-every", type=int, default=1, help="refresh visual embedding every N brain ticks")
    p.add_argument("--vision-model-id", default=None, help="override Hugging Face DINOv3 model ID")
    p.add_argument("--curiosity-weight", type=float, default=0.02)
    p.add_argument("--homeostasis-weight", type=float, default=0.01)

    p.add_argument("--visual", action="store_true", help="serve the live browser visualization")
    p.add_argument("--visual-host", default="127.0.0.1", help="visual server bind host")
    p.add_argument("--visual-port", type=int, default=8765, help="visual server port; use 0 for any free port")
    p.add_argument("--open-browser", action="store_true", help="try to open the visualizer in the Windows/default browser")
    return p


def _running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except Exception:
        return False


def _try_open_browser(url: str) -> bool:
    commands: list[list[str]] = []
    if _running_under_wsl():
        commands.extend([
            ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
            ["/mnt/c/Windows/System32/cmd.exe", "/c", "start", "", url],
            ["/mnt/c/Windows/explorer.exe", url],
            ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
            ["cmd.exe", "/c", "start", "", url],
        ])
    else:
        commands.extend([
            ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
            ["cmd.exe", "/c", "start", "", url],
        ])
    for cmd in commands:
        try:
            if Path(cmd[0]).is_absolute() and not Path(cmd[0]).exists():
                continue
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, OSError):
            continue
    if _running_under_wsl():
        return False  # do not fall through to Linux xdg-open noise in WSL
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _build_environment(args):
    if args.environment == "foraging":
        return ForagingWorld(seed=args.seed)
    from .embodied import AI2ThorConfig, AI2ThorEmbodiedEnv
    cfg = AI2ThorConfig(
        scene=args.scene,
        executable=args.thor_executable,
        width=args.thor_width,
        height=args.thor_height,
        fov=args.thor_fov,
        grid_size=args.grid_size,
        rotate_degrees=args.rotate_degrees,
        max_episode_steps=args.max_episode_steps,
        curriculum_level=args.curriculum_level,
        goal_signal=args.goal_signal,
        goal_dropout=max(0.0, min(1.0, args.goal_dropout)),
        random_start=not args.no_random_start,
        safety_distance=max(0.0, args.safety_distance),
        action_noise=max(0.0, min(0.5, args.action_noise)),
        orient_success_degrees=max(1.0, min(90.0, args.orient_success_degrees)),
        stagnation_steps=max(1, args.stagnation_steps),
        stagnation_epsilon=max(0.0, args.stagnation_epsilon),
        graduation_window=max(10, args.graduation_window),
        rehearsal_fraction=max(0.0, min(0.50, args.rehearsal_fraction)),
        level_epsilon_start=max(0.0, min(1.0, args.level_epsilon_start)),
        level_epsilon_end=max(0.0, min(1.0, args.level_epsilon_end)),
        level_epsilon_decay_steps=max(1, args.level_epsilon_decay_steps),
        nav_angular_scale=max(0.0, args.nav_angular_scale),
        nav_distance_scale=max(0.0, args.nav_distance_scale),
        nav_forward_bonus=max(0.0, args.nav_forward_bonus),
        nav_blocked_penalty=max(0.0, args.nav_blocked_penalty),
        nav_wrong_way_penalty=max(0.0, args.nav_wrong_way_penalty),
        nav_initial_path_steps=max(2, args.nav_initial_path_steps),
        nav_min_path_steps=max(2, args.nav_min_path_steps),
        nav_max_path_steps=max(2, args.nav_max_path_steps),
        nav_success_radius=max(0.03, args.nav_success_radius),
        nav_min_successful_translations=max(1, args.nav_min_successful_translations),
        nav_alignment_gate_degrees=max(0.0, min(90.0, args.nav_alignment_gate_degrees)),
        nav_idle_aligned_penalty=max(0.0, args.nav_idle_aligned_penalty),
        nav_edge_failure_threshold=max(1, args.nav_edge_failure_threshold),
        nav_native_collision_training=not bool(args.depth_safety_training),
        nav_graduation_blocked_rate=max(0.0, min(1.0, args.nav_graduation_blocked_rate)),
        seed=args.seed,
    )
    return AI2ThorEmbodiedEnv(cfg)


def main() -> None:
    args = build_parser().parse_args()
    cfg = BrainConfig(
        brain_hz=args.brain_hz,
        seed=args.seed,
        curiosity_weight=max(0.0, args.curiosity_weight),
        homeostasis_weight=max(0.0, args.homeostasis_weight),
        replay_checkpoint_items=max(0, args.replay_checkpoint_items),
    )
    try:
        env = _build_environment(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Environment setup failed: {exc}") from exc

    print(f"Loading vision backbone: {args.vision_backbone}", flush=True)
    try:
        vision = build_vision_backbone(
            args.vision_backbone,
            device=args.vision_device,
            model_id=args.vision_model_id,
        )
    except (RuntimeError, ValueError) as exc:
        try:
            env.close()
        except Exception:
            pass
        raise SystemExit(f"Vision setup failed: {exc}") from exc
    print(f"Vision: {vision.describe()}", flush=True)

    visualizer = LiveVisualizer(args.visual_host, args.visual_port) if args.visual else None
    agent = ContinuousAgent(
        env,
        cfg,
        state_dir=args.state_dir,
        device=args.device,
        ollama_model=args.ollama_model,
        planner_every=args.planner_every,
        visualizer=visualizer,
        vision_backbone=vision,
        vision_every=args.vision_every,
        bootstrap_checkpoint=args.bootstrap_checkpoint,
    )

    if visualizer:
        url = visualizer.start()
        print(f"Live visualization: {url}")
        print("Open that address in your Windows browser. Controls: Pause / Slower / Faster / 1×.")
        if args.open_browser and not _try_open_browser(url):
            print(f"Could not auto-open a browser. Open {url.replace('127.0.0.1', 'localhost')} in Windows manually.")

    stop_at = int(args.steps)
    if args.additional_steps > 0:
        stop_at = agent.total_steps + int(args.additional_steps)

    environment_label = args.environment
    if args.environment == "thor":
        environment_label += f":L{args.curriculum_level}:{args.scene}"
    print(
        f"Continuous Brain v0.4.3 | device={agent.device} | {cfg.brain_hz:g} Hz | "
        f"env={environment_label} | vision={vision.name} | Ctrl+C saves state and exits."
    )
    try:
        agent.run(max_steps=stop_at, realtime=not args.fast)
    finally:
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    main()
