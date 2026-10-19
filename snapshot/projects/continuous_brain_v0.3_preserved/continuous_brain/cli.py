from __future__ import annotations

import argparse
import subprocess

from .agent import ContinuousAgent
from .config import BrainConfig
from .environment import ForagingWorld
from .visualizer import LiveVisualizer


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
    p.add_argument("--visual", action="store_true", help="serve the live browser visualization")
    p.add_argument("--visual-host", default="127.0.0.1", help="visual server bind host")
    p.add_argument("--visual-port", type=int, default=8765, help="visual server port; use 0 for any free port")
    p.add_argument("--open-browser", action="store_true", help="try to open the visualizer in the Windows/default browser")
    return p


def _try_open_browser(url: str) -> None:
    # WSL can ask Windows to open a URL. Fall back to Python's browser helper elsewhere.
    commands = [
        ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
        ["cmd.exe", "/c", "start", "", url],
    ]
    for cmd in commands:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, OSError):
            continue
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    args = build_parser().parse_args()
    cfg = BrainConfig(brain_hz=args.brain_hz, seed=args.seed)
    env = ForagingWorld(seed=args.seed)
    visualizer = LiveVisualizer(args.visual_host, args.visual_port) if args.visual else None
    agent = ContinuousAgent(
        env,
        cfg,
        state_dir=args.state_dir,
        device=args.device,
        ollama_model=args.ollama_model,
        planner_every=args.planner_every,
        visualizer=visualizer,
    )

    if visualizer:
        url = visualizer.start()
        print(f"Live visualization: {url}")
        print("Open that address in your Windows browser. Controls: Pause / Slower / Faster / 1×.")
        if args.open_browser:
            _try_open_browser(url)

    print(
        f"Continuous Brain v0.2 | device={agent.device} | {cfg.brain_hz:g} Hz | "
        "Ctrl+C saves state and exits."
    )
    try:
        agent.run(max_steps=args.steps, realtime=not args.fast)
    finally:
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    main()
