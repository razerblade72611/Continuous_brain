from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import yaml

from .learning import (
    LearningPaths,
    append_jsonl,
    load_json,
    merge_jsonl_files,
    reset_learning_state,
    utc_now,
    write_json_atomic,
)
from .memory import EpisodicMemory
from .models import Action
from .nav_skill import NavigationSkill
from .navigation import NavigationConfig, path_to_actions, planar_distance, shortest_grid_path
from .replay import ReplayBuffer, ReplayTransition
from .thor_adapter import SharedThorWorld, ThorConfig
from .world_map import PersistentWorldMap


@dataclass
class WorkerSpec:
    worker_id: int
    episodes: int
    scene: str
    grid_size: float
    rotate_step_degrees: float
    visibility_distance: float
    field_of_view: int
    width: int
    height: int
    headless: bool
    thor_executable: str | None
    seed: int
    peer_encounter_rate: float
    peer_encounter_distance: float
    max_steps_per_episode: int
    worker_root: str
    replay_max_records: int


@dataclass
class WorkerMetrics:
    worker_id: int
    episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    primitive_steps: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    peer_training_episodes: int = 0
    peer_encounters: int = 0
    reachable_cells: int = 0
    unique_cells: int = 0
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        total_actions = self.successful_actions + self.failed_actions
        data["route_success_rate_pct"] = round(
            100.0 * self.successful_episodes / self.episodes, 2
        ) if self.episodes else 0.0
        data["action_success_rate_pct"] = round(
            100.0 * self.successful_actions / total_actions, 2
        ) if total_actions else 0.0
        data["steps_per_wall_second"] = round(
            self.primitive_steps / self.wall_seconds, 2
        ) if self.wall_seconds > 0 else 0.0
        data["coverage_pct"] = round(
            min(100.0, 100.0 * self.unique_cells / self.reachable_cells), 2
        ) if self.reachable_cells else 0.0
        return data


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target.resolve()}")
    value = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    return value


def _teleport(world: SharedThorWorld, agent_id: int, position: dict[str, float], yaw: float) -> bool:
    if world.controller is None:
        return False
    event = world.controller.step(
        action="TeleportFull",
        agentId=int(agent_id),
        position={
            "x": float(position["x"]),
            "y": float(position.get("y", 0.9)),
            "z": float(position["z"]),
        },
        rotation={"x": 0.0, "y": float(yaw) % 360.0, "z": 0.0},
        horizon=0.0,
        standing=True,
    )
    events = getattr(event, "events", None)
    selected = events[int(agent_id)] if events is not None else event
    return selected.metadata.get("lastActionSuccess") is not False


def _reward(before, after, target: dict[str, float], success: bool, reached: bool) -> float:
    before_d = planar_distance(before.pose.position, target)
    after_d = planar_distance(after.pose.position, target)
    progress = before_d - after_d
    value = progress * 2.0
    value += 0.08 if success else -1.0
    if reached:
        value += 2.0
    return round(value, 6)


def _run_route(
    *,
    adapter,
    agent_label: str,
    episode: int,
    target: dict[str, float],
    reachable: list[dict[str, float]],
    nav_cfg: NavigationConfig,
    world_map: PersistentWorldMap,
    skill: NavigationSkill,
    replay: ReplayBuffer,
    peer_training: bool,
    arrival_distance: float,
    max_steps: int,
) -> tuple[bool, int, int, int]:
    start_world = adapter.observe()
    world_map.record_pose(start_world.pose.position)
    path = shortest_grid_path(
        reachable,
        start_world.pose.position,
        target,
        NavigationConfig(
            grid_size=nav_cfg.grid_size,
            rotate_step_degrees=nav_cfg.rotate_step_degrees,
            interaction_distance=float(arrival_distance),
        ),
    )
    teacher_actions = path_to_actions(
        path,
        initial_yaw=float(start_world.pose.rotation.get("y", 0.0)),
        target_position=target,
        config=nav_cfg,
    )
    if not teacher_actions:
        reached = planar_distance(start_world.pose.position, target) <= arrival_distance + 1e-6
        return reached, 0, 0, 0

    steps = 0
    successes = 0
    failures = 0
    for primitive in teacher_actions:
        if steps >= max_steps:
            break
        before = adapter.observe()
        state = skill.state(
            before.pose.position,
            float(before.pose.rotation.get("y", 0.0)),
            target,
            grid_size=nav_cfg.grid_size,
        )
        execution = adapter.execute(Action(primitive.kind, dict(primitive.params), primitive.reason))
        after = execution.state
        steps += max(1, int(execution.primitive_steps))
        ok = after.last_action_success is not False
        reached = planar_distance(after.pose.position, target) <= arrival_distance + 1e-6
        reward = _reward(before, after, target, ok, reached)
        next_state = skill.state(
            after.pose.position,
            float(after.pose.rotation.get("y", 0.0)),
            target,
            grid_size=nav_cfg.grid_size,
        )
        skill.update(state, primitive.kind, reward, next_state)
        world_map.record_transition(before.pose.position, after.pose.position, success=ok)
        replay.add(
            ReplayTransition(
                scene=str(after.scene),
                agent=agent_label,
                episode=int(episode),
                action=primitive.kind,
                before_position=dict(before.pose.position),
                before_yaw=float(before.pose.rotation.get("y", 0.0)),
                after_position=dict(after.pose.position),
                after_yaw=float(after.pose.rotation.get("y", 0.0)),
                target_position=dict(target),
                success=bool(ok),
                reward=float(reward),
                peer_training=bool(peer_training),
            )
        )
        if ok:
            successes += 1
        else:
            failures += 1
            break
    final_world = adapter.observe()
    reached = planar_distance(final_world.pose.position, target) <= arrival_distance + 1e-6
    return reached, steps, successes, failures


def _worker_main(spec: WorkerSpec) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(spec.seed)
    worker_root = Path(spec.worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    map_path = worker_root / "map.json"
    skill_path = worker_root / "navigation_skill.json"
    replay_path = worker_root / "replay.jsonl"
    events_path = worker_root / "events.jsonl"

    world_map = PersistentWorldMap(spec.scene, spec.grid_size, map_path)
    skill = NavigationSkill(skill_path)
    replay = ReplayBuffer(replay_path, max_records=spec.replay_max_records)
    metrics = WorkerMetrics(worker_id=spec.worker_id)

    thor_cfg = ThorConfig(
        scene=spec.scene,
        grid_size=spec.grid_size,
        rotate_step_degrees=spec.rotate_step_degrees,
        visibility_distance=spec.visibility_distance,
        interaction_distance=1.25,
        width=spec.width,
        height=spec.height,
        field_of_view=spec.field_of_view,
        headless=spec.headless,
        step_delay_seconds=0.0,
        render_instance_segmentation=False,
        local_executable_path=spec.thor_executable,
        learning_state_dir=None,
        use_learned_navigation=False,
    )
    nav_cfg = NavigationConfig(
        grid_size=spec.grid_size,
        rotate_step_degrees=spec.rotate_step_degrees,
        interaction_distance=1.25,
    )

    shared = SharedThorWorld(thor_cfg, agent_count=2)
    try:
        shared.start()
        adapters = {agent_id: shared.agent(agent_id) for agent_id in (0, 1)}
        for adapter in adapters.values():
            adapter.start()
        reachable = adapters[0].get_reachable_positions()
        metrics.reachable_cells = len(reachable)
        if len(reachable) < 2:
            raise RuntimeError("AI2-THOR returned fewer than two reachable positions for training.")

        for episode in range(1, spec.episodes + 1):
            metrics.episodes += 1
            start_a, start_b = rng.sample(reachable, 2)
            yaw_a = rng.choice((0.0, 90.0, 180.0, 270.0))
            yaw_b = rng.choice((0.0, 90.0, 180.0, 270.0))
            if not _teleport(shared, 0, start_a, yaw_a):
                metrics.failed_episodes += 1
                continue
            if not _teleport(shared, 1, start_b, yaw_b):
                metrics.failed_episodes += 1
                continue

            for adapter in adapters.values():
                adapter.observe()
                world_map.record_pose(adapter.state.pose.position)

            peer_episode = rng.random() < spec.peer_encounter_rate
            if peer_episode:
                metrics.peer_training_episodes += 1
                seeker = rng.choice((0, 1))
                friend = 1 - seeker
                friend_world = adapters[friend].observe()
                target = dict(friend_world.pose.position)
                reached, steps, ok_actions, bad_actions = _run_route(
                    adapter=adapters[seeker],
                    agent_label="A" if seeker == 0 else "B",
                    episode=episode,
                    target=target,
                    reachable=reachable,
                    nav_cfg=nav_cfg,
                    world_map=world_map,
                    skill=skill,
                    replay=replay,
                    peer_training=True,
                    arrival_distance=spec.peer_encounter_distance,
                    max_steps=spec.max_steps_per_episode,
                )
                metrics.primitive_steps += steps
                metrics.successful_actions += ok_actions
                metrics.failed_actions += bad_actions
                seeker_world = adapters[seeker].observe()
                friend_world = adapters[friend].observe()
                actual_distance = planar_distance(seeker_world.pose.position, friend_world.pose.position)
                encountered = bool(reached and actual_distance <= spec.peer_encounter_distance + 0.15)
                if encountered:
                    metrics.peer_encounters += 1
                    append_jsonl(
                        events_path,
                        {
                            "timestamp": utc_now(),
                            "kind": "training_peer_encounter",
                            "episode": episode,
                            "seeker": "A" if seeker == 0 else "B",
                            "peer": "B" if seeker == 0 else "A",
                            "distance_m": round(actual_distance, 3),
                        },
                    )
                if encountered:
                    metrics.successful_episodes += 1
                else:
                    metrics.failed_episodes += 1
            else:
                active = episode % 2
                target = dict(rng.choice(reachable))
                reached, steps, ok_actions, bad_actions = _run_route(
                    adapter=adapters[active],
                    agent_label="A" if active == 0 else "B",
                    episode=episode,
                    target=target,
                    reachable=reachable,
                    nav_cfg=nav_cfg,
                    world_map=world_map,
                    skill=skill,
                    replay=replay,
                    peer_training=False,
                    arrival_distance=max(0.05, spec.grid_size * 0.6),
                    max_steps=spec.max_steps_per_episode,
                )
                metrics.primitive_steps += steps
                metrics.successful_actions += ok_actions
                metrics.failed_actions += bad_actions
                if reached:
                    metrics.successful_episodes += 1
                else:
                    metrics.failed_episodes += 1

        world_map.save()
        skill.save()
        replay.finalize()
        metrics.unique_cells = len(world_map.cells)
        metrics.wall_seconds = time.perf_counter() - started
        write_json_atomic(worker_root / "metrics.json", metrics.as_dict())
        return metrics.as_dict()
    finally:
        shared.close()


def _split_episodes(total: int, workers: int) -> list[int]:
    workers = max(1, min(int(workers), int(total)))
    base, extra = divmod(int(total), workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


def _merge_workers(paths: LearningPaths, scene: str, grid_size: float, worker_dirs: list[Path], replay_max: int) -> dict[str, Any]:
    global_map = PersistentWorldMap(scene, grid_size, paths.scene_map(scene))
    global_skill = NavigationSkill(paths.navigation_checkpoint)

    worker_metrics: list[dict[str, Any]] = []
    map_sources: list[PersistentWorldMap] = []
    skill_sources: list[NavigationSkill] = []
    replay_sources: list[Path] = []
    event_sources: list[Path] = []
    for directory in worker_dirs:
        metrics = load_json(directory / "metrics.json", {})
        if metrics:
            worker_metrics.append(metrics)
        map_path = directory / "map.json"
        if map_path.exists():
            map_sources.append(PersistentWorldMap(scene, grid_size, map_path))
        skill_path = directory / "navigation_skill.json"
        if skill_path.exists():
            skill_sources.append(NavigationSkill(skill_path))
        replay_sources.append(directory / "replay.jsonl")
        event_sources.append(directory / "events.jsonl")

    for item in map_sources:
        global_map.merge(item)
    global_map.save()
    for item in skill_sources:
        global_skill.merge(item)
    global_skill.save()
    merged_transitions = merge_jsonl_files(replay_sources, paths.replay_file, max_records=replay_max)

    # Persist only explicit encounter events as social memory. Training map internals
    # remain in the motor layer and are not dumped into the LLM context.
    peer_events: list[dict[str, Any]] = []
    for source in event_sources:
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                peer_events.append(row)

    totals = {
        "episodes": sum(int(row.get("episodes", 0)) for row in worker_metrics),
        "successful_episodes": sum(int(row.get("successful_episodes", 0)) for row in worker_metrics),
        "failed_episodes": sum(int(row.get("failed_episodes", 0)) for row in worker_metrics),
        "primitive_steps": sum(int(row.get("primitive_steps", 0)) for row in worker_metrics),
        "successful_actions": sum(int(row.get("successful_actions", 0)) for row in worker_metrics),
        "failed_actions": sum(int(row.get("failed_actions", 0)) for row in worker_metrics),
        "peer_training_episodes": sum(int(row.get("peer_training_episodes", 0)) for row in worker_metrics),
        "peer_encounters": sum(int(row.get("peer_encounters", 0)) for row in worker_metrics),
        "reachable_cells": max([int(row.get("reachable_cells", 0)) for row in worker_metrics] or [0]),
        "unique_cells": len(global_map.cells),
        "merged_replay_transitions": merged_transitions,
        "navigation_skill_transitions": global_skill.transitions,
    }
    totals["route_success_rate_pct"] = round(
        100.0 * totals["successful_episodes"] / totals["episodes"], 2
    ) if totals["episodes"] else 0.0
    totals["action_success_rate_pct"] = round(
        100.0 * totals["successful_actions"] /
        max(1, totals["successful_actions"] + totals["failed_actions"]),
        2,
    )
    totals["coverage_pct"] = round(
        100.0 * totals["unique_cells"] / totals["reachable_cells"], 2
    ) if totals["reachable_cells"] else 0.0
    return {"totals": totals, "workers": worker_metrics, "peer_events": peer_events}


def _append_training_memories(paths: LearningPaths, summary: dict[str, Any], max_records: int) -> None:
    safe_summary = {
        "scene": summary.get("scene"),
        "episodes": summary.get("totals", {}).get("episodes", 0),
        "primitive_steps": summary.get("totals", {}).get("primitive_steps", 0),
        "map_cells_learned": summary.get("totals", {}).get("unique_cells", 0),
        "coverage_pct": summary.get("totals", {}).get("coverage_pct", 0.0),
        "peer_encounters": summary.get("totals", {}).get("peer_encounters", 0),
        "navigation_checkpoint_saved": True,
    }
    for label in ("A", "B"):
        memory = EpisodicMemory(str(paths.agent_memory(label)), max_records=max_records, load_existing=True)
        memory.add("training_summary", safe_summary)
    for event in summary.get("peer_events", [])[-100:]:
        seeker = str(event.get("seeker", "A"))
        memory = EpisodicMemory(str(paths.agent_memory(seeker)), max_records=max_records, load_existing=True)
        memory.add(
            "training_peer_encounter",
            {
                "peer": event.get("peer"),
                "distance_m": event.get("distance_m"),
                "scene": summary.get("scene"),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI2-THOR v0.8 accelerated navigation/social encounter trainer (no LLM required)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=72611)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--thor-executable", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--peer-encounter-rate", type=float, default=None)
    parser.add_argument("--peer-encounter-distance", type=float, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the configured persistent state directory before training, then start from a clean state.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate/print the training plan without starting Unity.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    training_cfg = cfg.get("training", {})
    learning_cfg = cfg.get("learning", {})
    sim_cfg = cfg.get("simulation", {})

    scene = str(args.scene or cfg.get("scene", "FloorPlan1"))
    episodes = int(args.episodes if args.episodes is not None else training_cfg.get("episodes", 500))
    workers = int(args.workers if args.workers is not None else training_cfg.get("workers", 1))
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if workers <= 0:
        raise ValueError("--workers must be positive.")

    state_dir = Path(args.state_dir or learning_cfg.get("state_dir", "state")).expanduser().resolve()
    if args.reset_memory:
        resolved = reset_learning_state(state_dir)
        print(f"Cleared persistent learning/memory state: {resolved}", flush=True)
    paths = LearningPaths.from_root(state_dir).ensure()

    peer_rate = float(
        args.peer_encounter_rate
        if args.peer_encounter_rate is not None
        else training_cfg.get("peer_encounter_rate", 0.25)
    )
    if not 0.0 <= peer_rate <= 1.0:
        raise ValueError("--peer-encounter-rate must be in [0, 1].")
    peer_distance = float(
        args.peer_encounter_distance
        if args.peer_encounter_distance is not None
        else training_cfg.get("peer_encounter_distance_m", 1.25)
    )
    max_steps = int(
        args.max_steps_per_episode
        if args.max_steps_per_episode is not None
        else training_cfg.get("max_steps_per_episode", 120)
    )
    width = int(args.width if args.width is not None else training_cfg.get("image_width", 160))
    height = int(args.height if args.height is not None else training_cfg.get("image_height", 120))
    thor_executable = args.thor_executable or sim_cfg.get("local_executable_path") or None
    replay_max = int(learning_cfg.get("replay_max_records", 250_000))
    memory_max = int(learning_cfg.get("memory_max_records", 12_000))

    plan = {
        "version": "0.8.0",
        "scene": scene,
        "episodes": episodes,
        "workers": min(workers, episodes),
        "peer_encounter_rate": peer_rate,
        "peer_encounter_distance_m": peer_distance,
        "max_steps_per_episode": max_steps,
        "render_size": [width, height],
        "headless": bool(args.headless),
        "llm_calls": 0,
        "state_dir": str(paths.root),
        "custom_thor_executable": thor_executable,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_workers_root = paths.worker_dir / run_id
    run_workers_root.mkdir(parents=True, exist_ok=True)
    splits = _split_episodes(episodes, workers)
    specs: list[WorkerSpec] = []
    for worker_id, worker_episodes in enumerate(splits):
        worker_root = run_workers_root / f"worker_{worker_id:02d}"
        specs.append(
            WorkerSpec(
                worker_id=worker_id,
                episodes=worker_episodes,
                scene=scene,
                grid_size=float(cfg.get("grid_size", 0.25)),
                rotate_step_degrees=float(cfg.get("rotate_step_degrees", 90)),
                visibility_distance=float(cfg.get("visibility_distance", 1.5)),
                field_of_view=int(cfg.get("field_of_view", 90)),
                width=width,
                height=height,
                headless=bool(args.headless),
                thor_executable=thor_executable,
                seed=int(args.seed) + worker_id * 100003,
                peer_encounter_rate=peer_rate,
                peer_encounter_distance=peer_distance,
                max_steps_per_episode=max_steps,
                worker_root=str(worker_root),
                replay_max_records=replay_max,
            )
        )

    wall_started = time.perf_counter()
    if len(specs) == 1:
        results = [_worker_main(specs[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(specs)) as pool:
            results = pool.map(_worker_main, specs)

    merged = _merge_workers(
        paths,
        scene=scene,
        grid_size=float(cfg.get("grid_size", 0.25)),
        worker_dirs=[Path(spec.worker_root) for spec in specs],
        replay_max=replay_max,
    )
    total_wall = time.perf_counter() - wall_started
    summary = {
        "version": "0.8.0",
        "timestamp": utc_now(),
        "run_id": run_id,
        "scene": scene,
        "workers": len(specs),
        "wall_seconds": round(total_wall, 3),
        **merged,
    }
    totals = summary["totals"]
    totals["aggregate_steps_per_wall_second"] = round(
        totals["primitive_steps"] / total_wall, 2
    ) if total_wall > 0 else 0.0
    write_json_atomic(paths.training_summary, summary)
    append_jsonl(paths.training_history, {k: v for k, v in summary.items() if k != "peer_events"})
    _append_training_memories(paths, summary, max_records=memory_max)

    # Worker artifacts are useful while diagnosing a failed run, but successful
    # aggregation no longer needs them. Keep the current run by default if the
    # config asks for it; otherwise remove to avoid growing state indefinitely.
    if not bool(training_cfg.get("keep_worker_artifacts", False)):
        shutil.rmtree(run_workers_root, ignore_errors=True)

    print("\nTraining complete.", flush=True)
    print(json.dumps({"state_dir": str(paths.root), **totals}, indent=2), flush=True)
    print(f"Navigation checkpoint: {paths.navigation_checkpoint}", flush=True)
    print(f"Persistent map:       {paths.scene_map(scene)}", flush=True)
    print(f"Replay buffer:        {paths.replay_file}", flush=True)
    print(f"Training summary:     {paths.training_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
