from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import time
from typing import Any

import yaml

from .embodied_skills import EmbodiedSkillLibrary
from .executor import Verifier
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
from .models import Action, ActionKind
from .nav_skill import NavigationSkill
from .navigation import NavigationConfig, path_to_actions, planar_distance, shortest_grid_path
from .replay import ReplayBuffer, ReplayTransition
from .thor_adapter import SharedThorWorld, ThorConfig
from .world_map import PersistentWorldMap


DEFAULT_CURRICULUM = {
    "navigate": 0.24,
    "open_cross": 0.24,
    "pickup": 0.18,
    "place": 0.12,
    "toggle": 0.07,
    "peer_meet": 0.15,
}


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
    max_steps_per_episode: int
    worker_root: str
    replay_max_records: int
    curriculum: dict[str, float]
    export_sft: bool = True


@dataclass
class WorkerMetrics:
    worker_id: int
    episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    primitive_steps: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    reachable_cells: int = 0
    unique_cells: int = 0
    wall_seconds: float = 0.0
    curriculum_attempts: dict[str, int] = field(default_factory=dict)
    curriculum_successes: dict[str, int] = field(default_factory=dict)

    def note_episode(self, skill: str, success: bool) -> None:
        self.episodes += 1
        self.curriculum_attempts[skill] = self.curriculum_attempts.get(skill, 0) + 1
        if success:
            self.successful_episodes += 1
            self.curriculum_successes[skill] = self.curriculum_successes.get(skill, 0) + 1
        else:
            self.failed_episodes += 1

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        total_actions = self.successful_actions + self.failed_actions
        data["episode_success_rate_pct"] = round(
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
        data["curriculum_success_rate_pct"] = {
            skill: round(
                100.0 * self.curriculum_successes.get(skill, 0) / max(1, attempts), 2
            )
            for skill, attempts in sorted(self.curriculum_attempts.items())
        }
        return data


@dataclass
class EpisodeResult:
    skill: str
    success: bool
    primitive_steps: int
    successful_actions: int
    failed_actions: int
    agent_label: str
    sequence: list[str] = field(default_factory=list)
    object_type: str | None = None
    note: str | None = None


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


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    rows = [(str(name), max(0.0, float(weight))) for name, weight in weights.items() if float(weight) > 0]
    if not rows:
        return "navigate"
    total = sum(weight for _, weight in rows)
    pick = rng.random() * total
    acc = 0.0
    for name, weight in rows:
        acc += weight
        if pick <= acc:
            return name
    return rows[-1][0]


def _object_type(world, object_id: str | None) -> str | None:
    if not object_id:
        return None
    obj = world.objects.get(object_id)
    return None if obj is None else obj.object_type


def _append_embodied_transition(
    path: Path,
    *,
    scene: str,
    agent: str,
    episode: int,
    curriculum_skill: str,
    action: Action,
    before,
    after,
    success: bool,
    verification: dict[str, Any],
    primitive_steps: int,
) -> None:
    append_jsonl(
        path,
        {
            "timestamp": utc_now(),
            "scene": scene,
            "agent": agent,
            "episode": int(episode),
            "curriculum_skill": curriculum_skill,
            "action": action.kind,
            "target_object_type": _object_type(before, action.params.get("objectId")),
            "before_position": dict(before.pose.position),
            "before_yaw": float(before.pose.rotation.get("y", 0.0)),
            "after_position": dict(after.pose.position),
            "after_yaw": float(after.pose.rotation.get("y", 0.0)),
            "success": bool(success),
            "primitive_steps": int(primitive_steps),
            "verification": verification,
        },
    )


def _execute_teacher_action(
    *,
    adapter,
    action: Action,
    verifier: Verifier,
    replay_path: Path,
    scene: str,
    agent_label: str,
    episode: int,
    curriculum_skill: str,
    library: EmbodiedSkillLibrary,
    skill_name: str | None = None,
    object_type: str | None = None,
    max_steps: int | None = None,
) -> tuple[bool, int]:
    before = adapter.observe()
    try:
        execution = adapter.execute(action, step_budget=max_steps)
        after = execution.state
        primitive_steps = int(execution.primitive_steps)
        event_ok = after.last_action_success is not False
        verified, verification = verifier.verify(before, after, action)
        if action.kind == ActionKind.CROSS_THRESHOLD.value:
            moved = planar_distance(before.pose.position, after.pose.position)
            verification = {"distance_moved": round(moved, 3)}
            verified = moved >= 0.35
        success = bool(event_ok and verified)
    except Exception as exc:
        after = adapter.observe()
        primitive_steps = 0
        success = False
        verification = {"exception": f"{type(exc).__name__}: {exc}"}

    _append_embodied_transition(
        replay_path,
        scene=scene,
        agent=agent_label,
        episode=episode,
        curriculum_skill=curriculum_skill,
        action=action,
        before=before,
        after=after,
        success=success,
        verification=verification,
        primitive_steps=primitive_steps,
    )
    if skill_name:
        library.record(
            skill_name,
            success=success,
            primitive_steps=primitive_steps,
            object_type=object_type,
        )
    return success, primitive_steps


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
    arrival_distance: float,
    max_steps: int,
    embodied_library: EmbodiedSkillLibrary,
) -> EpisodeResult:
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
        embodied_library.record("navigate", success=reached, primitive_steps=0)
        return EpisodeResult("navigate", reached, 0, 0, 0, agent_label, [])

    steps = successes = failures = 0
    sequence: list[str] = []
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
        sequence.append(primitive.kind)
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
                peer_training=False,
            )
        )
        if ok:
            successes += 1
        else:
            failures += 1
            break
    final_world = adapter.observe()
    reached = planar_distance(final_world.pose.position, target) <= arrival_distance + 1e-6
    embodied_library.record("navigate", success=reached, primitive_steps=steps)
    return EpisodeResult("navigate", reached, steps, successes, failures, agent_label, sequence)


def _candidate_objects(world, predicate) -> list[Any]:
    return [obj for obj in world.objects.values() if obj.position is not None and predicate(obj)]


def _run_interaction_macro(
    *,
    adapter,
    agent_label: str,
    episode: int,
    curriculum_skill: str,
    object_id: str,
    final_action_kind: str,
    replay_path: Path,
    library: EmbodiedSkillLibrary,
    verifier: Verifier,
    interaction_distance: float,
    extra_params: dict[str, Any] | None = None,
    cross_after_open: bool = False,
    max_steps: int = 120,
) -> EpisodeResult:
    world = adapter.observe()
    obj = world.objects.get(object_id)
    object_type = None if obj is None else obj.object_type
    sequence: list[str] = []
    total_steps = good = bad = 0

    actions: list[tuple[Action, str | None]] = [
        (Action(ActionKind.NAVIGATE_TO.value, {"objectId": object_id, "interactionDistance": interaction_distance}, "teacher navigate to selected target"), "navigate"),
        (Action(ActionKind.FACE_TARGET.value, {"objectId": object_id}, "teacher face selected target"), "face_target"),
        (Action(final_action_kind, {"objectId": object_id, **(extra_params or {})}, "teacher direct interaction"), curriculum_skill if curriculum_skill != "open_cross" else "open"),
    ]
    if cross_after_open:
        actions.append((Action(ActionKind.CROSS_THRESHOLD.value, {"objectId": object_id}, "teacher cross selected open doorway"), "cross_door"))

    success = True
    for action, skill_name in actions:
        remaining = max(0, max_steps - total_steps)
        if remaining <= 0:
            success = False
            break
        ok, steps = _execute_teacher_action(
            adapter=adapter,
            action=action,
            verifier=verifier,
            replay_path=replay_path,
            scene=str(world.scene),
            agent_label=agent_label,
            episode=episode,
            curriculum_skill=curriculum_skill,
            library=library,
            skill_name=skill_name,
            object_type=object_type,
            max_steps=remaining,
        )
        sequence.append(action.kind)
        total_steps += steps
        good += int(ok)
        bad += int(not ok)
        if not ok:
            success = False
            break

    return EpisodeResult(
        curriculum_skill,
        success,
        total_steps,
        good,
        bad,
        agent_label,
        sequence,
        object_type=object_type,
    )


def _train_open_cross(*, adapter, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, interaction_distance: float, max_steps: int) -> EpisodeResult:
    world = adapter.observe()
    doors = _candidate_objects(world, lambda obj: obj.openable and obj.object_type.lower() == "door")
    if not doors:
        return EpisodeResult("open_cross", False, 0, 0, 0, agent_label, note="no Door objects")
    door = rng.choice(doors)

    # Restore a clean closed-door training condition when possible.
    if door.is_open is True:
        prep = _run_interaction_macro(
            adapter=adapter,
            agent_label=agent_label,
            episode=episode,
            curriculum_skill="close_cleanup",
            object_id=door.object_id,
            final_action_kind=ActionKind.CLOSE.value,
            replay_path=replay_path,
            library=library,
            verifier=verifier,
            interaction_distance=interaction_distance,
            max_steps=max_steps,
        )
        if not prep.success:
            return EpisodeResult("open_cross", False, prep.primitive_steps, prep.successful_actions, prep.failed_actions, agent_label, prep.sequence, door.object_type, "could not close training door")

    return _run_interaction_macro(
        adapter=adapter,
        agent_label=agent_label,
        episode=episode,
        curriculum_skill="open_cross",
        object_id=door.object_id,
        final_action_kind=ActionKind.OPEN.value,
        replay_path=replay_path,
        library=library,
        verifier=verifier,
        interaction_distance=interaction_distance,
        cross_after_open=True,
        max_steps=max_steps,
    )


def _train_pickup(*, adapter, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, interaction_distance: float, max_steps: int) -> EpisodeResult:
    world = adapter.observe()
    if world.held_object_id:
        try:
            adapter.execute(Action(ActionKind.DROP.value, {}, "training cleanup"))
        except Exception:
            pass
        world = adapter.observe()
    objects = _candidate_objects(world, lambda obj: obj.pickupable and not obj.is_picked_up)
    if not objects:
        return EpisodeResult("pickup", False, 0, 0, 0, agent_label, note="no pickupable objects")
    obj = rng.choice(objects)
    result = _run_interaction_macro(
        adapter=adapter,
        agent_label=agent_label,
        episode=episode,
        curriculum_skill="pickup",
        object_id=obj.object_id,
        final_action_kind=ActionKind.PICKUP.value,
        replay_path=replay_path,
        library=library,
        verifier=verifier,
        interaction_distance=interaction_distance,
        max_steps=max_steps,
    )
    # Keep the training world usable. If we can put it back where it came from, do so;
    # otherwise drop it. Cleanup is not counted as curriculum success.
    if result.success:
        original_parents = list(obj.parent_receptacles)
        try:
            if original_parents and original_parents[0] in adapter.observe().objects:
                target_id = original_parents[0]
                adapter.execute(Action(ActionKind.NAVIGATE_TO.value, {"objectId": target_id, "interactionDistance": interaction_distance}))
                adapter.execute(Action(ActionKind.FACE_TARGET.value, {"objectId": target_id}))
                adapter.execute(Action(ActionKind.PUT.value, {"objectId": target_id}))
            else:
                adapter.execute(Action(ActionKind.DROP.value, {}))
        except Exception:
            pass
    return result


def _train_place(*, adapter, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, interaction_distance: float, max_steps: int) -> EpisodeResult:
    world = adapter.observe()
    if world.held_object_id:
        try:
            adapter.execute(Action(ActionKind.DROP.value, {}, "training cleanup"))
        except Exception:
            pass
        world = adapter.observe()

    sources = [
        obj for obj in _candidate_objects(world, lambda o: o.pickupable and not o.is_picked_up)
        if obj.parent_receptacles and any(parent in world.objects and world.objects[parent].receptacle for parent in obj.parent_receptacles)
    ]
    if not sources:
        return EpisodeResult("place", False, 0, 0, 0, agent_label, note="no pickupable object with known parent receptacle")
    source = rng.choice(sources)
    target_id = next(parent for parent in source.parent_receptacles if parent in world.objects and world.objects[parent].receptacle)
    target_type = world.objects[target_id].object_type

    pickup = _run_interaction_macro(
        adapter=adapter,
        agent_label=agent_label,
        episode=episode,
        curriculum_skill="place_prepare_pickup",
        object_id=source.object_id,
        final_action_kind=ActionKind.PICKUP.value,
        replay_path=replay_path,
        library=library,
        verifier=verifier,
        interaction_distance=interaction_distance,
        max_steps=max_steps,
    )
    if not pickup.success:
        return EpisodeResult("place", False, pickup.primitive_steps, pickup.successful_actions, pickup.failed_actions, agent_label, pickup.sequence, target_type, "pickup stage failed")

    result = _run_interaction_macro(
        adapter=adapter,
        agent_label=agent_label,
        episode=episode,
        curriculum_skill="place",
        object_id=target_id,
        final_action_kind=ActionKind.PUT.value,
        replay_path=replay_path,
        library=library,
        verifier=verifier,
        interaction_distance=interaction_distance,
        max_steps=max(1, max_steps - pickup.primitive_steps),
    )
    result.primitive_steps += pickup.primitive_steps
    result.successful_actions += pickup.successful_actions
    result.failed_actions += pickup.failed_actions
    result.sequence = pickup.sequence + result.sequence
    return result


def _train_toggle(*, adapter, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, interaction_distance: float, max_steps: int) -> EpisodeResult:
    world = adapter.observe()
    objects = _candidate_objects(world, lambda obj: obj.toggleable and obj.is_toggled is not None)
    if not objects:
        return EpisodeResult("toggle", False, 0, 0, 0, agent_label, note="no toggleable objects")
    obj = rng.choice(objects)
    kind = ActionKind.TOGGLE_OFF.value if obj.is_toggled else ActionKind.TOGGLE_ON.value
    result = _run_interaction_macro(
        adapter=adapter,
        agent_label=agent_label,
        episode=episode,
        curriculum_skill="toggle",
        object_id=obj.object_id,
        final_action_kind=kind,
        replay_path=replay_path,
        library=library,
        verifier=verifier,
        interaction_distance=interaction_distance,
        max_steps=max_steps,
    )
    if result.success:
        try:
            reverse = ActionKind.TOGGLE_ON.value if kind == ActionKind.TOGGLE_OFF.value else ActionKind.TOGGLE_OFF.value
            adapter.execute(Action(reverse, {"objectId": obj.object_id}, "training cleanup"))
        except Exception:
            pass
    return result


def _train_peer_meet(*, adapters: dict[int, Any], active: int, agent_label: str, episode: int, library: EmbodiedSkillLibrary, max_steps: int, peer_distance: float = 1.25) -> EpisodeResult:
    peer = 1 - active
    target = dict(adapters[peer].observe().pose.position)
    reachable = adapters[active].get_reachable_positions()
    nav_cfg = NavigationConfig(grid_size=adapters[active].config.grid_size, rotate_step_degrees=adapters[active].config.rotate_step_degrees, interaction_distance=peer_distance)
    path = shortest_grid_path(reachable, adapters[active].observe().pose.position, target, nav_cfg)
    actions = path_to_actions(path, initial_yaw=float(adapters[active].observe().pose.rotation.get("y", 0.0)), target_position=target, config=nav_cfg)
    steps = good = bad = 0
    sequence: list[str] = []
    for action in actions:
        if steps >= max_steps:
            break
        execution = adapters[active].execute(action)
        steps += max(1, int(execution.primitive_steps))
        ok = execution.state.last_action_success is not False
        sequence.append(action.kind)
        good += int(ok)
        bad += int(not ok)
        if not ok:
            break
    actual = planar_distance(adapters[active].observe().pose.position, adapters[peer].observe().pose.position)
    success = actual <= peer_distance + 0.15
    library.record("peer_meet", success=success, primitive_steps=steps, object_type="Peer")
    return EpisodeResult("peer_meet", success, steps, good, bad, agent_label, sequence, "Peer", f"distance={actual:.3f}")


def _append_sft(path: Path, result: EpisodeResult, scene: str) -> None:
    if not result.success or not result.sequence:
        return
    target = result.object_type or "target"
    action_text = " -> ".join(result.sequence)
    append_jsonl(
        path,
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are learning an embodied motor skill. Choose actions only after a semantic goal/target has already been selected. Do not invent hidden targets.",
                },
                {
                    "role": "user",
                    "content": f"Scene={scene}. Skill={result.skill}. Selected target type={target}. Execute the skill reliably.",
                },
                {"role": "assistant", "content": action_text},
            ],
            "metadata": {
                "agent": result.agent_label,
                "skill": result.skill,
                "target_object_type": result.object_type,
                "primitive_steps": result.primitive_steps,
                "source": "AUTHOR accelerated teacher trajectory",
            },
        },
    )


def _worker_main(spec: WorkerSpec) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(spec.seed)
    worker_root = Path(spec.worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    map_path = worker_root / "map.json"
    nav_skill_path = worker_root / "navigation_skill.json"
    nav_replay_path = worker_root / "navigation_replay.jsonl"
    embodied_replay_path = worker_root / "embodied_replay.jsonl"
    sft_path = worker_root / "embodied_sft.jsonl"
    skill_paths = {"A": worker_root / "skills_A.json", "B": worker_root / "skills_B.json"}

    world_map = PersistentWorldMap(spec.scene, spec.grid_size, map_path)
    nav_skill = NavigationSkill(nav_skill_path)
    nav_replay = ReplayBuffer(nav_replay_path, max_records=spec.replay_max_records)
    skill_libs = {label: EmbodiedSkillLibrary(path, agent_label=label) for label, path in skill_paths.items()}
    metrics = WorkerMetrics(worker_id=spec.worker_id)
    verifier = Verifier(default_interaction_distance=1.5)

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
            # Re-query because open doors can expand the connected training space.
            reachable = adapters[0].get_reachable_positions()
            if len(reachable) >= 2:
                start_a, start_b = rng.sample(reachable, 2)
                _teleport(shared, 0, start_a, rng.choice((0.0, 90.0, 180.0, 270.0)))
                _teleport(shared, 1, start_b, rng.choice((0.0, 90.0, 180.0, 270.0)))
            active = (episode + spec.worker_id) % 2
            label = "A" if active == 0 else "B"
            adapter = adapters[active]
            library = skill_libs[label]
            world_map.record_pose(adapter.observe().pose.position)
            chosen = _weighted_choice(rng, spec.curriculum)

            if chosen == "navigate":
                target = dict(rng.choice(reachable))
                result = _run_route(
                    adapter=adapter,
                    agent_label=label,
                    episode=episode,
                    target=target,
                    reachable=reachable,
                    nav_cfg=nav_cfg,
                    world_map=world_map,
                    skill=nav_skill,
                    replay=nav_replay,
                    arrival_distance=max(0.05, spec.grid_size * 0.6),
                    max_steps=spec.max_steps_per_episode,
                    embodied_library=library,
                )
            elif chosen == "open_cross":
                result = _train_open_cross(adapter=adapter, agent_label=label, episode=episode, rng=rng, replay_path=embodied_replay_path, library=library, verifier=verifier, interaction_distance=1.25, max_steps=spec.max_steps_per_episode)
            elif chosen == "pickup":
                result = _train_pickup(adapter=adapter, agent_label=label, episode=episode, rng=rng, replay_path=embodied_replay_path, library=library, verifier=verifier, interaction_distance=1.25, max_steps=spec.max_steps_per_episode)
            elif chosen == "place":
                result = _train_place(adapter=adapter, agent_label=label, episode=episode, rng=rng, replay_path=embodied_replay_path, library=library, verifier=verifier, interaction_distance=1.25, max_steps=spec.max_steps_per_episode)
            elif chosen == "toggle":
                result = _train_toggle(adapter=adapter, agent_label=label, episode=episode, rng=rng, replay_path=embodied_replay_path, library=library, verifier=verifier, interaction_distance=1.25, max_steps=spec.max_steps_per_episode)
            elif chosen == "peer_meet":
                result = _train_peer_meet(adapters=adapters, active=active, agent_label=label, episode=episode, library=library, max_steps=spec.max_steps_per_episode)
            else:
                result = EpisodeResult(chosen, False, 0, 0, 0, label, note="unknown curriculum skill")

            metrics.note_episode(chosen, result.success)
            metrics.primitive_steps += result.primitive_steps
            metrics.successful_actions += result.successful_actions
            metrics.failed_actions += result.failed_actions
            if spec.export_sft:
                _append_sft(sft_path, result, spec.scene)

        world_map.save()
        nav_skill.save()
        nav_replay.finalize()
        for library in skill_libs.values():
            library.save()
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


def _curriculum_from_config(training_cfg: dict[str, Any], override: str | None = None) -> dict[str, float]:
    if override:
        requested = [part.strip().lower() for part in override.split(",") if part.strip()]
        if not requested:
            return dict(DEFAULT_CURRICULUM)
        unknown = [name for name in requested if name not in DEFAULT_CURRICULUM]
        if unknown:
            raise ValueError(f"Unknown curriculum skill(s): {', '.join(unknown)}")
        return {name: 1.0 for name in requested}
    configured = training_cfg.get("curriculum") or {}
    if not isinstance(configured, dict) or not configured:
        return dict(DEFAULT_CURRICULUM)
    merged = {name: float(configured.get(name, 0.0)) for name in DEFAULT_CURRICULUM}
    if sum(max(0.0, v) for v in merged.values()) <= 0:
        return dict(DEFAULT_CURRICULUM)
    return merged


def _merge_workers(paths: LearningPaths, scene: str, grid_size: float, worker_dirs: list[Path], replay_max: int) -> dict[str, Any]:
    global_map = PersistentWorldMap(scene, grid_size, paths.scene_map(scene))
    global_nav = NavigationSkill(paths.navigation_checkpoint)
    global_skills = {
        label: EmbodiedSkillLibrary(paths.agent_skill_checkpoint(label), agent_label=label)
        for label in ("A", "B")
    }

    worker_metrics: list[dict[str, Any]] = []
    nav_replays: list[Path] = []
    embodied_replays: list[Path] = []
    sft_sources: list[Path] = []
    for directory in worker_dirs:
        metrics = load_json(directory / "metrics.json", {})
        if metrics:
            worker_metrics.append(metrics)
        map_path = directory / "map.json"
        if map_path.exists():
            global_map.merge(PersistentWorldMap(scene, grid_size, map_path))
        nav_path = directory / "navigation_skill.json"
        if nav_path.exists():
            global_nav.merge(NavigationSkill(nav_path))
        for label in ("A", "B"):
            skill_path = directory / f"skills_{label}.json"
            if skill_path.exists():
                global_skills[label].merge(EmbodiedSkillLibrary(skill_path, agent_label=label))
        nav_replays.append(directory / "navigation_replay.jsonl")
        embodied_replays.append(directory / "embodied_replay.jsonl")
        sft_sources.append(directory / "embodied_sft.jsonl")

    global_map.save()
    global_nav.save()
    for library in global_skills.values():
        library.save()
    merged_nav = merge_jsonl_files(nav_replays, paths.replay_file, max_records=replay_max)
    merged_embodied = merge_jsonl_files(embodied_replays, paths.embodied_replay_file, max_records=replay_max)
    merged_sft = merge_jsonl_files(sft_sources, paths.embodied_sft_file, max_records=replay_max)

    curriculum_attempts: dict[str, int] = {}
    curriculum_successes: dict[str, int] = {}
    for row in worker_metrics:
        for skill, count in (row.get("curriculum_attempts") or {}).items():
            curriculum_attempts[skill] = curriculum_attempts.get(skill, 0) + int(count)
        for skill, count in (row.get("curriculum_successes") or {}).items():
            curriculum_successes[skill] = curriculum_successes.get(skill, 0) + int(count)

    totals = {
        "episodes": sum(int(row.get("episodes", 0)) for row in worker_metrics),
        "successful_episodes": sum(int(row.get("successful_episodes", 0)) for row in worker_metrics),
        "failed_episodes": sum(int(row.get("failed_episodes", 0)) for row in worker_metrics),
        "primitive_steps": sum(int(row.get("primitive_steps", 0)) for row in worker_metrics),
        "successful_actions": sum(int(row.get("successful_actions", 0)) for row in worker_metrics),
        "failed_actions": sum(int(row.get("failed_actions", 0)) for row in worker_metrics),
        "reachable_cells": max([int(row.get("reachable_cells", 0)) for row in worker_metrics] or [0]),
        "unique_cells": len(global_map.cells),
        "merged_navigation_replay_transitions": merged_nav,
        "merged_embodied_replay_transitions": merged_embodied,
        "merged_sft_examples": merged_sft,
        "navigation_skill_transitions": global_nav.transitions,
        "curriculum_attempts": curriculum_attempts,
        "curriculum_successes": curriculum_successes,
        "agent_skills": {label: library.compact_summary() for label, library in global_skills.items()},
    }
    totals["episode_success_rate_pct"] = round(100.0 * totals["successful_episodes"] / max(1, totals["episodes"]), 2)
    totals["action_success_rate_pct"] = round(100.0 * totals["successful_actions"] / max(1, totals["successful_actions"] + totals["failed_actions"]), 2)
    totals["coverage_pct"] = round(100.0 * totals["unique_cells"] / max(1, totals["reachable_cells"]), 2)
    totals["curriculum_success_rate_pct"] = {
        skill: round(100.0 * curriculum_successes.get(skill, 0) / max(1, count), 2)
        for skill, count in sorted(curriculum_attempts.items())
    }
    return {"totals": totals, "workers": worker_metrics}


def _append_training_memories(paths: LearningPaths, summary: dict[str, Any], max_records: int) -> None:
    for label in ("A", "B"):
        own_skills = summary.get("totals", {}).get("agent_skills", {}).get(label, {})
        memory = EpisodicMemory(str(paths.agent_memory(label)), max_records=max_records, load_existing=True)
        memory.add(
            "accelerated_embodied_training_summary",
            {
                "scene": summary.get("scene"),
                "episodes": summary.get("totals", {}).get("episodes", 0),
                "primitive_steps": summary.get("totals", {}).get("primitive_steps", 0),
                "coverage_pct": summary.get("totals", {}).get("coverage_pct", 0.0),
                "your_embodied_skill_training": own_skills,
                "note": "These are your own body-skill competence statistics; they do not reveal the peer's private state.",
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AUTHOR v0.10 accelerated embodied curriculum trainer (zero LLM calls by default)."
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
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument(
        "--curriculum",
        default=None,
        help="Comma-separated skills to train exclusively: navigate,open_cross,pickup,place,toggle,peer_meet",
    )
    parser.add_argument("--no-sft-export", action="store_true", help="Do not export successful teacher trajectories for later LoRA/SFT work.")
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

    scene = str(args.scene or cfg.get("scene", "FloorPlan1Discovery_physics"))
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

    max_steps = int(args.max_steps_per_episode if args.max_steps_per_episode is not None else training_cfg.get("max_steps_per_episode", 120))
    width = int(args.width if args.width is not None else training_cfg.get("image_width", 160))
    height = int(args.height if args.height is not None else training_cfg.get("image_height", 120))
    thor_executable = args.thor_executable or sim_cfg.get("local_executable_path") or None
    replay_max = int(learning_cfg.get("replay_max_records", 250_000))
    memory_max = int(learning_cfg.get("memory_max_records", 12_000))
    curriculum = _curriculum_from_config(training_cfg, args.curriculum)
    export_sft = bool(training_cfg.get("export_sft", True)) and not args.no_sft_export

    plan = {
        "version": "0.10.0",
        "trainer": "embodied_curriculum",
        "scene": scene,
        "episodes": episodes,
        "workers": min(workers, episodes),
        "curriculum": curriculum,
        "max_steps_per_episode": max_steps,
        "render_size": [width, height],
        "headless": bool(args.headless),
        "llm_calls": 0,
        "export_sft": export_sft,
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
                max_steps_per_episode=max_steps,
                worker_root=str(run_workers_root / f"worker_{worker_id:02d}"),
                replay_max_records=replay_max,
                curriculum=dict(curriculum),
                export_sft=export_sft,
            )
        )

    wall_started = time.perf_counter()
    if len(specs) == 1:
        _worker_main(specs[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(specs)) as pool:
            pool.map(_worker_main, specs)

    merged = _merge_workers(
        paths,
        scene=scene,
        grid_size=float(cfg.get("grid_size", 0.25)),
        worker_dirs=[Path(spec.worker_root) for spec in specs],
        replay_max=replay_max,
    )
    total_wall = time.perf_counter() - wall_started
    summary = {
        "version": "0.10.0",
        "timestamp": utc_now(),
        "run_id": run_id,
        "scene": scene,
        "workers": len(specs),
        "wall_seconds": round(total_wall, 3),
        "curriculum": curriculum,
        **merged,
    }
    totals = summary["totals"]
    totals["aggregate_steps_per_wall_second"] = round(totals["primitive_steps"] / total_wall, 2) if total_wall > 0 else 0.0
    write_json_atomic(paths.training_summary, summary)
    append_jsonl(paths.training_history, summary)
    _append_training_memories(paths, summary, max_records=memory_max)

    if not bool(training_cfg.get("keep_worker_artifacts", False)):
        shutil.rmtree(run_workers_root, ignore_errors=True)

    print("\nEmbodied curriculum training complete.", flush=True)
    print(json.dumps({"state_dir": str(paths.root), **totals}, indent=2), flush=True)
    print(f"Navigation checkpoint: {paths.navigation_checkpoint}", flush=True)
    print(f"Agent A skills:        {paths.agent_skill_checkpoint('A')}", flush=True)
    print(f"Agent B skills:        {paths.agent_skill_checkpoint('B')}", flush=True)
    print(f"Persistent map:        {paths.scene_map(scene)}", flush=True)
    print(f"Embodied replay:       {paths.embodied_replay_file}", flush=True)
    if export_sft:
        print(f"SFT/LoRA dataset:      {paths.embodied_sft_file}", flush=True)
    print(f"Training summary:      {paths.training_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
