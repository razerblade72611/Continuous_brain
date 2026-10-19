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
from typing import Any, Iterable

import yaml

from .embodied_skills import EmbodiedSkillLibrary
from .actionability import PoseActionabilityMemory
from .executor import Verifier
from .interaction_pose import InteractionPose
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
from .navigation import NavigationConfig, facing_yaw, path_to_actions, planar_distance, shortest_grid_path
from .replay import ReplayBuffer, ReplayTransition
from .thor_adapter import SharedThorWorld, ThorConfig, SimulatorStepTimeout, SimulatorTransportError
from .world_map import PersistentWorldMap


DEFAULT_CURRICULUM = {
    "navigate": 0.20,
    "door": 0.25,
    "pickup": 0.20,
    "place": 0.12,
    "toggle": 0.08,
    "peer_meet": 0.15,
}
CURRICULUM_ALIASES = {"open_cross": "door"}
VALID_LEVELS = range(0, 6)


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
    interaction_distance: float = 1.25
    max_interaction_pose_replans: int = 3
    max_interaction_action_retries: int = 3
    preferred_open_clearance_m: float = 1.0
    interaction_recovery_exclusion_radius_m: float = 0.50
    interaction_recovery_min_position_change_m: float = 0.25
    level: int = 0
    max_setup_attempts_per_episode: int = 12
    simulator_step_timeout_seconds: float = 30.0
    simulator_start_timeout_seconds: float = 60.0
    progress_every_episodes: int = 10
    export_sft: bool = True


@dataclass
class WorkerMetrics:
    worker_id: int
    episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    skipped_setups: int = 0
    primitive_steps: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    reachable_cells: int = 0
    unique_cells: int = 0
    wall_seconds: float = 0.0
    curriculum_attempts: dict[str, int] = field(default_factory=dict)
    curriculum_successes: dict[str, int] = field(default_factory=dict)
    setup_skip_reasons: dict[str, int] = field(default_factory=dict)
    simulator_timeouts: int = 0
    simulator_transport_failures: int = 0
    simulator_restarts: int = 0
    discovery_prop_restores: int = 0

    def note_episode(self, skill: str, success: bool) -> None:
        self.episodes += 1
        self.curriculum_attempts[skill] = self.curriculum_attempts.get(skill, 0) + 1
        if success:
            self.successful_episodes += 1
            self.curriculum_successes[skill] = self.curriculum_successes.get(skill, 0) + 1
        else:
            self.failed_episodes += 1

    def note_skip(self, reason: str) -> None:
        self.skipped_setups += 1
        key = str(reason or "invalid_setup")[:160]
        self.setup_skip_reasons[key] = self.setup_skip_reasons.get(key, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        total_actions = self.successful_actions + self.failed_actions
        data["episode_success_rate_pct"] = round(100.0 * self.successful_episodes / self.episodes, 2) if self.episodes else 0.0
        data["action_success_rate_pct"] = round(100.0 * self.successful_actions / total_actions, 2) if total_actions else 0.0
        data["steps_per_wall_second"] = round(self.primitive_steps / self.wall_seconds, 2) if self.wall_seconds > 0 else 0.0
        data["coverage_pct"] = round(min(100.0, 100.0 * self.unique_cells / self.reachable_cells), 2) if self.reachable_cells else 0.0
        data["curriculum_success_rate_pct"] = {
            skill: round(100.0 * self.curriculum_successes.get(skill, 0) / max(1, attempts), 2)
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
    scored: bool = True
    verification: list[dict[str, Any]] = field(default_factory=list)


def _skip(skill: str, label: str, reason: str, object_type: str | None = None) -> EpisodeResult:
    return EpisodeResult(skill, False, 0, 0, 0, label, object_type=object_type, note=reason, scored=False)


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target.resolve()}")
    value = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    return value


def _selected_event(event: Any, agent_id: int):
    events = getattr(event, "events", None)
    return events[int(agent_id)] if events is not None else event


def _teleport(world: SharedThorWorld, agent_id: int, position: dict[str, float], yaw: float, *, horizon: float = 0.0) -> bool:
    if world.controller is None:
        return False
    event = world.step_for(
        int(agent_id),
        "TeleportFull",
        position={"x": float(position["x"]), "y": float(position.get("y", 0.9)), "z": float(position["z"])},
        rotation={"x": 0.0, "y": float(yaw) % 360.0, "z": 0.0},
        horizon=float(horizon),
        standing=True,
    )
    return event.metadata.get("lastActionSuccess") is not False


def _teleport_pose(world: SharedThorWorld, agent_id: int, pose: InteractionPose) -> bool:
    return _teleport(world, agent_id, pose.position, pose.rotation, horizon=pose.horizon)


def _admin_step(world: SharedThorWorld, agent_id: int, action: str, **params) -> bool:
    if world.controller is None:
        return False
    event = world.step_for(int(agent_id), action, **params)
    return event.metadata.get("lastActionSuccess") is not False



def _set_all_door_state(shared: SharedThorWorld, adapter, agent_id: int, *, open_state: bool, exclude: set[str] | None = None) -> bool:
    """Authoritatively prepare Door state without creating agent skill labels."""
    excluded = exclude or set()
    world = adapter.observe()
    for door in list(world.objects.values()):
        if str(door.object_type).lower() != "door" or not door.openable or door.object_id in excluded:
            continue
        current = door.is_open is True
        if door.is_open is not None and current == bool(open_state):
            continue
        action = "OpenObject" if open_state else "CloseObject"
        if not _admin_step(shared, agent_id, action, objectId=door.object_id, forceAction=True):
            return False
    return True


def _prepare_level_topology(shared: SharedThorWorld, adapter, agent_id: int, level: int, *, selected_door_id: str | None = None) -> tuple[bool, str | None]:
    """Set topology difficulty before target-pose discovery.

    Levels 0-2 isolate the current connected room. Level 3 opens non-target
    connectors for open multi-room navigation. Levels 4-5 are prepared by the
    blocked-route helper after target discovery.
    """
    if level <= 2:
        if not _set_all_door_state(shared, adapter, agent_id, open_state=False):
            return False, "trainer could not close room doors"
    elif level == 3:
        exclude = {selected_door_id} if selected_door_id else set()
        if not _set_all_door_state(shared, adapter, agent_id, open_state=True, exclude=exclude):
            return False, "trainer could not open multi-room connectors"
    return True, None


def _position_key(position: dict[str, float], grid_size: float) -> tuple[int, int]:
    step = float(grid_size)
    return (round(float(position["x"]) / step), round(float(position["z"]) / step))


def _prepare_blocked_route_start(
    shared: SharedThorWorld,
    adapter,
    active: int,
    rng: random.Random,
    target_object_id: str,
    target_pose: InteractionPose,
    full_reachable: list[dict[str, float]],
) -> tuple[bool, str | None, list[dict[str, float]], str | None]:
    """Construct a verified one-door route without random-start rejection sampling.

    v0.10.13 chose a target first and then sampled up to 60 arbitrary starts hoping
    one happened to land on the opposite, actionable side of a closed door. In the
    Discovery world many semantic targets live on the base-room side while Door
    interaction poses are asymmetric, so most outer setup attempts were rejected.

    v0.10.14 instead partitions the all-open topology around each candidate door:
      1. place the trainer at a simulator-returned target interaction pose,
      2. close one door and measure the target-side reachable component,
      3. choose maneuverable cells from the opposite component,
      4. verify the closed door is actionable and the target is inaccessible there,
      5. face the first verified path segment and accept the setup.

    All of this remains trainer-private setup. No geometry is exposed to the semantic
    planner and no rejected candidate is scored against the agent.
    """
    if not _set_all_door_state(shared, adapter, active, open_state=True):
        return False, "trainer could not establish all-open discovery topology", [], None
    world = adapter.observe()
    doors = [
        obj for obj in world.objects.values()
        if str(obj.object_type).lower() == "door" and obj.openable and obj.position is not None
    ]
    if not doors:
        return False, "no Door objects available for blocked-route level", [], None

    grid = float(adapter.config.grid_size)
    full_rows = list(full_reachable)
    full_keys = {_position_key(p, grid) for p in full_rows}
    rng.shuffle(doors)

    for door in doors:
        # Determine which side contains a valid interaction pose for the already-
        # selected semantic target. Teleporting to target_pose is administrative
        # setup and therefore does not teach or score navigation.
        if not _set_all_door_state(shared, adapter, active, open_state=True):
            continue
        if not _teleport_pose(shared, active, target_pose):
            continue
        if not _admin_step(shared, active, "CloseObject", objectId=door.object_id, forceAction=True):
            continue
        try:
            target_side = adapter.get_reachable_positions()
            target_side_poses = adapter.get_interactable_poses(
                target_object_id, positions=target_side, standings=[True]
            )
        except (SimulatorStepTimeout, SimulatorTransportError):
            raise
        except Exception:
            continue
        if not target_side_poses:
            continue

        target_side_keys = {_position_key(p, grid) for p in target_side}
        opposite = [
            p for p in full_rows
            if _position_key(p, grid) in full_keys
            and _position_key(p, grid) not in target_side_keys
            and planar_distance(p, target_pose.position) >= 1.5
        ]
        if not opposite:
            continue

        # Require genuine local maneuvering room on the opposite side. Compute
        # degree inside that opposite component rather than in the all-open graph,
        # so the closed doorway itself cannot make a pocket look safer than it is.
        safe = _maneuverable_starts(opposite, opposite, grid, min_neighbors=2)
        if not safe:
            continue
        rng.shuffle(safe)
        safe.sort(key=lambda p: _cardinal_neighbor_degree(p, opposite, grid), reverse=True)

        for start in safe[:16]:
            # Every candidate begins from a known all-open topology, then closes the
            # same candidate door after placement. This avoids carrying door state
            # from one candidate into the next TeleportFull.
            if not _set_all_door_state(shared, adapter, active, open_state=True):
                continue
            if not _teleport(shared, active, start, 0.0, horizon=0.0):
                continue
            if not _admin_step(shared, active, "CloseObject", objectId=door.object_id, forceAction=True):
                continue
            try:
                reachable = adapter.get_reachable_positions()
                target_poses = adapter.get_interactable_poses(
                    target_object_id, positions=reachable, standings=[True]
                )
                door_poses = adapter.get_interactable_poses(
                    door.object_id, positions=reachable, standings=[True]
                )
            except (SimulatorStepTimeout, SimulatorTransportError):
                raise
            except Exception:
                continue
            if target_poses or not door_poses:
                continue

            nav_cfg = NavigationConfig(
                grid,
                adapter.config.rotate_step_degrees,
                max(0.05, grid * 0.6),
            )
            path_rows: list[tuple[list[dict[str, float]], InteractionPose]] = []
            for door_pose in door_poses:
                try:
                    path = shortest_grid_path(reachable, start, door_pose.position, nav_cfg)
                except Exception:
                    continue
                if path:
                    path_rows.append((path, door_pose))
            if not path_rows:
                continue
            path, door_pose = min(path_rows, key=lambda row: len(row[0]))
            yaw = facing_yaw(path[0], path[1]) if len(path) >= 2 else float(door_pose.rotation)
            if not _teleport(shared, active, start, yaw, horizon=0.0):
                continue
            return True, None, reachable, door.object_id

    _set_all_door_state(shared, adapter, active, open_state=True)
    return False, "target has no verified actionable one-door blocked route", [], None


def _try_blocked_route_for_object(
    shared: SharedThorWorld,
    adapter,
    active: int,
    rng: random.Random,
    obj,
    *,
    max_pose_trials: int = 4,
) -> tuple[list[dict[str, float]], InteractionPose, str] | None:
    """Try several target poses internally; rejection never consumes a raw episode."""
    if not _set_all_door_state(shared, adapter, active, open_state=True):
        return None
    try:
        reachable, poses = _pose_candidates(adapter, obj.object_id)
    except (SimulatorStepTimeout, SimulatorTransportError):
        raise
    except Exception:
        return None
    if not poses:
        return None
    pose_rows = list(poses)
    rng.shuffle(pose_rows)
    for pose in pose_rows[: max(1, int(max_pose_trials))]:
        placed, _reason, component, blocking = _prepare_blocked_route_start(
            shared, adapter, active, rng, obj.object_id, pose, reachable
        )
        if placed and blocking:
            return component, pose, blocking
    return None


def _find_object_with_blocked_route(
    shared: SharedThorWorld,
    adapter,
    active: int,
    rng: random.Random,
    candidates: list[Any],
) -> tuple[Any, list[dict[str, float]], InteractionPose, str] | None:
    """Jointly select semantic target + verified blocked route in one raw setup.

    This is the key v0.10.14 setup-efficiency change: an unrouteable semantic
    candidate is simply tried and discarded inside trainer setup instead of causing
    the worker to spend one of its max_setup_attempts_per_episode attempts.
    """
    rows = list(candidates)
    rng.shuffle(rows)
    for obj in rows:
        setup = _try_blocked_route_for_object(shared, adapter, active, rng, obj)
        if setup is not None:
            reachable, pose, blocking = setup
            return obj, reachable, pose, blocking
    return None

def _place_peer_near_but_not_on_target(shared: SharedThorWorld, peer_id: int, reachable: list[dict[str, float]], target: dict[str, float], active_position: dict[str, float]) -> None:
    """Level-5 nuisance presence: nearby, but never intentionally makes route impossible."""
    candidates = [
        p for p in reachable
        if 0.8 <= planar_distance(p, target) <= 1.8 and planar_distance(p, active_position) >= 0.75
    ]
    candidates.sort(key=lambda p: abs(planar_distance(p, target) - 1.25))
    for point in candidates[:30]:
        if _teleport(shared, peer_id, point, 180.0):
            return
    _park_peer(shared, peer_id, reachable, target)

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
    obj = None if not object_id else world.objects.get(object_id)
    return None if obj is None else obj.object_type


def _append_embodied_transition(path: Path, *, scene: str, agent: str, episode: int, curriculum_skill: str, stage_skill: str, curriculum_level: int | None, action: Action, before, after, success: bool, verification: dict[str, Any], primitive_steps: int, primitive_actions: list[str], execution_telemetry: list[dict[str, Any]] | None = None, auxiliary_skills: list[dict[str, Any]] | None = None) -> None:
    append_jsonl(path, {
        "schema": "author.embodied.v01015",
        "timestamp": utc_now(),
        "run_id": path.parent.parent.name if path.parent.parent else None,
        "worker": path.parent.name,
        "scene": scene,
        "agent": agent,
        "episode": int(episode),
        "curriculum_skill": curriculum_skill,
        "stage_skill": stage_skill,
        "curriculum_level": curriculum_level,
        "action": action.kind,
        "target_object_type": _object_type(before, action.params.get("objectId")),
        # Coordinates remain controller-private training telemetry; they are never
        # serialized into the normal Qwen prompt or SFT conversational messages.
        "before_position": dict(before.pose.position),
        "before_yaw": float(before.pose.rotation.get("y", 0.0)),
        "after_position": dict(after.pose.position),
        "after_yaw": float(after.pose.rotation.get("y", 0.0)),
        "success": bool(success),
        "primitive_steps": int(primitive_steps),
        "primitive_actions": list(primitive_actions),
        "verification": verification,
        "execution_telemetry": list(execution_telemetry or []),
        # Controller-private route evidence. These rows are never placed in the
        # semantic planner prompt; they exist so Level-4/5 qualification is auditable.
        "auxiliary_skills": list(auxiliary_skills or []),
    })


def _execute_stage(*, adapter, action: Action, verifier: Verifier, replay_path: Path, scene: str, agent_label: str, episode: int, curriculum_skill: str, stage_skill: str, library: EmbodiedSkillLibrary, object_type: str | None, max_steps: int) -> tuple[bool, int, list[str], dict[str, Any]]:
    before = adapter.observe()
    try:
        execution = adapter.execute(action, step_budget=max_steps)
        after = execution.state
        primitives = list(execution.primitive_actions or ([action.kind] if execution.primitive_steps else []))
        skill_auxiliary = [
            dict(row) for row in (execution.auxiliary_skills or [])
            if str(row.get("skill") or "").strip()
        ]
        for aux in skill_auxiliary:
            aux_skill = str(aux.get("skill") or "").strip()
            library.record(
                aux_skill,
                success=bool(aux.get("success")),
                primitive_steps=int(aux.get("primitive_steps", 0)),
                object_type=aux.get("object_type"),
                recovery=bool(aux.get("recovery", False)),
            )
        telemetry = [row for row in (execution.auxiliary_skills or []) if not str(row.get("skill") or "").strip()]
        recovery_summary = next((row for row in reversed(telemetry) if row.get("kind") == "interaction_recovery_summary"), {})
        recovery_used = bool(recovery_summary.get("recovery_attempted")) or any(bool(row.get("recovery")) for row in telemetry)
        verified, verification = verifier.verify(before, after, action, execution=execution)
        verification = dict(verification or {})
        verification["last_action_success"] = after.last_action_success
        verification["last_action_error"] = after.last_error
        verification["interaction_recovery_used"] = recovery_used
        if telemetry:
            verification["interaction_pose_attempts"] = telemetry
        success = bool(after.last_action_success is not False and verified)

        route_rows = [row for row in skill_auxiliary if str(row.get("skill") or "") in {"door_approach", "door_open", "door_cross"}]
        route_names = {str(row.get("skill") or "") for row in route_rows}
        route_required = bool(
            int(library.training_level or 0) >= 4
            and curriculum_skill in {"pickup", "place", "toggle"}
            and action.kind == ActionKind.NAVIGATE_TO.value
        )
        route_resume_verified = bool(
            success
            and {"door_approach", "door_open", "door_cross"}.issubset(route_names)
            and all(bool(row.get("success")) for row in route_rows)
        )
        verification["semantic_route_required"] = route_required
        verification["semantic_target_preserved"] = route_required
        verification["route_auxiliary_skills"] = route_rows
        verification["route_resume_target_verified"] = route_resume_verified
        if route_required:
            # Level-4/5 only counts the target approach as successful when the motor
            # layer used and successfully completed the closed-door infrastructure
            # route, then still reached the original non-door target pose.
            success = bool(success and route_resume_verified)
            library.record(
                "route_resume_target",
                success=route_resume_verified,
                primitive_steps=int(execution.primitive_steps),
                object_type=object_type,
            )

        recovery_success = bool(recovery_used and success)
        recovery_exhausted = bool(recovery_used and not success)
        verification["interaction_recovery_success"] = recovery_success
        verification["interaction_recovery_exhausted"] = recovery_exhausted
        steps = int(execution.primitive_steps)
    except (SimulatorStepTimeout, SimulatorTransportError):
        raise
    except Exception as exc:
        after = adapter.observe()
        primitives = []
        steps = 0
        success = False
        telemetry = []
        skill_auxiliary = []
        recovery_used = False
        recovery_success = False
        recovery_exhausted = False
        verification = {"exception": f"{type(exc).__name__}: {exc}"}

    _append_embodied_transition(
        replay_path,
        scene=scene,
        agent=agent_label,
        episode=episode,
        curriculum_skill=curriculum_skill,
        stage_skill=stage_skill,
        curriculum_level=library.training_level,
        action=action,
        before=before,
        after=after,
        success=success,
        verification=verification,
        primitive_steps=steps,
        primitive_actions=primitives,
        execution_telemetry=telemetry,
        auxiliary_skills=skill_auxiliary,
    )
    library.record(stage_skill, success=success, primitive_steps=steps, object_type=object_type, recovery=recovery_used, recovery_success=recovery_success, recovery_exhausted=recovery_exhausted)
    return success, steps, primitives, verification


def _candidate_objects(world, predicate) -> list[Any]:
    return [obj for obj in world.objects.values() if obj.position is not None and predicate(obj)]


def _pose_candidates(adapter, object_id: str) -> tuple[list[dict[str, float]], list[InteractionPose]]:
    reachable = adapter.get_reachable_positions()
    if not reachable:
        return [], []
    poses = adapter.get_interactable_poses(object_id, positions=reachable, standings=[True])
    return reachable, poses


def _choose_setup_pose(rng: random.Random, poses: list[InteractionPose]) -> InteractionPose:
    # Diversity matters in the trainer; runtime itself chooses the cheapest connected pose.
    return rng.choice(poses)


def _navigation_pair_for_level(rng: random.Random, reachable: list[dict[str, float]], level: int, grid_size: float) -> tuple[dict[str, float], dict[str, float]] | None:
    if len(reachable) < 2:
        return None
    if level <= 0:
        low, high = grid_size * 0.9, grid_size * 2.1
    elif level == 1:
        low, high = 0.45, 1.10
    elif level == 2:
        low, high = 1.0, 3.0
    else:
        low, high = 1.5, float("inf")
    rows = list(reachable)
    for _ in range(min(500, max(50, len(rows) * 3))):
        a, b = rng.sample(rows, 2)
        distance = planar_distance(a, b)
        if low <= distance <= high:
            return dict(a), dict(b)
    # Never fabricate a hard episode if the current component is too small.
    return None




def _cardinal_neighbor_degree(position: dict[str, float], reachable: Iterable[dict[str, float]], grid_size: float) -> int:
    step = float(grid_size)
    keys = {(round(float(p["x"]) / step), round(float(p["z"]) / step)) for p in reachable}
    key = (round(float(position["x"]) / step), round(float(position["z"]) / step))
    x, z = key
    return sum((nx, nz) in keys for nx, nz in ((x+1,z),(x-1,z),(x,z+1),(x,z-1)))


def _maneuverable_starts(candidates: list[dict[str, float]], reachable: list[dict[str, float]], grid_size: float, *, min_neighbors: int = 2) -> list[dict[str, float]]:
    return [p for p in candidates if _cardinal_neighbor_degree(p, reachable, grid_size) >= int(min_neighbors)]


def _start_candidates_for_level(level: int, reachable: Iterable[dict[str, float]], pose: InteractionPose, grid_size: float) -> list[dict[str, float]]:
    rows = list(reachable)
    if level <= 0:
        return []
    if level == 1:
        return [p for p in rows if 0.45 <= planar_distance(p, pose.position) <= 1.05]
    if level == 2:
        return [p for p in rows if 1.0 <= planar_distance(p, pose.position) <= 3.0]
    # Levels 3/4 use the entire current connected component but avoid trivial starts.
    if level in (3, 4):
        return [p for p in rows if planar_distance(p, pose.position) >= max(1.25, grid_size * 3.0)]
    return rows


def _prepare_agent_start(shared: SharedThorWorld, adapter, active: int, rng: random.Random, level: int, pose: InteractionPose, reachable: list[dict[str, float]]) -> tuple[bool, str | None]:
    if level == 0:
        return _teleport_pose(shared, active, pose), None
    candidates = _start_candidates_for_level(level, reachable, pose, adapter.config.grid_size)
    if level >= 3:
        safe = _maneuverable_starts(candidates, reachable, adapter.config.grid_size, min_neighbors=2)
        if safe:
            candidates = safe
    if not candidates:
        return False, f"no level-{level} start positions"
    # Verify a path before teleporting.  Levels >=3 also face the first path
    # segment so a valid lesson never begins with an avoidable corner rotation.
    rng.shuffle(candidates)
    if level >= 3:
        candidates.sort(key=lambda p: _cardinal_neighbor_degree(p, reachable, adapter.config.grid_size), reverse=True)
    cfg = NavigationConfig(adapter.config.grid_size, adapter.config.rotate_step_degrees, max(0.05, adapter.config.grid_size * 0.6))
    for start in candidates[:100]:
        try:
            path = shortest_grid_path(reachable, start, pose.position, cfg)
        except ValueError:
            continue
        if level >= 3 and len(path) > 1:
            yaw = facing_yaw(path[0], path[1])
        else:
            yaw = rng.choice((0.0, 90.0, 180.0, 270.0))
        if _teleport(shared, active, start, yaw, horizon=0.0):
            return True, None
    return False, f"could not place agent at connected level-{level} start"


def _park_peer(shared: SharedThorWorld, peer_id: int, reachable: list[dict[str, float]], avoid: dict[str, float]) -> None:
    if not reachable:
        return
    candidates = sorted(reachable, key=lambda p: planar_distance(p, avoid), reverse=True)
    for point in candidates[:40]:
        if planar_distance(point, avoid) < 2.0:
            break
        if _teleport(shared, peer_id, point, 180.0):
            return


def _restore_object_point(shared: SharedThorWorld, agent_id: int, object_id: str, position: dict[str, float]) -> None:
    # Trainer infrastructure only. Failure is intentionally not scored as agent behavior.
    _admin_step(shared, agent_id, "PlaceObjectAtPoint", objectId=object_id, position=dict(position))


def _capture_discovery_prop_positions(adapter) -> dict[str, dict[str, float]]:
    """Capture canonical trainer positions for generated movable Discovery props.

    The Unity scene intentionally contains pickupable semantic props. They can be
    nudged or knocked from furniture during long multi-episode runs, and AI2-THOR
    then correctly treats them as physical blockers. Curriculum setup should not
    accumulate that incidental debris state across independent lessons.
    """
    baseline: dict[str, dict[str, float]] = {}
    observe = getattr(adapter, "observe", None)
    if not callable(observe):
        return baseline
    world = observe()
    for obj in getattr(world, "objects", {}).values():
        if obj.position is None or not str(obj.name or "").startswith("Discovery_"):
            continue
        baseline[obj.object_id] = dict(obj.position)
    return baseline


def _restore_discovery_props(
    shared: SharedThorWorld,
    adapters: dict[int, Any],
    agent_id: int,
    baseline: dict[str, dict[str, float]],
    *,
    tolerance_m: float = 0.12,
) -> tuple[bool, str | None, int]:
    """Restore displaced generated props before a curriculum lesson.

    Only props whose centers moved materially are touched, minimizing simulator
    traffic. Objects currently held by either agent are skipped until the normal
    held-object cleanup releases them. This is trainer-only scene hygiene.
    """
    if not baseline:
        return True, None, 0
    held = {a.observe().held_object_id for a in adapters.values()}
    world = adapters[agent_id].observe()
    restored = 0
    for object_id, home in baseline.items():
        if object_id in held:
            continue
        obj = world.objects.get(object_id)
        if obj is None or obj.position is None:
            continue
        dx = float(obj.position.get("x", 0.0)) - float(home.get("x", 0.0))
        dy = float(obj.position.get("y", 0.0)) - float(home.get("y", 0.0))
        dz = float(obj.position.get("z", 0.0)) - float(home.get("z", 0.0))
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= float(tolerance_m):
            continue
        if not _admin_step(shared, agent_id, "PlaceObjectAtPoint", objectId=object_id, position=dict(home)):
            return False, f"trainer could not restore displaced Discovery prop {object_id}", restored
        restored += 1
        world = adapters[agent_id].observe()
    return True, None, restored


def _distance_to_path(point: dict[str, float], path: list[dict[str, float]]) -> float:
    if not path:
        return float("inf")
    return min(planar_distance(point, row) for row in path)


def _park_peer_off_level4_route(
    shared: SharedThorWorld,
    adapter,
    active: int,
    peer_id: int,
    reachable: list[dict[str, float]],
    blocking_door_id: str,
) -> bool:
    """Park the passive Level-4 peer away from the verified approach corridor.

    Peer interference belongs to Level 5. Level 4 isolates semantic door-route
    continuation, so the inactive agent must remain on the starting side while
    staying clear of both the active start and the shortest door-approach path.
    """
    if not reachable:
        return False
    world = adapter.observe()
    door = world.objects.get(blocking_door_id)
    if door is None or door.position is None:
        return False
    start = dict(world.pose.position)
    try:
        door_poses = adapter.get_interactable_poses(blocking_door_id, positions=reachable, standings=[True])
    except (SimulatorStepTimeout, SimulatorTransportError):
        raise
    except Exception:
        door_poses = []
    nav_cfg = NavigationConfig(
        adapter.config.grid_size,
        adapter.config.rotate_step_degrees,
        max(0.05, adapter.config.grid_size * 0.6),
    )
    paths: list[list[dict[str, float]]] = []
    for pose in door_poses:
        try:
            path = shortest_grid_path(reachable, start, pose.position, nav_cfg)
        except Exception:
            continue
        if path:
            paths.append(path)
    route = min(paths, key=len) if paths else [start, dict(door.position)]
    safe = _maneuverable_starts(reachable, reachable, adapter.config.grid_size, min_neighbors=2) or list(reachable)
    scored: list[tuple[float, dict[str, float]]] = []
    for point in safe:
        d_route = _distance_to_path(point, route)
        d_start = planar_distance(point, start)
        d_door = planar_distance(point, door.position)
        if d_route < 1.0 or d_start < 1.25 or d_door < 1.25:
            continue
        score = min(d_route, d_start, d_door) + 0.20 * d_route
        scored.append((score, point))
    scored.sort(key=lambda row: row[0], reverse=True)
    for _score, point in scored[:40]:
        if _teleport(shared, peer_id, point, 180.0):
            return True
    return False


def _ensure_empty_hand(adapter) -> bool:
    if not adapter.observe().held_object_id:
        return True
    try:
        adapter.execute(Action(ActionKind.DROP.value, {"forceAction": True}, "trainer cleanup"))
    except (SimulatorStepTimeout, SimulatorTransportError):
        raise
    except Exception:
        return False
    return adapter.observe().held_object_id is None


def _run_route(*, adapter, agent_label: str, episode: int, target: dict[str, float], reachable: list[dict[str, float]], nav_cfg: NavigationConfig, world_map: PersistentWorldMap, skill: NavigationSkill, replay: ReplayBuffer, arrival_distance: float, max_steps: int, embodied_library: EmbodiedSkillLibrary) -> EpisodeResult:
    start_world = adapter.observe()
    world_map.record_pose(start_world.pose.position)
    try:
        path = shortest_grid_path(reachable, start_world.pose.position, target, NavigationConfig(nav_cfg.grid_size, nav_cfg.rotate_step_degrees, float(arrival_distance)))
    except ValueError as exc:
        return _skip("navigate", agent_label, f"invalid point-navigation setup: {exc}")
    actions = path_to_actions(path, initial_yaw=float(start_world.pose.rotation.get("y", 0.0)), target_position=target, config=nav_cfg)
    steps = successes = failures = 0
    sequence: list[str] = []
    for primitive in actions:
        if steps >= max_steps:
            break
        before = adapter.observe()
        state = skill.state(before.pose.position, float(before.pose.rotation.get("y", 0.0)), target, grid_size=nav_cfg.grid_size)
        execution = adapter.execute(primitive)
        after = execution.state
        sequence.extend(execution.primitive_actions or [primitive.kind])
        used = max(1, int(execution.primitive_steps))
        steps += used
        ok = after.last_action_success is not False
        reached = planar_distance(after.pose.position, target) <= arrival_distance + 1e-6
        before_d = planar_distance(before.pose.position, target)
        after_d = planar_distance(after.pose.position, target)
        reward = round((before_d - after_d) * 2.0 + (0.08 if ok else -1.0) + (2.0 if reached else 0.0), 6)
        next_state = skill.state(after.pose.position, float(after.pose.rotation.get("y", 0.0)), target, grid_size=nav_cfg.grid_size)
        skill.update(state, primitive.kind, reward, next_state)
        world_map.record_transition(before.pose.position, after.pose.position, success=ok)
        replay.add(ReplayTransition(scene=str(after.scene), agent=agent_label, episode=int(episode), action=primitive.kind, before_position=dict(before.pose.position), before_yaw=float(before.pose.rotation.get("y", 0.0)), after_position=dict(after.pose.position), after_yaw=float(after.pose.rotation.get("y", 0.0)), target_position=dict(target), success=bool(ok), reward=float(reward), peer_training=False))
        successes += int(ok)
        failures += int(not ok)
        if not ok:
            break
    reached = planar_distance(adapter.observe().pose.position, target) <= arrival_distance + 1e-6
    embodied_library.record("point_navigation", success=reached, primitive_steps=steps)
    return EpisodeResult("navigate", reached, steps, successes, failures, agent_label, sequence, verification=[{"kind": "point_radius", "success": reached}])


def _approach_params(adapter, object_id: str, interaction_kind: str, level: int, pose: InteractionPose) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objectId": object_id,
        "interactionDistance": adapter.config.interaction_distance,
        "_interactionKind": interaction_kind,
    }
    # Levels 1-3 are staged around one exact simulator-returned pose. Pinning that
    # controller-private pose prevents the runtime selector from silently turning a
    # near/same-room lesson into navigation toward some unrelated preferred pose.
    if 1 <= int(level) <= 3:
        params["_targetInteractionPose"] = pose.as_dict()
    return params


def _train_door(*, shared: SharedThorWorld, adapters: dict[int, Any], active: int, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, level: int, max_steps: int) -> EpisodeResult:
    adapter = adapters[active]
    world = adapter.observe()
    doors = _candidate_objects(world, lambda obj: obj.openable and obj.object_type.lower() == "door")
    if not doors:
        return _skip("door", agent_label, "no Door objects")
    rng.shuffle(doors)
    door = doors[0]

    # Authoritative trainer reset: preparation is never learned/scored.
    topo_ok, topo_reason = _prepare_level_topology(shared, adapter, active, level, selected_door_id=door.object_id)
    if not topo_ok:
        return _skip("door", agent_label, topo_reason or "trainer topology preparation failed", "Door")
    current_door = adapter.observe().objects.get(door.object_id)
    if current_door is None:
        return _skip("door", agent_label, "selected Door disappeared during setup", "Door")
    if current_door.is_open is not False and not _admin_step(shared, active, "CloseObject", objectId=door.object_id, forceAction=True):
        return _skip("door", agent_label, "trainer could not close selected door", "Door")
    reachable, poses = _pose_candidates(adapter, door.object_id)
    if not poses:
        return _skip("door", agent_label, "closed door has no GetInteractablePoses pose in current component", "Door")
    pose = _choose_setup_pose(rng, poses)
    placed, reason = _prepare_agent_start(shared, adapter, active, rng, level, pose, reachable)
    if placed:
        if level >= 5:
            _place_peer_near_but_not_on_target(shared, 1 - active, reachable, pose.position, adapter.observe().pose.position)
        else:
            _park_peer(shared, 1 - active, reachable, pose.position)
    if not placed:
        return _skip("door", agent_label, reason or "trainer placement failed", "Door")

    total = good = bad = 0
    seq: list[str] = []
    checks: list[dict[str, Any]] = []
    if level > 0:
        ok, steps, primitive, verify = _execute_stage(
            adapter=adapter,
            action=Action(ActionKind.NAVIGATE_TO.value, _approach_params(adapter, door.object_id, ActionKind.OPEN.value, level, pose), "actionability-aware door approach"),
            verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label,
            episode=episode, curriculum_skill="door", stage_skill="door_approach", library=library,
            object_type="Door", max_steps=max_steps,
        )
        total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
        if not ok:
            return EpisodeResult("door", False, total, good, bad, agent_label, seq, "Door", verification=checks)

    remaining = max(1, max_steps - total)
    ok, steps, primitive, verify = _execute_stage(
        adapter=adapter,
        action=Action(ActionKind.OPEN.value, {"objectId": door.object_id, "forceAction": False}, "verified door opening"),
        verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label,
        episode=episode, curriculum_skill="door", stage_skill="door_open", library=library,
        object_type="Door", max_steps=remaining,
    )
    total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
    if not ok or level == 0:
        return EpisodeResult("door", ok, total, good, bad, agent_label, seq, "Door", verification=checks)

    # CrossThreshold re-queries GetReachablePositions after the door opens.
    remaining = max(1, max_steps - total)
    ok, steps, primitive, verify = _execute_stage(
        adapter=adapter,
        action=Action(ActionKind.CROSS_THRESHOLD.value, {"objectId": door.object_id}, "verified cross through selected open door"),
        verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label,
        episode=episode, curriculum_skill="door", stage_skill="door_cross", library=library,
        object_type="Door", max_steps=remaining,
    )
    total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
    return EpisodeResult("door", ok, total, good, bad, agent_label, seq, "Door", verification=checks)


def _find_object_with_poses(adapter, candidates: list[Any], rng: random.Random) -> tuple[Any, list[dict[str, float]], list[InteractionPose]] | None:
    rng.shuffle(candidates)
    for obj in candidates:
        try:
            reachable, poses = _pose_candidates(adapter, obj.object_id)
        except (SimulatorStepTimeout, SimulatorTransportError):
            raise
        except Exception:
            continue
        if poses:
            return obj, reachable, poses
    return None


def _train_pickup(*, shared: SharedThorWorld, adapters: dict[int, Any], active: int, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, level: int, max_steps: int) -> EpisodeResult:
    adapter = adapters[active]
    if not _ensure_empty_hand(adapter):
        return _skip("pickup", agent_label, "trainer could not clear held object")
    if level >= 4:
        if not _set_all_door_state(shared, adapter, active, open_state=True):
            return _skip("pickup", agent_label, "trainer could not establish all-open target discovery topology")
    else:
        topo_ok, topo_reason = _prepare_level_topology(shared, adapter, active, level)
        if not topo_ok:
            return _skip("pickup", agent_label, topo_reason or "trainer topology preparation failed")
    world = adapter.observe()
    pickup_candidates = _candidate_objects(world, lambda obj: obj.pickupable and not obj.is_picked_up)
    blocking_door_id = None
    if level >= 4:
        blocked = _find_object_with_blocked_route(shared, adapter, active, rng, pickup_candidates)
        if not blocked:
            return _skip("pickup", agent_label, "no pickupable target with a verified actionable one-door blocked route")
        obj, reachable, pose, blocking_door_id = blocked
        placed, reason = True, None
    else:
        found = _find_object_with_poses(adapter, pickup_candidates, rng)
        if not found:
            return _skip("pickup", agent_label, "no pickupable target with a connected interactable pose")
        obj, reachable, poses = found
        pose = _choose_setup_pose(rng, poses)
        placed, reason = _prepare_agent_start(shared, adapter, active, rng, level, pose, reachable)
    original = dict(obj.position)
    if not placed:
        return _skip("pickup", agent_label, reason or "trainer placement failed", obj.object_type)
    if level >= 5 and blocking_door_id:
        blocking = adapter.observe().objects.get(blocking_door_id)
        nuisance_target = blocking.position if blocking and blocking.position else pose.position
        _place_peer_near_but_not_on_target(shared, 1 - active, reachable, nuisance_target, adapter.observe().pose.position)
    elif level >= 4 and blocking_door_id:
        if not _park_peer_off_level4_route(shared, adapter, active, 1 - active, reachable, blocking_door_id):
            return _skip("pickup", agent_label, "trainer could not park passive peer clear of verified Level-4 route", obj.object_type)
    else:
        _park_peer(shared, 1 - active, reachable, pose.position)

    total = good = bad = 0; seq: list[str] = []; checks: list[dict[str, Any]] = []
    if level > 0:
        ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(ActionKind.NAVIGATE_TO.value, _approach_params(adapter, obj.object_id, ActionKind.PICKUP.value, level, pose), "actionability-aware pickup approach"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="pickup", stage_skill="pickup_approach", library=library, object_type=obj.object_type, max_steps=max_steps)
        total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
        if not ok:
            return EpisodeResult("pickup", False, total, good, bad, agent_label, seq, obj.object_type, verification=checks)
    ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(ActionKind.PICKUP.value, {"objectId": obj.object_id, "forceAction": False}, "verified pickup"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="pickup", stage_skill="pickup_grasp", library=library, object_type=obj.object_type, max_steps=max(1, max_steps-total))
    total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
    if ok:
        _ensure_empty_hand(adapter)
        _restore_object_point(shared, active, obj.object_id, original)
    return EpisodeResult("pickup", ok, total, good, bad, agent_label, seq, obj.object_type, verification=checks)


def _train_place(*, shared: SharedThorWorld, adapters: dict[int, Any], active: int, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, level: int, max_steps: int) -> EpisodeResult:
    adapter = adapters[active]
    if not _ensure_empty_hand(adapter):
        return _skip("place", agent_label, "trainer could not clear held object")
    if level >= 4:
        if not _set_all_door_state(shared, adapter, active, open_state=True):
            return _skip("place", agent_label, "trainer could not establish all-open target discovery topology")
    else:
        topo_ok, topo_reason = _prepare_level_topology(shared, adapter, active, level)
        if not topo_ok:
            return _skip("place", agent_label, topo_reason or "trainer topology preparation failed")
    world = adapter.observe()
    sources = [obj for obj in _candidate_objects(world, lambda o: o.pickupable and not o.is_picked_up) if obj.parent_receptacles]
    rng.shuffle(sources)
    selected = None
    blocking_door_id = None
    if level >= 4:
        # Jointly choose source/receptacle and a verified one-door route to the
        # receptacle. An unrouteable pair is rejected inside this raw setup.
        for source in sources:
            target_ids = [pid for pid in source.parent_receptacles if pid in world.objects and world.objects[pid].receptacle]
            rng.shuffle(target_ids)
            for target_id in target_ids:
                target = world.objects[target_id]
                setup = _try_blocked_route_for_object(shared, adapter, active, rng, target)
                if setup is not None:
                    reachable, pose, blocking_door_id = setup
                    selected = (source, target, reachable, pose)
                    break
            if selected:
                break
    else:
        for source in sources:
            target_ids = [pid for pid in source.parent_receptacles if pid in world.objects and world.objects[pid].receptacle]
            rng.shuffle(target_ids)
            for target_id in target_ids:
                try:
                    reachable, poses = _pose_candidates(adapter, target_id)
                except (SimulatorStepTimeout, SimulatorTransportError):
                    raise
                except Exception:
                    continue
                if poses:
                    selected = (source, world.objects[target_id], reachable, _choose_setup_pose(rng, poses))
                    break
            if selected:
                break
    if not selected:
        reason = "no source/receptacle pair with a verified actionable one-door blocked route" if level >= 4 else "no source/receptacle pair with connected target pose"
        return _skip("place", agent_label, reason)
    source, target, reachable, pose = selected
    original = dict(source.position)
    # Administrative setup: holding the source is a precondition, not a scored pickup.
    if not _admin_step(shared, active, "PickupObject", objectId=source.object_id, forceAction=True):
        return _skip("place", agent_label, "trainer could not prepare held source", target.object_type)
    if level >= 4:
        placed, reason = True, None
    else:
        placed, reason = _prepare_agent_start(shared, adapter, active, rng, level, pose, reachable)
    if not placed:
        _ensure_empty_hand(adapter); _restore_object_point(shared, active, source.object_id, original)
        return _skip("place", agent_label, reason or "trainer placement failed", target.object_type)
    if level >= 5 and blocking_door_id:
        blocking = adapter.observe().objects.get(blocking_door_id)
        nuisance_target = blocking.position if blocking and blocking.position else pose.position
        _place_peer_near_but_not_on_target(shared, 1 - active, reachable, nuisance_target, adapter.observe().pose.position)
    elif level >= 4 and blocking_door_id:
        if not _park_peer_off_level4_route(shared, adapter, active, 1 - active, reachable, blocking_door_id):
            _ensure_empty_hand(adapter); _restore_object_point(shared, active, source.object_id, original)
            return _skip("place", agent_label, "trainer could not park passive peer clear of verified Level-4 route", target.object_type)
    else:
        _park_peer(shared, 1 - active, reachable, pose.position)

    total = good = bad = 0; seq: list[str] = []; checks: list[dict[str, Any]] = []
    if level > 0:
        ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(ActionKind.NAVIGATE_TO.value, _approach_params(adapter, target.object_id, ActionKind.PUT.value, level, pose), "actionability-aware placement approach"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="place", stage_skill="place_approach", library=library, object_type=target.object_type, max_steps=max_steps)
        total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
        if not ok:
            _ensure_empty_hand(adapter); _restore_object_point(shared, active, source.object_id, original)
            return EpisodeResult("place", False, total, good, bad, agent_label, seq, target.object_type, verification=checks)
    ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(ActionKind.PUT.value, {"objectId": target.object_id, "forceAction": False, "placeStationary": True}, "verified placement"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="place", stage_skill="place_release", library=library, object_type=target.object_type, max_steps=max(1, max_steps-total))
    total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
    if not ok:
        _ensure_empty_hand(adapter)
    _restore_object_point(shared, active, source.object_id, original)
    return EpisodeResult("place", ok, total, good, bad, agent_label, seq, target.object_type, verification=checks)


def _train_toggle(*, shared: SharedThorWorld, adapters: dict[int, Any], active: int, agent_label: str, episode: int, rng: random.Random, replay_path: Path, library: EmbodiedSkillLibrary, verifier: Verifier, level: int, max_steps: int) -> EpisodeResult:
    adapter = adapters[active]
    if level >= 4:
        if not _set_all_door_state(shared, adapter, active, open_state=True):
            return _skip("toggle", agent_label, "trainer could not establish all-open target discovery topology")
    else:
        topo_ok, topo_reason = _prepare_level_topology(shared, adapter, active, level)
        if not topo_ok:
            return _skip("toggle", agent_label, topo_reason or "trainer topology preparation failed")
    world = adapter.observe()
    toggle_candidates = _candidate_objects(world, lambda obj: obj.toggleable and obj.is_toggled is not None)
    blocking_door_id = None
    if level >= 4:
        blocked = _find_object_with_blocked_route(shared, adapter, active, rng, toggle_candidates)
        if not blocked:
            return _skip("toggle", agent_label, "no toggleable target with a verified actionable one-door blocked route")
        obj, reachable, pose, blocking_door_id = blocked
        placed, reason = True, None
    else:
        found = _find_object_with_poses(adapter, toggle_candidates, rng)
        if not found:
            return _skip("toggle", agent_label, "no toggleable target with connected pose")
        obj, reachable, poses = found
        pose = _choose_setup_pose(rng, poses)
        placed, reason = _prepare_agent_start(shared, adapter, active, rng, level, pose, reachable)
    initial = bool(obj.is_toggled)
    kind = ActionKind.TOGGLE_OFF.value if initial else ActionKind.TOGGLE_ON.value
    if not placed:
        return _skip("toggle", agent_label, reason or "trainer placement failed", obj.object_type)
    if level >= 5 and blocking_door_id:
        blocking = adapter.observe().objects.get(blocking_door_id)
        nuisance_target = blocking.position if blocking and blocking.position else pose.position
        _place_peer_near_but_not_on_target(shared, 1-active, reachable, nuisance_target, adapter.observe().pose.position)
    elif level >= 4 and blocking_door_id:
        if not _park_peer_off_level4_route(shared, adapter, active, 1-active, reachable, blocking_door_id):
            return _skip("toggle", agent_label, "trainer could not park passive peer clear of verified Level-4 route", obj.object_type)
    else:
        _park_peer(shared, 1-active, reachable, pose.position)
    total = good = bad = 0; seq: list[str] = []; checks: list[dict[str, Any]] = []
    if level > 0:
        ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(ActionKind.NAVIGATE_TO.value, _approach_params(adapter, obj.object_id, kind, level, pose), "actionability-aware toggle approach"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="toggle", stage_skill="toggle_approach", library=library, object_type=obj.object_type, max_steps=max_steps)
        total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
        if not ok:
            return EpisodeResult("toggle", False, total, good, bad, agent_label, seq, obj.object_type, verification=checks)
    ok, steps, primitive, verify = _execute_stage(adapter=adapter, action=Action(kind, {"objectId": obj.object_id, "forceAction": False}, "verified toggle"), verifier=verifier, replay_path=replay_path, scene=str(world.scene), agent_label=agent_label, episode=episode, curriculum_skill="toggle", stage_skill="toggle_interaction", library=library, object_type=obj.object_type, max_steps=max(1, max_steps-total))
    total += steps; good += int(ok); bad += int(not ok); seq += primitive; checks.append(verify)
    reverse = ActionKind.TOGGLE_ON.value if initial else ActionKind.TOGGLE_OFF.value
    _admin_step(shared, active, reverse, objectId=obj.object_id, forceAction=True)
    return EpisodeResult("toggle", ok, total, good, bad, agent_label, seq, obj.object_type, verification=checks)


def _train_peer_meet(*, shared: SharedThorWorld, adapters: dict[int, Any], active: int, agent_label: str, episode: int, rng: random.Random, library: EmbodiedSkillLibrary, max_steps: int, peer_distance: float = 1.25) -> EpisodeResult:
    adapter = adapters[active]
    reachable = adapter.get_reachable_positions()
    if len(reachable) < 2:
        return _skip("peer_meet", agent_label, "fewer than two reachable positions", "Peer")
    starts = rng.sample(reachable, min(2, len(reachable)))
    if not _teleport(shared, active, starts[0], rng.choice((0.0,90.0,180.0,270.0))):
        return _skip("peer_meet", agent_label, "could not place active agent", "Peer")
    peer = 1-active
    peer_candidates = sorted(reachable, key=lambda p: planar_distance(p, starts[0]), reverse=True)
    peer_pos = next((p for p in peer_candidates if planar_distance(p, starts[0]) >= 1.5), peer_candidates[0])
    if not _teleport(shared, peer, peer_pos, 180.0):
        return _skip("peer_meet", agent_label, "could not place peer", "Peer")
    target = dict(adapters[peer].observe().pose.position)
    reachable = adapter.get_reachable_positions()
    cfg = NavigationConfig(adapter.config.grid_size, adapter.config.rotate_step_degrees, peer_distance)
    try:
        path = shortest_grid_path(reachable, adapter.observe().pose.position, target, cfg)
    except ValueError as exc:
        return _skip("peer_meet", agent_label, f"peer path invalid: {exc}", "Peer")
    actions = path_to_actions(path, initial_yaw=float(adapter.observe().pose.rotation.get("y",0.0)), target_position=target, config=cfg)
    steps=good=bad=0; seq=[]
    for action in actions:
        if steps >= max_steps: break
        execution=adapter.execute(action)
        used=max(1,int(execution.primitive_steps)); steps+=used
        seq += list(execution.primitive_actions or [action.kind])
        ok=execution.state.last_action_success is not False
        good+=int(ok); bad+=int(not ok)
        if not ok: break
    actual=planar_distance(adapter.observe().pose.position, adapters[peer].observe().pose.position)
    success=actual <= peer_distance + 0.15
    library.record("peer_approach", success=success, primitive_steps=steps, object_type="Peer")
    return EpisodeResult("peer_meet", success, steps, good, bad, agent_label, seq, "Peer", verification=[{"kind":"peer_distance","distance_m":round(actual,3),"success":success}])


def _append_sft(path: Path, result: EpisodeResult, scene: str, level: int) -> None:
    # Only successful, authoritative trajectories are eligible. Invalid setup and
    # failures remain in replay/metrics but never become positive demonstrations.
    if not result.scored or not result.success or not result.sequence:
        return
    append_jsonl(path, {
        "schema": "author.embodied_sft.v01015",
        "messages": [
            {"role": "system", "content": "A semantic goal and target have already been selected. Execute only the embodied motor skill; do not invent another target."},
            {"role": "user", "content": f"Scene={scene}. Curriculum={result.skill}. Level={level}. Selected target type={result.object_type or 'point'}."},
            {"role": "assistant", "content": " -> ".join(result.sequence)},
        ],
        "metadata": {
            "agent": result.agent_label,
            "skill": result.skill,
            "level": int(level),
            "target_object_type": result.object_type,
            "primitive_steps": result.primitive_steps,
            "verified": True,
            "source": "AUTHOR v0.10.15 route-safe semantic-route teacher trajectory",
        },
    })


def _worker_main(spec: WorkerSpec) -> dict[str, Any]:
    started=time.perf_counter(); rng=random.Random(spec.seed)
    root=Path(spec.worker_root); root.mkdir(parents=True,exist_ok=True)
    map_path=root/"map.json"; nav_path=root/"navigation_skill.json"; nav_replay_path=root/"navigation_replay.jsonl"
    embodied_replay=root/"embodied_replay.jsonl"; sft_path=root/"embodied_sft.jsonl"
    skill_paths={"A":root/"skills_A.json","B":root/"skills_B.json"}
    world_map=PersistentWorldMap(spec.scene,spec.grid_size,map_path); nav_skill=NavigationSkill(nav_path)
    nav_replay=ReplayBuffer(nav_replay_path,max_records=spec.replay_max_records)
    libraries={label:EmbodiedSkillLibrary(path,agent_label=label) for label,path in skill_paths.items()}
    for library in libraries.values():
        library.training_level = spec.level
    metrics=WorkerMetrics(worker_id=spec.worker_id); verifier=Verifier(default_interaction_distance=spec.visibility_distance)
    thor_cfg=ThorConfig(scene=spec.scene,grid_size=spec.grid_size,rotate_step_degrees=spec.rotate_step_degrees,visibility_distance=spec.visibility_distance,interaction_distance=spec.interaction_distance,width=spec.width,height=spec.height,field_of_view=spec.field_of_view,headless=spec.headless,step_delay_seconds=0.0,render_instance_segmentation=False,local_executable_path=spec.thor_executable,learning_state_dir=str(root),use_learned_navigation=False,use_interactable_poses=True,max_interaction_pose_replans=spec.max_interaction_pose_replans,allow_legacy_object_center_fallback=False,use_interaction_actionability=True,max_interaction_action_retries=spec.max_interaction_action_retries,preferred_open_clearance_m=spec.preferred_open_clearance_m,interaction_recovery_exclusion_radius_m=spec.interaction_recovery_exclusion_radius_m,interaction_recovery_min_position_change_m=spec.interaction_recovery_min_position_change_m,simulator_step_timeout_seconds=spec.simulator_step_timeout_seconds,simulator_start_timeout_seconds=spec.simulator_start_timeout_seconds)
    nav_cfg=NavigationConfig(spec.grid_size,spec.rotate_step_degrees,0.10)
    shared=SharedThorWorld(thor_cfg,agent_count=2,heartbeat_path=root/"heartbeat.json")
    try:
        shared.start(); adapters={i:shared.agent(i) for i in (0,1)}
        for adapter in adapters.values(): adapter.start()
        initial_reachable=adapters[0].get_reachable_positions(); metrics.reachable_cells=len(initial_reachable)
        if len(initial_reachable)<2: raise RuntimeError("AI2-THOR returned fewer than two reachable positions for training.")
        discovery_prop_baseline = _capture_discovery_prop_positions(adapters[0])

        raw_attempts=0; max_raw=max(spec.episodes*spec.max_setup_attempts_per_episode,spec.episodes); last_progress_reported=-1
        while metrics.episodes < spec.episodes and raw_attempts < max_raw:
            raw_attempts += 1
            active=(metrics.episodes + spec.worker_id) % 2; label="A" if active==0 else "B"; adapter=adapters[active]; library=libraries[label]
            chosen=_weighted_choice(rng,spec.curriculum)
            shared.set_heartbeat_context(worker_id=spec.worker_id,level=spec.level,episode=metrics.episodes+1,raw_attempt=raw_attempts,skill=chosen,agent=label)
            try:
                props_ok, props_reason, restored = _restore_discovery_props(shared, adapters, active, discovery_prop_baseline)
                metrics.discovery_prop_restores += int(restored)
                if not props_ok:
                    result=_skip(chosen,label,props_reason or "trainer could not restore Discovery props")
                elif chosen=="navigate":
                    if spec.level <= 2:
                        topo_ok, topo_reason = _prepare_level_topology(shared, adapter, active, spec.level)
                    else:
                        topo_ok = _set_all_door_state(shared, adapter, active, open_state=True)
                        topo_reason = None if topo_ok else "trainer could not open navigation connectors"
                    if not topo_ok:
                        result=_skip("navigate",label,topo_reason or "trainer topology preparation failed")
                    else:
                        reachable=adapter.get_reachable_positions()
                        pair=_navigation_pair_for_level(rng,reachable,spec.level,spec.grid_size)
                        if pair is None:
                            result=_skip("navigate",label,f"no valid level-{spec.level} point-navigation pair")
                        else:
                            start,target=pair
                            if spec.level >= 5:
                                _place_peer_near_but_not_on_target(shared,1-active,reachable,target,start)
                            else:
                                _park_peer(shared,1-active,reachable,target)
                            if not _teleport(shared,active,start,rng.choice((0.0,90.0,180.0,270.0))):
                                result=_skip("navigate",label,"trainer teleport failed")
                            else:
                                result=_run_route(adapter=adapter,agent_label=label,episode=metrics.episodes+1,target=dict(target),reachable=reachable,nav_cfg=nav_cfg,world_map=world_map,skill=nav_skill,replay=nav_replay,arrival_distance=max(0.05,spec.grid_size*0.6),max_steps=spec.max_steps_per_episode,embodied_library=library)
                elif chosen=="door":
                    result=_train_door(shared=shared,adapters=adapters,active=active,agent_label=label,episode=metrics.episodes+1,rng=rng,replay_path=embodied_replay,library=library,verifier=verifier,level=spec.level,max_steps=spec.max_steps_per_episode)
                elif chosen=="pickup":
                    result=_train_pickup(shared=shared,adapters=adapters,active=active,agent_label=label,episode=metrics.episodes+1,rng=rng,replay_path=embodied_replay,library=library,verifier=verifier,level=spec.level,max_steps=spec.max_steps_per_episode)
                elif chosen=="place":
                    result=_train_place(shared=shared,adapters=adapters,active=active,agent_label=label,episode=metrics.episodes+1,rng=rng,replay_path=embodied_replay,library=library,verifier=verifier,level=spec.level,max_steps=spec.max_steps_per_episode)
                elif chosen=="toggle":
                    result=_train_toggle(shared=shared,adapters=adapters,active=active,agent_label=label,episode=metrics.episodes+1,rng=rng,replay_path=embodied_replay,library=library,verifier=verifier,level=spec.level,max_steps=spec.max_steps_per_episode)
                elif chosen=="peer_meet":
                    result=_train_peer_meet(shared=shared,adapters=adapters,active=active,agent_label=label,episode=metrics.episodes+1,rng=rng,library=library,max_steps=spec.max_steps_per_episode)
                else:
                    result=_skip(chosen,label,"unknown curriculum skill")
            except SimulatorStepTimeout as exc:
                # A timed-out FIFO call leaves request/response framing ambiguous.
                # Never reuse that controller and never score the timeout against A/B.
                metrics.simulator_timeouts += 1
                result=_skip(chosen,label,f"simulator timeout during {exc.operation}")
                print(f"[worker {spec.worker_id}][level {spec.level}] simulator timeout during {exc.operation}; restarting Unity", flush=True)
                try:
                    shared.restart()
                    for live_adapter in adapters.values():
                        live_adapter.start()
                    metrics.simulator_restarts += 1
                except Exception as restart_exc:
                    raise RuntimeError(f"Simulator restart failed after timeout in {exc.operation}: {restart_exc}") from restart_exc
            except SimulatorTransportError as exc:
                # Broken pipe/EOF means the Unity transport is dead. Treat it exactly
                # like a poisoned timed-out controller: unscored, replace, continue.
                metrics.simulator_transport_failures += 1
                result=_skip(chosen,label,f"simulator transport failure during {exc.operation}")
                print(f"[worker {spec.worker_id}][level {spec.level}] simulator transport failure during {exc.operation}; restarting Unity", flush=True)
                try:
                    shared.restart()
                    for live_adapter in adapters.values():
                        live_adapter.start()
                    metrics.simulator_restarts += 1
                except Exception as restart_exc:
                    raise RuntimeError(f"Simulator restart failed after transport failure in {exc.operation}: {restart_exc}") from restart_exc
            except Exception as exc:
                # Simulator/setup exceptions are not competence failures unless the
                # action stage already recorded itself through _execute_stage.
                result=_skip(chosen,label,f"setup exception: {type(exc).__name__}: {exc}")

            if not result.scored:
                metrics.note_skip(result.note or "invalid setup")
                continue
            metrics.note_episode(chosen,result.success); metrics.primitive_steps+=result.primitive_steps
            metrics.successful_actions+=result.successful_actions; metrics.failed_actions+=result.failed_actions
            if spec.export_sft: _append_sft(sft_path,result,spec.scene,spec.level)
            every=max(1,int(spec.progress_every_episodes))
            if metrics.episodes % every == 0 and metrics.episodes != last_progress_reported:
                last_progress_reported=metrics.episodes
                rate=100.0*metrics.successful_episodes/max(1,metrics.episodes)
                print(f"[worker {spec.worker_id}][level {spec.level}] {metrics.episodes}/{spec.episodes} scored | success {rate:.2f}% | skips {metrics.skipped_setups} | timeouts {metrics.simulator_timeouts} | transport {metrics.simulator_transport_failures} | restarts {metrics.simulator_restarts}", flush=True)

        if metrics.episodes < spec.episodes:
            raise RuntimeError(f"Only {metrics.episodes}/{spec.episodes} valid episodes could be generated after {raw_attempts} setup attempts. Skip reasons: {metrics.setup_skip_reasons}")
        world_map.save(); nav_skill.save(); nav_replay.finalize()
        for lib in libraries.values(): lib.save()
        metrics.unique_cells=len(world_map.cells); metrics.wall_seconds=time.perf_counter()-started
        write_json_atomic(root/"metrics.json",metrics.as_dict()); return metrics.as_dict()
    finally:
        shared.close()


def _split_episodes(total: int, workers: int) -> list[int]:
    workers=max(1,min(int(workers),int(total))); base,extra=divmod(int(total),workers)
    return [base+(1 if i<extra else 0) for i in range(workers)]


def _canonical_curriculum_name(name: str) -> str:
    value=str(name).strip().lower(); return CURRICULUM_ALIASES.get(value,value)


def _curriculum_from_config(training_cfg: dict[str, Any], override: str | None = None) -> dict[str, float]:
    if override:
        requested=[_canonical_curriculum_name(p) for p in override.split(",") if p.strip()]
        if not requested: return dict(DEFAULT_CURRICULUM)
        unknown=[name for name in requested if name not in DEFAULT_CURRICULUM]
        if unknown: raise ValueError(f"Unknown curriculum skill(s): {', '.join(unknown)}")
        return {name:1.0 for name in dict.fromkeys(requested)}
    configured=training_cfg.get("curriculum") or {}
    if not isinstance(configured,dict) or not configured: return dict(DEFAULT_CURRICULUM)
    canonical={_canonical_curriculum_name(k):float(v) for k,v in configured.items()}
    merged={name:float(canonical.get(name,0.0)) for name in DEFAULT_CURRICULUM}
    return dict(DEFAULT_CURRICULUM) if sum(max(0.0,v) for v in merged.values())<=0 else merged


def _merge_workers(paths: LearningPaths, scene: str, grid_size: float, worker_dirs: list[Path], replay_max: int) -> dict[str, Any]:
    global_map=PersistentWorldMap(scene,grid_size,paths.scene_map(scene)); global_nav=NavigationSkill(paths.navigation_checkpoint)
    global_skills={label:EmbodiedSkillLibrary(paths.agent_skill_checkpoint(label),agent_label=label) for label in ("A","B")}
    global_actionability={label:PoseActionabilityMemory(paths.agent_actionability_checkpoint(label),agent_label=label,grid_size=grid_size) for label in ("A","B")}
    worker_metrics=[]; nav_replays=[]; embodied_replays=[]; sft_sources=[]; run_cells=set()
    for directory in worker_dirs:
        metrics=load_json(directory/"metrics.json",{})
        if metrics: worker_metrics.append(metrics)
        if (directory/"map.json").exists():
            worker_map=PersistentWorldMap(scene,grid_size,directory/"map.json")
            run_cells.update(worker_map.cells.keys())
            global_map.merge(worker_map)
        if (directory/"navigation_skill.json").exists(): global_nav.merge(NavigationSkill(directory/"navigation_skill.json"))
        for label in ("A","B"):
            p=directory/f"skills_{label}.json"
            if p.exists(): global_skills[label].merge(EmbodiedSkillLibrary(p,agent_label=label))
            ap=directory/"checkpoints"/label.lower()/"interaction_actionability.json"
            if ap.exists():
                worker_actionability=PoseActionabilityMemory(ap,agent_label=label,grid_size=grid_size)
                seed_path=ap.with_name("interaction_actionability_seed.json")
                if seed_path.exists():
                    seed=PoseActionabilityMemory(seed_path,agent_label=label,grid_size=grid_size)
                    worker_actionability=worker_actionability.delta_from(seed)
                global_actionability[label].merge(worker_actionability)
        nav_replays.append(directory/"navigation_replay.jsonl"); embodied_replays.append(directory/"embodied_replay.jsonl"); sft_sources.append(directory/"embodied_sft.jsonl")
    global_map.save(); global_nav.save()
    for lib in global_skills.values(): lib.save()
    for memory in global_actionability.values(): memory.save()
    merged_nav=merge_jsonl_files(nav_replays,paths.replay_file,max_records=replay_max)
    merged_embodied=merge_jsonl_files(embodied_replays,paths.embodied_replay_file,max_records=replay_max)
    merged_sft=merge_jsonl_files(sft_sources,paths.embodied_sft_file,max_records=replay_max)
    attempts={}; successes={}; skip_reasons={}
    for row in worker_metrics:
        for skill,count in (row.get("curriculum_attempts") or {}).items(): attempts[skill]=attempts.get(skill,0)+int(count)
        for skill,count in (row.get("curriculum_successes") or {}).items(): successes[skill]=successes.get(skill,0)+int(count)
        for reason,count in (row.get("setup_skip_reasons") or {}).items(): skip_reasons[reason]=skip_reasons.get(reason,0)+int(count)
    totals={
        "episodes":sum(int(r.get("episodes",0)) for r in worker_metrics),
        "successful_episodes":sum(int(r.get("successful_episodes",0)) for r in worker_metrics),
        "failed_episodes":sum(int(r.get("failed_episodes",0)) for r in worker_metrics),
        "skipped_setups":sum(int(r.get("skipped_setups",0)) for r in worker_metrics),
        "setup_skip_reasons":skip_reasons,
        "primitive_steps":sum(int(r.get("primitive_steps",0)) for r in worker_metrics),
        "successful_actions":sum(int(r.get("successful_actions",0)) for r in worker_metrics),
        "failed_actions":sum(int(r.get("failed_actions",0)) for r in worker_metrics),
        "simulator_timeouts":sum(int(r.get("simulator_timeouts",0)) for r in worker_metrics),
        "simulator_transport_failures":sum(int(r.get("simulator_transport_failures",0)) for r in worker_metrics),
        "simulator_restarts":sum(int(r.get("simulator_restarts",0)) for r in worker_metrics),
        "discovery_prop_restores":sum(int(r.get("discovery_prop_restores",0)) for r in worker_metrics),
        "reachable_cells":max([int(r.get("reachable_cells",0)) for r in worker_metrics] or [0]),
        "unique_cells":len(global_map.cells),
        "merged_navigation_replay_transitions":merged_nav,
        "merged_embodied_replay_transitions":merged_embodied,
        "merged_sft_examples":merged_sft,
        "navigation_skill_transitions":global_nav.transitions,
        "curriculum_attempts":attempts,
        "curriculum_successes":successes,
        "agent_skills":{label:lib.compact_summary() for label,lib in global_skills.items()},
    }
    totals["episode_success_rate_pct"]=round(100.0*totals["successful_episodes"]/max(1,totals["episodes"]),2)
    totals["action_success_rate_pct"]=round(100.0*totals["successful_actions"]/max(1,totals["successful_actions"]+totals["failed_actions"]),2)
    # v0.10.4: persistent world-map cells can span earlier runs. Report current
    # worker coverage separately so a cumulative map can never show >100% run coverage.
    run_unique=len(run_cells)
    totals["run_unique_cells"]=run_unique
    totals["persistent_unique_cells"]=totals["unique_cells"]
    totals["run_coverage_pct"]=round(min(100.0,100.0*run_unique/max(1,totals["reachable_cells"])),2)
    totals["coverage_pct"]=totals["run_coverage_pct"]
    totals["curriculum_success_rate_pct"]={skill:round(100.0*successes.get(skill,0)/max(1,count),2) for skill,count in sorted(attempts.items())}
    return {"totals":totals,"workers":worker_metrics}


def _append_training_memories(paths: LearningPaths, summary: dict[str, Any], max_records: int) -> None:
    for label in ("A","B"):
        own=summary.get("totals",{}).get("agent_skills",{}).get(label,{})
        memory=EpisodicMemory(str(paths.agent_memory(label)),max_records=max_records,load_existing=True)
        memory.add("accelerated_embodied_training_summary",{"scene":summary.get("scene"),"level":summary.get("level"),"episodes":summary.get("totals",{}).get("episodes",0),"primitive_steps":summary.get("totals",{}).get("primitive_steps",0),"your_embodied_skill_training":own,"note":"Your own verified staged body-skill competence; peer private state is not included."})


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="AUTHOR v0.10.15 route-safe semantic-route staged embodied curriculum trainer (zero LLM calls by default).")
    parser.add_argument("--config",default="config.yaml"); parser.add_argument("--scene",default=None)
    parser.add_argument("--episodes",type=int,default=None); parser.add_argument("--workers",type=int,default=None)
    parser.add_argument("--seed",type=int,default=72611); parser.add_argument("--state-dir",default=None)
    parser.add_argument("--thor-executable",default=None); parser.add_argument("--headless",action="store_true")
    parser.add_argument("--width",type=int,default=None); parser.add_argument("--height",type=int,default=None)
    parser.add_argument("--max-steps-per-episode",type=int,default=None)
    parser.add_argument("--level",type=int,choices=list(VALID_LEVELS),default=None,help="Curriculum difficulty: 0 direct pose, 1 near, 2 room-scale, 3 connected-component, 4 door-aware/full, 5 interference-ready.")
    parser.add_argument("--curriculum",default=None,help="Comma-separated: navigate,door (or open_cross),pickup,place,toggle,peer_meet")
    parser.add_argument("--no-sft-export",action="store_true")
    parser.add_argument("--reset-memory",action="store_true",help="Delete persistent state. Normally DO NOT use this when upgrading from v0.10.0.")
    parser.add_argument("--dry-run",action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args=build_parser().parse_args(argv); cfg=_load_config(args.config); training_cfg=cfg.get("training",{}); learning_cfg=cfg.get("learning",{}); sim_cfg=cfg.get("simulation",{})
    nav_cfg=cfg.get("navigation",{}); scene=str(args.scene or cfg.get("scene","FloorPlan1Discovery_physics")); episodes=int(args.episodes if args.episodes is not None else training_cfg.get("episodes",500)); workers=int(args.workers if args.workers is not None else training_cfg.get("workers",1))
    if episodes<=0: raise ValueError("--episodes must be positive.")
    if workers<=0: raise ValueError("--workers must be positive.")
    level=int(args.level if args.level is not None else training_cfg.get("level",0))
    state_dir=Path(args.state_dir or learning_cfg.get("state_dir","state")).expanduser().resolve()
    if args.reset_memory:
        print(f"Cleared persistent learning/memory state: {reset_learning_state(state_dir)}",flush=True)
    paths=LearningPaths.from_root(state_dir).ensure()
    max_steps=int(args.max_steps_per_episode if args.max_steps_per_episode is not None else training_cfg.get("max_steps_per_episode",140)); width=int(args.width if args.width is not None else training_cfg.get("image_width",160)); height=int(args.height if args.height is not None else training_cfg.get("image_height",120))
    thor_executable=args.thor_executable or sim_cfg.get("local_executable_path") or None; replay_max=int(learning_cfg.get("replay_max_records",250000)); memory_max=int(learning_cfg.get("memory_max_records",12000)); curriculum=_curriculum_from_config(training_cfg,args.curriculum); export_sft=bool(training_cfg.get("export_sft",True)) and not args.no_sft_export
    plan={"version":"0.10.15","trainer":"route_safe_semantic_route_curriculum","scene":scene,"episodes":episodes,"workers":min(workers,episodes),"level":level,"curriculum":curriculum,"max_steps_per_episode":max_steps,"render_size":[width,height],"headless":bool(args.headless),"llm_calls":0,"export_sft":export_sft,"state_dir":str(paths.root),"custom_thor_executable":thor_executable,"invalid_setup_policy":"skip_not_failure","legacy_checkpoint_policy":"preserve_and_migrate","simulator_step_timeout_seconds":float(training_cfg.get("simulator_step_timeout_seconds",30.0)),"simulator_start_timeout_seconds":float(training_cfg.get("simulator_start_timeout_seconds",60.0))}
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run: return 0
    run_id=time.strftime("%Y%m%d-%H%M%S"); root=paths.worker_dir/run_id; root.mkdir(parents=True,exist_ok=True); splits=_split_episodes(episodes,workers); specs=[]
    for wid,n in enumerate(splits):
        specs.append(WorkerSpec(worker_id=wid,episodes=n,scene=scene,grid_size=float(cfg.get("grid_size",0.25)),rotate_step_degrees=float(cfg.get("rotate_step_degrees",90)),visibility_distance=float(cfg.get("visibility_distance",1.5)),field_of_view=int(cfg.get("field_of_view",90)),width=width,height=height,headless=bool(args.headless),thor_executable=thor_executable,seed=int(args.seed)+wid*100003,max_steps_per_episode=max_steps,worker_root=str(root/f"worker_{wid:02d}"),replay_max_records=replay_max,curriculum=dict(curriculum),interaction_distance=float(nav_cfg.get("interaction_distance",1.25)),max_interaction_pose_replans=int(nav_cfg.get("max_interaction_pose_replans",3)),max_interaction_action_retries=int(nav_cfg.get("max_interaction_action_retries",3)),preferred_open_clearance_m=float(nav_cfg.get("preferred_open_clearance_m",1.0)),interaction_recovery_exclusion_radius_m=float(nav_cfg.get("interaction_recovery_exclusion_radius_m",0.50)),interaction_recovery_min_position_change_m=float(nav_cfg.get("interaction_recovery_min_position_change_m",0.25)),level=level,max_setup_attempts_per_episode=int(training_cfg.get("max_setup_attempts_per_episode",12)),simulator_step_timeout_seconds=float(training_cfg.get("simulator_step_timeout_seconds",30.0)),simulator_start_timeout_seconds=float(training_cfg.get("simulator_start_timeout_seconds",60.0)),progress_every_episodes=int(training_cfg.get("progress_every_episodes",10)),export_sft=export_sft))
    # Seed each worker with a private copy of persistent A/B pose-actionability.
    # Workers never write the same file concurrently; their updates merge afterward.
    for spec in specs:
        worker_paths=LearningPaths.from_root(Path(spec.worker_root)).ensure()
        for label in ("A","B"):
            source=paths.agent_actionability_checkpoint(label)
            destination=worker_paths.agent_actionability_checkpoint(label)
            if source.exists():
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
                shutil.copy2(source,destination.with_name("interaction_actionability_seed.json"))
    started=time.perf_counter()
    if len(specs)==1: _worker_main(specs[0])
    else:
        ctx=mp.get_context("spawn")
        with ctx.Pool(processes=len(specs)) as pool: pool.map(_worker_main,specs)
    merged=_merge_workers(paths,scene,float(cfg.get("grid_size",0.25)),[Path(s.worker_root) for s in specs],replay_max); wall=time.perf_counter()-started
    summary={"version":"0.10.15","timestamp":utc_now(),"run_id":run_id,"scene":scene,"level":level,"workers":len(specs),"wall_seconds":round(wall,3),"curriculum":curriculum,**merged}
    summary["totals"]["aggregate_steps_per_wall_second"]=round(summary["totals"]["primitive_steps"]/wall,2) if wall>0 else 0.0
    write_json_atomic(paths.training_summary,summary); append_jsonl(paths.training_history,summary); _append_training_memories(paths,summary,memory_max)
    if not bool(training_cfg.get("keep_worker_artifacts",False)): shutil.rmtree(root,ignore_errors=True)
    print("\nEmbodied curriculum training complete.",flush=True); print(json.dumps({"state_dir":str(paths.root),"level":level,**summary["totals"]},indent=2),flush=True)
    print(f"Agent A skills:        {paths.agent_skill_checkpoint('A')}",flush=True); print(f"Agent B skills:        {paths.agent_skill_checkpoint('B')}",flush=True); print(f"Embodied replay:       {paths.embodied_replay_file}",flush=True)
    if export_sft: print(f"SFT/LoRA dataset:      {paths.embodied_sft_file}",flush=True)
    print(f"Training summary:      {paths.training_summary}",flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
