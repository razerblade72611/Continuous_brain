from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
import errno
from pathlib import Path
import signal
import threading
import select
import struct
from collections import defaultdict
from typing import Any, Callable
import time

from .interaction_pose import InteractionPose, choose_best_interaction_pose, forward_point
from .models import Action, ActionKind
from .navigation import NavigationConfig, facing_yaw, path_to_actions, planar_distance, rotation_actions, shortest_grid_path
from .learning import LearningPaths
from .nav_skill import NavigationSkill
from .world_map import PersistentWorldMap
from .world_state import normalize_event
from .actionability import PoseActionabilityMemory


@dataclass
class ThorConfig:
    scene: str = "FloorPlan1"
    grid_size: float = 0.25
    rotate_step_degrees: float = 90.0
    visibility_distance: float = 1.5
    interaction_distance: float = 1.25
    width: int = 640
    height: int = 480
    field_of_view: int = 90
    headless: bool = False
    step_delay_seconds: float = 0.0
    render_instance_segmentation: bool = True
    local_executable_path: str | None = None
    learning_state_dir: str | None = None
    use_learned_navigation: bool = True
    # v0.10.3: interaction approach uses simulator-authoritative poses.  Failing
    # closed is deliberate; object-center fallback recreates v0.10's bad labels.
    use_interactable_poses: bool = True
    max_interaction_pose_replans: int = 3
    allow_legacy_object_center_fallback: bool = False
    # If the selected semantic target lies beyond a closed interior door, the body
    # may use already-known Door objects as controller-private route infrastructure.
    # It never changes the semantic target exposed to the LLM.
    door_aware_route_continuation: bool = True
    max_door_route_hops: int = 3
    # v0.10.3: GetInteractablePoses supplies visibility-valid candidates, not a
    # guarantee that every physical interaction will succeed. Learn per-pose
    # actionability and recover by trying alternate candidates before giving up.
    use_interaction_actionability: bool = True
    max_interaction_action_retries: int = 3
    preferred_open_clearance_m: float = 1.0
    # v0.10.3: after a moving-object interaction fails, avoid the whole nearby
    # physical region rather than wasting retries on yaw/horizon variants there.
    interaction_recovery_exclusion_radius_m: float = 0.50
    interaction_recovery_min_position_change_m: float = 0.25
    # v0.10.6: AUTHOR-side watchdog for blocking FIFO simulator calls.
    simulator_step_timeout_seconds: float = 30.0
    simulator_start_timeout_seconds: float = 60.0
    agent_label: str | None = None


class SimulatorStepTimeout(TimeoutError):
    """Raised when AUTHOR interrupts a simulator call that exceeded its deadline."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = str(operation)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(f"AI2-THOR operation {self.operation!r} exceeded {self.timeout_seconds:.1f}s")


class SimulatorTransportError(ConnectionError):
    """Raised when Unity/FIFO transport is dead and the controller must be replaced."""

    def __init__(self, operation: str, cause: BaseException | str):
        self.operation = str(operation)
        self.cause = cause
        super().__init__(f"AI2-THOR transport failed during {self.operation!r}: {cause}")


def _is_simulator_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, SimulatorTransportError):
        return True
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, EOFError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN, errno.EBADF, errno.ENXIO}:
        return True
    return False




def _install_ai2thor_fifo_timeout_patch(default_timeout_seconds: float) -> bool:
    """Patch ai2thor 4.3.x FIFO reads with select()-based deadlines.

    AI2-THOR 4.3.x performs blocking ``server_pipe.read`` calls without any
    readiness timeout.  Python signals are not reliable enough to recover that
    path on every WSL/kernel combination, so AUTHOR guards the actual FIFO file
    descriptor with ``select`` and raises ``SimulatorStepTimeout`` itself.
    """
    try:
        import ai2thor.fifo_server as fifo_server
        import msgpack
    except Exception:
        return False

    cls = fifo_server.FifoServer
    cls._author_default_timeout_seconds = float(default_timeout_seconds or 0.0)
    if getattr(cls, "_author_select_timeout_patch", False):
        return True

    original_recv = cls._recv_message

    def _recv_message_with_deadline(self):
        timeout = float(getattr(self, "_author_receive_timeout_seconds", getattr(cls, "_author_default_timeout_seconds", 0.0)) or 0.0)
        if timeout <= 0.0:
            return original_recv(self)
        operation = str(getattr(self, "_author_operation", "FIFO.receive"))
        deadline = time.monotonic() + timeout
        if self.server_pipe is None:
            # Unity has already opened the writer before normal action traffic.
            self.server_pipe = open(self.server_pipe_path, "rb", buffering=0)

        def _read_exact(size: int) -> bytes:
            remaining = int(size)
            chunks: list[bytes] = []
            fd = self.server_pipe.fileno()
            while remaining > 0:
                wait = deadline - time.monotonic()
                if wait <= 0.0:
                    raise SimulatorStepTimeout(operation, timeout)
                ready, _, _ = select.select([fd], [], [], wait)
                if not ready:
                    raise SimulatorStepTimeout(operation, timeout)
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        metadata = None
        files = defaultdict(list)
        while True:
            header = _read_exact(self.header_size)
            if len(header) == 0:
                raise SimulatorTransportError(operation, "Unity FIFO closed before a reply")
            if len(header) != self.header_size:
                raise SimulatorTransportError(operation, f"short Unity FIFO header: {len(header)}/{self.header_size} bytes")
            if header[0] == fifo_server.FieldType.END_OF_MESSAGE.value:
                break
            field_type_int, message_length = struct.unpack(self.header_format, header)
            field_type = self.field_types[field_type_int]
            body = _read_exact(message_length)
            if len(body) != message_length:
                raise SimulatorTransportError(operation, f"short Unity FIFO body: {len(body)}/{message_length} bytes")
            if field_type is fifo_server.FieldType.METADATA:
                metadata = msgpack.loads(body, raw=False, strict_map_key=False)
            elif field_type is fifo_server.FieldType.METADATA_PATCH:
                metadata_patch = msgpack.loads(body, raw=False, strict_map_key=False)
                agents = self.raw_metadata["agents"]
                metadata = dict(
                    agents=[{} for _ in range(len(agents))],
                    thirdPartyCameras=self.raw_metadata["thirdPartyCameras"],
                    sequenceId=self.sequence_id,
                    activeAgentId=metadata_patch["agentId"],
                )
                for i in range(len(agents)):
                    metadata["agents"][i].update(agents[i])
                metadata["agents"][metadata_patch["agentId"]].update(metadata_patch)
                files = self.raw_files
            elif field_type in self.image_fields:
                files[self.form_field_map[field_type]].append(body)
            else:
                raise ValueError(f"Invalid FIFO field type: {field_type}")
        self.raw_metadata = metadata
        self.raw_files = files
        return metadata, files

    cls._recv_message = _recv_message_with_deadline
    cls._author_select_timeout_patch = True
    return True


def _configure_controller_fifo_timeout(controller: Any, timeout_seconds: float, operation: str) -> None:
    server = getattr(controller, "server", None)
    if server is None:
        return
    setattr(server, "_author_receive_timeout_seconds", float(timeout_seconds or 0.0))
    setattr(server, "_author_operation", str(operation))

def _call_with_hard_timeout(fn: Callable[[], Any], timeout_seconds: float, operation: str):
    """Bound blocking AI2-THOR calls on POSIX worker main threads.

    ai2thor 4.3.x FIFO receive() performs a plain blocking pipe read. On
    Linux/WSL, SIGALRM safely interrupts that read and lets AUTHOR discard the
    now-desynchronized controller instead of hanging a training run forever.
    """
    timeout = float(timeout_seconds or 0.0)
    if timeout <= 0.0 or os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return fn()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def _deadline(_signum, _frame):
        raise SimulatorStepTimeout(operation, timeout)
    signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@dataclass
class ThorExecution:
    event: Any
    state: Any
    primitive_steps: int
    primitive_actions: list[str] | None = None
    auxiliary_skills: list[dict[str, Any]] | None = None
    # v0.10.4: explicit controller evidence for pseudo-action goals such as
    # NavigateTo(interaction pose). Higher-level verification must consume this
    # instead of re-imposing an object-center distance heuristic.
    goal_verification: dict[str, Any] | None = None


def _learned_route_actions(config: ThorConfig, world, target_position: dict[str, float], interaction_distance: float) -> tuple[list[dict[str, float]], list[Action]]:
    if not config.use_learned_navigation or not config.learning_state_dir:
        return [], []
    paths = LearningPaths.from_root(config.learning_state_dir)
    map_path = paths.scene_map(config.scene)
    if not map_path.exists():
        return [], []
    learned_map = PersistentWorldMap(config.scene, config.grid_size, map_path)
    path = learned_map.route(world.pose.position, target_position, interaction_distance=interaction_distance)
    if not path:
        return [], []
    skill = NavigationSkill(paths.navigation_checkpoint)
    actions = skill.route_actions(
        path,
        initial_yaw=float(world.pose.rotation.get("y", 0.0)),
        final_target=target_position,
        rotate_step_degrees=config.rotate_step_degrees,
    )
    return path, actions


def _choose_cross_threshold_target(current_position: dict[str, float], door_position: dict[str, float], reachable: list[dict[str, float]], *, min_beyond: float = 0.35, max_lateral: float = 0.9) -> dict[str, float] | None:
    cx, cz = float(current_position["x"]), float(current_position["z"])
    dx, dz = float(door_position["x"]) - cx, float(door_position["z"]) - cz
    norm = math.hypot(dx, dz)
    if norm < 1e-6:
        return None
    ux, uz = dx / norm, dz / norm
    candidates: list[tuple[float, float, dict[str, float]]] = []
    for point in reachable:
        px = float(point["x"]) - float(door_position["x"])
        pz = float(point["z"]) - float(door_position["z"])
        beyond = px * ux + pz * uz
        lateral = abs(px * (-uz) + pz * ux)
        if beyond < min_beyond or lateral > max_lateral:
            continue
        candidates.append((abs(beyond - 0.9) + lateral * 1.5, beyond, point))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]))
    return dict(candidates[0][2])


def _look_actions(current_horizon: float, desired_horizon: float) -> list[Action]:
    current = max(-30.0, min(60.0, round(float(current_horizon) / 30.0) * 30.0))
    desired = max(-30.0, min(60.0, round(float(desired_horizon) / 30.0) * 30.0))
    actions: list[Action] = []
    while current < desired - 1e-6:
        actions.append(Action(ActionKind.LOOK_DOWN.value, {"degrees": 30}, "Align camera with interaction pose."))
        current += 30.0
    while current > desired + 1e-6:
        actions.append(Action(ActionKind.LOOK_UP.value, {"degrees": 30}, "Align camera with interaction pose."))
        current -= 30.0
    return actions


def _execute_primitives(adapter, primitives: list[Action], step_budget: int | None) -> ThorExecution:
    if step_budget is not None:
        primitives = primitives[: max(0, int(step_budget))]
    if not primitives:
        return ThorExecution(adapter.raw_event(), adapter.observe(), 0, [])
    last = None
    count = 0
    executed_names: list[str] = []
    for primitive in primitives:
        last = adapter._execute_primitive(primitive)
        count += 1
        executed_names.append(primitive.kind)
        if last.state.last_action_success is False:
            break
    if last is None:
        return ThorExecution(adapter.raw_event(), adapter.observe(), 0, [])
    last.primitive_steps = count
    last.primitive_actions = executed_names
    return last


def _execute_pose_goal(adapter, pose: InteractionPose, reachable: list[dict[str, float]], step_budget: int | None = None) -> ThorExecution:
    world = adapter.observe()
    nav_cfg = NavigationConfig(
        grid_size=adapter.config.grid_size,
        rotate_step_degrees=adapter.config.rotate_step_degrees,
        interaction_distance=max(0.01, adapter.config.grid_size * 0.35),
    )
    path = shortest_grid_path(reachable, world.pose.position, pose.position, nav_cfg)
    primitives = path_to_actions(
        path,
        initial_yaw=float(world.pose.rotation.get("y", 0.0)),
        # Point one metre forward from the desired pose so path_to_actions ends at
        # the exact rotation returned by GetInteractablePoses rather than object center.
        target_position=forward_point(pose),
        config=nav_cfg,
    )
    movement = _execute_primitives(adapter, primitives, step_budget)
    used = movement.primitive_steps
    if movement.state.last_action_success is False:
        return movement
    remaining = None if step_budget is None else max(0, int(step_budget) - used)
    look = _look_actions(movement.state.pose.horizon, pose.horizon)
    looked = _execute_primitives(adapter, look, remaining)
    looked.primitive_steps += used
    looked.primitive_actions = list(movement.primitive_actions or []) + list(looked.primitive_actions or [])
    return looked




def _current_interaction_pose(adapter) -> InteractionPose:
    world = adapter.observe()
    return InteractionPose(
        x=float(world.pose.position["x"]),
        y=float(world.pose.position.get("y", 0.9)),
        z=float(world.pose.position["z"]),
        rotation=float(world.pose.rotation.get("y", 0.0)) % 360.0,
        horizon=float(world.pose.horizon),
        standing=True,
    )


def _interaction_action_succeeded(before, after, action: Action) -> bool:
    object_id = str(action.params.get("objectId", ""))
    obj = after.objects.get(object_id) if object_id else None
    if after.last_action_success is False:
        return False
    if action.kind == ActionKind.OPEN.value:
        return bool(obj is not None and obj.is_open is True)
    if action.kind == ActionKind.CLOSE.value:
        return bool(obj is not None and obj.is_open is False)
    if action.kind == ActionKind.PICKUP.value:
        return bool(object_id and after.held_object_id == object_id)
    if action.kind == ActionKind.PUT.value:
        held_before = before.held_object_id
        if not held_before or after.held_object_id is not None:
            return False
        source = after.objects.get(held_before)
        return bool(source is not None and object_id in (source.parent_receptacles or []))
    if action.kind == ActionKind.TOGGLE_ON.value:
        return bool(obj is not None and obj.is_toggled is True)
    if action.kind == ActionKind.TOGGLE_OFF.value:
        return bool(obj is not None and obj.is_toggled is False)
    return after.last_action_success is not False


def _interaction_kinds() -> set[str]:
    return {
        ActionKind.OPEN.value,
        ActionKind.CLOSE.value,
        ActionKind.PICKUP.value,
        ActionKind.PUT.value,
        ActionKind.TOGGLE_ON.value,
        ActionKind.TOGGLE_OFF.value,
    }


def _actionability_memory(adapter) -> PoseActionabilityMemory | None:
    if not adapter.config.use_interaction_actionability or not adapter.config.learning_state_dir:
        return None
    memory = getattr(adapter, "_pose_actionability_memory", None)
    if memory is not None:
        return memory
    label = getattr(adapter, "agent_label", None) or adapter.config.agent_label or "solo"
    paths = LearningPaths.from_root(adapter.config.learning_state_dir).ensure()
    memory = PoseActionabilityMemory(
        paths.agent_actionability_checkpoint(label),
        agent_label=label,
        grid_size=adapter.config.grid_size,
    )
    setattr(adapter, "_pose_actionability_memory", memory)
    return memory


def _pose_penalty_for_action(adapter, action: Action, target, memory: PoseActionabilityMemory | None):
    if memory is None:
        # Even before experience exists, open/close prefers enough body clearance
        # to avoid standing inside a door/drawer swing envelope. This is soft.
        def prior(pose: InteractionPose) -> float:
            if action.kind not in {ActionKind.OPEN.value, ActionKind.CLOSE.value} or target is None or target.position is None:
                return 0.0
            distance = planar_distance(pose.position, target.position)
            preferred = max(0.0, float(adapter.config.preferred_open_clearance_m))
            return max(0.0, preferred - distance) * 80.0
        return prior

    def learned(pose: InteractionPose) -> float:
        return memory.penalty(
            action.kind,
            str(action.params.get("objectId", "")),
            pose,
            object_position=None if target is None else target.position,
            preferred_open_clearance_m=adapter.config.preferred_open_clearance_m,
        )
    return learned


def _recovery_pose_pool(adapter, action: Action, target, poses: list[InteractionPose], failed_poses: list[InteractionPose]) -> list[InteractionPose]:
    """Return recovery candidates ordered only after safety/generalization filters.

    Open/Close failures often describe a swing/collision region, not a bad camera
    orientation.  Prefer a materially different position and, when available, a
    pose at or beyond the configured clearance.  If a narrow scene has no strict
    candidate, relax radius first but never retry the exact failed X/Z cell.
    """
    if not failed_poses:
        return list(poses)

    moving_object = action.kind in {ActionKind.OPEN.value, ActionKind.CLOSE.value}
    grid = max(0.01, float(adapter.config.grid_size))
    min_change = max(grid * 0.9, float(adapter.config.interaction_recovery_min_position_change_m))
    radius = max(min_change, float(adapter.config.interaction_recovery_exclusion_radius_m))
    failed_position_keys = {pose.position_key(grid) for pose in failed_poses}

    def changed_position(pose: InteractionPose, threshold: float) -> bool:
        if moving_object and pose.position_key(grid) in failed_position_keys:
            return False
        return all(planar_distance(pose.position, old.position) >= threshold - 1e-6 for old in failed_poses)

    strict = [pose for pose in poses if changed_position(pose, radius if moving_object else min_change)]
    if not strict:
        strict = [pose for pose in poses if changed_position(pose, min_change)]
    if not strict:
        strict = [pose for pose in poses if pose.position_key(grid) not in failed_position_keys] if moving_object else list(poses)

    if moving_object and target is not None and target.position is not None and strict:
        preferred = max(0.0, float(adapter.config.preferred_open_clearance_m))
        safe = [pose for pose in strict if planar_distance(pose.position, target.position) >= preferred - 1e-6]
        if safe:
            strict = safe
    return strict


def _execute_interaction_with_pose_recovery(adapter, action: Action, step_budget: int | None = None) -> ThorExecution:
    """Execute one semantic interaction with bounded spatially-aware recovery."""
    if bool(action.params.get("forceAction", False)) or not adapter.config.use_interaction_actionability:
        return adapter._execute_primitive(action)

    object_id = str(action.params.get("objectId", ""))
    if not object_id:
        return adapter._execute_primitive(action)

    max_retries = max(0, int(adapter.config.max_interaction_action_retries))
    parts: list[ThorExecution] = []
    telemetry: list[dict[str, Any]] = []
    excluded: set[tuple[int, int, int, int]] = set()
    failed_poses: list[InteractionPose] = []
    memory = _actionability_memory(adapter)
    recovery_started = False
    final_success = False

    for attempt in range(max_retries + 1):
        before = adapter.observe()
        target = before.objects.get(object_id)
        if target is None:
            break

        pose = _current_interaction_pose(adapter)
        used = sum(max(0, int(row.primitive_steps)) for row in parts)
        remaining = None if step_budget is None else max(0, int(step_budget) - used)
        if remaining is not None and remaining <= 0:
            break

        if attempt > 0:
            recovery_started = True
            reachable = adapter.get_reachable_positions()
            poses = adapter.get_interactable_poses(object_id, positions=reachable, standings=[True])
            poses = _recovery_pose_pool(adapter, action, target, poses, failed_poses)
            nav_cfg = NavigationConfig(
                grid_size=adapter.config.grid_size,
                rotate_step_degrees=adapter.config.rotate_step_degrees,
                interaction_distance=adapter.config.interaction_distance,
            )
            excluded_positions = (
                {p.position_key(adapter.config.grid_size) for p in failed_poses}
                if action.kind in {ActionKind.OPEN.value, ActionKind.CLOSE.value}
                else set()
            )
            chosen = choose_best_interaction_pose(
                poses,
                reachable,
                before.pose.position,
                float(before.pose.rotation.get("y", 0.0)),
                float(before.pose.horizon),
                nav_cfg,
                excluded_keys=excluded,
                excluded_position_keys=excluded_positions,
                pose_penalty=_pose_penalty_for_action(adapter, action, target, memory),
                # Recovery is actionability-first: once a physical interaction has
                # failed, walking a few extra cells is cheaper than another bad try.
                move_cost_weight=1.0,
                turn_cost_weight=0.25,
                look_cost_weight=0.10,
            )
            if chosen is None:
                break
            excluded.add(chosen.key(adapter.config.grid_size))
            moved = _execute_pose_goal(adapter, chosen, reachable, remaining)
            parts.append(moved)
            after_move = adapter.observe()
            if not _pose_goal_verified(adapter, after_move, object_id, chosen):
                telemetry.append({
                    "kind": "interaction_pose_recovery",
                    "action": action.kind,
                    "objectId": object_id,
                    "attempt": attempt,
                    "pose_reached": False,
                    "recovery": True,
                    "pose_key": list(chosen.key(adapter.config.grid_size)),
                    "position_key": list(chosen.position_key(adapter.config.grid_size)),
                })
                continue
            pose = chosen
            used = sum(max(0, int(row.primitive_steps)) for row in parts)
            remaining = None if step_budget is None else max(0, int(step_budget) - used)
            if remaining is not None and remaining <= 0:
                break
            before = adapter.observe()

        result = adapter._execute_primitive(action)
        parts.append(result)
        after = adapter.observe()
        success = _interaction_action_succeeded(before, after, action)
        final_success = bool(success)
        error = after.last_error
        if memory is not None:
            memory.record(action.kind, object_id, pose, success=success, error=error, recovery=(attempt > 0))
        telemetry.append({
            "kind": "interaction_pose_attempt",
            "action": action.kind,
            "objectId": object_id,
            "attempt": attempt + 1,
            "success": bool(success),
            "error": error,
            "recovery": bool(attempt > 0),
            "pose_key": list(pose.key(adapter.config.grid_size)),
            "position_key": list(pose.position_key(adapter.config.grid_size)),
            "distance_to_object": None if target.position is None else round(planar_distance(pose.position, target.position), 4),
        })
        if success:
            break
        excluded.add(pose.key(adapter.config.grid_size))
        failed_poses.append(pose)

    telemetry.append({
        "kind": "interaction_recovery_summary",
        "action": action.kind,
        "objectId": object_id,
        "recovery": bool(recovery_started),
        "recovery_attempted": bool(recovery_started),
        "recovery_success": bool(recovery_started and final_success),
        "recovery_exhausted": bool(recovery_started and not final_success),
        "success": bool(final_success),
        "error": None if final_success else (adapter.observe().last_error or None),
        "failed_positions": [list(p.position_key(adapter.config.grid_size)) for p in failed_poses],
    })
    return _merge_executions(adapter, parts, telemetry)


def _legacy_navigate_to(adapter, action: Action, step_budget: int | None = None) -> ThorExecution:
    object_id = str(action.params.get("objectId", ""))
    world = adapter.observe()
    target = world.objects.get(object_id)
    if target is None or target.position is None:
        raise ValueError(f"NavigateTo target not found or has no position: {object_id}")
    nav_cfg = NavigationConfig(
        grid_size=adapter.config.grid_size,
        rotate_step_degrees=adapter.config.rotate_step_degrees,
        interaction_distance=float(action.params.get("interactionDistance", adapter.config.interaction_distance)),
    )
    path, primitives = _learned_route_actions(adapter.config, world, target.position, nav_cfg.interaction_distance)
    if not path:
        reachable = adapter.get_reachable_positions()
        path = shortest_grid_path(reachable, world.pose.position, target.position, nav_cfg)
        primitives = path_to_actions(path, initial_yaw=float(world.pose.rotation.get("y", 0.0)), target_position=target.position, config=nav_cfg)
    return _execute_primitives(adapter, primitives, step_budget)


def _pose_goal_verified(adapter, state, object_id: str, pose: InteractionPose) -> bool:
    obj = state.objects.get(object_id)
    position_ok = planar_distance(state.pose.position, pose.position) <= max(0.16, adapter.config.grid_size * 0.6)
    orientation_ok = abs((float(state.pose.rotation.get("y", 0.0)) - pose.rotation + 180.0) % 360.0 - 180.0) <= adapter.config.rotate_step_degrees / 2.0 + 1e-6
    horizon_ok = abs(float(state.pose.horizon) - pose.horizon) <= 15.0 + 1e-6
    return bool(state.last_action_success is not False and position_ok and orientation_ok and horizon_ok and obj is not None and obj.visible)


def _merge_executions(
    adapter,
    parts: list[ThorExecution],
    auxiliary: list[dict[str, Any]] | None = None,
    *,
    goal_verification: dict[str, Any] | None = None,
) -> ThorExecution:
    if not parts:
        return ThorExecution(
            adapter.raw_event(), adapter.observe(), 0, [], list(auxiliary or []), goal_verification
        )
    last = parts[-1]
    return ThorExecution(
        last.event,
        last.state,
        sum(max(0, int(row.primitive_steps)) for row in parts),
        [name for row in parts for name in (row.primitive_actions or [])],
        list(auxiliary or []) + [item for row in parts for item in (row.auxiliary_skills or [])],
        goal_verification if goal_verification is not None else last.goal_verification,
    )


def _pose_goal_details(adapter, state, object_id: str, pose: InteractionPose | None, *, verified: bool, reason: str, replans: int) -> dict[str, Any]:
    obj = state.objects.get(object_id)
    details: dict[str, Any] = {
        "kind": "interaction_pose_goal",
        "objectId": object_id,
        "pose_goal_verified": bool(verified),
        "reason": str(reason),
        "replans": int(replans),
        "visible": None if obj is None else bool(obj.visible),
        # Object-center distance is retained only as diagnostics. It is not an
        # authority for a pose-aware NavigateTo because articulated/custom assets
        # can have interaction surfaces far from their metadata transform center.
        "object_center_distance_m": None if obj is None else obj.distance,
    }
    if pose is not None:
        details.update({
            "target_pose_key": list(pose.key(adapter.config.grid_size)),
            "target_position_key": list(pose.position_key(adapter.config.grid_size)),
            "pose_position_error_m": round(planar_distance(state.pose.position, pose.position), 4),
            "pose_yaw_error_deg": round(abs((float(state.pose.rotation.get("y", 0.0)) - pose.rotation + 180.0) % 360.0 - 180.0), 4),
            "pose_horizon_error_deg": round(abs(float(state.pose.horizon) - pose.horizon), 4),
        })
    return details


def _crossed_door(before, after, door_position: dict[str, float]) -> bool:
    bx, bz = float(before.pose.position["x"]), float(before.pose.position["z"])
    dx = float(door_position["x"]) - bx
    dz = float(door_position["z"]) - bz
    norm = math.hypot(dx, dz)
    if norm < 1e-6:
        return False
    ux, uz = dx / norm, dz / norm
    moved = math.hypot(float(after.pose.position["x"]) - bx, float(after.pose.position["z"]) - bz)
    ax = float(after.pose.position["x"]) - float(door_position["x"])
    az = float(after.pose.position["z"]) - float(door_position["z"])
    beyond = ax * ux + az * uz
    return bool(moved >= 0.35 and beyond >= 0.20)


def _choose_accessible_closed_door(adapter, target_position: dict[str, float] | None, reachable: list[dict[str, float]], visited: set[str]) -> tuple[Any, InteractionPose] | None:
    world = adapter.observe()
    nav_cfg = NavigationConfig(adapter.config.grid_size, adapter.config.rotate_step_degrees, adapter.config.interaction_distance)
    candidates: list[tuple[float, Any, InteractionPose]] = []
    for door in world.objects.values():
        if (
            door.object_id in visited
            or str(door.object_type).lower() != "door"
            or not door.openable
            or door.is_open is not False
            or door.position is None
        ):
            continue
        try:
            poses = adapter.get_interactable_poses(door.object_id, positions=reachable, standings=[True])
            open_action = Action(ActionKind.OPEN.value, {"objectId": door.object_id, "forceAction": False}, "route door actionability")
            pose = choose_best_interaction_pose(
                poses,
                reachable,
                world.pose.position,
                float(world.pose.rotation.get("y", 0.0)),
                float(world.pose.horizon),
                nav_cfg,
                pose_penalty=_pose_penalty_for_action(adapter, open_action, door, _actionability_memory(adapter)),
            )
        except (SimulatorStepTimeout, SimulatorTransportError):
            raise
        except Exception:
            continue
        if pose is None:
            continue
        # Prefer doors that are cheap to reach and whose far side trends toward the
        # already-selected semantic target.  This is motor-route planning only.
        try:
            exact = NavigationConfig(adapter.config.grid_size, adapter.config.rotate_step_degrees, max(0.01, adapter.config.grid_size * 0.35))
            path = shortest_grid_path(reachable, world.pose.position, pose.position, exact)
            path_cost = max(0, len(path) - 1)
        except ValueError:
            continue
        target_cost = 0.0 if target_position is None else planar_distance(door.position, target_position)
        candidates.append((path_cost * 0.35 + target_cost, door, pose))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0])
    return candidates[0][1], candidates[0][2]


def _navigate_to_interactable_pose(adapter, action: Action, step_budget: int | None = None) -> ThorExecution:
    object_id = str(action.params.get("objectId", ""))
    if not object_id:
        raise ValueError("NavigateTo requires params.objectId.")
    if not adapter.config.use_interactable_poses:
        return _legacy_navigate_to(adapter, action, step_budget)

    # v0.10.4 curriculum-only pin. Leading underscore keeps this controller-private
    # and _execute_primitive strips it before any Unity action. Normal live Qwen
    # actions never need or receive coordinates.
    pinned_raw = action.params.get("_targetInteractionPose")
    pinned_pose = InteractionPose.from_thor(pinned_raw) if isinstance(pinned_raw, dict) else None

    max_replans = max(1, int(adapter.config.max_interaction_pose_replans))
    max_door_hops = max(0, int(adapter.config.max_door_route_hops))
    excluded: set[tuple[int, int, int, int]] = set()
    visited_doors: set[str] = set()
    all_parts: list[ThorExecution] = []
    auxiliary: list[dict[str, Any]] = []
    pose_replans = 0
    door_hops = 0
    last_pose: InteractionPose | None = None

    while True:
        current = adapter.observe()
        target = current.objects.get(object_id)
        if target is None:
            if all_parts:
                details = _pose_goal_details(adapter, current, object_id, last_pose or pinned_pose, verified=False, reason="target_missing", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)
            raise ValueError(f"NavigateTo target not found: {object_id}")

        used = sum(row.primitive_steps for row in all_parts)
        remaining = None if step_budget is None else max(0, int(step_budget) - used)
        if remaining is not None and remaining <= 0:
            details = _pose_goal_details(adapter, current, object_id, last_pose or pinned_pose, verified=False, reason="step_budget_exhausted", replans=pose_replans)
            return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)

        reachable = adapter.get_reachable_positions()
        poses = adapter.get_interactable_poses(object_id, positions=reachable, standings=[True])
        nav_cfg = NavigationConfig(
            grid_size=adapter.config.grid_size,
            rotate_step_degrees=adapter.config.rotate_step_degrees,
            interaction_distance=adapter.config.interaction_distance,
        )
        prospective_kind = str(action.params.get("_interactionKind") or "").strip()
        prospective = None
        if prospective_kind in _interaction_kinds():
            prospective = Action(prospective_kind, {"objectId": object_id, "forceAction": False}, "prospective interaction actionability")

        pose: InteractionPose | None = None
        if pinned_pose is not None and poses:
            pinned_key = pinned_pose.key(adapter.config.grid_size)
            pose = next((candidate for candidate in poses if candidate.key(adapter.config.grid_size) == pinned_key), None)
            if pose is None:
                details = _pose_goal_details(adapter, current, object_id, pinned_pose, verified=False, reason="pinned_pose_not_available", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details) if all_parts else ThorExecution(adapter.raw_event(), current, 0, [], auxiliary, details)
        elif pinned_pose is None:
            pose = choose_best_interaction_pose(
                poses,
                reachable,
                current.pose.position,
                float(current.pose.rotation.get("y", 0.0)),
                float(current.pose.horizon),
                nav_cfg,
                excluded_keys=excluded,
                pose_penalty=(None if prospective is None else _pose_penalty_for_action(adapter, prospective, target, _actionability_memory(adapter))),
            )

        if pose is not None and pose_replans < max_replans:
            last_pose = pose
            if pinned_pose is None:
                excluded.add(pose.key(adapter.config.grid_size))
            pose_replans += 1
            execution = _execute_pose_goal(adapter, pose, reachable, remaining)
            all_parts.append(execution)
            after = adapter.observe()
            if _pose_goal_verified(adapter, after, object_id, pose):
                details = _pose_goal_details(adapter, after, object_id, pose, verified=True, reason="interaction_pose_reached", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)
            # Pinned curricula retry the same educational pose after a transient
            # route/primitive miss. Normal runtime excludes the failed candidate.
            continue

        # If the target already has poses in this component but all allowed replans
        # were exhausted, do not wander through unrelated doors looking for an
        # easier problem. Return the real attempted steps and explicit pose failure.
        if poses:
            reason = "pinned_pose_replans_exhausted" if pinned_pose is not None else "interaction_pose_replans_exhausted"
            details = _pose_goal_details(adapter, adapter.observe(), object_id, last_pose or pinned_pose, verified=False, reason=reason, replans=pose_replans)
            return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details) if all_parts else ThorExecution(adapter.raw_event(), adapter.observe(), 0, [], auxiliary, details)

        # No pose exists in the current connected component. If enabled, use an
        # accessible closed Door as route infrastructure, then re-query topology.
        can_route_door = bool(
            adapter.config.door_aware_route_continuation
            and door_hops < max_door_hops
            and target.position is not None
        )
        if can_route_door:
            planned = _choose_accessible_closed_door(adapter, target.position, reachable, visited_doors)
        else:
            planned = None
        if planned is not None:
            door, door_pose = planned
            visited_doors.add(door.object_id)
            door_hops += 1

            approach = _execute_pose_goal(adapter, door_pose, reachable, remaining)
            all_parts.append(approach)
            approach_after = adapter.observe()
            approach_ok = _pose_goal_verified(adapter, approach_after, door.object_id, door_pose)
            auxiliary.append({"skill": "door_approach", "success": approach_ok, "primitive_steps": approach.primitive_steps, "object_type": "Door"})
            if not approach_ok:
                details = _pose_goal_details(adapter, approach_after, object_id, last_pose or pinned_pose, verified=False, reason="blocking_door_approach_failed", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)

            used = sum(row.primitive_steps for row in all_parts)
            remaining = None if step_budget is None else max(0, int(step_budget) - used)
            if remaining is not None and remaining <= 0:
                details = _pose_goal_details(adapter, adapter.observe(), object_id, last_pose or pinned_pose, verified=False, reason="step_budget_exhausted", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)

            route_open_action = Action(ActionKind.OPEN.value, {"objectId": door.object_id, "forceAction": False}, "motor route: open blocking door")
            opened = _execute_interaction_with_pose_recovery(adapter, route_open_action, remaining)
            all_parts.append(opened)
            opened_state = adapter.observe()
            opened_obj = opened_state.objects.get(door.object_id)
            open_ok = bool(opened.state.last_action_success is not False and opened_obj is not None and opened_obj.is_open is True)
            recovery_summary = next((row for row in reversed(opened.auxiliary_skills or []) if row.get("kind") == "interaction_recovery_summary"), {})
            auxiliary.append({
                "skill": "door_open",
                "success": open_ok,
                "primitive_steps": opened.primitive_steps,
                "object_type": "Door",
                "recovery": bool(recovery_summary.get("recovery_attempted")),
                "recovery_success": bool(recovery_summary.get("recovery_success")),
                "recovery_exhausted": bool(recovery_summary.get("recovery_exhausted")),
            })
            if not open_ok:
                details = _pose_goal_details(adapter, opened_state, object_id, last_pose or pinned_pose, verified=False, reason="blocking_door_open_failed", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)

            used = sum(row.primitive_steps for row in all_parts)
            remaining = None if step_budget is None else max(0, int(step_budget) - used)
            if remaining is not None and remaining <= 0:
                details = _pose_goal_details(adapter, adapter.observe(), object_id, last_pose or pinned_pose, verified=False, reason="step_budget_exhausted", replans=pose_replans)
                return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)

            before_cross = adapter.observe()
            try:
                crossed = _cross_threshold_for_adapter(adapter, Action(ActionKind.CROSS_THRESHOLD.value, {"objectId": door.object_id}, "motor route: cross blocking door"), remaining)
                all_parts.append(crossed)
                cross_after = adapter.observe()
                cross_ok = bool(crossed.state.last_action_success is not False and _crossed_door(before_cross, cross_after, door.position))
                auxiliary.append({"skill": "door_cross", "success": cross_ok, "primitive_steps": crossed.primitive_steps, "object_type": "Door"})
            except (SimulatorStepTimeout, SimulatorTransportError):
                raise
            except Exception:
                # Opening can still expand the reachable component enough for the
                # normal path planner to continue. Record the cross miss, then let
                # the next topology query decide rather than fabricating failure.
                auxiliary.append({"skill": "door_cross", "success": False, "primitive_steps": 0, "object_type": "Door", "recovery": True})
            excluded.clear()
            pose_replans = 0
            continue

        if adapter.config.allow_legacy_object_center_fallback and not all_parts:
            return _legacy_navigate_to(adapter, action, remaining)
        if all_parts:
            # Crucial accounting behavior: never throw away primitive steps already
            # executed. Return explicit controller failure evidence to the verifier.
            details = _pose_goal_details(adapter, adapter.observe(), object_id, last_pose or pinned_pose, verified=False, reason="no_interaction_pose_in_component", replans=pose_replans)
            return _merge_executions(adapter, all_parts, auxiliary, goal_verification=details)
        raise ValueError(f"No connected GetInteractablePoses result for target {object_id}; no safe door route was available.")


def _face_target_for_adapter(adapter, action: Action, step_budget: int | None = None) -> ThorExecution:
    """Rotate the body toward the selected semantic target.

    Restored in v0.10.5 after the v0.10.4 pose-authority refactor
    accidentally dropped this live-dispatch helper.
    """
    object_id = str(action.params.get("objectId", ""))
    if not object_id:
        raise ValueError("FaceTarget requires params.objectId.")
    world = adapter.observe()
    target = world.objects.get(object_id)
    if target is None or target.position is None:
        raise ValueError(f"FaceTarget target not found or has no position: {object_id}")
    turns = rotation_actions(
        float(world.pose.rotation.get("y", 0.0)),
        facing_yaw(world.pose.position, target.position),
        adapter.config.rotate_step_degrees,
    )
    return _execute_primitives(adapter, turns, step_budget)


def _cross_threshold_for_adapter(adapter, action: Action, step_budget: int | None = None) -> ThorExecution:
    """Physically move through an already-open selected doorway.

    The connected reachable component is re-queried after opening, then a target
    point is chosen on the far side of the same door and reached with normal
    embodied primitives.  This implementation existed in v0.10.3 and was
    accidentally omitted during the v0.10.4 pose-authority refactor.
    """
    object_id = str(action.params.get("objectId", ""))
    if not object_id:
        raise ValueError("CrossThreshold requires params.objectId.")
    world = adapter.observe()
    door = world.objects.get(object_id)
    if door is None or door.position is None:
        raise ValueError(f"CrossThreshold target not found or has no position: {object_id}")
    if door.openable and door.is_open is not True:
        raise ValueError("CrossThreshold requires the selected door to already be open.")

    # Re-query after opening: the connected topology may have expanded.
    reachable = adapter.get_reachable_positions()
    target = _choose_cross_threshold_target(world.pose.position, door.position, reachable)
    if target is None:
        raise ValueError("No reachable position was found beyond the selected doorway.")

    nav_cfg = NavigationConfig(
        adapter.config.grid_size,
        adapter.config.rotate_step_degrees,
        max(0.05, adapter.config.grid_size * 0.6),
    )
    path = shortest_grid_path(reachable, world.pose.position, target, nav_cfg)
    primitives = path_to_actions(
        path,
        initial_yaw=float(world.pose.rotation.get("y", 0.0)),
        target_position=target,
        config=nav_cfg,
    )
    return _execute_primitives(adapter, primitives, step_budget)


class _AdapterMixin:
    config: ThorConfig

    def get_interactable_poses(self, object_id: str, *, positions: list[dict[str, float]] | None = None, rotations: list[float] | None = None, horizons: list[float] | None = None, standings: list[bool] | None = None) -> list[InteractionPose]:
        params: dict[str, Any] = {"objectId": str(object_id)}
        if positions is not None:
            params["positions"] = positions
        if rotations is not None:
            params["rotations"] = rotations
        if horizons is not None:
            params["horizons"] = horizons
        if standings is not None:
            params["standings"] = standings
        event = self._query_step("GetInteractablePoses", **params)
        if event.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(event.metadata.get("errorMessage") or "GetInteractablePoses failed.")
        raw = event.metadata.get("actionReturn") or []
        return [InteractionPose.from_thor(row) for row in raw if isinstance(row, dict)]

    def execute(self, action: Action, step_budget: int | None = None) -> ThorExecution:
        if action.kind == ActionKind.NAVIGATE_TO.value:
            return _navigate_to_interactable_pose(self, action, step_budget)
        if action.kind == ActionKind.FACE_TARGET.value:
            return _face_target_for_adapter(self, action, step_budget)
        if action.kind == ActionKind.CROSS_THRESHOLD.value:
            return _cross_threshold_for_adapter(self, action, step_budget)
        if action.kind in _interaction_kinds():
            return _execute_interaction_with_pose_recovery(self, action, step_budget)
        return self._execute_primitive(action)


class ThorAdapter(_AdapterMixin):
    """Boundary around AI2-THOR plus deterministic, pose-aware body mechanics."""

    def __init__(self, config: ThorConfig):
        self.config = config
        self.agent_label = config.agent_label or "solo"
        self._pose_actionability_memory = None
        self.controller = None
        self.step_index = 0
        self.state = None

    def start(self) -> None:
        _install_ai2thor_fifo_timeout_patch(self.config.simulator_start_timeout_seconds)
        from ai2thor.controller import Controller
        kwargs = dict(
            agentMode="default", visibilityDistance=self.config.visibility_distance,
            scene=self.config.scene, gridSize=self.config.grid_size, snapToGrid=True,
            rotateStepDegrees=self.config.rotate_step_degrees, renderDepthImage=False,
            renderInstanceSegmentation=self.config.render_instance_segmentation,
            width=self.config.width, height=self.config.height, fieldOfView=self.config.field_of_view,
        )
        if self.config.local_executable_path:
            kwargs["local_executable_path"] = self.config.local_executable_path
        if self.config.headless:
            from ai2thor.platform import CloudRendering
            kwargs["platform"] = CloudRendering
        self.controller = _call_with_hard_timeout(lambda: Controller(**kwargs), self.config.simulator_start_timeout_seconds, "Controller.start")
        _configure_controller_fifo_timeout(self.controller, self.config.simulator_step_timeout_seconds, "idle")
        self.step_index = 0
        self.state = normalize_event(self.controller.last_event, self.config.scene, self.step_index)

    def close(self) -> None:
        controller = self.controller
        self.controller = None
        if controller is None:
            return
        try:
            controller.stop()
        except Exception as exc:
            if not _is_simulator_transport_error(exc):
                raise

    def observe(self):
        if self.controller is None:
            raise RuntimeError("THOR adapter is not started.")
        self.state = normalize_event(self.controller.last_event, self.config.scene, self.step_index)
        return self.state

    def raw_event(self):
        return None if self.controller is None else self.controller.last_event

    def _query_step(self, action: str, **params):
        if self.controller is None:
            raise RuntimeError("THOR adapter is not started.")
        _configure_controller_fifo_timeout(self.controller, self.config.simulator_step_timeout_seconds, action)
        return _call_with_hard_timeout(
            lambda: self.controller.step(action=action, **params),
            self.config.simulator_step_timeout_seconds,
            action,
        )

    def get_reachable_positions(self) -> list[dict[str, float]]:
        event = self._query_step("GetReachablePositions")
        if event.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(event.metadata.get("errorMessage") or "GetReachablePositions failed.")
        return list(event.metadata.get("actionReturn") or [])

    def _execute_primitive(self, action: Action) -> ThorExecution:
        params = {k: v for k, v in action.params.items() if not str(k).startswith("_")}
        event = self._query_step(action.kind, **params)
        if self.config.step_delay_seconds > 0:
            time.sleep(self.config.step_delay_seconds)
        self.step_index += 1
        self.state = normalize_event(event, self.config.scene, self.step_index)
        return ThorExecution(event, self.state, 1, [action.kind])


class SharedThorWorld:
    """One AI2-THOR Controller/Unity scene shared by multiple embodied agents."""

    def __init__(self, config: ThorConfig, agent_count: int = 2, heartbeat_path: str | Path | None = None):
        if agent_count < 2:
            raise ValueError("SharedThorWorld requires agent_count >= 2.")
        self.config = config
        self.agent_count = int(agent_count)
        self.controller = None
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self.heartbeat_context: dict[str, Any] = {}
        self.restart_count = 0

    def set_heartbeat_context(self, **context: Any) -> None:
        self.heartbeat_context = dict(context)

    def _heartbeat(self, *, status: str, operation: str | None = None, agent_id: int | None = None, error: str | None = None) -> None:
        if self.heartbeat_path is None:
            return
        payload = {
            **self.heartbeat_context,
            "status": status,
            "operation": operation,
            "agent_id": agent_id,
            "restart_count": self.restart_count,
            "timestamp_unix": time.time(),
        }
        if error:
            payload["error"] = str(error)
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.heartbeat_path.with_suffix(self.heartbeat_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.heartbeat_path)

    def start(self) -> None:
        _install_ai2thor_fifo_timeout_patch(self.config.simulator_start_timeout_seconds)
        from ai2thor.controller import Controller
        kwargs = dict(
            agentMode="default", visibilityDistance=self.config.visibility_distance,
            scene=self.config.scene, gridSize=self.config.grid_size, snapToGrid=True,
            rotateStepDegrees=self.config.rotate_step_degrees, renderDepthImage=False,
            renderInstanceSegmentation=self.config.render_instance_segmentation,
            width=self.config.width, height=self.config.height, fieldOfView=self.config.field_of_view,
            agentCount=self.agent_count, makeAgentsVisible=True,
        )
        if self.config.local_executable_path:
            kwargs["local_executable_path"] = self.config.local_executable_path
        if self.config.headless:
            from ai2thor.platform import CloudRendering
            kwargs["platform"] = CloudRendering
        self._heartbeat(status="starting", operation="Controller.start")
        self.controller = _call_with_hard_timeout(lambda: Controller(**kwargs), self.config.simulator_start_timeout_seconds, "Controller.start")
        _configure_controller_fifo_timeout(self.controller, self.config.simulator_step_timeout_seconds, "idle")
        self._heartbeat(status="ready", operation="Controller.start")
        events = getattr(self.controller.last_event, "events", None)
        if events is None or len(events) < self.agent_count:
            self.close()
            raise RuntimeError(f"AI2-THOR did not initialize {self.agent_count} agents in one scene.")

    def close(self) -> None:
        controller = self.controller
        # Detach first so a failed/broken stop can never leave a poisoned controller reusable.
        self.controller = None
        if controller is None:
            return
        try:
            controller.stop()
        except Exception as exc:
            if not _is_simulator_transport_error(exc):
                raise
            self._heartbeat(status="closed_broken_transport", operation="Controller.stop", error=str(exc))

    def restart(self) -> None:
        """Discard a timed-out FIFO/controller and start a fresh two-agent world."""
        self._heartbeat(status="restarting", operation="Controller.restart")
        self.close()
        self.restart_count += 1
        self.start()

    def event_for(self, agent_id: int, event: Any | None = None):
        if self.controller is None:
            raise RuntimeError("Shared THOR world is not started.")
        if not 0 <= int(agent_id) < self.agent_count:
            raise IndexError(f"Invalid agent_id={agent_id}; count={self.agent_count}")
        multi = self.controller.last_event if event is None else event
        events = getattr(multi, "events", None)
        if events is None:
            raise RuntimeError("Expected AI2-THOR Multi-Agent Event with `.events`.")
        return events[int(agent_id)]

    def step_for(self, agent_id: int, action: str, **params):
        if self.controller is None:
            raise RuntimeError("Shared THOR world is not started.")
        aid = int(agent_id)
        self._heartbeat(status="waiting_for_unity", operation=action, agent_id=aid)
        try:
            _configure_controller_fifo_timeout(self.controller, self.config.simulator_step_timeout_seconds, action)
            multi = _call_with_hard_timeout(
                lambda: self.controller.step(action=action, agentId=aid, **params),
                self.config.simulator_step_timeout_seconds,
                action,
            )
        except SimulatorStepTimeout as exc:
            self._heartbeat(status="timeout", operation=action, agent_id=aid, error=str(exc))
            raise
        except Exception as exc:
            if not _is_simulator_transport_error(exc):
                raise
            wrapped = exc if isinstance(exc, SimulatorTransportError) else SimulatorTransportError(action, exc)
            self._heartbeat(status="transport_error", operation=action, agent_id=aid, error=str(wrapped))
            raise wrapped from exc
        self._heartbeat(status="unity_replied", operation=action, agent_id=aid)
        return self.event_for(aid, multi)

    def agent(self, agent_id: int) -> "ThorAgentAdapter":
        return ThorAgentAdapter(self, int(agent_id))


class ThorAgentAdapter(_AdapterMixin):
    def __init__(self, world: SharedThorWorld, agent_id: int):
        if not 0 <= agent_id < world.agent_count:
            raise IndexError(f"Invalid agent_id={agent_id}; count={world.agent_count}")
        self.world = world
        self.agent_id = int(agent_id)
        self.config = replace(world.config)
        self.agent_label = chr(ord("A") + self.agent_id) if self.agent_id < 26 else f"agent_{self.agent_id}"
        self._pose_actionability_memory = None
        self.step_index = 0
        self.state = None

    @property
    def controller(self):
        return self.world.controller

    def start(self) -> None:
        if self.world.controller is None:
            raise RuntimeError("Start SharedThorWorld before using agent adapters.")
        self.step_index = 0
        self.state = self.observe()

    def close(self) -> None:
        return None

    def observe(self):
        event = self.world.event_for(self.agent_id)
        self.state = normalize_event(event, self.config.scene, self.step_index)
        return self.state

    def raw_event(self):
        return None if self.world.controller is None else self.world.event_for(self.agent_id)

    def _query_step(self, action: str, **params):
        return self.world.step_for(self.agent_id, action, **params)

    def get_reachable_positions(self) -> list[dict[str, float]]:
        event = self._query_step("GetReachablePositions")
        if event.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(event.metadata.get("errorMessage") or "GetReachablePositions failed.")
        return list(event.metadata.get("actionReturn") or [])

    def _execute_primitive(self, action: Action) -> ThorExecution:
        params = {k: v for k, v in action.params.items() if not str(k).startswith("_")}
        event = self._query_step(action.kind, **params)
        if self.config.step_delay_seconds > 0:
            time.sleep(self.config.step_delay_seconds)
        self.step_index += 1
        self.state = normalize_event(event, self.config.scene, self.step_index)
        return ThorExecution(event, self.state, 1, [action.kind])
