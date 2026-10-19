from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

from .models import Action, ActionKind


@dataclass(frozen=True)
class NavigationConfig:
    grid_size: float = 0.25
    rotate_step_degrees: float = 90.0
    interaction_distance: float = 1.25


def planar_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"]))


def _grid_key(position: dict[str, float], grid_size: float) -> tuple[int, int]:
    return (
        round(float(position["x"]) / grid_size),
        round(float(position["z"]) / grid_size),
    )


def _nearest_key(
    position: dict[str, float],
    positions: dict[tuple[int, int], dict[str, float]],
) -> tuple[int, int]:
    return min(positions, key=lambda key: planar_distance(position, positions[key]))


def shortest_grid_path(
    reachable_positions: Iterable[dict[str, float]],
    start_position: dict[str, float],
    target_position: dict[str, float],
    config: NavigationConfig,
) -> list[dict[str, float]]:
    """Return a shortest grid path to the reachable point nearest the target."""
    positions = {
        _grid_key(p, config.grid_size): {
            "x": float(p["x"]),
            "y": float(p.get("y", start_position.get("y", 0.0))),
            "z": float(p["z"]),
        }
        for p in reachable_positions
    }
    if not positions:
        raise ValueError("AI2-THOR returned no reachable positions.")

    start = _nearest_key(start_position, positions)

    candidates = sorted(
        positions,
        key=lambda key: planar_distance(positions[key], target_position),
    )
    within_range = [
        key
        for key in candidates
        if planar_distance(positions[key], target_position) <= config.interaction_distance
    ]
    goals = set(within_range or candidates[:1])

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found: tuple[int, int] | None = None

    while queue:
        current = queue.popleft()
        if current in goals:
            found = current
            break
        x, z = current
        for neighbor in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            if neighbor in positions and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    if found is None:
        raise ValueError("No connected path to a reachable position near the target.")

    keys: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = found
    while cursor is not None:
        keys.append(cursor)
        cursor = parent[cursor]
    keys.reverse()
    return [positions[key] for key in keys]


def normalize_yaw(yaw: float) -> float:
    return yaw % 360.0


def cardinal_yaw(dx: float, dz: float) -> float:
    # AI2-THOR convention: 0=+z, 90=+x, 180=-z, 270=-x.
    if abs(dx) > abs(dz):
        return 90.0 if dx > 0 else 270.0
    return 0.0 if dz > 0 else 180.0


def facing_yaw(source: dict[str, float], target: dict[str, float]) -> float:
    dx = float(target["x"]) - float(source["x"])
    dz = float(target["z"]) - float(source["z"])
    return normalize_yaw(math.degrees(math.atan2(dx, dz)))


def rotation_actions(current_yaw: float, desired_yaw: float, step_degrees: float = 90.0) -> list[Action]:
    current = round(normalize_yaw(current_yaw) / step_degrees) * step_degrees
    desired = round(normalize_yaw(desired_yaw) / step_degrees) * step_degrees
    current = normalize_yaw(current)
    desired = normalize_yaw(desired)

    delta = (desired - current) % 360.0
    right_steps = round(delta / step_degrees)
    left_steps = round((360.0 - delta) / step_degrees) if delta else 0

    if right_steps <= left_steps:
        return [
            Action(ActionKind.ROTATE_RIGHT.value, {}, "Turn toward navigation waypoint.")
            for _ in range(right_steps)
        ]
    return [
        Action(ActionKind.ROTATE_LEFT.value, {}, "Turn toward navigation waypoint.")
        for _ in range(left_steps)
    ]


def path_to_actions(
    path: list[dict[str, float]],
    initial_yaw: float,
    target_position: dict[str, float],
    config: NavigationConfig,
    *,
    face_target: bool = True,
) -> list[Action]:
    if not path:
        return []

    actions: list[Action] = []
    yaw = normalize_yaw(initial_yaw)

    for previous, current in zip(path, path[1:]):
        dx = current["x"] - previous["x"]
        dz = current["z"] - previous["z"]
        desired = cardinal_yaw(dx, dz)
        turns = rotation_actions(yaw, desired, config.rotate_step_degrees)
        actions.extend(turns)
        yaw = desired
        actions.append(Action(ActionKind.MOVE_AHEAD.value, {}, "Follow shortest reachable-position path."))

    if face_target:
        final_yaw = facing_yaw(path[-1], target_position)
        turns = rotation_actions(yaw, final_yaw, config.rotate_step_degrees)
        actions.extend(turns)
    return actions
