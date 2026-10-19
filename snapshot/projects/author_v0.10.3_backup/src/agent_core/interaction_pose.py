from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

from .navigation import NavigationConfig, planar_distance, shortest_grid_path


@dataclass(frozen=True, slots=True)
class InteractionPose:
    """Controller-private body pose returned by AI2-THOR GetInteractablePoses."""

    x: float
    y: float
    z: float
    rotation: float
    horizon: float
    standing: bool = True

    @property
    def position(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_thor(cls, raw: dict[str, Any]) -> "InteractionPose":
        rotation = raw.get("rotation", 0.0)
        if isinstance(rotation, dict):
            rotation = rotation.get("y", 0.0)
        return cls(
            x=float(raw["x"]),
            y=float(raw.get("y", 0.9)),
            z=float(raw["z"]),
            rotation=float(rotation) % 360.0,
            horizon=max(-30.0, min(60.0, float(raw.get("horizon", 0.0)))),
            standing=bool(raw.get("standing", True)),
        )

    def position_key(self, grid_size: float = 0.25) -> tuple[int, int]:
        return (round(self.x / grid_size), round(self.z / grid_size))

    def key(self, grid_size: float = 0.25) -> tuple[int, int, int, int]:
        return (
            round(self.x / grid_size),
            round(self.z / grid_size),
            round(self.rotation) % 360,
            round(self.horizon),
        )


def angular_error(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def choose_best_interaction_pose(
    poses: Iterable[InteractionPose],
    reachable: list[dict[str, float]],
    current_position: dict[str, float],
    current_yaw: float,
    current_horizon: float,
    nav_config: NavigationConfig,
    *,
    excluded_keys: set[tuple[int, int, int, int]] | None = None,
    excluded_position_keys: set[tuple[int, int]] | None = None,
    pose_penalty: Callable[[InteractionPose], float] | None = None,
    pose_filter: Callable[[InteractionPose], bool] | None = None,
    move_cost_weight: float = 10.0,
    turn_cost_weight: float = 1.0,
    look_cost_weight: float = 0.5,
) -> InteractionPose | None:
    """Choose a connected pose by real grid-path cost, then orientation effort.

    GetInteractablePoses is already restricted to the current reachable component,
    but path validation is repeated here so a malformed/custom scene cannot turn a
    disconnected pose into a scored navigation failure.
    """

    excluded = excluded_keys or set()
    excluded_positions = excluded_position_keys or set()
    best: tuple[float, InteractionPose] | None = None
    exact_cfg = NavigationConfig(
        grid_size=nav_config.grid_size,
        rotate_step_degrees=nav_config.rotate_step_degrees,
        interaction_distance=max(0.01, nav_config.grid_size * 0.35),
    )
    for pose in poses:
        if pose.key(nav_config.grid_size) in excluded:
            continue
        if pose.position_key(nav_config.grid_size) in excluded_positions:
            continue
        if pose_filter is not None and not pose_filter(pose):
            continue
        try:
            path = shortest_grid_path(reachable, current_position, pose.position, exact_cfg)
        except ValueError:
            continue
        end = path[-1] if path else current_position
        if planar_distance(end, pose.position) > max(nav_config.grid_size * 0.55, 0.15):
            continue
        move_cost = max(0, len(path) - 1)
        turn_cost = angular_error(current_yaw, pose.rotation) / max(1.0, nav_config.rotate_step_degrees)
        look_cost = abs(float(current_horizon) - pose.horizon) / 30.0
        # Translation dominates; orientation is only a tie-breaker/secondary cost.
        score = move_cost * float(move_cost_weight) + turn_cost * float(turn_cost_weight) + look_cost * float(look_cost_weight)
        if pose_penalty is not None:
            score += float(pose_penalty(pose))
        if best is None or score < best[0]:
            best = (score, pose)
    return None if best is None else best[1]


def forward_point(pose: InteractionPose, distance: float = 1.0) -> dict[str, float]:
    radians = math.radians(pose.rotation)
    return {
        "x": pose.x + math.sin(radians) * distance,
        "y": pose.y,
        "z": pose.z + math.cos(radians) * distance,
    }
