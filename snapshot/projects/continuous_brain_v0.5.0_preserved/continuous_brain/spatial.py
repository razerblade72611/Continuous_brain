from __future__ import annotations

from dataclasses import dataclass, asdict
from heapq import heappop, heappush
import math
from typing import Any

import numpy as np


@dataclass(slots=True)
class PoseBelief2D:
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    updates: int = 0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.z], dtype=np.float32)


class SpatialMemory2D:
    """Persistent metric occupancy memory for the embodied branch.

    This intentionally starts simple and inspectable.  The map stores log-odds-like
    occupancy evidence on a fixed metric grid.  Depth updates mark free cells along
    rays and occupied cells at trustworthy endpoints.  Unknown cells stay traversable
    for planning, but with an extra cost so the planner prefers observed free space.

    v0.5 uses simulator pose metadata as a *pose-belief bootstrap*, not as a route
    teacher.  The class is deliberately isolated so future releases can replace the
    pose source with visual/depth odometry or SLAM without changing the brain API.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        resolution: float = 0.25,
        unknown_cost: float = 2.5,
        occupied_threshold: float = 0.80,
        free_threshold: float = -0.25,
        max_cells: int = 200_000,
    ) -> None:
        self.resolution = max(0.05, float(resolution))
        self.unknown_cost = max(1.0, float(unknown_cost))
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        self.max_cells = max(256, int(max_cells))
        self.bounds = [-5.0, 5.0, -5.0, 5.0]
        self.log_odds = np.zeros((1, 1), dtype=np.float32)
        self.visits = np.zeros((1, 1), dtype=np.uint16)
        self.pose = PoseBelief2D()
        self.map_updates = 0
        self.rays_integrated = 0
        self.plans = 0
        self.plan_failures = 0
        self.landmarks: dict[str, dict[str, Any]] = {}
        self._resize(self.bounds)

    # ---------- grid geometry ----------
    def _shape_for_bounds(self, bounds: list[float]) -> tuple[int, int]:
        xmin, xmax, zmin, zmax = [float(v) for v in bounds]
        nx = max(3, int(math.ceil((xmax - xmin) / self.resolution)) + 1)
        nz = max(3, int(math.ceil((zmax - zmin) / self.resolution)) + 1)
        if nx * nz > self.max_cells:
            scale = math.sqrt((nx * nz) / float(self.max_cells))
            self.resolution *= max(1.0, scale)
            nx = max(3, int(math.ceil((xmax - xmin) / self.resolution)) + 1)
            nz = max(3, int(math.ceil((zmax - zmin) / self.resolution)) + 1)
        return nz, nx

    def _resize(self, bounds: list[float]) -> None:
        xmin, xmax, zmin, zmax = [float(v) for v in bounds]
        if xmax <= xmin or zmax <= zmin:
            return
        self.bounds = [xmin, xmax, zmin, zmax]
        shape = self._shape_for_bounds(self.bounds)
        self.log_odds = np.zeros(shape, dtype=np.float32)
        self.visits = np.zeros(shape, dtype=np.uint16)

    def ensure_bounds(self, bounds: list[float], *, preserve: bool = True) -> None:
        requested = [float(v) for v in bounds]
        if len(requested) != 4 or requested[1] <= requested[0] or requested[3] <= requested[2]:
            return
        old_bounds = list(self.bounds)
        old_resolution = float(self.resolution)
        old_log = self.log_odds.copy()
        old_visits = self.visits.copy()
        if all(abs(a - b) < self.resolution * 0.25 for a, b in zip(requested, old_bounds)):
            return
        self._resize(requested)
        if not preserve or old_log.size <= 1:
            return
        # Reproject old cells by metric center.  This is slow only during scene/bounds
        # changes and keeps map persistence robust to small executable differences.
        for iz in range(old_log.shape[0]):
            for ix in range(old_log.shape[1]):
                if abs(float(old_log[iz, ix])) < 1e-6 and int(old_visits[iz, ix]) == 0:
                    continue
                x = old_bounds[0] + ix * old_resolution
                z = old_bounds[2] + iz * old_resolution
                cell = self.world_to_cell(x, z)
                if cell is None:
                    continue
                nz, nx = cell
                self.log_odds[nz, nx] = old_log[iz, ix]
                self.visits[nz, nx] = old_visits[iz, ix]

    def world_to_cell(self, x: float, z: float) -> tuple[int, int] | None:
        xmin, xmax, zmin, zmax = self.bounds
        if x < xmin - 1e-6 or x > xmax + 1e-6 or z < zmin - 1e-6 or z > zmax + 1e-6:
            return None
        ix = int(round((float(x) - xmin) / self.resolution))
        iz = int(round((float(z) - zmin) / self.resolution))
        if 0 <= iz < self.log_odds.shape[0] and 0 <= ix < self.log_odds.shape[1]:
            return iz, ix
        return None

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        iz, ix = cell
        return np.array([
            self.bounds[0] + ix * self.resolution,
            self.bounds[2] + iz * self.resolution,
        ], dtype=np.float32)

    # ---------- evidence ----------
    def observe_pose(self, x: float, z: float, yaw: float, confidence: float = 1.0) -> None:
        self.pose = PoseBelief2D(float(x), float(z), float(yaw) % 360.0, float(np.clip(confidence, 0.0, 1.0)), self.pose.updates + 1)
        cell = self.world_to_cell(x, z)
        if cell is not None:
            iz, ix = cell
            self.visits[iz, ix] = np.uint16(min(65535, int(self.visits[iz, ix]) + 1))
            self.log_odds[iz, ix] = min(float(self.log_odds[iz, ix]), -0.75)

    def _add_evidence(self, cell: tuple[int, int] | None, amount: float) -> None:
        if cell is None:
            return
        iz, ix = cell
        self.log_odds[iz, ix] = float(np.clip(float(self.log_odds[iz, ix]) + amount, -4.0, 4.0))

    def mark_free_world(self, x: float, z: float, amount: float = -0.18) -> None:
        self._add_evidence(self.world_to_cell(x, z), -abs(float(amount)))

    def mark_occupied_world(self, x: float, z: float, amount: float = 0.95) -> None:
        self._add_evidence(self.world_to_cell(x, z), abs(float(amount)))

    def mark_blocked_ahead(self, distance: float | None = None) -> None:
        d = self.resolution if distance is None else max(self.resolution * 0.75, float(distance))
        a = math.radians(self.pose.yaw)
        x = self.pose.x + math.sin(a) * d
        z = self.pose.z + math.cos(a) * d
        cell = self.world_to_cell(x, z)
        if cell is not None:
            # Native-physics collision is stronger evidence than a depth free-space
            # estimate. Force the cell occupied so stale/self-occluded depth cannot
            # immediately erase a confirmed blocked translation.
            self.log_odds[cell] = max(float(self.log_odds[cell]), 2.0)

    def integrate_depth(
        self,
        depth: np.ndarray | None,
        *,
        fov_degrees: float,
        max_range: float = 5.0,
        rays: int = 31,
        min_depth: float = 0.25,
    ) -> int:
        if depth is None:
            return 0
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2 or d.size == 0:
            return 0
        d = np.nan_to_num(d, nan=max_range, posinf=max_range, neginf=0.0)
        h, w = d.shape
        # Use an upper-middle band and a high percentile to reduce floor/body
        # self-occlusion.  The Discovery camera previously reported ~0.14 m center
        # medians while MoveAhead was physically valid; depths below min_depth are
        # therefore treated as untrusted rather than mapped as walls.
        y0, y1 = max(0, int(h * 0.30)), min(h, max(int(h * 0.58), int(h * 0.30) + 1))
        band = d[y0:y1, :]
        ray_count = max(5, min(int(rays), w))
        columns = np.linspace(0, w - 1, ray_count).astype(int)
        integrated = 0
        step = max(self.resolution * 0.45, 0.05)
        for col in columns:
            vals = band[:, col]
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= float(min_depth)) & (vals > 0.0)]
            if vals.size == 0:
                continue
            dist = float(np.percentile(vals, 70.0))
            dist = float(np.clip(dist, min_depth, max_range))
            frac = (float(col) / max(1.0, float(w - 1))) - 0.5
            ray_yaw = self.pose.yaw + frac * float(fov_degrees)
            angle = math.radians(ray_yaw)
            free_until = max(0.0, dist - self.resolution * 0.6)
            s = step
            while s <= free_until:
                x = self.pose.x + math.sin(angle) * s
                z = self.pose.z + math.cos(angle) * s
                self.mark_free_world(x, z)
                s += step
            # Only mark an endpoint occupied if the sensor ended well before its
            # maximum range. Unknown/max-range rays create free-space evidence only.
            if dist < max_range * 0.96:
                x = self.pose.x + math.sin(angle) * dist
                z = self.pose.z + math.cos(angle) * dist
                self.mark_occupied_world(x, z, amount=0.55)
            integrated += 1
        if integrated:
            self.map_updates += 1
            self.rays_integrated += integrated
        return integrated

    def remember_landmarks(self, objects: list[dict[str, Any]]) -> None:
        for obj in objects:
            if not bool(obj.get("visible", False)):
                continue
            p = obj.get("position") or {}
            if "x" not in p or "z" not in p:
                continue
            oid = str(obj.get("objectId") or obj.get("name") or obj.get("objectType") or "object")
            self.landmarks[oid] = {
                "object_id": oid,
                "object_type": str(obj.get("objectType", "object")),
                "x": float(p.get("x", 0.0)),
                "z": float(p.get("z", 0.0)),
                "seen": int(self.landmarks.get(oid, {}).get("seen", 0)) + 1,
                "pose_updates": int(self.pose.updates),
            }

    # ---------- planning ----------
    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        iz, ix = cell
        out = []
        for nz, nx in ((iz + 1, ix), (iz - 1, ix), (iz, ix + 1), (iz, ix - 1)):
            if 0 <= nz < self.log_odds.shape[0] and 0 <= nx < self.log_odds.shape[1]:
                if float(self.log_odds[nz, nx]) < self.occupied_threshold:
                    out.append((nz, nx))
        return out

    def _cell_cost(self, cell: tuple[int, int]) -> float:
        v = float(self.log_odds[cell])
        if v >= self.occupied_threshold:
            return float("inf")
        if v <= self.free_threshold:
            return 1.0
        # Unknown/weakly observed space is allowed but slightly discouraged.
        return self.unknown_cost

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_traversable(self, goal: tuple[int, int], radius: int = 4) -> tuple[int, int] | None:
        if self._cell_cost(goal) < float("inf"):
            return goal
        gz, gx = goal
        choices: list[tuple[float, tuple[int, int]]] = []
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                q = (gz + dz, gx + dx)
                if not (0 <= q[0] < self.log_odds.shape[0] and 0 <= q[1] < self.log_odds.shape[1]):
                    continue
                c = self._cell_cost(q)
                if c < float("inf"):
                    choices.append((self._heuristic(q, goal) + 0.1 * c, q))
        return min(choices, default=(0.0, None), key=lambda x: x[0])[1]

    def plan(self, start_world: np.ndarray, goal_world: np.ndarray) -> list[np.ndarray]:
        self.plans += 1
        start = self.world_to_cell(float(start_world[0]), float(start_world[1]))
        goal = self.world_to_cell(float(goal_world[0]), float(goal_world[1]))
        if start is None or goal is None:
            self.plan_failures += 1
            return []
        goal = self._nearest_traversable(goal) or goal
        if start == goal:
            return [self.cell_to_world(goal)]
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        gscore: dict[tuple[int, int], float] = {start: 0.0}
        found = False
        while frontier:
            _score, cur = heappop(frontier)
            if cur == goal:
                found = True
                break
            for nxt in self._neighbors(cur):
                cost = self._cell_cost(nxt)
                ng = gscore[cur] + cost
                if ng >= gscore.get(nxt, float("inf")):
                    continue
                gscore[nxt] = ng
                came[nxt] = cur
                heappush(frontier, (ng + self._heuristic(nxt, goal), nxt))
        if not found:
            self.plan_failures += 1
            return []
        cells: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal
        while cur is not None and cur != start:
            cells.append(cur)
            cur = came.get(cur)
        cells.reverse()
        return [self.cell_to_world(c) for c in cells]

    def choose_waypoint(self, path: list[np.ndarray], *, lookahead_cells: int = 2) -> np.ndarray | None:
        if not path:
            return None
        idx = min(len(path) - 1, max(0, int(lookahead_cells) - 1))
        return np.asarray(path[idx], dtype=np.float32).copy()

    # ---------- diagnostics / persistence ----------
    def counts(self) -> dict[str, int]:
        known = np.abs(self.log_odds) >= 0.15
        occupied = self.log_odds >= self.occupied_threshold
        free = self.log_odds <= self.free_threshold
        return {
            "cells": int(self.log_odds.size),
            "known": int(np.count_nonzero(known)),
            "free": int(np.count_nonzero(free)),
            "occupied": int(np.count_nonzero(occupied)),
            "visited": int(np.count_nonzero(self.visits)),
            "landmarks": int(len(self.landmarks)),
            "map_updates": int(self.map_updates),
            "rays_integrated": int(self.rays_integrated),
            "plans": int(self.plans),
            "plan_failures": int(self.plan_failures),
        }

    def compact_snapshot(self, max_side: int = 64) -> dict[str, Any]:
        arr = self.log_odds
        step_z = max(1, int(math.ceil(arr.shape[0] / max_side)))
        step_x = max(1, int(math.ceil(arr.shape[1] / max_side)))
        sampled = arr[::step_z, ::step_x]
        # Quantized values make the dashboard payload small and JSON-safe.
        q = np.clip(np.rint(sampled * 25.0), -100, 100).astype(np.int16)
        return {
            "bounds": [float(v) for v in self.bounds],
            "resolution": float(self.resolution * max(step_x, step_z)),
            "shape": [int(q.shape[0]), int(q.shape[1])],
            "values": q.reshape(-1).astype(int).tolist(),
            "pose": asdict(self.pose),
            "counts": self.counts(),
        }

    def state_dict(self, *, scene: str | None = None) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "scene": scene,
            "resolution": float(self.resolution),
            "unknown_cost": float(self.unknown_cost),
            "occupied_threshold": float(self.occupied_threshold),
            "free_threshold": float(self.free_threshold),
            "bounds": [float(v) for v in self.bounds],
            "log_odds": self.log_odds.astype(np.float16),
            "visits": self.visits.astype(np.uint16),
            "pose": asdict(self.pose),
            "map_updates": int(self.map_updates),
            "rays_integrated": int(self.rays_integrated),
            "plans": int(self.plans),
            "plan_failures": int(self.plan_failures),
            "landmarks": dict(self.landmarks),
        }

    def load_state_dict(self, state: dict[str, Any], *, scene: str | None = None) -> bool:
        if not isinstance(state, dict):
            return False
        saved_scene = state.get("scene")
        if scene and saved_scene and str(scene) != str(saved_scene):
            return False
        try:
            bounds = [float(v) for v in state.get("bounds", self.bounds)]
            log = np.asarray(state.get("log_odds"), dtype=np.float32)
            visits = np.asarray(state.get("visits"), dtype=np.uint16)
            if log.ndim != 2 or visits.shape != log.shape or log.size > self.max_cells:
                return False
            self.resolution = max(0.05, float(state.get("resolution", self.resolution)))
            self.unknown_cost = max(1.0, float(state.get("unknown_cost", self.unknown_cost)))
            self.occupied_threshold = float(state.get("occupied_threshold", self.occupied_threshold))
            self.free_threshold = float(state.get("free_threshold", self.free_threshold))
            self.bounds = bounds
            self.log_odds = np.clip(log, -4.0, 4.0).astype(np.float32)
            self.visits = visits.astype(np.uint16)
            p = dict(state.get("pose", {}))
            self.pose = PoseBelief2D(
                float(p.get("x", 0.0)), float(p.get("z", 0.0)), float(p.get("yaw", 0.0)),
                float(p.get("confidence", 0.0)), int(p.get("updates", 0)),
            )
            self.map_updates = int(state.get("map_updates", 0))
            self.rays_integrated = int(state.get("rays_integrated", 0))
            self.plans = int(state.get("plans", 0))
            self.plan_failures = int(state.get("plan_failures", 0))
            self.landmarks = {str(k): dict(v) for k, v in dict(state.get("landmarks", {})).items() if isinstance(v, dict)}
            return True
        except Exception:
            return False
