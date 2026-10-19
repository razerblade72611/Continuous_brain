from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

from .learning import load_json, utc_now, write_json_atomic


def planar_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"]))


@dataclass
class MapCell:
    x: float
    y: float
    z: float
    visits: int = 0
    successes: int = 0
    failures: int = 0

    def position(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


class PersistentWorldMap:
    """Experience-derived grid/topological map.

    Only positions actually visited (and transitions actually attempted) are stored.
    The detailed map is consumed by the motor layer, not serialized into LLM prompts.
    """

    def __init__(self, scene: str, grid_size: float = 0.25, path: str | Path | None = None):
        self.scene = str(scene)
        self.grid_size = float(grid_size)
        self.path = None if path is None else Path(path)
        self.cells: dict[str, MapCell] = {}
        self.edges: dict[str, dict[str, int]] = {}
        self.updated_at: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    def _key(self, position: dict[str, float]) -> str:
        ix = round(float(position["x"]) / self.grid_size)
        iz = round(float(position["z"]) / self.grid_size)
        return f"{ix},{iz}"

    def _edge_key(self, source: str, target: str) -> str:
        return f"{source}>{target}"

    def _load(self) -> None:
        payload = load_json(self.path, {})
        if not payload:
            return
        self.scene = str(payload.get("scene", self.scene))
        self.grid_size = float(payload.get("grid_size", self.grid_size))
        for key, raw in (payload.get("cells") or {}).items():
            if not isinstance(raw, dict):
                continue
            self.cells[str(key)] = MapCell(
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                z=float(raw.get("z", 0.0)),
                visits=int(raw.get("visits", 0)),
                successes=int(raw.get("successes", 0)),
                failures=int(raw.get("failures", 0)),
            )
        self.edges = {
            str(key): {
                "successes": int((raw or {}).get("successes", 0)),
                "failures": int((raw or {}).get("failures", 0)),
            }
            for key, raw in (payload.get("edges") or {}).items()
            if isinstance(raw, dict)
        }
        self.updated_at = payload.get("updated_at")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "scene": self.scene,
            "grid_size": self.grid_size,
            "updated_at": self.updated_at or utc_now(),
            "cells": {
                key: {
                    "x": cell.x,
                    "y": cell.y,
                    "z": cell.z,
                    "visits": cell.visits,
                    "successes": cell.successes,
                    "failures": cell.failures,
                }
                for key, cell in sorted(self.cells.items())
            },
            "edges": dict(sorted(self.edges.items())),
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("PersistentWorldMap.save requires a path.")
        self.updated_at = utc_now()
        write_json_atomic(target, self.as_dict())
        self.path = target

    def record_pose(self, position: dict[str, float], success: bool | None = None) -> str:
        key = self._key(position)
        cell = self.cells.get(key)
        if cell is None:
            cell = MapCell(
                x=float(position["x"]),
                y=float(position.get("y", 0.0)),
                z=float(position["z"]),
            )
            self.cells[key] = cell
        cell.visits += 1
        if success is True:
            cell.successes += 1
        elif success is False:
            cell.failures += 1
        return key

    def record_transition(
        self,
        before: dict[str, float],
        after: dict[str, float],
        success: bool,
    ) -> None:
        source = self.record_pose(before)
        target = self.record_pose(after, success=success)
        moved = planar_distance(before, after) >= max(0.01, self.grid_size * 0.2)
        edge_key = self._edge_key(source, target)
        stats = self.edges.setdefault(edge_key, {"successes": 0, "failures": 0})
        if success and moved:
            stats["successes"] += 1
            # Successful movement is traversable both ways for the default grid body.
            reverse = self.edges.setdefault(self._edge_key(target, source), {"successes": 0, "failures": 0})
            reverse["successes"] += 1
        else:
            stats["failures"] += 1

    def merge(self, other: "PersistentWorldMap") -> None:
        if abs(self.grid_size - other.grid_size) > 1e-9:
            raise ValueError("Cannot merge maps with different grid sizes.")
        for key, cell in other.cells.items():
            existing = self.cells.get(key)
            if existing is None:
                self.cells[key] = MapCell(**cell.__dict__)
                continue
            total_visits = existing.visits + cell.visits
            if total_visits > 0:
                existing.x = (existing.x * existing.visits + cell.x * cell.visits) / total_visits
                existing.y = (existing.y * existing.visits + cell.y * cell.visits) / total_visits
                existing.z = (existing.z * existing.visits + cell.z * cell.visits) / total_visits
            existing.visits = total_visits
            existing.successes += cell.successes
            existing.failures += cell.failures
        for key, stats in other.edges.items():
            current = self.edges.setdefault(key, {"successes": 0, "failures": 0})
            current["successes"] += int(stats.get("successes", 0))
            current["failures"] += int(stats.get("failures", 0))

    def _nearest_key(self, position: dict[str, float]) -> str | None:
        if not self.cells:
            return None
        return min(self.cells, key=lambda key: planar_distance(position, self.cells[key].position()))

    def route(
        self,
        start_position: dict[str, float],
        target_position: dict[str, float],
        interaction_distance: float = 0.35,
        min_edge_successes: int = 1,
    ) -> list[dict[str, float]]:
        if not self.cells:
            return []
        start = self._nearest_key(start_position)
        if start is None:
            return []
        # Do not "snap" a live agent to a distant learned component. If the
        # current pose is outside learned coverage, let AI2-THOR's deterministic
        # reachable-position planner handle the route and preserve correctness.
        if planar_distance(self.cells[start].position(), start_position) > self.grid_size * 1.5:
            return []
        target_candidates = sorted(
            self.cells,
            key=lambda key: planar_distance(self.cells[key].position(), target_position),
        )
        goals = {
            key
            for key in target_candidates
            if planar_distance(self.cells[key].position(), target_position) <= float(interaction_distance)
        }
        if not goals:
            # The learned map does not yet reach this target. Falling back is safer
            # than pretending the nearest remembered cell is an arrival location.
            return []

        adjacency: dict[str, list[str]] = {}
        for edge, stats in self.edges.items():
            if int(stats.get("successes", 0)) < int(min_edge_successes):
                continue
            success = int(stats.get("successes", 0))
            failure = int(stats.get("failures", 0))
            if success < failure:
                continue
            source, sep, target = edge.partition(">")
            if not sep or source not in self.cells or target not in self.cells:
                continue
            adjacency.setdefault(source, []).append(target)

        queue = deque([start])
        parent: dict[str, str | None] = {start: None}
        found: str | None = None
        while queue:
            current = queue.popleft()
            if current in goals:
                found = current
                break
            for neighbor in adjacency.get(current, []):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if found is None:
            return []

        keys: list[str] = []
        cursor: str | None = found
        while cursor is not None:
            keys.append(cursor)
            cursor = parent[cursor]
        keys.reverse()
        return [self.cells[key].position() for key in keys]

    def coverage(self, reachable_count: int | None = None) -> dict[str, Any]:
        unique = len(self.cells)
        payload: dict[str, Any] = {"unique_cells": unique}
        if reachable_count and reachable_count > 0:
            payload["reachable_cells"] = int(reachable_count)
            payload["coverage_pct"] = round(min(100.0, unique * 100.0 / int(reachable_count)), 2)
        return payload
