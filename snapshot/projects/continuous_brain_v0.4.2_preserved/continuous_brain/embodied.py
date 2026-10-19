from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import base64
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .environment import Environment


@dataclass(slots=True)
class CurriculumTarget:
    level: int
    kind: str
    label: str
    position: np.ndarray
    object_id: str | None = None
    stage: int = 0
    start_position: np.ndarray | None = None
    crossed_position: np.ndarray | None = None
    waypoints: list[np.ndarray] | None = None
    waypoint_index: int = 0


@dataclass(slots=True)
class AI2ThorConfig:
    scene: str = "FloorPlan1_physics"
    executable: str | None = None
    width: int = 320
    height: int = 240
    fov: float = 90.0
    grid_size: float = 0.25
    rotate_degrees: float = 30.0
    max_episode_steps: int = 300
    curriculum_level: int = 1
    goal_signal: str = "teacher-nav"  # teacher-nav | sensor-only
    goal_dropout: float = 0.0
    random_start: bool = True
    seed: int = 7
    safety_distance: float = 0.20
    action_noise: float = 0.0
    orient_success_degrees: float = 15.0
    orient_progress_scale: float = 0.008
    orient_correct_turn_bonus: float = 0.02
    orient_wrong_turn_penalty: float = 0.03
    stagnation_steps: int = 60
    stagnation_epsilon: float = 0.5
    graduation_window: int = 100
    graduation_success_rate: float = 0.95
    graduation_median_steps: float = 6.0
    graduation_failure_rate: float = 0.02

    # v0.4.2 hierarchical navigation curriculum.  L1 starts with short teacher
    # paths, preserves angular shaping from L0, and expands path length as recent
    # success improves.
    nav_angular_scale: float = 0.003
    nav_distance_scale: float = 0.80
    nav_forward_bonus: float = 0.02
    nav_blocked_penalty: float = 0.08
    nav_wrong_way_penalty: float = 0.04
    nav_progress_distance_epsilon: float = 0.01
    nav_progress_angle_epsilon: float = 2.0
    nav_initial_path_steps: int = 2
    nav_max_path_steps: int = 8
    waypoint_radius: float = 0.18
    rehearsal_fraction: float = 0.15
    level_epsilon_start: float = 0.35
    level_epsilon_end: float = 0.08
    level_epsilon_decay_steps: int = 5000


class AI2ThorEmbodiedEnv(Environment):
    """AI2-THOR adapter that preserves the v0.3 brain's 6x5 interface.

    Brain observation slots (all normalized):
      0 goal-forward component, 1 goal-right component,
      2 center depth clearance, 3 left-minus-right clearance,
      4 previous simulator action success, 5 previous extrinsic reward.

    The first two slots can be masked with ``goal_signal=sensor-only`` or gradually
    dropped during training. This creates an explicit curriculum from privileged
    point-goal supervision toward perception/memory-driven navigation.

    Brain action intents remain five-dimensional for checkpoint compatibility:
      idle, advance, turn_left, turn_right, interact.
    The adapter translates those high-level intents to simulator primitives.
    """

    observation_dim = 6
    action_dim = 5
    action_names = ("idle", "advance", "turn_left", "turn_right", "interact")

    def __init__(
        self,
        config: AI2ThorConfig,
        controller_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.energy = 1.0
        self.last_reward = 0.0
        self.last_action_success = 1.0
        self.steps = 0
        self.completed_subgoals = 0
        self._closed = False
        self._reachable: list[dict[str, float]] = []
        self._reachable_index: dict[tuple[int, int], np.ndarray] = {}
        self._bounds = [-1.0, 1.0, -1.0, 1.0]
        self.target: CurriculumTarget | None = None
        self._l0_anchor: dict[str, float] | None = None
        self._best_progress_metric = float("inf")
        self._stagnation_count = 0
        self._episode_action_counts = [0] * self.action_dim

        # v0.4.2: curriculum evidence is level-local.  This prevents mastered L0
        # episodes from making a brand-new L1 policy look successful and lets us
        # rehearse old skills without contaminating current-level graduation.
        self._recent_outcomes_by_level: dict[int, deque[dict[str, Any]]] = {}
        self._level_totals: dict[int, dict[str, int]] = {}
        self._level_action_steps: dict[int, int] = {}
        self._legacy_v041_stats: dict[str, Any] = {}

        if controller_factory is None:
            try:
                from ai2thor.controller import Controller
            except ImportError as exc:
                raise RuntimeError(
                    "AI2-THOR environment requested but ai2thor is not installed. "
                    "Install the Python package in this venv or run from an environment "
                    "that already contains your AI2-THOR fork."
                ) from exc
            controller_factory = Controller

        kwargs: dict[str, Any] = {
            "scene": config.scene,
            "width": int(config.width),
            "height": int(config.height),
            "fieldOfView": float(config.fov),
            "renderDepthImage": True,
        }
        if config.executable:
            kwargs["local_executable_path"] = str(Path(config.executable).expanduser())
        try:
            self.controller = controller_factory(**kwargs)
        except TypeError:
            # Some custom/forked Controller wrappers expose a smaller constructor.
            # Fall back while preserving the scene and executable when accepted.
            reduced = {"scene": config.scene}
            if config.executable:
                reduced["local_executable_path"] = str(Path(config.executable).expanduser())
            self.controller = controller_factory(**reduced)

        self.event = getattr(self.controller, "last_event", None)
        if self.event is None:
            self.event = self.controller.step(action="Pass")
        self.reset()

    def valid_action_indices_for_level(self, level: int | None) -> tuple[int, ...]:
        """Curriculum-specific action mask, while preserving the 5-output policy head."""
        level = int(self.config.curriculum_level if level is None else level)
        if level == 0:
            return (0, 2, 3)  # idle + rotate only
        if 1 <= level <= 5:
            return (0, 1, 2, 3)  # navigation; interaction stays locked
        return (0, 1, 2, 3, 4)

    def valid_action_indices(self) -> tuple[int, ...]:
        level = int(self.target.level if self.target is not None else self.config.curriculum_level)
        return self.valid_action_indices_for_level(level)

    def current_skill_level(self) -> int | None:
        return int(self.target.level if self.target is not None else self.config.curriculum_level)

    def exploration_floor(self) -> float:
        """Reset exploration when a newly unlocked curriculum level begins."""
        level = max(0, min(9, int(self.config.curriculum_level)))
        if level <= 0:
            return 0.0
        steps = int(self._level_action_steps.get(level, 0))
        t = min(1.0, steps / max(1, int(self.config.level_epsilon_decay_steps)))
        start = float(self.config.level_epsilon_start)
        end = float(self.config.level_epsilon_end)
        return max(0.0, start + t * (end - start))

    @property
    def curriculum_attempts(self) -> int:
        return int(self._totals(int(self.config.curriculum_level)).get("attempts", 0))

    @property
    def curriculum_successes(self) -> int:
        return int(self._totals(int(self.config.curriculum_level)).get("successes", 0))

    @property
    def curriculum_timeouts(self) -> int:
        return int(self._totals(int(self.config.curriculum_level)).get("timeouts", 0))

    @property
    def curriculum_stagnations(self) -> int:
        return int(self._totals(int(self.config.curriculum_level)).get("stagnations", 0))

    def _recent_queue(self, level: int) -> deque[dict[str, Any]]:
        level = int(level)
        q = self._recent_outcomes_by_level.get(level)
        if q is None:
            q = deque(maxlen=max(10, int(self.config.graduation_window)))
            self._recent_outcomes_by_level[level] = q
        return q

    def _totals(self, level: int) -> dict[str, int]:
        level = int(level)
        if level not in self._level_totals:
            self._level_totals[level] = {
                "attempts": 0, "successes": 0, "timeouts": 0, "stagnations": 0, "goals": 0
            }
        return self._level_totals[level]

    def _curriculum_stats(self, level: int | None = None) -> dict[str, Any]:
        level = int(self.current_skill_level() if level is None else level)
        recent = list(self._recent_queue(level))
        attempts = len(recent)
        successes = sum(1 for x in recent if x.get("success"))
        failures = attempts - successes
        completion = [int(x["steps"]) for x in recent if x.get("success")]
        median_steps = float(np.median(completion)) if completion else None
        success_rate = successes / attempts if attempts else 0.0
        failure_rate = failures / attempts if attempts else 0.0
        timeout_rate = sum(1 for x in recent if x.get("timeout")) / attempts if attempts else 0.0
        stagnation_rate = sum(1 for x in recent if x.get("stagnation")) / attempts if attempts else 0.0
        window = max(10, int(self.config.graduation_window))
        totals = self._totals(level)
        if level == 0:
            median_limit = float(self.config.graduation_median_steps)
        elif level == 1:
            median_limit = float(max(12, int(self.config.nav_max_path_steps) * 4))
        else:
            median_limit = float(max(30, int(self.config.graduation_median_steps)))
        graduation_ready = bool(
            attempts >= window
            and success_rate >= float(self.config.graduation_success_rate)
            and median_steps is not None
            and median_steps <= median_limit
            and timeout_rate <= float(self.config.graduation_failure_rate)
        )
        return {
            "level": level,
            "window": window,
            "recent_attempts": attempts,
            "recent_successes": successes,
            "recent_success_rate": success_rate,
            "recent_failure_rate": failure_rate,
            "recent_timeout_rate": timeout_rate,
            "recent_stagnation_rate": stagnation_rate,
            "median_success_steps": median_steps,
            "graduation_median_limit": median_limit,
            "graduation_ready": graduation_ready,
            "attempts_total": int(totals.get("attempts", 0)),
            "successes_total": int(totals.get("successes", 0)),
            "goals_total": int(totals.get("goals", 0)),
            "timeouts_total": int(totals.get("timeouts", 0)),
            "stagnations_total": int(totals.get("stagnations", 0)),
            "level_action_steps": int(self._level_action_steps.get(level, 0)),
            "episode_action_counts": list(self._episode_action_counts),
        }

    # ---------- simulator helpers ----------
    @staticmethod
    def _position_from_metadata(md: dict[str, Any]) -> np.ndarray:
        agent = md.get("agent", {}) or {}
        p = agent.get("position", {}) or {}
        return np.array([float(p.get("x", 0.0)), float(p.get("z", 0.0))], dtype=np.float32)

    @staticmethod
    def _yaw_from_metadata(md: dict[str, Any]) -> float:
        agent = md.get("agent", {}) or {}
        r = agent.get("rotation", {}) or {}
        return float(r.get("y", 0.0)) % 360.0

    @staticmethod
    def _horizon_from_metadata(md: dict[str, Any]) -> float:
        agent = md.get("agent", {}) or {}
        return float(agent.get("cameraHorizon", 0.0))

    def _metadata(self) -> dict[str, Any]:
        return dict(getattr(self.event, "metadata", {}) or {})

    def _objects(self) -> list[dict[str, Any]]:
        return list(self._metadata().get("objects", []) or [])

    def _object_by_id(self, object_id: str | None) -> dict[str, Any] | None:
        if not object_id:
            return None
        for obj in self._objects():
            if obj.get("objectId") == object_id:
                return obj
        return None

    @staticmethod
    def _obj_position(obj: dict[str, Any]) -> np.ndarray:
        p = obj.get("position", {}) or {}
        return np.array([float(p.get("x", 0.0)), float(p.get("z", 0.0))], dtype=np.float32)

    def _refresh_reachable(self) -> None:
        prior = self.event
        try:
            event = self.controller.step(action="GetReachablePositions")
            ret = (getattr(event, "metadata", {}) or {}).get("actionReturn") or []
            if ret:
                self._reachable = [dict(x) for x in ret]
            # Preserve the newest frame-bearing event when the fork provides one.
            self.event = event if getattr(event, "frame", None) is not None else prior
        except Exception:
            self.event = prior

        if self._reachable:
            xs = [float(p.get("x", 0.0)) for p in self._reachable]
            zs = [float(p.get("z", 0.0)) for p in self._reachable]
            pad = 0.5
            self._bounds = [min(xs) - pad, max(xs) + pad, min(zs) - pad, max(zs) + pad]
            self._build_reachable_index()

    def _teleport_random_start(self) -> None:
        if not self.config.random_start or len(self._reachable) < 2:
            return
        p = self.rng.choice(self._reachable)
        yaw = self.rng.choice((0.0, 90.0, 180.0, 270.0))
        try:
            self.event = self.controller.step(
                action="TeleportFull",
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                z=float(p.get("z", 0.0)),
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
                horizon=0.0,
                standing=True,
            )
        except Exception:
            pass

    def _capture_l0_anchor(self) -> None:
        md = self._metadata()
        agent = md.get("agent", {}) or {}
        p = agent.get("position", {}) or {}
        self._l0_anchor = {
            "x": float(p.get("x", 0.0)),
            "y": float(p.get("y", 0.0)),
            "z": float(p.get("z", 0.0)),
        }

    def _enforce_l0_position_lock(self) -> float:
        if not self.target or self.target.kind != "orient" or not self._l0_anchor:
            return 0.0
        md = self._metadata()
        p = self._position_from_metadata(md)
        anchor = np.array([self._l0_anchor["x"], self._l0_anchor["z"]], dtype=np.float32)
        drift = float(np.linalg.norm(p - anchor))
        if drift <= 1e-4:
            return drift
        original_success = self.last_action_success
        try:
            yaw = self._yaw_from_metadata(md)
            self.event = self.controller.step(
                action="TeleportFull",
                x=self._l0_anchor["x"], y=self._l0_anchor["y"], z=self._l0_anchor["z"],
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
                horizon=self._horizon_from_metadata(md), standing=True,
            )
        except Exception:
            pass
        self.last_action_success = original_success
        return drift

    # ---------- curriculum ----------
    def _grid_key(self, pos: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        g = max(1e-4, float(self.config.grid_size))
        return (int(round(float(pos[0]) / g)), int(round(float(pos[1]) / g)))

    def _build_reachable_index(self) -> None:
        self._reachable_index = {}
        for item in self._reachable:
            p = np.array([float(item.get("x", 0.0)), float(item.get("z", 0.0))], dtype=np.float32)
            self._reachable_index[self._grid_key(p)] = p

    def _nearest_reachable_key(self, pos: np.ndarray) -> tuple[int, int] | None:
        if not self._reachable_index:
            return None
        key = self._grid_key(pos)
        if key in self._reachable_index:
            return key
        return min(self._reachable_index, key=lambda k: float(np.linalg.norm(self._reachable_index[k] - pos)))

    def _grid_neighbors(self, key: tuple[int, int]) -> list[tuple[int, int]]:
        x, z = key
        candidates = ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1))
        return [q for q in candidates if q in self._reachable_index]

    def _short_l1_path(self, start: np.ndarray) -> list[np.ndarray]:
        """Choose a short collision-free teacher path over AI2-THOR reachable cells."""
        start_key = self._nearest_reachable_key(start)
        if start_key is None:
            return [self._reachable_point()]

        stats = self._curriculum_stats(1)
        attempts = int(stats.get("recent_attempts", 0))
        success_rate = float(stats.get("recent_success_rate", 0.0))
        base = max(1, int(self.config.nav_initial_path_steps))
        max_steps = max(base, int(self.config.nav_max_path_steps))
        if attempts < 20:
            difficulty = base
        elif success_rate < 0.70:
            difficulty = min(max_steps, base + 1)
        elif success_rate < 0.90:
            difficulty = min(max_steps, base + 3)
        else:
            difficulty = max_steps

        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
        depth: dict[tuple[int, int], int] = {start_key: 0}
        queue: deque[tuple[int, int]] = deque([start_key])
        candidates: list[tuple[int, int]] = []
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if 1 <= d <= difficulty:
                candidates.append(cur)
            if d >= difficulty:
                continue
            for nxt in self._grid_neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = cur
                depth[nxt] = d + 1
                queue.append(nxt)

        if not candidates:
            return [self._reachable_point()]
        # Favor the currently unlocked upper half of the difficulty band while
        # retaining variety so the agent cannot memorize one path length.
        min_depth = max(1, difficulty // 2)
        preferred = [k for k in candidates if depth[k] >= min_depth] or candidates
        goal_key = self.rng.choice(preferred)
        keys: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal_key
        while cur is not None and cur != start_key:
            keys.append(cur)
            cur = parent[cur]
        keys.reverse()
        return [self._reachable_index[k].copy() for k in keys]

    def _reachable_point(self, *, far: bool = False) -> np.ndarray:
        agent = self._position_from_metadata(self._metadata())
        if not self._reachable:
            angle = self.rng.random() * math.tau
            radius = 1.5 if far else 0.8
            return agent + np.array([math.sin(angle) * radius, math.cos(angle) * radius], dtype=np.float32)
        points = [np.array([float(p["x"]), float(p["z"])], dtype=np.float32) for p in self._reachable]
        if far:
            points.sort(key=lambda q: float(np.linalg.norm(q - agent)), reverse=True)
            pool = points[: max(1, len(points) // 4)]
        else:
            pool = [q for q in points if float(np.linalg.norm(q - agent)) > 0.75] or points
        return self.rng.choice(pool).copy()

    def _candidate_objects(self, *, visible_only: bool = False, doors_only: bool = False, interactive: bool = False) -> list[dict[str, Any]]:
        out = []
        for obj in self._objects():
            if visible_only and not bool(obj.get("visible", False)):
                continue
            typ = str(obj.get("objectType", ""))
            if doors_only and not (typ.lower() == "door" or bool(obj.get("openable", False)) and "door" in typ.lower()):
                continue
            if interactive and not any(bool(obj.get(k, False)) for k in ("openable", "toggleable", "pickupable")):
                continue
            if obj.get("position"):
                out.append(obj)
        return out

    def _choose_target(self, level_override: int | None = None) -> CurriculumTarget:
        level = max(0, min(9, int(self.config.curriculum_level if level_override is None else level_override)))
        start = self._position_from_metadata(self._metadata())

        if level == 0:
            pos = self._reachable_point(far=False)
            return CurriculumTarget(level, "orient", "orientation point", pos, start_position=start.copy())
        if level == 1:
            path = self._short_l1_path(start)
            goal = path[-1].copy() if path else self._reachable_point()
            return CurriculumTarget(
                level, "point", "reachable point", goal, start_position=start.copy(),
                waypoints=[p.copy() for p in path], waypoint_index=0,
            )
        if level == 2:
            objs = self._candidate_objects(visible_only=True)
            if objs:
                obj = self.rng.choice(objs)
                return CurriculumTarget(level, "visible_object", str(obj.get("objectType", "object")), self._obj_position(obj), str(obj.get("objectId")), start_position=start.copy())
            return CurriculumTarget(level, "point", "visible fallback point", self._reachable_point(), start_position=start.copy())
        if level == 3:
            objs = self._candidate_objects()
            if objs:
                obj = self.rng.choice(objs)
                return CurriculumTarget(level, "find_object", str(obj.get("objectType", "object")), self._obj_position(obj), str(obj.get("objectId")), start_position=start.copy())
            return CurriculumTarget(level, "point", "object fallback point", self._reachable_point(far=True), start_position=start.copy())
        if level == 4:
            return CurriculumTarget(level, "far_point", "far reachable point", self._reachable_point(far=True), start_position=start.copy())
        if level in (5, 6, 7):
            doors = self._candidate_objects(doors_only=True)
            if doors:
                obj = min(doors, key=lambda o: float(np.linalg.norm(self._obj_position(o) - start)))
                pos = self._obj_position(obj)
                crossed = self._crossing_point(start, pos) if level == 7 else None
                return CurriculumTarget(level, ("approach_door", "open_door", "cross_door")[level - 5], str(obj.get("objectType", "Door")), pos, str(obj.get("objectId")), start_position=start.copy(), crossed_position=crossed)
            return CurriculumTarget(level, "far_point", "door fallback point", self._reachable_point(far=True), start_position=start.copy())
        objs = self._candidate_objects(interactive=True)
        if objs:
            obj = self.rng.choice(objs)
            kind = "interact_object" if level == 8 else "multi_step"
            return CurriculumTarget(level, kind, str(obj.get("objectType", "interactive object")), self._obj_position(obj), str(obj.get("objectId")), start_position=start.copy())
        return CurriculumTarget(level, "far_point", "interaction fallback point", self._reachable_point(far=True), start_position=start.copy())

    def _crossing_point(self, start: np.ndarray, door: np.ndarray) -> np.ndarray:
        direction = door - start
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            return door.copy()
        direction /= norm
        desired = door + direction * 0.8
        if not self._reachable:
            return desired.astype(np.float32)
        pts = [np.array([float(p["x"]), float(p["z"])], dtype=np.float32) for p in self._reachable]
        beyond = [p for p in pts if float(np.dot(p - door, direction)) > 0.15 and float(np.linalg.norm(p - door)) < 1.5]
        if not beyond:
            beyond = pts
        return min(beyond, key=lambda p: float(np.linalg.norm(p - desired))).copy()

    # ---------- observations ----------
    def _depth_sectors(self) -> tuple[float, float, float, float]:
        depth = getattr(self.event, "depth_frame", None)
        if depth is None:
            return 1.0, 1.0, 1.0, 0.0
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2 or d.size == 0:
            return 1.0, 1.0, 1.0, 0.0
        d = np.nan_to_num(d, nan=5.0, posinf=5.0, neginf=0.0)
        d = np.clip(d, 0.0, 5.0)
        h, w = d.shape
        y0, y1 = int(h * 0.30), max(int(h * 0.80), int(h * 0.30) + 1)
        crop = d[y0:y1]
        third = max(1, w // 3)
        left = float(np.median(crop[:, :third])) / 5.0
        center = float(np.median(crop[:, third:2 * third])) / 5.0
        right = float(np.median(crop[:, 2 * third:])) / 5.0
        near = float(np.mean(crop < max(0.05, self.config.safety_distance)))
        return left, center, right, near

    def _final_goal_position(self) -> np.ndarray | None:
        if self.target is None:
            return None
        if self.target.kind == "cross_door" and self.target.crossed_position is not None:
            return self.target.crossed_position
        return self.target.position

    def _active_goal_position(self) -> np.ndarray | None:
        """Teacher-nav goal for the current decision; L1 uses a local waypoint."""
        if self.target is None:
            return None
        if (
            self.target.level == 1
            and self.config.goal_signal.strip().lower() == "teacher-nav"
            and self.target.waypoints
        ):
            idx = max(0, min(int(self.target.waypoint_index), len(self.target.waypoints) - 1))
            return self.target.waypoints[idx]
        return self._final_goal_position()

    def _advance_waypoint_if_reached(self) -> bool:
        if self.target is None or self.target.level != 1 or not self.target.waypoints:
            return False
        pos = self._position_from_metadata(self._metadata())
        advanced = False
        radius = max(0.05, float(self.config.waypoint_radius))
        while self.target.waypoint_index < len(self.target.waypoints) - 1:
            wp = self.target.waypoints[self.target.waypoint_index]
            if float(np.linalg.norm(wp - pos)) > radius:
                break
            self.target.waypoint_index += 1
            advanced = True
        return advanced

    def _goal_local(self) -> tuple[float, float, float, float]:
        if self.target is None:
            return 0.0, 0.0, 0.0, 0.0
        pos = self._position_from_metadata(self._metadata())
        goal = self._active_goal_position()
        if goal is None:
            return 0.0, 0.0, 0.0, 0.0
        delta = goal - pos
        dist = float(np.linalg.norm(delta))
        yaw = math.radians(self._yaw_from_metadata(self._metadata()))
        # AI2-THOR yaw 0 faces +Z; right vector is +X at yaw 0.
        forward = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        right = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
        f = float(np.dot(delta, forward))
        r = float(np.dot(delta, right))
        bearing = math.degrees(math.atan2(r, f)) if dist > 1e-6 else 0.0
        return f, r, dist, bearing

    def _final_goal_metrics(self) -> tuple[float, float]:
        goal = self._final_goal_position()
        if goal is None:
            return 0.0, 0.0
        pos = self._position_from_metadata(self._metadata())
        delta = goal - pos
        dist = float(np.linalg.norm(delta))
        yaw = math.radians(self._yaw_from_metadata(self._metadata()))
        forward = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        right = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
        f = float(np.dot(delta, forward)); r = float(np.dot(delta, right))
        bearing = math.degrees(math.atan2(r, f)) if dist > 1e-6 else 0.0
        return dist, bearing

    def _obs(self) -> np.ndarray:
        f, r, _dist, _bearing = self._goal_local()
        goal_enabled = self.config.goal_signal.strip().lower() != "sensor-only"
        if goal_enabled and self.config.goal_dropout > 0.0 and self.rng.random() < self.config.goal_dropout:
            goal_enabled = False
        if not goal_enabled:
            f = r = 0.0
        left, center, right, _near = self._depth_sectors()
        return np.array([
            float(np.clip(f / 5.0, -1.0, 1.0)),
            float(np.clip(r / 5.0, -1.0, 1.0)),
            float(np.clip(center, 0.0, 1.0)),
            float(np.clip(left - right, -1.0, 1.0)),
            float(np.clip(self.last_action_success, 0.0, 1.0)),
            float(np.clip(self.last_reward, -1.0, 1.0)),
        ], dtype=np.float32)

    # ---------- actions / reward ----------
    def _step_controller(self, **kwargs: Any) -> Any:
        try:
            event = self.controller.step(**kwargs)
        except TypeError:
            action = kwargs.pop("action")
            event = self.controller.step(action, **kwargs)
        self.event = event
        md = self._metadata()
        self.last_action_success = 1.0 if bool(md.get("lastActionSuccess", True)) else 0.0
        return event

    def _interact(self) -> tuple[bool, str]:
        obj = self._object_by_id(self.target.object_id if self.target else None)
        if obj is None:
            # Fall back to the nearest visible interactive object.
            pos = self._position_from_metadata(self._metadata())
            candidates = self._candidate_objects(visible_only=True, interactive=True)
            if candidates:
                obj = min(candidates, key=lambda o: float(np.linalg.norm(self._obj_position(o) - pos)))
        if obj is None:
            self._step_controller(action="Pass")
            return False, "no interactive target"
        object_id = str(obj.get("objectId"))
        if bool(obj.get("openable", False)) and not bool(obj.get("isOpen", False)):
            self._step_controller(action="OpenObject", objectId=object_id)
            return bool(self.last_action_success), "OpenObject"
        if bool(obj.get("toggleable", False)) and not bool(obj.get("isToggled", False)):
            self._step_controller(action="ToggleObjectOn", objectId=object_id)
            return bool(self.last_action_success), "ToggleObjectOn"
        if bool(obj.get("pickupable", False)) and not bool(obj.get("isPickedUp", False)):
            self._step_controller(action="PickupObject", objectId=object_id)
            return bool(self.last_action_success), "PickupObject"
        self._step_controller(action="Pass")
        return False, "already satisfied or unsupported"

    def _apply_intent(self, action: int) -> tuple[bool, str]:
        a = int(action)
        noise = max(0.0, min(0.5, self.config.action_noise))
        if noise > 0 and self.rng.random() < noise and a in (1, 2, 3):
            noisy_candidates = tuple(x for x in self.valid_action_indices() if x in (1, 2, 3))
            if noisy_candidates:
                a = self.rng.choice(noisy_candidates)
        if a == 0:
            self._step_controller(action="Pass")
            return bool(self.last_action_success), "Pass"
        if a == 1:
            _left, center, _right, _near = self._depth_sectors()
            if center * 5.0 < self.config.safety_distance:
                self._step_controller(action="Pass")
                self.last_action_success = 0.0
                return False, "SafetyStop"
            self._step_controller(action="MoveAhead", moveMagnitude=float(self.config.grid_size))
            return bool(self.last_action_success), "MoveAhead"
        if a == 2:
            self._step_controller(action="RotateLeft", degrees=float(self.config.rotate_degrees))
            return bool(self.last_action_success), "RotateLeft"
        if a == 3:
            self._step_controller(action="RotateRight", degrees=float(self.config.rotate_degrees))
            return bool(self.last_action_success), "RotateRight"
        return self._interact()

    def _task_success(self, interaction_success: bool) -> tuple[bool, str]:
        if self.target is None:
            return False, "no_target"
        _f, _r, dist, bearing = self._goal_local()
        final_dist, _final_bearing = self._final_goal_metrics()
        kind = self.target.kind
        if kind == "orient":
            return abs(bearing) <= float(self.config.orient_success_degrees), "oriented"
        if kind in {"point", "far_point", "visible_object", "find_object", "approach_door"}:
            threshold = 0.55 if "object" in kind or "door" in kind else 0.35
            return final_dist <= threshold, "arrived"
        if kind == "open_door":
            obj = self._object_by_id(self.target.object_id)
            return bool(obj and obj.get("isOpen", False)), "door_open"
        if kind == "cross_door":
            obj = self._object_by_id(self.target.object_id)
            open_ok = bool(obj is None or obj.get("isOpen", False))
            return open_ok and final_dist <= 0.45, "door_crossed"
        if kind in {"interact_object", "multi_step"}:
            if interaction_success:
                if kind == "multi_step":
                    self.completed_subgoals += 1
                    if self.completed_subgoals < 2:
                        self.target = self._choose_target()
                        self.target.level = 9
                        self.target.kind = "multi_step"
                        return False, "subgoal_complete"
                return True, "interaction_complete"
        return False, "in_progress"

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.last_reward = 0.0
        self.last_action_success = 1.0
        self.energy = 1.0
        self.completed_subgoals = 0
        try:
            if hasattr(self.controller, "reset"):
                self.event = self.controller.reset(scene=self.config.scene)
        except TypeError:
            self.event = self.controller.reset(self.config.scene)
        except Exception:
            pass
        if self.event is None:
            self.event = self.controller.step(action="Pass")
        self._refresh_reachable()
        self._teleport_random_start()

        configured = max(0, min(9, int(self.config.curriculum_level)))
        episode_level = configured
        rehearsal_fraction = max(0.0, min(0.50, float(self.config.rehearsal_fraction)))
        if configured > 0 and rehearsal_fraction > 0.0 and self.rng.random() < rehearsal_fraction:
            # Rehearse an earlier skill instead of merely replaying its data.  For L1
            # this is necessarily L0; later levels sample across the acquired stack.
            episode_level = self.rng.randrange(0, configured)

        self.target = self._choose_target(level_override=episode_level)
        if self.target.kind == "orient":
            self._capture_l0_anchor()
            self._best_progress_metric = abs(self._goal_local()[3])
        else:
            self._l0_anchor = None
            self._best_progress_metric = self._goal_local()[2]
        self._stagnation_count = 0
        self._episode_action_counts = [0] * self.action_dim
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        requested_action = int(action)
        level = int(self.target.level if self.target else self.config.curriculum_level)
        configured_level = int(self.config.curriculum_level)
        self._level_action_steps[level] = int(self._level_action_steps.get(level, 0)) + 1

        valid = self.valid_action_indices()
        if requested_action not in valid:
            # Keep a hard environment boundary even though the neural policy is masked.
            requested_action = 0
        self._episode_action_counts[requested_action] += 1

        prev_pos = self._position_from_metadata(self._metadata()).copy()
        _prev_f, _prev_r, prev_dist, prev_bearing = self._goal_local()
        interaction_success, primitive = self._apply_intent(requested_action)
        l0_drift = self._enforce_l0_position_lock()

        # Measure progress against the waypoint that was active when the action was
        # chosen.  Only after this measurement may the teacher advance the waypoint.
        _same_f, _same_r, same_goal_dist, same_goal_bearing = self._goal_local()
        post_pos = self._position_from_metadata(self._metadata()).copy()
        actual_move_distance = float(np.linalg.norm(post_pos - prev_pos))
        angular_delta = abs(prev_bearing) - abs(same_goal_bearing)
        distance_delta = prev_dist - same_goal_dist
        waypoint_advanced = self._advance_waypoint_if_reached()
        _f, _r, dist, bearing = self._goal_local()
        final_dist, final_bearing = self._final_goal_metrics()

        reward = -0.003
        is_orient = bool(self.target and self.target.kind == "orient")
        if is_orient:
            reward += float(self.config.orient_progress_scale) * angular_delta
            if abs(prev_bearing) > float(self.config.orient_success_degrees):
                correct = (prev_bearing > 0.0 and requested_action == 3) or (prev_bearing < 0.0 and requested_action == 2)
                wrong = (prev_bearing > 0.0 and requested_action == 2) or (prev_bearing < 0.0 and requested_action == 3)
                if correct:
                    reward += float(self.config.orient_correct_turn_bonus)
                elif wrong:
                    reward -= float(self.config.orient_wrong_turn_penalty)
                elif requested_action == 0:
                    reward -= 0.005
        elif level == 1:
            # Compose the mastered L0 angular primitive with newly unlocked forward
            # motion.  A good turn is rewarded even though Euclidean distance does
            # not change, while movement is judged against the collision-free waypoint.
            reward += float(self.config.nav_angular_scale) * angular_delta
            reward += float(self.config.nav_distance_scale) * distance_delta
            if requested_action in (2, 3) and abs(prev_bearing) > float(self.config.orient_success_degrees):
                correct = (prev_bearing > 0.0 and requested_action == 3) or (prev_bearing < 0.0 and requested_action == 2)
                wrong = (prev_bearing > 0.0 and requested_action == 2) or (prev_bearing < 0.0 and requested_action == 3)
                if correct:
                    reward += 0.5 * float(self.config.orient_correct_turn_bonus)
                elif wrong:
                    reward -= 0.5 * float(self.config.orient_wrong_turn_penalty)
            if requested_action == 1:
                blocked = primitive == "SafetyStop" or self.last_action_success < 0.5 or actual_move_distance < 1e-4
                if blocked:
                    reward -= float(self.config.nav_blocked_penalty)
                else:
                    reward += float(self.config.nav_forward_bonus)
                if distance_delta < -float(self.config.nav_progress_distance_epsilon):
                    reward -= float(self.config.nav_wrong_way_penalty)
            elif requested_action == 0 and abs(prev_bearing) > float(self.config.orient_success_degrees):
                reward -= 0.004
            if waypoint_advanced:
                reward += 0.08
        else:
            reward += 0.18 * distance_delta

        if self.last_action_success < 0.5 and not (level == 1 and requested_action == 1):
            reward -= 0.02

        # v0.4.2 stagnation is skill-aware.  Useful rotation in navigation resets the
        # watchdog even when distance is unchanged; previously this was misclassified
        # as no progress and could terminate a promising route.
        if is_orient:
            made_progress = angular_delta > max(0.0, float(self.config.stagnation_epsilon))
        elif level == 1:
            made_progress = bool(
                waypoint_advanced
                or distance_delta > max(0.0, float(self.config.nav_progress_distance_epsilon))
                or angular_delta > max(0.0, float(self.config.nav_progress_angle_epsilon))
            )
        else:
            eps = min(0.02, max(0.0, float(self.config.stagnation_epsilon)) / 100.0)
            made_progress = dist < self._best_progress_metric - eps

        metric = abs(bearing) if is_orient else dist
        if made_progress:
            self._best_progress_metric = min(self._best_progress_metric, metric)
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        success, reason = self._task_success(interaction_success)
        if success:
            reward += 1.0
        elif reason == "subgoal_complete":
            reward += 0.5

        stagnant = (not success) and self._stagnation_count >= max(1, int(self.config.stagnation_steps))
        timed_out = (not success) and self.steps >= int(self.config.max_episode_steps)
        if stagnant:
            reason = "stagnation"
            reward -= 0.10
        elif timed_out:
            reason = "timeout"
            reward -= 0.10

        self.last_reward = float(reward)
        self.energy = 1.0
        done = bool(success or stagnant or timed_out)
        if done:
            totals = self._totals(level)
            totals["attempts"] += 1
            if success:
                totals["successes"] += 1
                totals["goals"] += 1
            if stagnant:
                totals["stagnations"] += 1
            if timed_out:
                totals["timeouts"] += 1
            self._recent_queue(level).append({
                "success": bool(success), "steps": int(self.steps),
                "stagnation": bool(stagnant), "timeout": bool(timed_out),
            })

        episode_stats = self._curriculum_stats(level)
        configured_stats = self._curriculum_stats(configured_level)
        waypoint = self._active_goal_position()
        final_goal = self._final_goal_position()
        info = {
            "distance": float(dist),
            "final_distance": float(final_dist),
            "bearing": float(bearing),
            "final_bearing": float(final_bearing),
            "delta_bearing": float(angular_delta),
            "delta_distance": float(distance_delta),
            "actual_move_distance": float(actual_move_distance),
            "waypoint_advanced": bool(waypoint_advanced),
            "waypoint": None if waypoint is None else [float(waypoint[0]), float(waypoint[1])],
            "final_target": None if final_goal is None else [float(final_goal[0]), float(final_goal[1])],
            "waypoint_index": int(self.target.waypoint_index if self.target else 0),
            "waypoint_count": int(len(self.target.waypoints) if self.target and self.target.waypoints else 0),
            "path_steps": int(len(self.target.waypoints) if self.target and self.target.waypoints else 0),
            "reached": bool(success),
            "task_success": bool(success),
            "task_reason": reason,
            "curriculum_level": level,
            "configured_curriculum_level": configured_level,
            "is_rehearsal": bool(level != configured_level),
            "task_kind": self.target.kind if self.target else "none",
            "target_label": self.target.label if self.target else "none",
            "requested_action": int(action),
            "executed_action": int(requested_action),
            "allowed_actions": list(valid),
            "primitive_action": primitive,
            "last_action_success": bool(self.last_action_success),
            "goal_signal": self.config.goal_signal,
            "stagnation_steps": int(self._stagnation_count),
            "stagnation_limit": int(self.config.stagnation_steps),
            "stagnant": bool(stagnant),
            "timed_out": bool(timed_out),
            "l0_position_drift": float(l0_drift),
            "curriculum_stats": episode_stats,
            "configured_curriculum_stats": configured_stats,
            "graduation_ready": bool(configured_stats["graduation_ready"]),
            "level_epsilon_floor": float(self.exploration_floor()),
        }
        return self._obs(), float(reward), done, info

    def state_dict(self) -> dict:
        return {
            "format_version": 2,
            "recent_outcomes_by_level": {
                str(level): list(items) for level, items in self._recent_outcomes_by_level.items()
            },
            "level_totals": {str(level): dict(values) for level, values in self._level_totals.items()},
            "level_action_steps": {str(level): int(value) for level, value in self._level_action_steps.items()},
            "legacy_v041_stats": dict(self._legacy_v041_stats),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self._recent_outcomes_by_level.clear()
        self._level_totals.clear()
        self._level_action_steps.clear()

        if "recent_outcomes_by_level" in state or "level_totals" in state:
            for level_text, items in dict(state.get("recent_outcomes_by_level", {})).items():
                try:
                    level = int(level_text)
                except (TypeError, ValueError):
                    continue
                q = self._recent_queue(level)
                for item in list(items)[-q.maxlen:]:
                    if isinstance(item, dict):
                        q.append(dict(item))
            for level_text, values in dict(state.get("level_totals", {})).items():
                try:
                    level = int(level_text)
                except (TypeError, ValueError):
                    continue
                if isinstance(values, dict):
                    t = self._totals(level)
                    for key in t:
                        t[key] = int(values.get(key, t[key]))
            for level_text, value in dict(state.get("level_action_steps", {})).items():
                try:
                    self._level_action_steps[int(level_text)] = int(value)
                except (TypeError, ValueError):
                    pass
            self._legacy_v041_stats = dict(state.get("legacy_v041_stats", {}))
            return

        # Migration from v0.4.1: its single recent window mixed levels, so do not
        # pretend those outcomes belong to the configured level. Preserve them only
        # as diagnostic history and start trustworthy per-level evidence from zero.
        self._legacy_v041_stats = {
            "curriculum_attempts": int(state.get("curriculum_attempts", 0)),
            "curriculum_successes": int(state.get("curriculum_successes", 0)),
            "curriculum_timeouts": int(state.get("curriculum_timeouts", 0)),
            "curriculum_stagnations": int(state.get("curriculum_stagnations", 0)),
            "recent_outcomes": list(state.get("recent_outcomes", [])),
        }

    # ---------- sensory / dashboard ----------
    def render_rgb(self, size: int = 96) -> np.ndarray:
        frame = getattr(self.event, "frame", None)
        if frame is None:
            return np.zeros((max(32, size), max(32, size), 3), dtype=np.uint8)
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return np.zeros((max(32, size), max(32, size), 3), dtype=np.uint8)
        return rgb[..., :3].copy()

    def render_depth(self) -> np.ndarray | None:
        depth = getattr(self.event, "depth_frame", None)
        return None if depth is None else np.asarray(depth, dtype=np.float32).copy()

    @staticmethod
    def _thumbnail_payload(rgb: np.ndarray, max_width: int = 192) -> dict[str, Any]:
        x = np.asarray(rgb, dtype=np.uint8)
        h, w, _ = x.shape
        if w > max_width:
            scale = max_width / float(w)
            nh = max(1, int(round(h * scale)))
            ys = np.linspace(0, h - 1, nh).astype(int)
            xs = np.linspace(0, w - 1, max_width).astype(int)
            x = x[np.ix_(ys, xs)]
            h, w, _ = x.shape
        return {
            "width": int(w),
            "height": int(h),
            "rgb_b64": base64.b64encode(x.tobytes()).decode("ascii"),
        }

    def visual_snapshot(self, predicted_obs: np.ndarray | None = None) -> dict[str, Any]:
        md = self._metadata()
        pos = self._position_from_metadata(md)
        yaw = self._yaw_from_metadata(md)
        left, center, right, near = self._depth_sectors()
        final_goal = self._final_goal_position()
        waypoint = self._active_goal_position()
        level = int(self.current_skill_level() or 0)
        configured = int(self.config.curriculum_level)
        final_dist, final_bearing = self._final_goal_metrics()
        out = {
            "environment_type": "ai2thor",
            "render_mode": "embodied3d",
            "agent": [float(pos[0]), float(pos[1])],
            "target": None if final_goal is None else [float(final_goal[0]), float(final_goal[1])],
            "waypoint": None if waypoint is None else [float(waypoint[0]), float(waypoint[1])],
            "predicted_position": None,
            "world_bounds": [float(v) for v in self._bounds],
            "yaw": float(yaw),
            "camera_horizon": float(self._horizon_from_metadata(md)),
            "energy": float(self.energy),
            "depth_left": float(left * 5.0),
            "depth_center": float(center * 5.0),
            "depth_right": float(right * 5.0),
            "near_obstacle_fraction": float(near),
            "curriculum_level": level,
            "configured_curriculum_level": configured,
            "is_rehearsal": bool(level != configured),
            "task_kind": self.target.kind if self.target else "none",
            "target_label": self.target.label if self.target else "none",
            "goal_signal": self.config.goal_signal,
            "allowed_actions": list(self.valid_action_indices()),
            "curriculum_stats": self._curriculum_stats(configured),
            "episode_curriculum_stats": self._curriculum_stats(level),
            "configured_curriculum_stats": self._curriculum_stats(configured),
            "stagnation_steps": int(self._stagnation_count),
            "stagnation_limit": int(self.config.stagnation_steps),
            "final_distance": float(final_dist),
            "final_bearing": float(final_bearing),
            "waypoint_index": int(self.target.waypoint_index if self.target else 0),
            "waypoint_count": int(len(self.target.waypoints) if self.target and self.target.waypoints else 0),
            "path_steps": int(len(self.target.waypoints) if self.target and self.target.waypoints else 0),
            "level_epsilon_floor": float(self.exploration_floor()),
            "camera": self._thumbnail_payload(self.render_rgb()),
        }
        if predicted_obs is not None:
            p = np.asarray(predicted_obs, dtype=np.float32).reshape(-1)
            out["predicted_sensor"] = [float(v) for v in p[:6].tolist()]
        return out

    def describe(self) -> str:
        pos = self._position_from_metadata(self._metadata())
        _f, _r, dist, bearing = self._goal_local()
        final_dist, _final_bearing = self._final_goal_metrics()
        level = self.target.level if self.target else self.config.curriculum_level
        configured = int(self.config.curriculum_level)
        rehearsal = " rehearsal" if int(level) != configured else ""
        wp = ""
        if self.target and self.target.waypoints:
            wp = f" wp={self.target.waypoint_index + 1}/{len(self.target.waypoints)} final={final_dist:.2f}"
        return (
            f"scene={self.config.scene} level={level}{rehearsal} "
            f"task={self.target.kind if self.target else 'none'} target={self.target.label if self.target else 'none'} "
            f"agent=({pos[0]:+.2f},{pos[1]:+.2f}) yaw={self._yaw_from_metadata(self._metadata()):.1f} "
            f"distance={dist:.2f} bearing={bearing:+.1f}{wp} "
            f"stagnation={self._stagnation_count}/{self.config.stagnation_steps}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.controller.stop()
        except Exception:
            try:
                self.controller.close()
            except Exception:
                pass
