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
        self._bounds = [-1.0, 1.0, -1.0, 1.0]
        self.target: CurriculumTarget | None = None
        self._l0_anchor: dict[str, float] | None = None
        self._best_progress_metric = float("inf")
        self._stagnation_count = 0
        self._episode_action_counts = [0] * self.action_dim
        self._recent_outcomes: deque[dict[str, Any]] = deque(maxlen=max(10, int(config.graduation_window)))
        self.curriculum_attempts = 0
        self.curriculum_successes = 0
        self.curriculum_timeouts = 0
        self.curriculum_stagnations = 0

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

    def valid_action_indices(self) -> tuple[int, ...]:
        """Curriculum-specific action mask, while preserving the 5-output policy head."""
        level = int(self.target.level if self.target is not None else self.config.curriculum_level)
        if level == 0:
            return (0, 2, 3)  # idle + rotate only
        if 1 <= level <= 5:
            return (0, 1, 2, 3)  # navigation; interaction stays locked
        return (0, 1, 2, 3, 4)

    def _curriculum_stats(self) -> dict[str, Any]:
        recent = list(self._recent_outcomes)
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
        graduation_ready = bool(
            attempts >= window
            and success_rate >= float(self.config.graduation_success_rate)
            and median_steps is not None
            and median_steps <= float(self.config.graduation_median_steps)
            and timeout_rate <= float(self.config.graduation_failure_rate)
        )
        return {
            "window": window,
            "recent_attempts": attempts,
            "recent_successes": successes,
            "recent_success_rate": success_rate,
            "recent_failure_rate": failure_rate,
            "recent_timeout_rate": timeout_rate,
            "recent_stagnation_rate": stagnation_rate,
            "median_success_steps": median_steps,
            "graduation_ready": graduation_ready,
            "attempts_total": self.curriculum_attempts,
            "successes_total": self.curriculum_successes,
            "timeouts_total": self.curriculum_timeouts,
            "stagnations_total": self.curriculum_stagnations,
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

    def _choose_target(self) -> CurriculumTarget:
        level = max(0, min(9, int(self.config.curriculum_level)))
        start = self._position_from_metadata(self._metadata())

        if level == 0:
            pos = self._reachable_point(far=False)
            return CurriculumTarget(level, "orient", "orientation point", pos, start_position=start.copy())
        if level == 1:
            return CurriculumTarget(level, "point", "reachable point", self._reachable_point(), start_position=start.copy())
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

    def _goal_local(self) -> tuple[float, float, float, float]:
        if self.target is None:
            return 0.0, 0.0, 0.0, 0.0
        pos = self._position_from_metadata(self._metadata())
        goal = self.target.crossed_position if self.target.kind == "cross_door" and self.target.crossed_position is not None else self.target.position
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
        f, r, dist, bearing = self._goal_local()
        kind = self.target.kind
        if kind == "orient":
            return abs(bearing) <= float(self.config.orient_success_degrees), "oriented"
        if kind in {"point", "far_point", "visible_object", "find_object", "approach_door"}:
            threshold = 0.55 if "object" in kind or "door" in kind else 0.35
            return dist <= threshold, "arrived"
        if kind == "open_door":
            obj = self._object_by_id(self.target.object_id)
            return bool(obj and obj.get("isOpen", False)), "door_open"
        if kind == "cross_door":
            obj = self._object_by_id(self.target.object_id)
            open_ok = bool(obj is None or obj.get("isOpen", False))
            return open_ok and dist <= 0.45, "door_crossed"
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
        self.target = self._choose_target()
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
        valid = self.valid_action_indices()
        if requested_action not in valid:
            # This should not happen with the v0.4.1 agent-side mask. Keep a hard
            # environment boundary too so external planners cannot violate curriculum.
            requested_action = 0
        self._episode_action_counts[requested_action] += 1

        _prev_f, _prev_r, prev_dist, prev_bearing = self._goal_local()
        interaction_success, primitive = self._apply_intent(requested_action)
        l0_drift = self._enforce_l0_position_lock()
        _f, _r, dist, bearing = self._goal_local()

        reward = -0.003
        level = int(self.target.level if self.target else self.config.curriculum_level)
        angular_delta = abs(prev_bearing) - abs(bearing)
        distance_delta = prev_dist - dist
        if self.target and self.target.kind == "orient":
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
            metric = abs(bearing)
        else:
            reward += 0.18 * distance_delta
            metric = dist
        if self.last_action_success < 0.5:
            reward -= 0.02

        progress_eps = max(0.0, float(self.config.stagnation_epsilon))
        # Angular metric is degrees; distance metric is meters. Use a smaller meter epsilon.
        eps = progress_eps if self.target and self.target.kind == "orient" else min(0.02, progress_eps / 100.0)
        if metric < self._best_progress_metric - eps:
            self._best_progress_metric = metric
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
        self.energy = 1.0  # placeholder until real battery/body homeostasis is connected
        done = bool(success or stagnant or timed_out)
        if done:
            self.curriculum_attempts += 1
            if success:
                self.curriculum_successes += 1
            if stagnant:
                self.curriculum_stagnations += 1
            if timed_out:
                self.curriculum_timeouts += 1
            self._recent_outcomes.append({
                "success": bool(success), "steps": int(self.steps),
                "stagnation": bool(stagnant), "timeout": bool(timed_out),
            })

        stats = self._curriculum_stats()
        info = {
            "distance": float(dist),
            "bearing": float(bearing),
            "delta_bearing": float(angular_delta),
            "delta_distance": float(distance_delta),
            "reached": bool(success),
            "task_success": bool(success),
            "task_reason": reason,
            "curriculum_level": level,
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
            "curriculum_stats": stats,
            "graduation_ready": bool(stats["graduation_ready"]),
        }
        return self._obs(), float(reward), done, info

    def state_dict(self) -> dict:
        return {
            "curriculum_attempts": int(self.curriculum_attempts),
            "curriculum_successes": int(self.curriculum_successes),
            "curriculum_timeouts": int(self.curriculum_timeouts),
            "curriculum_stagnations": int(self.curriculum_stagnations),
            "recent_outcomes": list(self._recent_outcomes),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self.curriculum_attempts = int(state.get("curriculum_attempts", self.curriculum_attempts))
        self.curriculum_successes = int(state.get("curriculum_successes", self.curriculum_successes))
        self.curriculum_timeouts = int(state.get("curriculum_timeouts", self.curriculum_timeouts))
        self.curriculum_stagnations = int(state.get("curriculum_stagnations", self.curriculum_stagnations))
        self._recent_outcomes.clear()
        for item in list(state.get("recent_outcomes", []))[-self._recent_outcomes.maxlen:]:
            if isinstance(item, dict):
                self._recent_outcomes.append(dict(item))

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
        target_pos = None
        if self.target is not None:
            goal = self.target.crossed_position if self.target.kind == "cross_door" and self.target.crossed_position is not None else self.target.position
            target_pos = [float(goal[0]), float(goal[1])]
        out = {
            "environment_type": "ai2thor",
            "render_mode": "embodied3d",
            "agent": [float(pos[0]), float(pos[1])],
            "target": target_pos,
            "predicted_position": None,
            "world_bounds": [float(v) for v in self._bounds],
            "yaw": float(yaw),
            "camera_horizon": float(self._horizon_from_metadata(md)),
            "energy": float(self.energy),
            "depth_left": float(left * 5.0),
            "depth_center": float(center * 5.0),
            "depth_right": float(right * 5.0),
            "near_obstacle_fraction": float(near),
            "curriculum_level": int(self.target.level if self.target else self.config.curriculum_level),
            "task_kind": self.target.kind if self.target else "none",
            "target_label": self.target.label if self.target else "none",
            "goal_signal": self.config.goal_signal,
            "allowed_actions": list(self.valid_action_indices()),
            "curriculum_stats": self._curriculum_stats(),
            "stagnation_steps": int(self._stagnation_count),
            "stagnation_limit": int(self.config.stagnation_steps),
            "camera": self._thumbnail_payload(self.render_rgb()),
        }
        if predicted_obs is not None:
            p = np.asarray(predicted_obs, dtype=np.float32).reshape(-1)
            out["predicted_sensor"] = [float(v) for v in p[:6].tolist()]
        return out

    def describe(self) -> str:
        pos = self._position_from_metadata(self._metadata())
        _f, _r, dist, bearing = self._goal_local()
        return (
            f"scene={self.config.scene} level={self.target.level if self.target else self.config.curriculum_level} "
            f"task={self.target.kind if self.target else 'none'} target={self.target.label if self.target else 'none'} "
            f"agent=({pos[0]:+.2f},{pos[1]:+.2f}) yaw={self._yaw_from_metadata(self._metadata()):.1f} "
            f"distance={dist:.2f} bearing={bearing:+.1f} stagnation={self._stagnation_count}/{self.config.stagnation_steps}"
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
