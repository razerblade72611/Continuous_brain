from __future__ import annotations

from dataclasses import dataclass
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
            a = self.rng.choice((1, 2, 3))
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
            return abs(bearing) <= 15.0, "oriented"
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
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        prev_f, prev_r, prev_dist, prev_bearing = self._goal_local()
        interaction_success, primitive = self._apply_intent(action)
        f, r, dist, bearing = self._goal_local()

        # Dense shaping uses privileged task supervision, but the policy can have that
        # supervision masked from its observation via goal_signal/dropout.
        reward = -0.003
        level = int(self.target.level if self.target else self.config.curriculum_level)
        if self.target and self.target.kind == "orient":
            reward += 0.003 * (abs(prev_bearing) - abs(bearing))
        else:
            reward += 0.18 * (prev_dist - dist)
        if self.last_action_success < 0.5:
            reward -= 0.02

        success, reason = self._task_success(interaction_success)
        if success:
            reward += 1.0
        elif reason == "subgoal_complete":
            reward += 0.5

        self.last_reward = float(reward)
        self.energy = 1.0  # placeholder until real battery/body homeostasis is connected
        done = bool(success or self.steps >= int(self.config.max_episode_steps))
        info = {
            "distance": float(dist),
            "bearing": float(bearing),
            "reached": bool(success),
            "task_success": bool(success),
            "task_reason": reason,
            "curriculum_level": level,
            "task_kind": self.target.kind if self.target else "none",
            "target_label": self.target.label if self.target else "none",
            "primitive_action": primitive,
            "last_action_success": bool(self.last_action_success),
            "goal_signal": self.config.goal_signal,
        }
        return self._obs(), float(reward), done, info

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
            f"distance={dist:.2f} bearing={bearing:+.1f}"
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
