"""Persistent, deterministic egocentric development room (not AI2-THOR).

The brain receives a line-of-sight beacon bearing/range sensor, depth sectors,
action acknowledgement and reward. The top-down evaluator view is never input.
"""
from __future__ import annotations
import base64
import math
import random
import numpy as np
from .environment import Environment


class DevelopmentRoom(Environment):
    observation_dim = 6
    action_dim = 5
    action_names = ("idle", "advance", "turn_left", "turn_right", "interact")
    exact_resume = True

    def __init__(self, seed=7, max_episode_steps=300):
        self.rng = random.Random(seed)
        self.max_episode_steps = max_episode_steps
        self.position = np.array([0.0, 0.0])
        self.target = np.array([1.5, 1.5])
        self.yaw = 0.0
        self.energy = 1.0
        self.last_reward = 0.0
        self.last_action_success = 1.0
        self.steps = 0
        self.tasks = 0
        self.obstacles = [(-1.8, -1.1, 0.5, 1.3), (1.0, 1.7, -1.8, -1.0)]
        self.reset()

    def _blocked(self, x, z, margin=0.10):
        if abs(x) > 3-margin or abs(z) > 3-margin:
            return True
        return any(x0-margin <= x <= x1+margin and z0-margin <= z <= z1+margin
                   for x0, x1, z0, z1 in self.obstacles)

    def _point(self):
        for _ in range(1000):
            p = np.array([self.rng.uniform(-2.5, 2.5), self.rng.uniform(-2.5, 2.5)])
            if not self._blocked(*p, margin=0.4):
                return p
        raise RuntimeError("Cannot sample development position")

    def _ray(self, angle, limit=6.0):
        direction = np.array([math.sin(angle), math.cos(angle)])
        for d in np.arange(0.05, limit + 0.05, 0.05):
            if self._blocked(*(self.position + d * direction), margin=0):
                return float(d)
        return limit

    def _beacon(self):
        delta = self.target - self.position
        dist = float(np.linalg.norm(delta))
        yaw = math.radians(self.yaw)
        f = float(np.dot(delta, [math.sin(yaw), math.cos(yaw)]))
        r = float(np.dot(delta, [math.cos(yaw), -math.sin(yaw)]))
        angle = math.atan2(r, f)
        visible = abs(angle) <= math.radians(60) and dist <= 4.5
        if visible:
            visible = self._ray(yaw + angle, max(0.05, dist)) >= dist - 0.08
        return f, r, dist, angle, visible

    def _obs(self):
        f, r, _, _, visible = self._beacon()
        yaw = math.radians(self.yaw)
        left, center, right = [min(1.0, self._ray(yaw+a)/3) for a in (-0.5, 0, 0.5)]
        return np.array([f/5 if visible else 0, r/5 if visible else 0,
                         center, left-right, self.last_action_success,
                         np.clip(self.last_reward, -1, 1)], dtype=np.float32)

    def reset(self):
        self.position = self._point()
        self.target = self._point()
        self.yaw = self.rng.uniform(0, 360)
        self.steps = 0
        self.energy = 1.0
        self.last_reward = 0.0
        self.last_action_success = 1.0
        return self._obs()

    def finish_intention(self):
        # Curriculum generator changes the beacon, not the individual or body.
        self.tasks += 1
        self.target = self._point()
        return self._obs()

    def step(self, action):
        if action not in range(5):
            raise ValueError("Invalid primitive")
        _, _, before, _, was_visible = self._beacon()
        self.steps += 1
        self.last_action_success = 1.0
        if action == 1:
            a = math.radians(self.yaw)
            candidate = self.position + np.array([math.sin(a), math.cos(a)]) * 0.2
            if self._blocked(*candidate):
                self.last_action_success = 0.0
            else:
                self.position = candidate
        elif action in (2, 3):
            self.yaw = (self.yaw + (-15 if action == 2 else 15)) % 360
        self.energy = float(np.clip(self.energy + (0.004 if action == 0 else -0.001), 0, 1))
        _, _, after, angle, visible = self._beacon()
        inspected = action == 4 and visible and after <= 0.55
        if action == 4 and not inspected:
            self.last_action_success = 0.0
        reward = -0.002 - 0.03 * (1-self.last_action_success)
        # Teacher/evaluator shaping; not an omniscient coordinate input.
        if was_visible and visible:
            reward += 0.3 * (before-after)
        if inspected:
            reward += 1.0
        self.last_reward = reward
        done = self.steps >= self.max_episode_steps or self.energy <= 0
        return self._obs(), float(reward), bool(done), {
            "reached": inspected, "inspected": inspected, "distance": after,
            "bearing": math.degrees(angle), "last_action_success": bool(self.last_action_success),
            "infrastructure_fault": False,
        }

    def render_rgb(self, size=64):
        h, w = size, size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:h//2] = [35, 48, 65]
        img[h//2:] = [58, 53, 45]
        yaw = math.radians(self.yaw)
        for col in range(w):
            angle = (col/max(1, w-1)-0.5) * math.radians(120)
            dist = self._ray(yaw+angle)
            extent = min(h//2, int(h / max(0.5, dist) * 0.3))
            shade = int(180/(1+dist*0.2))
            img[h//2-extent:h//2+extent, col] = [shade, shade, shade+10]
        _, _, dist, angle, visible = self._beacon()
        if visible:
            col = int((angle/math.radians(120)+0.5) * (w-1))
            extent = max(2, min(h//3, int(h*0.15/max(0.2, dist))))
            img[h//2-extent:h//2+extent, max(0,col-2):min(w,col+3)] = [250, 195, 45]
        return img

    def state_dict(self):
        return {"position": self.position, "target": self.target, "yaw": self.yaw,
                "energy": self.energy, "last_reward": self.last_reward,
                "last_action_success": self.last_action_success, "steps": self.steps,
                "tasks": self.tasks, "rng": self.rng.getstate(),
                "max_episode_steps": self.max_episode_steps}

    def load_state_dict(self, state):
        for key in ("position", "target", "yaw", "energy", "last_reward", "last_action_success",
                    "steps", "tasks", "max_episode_steps"):
            setattr(self, key, state[key])
        self.rng.setstate(state["rng"])

    def describe(self):
        return "development-room:v1"  # no coordinates in cognition

    def visual_snapshot(self, predicted_obs=None):
        rgb = self.render_rgb()
        return {"agent": self.position.tolist(), "target": self.target.tolist(),
                "world_bounds": [-3, 3, -3, 3], "yaw": self.yaw,
                "camera": {"rgb_b64": base64.b64encode(rgb.tobytes()).decode(),
                           "width": rgb.shape[1], "height": rgb.shape[0],
                           "seq": self.steps + self.tasks * 100000},
                "render_mode": "egocentric", "energy": self.energy}
