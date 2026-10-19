from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np


class Environment:
    observation_dim: int
    action_dim: int
    action_names: tuple[str, ...]

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    def valid_action_indices(self) -> tuple[int, ...]:
        """Actions currently available to the policy. Environments may curriculum-mask actions."""
        return tuple(range(int(self.action_dim)))

    def valid_action_indices_for_level(self, _level: int | None) -> tuple[int, ...]:
        """Action mask for a replayed curriculum level; default environments are unmasked."""
        return self.valid_action_indices()

    def current_skill_level(self) -> int | None:
        """Current curriculum skill tag for replay/diagnostics, if the environment has one."""
        return None

    def exploration_floor(self) -> float:
        """Optional curriculum-specific epsilon floor used when a new skill is unlocked."""
        return 0.0

    def describe(self) -> str:
        return ""

    def render_rgb(self, size: int = 96) -> np.ndarray:
        raise NotImplementedError("environment does not provide RGB observations")

    def render_depth(self) -> np.ndarray | None:
        return None

    def visual_snapshot(self, predicted_obs: np.ndarray | None = None) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None

    def recover_sensor_fault(self) -> dict:
        return {"attempted": False, "recovered": False, "method": "unsupported", "fatal": True}

    def close(self) -> None:
        return None


@dataclass
class ForagingWorld(Environment):
    """Tiny endless environment used to prove the continuous loop works.

    The agent moves in [-1,1]^2. Reaching a target gives reward and spawns a new target.
    Energy slowly falls and is partially restored by reaching targets.
    """

    seed: int = 7
    max_episode_steps: int = 500

    observation_dim: int = 6
    action_dim: int = 5
    action_names: tuple[str, ...] = ("idle", "up", "down", "left", "right")

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.agent = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        self.reset()

    def _spawn_target(self) -> None:
        self.target[:] = [self.rng.uniform(-0.9, 0.9), self.rng.uniform(-0.9, 0.9)]

    def _obs(self) -> np.ndarray:
        delta = self.target - self.agent
        return np.array([
            self.agent[0], self.agent[1], delta[0], delta[1], self.energy, self.last_reward
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.agent[:] = [self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.5, 0.5)]
        self._spawn_target()
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        previous_dist = float(np.linalg.norm(self.target - self.agent))
        move = {
            0: (0.0, 0.0), 1: (0.0, 0.08), 2: (0.0, -0.08),
            3: (-0.08, 0.0), 4: (0.08, 0.0),
        }.get(int(action), (0.0, 0.0))
        self.agent += np.asarray(move, dtype=np.float32)
        self.agent[:] = np.clip(self.agent, -1.0, 1.0)
        self.energy = max(0.0, self.energy - 0.002)

        dist = float(np.linalg.norm(self.target - self.agent))
        # Dense potential-based shaping: moving toward the target is immediately useful,
        # while the +1.0 target reward remains the main objective.
        reward = -0.002 + 0.60 * (previous_dist - dist)
        reached = dist < 0.12
        if reached:
            reward += 1.0
            self.energy = min(1.0, self.energy + 0.35)
            self._spawn_target()

        done = self.energy <= 0.0 or self.steps >= self.max_episode_steps
        self.last_reward = reward
        return self._obs(), float(reward), done, {"distance": dist, "reached": reached}


    def render_rgb(self, size: int = 96) -> np.ndarray:
        """Render the tiny world as an RGB camera-like frame for vision backbones."""
        size = max(32, int(size))
        img = np.full((size, size, 3), 20, dtype=np.uint8)
        # subtle grid gives the visual encoder stable spatial structure
        for i in range(0, size, max(8, size // 8)):
            img[i:i + 1, :, :] = 35
            img[:, i:i + 1, :] = 35

        def pix(pos):
            x = int(round((float(pos[0]) + 1.0) * 0.5 * (size - 1)))
            y = int(round((1.0 - (float(pos[1]) + 1.0) * 0.5) * (size - 1)))
            return x, y

        yy, xx = np.ogrid[:size, :size]
        tx, ty = pix(self.target)
        ax, ay = pix(self.agent)
        tr = max(3, size // 18)
        ar = max(3, size // 20)
        tmask = (xx - tx) ** 2 + (yy - ty) ** 2 <= tr ** 2
        amask = (xx - ax) ** 2 + (yy - ay) ** 2 <= ar ** 2
        img[tmask] = np.array([240, 190, 45], dtype=np.uint8)
        img[amask] = np.array([65, 145, 245], dtype=np.uint8)
        # energy bar at the bottom is part of the sensory scene
        bar = int(round((size - 4) * float(self.energy)))
        img[size - 4:size - 2, 2:2 + bar] = np.array([85, 210, 120], dtype=np.uint8)
        return img


    def visual_snapshot(self, predicted_obs: np.ndarray | None = None) -> dict:
        pred = None
        if predicted_obs is not None:
            p = np.asarray(predicted_obs, dtype=np.float32).reshape(-1)
            if p.size >= 2:
                pred = [float(p[0]), float(p[1])]
        return {
            "environment_type": "foraging",
            "render_mode": "2d",
            "agent": [float(self.agent[0]), float(self.agent[1])],
            "target": [float(self.target[0]), float(self.target[1])],
            "predicted_position": pred,
            "world_bounds": [-1.0, 1.0, -1.0, 1.0],
            "energy": float(self.energy),
            "curriculum_level": None,
            "task_kind": "foraging",
            "target_label": "resource",
        }

    def describe(self) -> str:
        return (
            f"agent=({self.agent[0]:+.2f},{self.agent[1]:+.2f}) "
            f"target=({self.target[0]:+.2f},{self.target[1]:+.2f}) energy={self.energy:.2f}"
        )
