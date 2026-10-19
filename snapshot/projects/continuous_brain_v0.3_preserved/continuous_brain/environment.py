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

    def describe(self) -> str:
        return ""


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

    def describe(self) -> str:
        return (
            f"agent=({self.agent[0]:+.2f},{self.agent[1]:+.2f}) "
            f"target=({self.target[0]:+.2f},{self.target[1]:+.2f}) energy={self.energy:.2f}"
        )
