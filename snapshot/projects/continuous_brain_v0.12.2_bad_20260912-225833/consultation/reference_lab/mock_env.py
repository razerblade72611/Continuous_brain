"""
Minimal stand-in gridworld for exercising the Milestone 1 code end to end.

This is NOT a reconstruction of your actual simulator. Energy/hydration
decay rates, resource placement, and collision rules here are simplified
and not meant to match your real dynamics. Swap this module out for your
real environment; the only thing downstream code depends on is the
observation shape ((4, 9, 9) grid, 3-value body state) and the 5-action
space, matching your documented v0.12 input shapes.
"""

from __future__ import annotations
import numpy as np

GRID_SIZE = 9
N_CHANNELS = 4  # walls, self, food, water
ACTIONS = ["rest", "up", "down", "left", "right"]
_MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


class MockGridWorld:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.pos = None
        self.energy = self.hydration = self.integrity = 1.0
        self.food_pos = self.water_pos = None
        self.t = 0
        self.reset()

    def reset(self):
        self.pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        self.energy, self.hydration, self.integrity = 1.0, 1.0, 1.0
        self.food_pos = self._random_empty_cell()
        self.water_pos = self._random_empty_cell()
        self.t = 0
        return self._obs()

    def _random_empty_cell(self):
        while True:
            p = self.rng.integers(1, GRID_SIZE - 1, size=2)
            if not np.array_equal(p, self.pos):
                return p

    def _obs(self):
        grid = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[0, 0, :] = 1.0
        grid[0, -1, :] = 1.0
        grid[0, :, 0] = 1.0
        grid[0, :, -1] = 1.0  # channel 0: walls
        grid[1, self.pos[0], self.pos[1]] = 1.0
        grid[2, self.food_pos[0], self.food_pos[1]] = 1.0
        grid[3, self.water_pos[0], self.water_pos[1]] = 1.0
        body = np.array([self.energy, self.hydration, self.integrity], dtype=np.float32)
        return grid, body

    def step(self, action_idx: int):
        move = _MOVES[action_idx]
        proposed = np.clip(self.pos + np.array(move), 1, GRID_SIZE - 2)
        collided = action_idx != 0 and np.array_equal(proposed, self.pos)
        self.pos = proposed

        self.energy = max(0.0, self.energy - 0.02)
        self.hydration = max(0.0, self.hydration - 0.03)

        resource = "none"
        if np.array_equal(self.pos, self.food_pos):
            self.energy = min(1.0, self.energy + 0.5)
            self.food_pos = self._random_empty_cell()
            resource = "food"
        if np.array_equal(self.pos, self.water_pos):
            self.hydration = min(1.0, self.hydration + 0.5)
            self.water_pos = self._random_empty_cell()
            resource = "water"
        if collided:
            self.integrity = max(0.0, self.integrity - 0.1)

        reward = -0.01 + (1.0 if resource != "none" else 0.0) - (0.5 if collided else 0.0)
        self.t += 1
        done = self.energy <= 0 or self.hydration <= 0 or self.integrity <= 0 or self.t >= 200
        return self._obs(), reward, done, {"collided": collided, "resource": resource}
