"""Survival world with need management under partial observability.

Drain rates are set so a competent agent that finds resources can survive
well beyond the passive-death horizon; a non-navigating agent still dies early.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Observation:
    thirst: float
    hunger: float
    health: float
    x: int
    y: int
    known_water: bool
    known_food: bool
    water_dist: float
    food_dist: float
    step: int
    on_water: bool
    on_food: bool


class SurvivalWorld:
    def __init__(self, seed: int, size: int = 12, max_steps: int = 500) -> None:
        self.rng = random.Random(seed)
        self.size = size
        self.max_steps = max_steps
        # Two water sources, one food, so exploration has a realistic chance
        self.water_positions = [
            (self.rng.randint(0, size - 1), self.rng.randint(0, size - 1)),
            (self.rng.randint(0, size - 1), self.rng.randint(0, size - 1)),
        ]
        self.food_pos = (self.rng.randint(0, size - 1), self.rng.randint(0, size - 1))
        self.reset()

    def reset(self) -> Observation:
        self.x = self.rng.randint(0, self.size - 1)
        self.y = self.rng.randint(0, self.size - 1)
        self.thirst = 0.2
        self.hunger = 0.2
        self.health = 1.0
        self.tick = 0
        self.known_water = False
        self.known_food = False
        self.alive = True
        self.death_cause = None
        return self._obs()

    def _nearest_water_dist(self) -> float:
        return min(abs(self.x - wx) + abs(self.y - wy) for wx, wy in self.water_positions)

    def _obs(self) -> Observation:
        on_water = (self.x, self.y) in self.water_positions
        on_food = (self.x, self.y) == self.food_pos
        if on_water:
            self.known_water = True
        if on_food:
            self.known_food = True
        wd = self._nearest_water_dist() if self.known_water else -1.0
        fd = abs(self.x - self.food_pos[0]) + abs(self.y - self.food_pos[1]) if self.known_food else -1.0
        return Observation(
            thirst=self.thirst,
            hunger=self.hunger,
            health=self.health,
            x=self.x,
            y=self.y,
            known_water=self.known_water,
            known_food=self.known_food,
            water_dist=wd,
            food_dist=fd,
            step=self.tick,
            on_water=on_water,
            on_food=on_food,
        )

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict]:
        if not self.alive:
            return self._obs(), 0.0, True, {"cause": self.death_cause}

        self.tick += 1
        reward = 0.0
        info: Dict = {}

        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(action, (0, 0))
        self.x = max(0, min(self.size - 1, self.x + dx))
        self.y = max(0, min(self.size - 1, self.y + dy))

        on_water = (self.x, self.y) in self.water_positions
        on_food = (self.x, self.y) == self.food_pos

        if action == "drink" and on_water:
            self.thirst = max(0.0, self.thirst - 0.7)
            reward += 2.0
        if action == "eat" and on_food:
            self.hunger = max(0.0, self.hunger - 0.7)
            reward += 2.0

        # Slower drain so navigation can succeed
        self.thirst = min(1.0, self.thirst + 0.006)
        self.hunger = min(1.0, self.hunger + 0.005)

        if self.thirst > 0.75 or self.hunger > 0.75:
            self.health -= 0.04
            reward -= 0.3
        if self.thirst > 0.95:
            self.alive = False
            self.death_cause = "thirst"
            reward -= 10.0
        elif self.hunger > 0.95:
            self.alive = False
            self.death_cause = "hunger"
            reward -= 10.0
        elif self.health <= 0:
            self.alive = False
            self.death_cause = "health"
            reward -= 10.0

        done = not self.alive or self.tick >= self.max_steps
        if done and self.alive:
            reward += 5.0
            info["cause"] = "timeout"
        else:
            info["cause"] = self.death_cause

        return self._obs(), reward, done, info
