"""Baseline (a): uniform random over Crafter's 17 actions; never speaks."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    name = "uniform_random"
    privileged = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, obs, info=None):
        pass

    def act(self, obs, info=None):
        return int(self.rng.integers(17)), None
