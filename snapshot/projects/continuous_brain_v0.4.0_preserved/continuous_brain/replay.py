from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    prev_hidden: np.ndarray
    prev_action: int
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    visual_embedding: np.ndarray | None = None
    next_visual_embedding: np.ndarray | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self._items: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, n: int) -> list[Transition]:
        if not self._items:
            return []
        n = min(n, len(self._items))
        return self._rng.sample(list(self._items), n)
