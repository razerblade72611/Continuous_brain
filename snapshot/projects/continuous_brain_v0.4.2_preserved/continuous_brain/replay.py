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
    # v0.4.2: tag experience by curriculum skill so later skills can rehearse
    # earlier ones without mixing incompatible action masks.
    skill_level: int | None = None
    valid_actions: tuple[int, ...] | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self._items: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def level_counts(self) -> dict[int | None, int]:
        out: dict[int | None, int] = {}
        for item in self._items:
            out[item.skill_level] = out.get(item.skill_level, 0) + 1
        return out

    def sample(
        self,
        n: int,
        *,
        current_level: int | None = None,
        current_fraction: float = 0.80,
    ) -> list[Transition]:
        """Sample replay with optional current-skill emphasis.

        v0.4.2 keeps transitions tagged by skill level.  During L1+ training we
        bias most of the batch toward the current skill while reserving a slice
        for earlier skills.  This provides lightweight rehearsal against
        catastrophic forgetting without throwing away diverse experience.
        """
        if not self._items:
            return []
        n = min(int(n), len(self._items))
        if current_level is None:
            return self._rng.sample(list(self._items), n)

        frac = max(0.0, min(1.0, float(current_fraction)))
        items = list(self._items)
        current = [x for x in items if x.skill_level == current_level]
        prior = [x for x in items if x.skill_level is not None and x.skill_level < current_level]

        want_current = min(len(current), int(round(n * frac)))
        chosen: list[Transition] = self._rng.sample(current, want_current) if want_current else []
        remaining = n - len(chosen)

        if remaining and prior:
            take = min(len(prior), remaining)
            chosen.extend(self._rng.sample(prior, take))
            remaining -= take
        if remaining:
            chosen_ids = {id(x) for x in chosen}
            pool = [x for x in items if id(x) not in chosen_ids]
            if pool:
                chosen.extend(self._rng.sample(pool, min(remaining, len(pool))))

        self._rng.shuffle(chosen)
        return chosen
