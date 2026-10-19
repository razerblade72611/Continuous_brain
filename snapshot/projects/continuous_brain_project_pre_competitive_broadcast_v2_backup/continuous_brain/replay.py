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

    def state_dict(self, limit: int | None = None) -> dict:
        """Serialize a bounded tail of replay for restart-safe continual learning."""
        items = list(self._items)
        if limit is not None and int(limit) <= 0:
            items = []
        elif limit is not None and len(items) > int(limit):
            cap = int(limit)
            indexed = list(enumerate(items))
            levels = sorted({t.skill_level for t in items}, key=lambda x: (-1 if x is None else int(x)))
            # Reserve a durable recent slice for every represented skill, then fill
            # the remainder with the newest overall experience. This prevents a long
            # L1/L2 run from evicting all mastered-skill replay at checkpoint time.
            reserve_each = max(1, cap // max(2, 2 * len(levels)))
            chosen_idx: set[int] = set()
            for level in levels:
                group = [(i, t) for i, t in indexed if t.skill_level == level]
                chosen_idx.update(i for i, _ in group[-reserve_each:])
            for i, _ in reversed(indexed):
                if len(chosen_idx) >= cap:
                    break
                chosen_idx.add(i)
            items = [t for i, t in indexed if i in chosen_idx]
            if len(items) > cap:
                items = items[-cap:]
        rows = []
        for t in items:
            rows.append({
                "observation": np.asarray(t.observation, dtype=np.float32),
                "prev_hidden": np.asarray(t.prev_hidden, dtype=np.float32),
                "prev_action": int(t.prev_action),
                "action": int(t.action),
                "reward": float(t.reward),
                "next_observation": np.asarray(t.next_observation, dtype=np.float32),
                "done": bool(t.done),
                "visual_embedding": None if t.visual_embedding is None else np.asarray(t.visual_embedding, dtype=np.float32),
                "next_visual_embedding": None if t.next_visual_embedding is None else np.asarray(t.next_visual_embedding, dtype=np.float32),
                "skill_level": None if t.skill_level is None else int(t.skill_level),
                "valid_actions": None if t.valid_actions is None else tuple(int(x) for x in t.valid_actions),
            })
        return {"format_version": 1, "items": rows}

    def load_state_dict(self, state: dict | None) -> int:
        """Restore replay rows from a trusted local checkpoint; returns loaded count."""
        if not isinstance(state, dict):
            return 0
        rows = list(state.get("items", []))
        loaded = 0
        for row in rows[-self._items.maxlen:]:
            if not isinstance(row, dict):
                continue
            try:
                self.append(Transition(
                    observation=np.asarray(row["observation"], dtype=np.float32),
                    prev_hidden=np.asarray(row["prev_hidden"], dtype=np.float32),
                    prev_action=int(row["prev_action"]),
                    action=int(row["action"]),
                    reward=float(row["reward"]),
                    next_observation=np.asarray(row["next_observation"], dtype=np.float32),
                    done=bool(row["done"]),
                    visual_embedding=None if row.get("visual_embedding") is None else np.asarray(row["visual_embedding"], dtype=np.float32),
                    next_visual_embedding=None if row.get("next_visual_embedding") is None else np.asarray(row["next_visual_embedding"], dtype=np.float32),
                    skill_level=None if row.get("skill_level") is None else int(row["skill_level"]),
                    valid_actions=None if row.get("valid_actions") is None else tuple(int(x) for x in row["valid_actions"]),
                ))
                loaded += 1
            except (KeyError, TypeError, ValueError):
                continue
        return loaded
