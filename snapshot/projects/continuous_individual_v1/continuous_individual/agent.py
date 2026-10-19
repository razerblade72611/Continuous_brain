"""Agent with remembered map in the canonical store and directed navigation."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Set, Tuple

from .store import CanonicalStore
from .world import Observation


class NeedController(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),  # explore, seek_water, seek_food
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def obs_to_tensor(obs: Observation) -> torch.Tensor:
    return torch.tensor(
        [
            obs.thirst,
            obs.hunger,
            obs.health,
            1.0 if obs.known_water else 0.0,
            1.0 if obs.known_food else 0.0,
            obs.water_dist / 20.0 if obs.water_dist >= 0 else -1.0,
            obs.food_dist / 20.0 if obs.food_dist >= 0 else -1.0,
        ],
        dtype=torch.float32,
    )


class Individual:
    """All behaviorally relevant mutable state lives in the store."""

    def __init__(self, store: Optional[CanonicalStore] = None) -> None:
        self.store = store or CanonicalStore()
        self.controller = NeedController()
        if self.store.has("controller.weights"):
            self.controller.load_state_dict(self.store.read("controller.weights"))
        else:
            self._save_weights()
        if not self.store.has("map"):
            self.store.write("map", {"water": [], "food": None, "visited": []})

    def _save_weights(self) -> None:
        self.store.write("controller.weights", self.controller.state_dict())

    def _map(self) -> dict:
        return self.store.read("map")

    def _update_map(self, obs: Observation) -> None:
        m = self._map()
        visited: Set[Tuple[int, int]] = set(tuple(p) for p in m["visited"])
        visited.add((obs.x, obs.y))
        m["visited"] = list(visited)
        if obs.on_water and (obs.x, obs.y) not in [tuple(p) for p in m["water"]]:
            m["water"].append((obs.x, obs.y))
        if obs.on_food:
            m["food"] = (obs.x, obs.y)
        self.store.write("map", m)

    def rebuild_from_store(self) -> "Individual":
        return Individual(store=self.store)

    def _direction_toward(self, ox: int, oy: int, tx: int, ty: int) -> str:
        if abs(tx - ox) >= abs(ty - oy):
            return "right" if tx > ox else "left"
        return "down" if ty > oy else "up"

    def _explore_move(self, obs: Observation) -> str:
        m = self._map()
        visited = set(tuple(p) for p in m["visited"])
        # Prefer an adjacent unvisited cell
        for d, (dx, dy) in [("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))]:
            nx, ny = obs.x + dx, obs.y + dy
            if 0 <= nx < 12 and 0 <= ny < 12 and (nx, ny) not in visited:
                return d
        # Otherwise cycle
        return ["up", "right", "down", "left"][obs.step % 4]

    def _move_to_resource(self, obs: Observation, kind: str) -> str:
        m = self._map()
        if kind == "water" and m["water"]:
            # nearest known water
            best = min(m["water"], key=lambda p: abs(p[0] - obs.x) + abs(p[1] - obs.y))
            return self._direction_toward(obs.x, obs.y, best[0], best[1])
        if kind == "food" and m["food"] is not None:
            fx, fy = m["food"]
            return self._direction_toward(obs.x, obs.y, fx, fy)
        return self._explore_move(obs)

    def routine_choice(self, obs: Observation) -> str:
        if obs.thirst > 0.45 and obs.known_water:
            return "seek_water"
        if obs.hunger > 0.45 and obs.known_food:
            return "seek_food"
        if obs.thirst > 0.6:
            return "explore"
        return "explore"

    def learned_choice(self, obs: Observation) -> str:
        with torch.no_grad():
            logits = self.controller(obs_to_tensor(obs).unsqueeze(0))
            idx = int(logits.argmax(-1).item())
        return ["explore", "seek_water", "seek_food"][idx]

    def act(self, obs: Observation, use_learned: bool) -> str:
        self._update_map(obs)
        choice = self.learned_choice(obs) if use_learned else self.routine_choice(obs)

        if choice == "seek_water" and obs.on_water:
            return "drink"
        if choice == "seek_food" and obs.on_food:
            return "eat"
        if choice == "seek_water":
            return self._move_to_resource(obs, "water")
        if choice == "seek_food":
            return self._move_to_resource(obs, "food")
        return self._explore_move(obs)
