from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Any

from .learning import load_json, utc_now, write_json_atomic
from .models import Action, ActionKind
from .navigation import cardinal_yaw, normalize_yaw, rotation_actions


ACTIONS = (
    ActionKind.MOVE_AHEAD.value,
    ActionKind.ROTATE_LEFT.value,
    ActionKind.ROTATE_RIGHT.value,
)


def _yaw_bucket(yaw: float) -> int:
    return int(round(normalize_yaw(yaw) / 90.0)) % 4


def _desired_yaw(source: dict[str, float], target: dict[str, float]) -> float:
    return cardinal_yaw(float(target["x"]) - float(source["x"]), float(target["z"]) - float(source["z"]))


@dataclass(frozen=True)
class SkillState:
    relative_heading: int
    distance_bucket: int

    def key(self) -> str:
        return f"h{self.relative_heading}:d{self.distance_bucket}"


class NavigationSkill:
    """Small persistent tabular motor policy trained from navigation transitions.

    The persistent world map handles room topology; this policy learns the local
    turn/move mechanics used while following a route. It is deliberately small so
    it trains rapidly on CPU and survives across runs as a JSON checkpoint.
    """

    def __init__(self, path: str | Path | None = None, alpha: float = 0.25, gamma: float = 0.85):
        self.path = None if path is None else Path(path)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q: dict[str, dict[str, float]] = {}
        self.visits: dict[str, int] = {}
        self.transitions = 0
        self.updated_at: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        payload = load_json(self.path, {})
        self.alpha = float(payload.get("alpha", self.alpha))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.transitions = int(payload.get("transitions", 0))
        self.updated_at = payload.get("updated_at")
        self.visits = {str(k): int(v) for k, v in (payload.get("visits") or {}).items()}
        self.q = {
            str(key): {action: float(value) for action, value in (row or {}).items() if action in ACTIONS}
            for key, row in (payload.get("q") or {}).items()
            if isinstance(row, dict)
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "tabular_local_navigation",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "transitions": self.transitions,
            "updated_at": self.updated_at or utc_now(),
            "visits": dict(sorted(self.visits.items())),
            "q": {key: dict(sorted(row.items())) for key, row in sorted(self.q.items())},
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("NavigationSkill.save requires a path.")
        self.updated_at = utc_now()
        write_json_atomic(target, self.as_dict())
        self.path = target

    @staticmethod
    def state(
        position: dict[str, float],
        yaw: float,
        target_position: dict[str, float],
        grid_size: float = 0.25,
    ) -> SkillState:
        desired = _desired_yaw(position, target_position)
        relative = (_yaw_bucket(desired) - _yaw_bucket(yaw)) % 4
        distance = math.hypot(
            float(target_position["x"]) - float(position["x"]),
            float(target_position["z"]) - float(position["z"]),
        )
        if distance <= grid_size * 1.25:
            bucket = 0
        elif distance <= grid_size * 4.0:
            bucket = 1
        else:
            bucket = 2
        return SkillState(relative, bucket)

    def _row(self, state: SkillState) -> dict[str, float]:
        return self.q.setdefault(state.key(), {action: 0.0 for action in ACTIONS})

    def update(self, state: SkillState, action: str, reward: float, next_state: SkillState | None) -> None:
        if action not in ACTIONS:
            return
        row = self._row(state)
        current = row.get(action, 0.0)
        future = 0.0
        if next_state is not None:
            future_row = self._row(next_state)
            future = max(future_row.values()) if future_row else 0.0
        row[action] = current + self.alpha * (float(reward) + self.gamma * future - current)
        self.visits[state.key()] = self.visits.get(state.key(), 0) + 1
        self.transitions += 1

    def best_action(self, state: SkillState, epsilon: float = 0.0, rng: random.Random | None = None) -> str:
        rng = rng or random
        if epsilon > 0 and rng.random() < epsilon:
            return rng.choice(ACTIONS)
        row = self._row(state)
        # Deterministic tie order keeps behavior reproducible.
        return max(ACTIONS, key=lambda action: (row.get(action, 0.0), -ACTIONS.index(action)))

    def merge(self, other: "NavigationSkill") -> None:
        # Weighted average by state visits avoids multiplying preexisting weights.
        keys = set(self.q) | set(other.q)
        for key in keys:
            mine_n = int(self.visits.get(key, 0))
            other_n = int(other.visits.get(key, 0))
            mine = self.q.setdefault(key, {action: 0.0 for action in ACTIONS})
            theirs = other.q.get(key, {})
            total = mine_n + other_n
            if total <= 0:
                continue
            for action in ACTIONS:
                mine[action] = (
                    mine.get(action, 0.0) * mine_n + float(theirs.get(action, 0.0)) * other_n
                ) / total
            self.visits[key] = total
        self.transitions += other.transitions

    def route_actions(
        self,
        path: list[dict[str, float]],
        initial_yaw: float,
        final_target: dict[str, float],
        rotate_step_degrees: float = 90.0,
        min_state_visits: int = 4,
    ) -> list[Action]:
        """Turn an already learned topological route into primitive actions.

        Q values only decide among mechanically valid local choices. When a state
        has insufficient training, deterministic navigation remains the fallback.
        """
        if not path:
            return []
        actions: list[Action] = []
        yaw = normalize_yaw(initial_yaw)
        for previous, current in zip(path, path[1:]):
            desired = _desired_yaw(previous, current)
            relative = (_yaw_bucket(desired) - _yaw_bucket(yaw)) % 4
            state = self.state(previous, yaw, current)
            trained = self.visits.get(state.key(), 0) >= int(min_state_visits)
            if relative == 0:
                actions.append(Action(ActionKind.MOVE_AHEAD.value, {}, "Follow learned room route."))
                yaw = desired
                continue
            if relative == 1:
                actions.append(Action(ActionKind.ROTATE_RIGHT.value, {}, "Follow learned local navigation policy."))
                yaw = normalize_yaw(yaw + rotate_step_degrees)
            elif relative == 3:
                actions.append(Action(ActionKind.ROTATE_LEFT.value, {}, "Follow learned local navigation policy."))
                yaw = normalize_yaw(yaw - rotate_step_degrees)
            else:
                # For a 180-degree turn, use the trained preference if meaningful;
                # otherwise fall back to deterministic shortest-turn behavior.
                if trained:
                    row = self._row(state)
                    turn = max(
                        (ActionKind.ROTATE_LEFT.value, ActionKind.ROTATE_RIGHT.value),
                        key=lambda action: row.get(action, 0.0),
                    )
                    actions.extend([Action(turn, {}, "Follow learned 180-degree turn preference.")] * 2)
                else:
                    actions.extend(rotation_actions(yaw, desired, rotate_step_degrees))
                yaw = desired
            actions.append(Action(ActionKind.MOVE_AHEAD.value, {}, "Follow learned room route."))

        # Face the final semantic target after route arrival using deterministic body mechanics.
        if path:
            dx = float(final_target["x"]) - float(path[-1]["x"])
            dz = float(final_target["z"]) - float(path[-1]["z"])
            if abs(dx) + abs(dz) > 1e-6:
                desired_final = normalize_yaw(math.degrees(math.atan2(dx, dz)))
                actions.extend(rotation_actions(yaw, desired_final, rotate_step_degrees))
        return actions
