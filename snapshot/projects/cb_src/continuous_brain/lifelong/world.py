"""A fully observed, simulated body with two consumable resources.

The grid and resource beacons are explicit simulated sensors, not a claim of
learned physical perception. Only the training demonstrator uses shortest paths.
Body variables belong to the environment; predictions cannot write them.
"""
from __future__ import annotations

from collections import deque
import copy
import random

import numpy as np

SIZE = 9
ACTIONS = ("rest", "up", "down", "left", "right")
MOVES = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
BODY_NAMES = ("energy", "hydration", "integrity")
SENSOR_DIM = 18
MEMORY_DIM = 8
STYLE_DIM = 4


class ResourceWorld:
    def __init__(self, seed=1, *, kind="open", horizon=96):
        if kind not in ("open", "detour"):
            raise ValueError("kind must be open or detour")
        if horizon < 8:
            raise ValueError("horizon must be at least 8")
        self.seed, self.kind, self.horizon = int(seed), kind, int(horizon)
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self):
        self.walls = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.walls[0, :] = self.walls[-1, :] = 1
        self.walls[:, 0] = self.walls[:, -1] = 1
        self.gap = self.rng.randint(1, SIZE - 2)
        if self.kind == "detour":
            self.walls[1:-1, SIZE // 2] = 1
            self.walls[self.gap, SIZE // 2] = 0
        self.agent = (self.rng.randint(1, 3), self.rng.randint(1, 7))
        self.food = (self.rng.randint(5, 7), self.rng.randint(1, 7))
        self.water = self._free_cell({self.agent, self.food})
        self.body = np.array([self.rng.uniform(.42, .8), self.rng.uniform(.42, .8), 1.], np.float32)
        self.steps = self.collisions = self.harvests = 0
        self.food_count = self.water_count = 0
        self.last_action = 0
        self.last_positive = self.last_negative = 0.
        self.total_return = 0.
        self.visited = {self.agent}
        return self.observe()

    def _free_cell(self, exclude):
        cells = [(x, y) for y in range(1, SIZE - 1) for x in range(1, SIZE - 1)
                 if not self.walls[y, x] and (x, y) not in exclude]
        return self.rng.choice(cells)

    def observe(self):
        # Full grid sensor; coordinates are permitted in this benchmark.
        grid = np.zeros((4, SIZE, SIZE), dtype=np.float32)
        grid[0] = self.walls
        for channel, cell in ((1, self.food), (2, self.water), (3, self.agent)):
            grid[channel, cell[1], cell[0]] = 1
        x, y = self.agent
        sensors = np.zeros(SENSOR_DIM, np.float32)
        sensors[:3] = self.body
        sensors[3:7] = [(self.food[0] - x) / 8, (self.food[1] - y) / 8,
                         (self.water[0] - x) / 8, (self.water[1] - y) / 8]
        sensors[7 + self.last_action] = 1
        sensors[12:14] = [self.last_positive, self.last_negative]
        sensors[14] = self.steps / self.horizon
        # Local contact probes, in action order up/down/left/right, are encoded
        # in the camera too. The final three values give stable orientation.
        sensors[15:18] = [x / 8, y / 8, float(self.kind == "detour")]
        return {"grid": grid, "sensors": sensors}

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS):
            raise ValueError("action must be an integer from 0 through 4")
        if self.steps >= self.horizon or min(self.body) <= 0:
            raise RuntimeError("Episode ended; reset before taking another action")
        self.steps += 1
        before = self.body.copy()
        x, y = self.agent
        dx, dy = MOVES[action]
        proposed = (x + dx, y + dy)
        collision = bool(self.walls[proposed[1], proposed[0]])
        self.agent = self.agent if collision else proposed
        self.collisions += int(collision)
        self.body[0] -= .006 if action else .004
        self.body[1] -= .007
        if collision:
            self.body[2] -= .035
        elif action == 0:
            self.body[2] = min(1., self.body[2] + .025)
        resource = None
        if self.agent == self.food:
            self.body[0] = min(1., self.body[0] + .60)
            self.food = self._free_cell({self.agent, self.water})
            self.food_count += 1
            resource = "food"
        elif self.agent == self.water:
            self.body[1] = min(1., self.body[1] + .60)
            self.water = self._free_cell({self.agent, self.food})
            self.water_count += 1
            resource = "water"
        self.harvests += int(resource is not None)
        self.body = np.clip(self.body, 0., 1.).astype(np.float32)
        change = self.body - before
        positive = float(2. * np.maximum(change, 0.).sum())
        new_place = self.agent not in self.visited
        positive += .005 * new_place
        negative = float(2. * np.minimum(change, 0.).sum()) - .003 - .08 * collision
        reward = positive + negative
        self.visited.add(self.agent)
        self.last_action = int(action)
        self.last_positive, self.last_negative = positive, negative
        self.total_return += reward
        depleted = bool(min(self.body) <= 0)
        done = depleted or self.steps >= self.horizon
        info = {"collision": collision, "resource": resource, "positive": positive,
                "negative": negative, "depleted": depleted,
                "body_before": before.tolist(), "body_after": self.body.tolist(),
                "new_place": bool(new_place), "kind": self.kind}
        return self.observe(), float(reward), bool(done), info

    def state_dict(self):
        return {"seed": self.seed, "kind": self.kind, "horizon": self.horizon,
                "rng": self.rng.getstate(), "walls": self.walls.copy(),
                **{key: copy.deepcopy(getattr(self, key)) for key in (
                    "gap", "agent", "food", "water", "body", "steps", "collisions",
                    "harvests", "food_count", "water_count", "last_action",
                    "last_positive", "last_negative", "total_return", "visited")}}

    @classmethod
    def restore(cls, state):
        obj = cls(state["seed"], kind=state["kind"], horizon=state["horizon"])
        for key, value in state.items():
            if key != "rng":
                setattr(obj, key, copy.deepcopy(value))
        obj.rng.setstate(state["rng"])
        return obj


def teacher_action(world):
    """Authored shortest-path demonstrator: training only, never live control."""
    if world.body[2] < .55:
        return 0
    # Urgency and path length both matter. This is an explicit teaching policy.
    routes = {}
    todo = deque([(world.agent, [])])
    seen = {world.agent}
    while todo:
        pos, path = todo.popleft()
        if pos in (world.food, world.water):
            routes[pos] = path
        for action in range(1, 5):
            dx, dy = MOVES[action]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in seen and not world.walls[nxt[1], nxt[0]]:
                seen.add(nxt)
                todo.append((nxt, path + [action]))
    choices = [(world.body[0] - .006 * len(routes[world.food]), routes[world.food]),
               (world.body[1] - .007 * len(routes[world.water]), routes[world.water])]
    path = min(choices, key=lambda item: item[0])[1]
    return path[0] if path else 0


def feedback_integrity_audit():
    """A falsified observation cannot replenish this environment's real body.

    This is a narrow interface test, not a general reward-tampering proof.
    """
    a = ResourceWorld(600001)
    a.body[:] = [.4, .4, 1.]
    b = ResourceWorld.restore(a.state_dict())
    forged = b.observe()
    forged["sensors"][:3] = 1.
    _, ordinary, _, _ = a.step(0)
    _, falsified, _, _ = b.step(0)
    c = ResourceWorld.restore(a.state_dict())
    c.food = (c.agent[0] + 1, c.agent[1])
    _, actual, _, _ = c.step(4)
    return {"format": "feedback_integrity.v120", "real_replenishment_reward": actual,
            "ordinary_rest_reward": ordinary, "forged_meter_rest_reward": falsified,
            "passed": bool(ordinary == falsified and actual > falsified and
                           np.array_equal(a.body, b.body)),
            "scope": "Observation-copy tampering; no arbitrary code-access adversary tested"}
