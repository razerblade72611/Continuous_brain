"""Supervised developmental data for the vision organ: W1 frames paired with TEACHER labels.

Every frame is a genuine W1 observation (``obs["image"]``) and its labels come from W1's privileged ground truth
at the same instant (``labels_from_privileged(info["privileged"])``).  Behaviours that generate the frames:

* ``scripted`` -- W1's PRIVILEGED scripted baseline (``continuous_brain_world.baselines.ScriptedAgent``): it builds
  tables and furnaces, plants saplings, mines, fights zombies, shelters and sleeps at night.
* ``random``   -- W1's uniform random baseline (dies early; grassland around the spawn point, early night).
* ``explorer`` -- a teacher-assisted exploratory policy defined here (privileged navigation, data generation
  only).  At reset the teacher grants it pickaxes, a sword, wood, stone and saplings, and every step it tops up
  health, food, drink and energy, so it can roam far, tunnel into mountains, and stay out through the night.
  Each leg it picks a target kind (lava, diamond, iron, coal, water, sand, tree, tunnels, skeletons, zombies,
  cows or a random tile), walks/tunnels there along a shortest path (never onto lava), lingers looking around,
  sometimes places a table, furnace, stone or sapling, and takes 10 % uniformly random (lava-safe) actions.
  The world it sees is still Crafter's: only the player's inventory is edited, and the inventory strip is not
  part of the organ's input.

Everything is seeded: the same (seed, behaviour, max_steps, stride) gives byte-identical arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from crafter import constants

from continuous_brain_world import W1Config, W1Env
from continuous_brain_world.baselines import RandomAgent, ScriptedAgent
from continuous_brain_world.baselines.scripted import H, INF, NEIGHBOURS, W, flood
from continuous_brain_world.perception import perceivable_grid

from .vision import GRID_COLS, GRID_ROWS, labels_from_privileged

BEHAVIOURS = ("scripted", "explorer", "random", "gardener")
ACTIONS = {name: i for i, name in enumerate(constants.actions)}
MOVES = {(-1, 0): "move_left", (1, 0): "move_right", (0, -1): "move_up", (0, 1): "move_down"}
DIRS = tuple(MOVES)


class ExplorerAgent:
    """Teacher-assisted exploratory data-generation policy (privileged; see the module docstring)."""

    name = "teacher_assisted_explorer"
    privileged = True
    TARGETS = ("lava", "diamond", "iron", "coal", "water", "sand", "tree", "tunnel", "skeleton", "zombie", "cow",
               "sapling", "random")
    WEIGHTS = (0.13, 0.11, 0.08, 0.05, 0.10, 0.05, 0.05, 0.09, 0.09, 0.05, 0.05, 0.08, 0.07)
    GRANT = {"wood_pickaxe": 1, "stone_pickaxe": 1, "iron_pickaxe": 1, "wood_sword": 1, "stone_sword": 1}
    STOCK = {"wood": 4, "stone": 6, "sapling": 2}
    VITALS = ("health", "food", "drink", "energy")
    MINEABLE_COST = {"stone": 2, "coal": 2, "iron": 2, "diamond": 2, "tree": 6}
    WALK = ("grass", "sand", "path")

    # ``gardener`` mode: stays within ``HOME_RADIUS`` of its spawn point and plants saplings, so that plants stay
    # within Crafter's object-update radius (2 * 9 tiles of the player) long enough to ripen (grown > 300).
    GARDEN_TARGETS = ("sapling", "random", "water", "tree", "cow")
    GARDEN_WEIGHTS = (0.45, 0.35, 0.08, 0.07, 0.05)
    HOME_RADIUS = 9

    def __init__(self, seed: int = 0, epsilon: float = 0.1, mode: str = "explorer"):
        if mode not in ("explorer", "gardener"):
            raise ValueError(mode)
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(epsilon)
        self.mode = mode
        self.plant_prob = 0.08 if mode == "explorer" else 0.3
        self.home = None
        self.path, self.k, self.target, self.linger, self.deadline = [], 0, None, 0, 0

    # the teacher's interventions on the player's inventory (never on the world)
    def assist(self, env: W1Env, first: bool = False):
        inv = env.bridge.player.inventory
        if first:
            for k, v in self.GRANT.items():
                inv[k] = max(inv[k], v)
        for k, v in self.STOCK.items():
            inv[k] = max(inv[k], v)
        for k in self.VITALS:
            inv[k] = constants.items[k]["max"]

    def reset(self, obs=None, info=None):
        self.home = None
        self.path, self.k, self.target, self.linger, self.deadline = [], 0, None, 0, 0

    # -- planning ------------------------------------------------------------------------------------------
    def _costs(self, priv):
        names = {i: n for n, i in priv["material_ids"].items()}
        mat = np.asarray(priv["material_map"])
        cost = np.zeros((W, H), np.int64)
        for name, c in [(n, 1) for n in self.WALK] + list(self.MINEABLE_COST.items()):
            cost[mat == priv["material_ids"][name]] = c
        for o in priv["objects"]:
            x, y = o["pos"]
            if o["kind"] == "sapling":     # plants never move and cannot be pushed through
                cost[x, y] = 0
            elif o["kind"] != "player" and cost[x, y] == 1:
                cost[x, y] = 3
        return cost, mat, names

    def _plan(self, priv, step):
        cost, mat, names = self._costs(priv)
        pos = tuple(priv["player_pos"])
        dist, par = flood(cost.ravel().tolist(), pos[0] * H + pos[1])
        dist = np.asarray(dist).reshape(W, H)
        if self.home is None:
            self.home = pos
        targets, weights = ((self.TARGETS, self.WEIGHTS) if self.mode == "explorer"
                            else (self.GARDEN_TARGETS, self.GARDEN_WEIGHTS))
        kind = targets[int(self.rng.choice(len(targets), p=np.asarray(weights) / sum(weights)))]
        cands = self._candidates(kind, priv, mat, pos)
        if self.mode == "gardener":
            hx, hy = self.home
            r = self.HOME_RADIUS
            cands = [t for t in cands if max(abs(t[0] - hx), abs(t[1] - hy)) <= r]
            far = np.ones((W, H), bool)
            far[max(0, hx - r):hx + r + 1, max(0, hy - r):hy + r + 1] = False
            dist = np.where(far, INF, dist)
        best = []
        for t in cands:
            g = self._approach(t, dist)
            if g is not None:
                best.append((dist[g], t, g))
        if not best:
            xs, ys = np.nonzero((dist < INF) & (dist >= 5) & (dist <= 30))
            if len(xs) == 0:
                self.path, self.target = [], None
                return
            i = int(self.rng.integers(len(xs)))
            best = [(dist[xs[i], ys[i]], (int(xs[i]), int(ys[i])), (int(xs[i]), int(ys[i])))]
        best.sort()
        _, t, g = best[int(self.rng.integers(min(8, len(best))))]
        path, u = [], g[0] * H + g[1]
        start = pos[0] * H + pos[1]
        while u != start and u != -1:
            path.append(divmod(u, H))
            u = par[u]
        self.path, self.k, self.target = path[::-1], 0, t
        self.deadline = step + 2 * len(path) + 25

    def _candidates(self, kind, priv, mat, pos):
        ids = priv["material_ids"]
        if kind in ("skeleton", "zombie", "cow", "sapling"):  # sapling: any plant, ripe or not
            return [tuple(o["pos"]) for o in priv["objects"] if o["kind"] == kind]
        if kind == "random":
            return []
        name = "path" if kind == "tunnel" else kind
        xs, ys = np.nonzero(mat == ids[name])
        if kind == "tunnel":  # path tiles far from the spawn meadow: caves and tunnels
            keep = np.abs(xs - pos[0]) + np.abs(ys - pos[1]) >= 4
            xs, ys = xs[keep], ys[keep]
        if len(xs) > 60:
            sel = self.rng.choice(len(xs), 60, replace=False)
            xs, ys = xs[sel], ys[sel]
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    @staticmethod
    def _approach(t, dist):
        best, g = INF, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                x, y = t[0] + dx, t[1] + dy
                if 0 <= x < W and 0 <= y < H and dist[x, y] < best:
                    best, g = dist[x, y], (x, y)
        return g

    # -- acting --------------------------------------------------------------------------------------------
    def _mat(self, priv, p):
        x, y = p
        if not (0 <= x < W and 0 <= y < H):
            return None
        return self._names[int(priv["material_map"][x, y])]

    def _safe(self, priv, name):
        if name.startswith("move_"):
            u = {a: d for d, a in MOVES.items()}[name]
            p = priv["player_pos"]
            if self._mat(priv, (p[0] + u[0], p[1] + u[1])) == "lava":
                return "noop"
        return name

    def act(self, obs, info):
        priv = info["privileged"]
        self._names = {i: n for n, i in priv["material_ids"].items()}
        step = int(obs["step"])
        name = self._decide(priv, step)
        if self.rng.random() < self.epsilon:
            name = constants.actions[int(self.rng.integers(len(constants.actions)))]
        return ACTIONS[self._safe(priv, name)], None

    def _decide(self, priv, step):
        pos = tuple(priv["player_pos"])
        facing = tuple(priv["facing"])
        occupied = {tuple(o["pos"]): o["kind"] for o in priv["objects"] if o["kind"] != "player"}
        front = (pos[0] + facing[0], pos[1] + facing[1])
        if occupied.get(front) in ("zombie", "skeleton") and self.rng.random() < 0.7:
            return "do"
        if self.linger > 0:
            self.linger -= 1
            r = self.rng.random()
            if r < 0.04:
                return str(self.rng.choice(["place_table", "place_furnace", "place_stone"]))
            if r < 0.04 + self.plant_prob:
                return "place_plant"
            if r < 0.45 + self.plant_prob:
                return MOVES[DIRS[int(self.rng.integers(4))]]
            if r < 0.55 + self.plant_prob:
                return "do"
            return "noop"
        if self.target is None or self.k >= len(self.path) or step > self.deadline:
            if self.target is not None and self.k >= len(self.path):
                self.linger = int(self.rng.integers(3, 14))
                self.target = None
                return "noop"
            self._plan(priv, step)
            if not self.path:
                return MOVES[DIRS[int(self.rng.integers(4))]]
        while self.k < len(self.path) and tuple(self.path[self.k]) == pos:
            self.k += 1
        if self.k >= len(self.path):
            return "noop"
        n = tuple(self.path[self.k])
        u = (n[0] - pos[0], n[1] - pos[1])
        if u not in MOVES:  # pushed off the path (e.g. by a random action): replan next step
            self.target = None
            self.path = []
            return "noop"
        m = self._mat(priv, n)
        if n in occupied:
            return "do" if facing == u else MOVES[u]
        if m in self.MINEABLE_COST:
            return "do" if facing == u else MOVES[u]
        return MOVES[u]


def make_agent(behaviour: str, seed: int):
    if behaviour == "scripted":
        return ScriptedAgent(seed=seed)
    if behaviour == "random":
        return RandomAgent(seed=seed)
    if behaviour == "explorer":
        return ExplorerAgent(seed=seed)
    if behaviour == "gardener":
        return ExplorerAgent(seed=seed, mode="gardener")
    raise ValueError(f"unknown behaviour {behaviour!r}")


_ENVS: dict = {}


def _env(max_steps: int) -> W1Env:
    # One env per episode length, reused: ``reset(seed)`` rebuilds the world, the layer and the RNGs from the
    # seed alone (W1's determinism contract); only Crafter's texture cache is shared.
    if max_steps not in _ENVS:
        _ENVS[max_steps] = W1Env(W1Config(episode_length=max_steps, privileged_semantic=False))
    return _ENVS[max_steps]


FIELDS = {"frames": np.uint8, "material": np.int8, "object": np.int8, "perceivable": bool, "seed": np.int32,
          "step": np.int32, "behaviour": np.int8, "daylight": np.float32, "sleeping": bool,
          "player_pos": np.int16}


def generate_episode(seed: int, behaviour: str, max_steps: int, stride: int, agent_seed: int | None = None) -> dict:
    """Run one W1 episode and record every ``stride``-th frame (from step 0) with its teacher labels."""
    env = _env(int(max_steps))
    obs = env.reset(int(seed))
    agent = make_agent(behaviour, int(seed if agent_seed is None else agent_seed))
    agent.reset(obs)
    if isinstance(agent, ExplorerAgent):
        agent.assist(env, first=True)
    info = env.privileged_now()
    rec = {k: [] for k in FIELDS}
    done = False
    while True:
        s = int(obs["step"])
        if s % stride == 0:
            priv = info["privileged"]
            mat, obj = labels_from_privileged(priv)
            _, vis = perceivable_grid(env.bridge)            # [x, y] -> [row, col]
            rec["frames"].append(np.array(obs["image"]))          # the full 64 x 64 x 3 observation
            rec["material"].append(mat)
            rec["object"].append(obj)
            rec["perceivable"].append(vis.T.copy())
            rec["seed"].append(seed)
            rec["step"].append(s)
            rec["behaviour"].append(BEHAVIOURS.index(behaviour))
            rec["daylight"].append(priv["daylight"])
            rec["sleeping"].append(priv["sleeping"])
            rec["player_pos"].append(priv["player_pos"])
        if done:
            break
        action, utter = agent.act(obs, info)
        obs, _, done, info = env.step(action, utter)
        if isinstance(agent, ExplorerAgent) and not done:
            agent.assist(env)  # inventory only, after the frame was rendered (info's inventory is stale; unused)
    out = {k: np.asarray(v, dtype=FIELDS[k]) for k, v in rec.items()}
    assert out["material"].shape[1:] == (GRID_ROWS, GRID_COLS)
    return out


def concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def digest(data: dict) -> str:
    h = hashlib.sha256()
    for k in FIELDS:
        a = np.ascontiguousarray(data[k])
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_split(data: dict, directory, meta: dict | None = None) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for k in FIELDS:
        np.save(d / f"{k}.npy", data[k])
    manifest = {"n": int(len(data["frames"])), "sha256": digest(data), "meta": meta or {}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_split(directory, mmap: bool = False) -> dict:
    d = Path(directory)
    return {k: np.load(d / f"{k}.npy", mmap_mode="r" if mmap else None) for k in FIELDS}
