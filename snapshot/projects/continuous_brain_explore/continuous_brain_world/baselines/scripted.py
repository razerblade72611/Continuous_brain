"""Baseline (b): the PRIVILEGED SCRIPTED agent -- an upper-reference heuristic, not a learner.

It reads ``info["privileged"]`` (the full material map, every object, its own inventory, the structured
obligation specs, the visibility log, the request history and the per-day achievement snapshots) and never looks
at pixels or parses the caretaker's English.  It shows what the world permits, so that learners can be placed
between the random agent and it.  It is hand-tuned on engineering seeds 1-12 (see ``DEFAULTS``), never on the
baseline seeds.

Behaviour, in priority order every step:

1. answer every open report question at once from the privileged visibility log / request history, and
   acknowledge every new request (``ack <id>``);
2. fight a zombie that is next to it;
3. shelter (a sealed two-tile pocket, below) when there is a reason: to sleep, for the night (from evening dusk
   until dawn, longer while zombies wait outside), for an "avoid damage" request when a threat is near or it is
   dark, or to heal; drink/eat first when the shelter stay would otherwise starve it; leave when food or drink
   run out (at night only if dawn is far or health is low);
4. needs already being served (keep drinking/eating to full) and emergencies (drink or food <= 2);
5. the most urgent active task obligation (highest value / time-to-deadline first), each family with its own
   routine (collect / craft / place / visit; maintain raises need thresholds; avoid triggers sheltering);
6. stone for tonight's shelter in the afternoon; ordinary needs (thresholds raised in the evening and by any
   active "maintain" request);
7. the Crafter tech tree (wood, table, wood tools, stone, stone tools, sapling, furnace, coal, iron, iron
   tools, diamond, ripe plants, cows, stocks for future requests).

Movement is a shortest-path flood fill (Dial's algorithm) over the true map, where trees, stone and ores the
agent can mine cost 2 (one ``do`` plus one move) and tiles near zombies and skeletons cost extra.  Facing a
*free* tile (to place something or collect a sapling) is planned by arriving at it from the opposite side,
because in Crafter you can only turn without moving toward a blocked tile.

The shelter is a straight pocket ``q, s1, s2`` (entrance ``q``) whose five ring tiles (both sides of ``s1`` and
``s2`` and the far end) must be zombie- and arrow-proof: natural walls (stone, tree, ores; water and lava too
when ``water_walls``) or stone the agent places first, from outside.  The agent then mines/enters ``s1`` and
``s2``, walks back to ``s1`` (now facing ``q``) and places stone on ``q``.  Inside it sleeps (zombies hit a
sleeper for 7), drinks from an adjacent water tile if there is one, and leaves by mining ``q`` -- not while
zombies wait at the door unless food or drink have run out.  It never steps on lava (a final safety check).
"""
from __future__ import annotations

import numpy as np
from crafter import constants
from scipy.ndimage import binary_dilation

from .. import daylight as dl
from ..utterance import format_offset

ACTIONS = {name: i for i, name in enumerate(constants.actions)}
MOVE = {(-1, 0): "move_left", (1, 0): "move_right", (0, -1): "move_up", (0, 1): "move_down"}
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
W = H = 64
N = W * H
INF = 1 << 30
WALKABLE = ("grass", "sand", "path")
# Tiles a zombie cannot enter AND an arrow cannot cross (arrows fly over water and lava and break tables).
WALLS = ("stone", "tree", "coal", "iron", "diamond")
MATERIAL_OF = {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond",
               "drink": "water"}
TOOL_FOR = {"stone": "wood_pickaxe", "coal": "wood_pickaxe", "iron": "stone_pickaxe", "diamond": "iron_pickaxe"}


def _neighbours():
    out = []
    for x in range(W):
        for y in range(H):
            nb = []
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    nb.append(nx * H + ny)
            out.append(tuple(nb))
    return tuple(out)


NEIGHBOURS = _neighbours()


def shift(a: np.ndarray, dx: int, dy: int, fill=False) -> np.ndarray:
    """B[x, y] = a[x + dx, y + dy] (``fill`` outside)."""
    b = np.full_like(a, fill)
    xs = slice(max(0, -dx), W - max(0, dx))
    ys = slice(max(0, -dy), H - max(0, dy))
    xd = slice(max(0, dx), W - max(0, -dx))
    yd = slice(max(0, dy), H - max(0, -dy))
    b[xs, ys] = a[xd, yd]
    return b


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))


def flood(cost_flat: list, start: int):
    """Dial's algorithm: dist/parent over the grid with small positive integer costs (0 = impassable)."""
    dist = [INF] * N
    par = [-1] * N
    dist[start] = 0
    buckets = [[start]]
    d = 0
    while d < len(buckets):
        for u in buckets[d]:
            if dist[u] != d:
                continue
            for v in NEIGHBOURS[u]:
                c = cost_flat[v]
                if c:
                    nd = d + c
                    if nd < dist[v]:
                        dist[v] = nd
                        par[v] = u
                        while len(buckets) <= nd:
                            buckets.append([])
                        buckets[nd].append(v)
        d += 1
    return dist, par


class View:
    """One step's privileged ground truth, as numpy masks."""

    def __init__(self, obs, priv):
        self.step = int(obs["step"])
        self.mat = np.asarray(priv["material_map"])
        self.ids = priv["material_ids"]
        self.pos = tuple(priv["player_pos"])
        self.facing = tuple(priv["facing"])
        self.inv = priv["inventory"]
        self.ach = priv["achievements"]
        self.sleeping = priv["sleeping"]
        self.priv = priv
        self.objkind = {}
        self.occupied = np.zeros((W, H), bool)
        self.zombies, self.skeletons, self.cows, self.ripe, self.saplings = [], [], [], [], []
        for o in priv["objects"]:
            p = tuple(o["pos"])
            if o["kind"] == "player":
                continue
            self.objkind[p] = o["kind"]
            self.occupied[p] = True
            if o["kind"] == "zombie":
                self.zombies.append(p)
            elif o["kind"] == "skeleton":
                self.skeletons.append(p)
            elif o["kind"] == "cow":
                self.cows.append(p)
            elif o["kind"] == "sapling":
                self.saplings.append(p)
                if o["ripe"]:
                    self.ripe.append(p)
        self._m = {}

    def m(self, name: str) -> np.ndarray:
        if name not in self._m:
            mid = self.ids.get(name)
            self._m[name] = (self.mat == mid) if mid is not None else np.zeros((W, H), bool)
        return self._m[name]

    def any_of(self, names) -> np.ndarray:
        out = np.zeros((W, H), bool)
        for n in names:
            out |= self.m(n)
        return out

    def mat_name(self, p) -> str | None:
        x, y = p
        if not (0 <= x < W and 0 <= y < H):
            return None
        for k, v in self.ids.items():
            if v == self.mat[x, y]:
                return k
        return None

    def objmask(self, cells) -> np.ndarray:
        m = np.zeros((W, H), bool)
        for p in cells:
            m[p] = True
        return m


def _add(p, u, k=1):
    return (p[0] + k * u[0], p[1] + k * u[1])


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class ScriptedAgent:
    name = "privileged_scripted"
    privileged = True

    DEFAULTS = {"evening_light": 0.75, "morning_light": 0.85, "drink_evening": 8, "food_evening": 7,
                "open_ground": True, "water_walls": True, "flee": False, "crowd_wait": True, "wet_bonus": 12,
                "build_cost": 10, "urgent_value": 3}

    def __init__(self, seed: int = 0, **params):
        self.rng = np.random.default_rng(seed)
        self.p = dict(self.DEFAULTS)
        unknown = set(params) - set(self.p)
        if unknown:
            raise TypeError(f"unknown parameters {unknown}")
        self.p.update(params)
        self.reset(None)

    def reset(self, obs=None, info=None):
        self.answered: set = set()
        self.acked: set = set()
        self.drinking = self.eating = False
        self.shelter = None          # dict(q, d, reason)
        self.place_plan = None
        self.planted: list = []
        self.last_task = None

    # =====================================================================================================
    def act(self, obs, info):
        priv = info["privileged"]
        v = View(obs, priv)
        utter = self._speak(v, priv)
        if v.sleeping:
            return ACTIONS["noop"], utter
        self._prepare(v)
        name = self._decide(v) or "noop"
        name = self._safety(v, name)
        return ACTIONS[name], utter

    # -- speech ------------------------------------------------------------------------------------------
    def _speak(self, v, priv):
        clauses = []
        for ob in priv["obligations"]:
            if ob["family"] == "report" and ob["id"] not in self.answered:
                ans = self._answer(ob, v, priv)
                if ans is not None:
                    clauses.append(f"answer {ob['id']}: {ans}")
                    self.answered.add(ob["id"])
            elif ob["family"] != "report" and ob["id"] not in self.acked:
                clauses.append(f"ack {ob['id']}")
                self.acked.add(ob["id"])
        return "; ".join(clauses) or None

    def _answer(self, ob, v, priv):
        spec = ob["spec"]
        q = spec["question"]
        if q == "last_seen":
            entry = priv["visibility"].get(spec["kind"])
            if entry is None:
                return "never"
            best = min(entry["positions"], key=lambda p: (_cheb(p, v.pos), list(p)))
            return format_offset(best[0] - v.pos[0], best[1] - v.pos[1])
        if q == "count_today":
            snaps = priv["day_achievements"]
            start = snaps[str(spec["day"])][spec["event"]]
            if dl.day_index(v.step) > spec["day"] and str(spec["day"] + 1) in snaps:
                end = snaps[str(spec["day"] + 1)][spec["event"]]
            else:
                end = v.ach[spec["event"]]
            return str(end - start)
        hist = priv["history"]
        if q in ("first_request", "count_requests"):
            lo, hi = spec["window"]
            c = sorted((h for h in hist if h["family"] != "report" and lo <= h["issue_step"] < hi),
                       key=lambda h: (h["issue_step"], int(h["id"][1:])))
            if q == "count_requests":
                return str(len(c))
            return c[0]["id"] if c else "nothing"
        if q == "last_cancelled":
            c = [h for h in hist if h["status"] == "cancelled"]
            if not c:
                return "nothing"
            return max(c, key=lambda h: (h["resolved_step"], int(h["id"][1:])))["id"]
        return None

    # -- per-step preparation ---------------------------------------------------------------------------
    def _prepare(self, v):
        inv = v.inv
        self.mineable = v.m("tree").copy()
        if inv["wood_pickaxe"]:
            self.mineable |= v.m("stone") | v.m("coal")
        if inv["stone_pickaxe"]:
            self.mineable |= v.m("iron")
        if inv["iron_pickaxe"]:
            self.mineable |= v.m("diamond")
        self.mineable &= ~v.occupied
        if self.shelter is not None:  # never dig through the walls of our own shelter
            q = self.shelter["q"]
            for off in self._ring_offsets(self.shelter["d"]):
                c = _add(q, off)
                if 0 <= c[0] < W and 0 <= c[1] < H:
                    self.mineable[c] = False
        self.walkfree = v.any_of(WALKABLE) & ~v.occupied
        cost = np.zeros((W, H), np.int64)
        cost[self.walkfree] = 1
        cost[self.mineable] = 2
        danger = np.zeros((W, H), np.int64)
        zpen = 8 if (self._avoid_active(v) or inv["health"] < 7) else 3
        for z in v.zombies:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = z[0] + dx, z[1] + dy
                    if abs(dx) + abs(dy) <= 2 and 0 <= x < W and 0 <= y < H:
                        danger[x, y] = max(danger[x, y], zpen)
        for sk in v.skeletons:
            sl = (slice(max(0, sk[0] - 4), sk[0] + 5), slice(max(0, sk[1] - 4), sk[1] + 5))
            danger[sl] = np.maximum(danger[sl], 6)
        cost = np.where(cost > 0, cost + danger, 0)
        cost[v.pos] = 1
        self.cost = cost
        dist, self.par = flood(cost.ravel().tolist(), v.pos[0] * H + v.pos[1])
        self.dist = np.array(dist, np.int64).reshape(W, H)
        self.dist[v.pos] = 0

    def _safety(self, v, name):
        if name.startswith("move_"):
            u = {a: d for d, a in MOVE.items()}[name]
            if v.mat_name(_add(v.pos, u)) == "lava":
                return "noop"
        return name

    # -- movement primitives -----------------------------------------------------------------------------
    def _first_step(self, goal):
        g = goal[0] * H + goal[1]
        start = self.view.pos[0] * H + self.view.pos[1]
        if self.dist[goal] >= INF:
            return None
        while self.par[g] != start and self.par[g] != -1:
            g = self.par[g]
        return divmod(g, H)

    def _step_toward(self, v, goal):
        if goal == v.pos:
            return "noop"
        n = self._first_step(goal)
        if n is None:
            return None
        u = (n[0] - v.pos[0], n[1] - v.pos[1])
        if self.mineable[n]:
            return "do" if v.facing == u else MOVE[u]
        return MOVE[u]

    def _goto(self, v, mask):
        """Walk to the nearest cell of ``mask``; returns 'READY' when standing in it."""
        if mask[v.pos]:
            return "READY"
        d = np.where(mask, self.dist, INF)
        i = int(np.argmin(d))
        if d.flat[i] >= INF:
            return None
        return self._step_toward(v, divmod(i, H))

    def _face(self, v, target_mask, stand_mask=None, max_cost=INF):
        """Get to a state where the faced tile is in ``target_mask`` (standing in ``stand_mask``).
        Returns 'READY', a move action, or None if impossible."""
        front = _add(v.pos, v.facing)
        if 0 <= front[0] < W and 0 <= front[1] < H and target_mask[front] and (stand_mask is None or stand_mask[v.pos]):
            return "READY"
        best, plan = INF, None
        standable = self.walkfree.copy()
        standable[v.pos] = True
        if stand_mask is not None:
            standable &= stand_mask
        blocked_target = target_mask & ~self.walkfree
        free_target = target_mask & self.walkfree
        for u in DIRS:
            s_has_blocked = shift(blocked_target, u[0], u[1]) & standable
            if s_has_blocked.any():
                d = np.where(s_has_blocked, self.dist, INF)
                i = int(np.argmin(d))
                if d.flat[i] + 1 < best:
                    best, plan = d.flat[i] + 1, ("turn", divmod(i, H), u)
            s_has_free = shift(free_target, u[0], u[1]) & standable
            if s_has_free.any():
                # arrive at s from a = s - u, so we face s + u
                a_ok = shift(self.walkfree | self._posmask(v), -u[0], -u[1])
                da = shift(self.dist, -u[0], -u[1], INF)
                m = s_has_free & a_ok
                if m.any():
                    d = np.where(m, da, INF)
                    i = int(np.argmin(d))
                    if d.flat[i] + 1 < best:
                        best, plan = d.flat[i] + 1, ("arrive", divmod(i, H), u)
        if plan is None or best > max_cost:
            return None
        kind, s, u = plan
        if kind == "turn":
            if v.pos == s:
                return MOVE[u]
            return self._step_toward(v, s)
        a = (s[0] - u[0], s[1] - u[1])
        if v.pos == a:
            return MOVE[u]
        return self._step_toward(v, a)

    def _posmask(self, v):
        m = np.zeros((W, H), bool)
        m[v.pos] = True
        return m

    # =====================================================================================================
    def _decide(self, v):
        self.view = v
        inv = v.inv
        # 2. adjacent zombie: fight
        for z in v.zombies:
            if abs(z[0] - v.pos[0]) + abs(z[1] - v.pos[1]) == 1:
                u = (z[0] - v.pos[0], z[1] - v.pos[1])
                return "do" if v.facing == u else MOVE[u]
        if self.p["flee"] and inv["health"] <= 4 and self.shelter is None:
            act = self._flee(v)
            if act:
                return act
        thr = self._thresholds(v)
        reasons = self._shelter_reasons(v, thr)
        inside = self._inside(v)
        critical = inv["drink"] == 0 or inv["food"] == 0      # health cannot regenerate any more
        if dl.is_dark(v.step):
            # leaving at night means zombies; starving means a slow drain -- wait unless dawn is far
            must_eat = critical and (dl.next_dawn(v.step) - v.step > 60 or inv["health"] <= 4)
        elif inside:
            q = self.shelter["q"]
            crowded = any(abs(z[0] - q[0]) + abs(z[1] - q[1]) <= 3 for z in v.zombies)
            must_eat = critical or ((inv["drink"] <= 2 or inv["food"] <= 2)
                                    and not (crowded and self.p["crowd_wait"]))
        else:
            must_eat = inv["drink"] <= 2 or inv["food"] <= 2
        # 3. shelter: sleep, night, avoid-requests, healing
        if reasons and not must_eat:
            if not inside and self.shelter is None:
                # top up before hiding (in the dark only from water that is very close)
                horizon = self._shelter_horizon(v, reasons)
                dark = dl.is_dark(v.step)
                if inv["drink"] < min(9, horizon // 30 + 3):
                    act = self._drink(v, max_cost=10 if dark else 30)
                    if act:
                        return act
                if inv["food"] < min(8, horizon // 50 + 2) and not dark:
                    act = self._eat(v, max_cost=20)
                    if act:
                        return act
            act = self._enter_shelter(v, reasons)
            if act:
                return act
            if inv["energy"] <= 1:
                return "sleep"
        elif self.shelter is not None:
            act = self._shelter_step(v, needed=False)
            if act:
                return act
        # 4. emergencies and needs already being served
        if inv["drink"] <= 2 or (self.drinking and inv["drink"] < 9):
            act = self._drink(v)
            if act:
                return act
        if inv["food"] <= 2 or (self.eating and inv["food"] < 9):
            act = self._eat(v)
            if act:
                return act
        # 6. obligations
        act = self._obligations(v)
        if act:
            return act
        # stone for tonight's shelter
        if inv["wood_pickaxe"] and inv["stone"] < 6 and dl.phase(v.step) == "afternoon":
            act = self._acquire(v, "stone", 6)
            if act:
                return act
        # 7. ordinary needs
        if inv["drink"] <= thr["drink"]:
            act = self._drink(v)
            if act:
                return act
        if inv["food"] <= thr["food"]:
            act = self._eat(v)
            if act:
                return act
        # 8. tech tree
        act = self._tech_tree(v)
        if act:
            return act
        return self._wander(v)

    def _flee(self, v):
        close = [z for z in v.zombies if abs(z[0] - v.pos[0]) + abs(z[1] - v.pos[1]) <= 3]
        if not close:
            return None

        def score(p):
            return min(abs(z[0] - p[0]) + abs(z[1] - p[1]) for z in close)

        best, act = score(v.pos), None
        for u in DIRS:
            n = _add(v.pos, u)
            if 0 <= n[0] < W and 0 <= n[1] < H and self.walkfree[n] and score(n) > best:
                best, act = score(n), MOVE[u]
        return act

    def _inside(self, v):
        if self.shelter is None:
            return False
        q, s1, s2, end, d = self._shelter_cells()
        return v.pos in (s1, s2)

    def _shelter_horizon(self, v, reasons):
        h = 60
        if "night" in reasons or "sleep" in reasons:
            h = max(h, dl.next_dawn(v.step) - v.step if dl.is_dark(v.step) else 100)
        if "avoid" in reasons:
            h = max([h] + [o["deadline"] - v.step for o in v.priv["obligations"] if o["family"] == "avoid"])
        return h

    def _shelter_reasons(self, v, thr):
        inv, r = v.inv, set()
        dark = dl.is_dark(v.step)
        if inv["energy"] <= max(2, thr["energy"]):
            r.add("sleep")
        if self._inside(v) and inv["energy"] < 9 and (dark or "sleep" in self.shelter.get("reasons", ())):
            r.add("sleep")
        light = float(v.priv["daylight"])
        ph = dl.phase(v.step)
        if not self._urgent_tonight(v):
            if dark or (ph == "afternoon" and light < self.p["evening_light"]):
                r.add("night")
            elif ph == "morning" and light < self.p["morning_light"] and self._inside(v) and self._threat_near(v, 8):
                r.add("night")
        if self._avoid_active(v) and (dark or self._threat_near(v, 6)):
            r.add("avoid")
        heal_to = 7 if self._inside(v) else 4
        can_heal = inv["food"] > 0 and inv["drink"] > 0
        if can_heal and inv["health"] <= max(heal_to, thr["health"]) and (dark or self._threat_near(v, 10) or self._inside(v)):
            r.add("heal")
        return r

    # -- needs ---------------------------------------------------------------------------------------------
    def _thresholds(self, v):
        evening = dl.phase(v.step) == "afternoon"
        thr = {"drink": self.p["drink_evening"] if evening else 5, "food": self.p["food_evening"] if evening else 5,
               "energy": 2, "health": 0}
        for ob in v.priv["obligations"]:
            if ob["family"] == "maintain":
                st, t = ob["spec"]["stat"], ob["spec"]["threshold"]
                if st in thr:
                    thr[st] = max(thr[st], t + (1 if st == "energy" else 2))
        return thr

    def _drink(self, v, max_cost=INF):
        act = self._face(v, v.m("water"), max_cost=max_cost)
        if act == "READY":
            self.drinking = v.inv["drink"] < 8
            return "do"
        if act is None:
            self.drinking = False
        else:
            self.drinking = True
        return act

    def _eat(self, v, max_cost=INF):
        if v.ripe:
            act = self._face(v, v.objmask(v.ripe), max_cost=min(25, max_cost))
            if act == "READY":
                return "do"
            if act:
                self.eating = True
                return act
        if v.cows:
            act = self._face(v, v.objmask(v.cows), max_cost=max_cost)
            if act == "READY":
                self.eating = v.inv["food"] < 7
                return "do"
            if act:
                self.eating = True
                return act
        self.eating = False
        return None

    def _avoid_active(self, v):
        return any(o["family"] == "avoid" for o in v.priv["obligations"])

    def _threat_near(self, v, r):
        return any(_cheb(z, v.pos) <= r for z in v.zombies + v.skeletons)

    def _urgent_tonight(self, v):
        dawn = dl.next_dawn(v.step)
        for o in v.priv["obligations"]:
            if o["family"] in ("collect", "craft", "place", "visit") and o["deadline"] < dawn + 40 \
                    and o["value"] >= self.p["urgent_value"]:
                return True
        return False

    # -- shelter -------------------------------------------------------------------------------------------
    @staticmethod
    def _ring_offsets(d):
        """Offsets (from q) of the five wall tiles that must be blocked before entering: the two sides of s1,
        the two sides of s2, and the far end."""
        e = (d[1], d[0])
        return [(d[0] + e[0], d[1] + e[1]), (d[0] - e[0], d[1] - e[1]), (2 * d[0] + e[0], 2 * d[1] + e[1]),
                (2 * d[0] - e[0], 2 * d[1] - e[1]), (3 * d[0], 3 * d[1])]

    def _find_shelter(self, v):
        if not v.inv["wood_pickaxe"]:
            return None
        zone = np.zeros((W, H), bool)
        for sk in v.skeletons:
            zone[max(0, sk[0] - 8):sk[0] + 9, max(0, sk[1] - 8):sk[1] + 9] = True
        walls = v.any_of(WALLS + (("water", "lava") if self.p["water_walls"] else ())) & ~v.occupied
        placeable = v.any_of(WALKABLE) & ~v.occupied
        if not self.p["open_ground"]:
            placeable = np.zeros((W, H), bool)
        okcell = self.walkfree | self.mineable
        stone = v.m("stone")
        q_ok = (self.walkfree | self._posmask(v)) & (self.dist < 60)
        best, choice = INF, None
        for d in DIRS:
            ok = q_ok & shift(okcell, d[0], d[1]) & shift(okcell, 2 * d[0], 2 * d[1])
            need = np.zeros((W, H), np.int64)
            for off in self._ring_offsets(d):
                ok &= shift(walls | placeable, off[0], off[1])
                need += shift(~walls, off[0], off[1], True).astype(np.int64)
            if not ok.any():
                continue
            gain = shift(stone, d[0], d[1]).astype(np.int64) + shift(stone, 2 * d[0], 2 * d[1]).astype(np.int64)
            ok &= (v.inv["stone"] >= need) & (v.inv["stone"] + gain >= need + 1)
            mined = shift(~self.walkfree, d[0], d[1]).astype(np.int64) + shift(~self.walkfree, 2 * d[0], 2 * d[1]).astype(np.int64)
            wet = np.zeros((W, H), bool)
            if self.p["water_walls"]:
                for off in self._ring_offsets(d)[:4]:
                    wet |= shift(v.m("water"), off[0], off[1])
            score = np.where(ok, self.dist + self.p["build_cost"] * need + 2 * mined + 25 * zone
                             - self.p["wet_bonus"] * wet, INF)
            i = int(np.argmin(score))
            if score.flat[i] < best:
                best, choice = score.flat[i], (divmod(i, H), d)
        if choice is None or best > 120:
            return None
        return {"q": choice[0], "d": choice[1]}

    def _enter_shelter(self, v, reasons):
        if self.shelter is None:
            sh = self._find_shelter(v)
            if sh is None:
                return None
            self.shelter = sh
        self.shelter["reasons"] = sorted(reasons)
        return self._shelter_step(v, needed=True)

    def _shelter_cells(self):
        q, d = self.shelter["q"], self.shelter["d"]
        return q, _add(q, d), _add(q, d, 2), _add(q, d, 3), d

    def _is_wall(self, v, c):
        if not (0 <= c[0] < W and 0 <= c[1] < H):
            return True
        return v.mat_name(c) not in WALKABLE and not v.occupied[c]

    def _sealed(self, v):
        q = self.shelter["q"]
        cells = [_add(q, off) for off in self._ring_offsets(self.shelter["d"])] + [q]
        return all(self._is_wall(v, c) for c in cells)

    def _shelter_step(self, v, needed):
        q, s1, s2, end, d = self._shelter_cells()
        back = (-d[0], -d[1])
        inside = v.pos in (s1, s2)
        if not needed:
            if not inside:
                self.shelter = None
                return None
            return self._leave(v)
        ring = [_add(q, off) for off in self._ring_offsets(d)]
        pending = [c for c in ring if not self._is_wall(v, c)]
        if pending:
            if v.inv["stone"] < 1 or any(v.occupied[c] for c in pending if 0 <= c[0] < W and 0 <= c[1] < H):
                if inside:
                    return "noop" if not v.occupied[q] else ("do" if v.objkind.get(q) == "zombie" else "noop")
                self.shelter = None
                return None
            target = np.zeros((W, H), bool)
            for c in pending:
                target[c] = True
            act = self._face(v, target, stand_mask=~target)
            if act == "READY":
                return "place_stone"
            if act is None:
                self.shelter = None
            return act
        if not inside:
            if v.occupied[s1] or v.occupied[s2]:
                self.shelter = None
                return None
            if v.pos == q:
                if self.walkfree[s1]:
                    return MOVE[d]
                if self.mineable[s1]:
                    return "do" if v.facing == d else MOVE[d]
                self.shelter = None
                return None
            act = self._step_toward(v, q)
            if act is None:
                self.shelter = None
            return act
        if self._sealed(v):
            act = self._drink_inside(v)
            if act:
                return act
            return "sleep" if v.inv["energy"] < 9 else "noop"
        # ring complete, q open: get to s1 facing q and close it
        if v.pos == s2:
            return MOVE[back]
        if v.facing == d:
            if self.walkfree[s2]:
                return MOVE[d]
            if self.mineable[s2]:
                return "do"
            return MOVE[back]
        if v.facing != back:
            return MOVE[d] if self.walkfree[s2] else MOVE[back]
        if v.occupied[q]:
            return "do" if v.objkind.get(q) == "zombie" else "noop"
        if v.inv["stone"] > 0:
            return "place_stone"
        self.shelter = None
        return None

    def _drink_inside(self, v):
        """Drink from a water tile next to the pocket, if there is one (turning toward water never moves)."""
        if v.inv["drink"] >= 9 and not (v.inv["drink"] >= 8 and self.drinking):
            self.drinking = False
            return None
        for u in DIRS:
            c = _add(v.pos, u)
            if v.mat_name(c) == "water":
                self.drinking = v.inv["drink"] < 9
                return "do" if v.facing == u else MOVE[u]
        return None

    def _leave(self, v):
        q, s1, s2, end, d = self._shelter_cells()
        back = (-d[0], -d[1])
        if self._sealed(v) and any(abs(z[0] - q[0]) + abs(z[1] - q[1]) <= 3 for z in v.zombies) \
                and v.inv["drink"] > 0 and v.inv["food"] > 0:
            act = self._drink_inside(v)
            return act or ("sleep" if v.inv["energy"] < 9 else "noop")   # wait for the zombies to leave
        if v.pos == s2:
            return MOVE[back]
        if v.facing != back:
            return MOVE[back]  # turns toward q if q is stone, else walks out
        mat = v.mat_name(q)
        if mat in WALKABLE and not v.occupied[q]:
            self.shelter = None
            return MOVE[back]
        if v.occupied[q]:
            return "do" if v.objkind.get(q) == "zombie" else "noop"
        if mat in ("stone", "coal") and v.inv["wood_pickaxe"] or mat == "tree":
            return "do"
        self.shelter = None
        return None

    # -- obligations -----------------------------------------------------------------------------------------
    def _obligations(self, v):
        tasks = [o for o in v.priv["obligations"] if o["family"] in ("collect", "craft", "place", "visit")]
        if not tasks:
            return None

        def urgency(o):
            left = max(1, o["deadline"] - v.step)
            return (-o["value"] / left, o["deadline"], o["id"])

        for o in sorted(tasks, key=urgency):
            act = getattr(self, "_ob_" + o["family"])(v, o)
            if act:
                self.last_task = o["id"]
                return act
        return None

    def _ob_collect(self, v, o):
        return self._acquire(v, o["spec"]["item"], o["spec"]["target"])

    def _ob_craft(self, v, o):
        return self._make(v, o["spec"]["tool"], force=True)

    def _ob_place(self, v, o):
        spec = o["spec"]
        what = spec["what"]
        item = {"table": "wood", "plant": "sapling", "furnace": "stone", "stone": "stone"}[what]
        need = constants.place[what]["uses"][item]
        if v.inv[item] < need:
            return self._acquire(v, item, need)
        where = v.any_of(constants.place[what]["where"]) & ~v.occupied
        if "near" in spec:
            near = v.m(spec["near"])
            if not near.any():
                if spec["near"] == "table":
                    return self._place_table(v)
                return None
            region = dilate(near, spec["radius"]) & ~near
            target = where & region
        else:
            target = where & v.m(spec["onto"])
        act = self._face(v, target)
        if act == "READY":
            return "place_" + what
        return act

    def _ob_visit(self, v, o):
        spec = o["spec"]
        prog = o.get("progress") or {}
        start = tuple(spec["start"])
        if prog.get("reached_step") is None:
            feat = v.m(spec["feature"]).copy()
            xs, ys = np.nonzero(feat)
            ok = np.zeros((W, H), bool)
            from ..obligations import in_cone
            for x, y in zip(xs.tolist(), ys.tolist()):
                if _cheb((x, y), start) >= spec["min_dist"] and in_cone(start, (x, y), spec["direction"]):
                    ok[x, y] = True
            goal = dilate(ok, spec["reach_radius"]) & (self.walkfree | self._posmask(v))
            act = self._goto(v, goal)
            return "noop" if act == "READY" else act
        goal = np.zeros((W, H), bool)
        goal[max(0, start[0] - 3):start[0] + 4, max(0, start[1] - 3):start[1] + 4] = True
        goal &= self.walkfree | self._posmask(v)
        act = self._goto(v, goal)
        return "noop" if act == "READY" else act

    # -- resources and crafting --------------------------------------------------------------------------------
    def _acquire(self, v, item, count):
        """Act toward having ``inventory[item] >= count``."""
        if v.inv[item] >= count:
            return None
        if item == "sapling":
            act = self._face(v, v.m("grass") & ~v.occupied)
            return "do" if act == "READY" else act
        tool = TOOL_FOR.get(item)
        if tool and not v.inv[tool]:
            return self._make(v, tool)
        act = self._face(v, v.m(MATERIAL_OF[item]) & ~v.occupied)
        return "do" if act == "READY" else act

    def _station_mask(self, v, needs):
        m = np.ones((W, H), bool)
        for n in needs:
            m &= dilate(v.m(n), 1)
        return m

    def _make(self, v, tool, force=False):
        recipe = constants.make[tool]
        needs = recipe["nearby"]
        standable = self.walkfree | self._posmask(v)
        station = self._station_mask(v, needs) & standable
        station_close = bool((np.where(station, self.dist, INF) < 25).any())
        table_close = bool((dilate(v.m("table"), 1) & standable & (self.dist < 25)).any())
        extra = {"wood": 0 if table_close else constants.place["table"]["uses"]["wood"],
                 "stone": 0 if (station_close or "furnace" not in needs) else constants.place["furnace"]["uses"]["stone"]}
        want = dict(recipe["uses"])
        for item, n in extra.items():
            if n:
                want[item] = want.get(item, 0) + n
        for item, n in want.items():
            if v.inv[item] < n:
                return self._acquire(v, item, n)
        if station[v.pos]:
            return "make_" + tool
        if station_close:
            return self._goto(v, station)
        if not table_close:
            return self._place_table(v)
        return self._place_furnace(v)

    def _place_table(self, v):
        cost = constants.place["table"]["uses"]["wood"]
        if v.inv["wood"] < cost:
            return self._acquire(v, "wood", cost)
        where = v.any_of(constants.place["table"]["where"]) & ~v.occupied
        act = self._face(v, where)
        return "place_table" if act == "READY" else act

    def _place_furnace(self, v):
        cost = constants.place["furnace"]["uses"]["stone"]
        if v.inv["stone"] < cost:
            return self._acquire(v, "stone", cost)
        where = v.any_of(constants.place["furnace"]["where"]) & ~v.occupied
        stand = dilate(v.m("table"), 1)
        act = self._face(v, where & dilate(v.m("table"), 2), stand_mask=stand)
        return "place_furnace" if act == "READY" else act

    def _tech_tree(self, v):
        inv, ach = v.inv, v.ach
        if not inv["wood_pickaxe"]:
            return self._make(v, "wood_pickaxe")
        if not inv["wood_sword"]:
            return self._make(v, "wood_sword")
        if inv["stone"] < 3 and not inv["stone_pickaxe"]:
            return self._acquire(v, "stone", 3)
        if not inv["stone_pickaxe"]:
            return self._make(v, "stone_pickaxe")
        if not inv["stone_sword"]:
            return self._make(v, "stone_sword")
        if not ach["collect_sapling"] or (not ach["place_plant"] and inv["sapling"] == 0):
            act = self._acquire(v, "sapling", 1)
            if act:
                return act
        if not ach["place_plant"] and inv["sapling"]:
            act = self._face(v, v.m("grass") & ~v.occupied)
            if act == "READY":
                return "place_plant"
            if act:
                return act
        if not ach["place_furnace"]:
            fc = constants.place["furnace"]["uses"]["stone"]
            if inv["stone"] < fc + 1:
                return self._acquire(v, "stone", fc + 1)
            return self._place_furnace(v)
        if not inv["iron_pickaxe"] or not inv["iron_sword"]:
            for item, n in (("coal", 2), ("iron", 2), ("wood", 2)):
                have_needed = n - (1 if inv["iron_pickaxe"] else 0)
                if inv[item] < have_needed:
                    act = self._acquire(v, item, have_needed)
                    if act:
                        return act
            if not inv["iron_pickaxe"]:
                act = self._make(v, "iron_pickaxe")
                if act:
                    return act
            if not inv["iron_sword"]:
                act = self._make(v, "iron_sword")
                if act:
                    return act
        if not ach["collect_diamond"] and inv["iron_pickaxe"]:
            act = self._acquire(v, "diamond", 1)
            if act:
                return act
        if not ach["eat_plant"] and v.ripe:
            act = self._face(v, v.objmask(v.ripe))
            if act == "READY":
                return "do"
            if act:
                return act
        if not ach["eat_cow"] and v.cows:
            act = self._eat(v)
            if act:
                return act
        if not ach["collect_drink"]:
            act = self._drink(v)
            if act:
                return act
        # keep stocks for future requests
        for item, n in (("wood", 5), ("stone", 5), ("sapling", 2), ("coal", 2)):
            if inv[item] < n:
                act = self._acquire(v, item, n)
                if act:
                    return act
        if len(v.saplings) < 3 and inv["sapling"]:
            act = self._face(v, v.m("grass") & ~v.occupied, max_cost=10)
            if act == "READY":
                return "place_plant"
        return None

    def _wander(self, v):
        if v.inv["drink"] < 9:
            act = self._drink(v)
            if act:
                return act
        if v.inv["food"] < 8 and v.cows:
            act = self._eat(v)
            if act:
                return act
        return "noop"
