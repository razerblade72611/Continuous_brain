"""The skill library: engineered routines driven by the individual's own beliefs (exploration lane, v1).

This is the honest status of the individual's *skills*: they are the routine library of W1's privileged scripted
reference agent (``continuous_brain_world.baselines.scripted``: survival, sheltering, gathering, the tech tree,
pursuing commitments), but run over what the individual itself believes -- its remembered map (``memory``), the
objects it perceives now, its body sense, and the commitments it understood (``ledger``) -- instead of the true
world.  Learning these skills (imitation, then reinforcement) is the next step; see docs/explore/INDIVIDUAL_W1.md.

Two things are the individual's own here:

* **Belief view.**  Tiles never seen are treated as open grass for path planning (optimism under uncertainty)
  and are the targets of exploration.  Objects are those perceived at this step.
* **Exploration.**  When the routines have nothing to do (no known target for the current need or task), or
  when it is hungry or thirsty and knows no source of food or water, the individual walks to the nearest
  frontier: a never-seen tile reachable over known ground.
* **Caution learned from pain.**  If it is hurt repeatedly inside its shelter for no reason it can feel (not
  hunger, thirst or fatigue), it leaves, remembers the spot as dangerous (treated like a skeleton's surroundings
  when planning paths and shelters) and, for a while, stops trusting water as a wall.  Water is also not
  trusted as a wall while it believes tunnels (skeleton ground) are near, because arrows fly over water.
* **Object memory.**  Cows it saw recently and plants it saw are part of its belief view where they were last
  seen (``memory.believed_objects``); arriving and not finding them corrects the belief.

The routines' own memory between steps (whether it is drinking or eating, its shelter plan, its last task, its
random generator) is part of the individual's store-held state (``to_state``/``from_state``), so nothing is kept
in Python between ticks.
"""
from __future__ import annotations

import copy

import numpy as np

from continuous_brain_organs.vision import MATERIALS
from continuous_brain_world import daylight as dl
from continuous_brain_world.baselines.scripted import ACTIONS, INF, MOVE, ScriptedAgent, View

from . import memory as mem

MATERIAL_IDS = {name: i for i, name in enumerate(MATERIALS)}
UNKNOWN_AS = MATERIAL_IDS["grass"]
WALKABLE_IDS = {MATERIAL_IDS[n] for n in ("grass", "sand", "path")}
USE_BUMPS = False          # bump memory is recorded but not acted on: acting on it made lives shorter (see docs)
ACTION_NAMES = {i: n for n, i in ACTIONS.items()}
ROUTINE_STATE = ("drinking", "eating", "shelter", "place_plan", "planted", "last_task")
NEED_EXPLORE = {"day": 5, "afternoon": 7}   # food/drink at or below this and no known source: go looking (earlier
                                             # in the afternoon: the night will be spent without food or water)
TUNNEL_FRONTIER_PENALTY = 40                 # frontiers next to tunnel ground (skeletons) are explored last
TUNNEL_RADIUS = 12        # path tiles (tunnels, where skeletons live) this close: water is no wall (arrows cross it)
PAIN_WINDOW, PAIN_HITS = 20, 2   # hurt this often inside a shelter, with no hunger/thirst/fatigue: leave it
BAD_SPOT_MEMORY, CAUTION = 1500, 600
OPPORTUNISTIC_FOOD = 7    # eats a cow in sight below this (it cannot see food beyond its view)
PARAMS = {"food_evening": 8}   # evening food target one above the reference's: it cannot rely on knowing every cow
OPTIONS = ("routine", "drink", "eat", "rest", "retreat", "explore", "requests", "build")
OPTION_MAX = {"routine": 25, "drink": 80, "eat": 80, "rest": 300, "retreat": 15, "explore": 30, "requests": 40,
              "build": 40}


def belief_priv(memory: dict, body: dict, step: int, commitments: list[dict]) -> dict:
    """The routines' view of the world, built only from the individual's own beliefs and body sense."""
    belief = memory["map"].astype(np.int64)
    known = belief >= 0
    mat = np.where(known, belief, UNKNOWN_AS)
    objects = [{"kind": o["kind"], "pos": list(o["pos"]), "ripe": bool(o["ripe"])}
               for o in mem.believed_objects(memory)]
    taken = {(o["pos"][0], o["pos"][1]) for o in objects}
    for key in (memory.get("blocked", {}) if USE_BUMPS else {}):   # bumped into: something is there for now
        x, y = (int(v) for v in key.split(","))
        if (x, y) not in taken:
            objects.append({"kind": "obstacle", "pos": [x, y], "ripe": False})
    return {"material_map": mat, "material_ids": MATERIAL_IDS, "player_pos": list(body["player_pos"]),
            "facing": list(body["facing"]), "inventory": dict(body["inventory"]),
            "achievements": dict(body["achievements"]), "sleeping": bool(body["sleeping"]), "objects": objects,
            "obligations": commitments, "daylight": dl.daylight(step)}


class SkillLibrary(ScriptedAgent):
    name = "individual_skills"
    privileged = False

    def __init__(self, seed: int = 0, **params):
        super().__init__(seed=seed, **{**PARAMS, **params})

    def to_state(self) -> dict:
        st = {k: copy.deepcopy(getattr(self, k)) for k in ROUTINE_STATE}
        st["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        st["explore_target"] = getattr(self, "explore_target", None)
        st["visit_reached"] = dict(getattr(self, "visit_reached", None) or {})
        for k, d in (("last_health", None), ("pain_steps", []), ("bad_spots", []), ("caution_until", -1)):
            st[k] = copy.deepcopy(getattr(self, k, d))
        return st

    @classmethod
    def from_state(cls, state: dict | None, seed: int = 0) -> "SkillLibrary":
        lib = cls(seed=seed)
        if state:
            for k in ROUTINE_STATE:
                setattr(lib, k, copy.deepcopy(state[k]))
            lib.rng.bit_generator.state = copy.deepcopy(state["rng"])
            lib.explore_target = state.get("explore_target")
            lib.visit_reached = dict(state.get("visit_reached") or {})
            for k, d in (("last_health", None), ("pain_steps", []), ("bad_spots", []), ("caution_until", -1)):
                setattr(lib, k, copy.deepcopy(state.get(k, d)))
        else:
            lib.explore_target = None
            lib.visit_reached = {}
        return lib

    def belief_view(self, memory: dict, body: dict, step: int, commitments: list[dict]) -> View:
        """The routines' belief view for this step (records visit progress and pain on the way)."""
        commitments = self._visit_progress(memory, body, commitments)
        priv = belief_priv(memory, body, step, commitments)
        self._pain(body, step)
        for x, y, s in getattr(self, "bad_spots", []):          # remembered danger: planned around like a skeleton
            if step - s <= BAD_SPOT_MEMORY:
                priv["objects"].append({"kind": "skeleton", "pos": [x, y], "ripe": False})
        px, py = body["player_pos"]
        path = memory["map"] == MATERIAL_IDS["path"]
        near = path[max(0, px - TUNNEL_RADIUS):px + TUNNEL_RADIUS + 1, max(0, py - TUNNEL_RADIUS):py + TUNNEL_RADIUS + 1]
        self.p["water_walls"] = not (near.any() or step < getattr(self, "caution_until", -1))
        return View({"step": step}, priv)

    def decide(self, memory: dict, body: dict, step: int, commitments: list[dict]) -> tuple[str, str]:
        """(action name, option) for this step.  option names what the step serves (for the store)."""
        v = self.belief_view(memory, body, step, commitments)
        if v.sleeping:
            return "noop", "sleeping"
        self._prepare(v)
        return self.routine(v, memory, step)

    def routine(self, v: View, memory: dict, step: int) -> tuple[str, str]:
        """The whole routine library's choice on a prepared view (v1 behaviour)."""
        name = self._decide(v) or "noop"
        option = "routine"
        visible_cow = any(o["kind"] == "cow" for o in memory["objects"])
        if visible_cow and v.inv["food"] <= OPPORTUNISTIC_FOOD and self.shelter is None and not dl.is_dark(step) \
                and name not in ("do", "sleep") and not any(abs(z[0] - v.pos[0]) + abs(z[1] - v.pos[1]) <= 2
                                                            for z in v.zombies):
            act = self._eat(v, max_cost=12)            # food in sight, and it cannot count on finding more later
            if act:
                name, option = act, "eat"
        need = NEED_EXPLORE["afternoon" if dl.phase(step) == "afternoon" else "day"]
        hungry = v.inv["food"] <= need and not v.cows and not v.ripe
        thirsty = v.inv["drink"] <= need and not v.m("water").any()
        if (name == "noop" or hungry or thirsty) and self.shelter is None and not dl.is_dark(step):
            act = self._explore(v, memory)
            if act:
                name, option = act, "explore"
        name = self._safety(v, name)
        if option == "routine" and self.last_task is not None and name != "noop":
            option = f"routine:{self.last_task}"
        return name, option

    # -- options: temporally extended courses of action for the learned selector ------------------------------
    def option_feasible(self, v: View, memory: dict) -> np.ndarray:
        """Which of OPTIONS can start now (on its beliefs)."""
        inv = v.inv
        threats = self._threats(v, 6)
        return np.array([True, inv["drink"] < 9, inv["food"] < 9,
                         bool(inv["wood_pickaxe"]) or self._inside(v),
                         bool(threats), bool((memory["map"] < 0).any()),
                         any(o["family"] in ("collect", "craft", "place", "visit") for o in v.priv["obligations"]),
                         True], dtype=bool)

    def option_done(self, o: int, v: View, steps_in: int) -> bool:
        inv, dark = v.inv, dl.is_dark(v.step)
        name = OPTIONS[o]
        if steps_in >= OPTION_MAX[name]:
            return True
        if name == "drink":
            return inv["drink"] >= 9
        if name == "eat":
            return inv["food"] >= 8
        if name == "rest":
            return not dark and inv["energy"] >= 7 and not v.sleeping
        if name == "retreat":
            return not self._threats(v, 6)
        return False

    def option_act(self, o: int, v: View, memory: dict, step: int) -> str | None:
        """The option's action now, or None when it cannot go on (the selector then chooses again)."""
        name = OPTIONS[o]
        self.view = v                                              # the movement primitives plan on this view
        for z in v.zombies:                                        # reflex: an adjacent zombie is fought
            if abs(z[0] - v.pos[0]) + abs(z[1] - v.pos[1]) == 1:
                u = (z[0] - v.pos[0], z[1] - v.pos[1])
                return "do" if v.facing == u else MOVE[u]
        act = None
        if name == "drink":
            act = self._drink(v) or self._explore(v, memory)
        elif name == "eat":
            act = self._eat(v) or self._explore(v, memory)
        elif name == "rest":
            dark = dl.is_dark(step)
            if self._inside(v) and (dark or v.inv["energy"] < 9):
                act = self._shelter_step(v, needed=True)
            else:
                act = self._enter_shelter(v, {"night"} if dark else {"sleep"})
            if act is None and v.inv["energy"] <= 3:
                act = "sleep"                                      # exhausted: sleep where it stands
        elif name == "retreat":
            act = self._retreat(v)
        elif name == "explore":
            act = self._explore(v, memory)
        elif name == "requests":
            act = self._obligations(v)
        elif name == "build":
            act = self._tech_tree(v)
            if act is None and v.inv["wood_pickaxe"] and v.inv["stone"] < 6:
                act = self._acquire(v, "stone", 6)
        return None if act is None else self._safety(v, act)

    def _threats(self, v: View, r: int) -> list:
        return [z for z in v.zombies + v.skeletons if abs(z[0] - v.pos[0]) + abs(z[1] - v.pos[1]) <= r]

    def _retreat(self, v: View) -> str | None:
        threats = self._threats(v, 8)
        if not threats:
            return None

        def score(p):
            return min(abs(z[0] - p[0]) + abs(z[1] - p[1]) for z in threats)
        best, act = score(v.pos), None
        for u in sorted(MOVE):
            n = (v.pos[0] + u[0], v.pos[1] + u[1])
            if 0 <= n[0] < mem.W and 0 <= n[1] < mem.H and self.walkfree[n] and score(n) > best:
                best, act = score(n), MOVE[u]
        return act

    def _pain(self, body, step):
        inv = body["inventory"]
        last = getattr(self, "last_health", None)
        self.last_health = int(inv["health"])
        if self.shelter is None or last is None or inv["health"] >= last:
            return
        if inv["food"] == 0 or inv["drink"] == 0 or inv["energy"] == 0:
            return                                              # a feeling it understands, not an unseen attacker
        hurts = [s for s in getattr(self, "pain_steps", []) if step - s <= PAIN_WINDOW] + [step]
        self.pain_steps = hurts
        if len(hurts) >= PAIN_HITS:
            q = self.shelter["q"]
            self.bad_spots = list(getattr(self, "bad_spots", [])) + [[int(q[0]), int(q[1]), int(step)]]
            self.shelter = None
            self.caution_until = step + CAUTION
            self.pain_steps = []

    def _visit_progress(self, memory, body, commitments):
        """Its own record of having reached a visit target: within reach of a believed feature tile that lies
        in the stated direction, at least min_dist from the start.  Kept in the skill state."""
        from continuous_brain_world.obligations import in_cone
        reached = dict(getattr(self, "visit_reached", None) or {})
        pos = tuple(body["player_pos"])
        out = []
        for o in commitments:
            if o["family"] == "visit":
                spec = o["spec"]
                if o["id"] not in reached:
                    mid = MATERIAL_IDS.get(spec["feature"])
                    xs, ys = np.nonzero(memory["map"] == mid)
                    start = tuple(spec["start"])
                    for x, y in zip(xs.tolist(), ys.tolist()):
                        if max(abs(x - pos[0]), abs(y - pos[1])) <= spec["reach_radius"] and \
                                max(abs(x - start[0]), abs(y - start[1])) >= spec["min_dist"] and \
                                in_cone(start, (x, y), spec["direction"]):
                            reached[o["id"]] = int(memory["steps"])
                            break
                o = {**o, "progress": {"reached_step": reached.get(o["id"])}}
            out.append(o)
        self.visit_reached = reached
        return out

    def _explore(self, v, memory) -> str | None:
        unknown = memory["map"] < 0
        if not unknown.any():
            return None
        reach = np.where(unknown & (self.dist < INF), self.dist, INF)
        tunnels = memory["map"] == MATERIAL_IDS["path"]
        if tunnels.any():
            from scipy.ndimage import binary_dilation
            near_tunnel = binary_dilation(tunnels, iterations=3)
            reach = np.where((reach < INF) & near_tunnel, reach + TUNNEL_FRONTIER_PENALTY, reach)
        tgt = getattr(self, "explore_target", None)
        if tgt is not None and not (unknown[tuple(tgt)] and reach[tuple(tgt)] < INF):
            tgt = None
        if tgt is None:
            if reach.min() >= INF:
                return None
            flat = int(np.argmin(reach))          # nearest frontier; ties broken by position (deterministic)
            tgt = [flat // mem.H, flat % mem.H]
        self.explore_target = tgt
        return self._step_toward(v, tuple(tgt))
