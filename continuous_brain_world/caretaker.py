"""The caretaker: a seeded, scripted speaker that issues, amends and cancels requests in English.

Everything here draws from one ``numpy.random.Generator`` (PCG64) whose state lives in the layer's JSON state,
so the caretaker is a deterministic function of (seed, config, what happened in the world).

Curriculum of pressure (``day`` = daylight day index, see ``daylight.py``):

* arrival hazard per step  ``rate(day) = min(rate_max, rate0 * (1 + rate_growth * day))`` while fewer than
  ``max_concurrent`` obligations are active;
* difficulty ``d(day) = min(1, difficulty0 + difficulty_growth * day)`` gates which items/tools/places may be
  asked for, scales quantities, and shrinks the deadline slack ``max(1.3, 3.0 - 1.6 d)`` (multiplier on a
  per-request estimate of the time a competent agent needs from its current inventory);
* change hazard per step ``change_rate * n_changeable``; a change is a cancellation with probability
  ``cancel_fraction`` and otherwise an amendment (new quantity / tool / threshold / duration / deadline).

The language is template-based (see ``language_inventory()`` for the counts): several paraphrases per family,
synonyms for items and features, digit-or-word quantities, five deadline phrasings computed from the daylight
cycle, and value markers.  It is varied but closed-world; docs/explore/W1_WORLD.md states the limits.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import daylight as dl
from .obligations import chebyshev, is_active
from .utterance import compass

# --- vocabulary ------------------------------------------------------------------------------------------
ITEM_NOUNS = {
    "wood": [("wood", "wood"), ("piece of wood", "pieces of wood"), ("log", "logs")],
    "sapling": [("sapling", "saplings")],
    "stone": [("stone", "stones"), ("piece of stone", "pieces of stone")],
    "coal": [("lump of coal", "lumps of coal"), ("coal", "coal")],
    "iron": [("piece of iron", "pieces of iron"), ("iron", "iron")],
    "diamond": [("diamond", "diamonds")],
}
ITEM_KEYWORDS = {"wood": ["wood", "logs", "log"], "sapling": ["sapling", "saplings"],
                 "stone": ["stone", "stones"], "coal": ["coal"], "iron": ["iron"], "diamond": ["diamond", "diamonds"]}
TOOL_NAMES = {
    "wood_pickaxe": ["wood pickaxe", "wooden pickaxe"], "stone_pickaxe": ["stone pickaxe"],
    "iron_pickaxe": ["iron pickaxe"], "wood_sword": ["wood sword", "wooden sword"],
    "stone_sword": ["stone sword"], "iron_sword": ["iron sword"],
}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FEATURE_NAMES = {"water": ["the lake", "the water", "the pond"], "sand": ["the beach", "the sand"],
                 "tree": ["the trees", "the woods", "the grove"], "stone": ["the rocks", "the mountain", "the cliffs"]}
FEATURE_KEYWORDS = {"water": ["lake", "water", "pond"], "sand": ["beach", "sand"],
                    "tree": ["trees", "woods", "grove", "tree"], "stone": ["rocks", "mountain", "cliffs", "rock"]}
STAT_NAMES = {"food": ["food", "hunger bar"], "drink": ["drink", "thirst bar", "water level"],
              "energy": ["energy", "energy level"], "health": ["health"]}
KIND_PHRASES = {"cow": "a cow", "zombie": "a zombie", "skeleton": "a skeleton", "sapling": "a sapling",
                "water": "water", "tree": "a tree", "coal": "coal", "iron": "iron", "diamond": "a diamond",
                "lava": "lava", "table": "a table", "furnace": "a furnace"}
COUNT_EVENTS = {
    "defeat_zombie": ["How many zombies did you defeat today?", "How many zombies have you killed today?"],
    "eat_cow": ["How many cows did you eat today?", "How many cows have you eaten today?"],
    "collect_drink": ["How many times did you drink today?", "How often have you had a drink today?"],
    "place_plant": ["How many saplings did you plant today?"],
    "collect_wood": ["How many times did you collect wood today?", "How much wood did you chop today?"],
    "defeat_skeleton": ["How many skeletons did you defeat today?"],
}

T_COLLECT = ["Please gather {q} {dl}.", "Could you collect {q} for me {dl}?", "I need {q} {dl}.",
             "Bring me {q} {dl}, please.", "Fetch {q} {dl}.", "Can you get hold of {q} {dl}?",
             "Go and get {q} {dl}."]
T_CRAFT = ["Make me a {t} {dl}.", "Please craft a {t} {dl}.", "I'd like a new {t} {dl}.",
           "Could you make a {t} {dl}?", "Craft one {t} {dl}, would you?", "Build me a {t} {dl}."]
T_PLACE = {
    ("table", "near", "water"): ["Put a table within {r} tiles of water {dl}.",
                                 "Set up a crafting table no more than {r} tiles from the water {dl}.",
                                 "Build a table close to water (within {r} tiles) {dl}."],
    ("plant", "near", "table"): ["Plant a sapling near your table, within {r} tiles, {dl}.",
                                 "Put a sapling in the ground no more than {r} tiles from a table {dl}."],
    ("plant", "near", "water"): ["Plant a sapling within {r} tiles of water {dl}.",
                                 "Grow something by the water: plant a sapling within {r} tiles of it {dl}."],
    ("furnace", "near", "table"): ["Place a furnace within {r} tiles of a table {dl}.",
                                   "Build a furnace beside a table (at most {r} tiles away) {dl}."],
    ("stone", "onto", "water"): ["Drop a stone into the water {dl}.", "Fill in a water tile with stone {dl}.",
                                 "Lay a stone block on the water {dl}."],
}
T_MAINTAIN = ["Keep your {s} at {n} or above {u}.", "Don't let your {s} drop below {n} {u}.",
              "Make sure your {s} stays at least {n} {u}."]
T_AVOID = {
    "zombie": ["Don't take damage from zombies {u}.", "Don't let a zombie hit you {u}.",
               "No zombie bites {u}, please."],
    "arrow": ["Don't get hit by any arrows {u}.", "Stay clear of skeleton arrows {u}."],
    "any": ["Try not to get hurt at all {u}.", "Don't lose any health {u}.", "Stay unharmed {u}."],
}
T_VISIT = ["Go look at {f} to the {d} and come back {dl}.",
           "Head {d} to {f}, take a look, and return here {dl}.",
           "There is {f} off to the {d}, about {n} tiles away. Go and see it, then come back {dl}."]
T_LAST_SEEN = ["Where did you last see {k}?", "Remind me: where did you last spot {k}?",
               "Which way, and how far, is the last {k2} you saw?"]
T_FIRST = {"this_morning": ["What did I ask you for first this morning?",
                            "What was my first request this morning?"],
           "today": ["What was the first thing I asked you for today?"]}
T_COUNT_REQ = ["How many tasks (not questions) have I given you today?", "How many requests have I made today?"]
T_LAST_CANCEL = ["Which request did I cancel most recently?", "What did I last call off?"]
T_CANCEL = ["Never mind the {k}.", "Forget about the {k}, I don't need it any more.",
            "Cancel that request, please; ignore it."]
VALUE_MARK = {3: ["This is important: ", "Top priority: ", "Urgent: "], 2: [""],
              1: ["When you can: ", "Low priority: "]}


def language_inventory() -> dict:
    """Counts of the caretaker's templates and synonyms (reported in the docs)."""
    return {
        "request_templates": len(T_COLLECT) + len(T_CRAFT) + sum(map(len, T_PLACE.values())) + len(T_MAINTAIN)
        + sum(map(len, T_AVOID.values())) + len(T_VISIT),
        "question_templates": len(T_LAST_SEEN) + sum(map(len, COUNT_EVENTS.values())) + sum(map(len, T_FIRST.values()))
        + len(T_COUNT_REQ) + len(T_LAST_CANCEL),
        "change_templates": len(T_CANCEL) + 8,
        "deadline_phrasings": 7, "item_synonyms": sum(map(len, ITEM_NOUNS.values())),
        "feature_synonyms": sum(map(len, FEATURE_NAMES.values())), "value_markers": sum(map(len, VALUE_MARK.values())),
    }


# --- helpers ---------------------------------------------------------------------------------------------
def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _qty(rng, n: int, item: str) -> str:
    sing, plur = _pick(rng, ITEM_NOUNS[item])
    if n == 1:
        art = "an" if sing[0] in "aeiou" else "a"
        return f"{_pick(rng, [art, 'one'])} {sing}" if sing != plur else f"{_pick(rng, ['one', '1'])} {sing}"
    word = NUMBER_WORDS[n] if rng.random() < 0.5 else str(n)
    return f"{word} more {plur}" if rng.random() < 0.5 else f"{word} {plur}"


def max_quantity(item: str, d: float) -> int:
    """Largest quantity the caretaker asks for at difficulty ``d`` (an amendment may ask for one more)."""
    return {"wood": 1 + round(4 * d), "stone": 1 + round(4 * d), "sapling": 1 + round(d),
            "coal": 1 + round(2 * d), "iron": 1, "diamond": 1}[item]


def _round10(x: float) -> int:
    return max(10, int(round(x / 10.0)) * 10)


def dark_steps(start: int, end: int) -> int:
    """Number of dark steps in the half-open interval [start, end)."""
    full, rest = divmod(max(0, end - start), dl.DAY_LENGTH)
    n = full * (dl.DAWN_T - dl.NIGHTFALL_T)
    return n + sum(dl.NIGHTFALL_T <= (start + full * dl.DAY_LENGTH + i) % dl.DAY_LENGTH < dl.DAWN_T
                   for i in range(rest))


def choose_deadline(rng, step: int, dur: float, until: bool = False) -> tuple[int, str]:
    """Pick a deadline phrase whose implied deadline is close to ``step + dur``; return (deadline, phrase).

    For tasks (``until=False``) ``dur`` is working time: the dark steps it would span are added on top, so a
    request never silently expects work through the night (the caretaker knows nights are for sheltering).
    For "keep/avoid ... until" windows ``dur`` is the window itself."""
    if not until:
        dur = dur + dark_steps(step, step + int(dur))
    n = _round10(dur)
    options = [(step + n, (f"until step {step + n}" if until else f"by step {step + n}")),
               (step + n, (f"for the next {n} steps" if until else f"within {n} steps"))]
    ph = dl.phase(step)
    lo, hi = 0.6 * dur, 2.0 * dur
    if ph != "night":
        d = dl.next_nightfall(step) - 1
        if lo <= d - step <= hi and d - step >= 30:
            options.append((d, "until nightfall" if until else _pick(rng, ["before nightfall", "before it gets dark"])))
    d = dl.next_dawn(step)
    if lo <= d - step <= hi * 1.1 and d - step >= 30:
        if until:
            options.append((d, "until morning" if ph == "night" else "through the night until morning"))
        else:
            options.append((d, "by morning" if ph == "night" else "by tomorrow morning"))
    if ph == "morning" and not until:
        d = dl.next_noon(step)
        if lo <= d - step <= hi and d - step >= 30:
            options.append((d, "by noon"))
    return _pick(rng, options)


def _reachable(world, start, walkable_ids) -> np.ndarray:
    """Cells reachable from ``start`` over walkable materials (objects ignored)."""
    W, H = world.area
    mat = world._mat_map
    seen = np.zeros((W, H), bool)
    sx, sy = int(start[0]), int(start[1])
    seen[sx, sy] = True
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H and not seen[nx, ny] and int(mat[nx, ny]) in walkable_ids:
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen


class Caretaker:
    """Stateless apart from the layer state it is handed (rng lives there)."""

    def __init__(self, cfg):
        self.cfg = cfg

    # -- curriculum -------------------------------------------------------------------------------------
    def rate(self, day: int) -> float:
        c = self.cfg
        return min(c.arrival_rate_max, c.arrival_rate0 * (1.0 + c.arrival_growth * day))

    def difficulty(self, day: int) -> float:
        return min(1.0, self.cfg.difficulty0 + self.cfg.difficulty_growth * day)

    def slack(self, day: int) -> float:
        return max(1.3, 3.0 - 1.6 * self.difficulty(day))

    # -- generation -------------------------------------------------------------------------------------
    def propose(self, rng, layer, ctx) -> dict | None:
        """Draw one new obligation (dict without id/text prefix) or None if no family is feasible."""
        day = dl.day_index(ctx.step)
        weights = dict(self.cfg.family_weights)
        for _ in range(4):
            fams = [f for f, w in weights.items() if w > 0]
            if not fams:
                return None
            p = np.array([weights[f] for f in fams], float)
            fam = fams[int(rng.choice(len(fams), p=p / p.sum()))]
            ob = getattr(self, "_gen_" + fam)(rng, layer, ctx, self.difficulty(day), self.slack(day))
            if ob is not None:
                ob["family"] = fam
                return ob
            weights[fam] = 0.0
        return None

    def _value(self, rng) -> int:
        return int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))

    def _busy(self, layer, fam, key, val) -> bool:
        return any(is_active(o) and o["family"] == fam and o["spec"].get(key) == val for o in layer.obligations)

    def _gen_collect(self, rng, layer, ctx, d, slack):
        inv = ctx.inventory
        items = ["wood", "sapling"] + (["stone", "coal"] if d >= 0.2 else []) + (["iron"] if d >= 0.5 else []) \
            + (["diamond"] if d >= 0.85 else [])
        items = [i for i in items if inv[i] < 9 and not self._busy(layer, "collect", "item", i)]
        if not items:
            return None
        item = _pick(rng, items)
        k = int(rng.integers(1, max_quantity(item, d) + 1))
        k = max(1, min(k, 9 - inv[item]))
        est = self._collect_time(item, k, inv)
        value = self._value(rng)
        deadline, dphrase = choose_deadline(rng, ctx.step, est * slack)
        text = _pick(rng, T_COLLECT).format(q=_qty(rng, k, item), dl=dphrase)
        return {"spec": {"item": item, "quantity": k, "base": int(inv[item]), "target": int(inv[item]) + k},
                "deadline": deadline, "value": value, "text": text, "keywords": list(ITEM_KEYWORDS[item]),
                "estimate": est}

    @staticmethod
    def _collect_time(item, k, inv) -> float:
        need_wp = 0 if inv["wood_pickaxe"] else 90
        need_sp = 0 if inv["stone_pickaxe"] else 120 + need_wp
        need_ip = 0 if inv["iron_pickaxe"] else 450 + need_sp
        return {"wood": 30 + 20 * k, "sapling": 40 + 45 * k, "stone": 50 + need_wp + 15 * k,
                "coal": 80 + need_wp + 25 * k, "iron": 160 + need_sp + 60 * k,
                "diamond": 300 + need_ip + 100 * k}[item]

    def _gen_craft(self, rng, layer, ctx, d, slack):
        inv = ctx.inventory
        tools = ["wood_pickaxe", "wood_sword"] + (["stone_pickaxe", "stone_sword"] if d >= 0.3 else []) \
            + (["iron_pickaxe", "iron_sword"] if d >= 0.65 else [])
        tools = [t for t in tools if inv[t] < 9 and not self._busy(layer, "craft", "tool", t)]
        if not tools:
            return None
        tool = _pick(rng, tools)
        est = self._craft_time(tool, inv)
        deadline, dphrase = choose_deadline(rng, ctx.step, est * slack)
        name = _pick(rng, TOOL_NAMES[tool])
        return {"spec": {"tool": tool, "base": int(ctx.achievements["make_" + tool])},
                "deadline": deadline, "value": self._value(rng),
                "text": _pick(rng, T_CRAFT).format(t=name, dl=dphrase),
                "keywords": list(TOOL_NAMES[tool]), "estimate": est}

    @staticmethod
    def _craft_time(tool, inv) -> float:
        tier = tool.split("_")[0]
        if tier == "wood":
            return 90
        if tier == "stone":
            return 120 + (0 if inv["wood_pickaxe"] else 90)
        return 450 + (0 if inv["stone_pickaxe"] else 200)

    def _gen_place(self, rng, layer, ctx, d, slack):
        variants = [("table", "near", "water", 4, 120), ("plant", "near", "table", 3, 160),
                    ("plant", "near", "water", 3, 140)]
        if d >= 0.3:
            variants.append(("stone", "onto", "water", 0, 150))
        if d >= 0.4:
            variants.append(("furnace", "near", "table", 2, 260))
        variants = [v for v in variants if not self._busy(layer, "place", "what", v[0])]
        if not variants:
            return None
        what, rel, mat, radius, est = variants[int(rng.integers(len(variants)))]
        if what == "stone" and ctx.inventory["stone"] > 0:
            est = 80
        deadline, dphrase = choose_deadline(rng, ctx.step, est * slack)
        spec = {"what": what, rel: mat}
        if rel == "near":
            spec["radius"] = radius
        spec["base"] = int(ctx.achievements["place_" + what])
        text = _pick(rng, T_PLACE[(what, rel, mat)]).format(r=radius, dl=dphrase)
        kw = {"table": ["table", "crafting table"], "plant": ["sapling", "plant"], "furnace": ["furnace"],
              "stone": ["stone", "stone block"]}[what]
        return {"spec": spec, "deadline": deadline, "value": self._value(rng), "text": text, "keywords": kw,
                "estimate": est}

    def _gen_maintain(self, rng, layer, ctx, d, slack):
        inv = ctx.inventory
        stats = [s for s in ("food", "drink", "energy", "health") if not self._busy(layer, "maintain", "stat", s)]
        stats = [s for s in stats if inv[s] >= 4]
        if not stats:
            return None
        stat = _pick(rng, stats)
        hi = min(inv[stat] - 1, 3 + round(3 * d))
        thr = int(rng.integers(2, max(2, hi) + 1))
        dur = float(rng.integers(150, 300 + round(250 * d)))
        deadline, uphrase = choose_deadline(rng, ctx.step, dur, until=True)
        text = _pick(rng, T_MAINTAIN).format(s=_pick(rng, STAT_NAMES[stat]), n=thr, u=uphrase)
        return {"spec": {"stat": stat, "threshold": thr}, "deadline": deadline, "value": self._value(rng),
                "text": text, "keywords": [stat] + STAT_NAMES[stat][1:], "estimate": dur}

    def _gen_avoid(self, rng, layer, ctx, d, slack):
        sources = [s for s in ("zombie", "any", "arrow") if not self._busy(layer, "avoid", "source", s)]
        if not sources:
            return None
        src = _pick(rng, sources)
        lo, hi = {"zombie": (150, 300), "arrow": (200, 400), "any": (100, 200)}[src]
        dur = float(rng.integers(lo, hi + round(150 * d) + 1))
        deadline, uphrase = choose_deadline(rng, ctx.step, dur, until=True)
        text = _pick(rng, T_AVOID[src]).format(u=uphrase)
        kw = {"zombie": ["zombies", "zombie", "zombie damage"], "arrow": ["arrows", "arrow", "skeletons"],
              "any": ["damage", "getting hurt", "health"]}[src]
        return {"spec": {"source": src}, "deadline": deadline, "value": self._value(rng), "text": text,
                "keywords": kw, "estimate": dur}

    def _gen_visit(self, rng, layer, ctx, d, slack):
        if any(is_active(o) and o["family"] == "visit" for o in layer.obligations):
            return None
        world = ctx.world
        start = (int(ctx.player_pos[0]), int(ctx.player_pos[1]))
        walk = {world._mat_ids[m] for m in ("grass", "sand", "path")}
        reach = _reachable(world, start, walk)
        dmin, dmax = 10, 14 + round(10 * d)
        mat = world._mat_map
        r = 3
        # approachable[x, y]: some reachable cell within Chebyshev r (dilation of the reachable mask)
        W, H = world.area
        pad = np.pad(reach, r)
        approach = np.zeros_like(reach)
        for ox in range(2 * r + 1):
            for oy in range(2 * r + 1):
                approach |= pad[ox:ox + W, oy:oy + H]
        features = [f for f in ("water", "sand", "tree", "stone") if f in world._mat_ids]
        rng_order = [features[i] for i in rng.permutation(len(features))]
        for feat in rng_order:
            xs, ys = np.nonzero((mat == world._mat_ids[feat]) & approach)
            cands = [(int(x), int(y)) for x, y in zip(xs, ys) if dmin <= chebyshev((x, y), start) <= dmax]
            if not cands:
                continue
            tx, ty = cands[int(rng.integers(len(cands)))]
            direction = compass(tx - start[0], ty - start[1])
            dist = chebyshev((tx, ty), start)
            est = 3.0 * dist + 40
            deadline, dphrase = choose_deadline(rng, ctx.step, est * slack)
            fname = _pick(rng, FEATURE_NAMES[feat])
            text = _pick(rng, T_VISIT).format(f=fname, d=direction, n=int(round(dist / 5.0) * 5) or dist, dl=dphrase)
            spec = {"feature": feat, "direction": direction, "start": list(start), "min_dist": max(7, dist - 6),
                    "reach_radius": r, "return_radius": 3, "example_target": [tx, ty]}
            return {"spec": spec, "deadline": deadline, "value": self._value(rng), "text": text,
                    "keywords": list(FEATURE_KEYWORDS[feat]), "estimate": est, "progress": {"reached_step": None}}
        return None

    def _gen_report(self, rng, layer, ctx, d, slack):
        if ctx.step < 60:
            return None
        day = dl.day_index(ctx.step)
        options = []
        seen = layer.state["visibility"]["last_seen"]
        hidden = [k for k in KIND_PHRASES if k in seen and not layer.visibility.visible_now(k)]
        if hidden:
            options += ["last_seen"] * 3
        unseen = [k for k in ("diamond", "skeleton", "lava", "iron", "coal", "furnace", "table") if k not in seen]
        if unseen:
            options.append("last_seen_never")
        options += ["count_today"] * 2
        if dl.phase(ctx.step) != "morning":
            options.append("first_this_morning")
        options += ["first_today", "count_requests"]
        if any(o["status"] == "cancelled" for o in layer.obligations):
            options.append("last_cancelled")
        choice = _pick(rng, options)
        window = int(self.cfg.report_window)
        spec: dict
        if choice in ("last_seen", "last_seen_never"):
            kind = _pick(rng, hidden if choice == "last_seen" else unseen)
            k2 = KIND_PHRASES[kind].split(" ", 1)[-1]
            text = _pick(rng, T_LAST_SEEN).format(k=KIND_PHRASES[kind], k2=k2)
            spec = {"question": "last_seen", "kind": kind}
        elif choice == "count_today":
            event = _pick(rng, sorted(COUNT_EVENTS))
            text = _pick(rng, COUNT_EVENTS[event])
            spec = {"question": "count_today", "event": event, "day": day}
        elif choice == "first_this_morning":
            spec = {"question": "first_request", "window": list(dl.morning_window(day)), "which": "this_morning"}
            text = _pick(rng, T_FIRST["this_morning"])
        elif choice == "first_today":
            spec = {"question": "first_request", "window": [dl.day_start(day), ctx.step], "which": "today"}
            text = _pick(rng, T_FIRST["today"])
        elif choice == "count_requests":
            spec = {"question": "count_requests", "window": [dl.day_start(day), ctx.step]}
            text = _pick(rng, T_COUNT_REQ)
        else:
            spec = {"question": "last_cancelled"}
            text = _pick(rng, T_LAST_CANCEL)
        text = f"{text} (Answer within {window} steps.)"
        return {"spec": spec, "deadline": ctx.step + window, "value": 1 + int(rng.random() < 0.3), "text": text,
                "keywords": ["question"], "estimate": window}

    # -- changes ----------------------------------------------------------------------------------------
    def change(self, rng, ob: dict, ctx) -> tuple[str, str, dict] | None:
        """Return ('cancel'|'amend', message_body, changes) for an active obligation, or None."""
        if rng.random() < self.cfg.cancel_fraction:
            k = ob["keywords"][0]
            return "cancel", _pick(rng, T_CANCEL).format(k=k), {}
        fam, spec, step = ob["family"], ob["spec"], ctx.step
        left = ob["deadline"] - step
        if left < 40:
            return None
        if fam == "collect":
            room = min(9 - spec["base"], max_quantity(spec["item"], self.difficulty(dl.day_index(step))) + 1)
            choices = [k for k in range(1, room + 1) if k != spec["quantity"]]
            if not choices:
                return None
            k = int(_pick(rng, choices))
            deadline, dphrase = choose_deadline(rng, step, max(left, 60.0))
            body = (f"Change of plan: I need {_qty(rng, k, spec['item'])} after all "
                    f"(counting from when I first asked), {dphrase}.")
            return "amend", body, {"spec": {"quantity": k, "target": spec["base"] + k}, "deadline": deadline}
        if fam == "craft":
            tier = spec["tool"].split("_")[0]
            other = {"pickaxe": "sword", "sword": "pickaxe"}[spec["tool"].split("_")[1]]
            tool = f"{tier}_{other}"
            if rng.random() < 0.5 and ctx.inventory[tool] < 9:
                deadline, dphrase = choose_deadline(rng, step, max(left, 60.0))
                body = f"Change of plan: make it a {TOOL_NAMES[tool][0]} instead, {dphrase}."
                return "amend", body, {"spec": {"tool": tool, "base": int(ctx.achievements["make_" + tool])},
                                       "deadline": deadline, "keywords": list(TOOL_NAMES[tool])}
            deadline, dphrase = choose_deadline(rng, step, left * float(rng.uniform(0.7, 1.6)))
            return "amend", f"Update: I now need the {TOOL_NAMES[spec['tool']][0]} {dphrase}.", {"deadline": deadline}
        if fam == "maintain":
            if rng.random() < 0.5:
                thr = int(rng.integers(2, max(2, min(ctx.inventory[spec["stat"]], 7)) + 1))
                if thr == spec["threshold"]:
                    return None
                return "amend", f"Change: keep your {spec['stat']} at {thr} or above instead, same time frame.", \
                    {"spec": {"threshold": thr}}
            deadline, uphrase = choose_deadline(rng, step, left + float(rng.integers(50, 200)), until=True)
            if deadline <= ob["deadline"]:
                return None
            return "amend", f"Please keep that up {uphrase}.", {"deadline": deadline}
        if fam == "avoid":
            deadline, uphrase = choose_deadline(rng, step, left + float(rng.integers(50, 200)), until=True)
            if deadline <= ob["deadline"]:
                return None
            return "amend", f"Let's extend that: same rule {uphrase}.", {"deadline": deadline}
        # place / visit: new deadline
        deadline, dphrase = choose_deadline(rng, step, left * float(rng.uniform(0.7, 1.6)))
        noun = ob["keywords"][0] if fam == "place" else "trip"
        body = _pick(rng, [f"Update: I need the {noun} {dphrase} now.", f"New deadline for the {noun}: {dphrase}."])
        return "amend", body, {"deadline": deadline}
