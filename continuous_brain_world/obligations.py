"""Obligation records and the exact verification rule of every family.

An obligation is a plain JSON dict (it lives in the canonical store)::

    {"id": "o3", "family": "collect", "text": "[o3] Please gather 3 wood before nightfall.",
     "spec": {...}, "issue_step": 120, "deadline": 147, "value": 2,
     "status": "open" | "amended" | "fulfilled" | "failed" | "cancelled",
     "resolved_step": None | int, "reason": None | str, "keywords": [...], "history": [...]}

``open`` and ``amended`` are the two *active* statuses (``amended`` = still active, terms revised at least once).
Every verification reads Crafter's internal state through a ``StepContext`` -- never the agent's claims.  An
obligation issued at step ``i`` is first evaluated at step ``i + 1``; its ``deadline`` is the last step at
which it can still be met.  Rules (``s`` = the step being evaluated, ``i < s <= deadline``):

collect   fulfilled at the first ``s`` with ``inventory[item] >= spec.target`` where ``target = base + quantity``
          and ``base`` = inventory at issue (net gain since the request; capped at Crafter's max of 9 by
          construction); failed at ``s == deadline`` otherwise.
craft     fulfilled at the first ``s`` with ``achievements['make_' + tool] > spec.base`` (a *new* item was
          made after the request; ``base`` is re-read when an amendment changes the tool).
place     fulfilled at the first ``s`` at which Crafter registers a placement of ``spec.what`` (its
          ``place_*`` achievement counter rises) whose tile satisfies the spatial clause on the map at ``s``:
          ``near``: some tile of material ``spec.near`` other than the placed tile itself lies within
          Chebyshev distance ``spec.radius``; ``onto``: the tile's material just before placement was
          ``spec.onto``.
maintain  failed at the first ``s`` with ``inventory[stat] < spec.threshold``; fulfilled at ``s == deadline``
          if it never happened.
avoid     failed at the first ``s`` in which the player took damage of ``spec.source``; fulfilled at
          ``s == deadline`` otherwise.  ``zombie``: some zombie's attack cooldown went 0 -> 5 during the step
          (Crafter sets it to 5 exactly when that zombie hits the player); ``arrow``: an arrow whose next tile
          was the player's tile was consumed during the step; ``any``: the player's health is lower than before
          the step (regeneration is at most +1 and every damage source is at least -1 within a step, so a net
          drop occurs iff damage occurred).
visit     two stages.  *reached* at the first ``s`` with the player within Chebyshev ``reach_radius`` of a tile
          of material ``spec.feature`` that lies at Chebyshev distance ``>= spec.min_dist`` from ``spec.start``
          and within 45 degrees of the stated direction (as seen from ``start``); then fulfilled at the first
          later ``s`` with the player within Chebyshev ``return_radius`` of ``start``.  Failed at the deadline.
report    graded once, at the step in which the first ``answer <id>: ...`` clause for it arrives (against the
          state the agent was looking at when it spoke); correct -> fulfilled, wrong -> failed; no answer by the
          deadline -> failed (reason ``no_answer``).

Death fails every active obligation (reason ``died``).  Obligations still active when the episode reaches its
step limit stay active and are counted as *unresolved* (excluded from rates, reported separately).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import daylight as dl
from .utterance import compass, distance_bucket, format_offset, keyword_norm

FAMILIES = ("collect", "craft", "place", "maintain", "avoid", "visit", "report")
ACTIVE = ("open", "amended")
DIRECTION_VECTORS = {
    "east": (1.0, 0.0), "north-east": (1.0, -1.0), "north": (0.0, -1.0), "north-west": (-1.0, -1.0),
    "west": (-1.0, 0.0), "south-west": (-1.0, 1.0), "south": (0.0, 1.0), "south-east": (1.0, 1.0),
}
COS45 = math.cos(math.radians(45.0))


def is_active(ob: dict) -> bool:
    return ob["status"] in ACTIVE


def chebyshev(a, b) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def in_cone(start, tile, direction: str) -> bool:
    """Whether ``tile`` is within 45 degrees of ``direction`` as seen from ``start``."""
    vx, vy = float(tile[0] - start[0]), float(tile[1] - start[1])
    ux, uy = DIRECTION_VECTORS[direction]
    norm_v = math.hypot(vx, vy)
    if norm_v == 0:
        return False
    return (vx * ux + vy * uy) / (norm_v * math.hypot(ux, uy)) >= COS45 - 1e-12


@dataclass
class StepContext:
    """Ground truth of one environment step, read from Crafter's internals by the layer."""
    step: int
    inventory: dict
    achievements: dict
    player_pos: tuple
    alive: bool
    world: object                      # crafter.engine.World (read-only use)
    placements: list = field(default_factory=list)   # [{'what','pos','onto'}]
    zombie_hits: int = 0
    arrow_hits: int = 0
    health_drop: bool = False


def material_at(world, pos) -> str | None:
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= x < world.area[0] and 0 <= y < world.area[1]):
        return None
    return world._mat_names[int(world._mat_map[x, y])]


def material_within(world, pos, material: str, radius: int, exclude_center: bool = True) -> bool:
    mid = world._mat_ids.get(material)
    if mid is None:
        return False
    x, y = int(pos[0]), int(pos[1])
    x0, x1 = max(0, x - radius), min(world.area[0], x + radius + 1)
    y0, y1 = max(0, y - radius), min(world.area[1], y + radius + 1)
    region = world._mat_map[x0:x1, y0:y1] == mid
    if exclude_center:
        region[x - x0, y - y0] = False
    return bool(region.any())


def visit_reached(world, spec: dict, player_pos) -> bool:
    mid = world._mat_ids.get(spec["feature"])
    r = spec["reach_radius"]
    px, py = int(player_pos[0]), int(player_pos[1])
    for x in range(max(0, px - r), min(world.area[0], px + r + 1)):
        for y in range(max(0, py - r), min(world.area[1], py + r + 1)):
            if int(world._mat_map[x, y]) != mid:
                continue
            if chebyshev((x, y), spec["start"]) >= spec["min_dist"] and in_cone(spec["start"], (x, y),
                                                                             spec["direction"]):
                return True
    return False


def evaluate(ob: dict, ctx: StepContext) -> tuple[str, str] | None:
    """Return (new_status, reason) if ``ob`` resolves at ``ctx.step``, else None.  Pure; mutates only
    ``ob['progress']`` (the visit stage)."""
    fam, spec, s = ob["family"], ob["spec"], ctx.step
    if s <= ob["issue_step"]:
        return None
    due = s >= ob["deadline"]
    if fam == "collect":
        if ctx.inventory[spec["item"]] >= spec["target"]:
            return "fulfilled", "target_reached"
    elif fam == "craft":
        if ctx.achievements["make_" + spec["tool"]] > spec["base"]:
            return "fulfilled", "made"
    elif fam == "place":
        for p in ctx.placements:
            if p["what"] != spec["what"]:
                continue
            if "near" in spec and not material_within(ctx.world, p["pos"], spec["near"], spec["radius"]):
                continue
            if "onto" in spec and p["onto"] != spec["onto"]:
                continue
            return "fulfilled", "placed"
    elif fam == "maintain":
        if ctx.inventory[spec["stat"]] < spec["threshold"]:
            return "failed", "violated"
        if due:
            return "fulfilled", "kept"
        return None
    elif fam == "avoid":
        hit = {"zombie": ctx.zombie_hits > 0, "arrow": ctx.arrow_hits > 0, "any": ctx.health_drop}[spec["source"]]
        if hit:
            return "failed", "violated"
        if due:
            return "fulfilled", "kept"
        return None
    elif fam == "visit":
        prog = ob.setdefault("progress", {})
        if prog.get("reached_step") is None:
            if visit_reached(ctx.world, spec, ctx.player_pos):
                prog["reached_step"] = s
        elif chebyshev(ctx.player_pos, spec["start"]) <= spec["return_radius"]:
            return "fulfilled", "returned"
    elif fam == "report":
        pass  # graded by the utterance channel
    if due:
        return "failed", "no_answer" if fam == "report" else "deadline"
    return None


# --- report questions -----------------------------------------------------------------------------------

def report_truth(ob: dict, layer_state: dict, player_pos, achievements: dict, step: int) -> dict:
    """The ground-truth answer of a report question at ``step`` (when the agent's answer is graded)."""
    spec = ob["spec"]
    q = spec["question"]
    if q == "last_seen":
        entry = layer_state["visibility"]["last_seen"].get(spec["kind"])
        if entry is None:
            return {"kind": "never", "text": "never"}
        positions = entry["positions"]
        best = min(positions, key=lambda p: (chebyshev(p, player_pos), p))
        dx, dy = best[0] - int(player_pos[0]), best[1] - int(player_pos[1])
        return {"kind": "location", "positions": positions, "seen_step": entry["step"],
                "text": f"{format_offset(dx, dy)} ({compass(dx, dy)} {distance_bucket(dx, dy)})"}
    if q == "count_today":
        day = spec["day"]
        snaps = layer_state["day_achievements"]
        start = snaps[str(day)][spec["event"]]
        if dl.day_index(step) > day and str(day + 1) in snaps:
            end = snaps[str(day + 1)][spec["event"]]
        else:
            end = achievements[spec["event"]]
        return {"kind": "int", "value": int(end - start), "text": str(int(end - start))}
    if q in ("first_request", "count_requests"):
        lo, hi = spec["window"]
        cands = [o for o in layer_state["obligations"]
                 if o["family"] != "report" and lo <= o["issue_step"] < hi]
        cands.sort(key=lambda o: (o["issue_step"], int(o["id"][1:])))
        if q == "count_requests":
            return {"kind": "int", "value": len(cands), "text": str(len(cands))}
        if not cands:
            return {"kind": "nothing", "text": "nothing"}
        first = cands[0]
        return {"kind": "request", "id": first["id"], "keywords": first["keywords"],
                "text": f"{first['id']} ({first['keywords'][0]})"}
    if q == "last_cancelled":
        cands = [o for o in layer_state["obligations"] if o["status"] == "cancelled" and o["resolved_step"] <= step]
        if not cands:
            return {"kind": "nothing", "text": "nothing"}
        last = max(cands, key=lambda o: (o["resolved_step"], int(o["id"][1:])))
        return {"kind": "request", "id": last["id"], "keywords": last["keywords"],
                "text": f"{last['id']} ({last['keywords'][0]})"}
    raise ValueError(f"unknown report question {q!r}")


def grade_report(truth: dict, payload: dict, player_pos, tolerance: int) -> bool:
    kind, ptype = truth["kind"], payload["type"]
    if kind == "never":
        return ptype == "none"
    if kind == "nothing":
        return ptype == "none"
    if kind == "int":
        return ptype == "int" and payload["value"] == truth["value"]
    if kind == "request":
        if ptype == "id":
            return payload["value"] == truth["id"]
        if ptype == "text":
            said = keyword_norm(payload["value"])
            return bool(said) and said in {keyword_norm(k) for k in truth["keywords"]}
        return False
    if kind == "location":
        px, py = int(player_pos[0]), int(player_pos[1])
        if ptype == "offset":
            stated = (px + payload["dx"], py + payload["dy"])
            return any(chebyshev(stated, p) <= tolerance for p in truth["positions"])
        if ptype == "bucket":
            for p in truth["positions"]:
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        dx, dy = p[0] + ddx - px, p[1] + ddy - py
                        if compass(dx, dy) != payload["dir"]:
                            continue
                        if payload["dist"] is None or distance_bucket(dx, dy) == payload["dist"]:
                            return True
            return False
        return False
    raise ValueError(kind)
