"""Episodic memory: what the individual has seen of its world, and what it has done, kept in the store.

State (a plain dict, encoded by ``codec``)::

    {"map":      int8[64, 64]   believed material class per world tile (vision MATERIALS index), -1 = never seen
     "seen":     int32[64, 64]  step at which the tile was last perceived confidently, -1 = never
     "last_seen": {kind: {"step": s, "positions": [[x, y], ...]}}   the last step each kind was perceived, and
                                                                     every position it was perceived at then
     "day_snap": {day (str): {achievement: count}}                   own counters at the first step of each day
     "objects":  [{"kind", "pos": [x, y], "ripe", "step"}]           objects perceived at the latest step
     "remembered": [{"kind", "pos": [x, y], "ripe", "step"}]         cows and plants out of view, as last seen
     "blocked":  {"x,y": step}       tiles it bumped into (a move that did not move it), for BUMP_MEMORY steps
     "last":     {"pos": [x, y], "action": a}                         its previous position and action
     "steps": int}                                                   steps lived

Learning from bumping: if the previous action was a move and the body did not move (and it was not asleep), the
tile in that direction is not walkable now; for ``BUMP_MEMORY`` steps something is believed to stand there (most
often an animal that then moves on, sometimes a wall it misperceived).  Tiles the vision organ is unsure of are written into the map only where the map knows
nothing better (a best guess beats "unknown"); they never count as *seen* for memory answers.

Remembered objects: a cow is remembered where it was last seen for ``COW_MEMORY`` steps, a plant (sapling,
ripe or not) until its tile is seen again without it.  Whenever a remembered position is back in view, the
current view decides: the object is there (and is refreshed) or it is forgotten.

Perception enters memory only where the vision organ is confident (``CONF`` and above): a tile drowned in night
noise is not remembered as seen, which mirrors what an individual could honestly claim to have seen.  With
``mode="none"`` (an ablation) the map and the last-seen log keep only the current view.
"""
from __future__ import annotations

import numpy as np

from continuous_brain_organs.vision import MATERIALS, OBJECTS
from continuous_brain_world import daylight as dl

W = H = 64
ROWS, COLS, PR, PC = 7, 9, 3, 4          # the perceived view and the player's cell in it (vision organ geometry)
CONF = 0.9
COW_MEMORY = 300
BUMP_MEMORY = 20
GUESS_UNSURE = False
from crafter import constants as _cc
MOVES = {_cc.actions.index(n): d for n, d in (("move_left", (-1, 0)), ("move_right", (1, 0)),
                                             ("move_up", (0, -1)), ("move_down", (0, 1)))}
REMEMBERED_KINDS = ("cow", "sapling")
MATERIAL_KINDS = ("water", "tree", "coal", "iron", "diamond", "lava", "table", "furnace")
OBJECT_KINDS = ("cow", "zombie", "skeleton", "sapling")
MAT = {n: i for i, n in enumerate(MATERIALS)}
OBJ = {n: i for i, n in enumerate(OBJECTS)}


def empty() -> dict:
    return {"map": np.full((W, H), -1, np.int8), "seen": np.full((W, H), -1, np.int32), "last_seen": {},
            "day_snap": {}, "objects": [], "remembered": [], "blocked": {}, "last": None, "steps": 0}


def view_cells(pos):
    """World coordinates (x, y) of every perceived cell [r, c] (rows north->south, cols west->east)."""
    px, py = int(pos[0]), int(pos[1])
    for r in range(ROWS):
        for c in range(COLS):
            yield r, c, px + c - PC, py + r - PR


def integrate(state: dict | None, tiles: dict, body: dict, step: int, mode: str = "full",
              last_action: int | None = None) -> dict:
    """Return the new memory state after perceiving ``tiles`` (vision organ output) with ``body``."""
    st = empty() if state is None else state
    if mode == "none":                                  # ablation: no memory beyond the current view
        st = {**empty(), "day_snap": st["day_snap"], "steps": st["steps"], "last": st.get("last")}
    else:
        st = {**st, "map": st["map"].copy(), "seen": st["seen"].copy(), "last_seen": dict(st["last_seen"]),
              "day_snap": dict(st["day_snap"])}
    mat, obj = np.asarray(tiles["material"]), np.asarray(tiles["object"])
    mconf, oconf = np.asarray(tiles["material_conf"]), np.asarray(tiles["object_conf"])
    seen_now: dict[str, list] = {}
    objects = []
    blocked = {k: v for k, v in st.get("blocked", {}).items() if step - v <= BUMP_MEMORY}
    last = st.get("last")
    if last is not None and last_action in MOVES and not body["sleeping"] \
            and list(last["pos"]) == list(body["player_pos"]):
        d = MOVES[last_action]
        bx, by = body["player_pos"][0] + d[0], body["player_pos"][1] + d[1]
        if 0 <= bx < W and 0 <= by < H:
            blocked[f"{bx},{by}"] = step
    for r, c, x, y in view_cells(body["player_pos"]):
        if not (0 <= x < W and 0 <= y < H):
            continue
        if GUESS_UNSURE and mconf[r, c] < CONF and st["map"][x, y] < 0:
            st["map"][x, y] = int(mat[r, c])            # a best guess, not a sighting (seen stays -1)
        if mconf[r, c] >= CONF:
            m = int(mat[r, c])
            st["map"][x, y] = m
            st["seen"][x, y] = step
            name = MATERIALS[m]
            if name in MATERIAL_KINDS:
                seen_now.setdefault(name, []).append([x, y])
        if oconf[r, c] >= CONF:
            o = OBJECTS[int(obj[r, c])]
            if o in ("none", "player"):
                continue
            kind = "sapling" if o == "plant_ripe" else o
            objects.append({"kind": kind, "pos": [x, y], "ripe": o == "plant_ripe", "step": step})
            if kind in OBJECT_KINDS:
                seen_now.setdefault(kind, []).append([x, y])
    for kind, positions in seen_now.items():
        st["last_seen"][kind] = {"step": step, "positions": sorted(positions)}
    st["objects"] = objects
    in_view = {(x, y) for _, _, x, y in view_cells(body["player_pos"])}
    kept = [o for o in st.get("remembered", [])
            if (o["pos"][0], o["pos"][1]) not in in_view
            and (o["kind"] != "cow" or step - o["step"] <= COW_MEMORY)]
    if mode == "none":
        kept = []
    st["remembered"] = kept + [dict(o) for o in objects if o["kind"] in REMEMBERED_KINDS]
    day = str(dl.day_index(step))
    if day not in st["day_snap"]:
        st["day_snap"][day] = dict(body["achievements"])
    st["blocked"] = blocked if mode != "none" else {}
    st["last"] = {"pos": list(body["player_pos"]), "action": None if last_action is None else int(last_action)}
    st["steps"] = st["steps"] + 1
    return st


def counts_today(state: dict, body: dict, step: int) -> dict:
    """Own achievement counts since the first step of today (from the individual's own day snapshots)."""
    snap = state["day_snap"].get(str(dl.day_index(step)))
    if snap is None:
        return {}
    return {k: int(body["achievements"].get(k, 0) - snap.get(k, 0)) for k in body["achievements"]}


def believed_objects(state: dict) -> list[dict]:
    """Objects the individual believes are where it thinks: the current view, plus remembered cows and plants
    out of view (the current view always wins)."""
    now = {(o["pos"][0], o["pos"][1]) for o in state["objects"]}
    out = [dict(o) for o in state["objects"]]
    out += [dict(o) for o in state.get("remembered", []) if (o["pos"][0], o["pos"][1]) not in now]
    return out


def known_fraction(state: dict) -> float:
    return float((state["map"] >= 0).mean())


def summary_vector(state: dict) -> np.ndarray:
    """episodic_memory.slots: [fraction of the world known, kinds ever seen, objects in view, steps lived]."""
    return np.array([known_fraction(state), len(state["last_seen"]), len(state["objects"]), state["steps"]],
                    dtype=np.float64)
