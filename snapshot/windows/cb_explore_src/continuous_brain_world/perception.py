"""Ground truth of what the agent could perceive: the visibility log.

The agent's pixels show Crafter's local view: a 9 x 7 grid of world tiles centred on the player (the bottom two
of the nine rows show the inventory).  A tile is **perceivable** at a step iff

1. it lies inside that 9 x 7 window and inside the 64 x 64 world, and
2. it is not drowned by night noise: when ``daylight < 0.5`` Crafter blends the frame with uniform noise using
   the per-pixel weight ``mask = 2 (0.5 - daylight) * vignette`` (``engine.LocalView._noise``), and W1 counts a
   tile as perceivable only while ``mask < 0.5`` at the tile's centre pixel -- i.e. the true image still
   dominates the noise there.  The vignette is read from Crafter's own (cached) function for the exact canvas
   shape of the primary observation, so this rule is the renderer's rule, not an approximation of it.

While the player sleeps Crafter still draws the scene (desaturated); W1 follows the pixels and keeps counting it.

The log records, for every *kind* (objects: cow, zombie, skeleton, sapling; materials: water, tree, stone,
coal, iron, diamond, lava, sand, table, furnace), the last step at which at least one instance was perceivable
and the absolute positions of all perceivable instances at that step.  Memory questions are graded only against
this log, so a cow that stood just outside the window -- or in the dark corner of a night frame -- never counts.
"""
from __future__ import annotations

import numpy as np

from crafter import objects

from . import daylight as dl

OBJECT_KINDS = {objects.Cow: "cow", objects.Zombie: "zombie", objects.Skeleton: "skeleton",
                objects.Plant: "sapling"}
MATERIAL_KINDS = ("water", "tree", "stone", "coal", "iron", "diamond", "lava", "sand", "table", "furnace")
ALL_KINDS = tuple(OBJECT_KINDS.values()) + MATERIAL_KINDS
NOISE_VISIBILITY_LIMIT = 0.5


def perceivable_grid(bridge) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions[gx, gy, 2] absolute tile coords, visible[gx, gy] bool) for the current state."""
    env = bridge.env
    gw, gh = bridge.view_grid
    unit = bridge.unit
    center = np.asarray(env._player.pos)
    offset = np.array([gw // 2, gh // 2])
    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh), indexing="ij")
    pos = np.stack([center[0] + gx - offset[0], center[1] + gy - offset[1]], -1)
    area = env._world.area
    inside = (pos[..., 0] >= 0) & (pos[..., 0] < area[0]) & (pos[..., 1] >= 0) & (pos[..., 1] < area[1])
    light = env._world.daylight
    if light < 0.5:
        vignette = env._local_view._vignette((gw * unit, gh * unit, 3), 0.5)
        centres = vignette[gx * unit + unit // 2, gy * unit + unit // 2]
        mask = 2 * (0.5 - light) * centres
        inside &= mask < NOISE_VISIBILITY_LIMIT
    return pos, inside


def perceived_now(bridge) -> dict[str, list[list[int]]]:
    """Kind -> sorted list of absolute positions perceivable in the current frame."""
    pos, vis = perceivable_grid(bridge)
    world = bridge.world
    out: dict[str, list[list[int]]] = {}
    names = world._mat_names
    vis_pos = pos[vis]
    if len(vis_pos):
        mats = world._mat_map[vis_pos[:, 0], vis_pos[:, 1]]
        for (x, y), m in zip(vis_pos.tolist(), mats.tolist()):
            name = names[m]
            if name in MATERIAL_KINDS:
                out.setdefault(name, []).append([x, y])
    visible_set = {(int(x), int(y)) for x, y in vis_pos.tolist()}
    for obj in world.objects:
        kind = OBJECT_KINDS.get(type(obj))
        if kind is None:
            continue
        p = (int(obj.pos[0]), int(obj.pos[1]))
        if p in visible_set:
            out.setdefault(kind, []).append([p[0], p[1]])
    for k in out:
        out[k].sort()
    return out


class VisibilityLog:
    """JSON-serialisable record of what was perceivable, step by step (last-seen and first-seen per kind)."""

    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {"last_seen": {}, "first_seen": {}, "now": {}}

    def update(self, bridge, step: int) -> dict:
        now = perceived_now(bridge)
        ls, fs = self.state["last_seen"], self.state["first_seen"]
        for kind, positions in now.items():
            ls[kind] = {"step": int(step), "day": dl.day_index(step), "positions": positions}
            fs.setdefault(kind, int(step))
        self.state["now"] = {k: len(v) for k, v in now.items()}
        return now

    def last_seen(self, kind: str) -> dict | None:
        return self.state["last_seen"].get(kind)

    def visible_now(self, kind: str) -> bool:
        return self.state["now"].get(kind, 0) > 0
