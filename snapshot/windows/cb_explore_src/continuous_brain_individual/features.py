"""What the individual's learned faculties see: a fixed vector of features built only from its own beliefs.

Inputs are the individual's memory (remembered map, objects in view, remembered cows and plants), its body sense,
its commitment ledger, a few things its skills remember (its shelter, places that hurt it), and a short history of
its own vitals.  Nothing privileged.  Distances are city-block distances on its believed map, scaled so that 1 is
"here" and 0 is "unknown or 32+ tiles away".
"""
from __future__ import annotations

import numpy as np

from continuous_brain_organs.vision import MATERIALS
from continuous_brain_world import daylight as dl

from . import ledger, memory as mem

MAT = {n: i for i, n in enumerate(MATERIALS)}
NEAR_KINDS = ("water", "tree", "stone", "coal", "iron", "table", "furnace")
ITEMS = ("wood", "stone", "sapling", "coal", "iron", "diamond")
TOOLS = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
FAR = 32.0

NAMES = (["health", "food", "drink", "energy", "asleep", "day sin", "day cos", "dark", "until nightfall",
          "until dawn"]
         + [f"{k} {w}" for k in NEAR_KINDS for w in ("known", "near")]
         + ["lava close", "cow known", "cow near", "plant known", "plant near", "zombies in view", "zombie near",
            "skeletons in view", "skeleton near", "arrow in view", "tunnels near", "hurt-place near", "in shelter"]
         + [f"has {i}" for i in ITEMS] + [f"has {t}" for t in TOOLS]
         + ["world known", "recent damage", "drink trend", "food trend"]
         + [f"open {f}" for f in ledger.TASK_FAMILIES] + ["deadline pressure", "question pending",
                                                           "things done"])
DIM = len(NAMES)


def _near(dist: float | None) -> float:
    return 0.0 if dist is None else max(0.0, 1.0 - dist / FAR)


def _nearest(pos, cells) -> float | None:
    if len(cells) == 0:
        return None
    c = np.asarray(cells)
    return float(np.min(np.abs(c[:, 0] - pos[0]) + np.abs(c[:, 1] - pos[1])))


def belief_features(memory: dict, body: dict, step: int, led: dict, skills_state: dict | None,
                    vitals_history: list[list[int]]) -> np.ndarray:
    inv = body["inventory"]
    pos = (int(body["player_pos"][0]), int(body["player_pos"][1]))
    t = (step % dl.DAY_LENGTH) / dl.DAY_LENGTH
    dark = dl.is_dark(step)
    f = [inv["health"] / 9, inv["food"] / 9, inv["drink"] / 9, inv["energy"] / 9, float(body["sleeping"]),
         np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), float(dark),
         0.0 if dark else min(1.0, (dl.next_nightfall(step) - step) / dl.DAY_LENGTH),
         min(1.0, (dl.next_dawn(step) - step) / dl.DAY_LENGTH) if dark else 0.0]
    m = memory["map"]
    for k in NEAR_KINDS:
        xs, ys = np.nonzero(m == MAT[k])
        d = _nearest(pos, np.stack([xs, ys], 1)) if len(xs) else None
        f += [float(d is not None), _near(d)]
    xs, ys = np.nonzero(m == MAT["lava"])
    dl_ = _nearest(pos, np.stack([xs, ys], 1)) if len(xs) else None
    f.append(float(dl_ is not None and dl_ <= 3))
    objs = mem.believed_objects(memory)

    def kind(*names):
        return [o["pos"] for o in objs if o["kind"] in names]
    cows, plants = kind("cow"), kind("sapling")
    zombies, skeletons, arrows = kind("zombie"), kind("skeleton"), kind("arrow")
    f += [float(bool(cows)), _near(_nearest(pos, cows)), float(bool(plants)), _near(_nearest(pos, plants)),
          min(1.0, len(zombies) / 3), max(0.0, 1 - (_nearest(pos, zombies) or 99) / 9),
          min(1.0, len(skeletons) / 2), max(0.0, 1 - (_nearest(pos, skeletons) or 99) / 9), float(bool(arrows))]
    path = m == MAT["path"]
    r = 12
    f.append(float(path[max(0, pos[0] - r):pos[0] + r + 1, max(0, pos[1] - r):pos[1] + r + 1].any()))
    sk = skills_state or {}
    bad = [b[:2] for b in sk.get("bad_spots", [])]
    f.append(float(bool(bad) and (_nearest(pos, bad) or 99) <= 8))
    f.append(float(sk.get("shelter") is not None))
    f += [min(1.0, inv[i] / 9) for i in ITEMS] + [float(inv[t_] > 0) for t_ in TOOLS]
    f.append(float((m >= 0).mean()))
    hist = vitals_history[-10:] if vitals_history else [[inv["health"], inv["food"], inv["drink"], inv["energy"]]]
    h0 = hist[0]
    drops = sum(max(0, a[0] - b[0]) for a, b in zip(hist, hist[1:]))
    f += [min(1.0, drops / 5), float(np.clip((inv["drink"] - h0[2]) / 3, -1, 1)),
          float(np.clip((inv["food"] - h0[1]) / 3, -1, 1))]
    act = ledger.active(led, step)
    for fam in ledger.TASK_FAMILIES:
        f.append(min(1.0, sum(1 for c in act if c["family"] == fam) / 3))
    slack = [c["deadline"] - step for c in act]
    f.append(0.0 if not slack else float(np.clip(1 - min(slack) / dl.DAY_LENGTH, 0, 1)))
    f.append(float(bool(ledger.pending_questions(led))))
    f.append(sum(1 for v in body["achievements"].values() if v) / 22)
    out = np.asarray(f, dtype=np.float32)
    assert out.shape == (DIM,), (out.shape, DIM)
    return out
