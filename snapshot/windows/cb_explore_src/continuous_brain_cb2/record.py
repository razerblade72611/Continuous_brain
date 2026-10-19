"""Turning a W1 life into archive rows: what the actor sensed, what it did, what followed, what a teacher knew.

``record_life`` runs one life of an ``Individual`` in a ``W1Env`` and appends every transition to a
``StreamWriter``.  The recorder only reads: the individual's store is never written, so a recorded life is the
same life as an unrecorded one (``tests/test_cb2_archive.py``: record/no-record twins).

Rows (see ``archive.SCHEMAS``):

- student: the frame and body sense the actor had when it chose, its action and utterance, the reward and
  ending that followed, who chose (SELF or TEACHER) and, on the lived stream, the individual's state hash after
  the tick;
- teacher: the routine's action and option (the actor's own when the routine acted; a COUNTERFACTUAL label when
  something else did), and the true tiles of the view;
- organ: the actor's own memory read out as belief features (with no routine-internal state) and an egocentric
  window of its remembered map.
"""
from __future__ import annotations

import json

import numpy as np

from continuous_brain_individual import codec
from continuous_brain_individual.body import read_body
from continuous_brain_individual.features import belief_features
from continuous_brain_individual import ledger
from continuous_brain_individual import memory as mem
from continuous_brain_organs.vision import labels_from_privileged

from .archive import CROP, INVENTORY, WHO

VITALS = ("health", "food", "drink", "energy")


def map_crop(memory: dict, pos) -> np.ndarray:
    m = memory["map"]
    r = CROP // 2
    out = np.full((CROP, CROP), -1, np.int8)
    x0, y0 = int(pos[0]) - r, int(pos[1]) - r
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(m.shape[0], x0 + CROP), min(m.shape[1], y0 + CROP)
    if xs1 > xs0 and ys1 > ys0:
        out[xs0 - x0:xs1 - x0, ys0 - y0:ys1 - y0] = m[xs0:xs1, ys0:ys1]
    return out


class OrganReadout:
    """The actor's memory, read out as an organ would: belief features and the remembered map around it."""

    def __init__(self):
        self.hist: list[list[int]] = []

    def reset(self):
        self.hist = []

    def __call__(self, ind) -> dict:
        store = ind.rt.store
        body = read_body(store.read("driver.body"))
        memory = codec.unblob(store.read("episodic_memory.state")) or mem.empty()
        led = codec.unblob(store.read("goal_genesis.state")) or ledger.empty()
        inv = body["inventory"]
        self.hist = (self.hist + [[int(inv[k]) for k in VITALS]])[-11:]
        phi = belief_features(memory, body, int(body["step"]), led, None, self.hist)
        return {"features": phi.astype(np.float16).tolist(),
                "map_crop": map_crop(memory, body["player_pos"]).reshape(-1).tolist()}


def student_row(obs, info, action: int, utterance, who: str, reward: dict, done: bool, life: int, seed: int,
                store_hash: str = "", dying: bool = False) -> dict:
    p = info["privileged"]
    return {"life": int(life), "step": int(obs["step"]), "seed": int(seed),
            "frame": np.ascontiguousarray(obs["image"], dtype=np.uint8).tobytes(),
            "inventory": [int(p["inventory"][k]) for k in INVENTORY],
            "pos": [int(p["player_pos"][0]), int(p["player_pos"][1])],
            "facing": [int(p["facing"][0]), int(p["facing"][1])], "sleeping": bool(p["sleeping"]),
            "messages": json.dumps(list(obs["messages"])), "action": int(action), "utterance": utterance or "",
            "who": WHO.index(who), "reward": float(reward["crafter"]), "reward_caretaker": float(reward["obligations"]), "done": bool(done), "dying": bool(dying),
            "store_hash": store_hash}


def teacher_row(info, teacher_action: int, teacher_option: int) -> dict:
    mat, obj = labels_from_privileged(info["privileged"])
    return {"teacher_action": int(teacher_action), "teacher_option": int(teacher_option),
            "view_material": mat.reshape(-1).astype(np.int8).tolist(),
            "view_object": obj.reshape(-1).astype(np.int8).tolist()}


class ShadowRoutine:
    """The routine library run beside an actor that is not the routine, on the actor's own beliefs: what the
    routine would have done (a COUNTERFACTUAL label).  It keeps its own plan state outside the individual's store
    and replans from the actor's beliefs every step, so its labels follow the actor's trajectory."""

    def __init__(self, seed: int = 0):
        from continuous_brain_individual.skills import SkillLibrary
        self.lib = SkillLibrary(seed=seed)

    def __call__(self, ind) -> tuple[int, int]:
        from continuous_brain_individual.hosts import OPTION_CODES
        from continuous_brain_individual.skills import ACTIONS
        store = ind.rt.store
        body = read_body(store.read("driver.body"))
        memory = codec.unblob(store.read("episodic_memory.state")) or mem.empty()
        led = codec.unblob(store.read("goal_genesis.state")) or ledger.empty()
        step = int(body["step"])
        name, option = self.lib.decide(memory, body, step, ledger.active(led, step))
        return ACTIONS[name], OPTION_CODES.get(option.split(":")[0], 0)


def routine_choice(ind) -> tuple[int, int]:
    """The routine's action and option this tick, as the individual's skills wrote them."""
    store = ind.rt.store
    return int(np.asarray(store.read("skills.action"))[0]), int(np.asarray(store.read("skills.selected_option"))[0])


def record_life(ind, env, seed: int, writer, steps: int = 10000, who: str = "SELF", lived: bool = True,
                teacher: bool = True, organ: bool = True, on_step=None, stop=None) -> dict:
    """One life, every transition archived.  ``ind`` acts (with its own skills); ``who`` labels whose choice that
    is.  Returns the life's summary.  A lifelong individual's last moment is ticked (``die``) but not archived as
    a transition: nothing it does then reaches the world."""
    from continuous_brain_individual.individual import first_info
    obs = env.reset(seed)
    info = first_info(env)
    readout = OrganReadout()
    shadow = ShadowRoutine(seed) if teacher and ind.config.skills != "routine" else None
    n, ret, done = 0, 0.0, False
    while n < steps and not done and not (stop is not None and stop()):
        action, utter = ind.decide(obs, info)
        orow = readout(ind) if organ else None
        trow = None
        if teacher:
            trow = teacher_row(info, *(shadow(ind) if shadow is not None else routine_choice(ind)))
        obs2, reward, done, info2 = env.step(action, utter)
        writer.append(student_row(obs, info, action, utter, who, reward, done, ind.life, seed,
                                  ind.state_hash() if lived else ""), trow, orow)
        ret += float(reward["total"])
        n += 1
        if on_step is not None:
            on_step(n, action, utter, obs2, reward, info2)
        obs, info = obs2, info2
    if done and ind.config.lifelong:
        ind.die(obs, info)
    return {"steps": n, "done": bool(done), "reward": ret,
            "achievements": sum(1 for v in info["privileged"]["achievements"].values() if v)}
