"""The routine's intent at a step: which of its routines produced the action, and where it was walking.

DIAGNOSTIC ONLY (r3-T).  This is privileged information about the teacher's private plan, never part of the
student's legal input contract.  It exists to answer one question: is the learned skill's closed-loop collapse
caused by not knowing *what the routine is trying to do* (hidden intent), or by not being able to *do it*?

``IntentLibrary`` is the individual's own skill library (``continuous_brain_individual.skills.SkillLibrary``)
with its routine entry points and its one movement primitive watched.  ``decide_with_intent`` returns exactly what
``decide`` returns (the library's behaviour is unchanged; ``tests/test_cb2_intent.py`` checks it on whole lives)
plus a vector of ``INTENT_DIM`` numbers:

- ``BRANCHES`` one-hot: the outermost routine that produced this step's action ("inline" when ``_decide``
  answered itself: fighting an adjacent zombie, sleeping where it stands);
- the goal of the last path step taken inside that routine: has-goal, dx/32, dy/32 (clipped to +-1) and its path
  cost/64 (clipped to 1).  Actions that are not walks (do, place, craft, turning to face) have no goal.
"""
from __future__ import annotations

import numpy as np

from continuous_brain_individual.skills import SkillLibrary

BRANCHES = ("inline", "flee", "drink", "eat", "enter_shelter", "shelter_step", "obligations", "acquire",
            "tech_tree", "wander", "explore", "sleeping")
GOAL_DIM = 4
INTENT_DIM = len(BRANCHES) + GOAL_DIM
MASKS = ("none", "branch", "full")          # which part of the intent a policy is given (r3-T arms)
_TRACKED = ("_flee", "_drink", "_eat", "_enter_shelter", "_shelter_step", "_obligations", "_acquire",
            "_tech_tree", "_wander", "_explore")


def mask_vector(mask: str) -> np.ndarray:
    m = np.zeros(INTENT_DIM, np.float32)
    if mask in ("branch", "full"):
        m[:len(BRANCHES)] = 1
    if mask == "full":
        m[len(BRANCHES):] = 1
    if mask not in MASKS:
        raise ValueError(mask)
    return m


def _tracked(name: str):
    base = getattr(SkillLibrary, name)

    def wrapper(self, *args, **kw):
        outer = self._i_depth == 0
        if outer:
            self._i_cur_goal = None
        self._i_depth += 1
        try:
            r = base(self, *args, **kw)
        finally:
            self._i_depth -= 1
        if outer and r:
            self._i_branch, self._i_goal = name.lstrip("_"), self._i_cur_goal
        return r
    wrapper.__name__ = name
    return wrapper


class IntentLibrary(SkillLibrary):
    _i_depth = 0
    _i_cur_goal = None
    _i_branch = None
    _i_goal = None

    def _step_toward(self, v, goal):
        r = super()._step_toward(v, goal)
        if r and self._i_depth:
            self._i_cur_goal = (int(goal[0]), int(goal[1]))
        return r

    def decide_with_intent(self, memory: dict, body: dict, step: int, commitments: list[dict]):
        """(action name, option, intent vector float32[INTENT_DIM])."""
        self._i_depth, self._i_branch, self._i_goal, self._i_cur_goal = 0, None, None, None
        name, option = self.decide(memory, body, step, commitments)
        vec = np.zeros(INTENT_DIM, np.float32)
        branch = "sleeping" if option == "sleeping" else (self._i_branch or "inline")
        vec[BRANCHES.index(branch)] = 1
        g = self._i_goal if branch != "sleeping" else None
        if g is not None:
            px, py = (int(c) for c in body["player_pos"])
            dist = getattr(self, "dist", None)
            cost = float(dist[g]) if dist is not None else 0.0
            vec[len(BRANCHES):] = (1.0, np.clip((g[0] - px) / 32, -1, 1), np.clip((g[1] - py) / 32, -1, 1),
                                   min(cost, 64.0) / 64.0)
        return name, option, vec


for _n in _TRACKED:
    setattr(IntentLibrary, _n, _tracked(_n))


class IntentShadow:
    """The routine library run beside an actor on the actor's own beliefs, as the routine host runs it (seed 0,
    fresh plans each life), reporting its intent.  Its plan state can live in a store (``state``)."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, memory: dict, body: dict, step: int, commitments: list[dict]):
        lib = IntentLibrary.from_state(self.state, seed=self.seed)
        out = lib.decide_with_intent(memory, body, step, commitments)
        self.state = lib.to_state()
        return out

    @staticmethod
    def read(ind):
        """The actor's beliefs as the routine host reads them: (memory, body, step, active commitments)."""
        from continuous_brain_individual import codec, ledger
        from continuous_brain_individual import memory as mem
        from continuous_brain_individual.body import read_body
        store = ind.rt.store
        body = read_body(store.read("driver.body"))
        memory = codec.unblob(store.read("episodic_memory.state")) or mem.empty()
        led = codec.unblob(store.read("goal_genesis.state")) or ledger.empty()
        step = int(body["step"])
        return memory, body, step, ledger.active(led, step)
