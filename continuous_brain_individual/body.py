"""Body sense: the only part of W1's privileged ``info`` the individual may read.

An animal knows its own body without looking: how hungry, thirsty, tired and hurt it is, what it is carrying,
where its body is and which way it faces, whether it is asleep, and what it has done (its own achievement
counters).  ``body_sense`` copies exactly those fields and nothing else; the world map, the objects, the
obligation specs, the visibility log and the request history stay hidden.  ``GuardedInfo`` turns any other read
into an error, so the separation is enforced, not promised.
"""
from __future__ import annotations

import json

from continuous_brain_world import daylight as dl

BODY_KEYS = ("inventory", "player_pos", "facing", "sleeping", "achievements")
VITALS = ("health", "food", "drink", "energy")


class PrivilegedRead(KeyError):
    pass


class GuardedInfo:
    """Wraps W1's ``info``; only ``info["privileged"][k]`` for k in BODY_KEYS can be read."""

    def __init__(self, info: dict):
        self._priv = info["privileged"]

    def body(self) -> dict:
        return body_sense({"privileged": self._priv})

    def __getitem__(self, key):
        raise PrivilegedRead(f"the individual may not read info[{key!r}]; use body()")


def body_sense(info: dict) -> dict:
    priv = info["privileged"]
    return {"inventory": {k: int(v) for k, v in priv["inventory"].items()},
            "player_pos": [int(priv["player_pos"][0]), int(priv["player_pos"][1])],
            "facing": [int(priv["facing"][0]), int(priv["facing"][1])],
            "sleeping": bool(priv["sleeping"]),
            "achievements": {k: int(v) for k, v in priv["achievements"].items()}}


def body_message(body: dict, step: int) -> bytes:
    """The driver.body blob: the body sense plus the step (the individual's clock)."""
    return json.dumps({"step": int(step), **body, "time": dl.describe(int(step))}, sort_keys=True,
                      separators=(",", ":")).encode()


def read_body(arr) -> dict:
    import numpy as np
    raw = np.asarray(arr, dtype=np.uint8).tobytes()
    return json.loads(raw.decode()) if raw else {}
