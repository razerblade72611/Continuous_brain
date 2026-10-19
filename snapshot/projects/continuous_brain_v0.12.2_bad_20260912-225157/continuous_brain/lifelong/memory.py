"""Sensor-keyed episodic retrieval and descriptive behavioral history."""
from __future__ import annotations

import copy

import numpy as np

from .world import MEMORY_DIM


class ExperienceMemory:
    def __init__(self, state=None):
        state = state or {}
        self.recent = copy.deepcopy(state.get("recent", []))
        self.formative = copy.deepcopy(state.get("formative", []))

    def retrieve(self, sensors):
        rows = self.recent + self.formative
        if not rows:
            return np.zeros(MEMORY_DIM, np.float32)
        key = np.asarray(sensors[:7], np.float32)
        keys = np.asarray([r["key"] for r in rows], np.float32)
        distances = np.square(keys - key).sum(-1)
        nearest = np.argsort(distances, kind="stable")[:3]
        weights = 1. / (distances[nearest] + .05)
        values = np.asarray([rows[int(i)]["value"] for i in nearest], np.float32)
        return np.asarray((values * weights[:, None]).sum(0) / weights.sum(), np.float32)

    def add(self, sensors, action, reward, info):
        value = [0.] * MEMORY_DIM
        value[action] = 1.
        value[5:] = [float(np.clip(reward, -1, 1)), float(info["resource"] is not None),
                     float(info["collision"])]
        row = {"key": np.asarray(sensors[:7]).tolist(), "value": value}
        self.recent.append(row)
        self.recent = self.recent[-256:]
        if info["resource"] is not None:
            self.formative.append(row)
            self.formative = self.formative[-64:]

    def state_dict(self):
        return {"recent": self.recent, "formative": self.formative}


class BehaviorHistory:
    def __init__(self, state=None):
        self.counts = dict(state or {"steps": 0, "collisions": 0, "rest": 0,
                                   "food": 0, "water": 0, "discoveries": 0})

    def observe(self, action, info):
        c = self.counts
        c["steps"] += 1
        c["collisions"] += int(info["collision"])
        c["rest"] += int(action == 0)
        c["discoveries"] += int(info["new_place"])
        if info["resource"]:
            c[info["resource"]] += 1

    def vector(self):
        c = self.counts
        n = max(1, c["steps"])
        resources = max(1, c["food"] + c["water"])
        return np.array([c["collisions"] / n, c["rest"] / n,
                         (c["food"] - c["water"]) / resources,
                         c["discoveries"] / n], np.float32)

    def describe(self):
        v = self.vector()
        return dict(zip(("collision_frequency", "rest_frequency", "food_water_balance",
                         "new_place_frequency"), [float(x) for x in v]))
