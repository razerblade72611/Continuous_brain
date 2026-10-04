"""W1Env: Crafter plus the obligations layer, behind a Gym-like API.

    env = W1Env(W1Config())
    obs = env.reset(seed)
    obs, reward, done, info = env.step(action, utterance=None)

``obs`` (what the agent may use)::

    {"image": uint8[obs_size, obs_size, 3],          # Crafter's pixels (local view + inventory)
     "image_large": uint8[L, L, 3],                  # only if config.large_render_size is set
     "messages": [str, ...],                         # caretaker messages issued at this step
     "step": int, "day": int, "phase": "morning" | "afternoon" | "night",
     "obligations": [...]}                           # only in interface="structured" (baseline mode)

``reward`` is a dict ``{"crafter": r_c, "obligations": r_o, "total": r_c + r_o}``: Crafter's own reward
(health change / 10 + 1 per first unlock of an achievement) and the layer's (+value when an obligation is
fulfilled, -fail_penalty * value when one fails, 0 for a cancellation).

``info`` is **privileged ground truth for evaluation and for the privileged baseline only**::

    {"privileged": {inventory, achievements, player_pos, facing, sleeping, health, daylight, material_map,
                    material_ids, objects, semantic?, obligations (active, full specs), history (all,
                    compact), visibility (last-seen log), day_achievements},
     "events": {placements, zombie_hits, arrow_hits, health_drop, fulfilled, failed, issued, amended,
                cancelled, answers},
     "crafter_reward": float}

Determinism: given ``(seed, config)`` and the same sequence of ``(action, utterance)`` the observations,
messages, rewards and obligation states are identical (Crafter's despawn-order fix: ``crafter_bridge``).
Snapshots: ``snapshot()`` returns ``{"format", "config", "crafter": bytes, "layer": json-dict}``;
``restore(snapshot)`` continues bit-exactly.  ``snapshot_to_json``/``snapshot_from_json`` wrap it as text.
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import daylight as dl
from .crafter_bridge import CrafterBridge, crafter_provenance, object_kind
from .layer import ObligationLayer
from .scoring import episode_summary

SNAPSHOT_FORMAT = "w1-snapshot/1"


@dataclass
class W1Config:
    episode_length: int = 10000
    obs_size: int = 64
    large_render_size: int | None = None
    interface: str = "text"                 # "text" | "structured"
    max_concurrent: int = 5
    first_request_step: int = 10
    arrival_rate0: float = 1 / 110
    arrival_growth: float = 0.15            # per day
    arrival_rate_max: float = 1 / 35
    difficulty0: float = 0.15
    difficulty_growth: float = 0.1          # per day
    change_rate: float = 1 / 900            # per step, per changeable obligation
    cancel_fraction: float = 0.35
    report_window: int = 120
    location_tolerance: int = 2
    fail_penalty: float = 0.5
    feedback: bool = True
    reveal_answers: bool = True
    privileged_semantic: bool = True
    log_keep: int = 200
    family_weights: dict = field(default_factory=lambda: {
        "collect": 0.20, "craft": 0.12, "place": 0.13, "maintain": 0.09, "avoid": 0.08, "visit": 0.10,
        "report": 0.28})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "W1Config":
        return cls(**d)

    def validate(self):
        if self.interface not in ("text", "structured"):
            raise ValueError("interface must be 'text' or 'structured'")
        if self.obs_size < 9:
            raise ValueError("obs_size too small")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        from .obligations import FAMILIES
        unknown = set(self.family_weights) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown families {unknown}")


class W1Env:
    def __init__(self, config: W1Config | None = None):
        self.config = config or W1Config()
        self.config.validate()
        self.bridge = CrafterBridge(self.config.obs_size, self.config.episode_length)
        self.layer: ObligationLayer | None = None
        self.seed: int | None = None
        self.done = True
        self.timing = {"crafter_s": 0.0, "layer_s": 0.0, "steps": 0}

    # -- API ----------------------------------------------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self.seed = int(seed)
        image = self.bridge.reset(self.seed)
        self.layer = ObligationLayer(self.config, self.seed)
        msgs = self.layer.start(self.bridge)
        self.done = False
        self._last_info = None
        return self._obs(image, msgs)

    def step(self, action: int, utterance: str | None = None):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if not 0 <= int(action) < 17:
            raise ValueError("action must be in [0, 17)")
        t0 = time.perf_counter()
        hear_msgs, answer_events = self.layer.hear(utterance, self.bridge)
        pre = self.layer.capture(self.bridge)
        t1 = time.perf_counter()
        image, r_crafter, done = self.bridge.step(int(action))
        t2 = time.perf_counter()
        msgs, r_obl, events = self.layer.advance(self.bridge, pre, r_crafter)
        r_obl += sum(self.layer._reward(self.layer._index[e["id"]]) for e in answer_events
                     if e["event"] == "answered")
        events["answers"] = answer_events
        for e in answer_events:
            if e["event"] == "answered":
                events["fulfilled" if e["correct"] else "failed"].append(e["id"])
        self.done = bool(done)
        self.layer.state["done"] = self.done
        obs = self._obs(image, hear_msgs + msgs)
        info = self._info(events, r_crafter)
        t3 = time.perf_counter()
        self.timing["crafter_s"] += t2 - t1
        self.timing["layer_s"] += (t1 - t0) + (t3 - t2)
        self.timing["steps"] += 1
        reward = {"crafter": float(r_crafter), "obligations": float(r_obl), "total": float(r_crafter + r_obl)}
        return obs, reward, self.done, info

    # -- observation / info ---------------------------------------------------------------------------------
    def _obs(self, image, msgs) -> dict:
        s = self.bridge.step_count
        obs = {"image": image, "messages": list(msgs), "step": s, "day": dl.day_index(s), "phase": dl.phase(s)}
        if self.config.large_render_size:
            obs["image_large"] = self.bridge.render(int(self.config.large_render_size))
        if self.config.interface == "structured":
            obs["obligations"] = [self._public(o) for o in self.layer.active()]
        return obs

    @staticmethod
    def _public(o: dict) -> dict:
        return {"id": o["id"], "family": o["family"], "text": o["text"], "spec": copy.deepcopy(o["spec"]),
                "issue_step": o["issue_step"], "deadline": o["deadline"], "value": o["value"],
                "status": o["status"], "progress": copy.deepcopy(o.get("progress"))}

    def _info(self, events, r_crafter) -> dict:
        b, st = self.bridge, self.layer.state
        p = b.player
        priv = {
            "inventory": dict(p.inventory), "achievements": dict(p.achievements),
            "player_pos": (int(p.pos[0]), int(p.pos[1])), "facing": (int(p.facing[0]), int(p.facing[1])),
            "sleeping": bool(p.sleeping), "health": int(p.health), "daylight": float(b.world.daylight),
            "material_map": b.world._mat_map.copy(), "material_ids": dict(b.world._mat_ids),
            "objects": [{"kind": object_kind(o), "pos": (int(o.pos[0]), int(o.pos[1])), "health": int(o.health),
                         "ripe": bool(getattr(o, "ripe", False))} for o in b.world.objects],
            "obligations": [self._public(o) for o in self.layer.active()],
            "history": [{"id": o["id"], "family": o["family"], "issue_step": o["issue_step"], "status": o["status"],
                         "resolved_step": o["resolved_step"], "keywords": list(o["keywords"])}
                        for o in st["obligations"]],
            "visibility": copy.deepcopy(st["visibility"]["last_seen"]),
            "day_achievements": copy.deepcopy(st["day_achievements"]),
        }
        if self.config.privileged_semantic:
            priv["semantic"] = b.semantic()
        return {"privileged": priv, "events": events, "crafter_reward": float(r_crafter)}

    def privileged_now(self) -> dict:
        """The privileged info for the current state (e.g. right after reset)."""
        return self._info({}, 0.0)

    # -- persistence ------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        if self.layer is None:
            raise RuntimeError("nothing to snapshot before reset()")
        return {"format": SNAPSHOT_FORMAT, "config": self.config.to_dict(), "seed": self.seed,
                "done": self.done, "crafter": self.bridge.snapshot_bytes(), "layer": self.layer.export_state(),
                "provenance": crafter_provenance()}

    def restore(self, snap: dict) -> None:
        if snap.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("not a W1 snapshot")
        if snap["config"] != self.config.to_dict():
            raise ValueError("snapshot was taken under a different W1Config")
        self.bridge.restore_bytes(snap["crafter"])
        self.layer = ObligationLayer(self.config, snap["seed"], json.loads(json.dumps(snap["layer"])))
        self.seed = snap["seed"]
        self.done = bool(snap["done"])

    @staticmethod
    def snapshot_to_json(snap: dict) -> str:
        d = dict(snap)
        d["crafter"] = base64.b64encode(snap["crafter"]).decode("ascii")
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def snapshot_from_json(text: str) -> dict:
        d = json.loads(text)
        d["crafter"] = base64.b64decode(d["crafter"])
        return d

    # -- evaluation -------------------------------------------------------------------------------------
    def episode_summary(self) -> dict:
        return episode_summary(self.layer.state, dict(self.bridge.player.achievements), self.config.episode_length)

    def layer_state(self) -> dict:
        return self.layer.export_state()


def action_names() -> list[str]:
    from crafter import constants
    return list(constants.actions)
