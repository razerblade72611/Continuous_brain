"""Consolidation: the one recorded event through which what was learned in sleep changes the individual.

``install(src, dst, weights, lineage)`` makes ``dst`` the individual ``src`` was, with a new skill faculty:

- every channel outside the ``skills`` slot is carried over exactly (memory, commitments, self-model, diary,
  speech, perception), with the clock and the life number;
- ``skills.weights`` holds the installed core and policy, ``skills.state`` its lineage (every event so far);
- if ``dst`` binds the slot differently from ``src`` (the routine library replaced by the learned skill), the
  event is a *transplant*; otherwise it is an *install* that replaces only the weights.

The event records the state hash before and after, the weights' SHA-256, and the lineage it was given (archive
shards and their manifest hashes, training code, configs, seeds, validation).  ``ConsolidationLog`` keeps events
append-only and hash-chained beside the individual; ``verify`` recomputes the chain.  Rolling back is restoring
the snapshot whose hash the event names, and is itself an event.  Training batches never enter its memory or its
diary: only this event does.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from continuous_brain_individual import codec

from .skill_host import weights_sha

SLOT = "skills"


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def install(src, dst, weights: bytes, lineage: dict) -> dict:
    """Carry ``src`` into ``dst`` with ``weights`` installed in its skill slot.  ``dst`` must be freshly built
    (never ticked).  Returns the event (also appended to the lineage inside ``skills.state``)."""
    old = src.rt.state_dict()
    new = dst.rt.state_dict()
    if new["store"]["current"]:
        raise ValueError("install into a freshly built individual")
    kind = "install" if old["identity"]["binding"] == new["identity"]["binding"] else "transplant"
    dst_channels = set(new["store"]["current"]) | set(dst.rt.store.names())
    own = {n for n in dst.rt.store.names() if n.startswith(SLOT + ".")}
    merged_store = {"tick": old["store"]["tick"], "written_this_tick": [], "current": {}, "previous": {}}
    for part in ("current", "previous"):
        for k, v in old["store"][part].items():
            if k in dst_channels and k not in own:
                merged_store[part][k] = v
    prior = codec.unblob(old["store"]["current"].get("skills.state", np.zeros(0, np.uint8))) or {}
    history = list(prior.get("lineage", [])) if kind == "install" else []
    event = {"format": "cb2-consolidation/1", "kind": kind, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "tick": int(old["tick"]), "life": int(src.life), "parent_state_hash": src.state_hash(),
             "weights_sha256": weights_sha(np.frombuffer(weights, np.uint8)), "weights_bytes": len(weights),
             "from_binding": [f"{b['slot']}:{b['component']}" for b in old["identity"]["binding"] if b["slot"] == SLOT],
             "to_binding": [f"{b['slot']}:{b['component']}" for b in new["identity"]["binding"] if b["slot"] == SLOT],
             "lineage": lineage, "prev_event_sha": history[-1]["event_sha"] if history else None}
    event["event_sha"] = hashlib.sha256(_canon(event)).hexdigest()
    st = {"lineage": history + [event]}
    if kind == "install":
        st.update({k: v for k, v in prior.items() if k not in ("lineage", "t", "prev_action", "hist")})
    w = np.frombuffer(weights, np.uint8).copy()
    for part in ("current", "previous"):
        merged_store[part]["skills.weights"] = w
        merged_store[part]["skills.state"] = codec.blob(st)
        merged_store[part]["skills.core_state"] = np.zeros(0, np.uint8)
    state = copy.copy(new)
    state["store"] = merged_store
    state["tick"] = old["tick"]
    state["interventions"] = old["interventions"] if kind == "install" else new["interventions"]
    state["coprocessor"] = old["coprocessor"]
    state["components"] = {s: old["components"].get(s, c) if s != SLOT else c for s, c in new["components"].items()}
    dst.rt.load_state_dict(state)
    dst.life = src.life
    event["state_hash"] = dst.state_hash()
    return event


class ConsolidationLog:
    """Append-only, hash-chained record of every consolidation of one individual (``consolidations.jsonl``)."""

    def __init__(self, path):
        self.path = Path(path)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def append(self, event: dict) -> dict:
        prev = self.events()
        rec = {"seq": len(prev), "prev_record_sha": prev[-1]["record_sha"] if prev else None, "event": event}
        rec["record_sha"] = hashlib.sha256(_canon(rec)).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return rec

    def verify(self) -> int:
        prev = None
        for i, rec in enumerate(self.events()):
            body = {k: v for k, v in rec.items() if k != "record_sha"}
            if rec["seq"] != i or rec["prev_record_sha"] != prev or \
                    hashlib.sha256(_canon(body)).hexdigest() != rec["record_sha"]:
                raise ValueError(f"consolidation log broken at record {i}")
            prev = rec["record_sha"]
        return len(self.events())
