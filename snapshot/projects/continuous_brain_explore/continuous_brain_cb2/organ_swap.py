"""Organ swap (Gate 7, DAV-90): replace an organ, keep the same individual.

An organ (the vision organ's tile decoder, later the language organ's model) is part of the runtime's identity
*binding*: the perception host's provenance names the organ's weights.  So a snapshot saved with one organ refuses to
restore into a runtime built with another (``UnifiedRuntime.load_state_dict`` compares bindings).  The only way an
organ changes is this recorded event:

- ``swap_organ(src, dst, organ, lineage)``: ``dst`` is a freshly built individual whose configuration is ``src``'s
  except for the organ object.  Every store channel, the clock, the life number, the components' state and the
  interventions are carried over exactly -- so the individual's state hash is unchanged by the swap -- and only the
  binding's record of that one organ differs.  Anything else differing is refused.
- the event records the organ before and after (provenance: weights SHA-256, card), the parent and resulting state
  hash, and the lineage it was given; ``OrganSwapLog`` keeps events append-only and hash-chained beside the
  individual (``organ_swaps.jsonl``), like consolidations.
- a rollback is a swap back to the earlier organ, itself an event.

What the organ perceives from the next tick on is new; what the individual *is* (memory, commitments, self-model,
autobiography, consolidation history, learned skill, clock, life) is untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

from .consolidation import ConsolidationLog, _canon

ORGAN_SLOTS = {"vision": "perception", "language": None}     # organ -> the slot whose provenance names it


class OrganSwapError(ValueError):
    pass


def organ_provenance(ind, organ: str) -> dict:
    slot = ORGAN_SLOTS.get(organ)
    if slot is None:
        raise OrganSwapError(f"no swap protocol for the {organ} organ yet")
    for b in ind.rt.identity()["binding"]:
        if b["slot"] == slot:
            return _organ_of(b)
    raise OrganSwapError(f"{slot} is not bound")


def _organ_of(binding: dict) -> dict:
    return dict(((binding.get("provenance") or {}).get("detail") or {}).get("organ") or {})


def _slot_diff(a: dict, b: dict) -> list[str]:
    return [x["slot"] for x, y in zip(a["binding"], b["binding"]) if x != y]


def swap_organ(src, dst, organ: str, lineage: dict, prev_event_sha: str | None = None) -> dict:
    slot = ORGAN_SLOTS.get(organ)
    if slot is None:
        raise OrganSwapError(f"no swap protocol for the {organ} organ yet")
    old, new = src.rt.state_dict(), dst.rt.state_dict()
    if new["store"]["current"]:
        raise OrganSwapError("swap into a freshly built individual")
    mi, ti = old["identity"], new["identity"]
    for key in ("config_hash", "channels", "interventions", "intervention_payloads"):
        if mi[key] != ti[key]:
            raise OrganSwapError(f"the new individual differs in {key}: only the organ may change")
    if len(mi["binding"]) != len(ti["binding"]) or [x["slot"] for x in mi["binding"]] != [x["slot"] for x in ti["binding"]]:
        raise OrganSwapError("the new individual binds different slots")
    diff = _slot_diff(mi, ti)
    if diff != [slot]:
        raise OrganSwapError(f"bindings differ in {diff}; an organ swap may change only {slot}")
    a = next(x for x in mi["binding"] if x["slot"] == slot)
    b = next(x for x in ti["binding"] if x["slot"] == slot)
    if a["component"] != b["component"] or \
            {k: v for k, v in (a["provenance"]["detail"]).items() if k != "organ"} != \
            {k: v for k, v in (b["provenance"]["detail"]).items() if k != "organ"}:
        raise OrganSwapError("the host component changed, not only its organ")
    parent_hash = src.state_hash()
    state = copy.deepcopy(old)
    state["identity"] = ti
    dst.rt.load_state_dict(state)
    dst.life = src.life
    after = dst.state_hash()
    if after != parent_hash:
        raise OrganSwapError("the state hash changed: the swap carried something over inexactly")
    event = {"format": "cb2-organ-swap/1", "kind": lineage.get("kind", "organ_swap"), "organ": organ, "slot": slot,
             "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tick": int(old["tick"]), "life": int(src.life),
             "from": _organ_of(a), "to": _organ_of(b),
             "parent_state_hash": parent_hash, "state_hash": after, "lineage": lineage,
             "prev_event_sha": prev_event_sha}
    event["event_sha"] = hashlib.sha256(_canon(event)).hexdigest()
    return event


class OrganSwapLog(ConsolidationLog):
    """Append-only, hash-chained record of every organ swap of one individual (``organ_swaps.jsonl``)."""


def identity_digest(ind, exclude_slots=("perception",)) -> str:
    """SHA-256 over every store channel outside the excluded slots' own channels, with the tick and life: what an
    organ swap must leave exactly as it was."""
    names = [n for n in ind.rt.store.names() if n.split(".")[0] not in exclude_slots]
    h = hashlib.sha256()
    h.update(json.dumps({"tick": int(ind.rt.tick), "life": int(ind.life)}).encode())
    for n in sorted(names):
        h.update(n.encode())
        h.update(ind.rt.store.value_digest(n).encode())
    return h.hexdigest()
