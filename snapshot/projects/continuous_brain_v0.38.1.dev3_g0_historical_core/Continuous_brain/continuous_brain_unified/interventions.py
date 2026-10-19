"""Intervention manager: the only way anything other than a channel's owner changes it.

An intervention is applied immediately after the target channel's writer runs
(or after the observation is written, for environment channels), so every
downstream reader in the same tick sees the intervened value.

  ERASE    the channel's declared neutral value
  FREEZE   the value the channel held at the start of the window (start-of-tick
           value at `start_tick`), held for the whole window
  DONOR    a recorded trace from another run: {tick: value} or one constant value
  REPLACE  a fixed payload

Specs are registered before the run and locked on the first tick, so a gate
run cannot add interventions after seeing data.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from continuous_brain_protocols import (InterventionError, InterventionKind, InterventionRecord,
                                        InterventionSpec, state_digest)

from .channels import ChannelStore, value_sha256


class InterventionManager:
    def __init__(self, store: ChannelStore):
        self.store = store
        self._token = store.issue_override_token()
        self.specs: list[InterventionSpec] = []
        self.payloads: dict[str, Any] = {}
        self.records: list[InterventionRecord] = []
        self._frozen_values: dict[str, np.ndarray] = {}
        self.locked = False

    def add_payload(self, ref: str, value: Any) -> None:
        """A constant array, or a donor trace {tick: array}. Stored as a read-only private copy."""
        if self.locked:
            raise InterventionError("interventions are locked once the run has started")
        if ref in self.payloads:
            raise InterventionError(f"payload {ref!r} already registered")
        self.payloads[ref] = _frozen_copy(value)

    def payload_digest(self) -> str:
        return state_digest({k: v for k, v in sorted(self.payloads.items())})

    def register(self, spec: InterventionSpec) -> None:
        if self.locked:
            raise InterventionError("interventions are locked once the run has started")
        decl = self.store.decl(spec.target)
        if not decl.intervenable:
            raise InterventionError(f"{spec.target!r} is not declared intervenable")
        if spec.kind == InterventionKind.LESION:
            raise InterventionError("lesions act inside substrates (Substrate.lesion), not on channels")
        if spec.payload_ref is not None and spec.payload_ref not in self.payloads:
            raise InterventionError(f"payload {spec.payload_ref!r} not registered")
        for other in self.specs:
            if other.target == spec.target and _windows_overlap(other, spec):
                raise InterventionError(f"overlapping interventions on {spec.target!r}")
        self.specs.append(spec)

    def lock(self) -> None:
        self.locked = True
        self.specs = tuple(self.specs)                 # no appends after the run starts
        self.payloads = _ReadOnlyDict(self.payloads)

    def targets_active(self, tick: int) -> set[str]:
        return {s.target for s in self.specs if s.active(tick)}

    def apply(self, channel: str, tick: int) -> None:
        for spec in self.specs:
            if spec.target != channel or not spec.active(tick):
                continue
            decl = self.store.decl(channel)
            before = self.store.read(channel)
            if spec.kind == InterventionKind.ERASE:
                value = decl.neutral_value()
            elif spec.kind == InterventionKind.FREEZE:
                key = spec.spec_hash()
                if tick == spec.start_tick or key not in self._frozen_values:
                    self._frozen_values[key] = np.array(self.store.read(channel, delayed=True))
                value = self._frozen_values[key]
            elif spec.kind == InterventionKind.DONOR:
                value = _donor_value(self.payloads[spec.payload_ref], tick, channel)
            elif spec.kind == InterventionKind.REPLACE:
                value = self.payloads[spec.payload_ref]
            else:
                raise InterventionError(f"unsupported kind {spec.kind}")
            after_sha = self.store.override(channel, value, self._token, spec.kind.value)
            self.records.append(InterventionRecord(spec.spec_hash(), tick, channel, spec.kind.value,
                                                   value_sha256(before), after_sha))

    def state_dict(self) -> dict:
        return {"specs": [s.as_dict() for s in self.specs], "locked": self.locked,
                "frozen_values": {k: np.array(v) for k, v in self._frozen_values.items()},
                "records": [asdict(r) for r in self.records]}

    def load_state_dict(self, state: dict) -> None:
        mine = [s.as_dict() for s in self.specs]
        if mine != state["specs"]:
            raise InterventionError("checkpoint was taken with a different intervention schedule")
        if state["locked"] and not self.locked:
            self.lock()
        self._frozen_values = {k: np.array(v) for k, v in state["frozen_values"].items()}
        self.records = [InterventionRecord(**r) for r in state["records"]]


class _ReadOnlyDict(dict):
    def _refuse(self, *a, **k):
        raise InterventionError("payloads are locked once the run has started")

    __setitem__ = __delitem__ = update = pop = popitem = clear = setdefault = _refuse


def _frozen_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _frozen_copy(v) for k, v in value.items()}
    arr = np.array(value, copy=True)
    arr.setflags(write=False)
    return arr


def _windows_overlap(a: InterventionSpec, b: InterventionSpec) -> bool:
    a_end = a.end_tick if a.end_tick is not None else float("inf")
    b_end = b.end_tick if b.end_tick is not None else float("inf")
    return a.start_tick < b_end and b.start_tick < a_end


def _donor_value(payload: Any, tick: int, channel: str) -> Any:
    if isinstance(payload, Mapping):
        if tick not in payload:
            raise InterventionError(f"donor trace for {channel!r} has no value at tick {tick}")
        return payload[tick]
    return payload
