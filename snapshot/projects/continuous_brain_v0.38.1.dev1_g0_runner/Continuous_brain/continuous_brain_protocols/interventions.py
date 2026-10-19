"""Declared interventions.

Evaluators are read-only with respect to agent state *except* through a
declared intervention (Joint Contract §5). An intervention names its kind,
its target (a channel or a state key the owner declared intervenable), the
tick window, and, for gate runs, the preregistration it belongs to. Every
application produces a record for the run log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .provenance import sha256_json


class InterventionKind(str, Enum):
    ERASE = "erase"          # replace with the owner's declared neutral value
    DONOR = "donor"          # substitute a value recorded from another run/subject
    FREEZE = "freeze"        # hold the value observed at the window start
    REPLACE = "replace"      # write a fixed payload
    LESION = "lesion"        # structural removal inside a substrate


class InterventionError(RuntimeError):
    pass


@dataclass(frozen=True)
class InterventionSpec:
    kind: InterventionKind
    target: str
    start_tick: int
    end_tick: int | None = None          # exclusive; None = until the end of the run
    payload_ref: str | None = None       # donor record id or payload id
    prereg_id: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.start_tick < 0:
            raise ValueError("start_tick must be >= 0")
        if self.end_tick is not None and self.end_tick <= self.start_tick:
            raise ValueError("end_tick must be greater than start_tick")
        if self.kind in (InterventionKind.DONOR, InterventionKind.REPLACE) and not self.payload_ref:
            raise ValueError(f"{self.kind.value} interventions need a payload_ref")

    def active(self, tick: int) -> bool:
        return tick >= self.start_tick and (self.end_tick is None or tick < self.end_tick)

    def as_dict(self) -> dict:
        return {"kind": self.kind.value, "target": self.target, "start_tick": self.start_tick,
                "end_tick": self.end_tick, "payload_ref": self.payload_ref, "prereg_id": self.prereg_id,
                "note": self.note}

    def spec_hash(self) -> str:
        return sha256_json(self.as_dict())


@dataclass
class InterventionRecord:
    spec_hash: str
    tick: int
    target: str
    kind: str
    before_sha256: str | None = None
    after_sha256: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InterventionTarget(Protocol):
    def intervention_targets(self) -> tuple[str, ...]:
        """Names this object allows interventions on."""
        ...
