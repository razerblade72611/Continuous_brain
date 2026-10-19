"""Components: mechanisms bound into slots.

A component declares what it reads (this tick, or delayed = as of the start
of the tick, for recurrent edges), what it writes, and where it came from.
The runtime accepts three provenance kinds:

  LEGACY_VERIFIED     an adapter over a legacy mechanism whose module,
                      evaluator, result, and checkpoints were hash-verified
                      against a VERIFIED legacy manifest entry (C1)
  EXPERIMENTAL_PREREG a new mechanism admitted by an APPROVED preregistration
                      (C4: the first workspace/attention-schema mechanism)
  TEST_FIXTURE        test code; only accepted by a runtime built with the
                      TEST_FIXTURES binding policy, which marks every output
                      of that runtime as non-scientific
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .channels import ChannelDecl


class ProvenanceKind(str, Enum):
    LEGACY_VERIFIED = "legacy_verified"
    EXPERIMENTAL_PREREG = "experimental_prereg"
    TEST_FIXTURE = "test_fixture"


@dataclass(frozen=True)
class ComponentProvenance:
    kind: ProvenanceKind
    detail: Mapping[str, Any] = field(default_factory=dict)   # e.g. manifest entry digest, prereg id

    def as_dict(self) -> dict:
        return {"kind": self.kind.value, "detail": dict(self.detail)}


@dataclass(frozen=True)
class TickContext:
    tick: int
    time_s: float
    dt_s: float
    inputs: Mapping[str, np.ndarray]           # read-only arrays, this tick's values
    delayed: Mapping[str, np.ndarray]          # read-only arrays, start-of-tick values
    coprocessor: Any = None                    # CoprocessorClient; None when the component didn't ask


class Component(abc.ABC):
    """Subclasses set name, slot, reads, reads_delayed, writes, and implement step()."""

    name: str
    slot: str
    reads: tuple[str, ...] = ()
    reads_delayed: tuple[str, ...] = ()
    writes: tuple[ChannelDecl, ...] = ()
    uses_coprocessor: bool = False

    @abc.abstractmethod
    def provenance(self) -> ComponentProvenance: ...

    @abc.abstractmethod
    def step(self, ctx: TickContext) -> Mapping[str, Any]:
        """Return a value for every declared write."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError(f"{self.name} has no state but was given some")

    def reset(self) -> None:
        pass
