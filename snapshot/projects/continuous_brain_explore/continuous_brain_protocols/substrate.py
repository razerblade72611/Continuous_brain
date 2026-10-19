"""The common substrate API (Joint Contract §3, amended by C2).

Both the SNN (H1) and the dense recurrent comparator (H2) implement this, so
the battery can run matched assays without importing either track.

C2: local plasticity and gradient training are separate calls.
  * `plasticity_step(modulators)`  local rules (STDP, homeostasis, structural)
  * `train_step(batch)`            gradient training (backprop)
A substrate that must never receive gradients (the SNN, Charter invariant 3)
raises `GradientFirewallError` from `train_step`. A substrate without a
capability raises `CapabilityUnavailable`, which names why (e.g. the Linear
issue that will add it), so a battery item can report CAPABILITY_ABSENT
instead of silently running a no-op.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .features import FeatureSource
from .provenance import sha256_json
from .streams import StreamSpec


class Capability(str, Enum):
    LOCAL_PLASTICITY = "local_plasticity"
    GRADIENT_TRAINING = "gradient_training"
    HOMEOSTASIS = "homeostasis"
    LESION = "lesion"
    FREEZE = "freeze"
    SPONTANEOUS_ACTIVITY = "spontaneous_activity"
    EXACT_RESTORE = "exact_restore"


class GradientFirewallError(RuntimeError):
    """Raised when gradient training is attempted on a substrate that forbids it."""


class CapabilityUnavailable(NotImplementedError):
    def __init__(self, capability: Capability, reason: str):
        super().__init__(f"{capability.value} unavailable: {reason}")
        self.capability = capability
        self.reason = reason


class SubstrateFrozenError(RuntimeError):
    """Raised when a frozen substrate is asked to learn."""


@dataclass(frozen=True)
class SubstrateMetadata:
    kind: str                         # e.g. "snn-e0", "dense-rnn-gru"
    track: str                        # "H1" or "H2"
    seed: int
    config_hash: str
    input_spec_hash: str
    version: str
    capabilities: frozenset[Capability]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["capabilities"] = sorted(c.value for c in self.capabilities)
        return d


@dataclass
class BudgetReport:
    """Compute and capacity accounting used for budget matching (H1/H2 doc)."""
    kind: str
    learned_scalars: int                 # parameters (dense) or active synapse weights (SNN)
    active_synapses: int | None
    parameter_count: int
    frames: int
    simulated_s: float
    wall_s: float
    peak_memory_bytes: int | None
    device: str
    gradient_updates: int = 0
    plasticity_updates: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LesionSpec:
    """Structural damage: `fraction` of the elements selected by `target`.

    target grammar:
      "synapses:<region>"  fraction of active synapses onto neurons of a region (SNN)
      "neurons:<region>"   fraction of neurons in a region, all their synapses (SNN)
      "units:hidden"       fraction of recurrent hidden units (dense)
      "weights:recurrent"  fraction of recurrent weights (dense)
    """
    target: str
    fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("lesion fraction must be in (0, 1]")
        if ":" not in self.target:
            raise ValueError("lesion target must look like '<what>:<where>'")

    def spec_hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass
class LesionRecord:
    spec_hash: str
    target: str
    requested_fraction: float
    removed: int
    of_total: int
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlasticityReport:
    updated: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainReport:
    updated: bool
    loss: float | None
    targets_used: int
    detail: dict[str, Any] = field(default_factory=dict)


class Substrate(abc.ABC):
    """A continuously stepped recurrent system with read-only state exposure."""

    # ------------------------------------------------------------------ identity
    @property
    @abc.abstractmethod
    def metadata(self) -> SubstrateMetadata: ...

    @abc.abstractmethod
    def input_spec(self) -> StreamSpec:
        """Sensor and efference channels this substrate consumes."""

    @abc.abstractmethod
    def output_spec(self) -> StreamSpec:
        """Motor channels this substrate emits."""

    # ------------------------------------------------------------------ dynamics
    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset dynamic state (not learned parameters) to genesis values."""

    @abc.abstractmethod
    def step(self, frame: Mapping[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        """Advance by dt seconds with the frame held constant; return motor outputs."""

    @abc.abstractmethod
    def expose_state(self) -> Mapping[str, np.ndarray]:
        """Read-only copies of the dynamic state (arrays with writeable=False)."""

    @abc.abstractmethod
    def features(self) -> FeatureSource: ...

    # ------------------------------------------------------------------ learning (C2)
    @abc.abstractmethod
    def plasticity_step(self, modulators: Mapping[str, float] | None = None) -> PlasticityReport: ...

    @abc.abstractmethod
    def train_step(self, batch: Any = None) -> TrainReport: ...

    @abc.abstractmethod
    def freeze(self) -> None:
        """Stop every adaptive process (plasticity, homeostasis, training)."""

    @abc.abstractmethod
    def unfreeze(self) -> None: ...

    @property
    @abc.abstractmethod
    def frozen(self) -> bool: ...

    # ------------------------------------------------------------------ damage, budget, persistence
    @abc.abstractmethod
    def lesion(self, spec: LesionSpec) -> LesionRecord: ...

    @abc.abstractmethod
    def budget(self) -> BudgetReport: ...

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

    @abc.abstractmethod
    def state_hash(self) -> str: ...

    # ------------------------------------------------------------------ helpers
    def has(self, capability: Capability) -> bool:
        return capability in self.metadata.capabilities


def read_only(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out
