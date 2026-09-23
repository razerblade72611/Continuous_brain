"""The contract a spiking executive core must satisfy.

The canonical core is InfantBrain / CorticalNucleus v0.11.3, which lives on
the home machine and has NOT been ported into this repo. It plugs in by
implementing `SpikingCore` in an adapter (see core/infantbrain_adapter.py).
Nothing else in the system depends on its internals.

Learning boundary: the core is never in the backprop graph. Spikes are
discrete events; the transducer reads them after the fact. The core's own
weights change only through `apply_plasticity`, a local rule gated by a scalar
neuromodulator, and (per the spec) only during sleep replay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..transducer.events import SpikeEvents


class SpikingCore(ABC):
    #: Neurons that receive external current (sensory input + backbone write-back).
    n_input: int
    #: Neurons whose spikes are exported to the transducer.
    n_projection: int
    #: Parallel instances sharing one set of weights.
    batch_size: int
    #: Integration step, seconds (InfantBrain: 1 ms).
    dt: float

    @property
    @abstractmethod
    def time_s(self) -> float:
        """Absolute simulation time in seconds (float64 precision)."""

    @abstractmethod
    def step(self, input_current: torch.Tensor, duration_s: float) -> SpikeEvents:
        """Advance by `duration_s`, holding `input_current` [B, n_input] constant.

        Returns projection-neuron spikes emitted during the window, with
        absolute times at integration-step ends.
        """

    @abstractmethod
    def reset_state(self) -> None:
        """Clear transient dynamics (membranes, synaptic currents, eligibility),
        keeping learned weights. Used before each sleep replay."""

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reset the core's noise RNG. Regression probes use fixed seeds so that
        pre-sleep and post-sleep evaluations see identical noise and differ only
        by what sleep changed."""

    @abstractmethod
    def set_plasticity_tracking(self, enabled: bool) -> None:
        """Enable/disable accumulation of eligibility traces while stepping."""

    @abstractmethod
    def apply_plasticity(self, modulator: float) -> dict[str, float]:
        """Apply the accumulated eligibility scaled by a neuromodulator
        (dopamine-like; positive = reinforce, negative = depress). Returns stats."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to restore the core exactly (weights + RNG + time)."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def projection_regions(self) -> list[str] | None:
        """Optional anatomical label per projection neuron (used to build populations)."""
        return None
