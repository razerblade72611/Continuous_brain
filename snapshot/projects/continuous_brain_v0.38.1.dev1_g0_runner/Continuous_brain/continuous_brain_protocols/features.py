"""Feature sources read by probes.

A feature source exposes a fixed-length, causal feature vector: the value at
frame t depends only on inputs up to and including frame t. Probes never see
anything else about a substrate, so every network and baseline is read by the
identical procedure (E2 prereg, "Probes").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    size: int
    description: str
    filter_tau_s: float | None = None     # e.g. 0.025 for spike trains filtered at 25 ms


@runtime_checkable
class FeatureSource(Protocol):
    def feature_spec(self) -> FeatureSpec: ...

    def read_features(self) -> np.ndarray:
        """A float64 copy, shape [size], of the features at the current frame."""
        ...
