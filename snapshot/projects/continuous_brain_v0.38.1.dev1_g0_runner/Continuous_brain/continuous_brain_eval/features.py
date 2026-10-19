"""Feature adapters read by every probe with one identical procedure (E2 prereg, "Probes").

  SpikeFeatures      the SNN's per-neuron spike trains filtered at 25 ms
  DenseStateFeatures the GRU hidden state
  RawHistoryFeatures the raw physical input channels with L lags (RAW-INPUT
                     baseline: 20 lags at 10 ms = 200 ms of history)

Subset rule (same for every network): all features if there are at most
`max_features`, otherwise a fixed random subset of exactly `max_features`
drawn from `seed`. The selected indices are part of the result record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from continuous_brain_protocols import FeatureSource, FeatureSpec, StreamSpec


def select_features(n_total: int, max_features: int = 2000, seed: int = 0) -> np.ndarray:
    if n_total <= max_features:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=max_features, replace=False))


@dataclass
class SubsetFeatures:
    """Wrap any FeatureSource with the subset rule."""
    source: FeatureSource
    max_features: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        self.index = select_features(self.source.feature_spec().size, self.max_features, self.seed)

    def feature_spec(self) -> FeatureSpec:
        s = self.source.feature_spec()
        return FeatureSpec(f"{s.name}[subset]", int(self.index.size), s.description, s.filter_tau_s)

    def read_features(self) -> np.ndarray:
        return self.source.read_features()[self.index]


def SpikeFeatures(substrate, max_features: int = 2000, seed: int = 0) -> SubsetFeatures:  # noqa: N802
    return SubsetFeatures(substrate.features(), max_features, seed)


def DenseStateFeatures(substrate, max_features: int = 2000, seed: int = 0) -> SubsetFeatures:  # noqa: N802
    return SubsetFeatures(substrate.features(), max_features, seed)


class RawHistoryFeatures:
    """Raw physical channels of the last `lags` frames, newest first (zeros before the stream starts)."""

    def __init__(self, spec: StreamSpec, lags: int = 20, channels: tuple[str, ...] | None = None):
        self.spec = spec
        self.lags = lags
        self.channels = channels or tuple(c.name for c in spec.channels)
        self.width = sum(spec.channel(c).size for c in self.channels)
        self._hist: deque[np.ndarray] = deque(maxlen=lags)

    def push(self, frame: Mapping[str, np.ndarray]) -> None:
        self._hist.appendleft(np.concatenate([np.asarray(frame[c], dtype=np.float64).reshape(-1)
                                              for c in self.channels]))

    def reset(self) -> None:
        self._hist.clear()

    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec("raw_history", self.width * self.lags,
                           f"{len(self.channels)} raw channels x {self.lags} lags")

    def read_features(self) -> np.ndarray:
        out = np.zeros(self.width * self.lags)
        for i, v in enumerate(self._hist):
            out[i * self.width:(i + 1) * self.width] = v
        return out
