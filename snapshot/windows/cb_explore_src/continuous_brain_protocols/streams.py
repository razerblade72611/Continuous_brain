"""Embodied experience streams.

A stream delivers frames of physically observable signals at a fixed dt.
Channel units are restricted to a physical vocabulary: this is the protocol
half of the semantic firewall (Charter invariant 2). A channel named
"object_class" with units "label" cannot even be declared.

A frame is a mapping {channel name: float64 array} plus the frame index and
simulated time. Efference channels carry the motor command issued at the
current frame; streams never expose future efference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .provenance import sha256_json

PHYSICAL_UNITS = frozenset({
    "luminance",           # linear intensity, nominally [0, 1]
    "rad", "rad/s", "rad/s^2",
    "m", "m/s", "m/s^2",
    "N", "N*m",
    "Hz",
    "normalized_command",  # dimensionless actuator drive in [0, 1] or [-1, 1]
    "s",
    "1",                   # dimensionless physical ratio (e.g. contact fraction)
})

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# Words that indicate semantic content rather than physics.
_FORBIDDEN_NAME_PARTS = ("label", "class", "object", "word", "token", "caption", "text", "category", "name")


class ChannelKind(str, Enum):
    SENSOR = "sensor"
    EFFERENCE = "efference"        # copy of the motor command issued at this frame
    MOTOR = "motor"                # substrate output


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    shape: tuple[int, ...]
    units: str
    kind: ChannelKind
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"channel name {self.name!r} must be snake_case")
        if any(part in self.name.split("_") for part in _FORBIDDEN_NAME_PARTS):
            raise ValueError(f"channel {self.name!r}: name suggests semantic content (semantic firewall)")
        if self.units not in PHYSICAL_UNITS:
            raise ValueError(f"channel {self.name!r}: units {self.units!r} are not physical "
                             f"(allowed: {sorted(PHYSICAL_UNITS)})")
        if any(int(s) <= 0 for s in self.shape):
            raise ValueError(f"channel {self.name!r}: shape must be positive")
        if self.low is not None and self.high is not None and self.high < self.low:
            raise ValueError(f"channel {self.name!r}: high < low")

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    def as_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "units": self.units, "kind": self.kind.value,
                "low": self.low, "high": self.high}


@dataclass(frozen=True)
class StreamSpec:
    channels: tuple[ChannelSpec, ...]
    dt_s: float

    def __post_init__(self) -> None:
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel names")
        if not self.dt_s > 0:
            raise ValueError("dt must be positive")

    def channel(self, name: str) -> ChannelSpec:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    def of_kind(self, kind: ChannelKind) -> tuple[ChannelSpec, ...]:
        return tuple(c for c in self.channels if c.kind == kind)

    def spec_hash(self) -> str:
        return sha256_json({"dt_s": self.dt_s, "channels": [c.as_dict() for c in self.channels]})

    def validate(self, values: Mapping[str, np.ndarray], kinds: tuple[ChannelKind, ...] | None = None) -> list[str]:
        """Problems with a frame's values: missing/extra channels, shapes, non-finite, out of bounds."""
        wanted = [c for c in self.channels if kinds is None or c.kind in kinds]
        problems = []
        extra = set(values) - {c.name for c in wanted}
        if extra:
            problems.append(f"undeclared channels: {sorted(extra)}")
        for c in wanted:
            if c.name not in values:
                problems.append(f"missing channel {c.name}")
                continue
            v = np.asarray(values[c.name])
            if tuple(v.shape) != tuple(c.shape):
                problems.append(f"{c.name}: shape {tuple(v.shape)} != {c.shape}")
                continue
            if not np.all(np.isfinite(v)):
                problems.append(f"{c.name}: non-finite values")
            # 1e-9 slack: bounds are physical, not bit-exact
            if c.low is not None and v.size and float(v.min()) < c.low - 1e-9:
                problems.append(f"{c.name}: value {float(v.min()):.4g} below {c.low}")
            if c.high is not None and v.size and float(v.max()) > c.high + 1e-9:
                problems.append(f"{c.name}: value {float(v.max()):.4g} above {c.high}")
        return problems


@dataclass(frozen=True)
class Frame:
    index: int
    time_s: float
    values: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]


@runtime_checkable
class EmbodiedStream(Protocol):
    """A source of frames. Closed-loop streams accept the substrate's motor output."""

    @property
    def spec(self) -> StreamSpec: ...

    def reset(self, seed: int) -> Frame: ...

    def next(self, motor: Mapping[str, np.ndarray] | None = None) -> Frame: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...
