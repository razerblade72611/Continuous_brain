"""Deterministic provenance and seed records.

Every result in the program must be traceable to: code revision, config hash,
seed family, checkpoint hash, substrate, compute budget, and artifact SHA-256
(Joint Contract §6; DAV-76 evidence rule). These helpers make the hashes
canonical so two agents computing the same record get the same digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


def _default(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if hasattr(o, "tolist"):                 # numpy arrays / scalars
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"not canonically serializable: {type(o).__name__}")


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, NaN forbidden: one byte string per value."""
    def check(x: Any) -> None:
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("canonical JSON forbids NaN/Inf; encode them explicitly")
        if isinstance(x, Mapping):
            for v in x.values():
                check(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                check(v)
    check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class SeedPurpose(str, Enum):
    DEV = "DEV"                  # development: any tuning allowed
    TEST = "TEST"                # sealed held-out: opened once, never tuned against
    ENGINEERING = "ENGINEERING"  # harness validation only; never scientific evidence
    HISTORICAL = "HISTORICAL"    # ranges recorded by the legacy line


@dataclass(frozen=True)
class SeedFamily:
    """A contiguous, inclusive integer seed range with a purpose."""
    name: str
    start: int
    stop: int                    # inclusive
    purpose: SeedPurpose
    sealed: bool = False
    spent: bool = False

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError(f"seed family {self.name}: stop < start")

    def __len__(self) -> int:
        return self.stop - self.start + 1

    def seeds(self) -> range:
        return range(self.start, self.stop + 1)

    def overlaps(self, other: "SeedFamily") -> bool:
        return not (self.stop < other.start or other.stop < self.start)

    def as_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "stop": self.stop, "purpose": self.purpose.value,
                "sealed": self.sealed, "spent": self.spent}


@dataclass
class ProvenanceRecord:
    """What every result record must carry (Joint Contract §6)."""
    config_hash: str
    code_revision: Mapping[str, Any]
    seed_family: Mapping[str, Any]
    substrate: str
    budget: Mapping[str, Any]
    checkpoint_sha256: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)       # relative path -> sha256
    decision_rule: str | None = None
    linear_refs: tuple[str, ...] = ()

    def attach(self, path: str | os.PathLike, label: str | None = None) -> str:
        digest = sha256_file(path)
        self.artifacts[label or os.fspath(path)] = digest
        return digest

    def as_dict(self) -> dict:
        d = asdict(self)
        d["linear_refs"] = list(self.linear_refs)
        return d

    def digest(self) -> str:
        return sha256_json(self.as_dict())
