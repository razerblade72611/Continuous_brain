"""Channel store: the only shared state between components.

Rules:
  * single writer: every channel has exactly one owner; a write by anyone else
    raises ChannelWriteError. Mechanism channels are namespaced by slot.
  * declared shape and dtype; values must be finite
  * reads return read-only copies, so a reader can never mutate shared state
  * two views per channel: `current` (latest write, possibly this tick) and
    `previous` (as it stood at the start of the tick) for recurrent edges
  * every write and every intervention override appends to the write log
    (tick, channel, writer, SHA-256 of the value bytes)
  * every read made on behalf of a named reader is counted in the read log
    (reader, channel, number of reads, reads of a value actually written
    rather than the neutral default, and a running SHA-256 chain over the
    (tick, value digest) pairs read), so an attestation can prove which
    written values each component consumed
  * a channel declared with shape=None is a variable-length 1-D uint8 blob
    (used for legacy call payloads); its neutral value is the empty blob

Interventions do not go through `write`; they go through `override`, which
requires the InterventionManager's token, so an evaluator holding the store
cannot fake one.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuous_brain_protocols import read_only


class ChannelError(RuntimeError):
    pass


class ChannelWriteError(ChannelError):
    pass


@dataclass(frozen=True)
class ChannelDecl:
    name: str
    owner: str
    shape: tuple[int, ...] | None       # None: variable-length 1-D uint8 blob
    dtype: str = "float64"
    neutral: float = 0.0               # value used by ERASE and before the first write
    intervenable: bool = True
    exact_contract: bool = False       # the owner promises exact preservation (G4 checks exactly)
    description: str = ""

    @property
    def is_blob(self) -> bool:
        return self.shape is None

    def neutral_value(self) -> np.ndarray:
        if self.is_blob:
            return np.zeros(0, dtype=np.uint8)
        return np.full(self.shape, self.neutral, dtype=self.dtype)


def value_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(f"{arr.dtype}{arr.shape}".encode() + arr.tobytes()).hexdigest()


class ChannelStore:
    def __init__(self, log_path: str | os.PathLike | None = None, log_memory: int = 100_000):
        self._decl: dict[str, ChannelDecl] = {}
        self._current: dict[str, np.ndarray] = {}
        self._previous: dict[str, np.ndarray] = {}
        self._written_this_tick: set[str] = set()
        self.tick = -1
        self.log: deque[dict] = deque(maxlen=log_memory)   # in-memory tail; the JSONL file (if set) is complete
        self.log_total = 0
        self.log_path = Path(log_path) if log_path else None
        self._override_token: object | None = None
        self.read_stats: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------ declaration
    def declare(self, decl: ChannelDecl) -> None:
        if decl.is_blob and decl.dtype != "uint8":
            raise ChannelError(f"blob channel {decl.name!r} must be declared with dtype uint8")
        if decl.name in self._decl:
            raise ChannelError(f"channel {decl.name!r} already declared by {self._decl[decl.name].owner!r}")
        prefix = decl.name.split(".", 1)[0]
        if "." in decl.name and prefix != decl.owner:
            raise ChannelError(f"channel {decl.name!r} is in namespace {prefix!r} but owned by {decl.owner!r}")
        self._decl[decl.name] = decl

    def undeclare_owner(self, owner: str) -> None:
        for name in [n for n, d in self._decl.items() if d.owner == owner]:
            del self._decl[name]
            self._current.pop(name, None)
            self._previous.pop(name, None)

    def decl(self, name: str) -> ChannelDecl:
        try:
            return self._decl[name]
        except KeyError:
            raise ChannelError(f"undeclared channel {name!r}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._decl))

    def owners(self) -> dict[str, str]:
        return {n: d.owner for n, d in self._decl.items()}

    # ------------------------------------------------------------------ ticks
    def begin_tick(self, tick: int) -> None:
        if tick <= self.tick:
            raise ChannelError(f"tick must increase ({tick} after {self.tick})")
        self.tick = tick
        self._previous = {k: v for k, v in self._current.items()}   # arrays are never mutated in place
        self._written_this_tick.clear()

    # ------------------------------------------------------------------ access
    def _coerce(self, decl: ChannelDecl, value: Any) -> np.ndarray:
        if decl.is_blob:
            if isinstance(value, (bytes, bytearray, memoryview)):
                arr = np.frombuffer(bytes(value), dtype=np.uint8).copy()
            else:
                arr = np.array(value, copy=True)
                if arr.dtype != np.uint8:
                    raise ChannelWriteError(f"{decl.name}: blob values must be bytes or uint8, got {arr.dtype}")
            if arr.ndim != 1:
                raise ChannelWriteError(f"{decl.name}: blob values must be 1-D")
            arr.setflags(write=False)
            return arr
        arr = np.array(value, dtype=decl.dtype, copy=True)
        if tuple(arr.shape) != tuple(decl.shape):
            raise ChannelWriteError(f"{decl.name}: shape {tuple(arr.shape)} != declared {decl.shape}")
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise ChannelWriteError(f"{decl.name}: non-finite value")
        arr.setflags(write=False)
        return arr

    def write(self, name: str, value: Any, writer: str) -> str:
        decl = self.decl(name)
        if writer != decl.owner:
            raise ChannelWriteError(f"{writer!r} may not write {name!r} (owner {decl.owner!r})")
        if name in self._written_this_tick:
            raise ChannelWriteError(f"{name!r} written twice in tick {self.tick}")
        arr = self._coerce(decl, value)
        self._current[name] = arr
        self._written_this_tick.add(name)
        return self._log(name, writer, arr, "write")

    def override(self, name: str, value: Any, token: object, cause: str) -> str:
        if token is None or token is not self._override_token:
            raise ChannelWriteError("overrides are only available to the runtime's intervention manager")
        decl = self.decl(name)
        if not decl.intervenable:
            raise ChannelWriteError(f"{name!r} is not declared intervenable")
        arr = self._coerce(decl, value)
        self._current[name] = arr
        return self._log(name, f"intervention:{cause}", arr, "override")

    def issue_override_token(self) -> object:
        if self._override_token is not None:
            raise ChannelError("override token already issued")
        self._override_token = object()
        return self._override_token

    def read(self, name: str, delayed: bool = False, reader: str | None = None) -> np.ndarray:
        decl = self.decl(name)
        src = self._previous if delayed else self._current
        written = name in src
        value = src[name].view() if written else read_only(decl.neutral_value())
        # a view of a read-only base cannot be made writable again
        if reader is not None:
            self._count_read(reader, name, value, written, delayed)
        return value

    def _count_read(self, reader: str, name: str, value: np.ndarray, written: bool, delayed: bool) -> None:
        st = self.read_stats.setdefault((reader, name), {"reads": 0, "written_reads": 0, "nonempty_reads": 0,
                                                         "delayed_reads": 0, "chain": hashlib.sha256().hexdigest()})
        st["reads"] += 1
        st["delayed_reads"] += int(delayed)
        st["written_reads"] += int(written)
        st["nonempty_reads"] += int(written and value.size > 0)
        st["chain"] = hashlib.sha256(f"{st['chain']}|{self.tick}|{value_sha256(value)}".encode()).hexdigest()

    def read_report(self) -> list[dict]:
        """Per (reader, channel): counts and the read chain, with the channel's writer."""
        return [{"reader": r, "channel": c, "writer": self._decl[c].owner, **dict(st)}
                for (r, c), st in sorted(self.read_stats.items())]

    def written(self, name: str) -> bool:
        return name in self._current

    def _log(self, name: str, writer: str, arr: np.ndarray, op: str) -> str:
        digest = value_sha256(arr)
        entry = {"tick": self.tick, "channel": name, "writer": writer, "op": op, "sha256": digest}
        self.log.append(entry)
        self.log_total += 1
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return digest

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"tick": self.tick,
                "current": {k: np.array(v) for k, v in self._current.items()},
                "previous": {k: np.array(v) for k, v in self._previous.items()},
                "written_this_tick": sorted(self._written_this_tick)}

    def load_state_dict(self, state: dict) -> None:
        unknown = set(state["current"]) - set(self._decl)
        if unknown:
            raise ChannelError(f"state holds undeclared channels: {sorted(unknown)}")
        self.tick = int(state["tick"])
        self._current = {k: self._coerce(self._decl[k], v) for k, v in state["current"].items()}
        self._previous = {k: self._coerce(self._decl[k], v) for k, v in state["previous"].items()}
        self._written_this_tick = set(state["written_this_tick"])

    def digest(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self._current):
            h.update(k.encode() + value_sha256(self._current[k]).encode())
        return h.hexdigest()
