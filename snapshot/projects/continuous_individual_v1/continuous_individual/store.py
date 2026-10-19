"""Canonical store: single source of authoritative state for one individual."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class CanonicalStore:
    """Single-writer blob store. All persistent individual state lives here."""

    def __init__(self) -> None:
        self._channels: Dict[str, bytes] = {}
        self._write_log: list[tuple[str, str]] = []

    def write(self, channel: str, value: Any) -> None:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._channels[channel] = blob
        digest = hashlib.sha256(blob).hexdigest()
        self._write_log.append((channel, digest))

    def read(self, channel: str) -> Any:
        if channel not in self._channels:
            raise KeyError(f"channel {channel!r} not present")
        return pickle.loads(self._channels[channel])

    def erase(self, channel: str) -> None:
        self._channels.pop(channel, None)
        self._write_log.append((channel, "ERASED"))

    def has(self, channel: str) -> bool:
        return channel in self._channels

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self._channels):
            h.update(k.encode())
            h.update(self._channels[k])
        return h.hexdigest()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"channels": self._channels, "log": self._write_log}, f)

    @classmethod
    def load(cls, path: Path) -> "CanonicalStore":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        store = cls()
        store._channels = data["channels"]
        store._write_log = data["log"]
        return store

    def snapshot_bytes(self) -> bytes:
        return pickle.dumps({"channels": self._channels, "log": self._write_log}, protocol=pickle.HIGHEST_PROTOCOL)
