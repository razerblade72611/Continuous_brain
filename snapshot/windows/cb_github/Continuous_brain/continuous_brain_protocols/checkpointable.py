"""Checkpointable contract and a canonical digest of nested state."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Checkpointable(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


def state_digest(state: Any) -> str:
    """SHA-256 over a nested state (dicts, lists, numpy arrays, torch tensors, scalars).

    Keys are visited in sorted order and each leaf contributes its type, dtype,
    shape, and bytes, so equal digests mean bit-identical state.
    """
    h = hashlib.sha256()

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            h.update(b"{")
            for k in sorted(x, key=str):
                h.update(str(k).encode() + b":")
                visit(x[k])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                visit(v)
            h.update(b"]")
        elif hasattr(x, "detach") and hasattr(x, "cpu"):          # torch tensor, without importing torch
            arr = x.detach().cpu().contiguous().numpy()
            h.update(f"T{arr.dtype}{arr.shape}".encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        elif isinstance(x, np.ndarray):
            h.update(f"A{x.dtype}{x.shape}".encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, (bytes, bytearray)):
            h.update(b"B" + bytes(x))
        elif x is None or isinstance(x, (bool, int, float, str, np.generic)):
            h.update(f"S{type(x).__name__}:{x!r}".encode())
        else:
            raise TypeError(f"state_digest cannot hash {type(x).__name__}")

    visit(state)
    return h.hexdigest()
