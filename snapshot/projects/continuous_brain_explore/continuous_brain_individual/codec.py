"""Exact, inspectable encoding of the individual's store-held state.

Every host keeps its state in a blob channel of the canonical store (``<slot>.state``) and nowhere else.  The
encoding is canonical JSON (sorted keys, no whitespace) with tagged containers for the types JSON cannot hold,
so ``decode(encode(x)) == x`` exactly, including tuple-vs-list, sets, integer dict keys and numpy arrays:

    {"__t": [..]}                         tuple
    {"__s": [..]}                         set (items sorted by their own encoding)
    {"__d": [[key, value], ...]}          dict with any non-string key (pairs sorted by the key's encoding)
    {"__a": dtype, "shape": [...], "b64": base64 of the C-order bytes}   numpy array
    {"__n": dtype, "v": value}            numpy scalar

The same object always encodes to the same bytes, so a state's SHA-256 identifies it.
"""
from __future__ import annotations

import base64
import hashlib
import json

import numpy as np

FORMAT = "cb-individual-state/1"


def _enc(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)) and not isinstance(o, np.integer):
        return o
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            raise ValueError("non-finite float in state")
        return o
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__a": a.dtype.str, "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    if isinstance(o, np.generic):
        return {"__n": o.dtype.str, "v": o.item()}
    if isinstance(o, tuple):
        return {"__t": [_enc(x) for x in o]}
    if isinstance(o, list):
        return [_enc(x) for x in o]
    if isinstance(o, (set, frozenset)):
        items = [_enc(x) for x in o]
        return {"__s": sorted(items, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))}
    if isinstance(o, dict):
        if all(isinstance(k, str) for k in o):
            if any(k.startswith("__") for k in o):
                raise ValueError("state dict keys may not start with '__'")
            return {k: _enc(v) for k, v in o.items()}
        pairs = [[_enc(k), _enc(v)] for k, v in o.items()]
        return {"__d": sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True, separators=(",", ":")))}
    raise TypeError(f"cannot encode {type(o).__name__} in individual state")


def _dec(o):
    if isinstance(o, list):
        return [_dec(x) for x in o]
    if isinstance(o, dict):
        if "__t" in o:
            return tuple(_dec(x) for x in o["__t"])
        if "__s" in o:
            return {_hashable(_dec(x)) for x in o["__s"]}
        if "__d" in o:
            return {_hashable(_dec(k)): _dec(v) for k, v in o["__d"]}
        if "__a" in o:
            a = np.frombuffer(base64.b64decode(o["b64"]), dtype=np.dtype(o["__a"])).reshape(o["shape"])
            return a.copy()
        if "__n" in o:
            return np.dtype(o["__n"]).type(o["v"])
        return {k: _dec(v) for k, v in o.items()}
    return o


def _hashable(x):
    if isinstance(x, list):
        return tuple(_hashable(i) for i in x)
    return x


def encode(obj) -> bytes:
    """Canonical bytes of a state object (wrapped with the format tag)."""
    return json.dumps({"format": FORMAT, "state": _enc(obj)}, sort_keys=True, separators=(",", ":")).encode()


def decode(data: bytes):
    """The state object; b"" (an erased channel) decodes to None."""
    if len(data) == 0:
        return None
    doc = json.loads(bytes(data).decode())
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} blob")
    return _dec(doc["state"])


def blob(obj) -> np.ndarray:
    """``encode`` as the 1-D uint8 array a blob channel holds."""
    return np.frombuffer(encode(obj), dtype=np.uint8).copy()


def unblob(arr) -> object:
    return decode(np.asarray(arr, dtype=np.uint8).tobytes())


def digest(obj) -> str:
    return hashlib.sha256(encode(obj)).hexdigest()
