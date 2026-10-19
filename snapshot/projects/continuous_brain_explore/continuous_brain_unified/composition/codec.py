"""Exact, deterministic value codec for store-mediated legacy calls.

Every argument and result that crosses a mechanism boundary is serialized to
bytes, written to a blob channel of the canonical store, and decoded on the
other side. Nothing crosses by Python reference except registered frozen
model/controller instances, which travel as instance tokens.

Guarantees
  * bit-exact: tensors and arrays round-trip dtype, shape, requires_grad and
    raw bytes; floats round-trip their IEEE-754 bits
  * deterministic: the same value always encodes to the same bytes (dict
    insertion order is preserved and encoded, because it is observable)
  * aliasing preserved within one payload: an object reachable twice is
    decoded as one object (back-references)
  * fail closed: any type not listed below raises CodecError; the route then
    reports BLOCKED instead of silently approximating a value

Supported: None, bool, int, float, complex, str, bytes, list, tuple, dict,
range, torch.Size, torch.Tensor (CPU, strided), numpy arrays and scalars
(non-object dtypes), Enum members and plain classes / dataclasses defined in
the legacy `continuous_brain` package or in the frozen evaluator modules
(state travels through their __dict__), and registered instances (tokens).

Known limit, recorded rather than hidden: tensors are decoded contiguous;
stride layout and storage sharing between distinct tensors are not
reproduced. The null-composition equivalence test (E1) is what shows this
does not change any frozen evaluator result.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import math
import struct
import sys
from typing import Any, Callable

import numpy as np
import torch

CODEC_VERSION = "cb-exact-codec/1"
MAGIC = b"CBX1"


class CodecError(TypeError):
    pass


_EXTRA_MODULES: dict[str, Any] = {}


def register_extra_module(module: Any) -> None:
    """Make classes of a module that is deliberately not in sys.modules (the frozen evaluator) resolvable."""
    _EXTRA_MODULES[module.__name__] = module


def clear_extra_modules() -> None:
    _EXTRA_MODULES.clear()


def _legacy_class(cls: type, allowed_modules: Callable[[str], bool]) -> bool:
    mod = getattr(cls, "__module__", "") or ""
    if not allowed_modules(mod):
        return False
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        if isinstance(base, type) and issubclass(base, enum.Enum) and base.__module__ == "enum":
            continue
        if not allowed_modules(getattr(base, "__module__", "") or ""):
            return False          # e.g. a legacy class subclassing list/dict/str: __dict__ would not hold its contents
    return True


def default_allowed_module(mod: str) -> bool:
    """Classes whose instances may travel by state: the legacy package and frozen evaluator modules."""
    if mod == "continuous_brain" or mod.startswith("continuous_brain."):
        return True
    top = mod.split(".", 1)[0]
    return top.startswith(("evaluate_", "train_evaluate_", "diagnose_")) or mod == "cb_g0_frozen_evaluator"


def _resolve_class(module: str, qualname: str) -> Any:
    obj: Any = _EXTRA_MODULES.get(module) or sys.modules.get(module)
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


class InstanceTable:
    """Registered objects that cross boundaries by token: frozen models, stateless controllers."""

    def __init__(self):
        self._by_id: dict[int, str] = {}
        self._by_key: dict[str, Any] = {}

    def register(self, key: str, obj: Any) -> None:
        if key in self._by_key and self._by_key[key] is not obj:
            raise CodecError(f"instance key {key!r} already registered to a different object")
        if id(obj) in self._by_id and self._by_id[id(obj)] != key:
            raise CodecError(f"object already registered as {self._by_id[id(obj)]!r}")
        self._by_id[id(obj)] = key
        self._by_key[key] = obj

    def key_of(self, obj: Any) -> str | None:
        k = self._by_id.get(id(obj))
        return k if k is not None and self._by_key[k] is obj else None

    def get(self, key: str) -> Any:
        try:
            return self._by_key[key]
        except KeyError:
            raise CodecError(f"unknown instance token {key!r}") from None

    def keys(self) -> list[str]:
        return sorted(self._by_key)

    def objects(self) -> list[Any]:
        return list(self._by_key.values())


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous()
    if t.numel() == 0:
        return b""
    flat = t.reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def informative(t: torch.Tensor) -> bool:
    """Content worth tracing between hosts: at least two elements, not all equal (zeros, fills and
    scalars are excluded, since equal content there says nothing about provenance)."""
    if t.numel() < 2:
        return False
    flat = t.detach().reshape(-1)
    return not bool((flat == flat[0]).all())


def tensor_digest(t: torch.Tensor) -> str:
    if t.device.type != "cpu":
        raise CodecError("only CPU tensors cross the store")
    return hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode() + _tensor_bytes(t)).hexdigest()


_TORCH_DTYPES = {str(d): d for d in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                                     torch.complex64, torch.complex128, torch.uint8, torch.int8,
                                     torch.int16, torch.int32, torch.int64, torch.bool)}


class _Writer:
    def __init__(self, instances: InstanceTable | None, allowed: Callable[[str], bool]):
        self.parts: list[bytes] = [MAGIC]
        self.memo: dict[int, int] = {}
        self.keep: list[Any] = []
        self.instances = instances
        self.allowed = allowed
        self.tensor_digests: list[str] = []
        self.object_digests: list[str] = []
        self.storages: dict[int, Any] = {}

    def w(self, b: bytes) -> None:
        self.parts.append(b)

    def u(self, n: int) -> None:
        self.w(struct.pack("<Q", n))

    def s(self, text: str) -> None:
        b = text.encode("utf-8")
        self.u(len(b)); self.w(b)

    def _memo(self, obj: Any) -> bool:
        """True if obj was already encoded (a back-reference was written)."""
        idx = self.memo.get(id(obj))
        if idx is not None:
            self.w(b"P"); self.u(idx)
            return True
        self.memo[id(obj)] = len(self.memo)
        self.keep.append(obj)
        return False

    def enc(self, v: Any) -> None:
        if self.instances is not None:
            key = self.instances.key_of(v)
            if key is not None:
                self.w(b"K"); self.s(key)
                return
        t = type(v)
        if v is None:
            self.w(b"N")
        elif t is bool:
            self.w(b"T" if v else b"F")
        elif t is int:
            self.w(b"I"); self.s(str(v))
        elif t is float:
            self.w(b"D"); self.w(struct.pack("<d", v))
        elif t is complex:
            self.w(b"J"); self.w(struct.pack("<dd", v.real, v.imag))
        elif t is str:
            self.w(b"S"); self.s(v)
        elif t is bytes:
            self.w(b"B"); self.u(len(v)); self.w(v)
        elif t is range:
            self.w(b"R"); self.s(str(v.start)); self.s(str(v.stop)); self.s(str(v.step))
        elif t is torch.Size:
            self.w(b"Z"); self.u(len(v))
            for x in v:
                self.s(str(int(x)))
        elif t is tuple:
            self.w(b"U"); self.u(len(v))
            for x in v:
                self.enc(x)
        elif t is list:
            if self._memo(v):
                return
            self.w(b"L"); self.u(len(v))
            for x in v:
                self.enc(x)
        elif t is dict:
            if self._memo(v):
                return
            self.w(b"M"); self.u(len(v))
            for k, x in v.items():
                self.enc(k); self.enc(x)
        elif isinstance(v, torch.Tensor):
            self._tensor(v)
        elif type(v) is np.ndarray:
            self._ndarray(v)
        elif isinstance(v, np.ndarray):
            raise CodecError(f"ndarray subclass {t.__qualname__} cannot cross the store")
        elif isinstance(v, np.generic):
            if v.dtype.kind in "OV" or v.dtype.fields is not None:
                raise CodecError("numpy object/void/structured scalars do not cross the store")
            self.w(b"G"); self.s(v.dtype.str); b = v.tobytes(); self.u(len(b)); self.w(b)
        elif isinstance(v, enum.Enum):
            cls = type(v)
            if isinstance(v, enum.Flag) and v.name is None:
                raise CodecError("enum.Flag combinations cannot cross the store")
            if not _legacy_class(cls, self.allowed):
                raise CodecError(f"enum {cls.__module__}.{cls.__qualname__} is not a legacy type")
            self.w(b"E"); self.s(cls.__module__); self.s(cls.__qualname__); self.s(v.name)
        elif isinstance(v, torch.nn.Module):
            raise CodecError(f"unregistered module instance {t.__module__}.{t.__qualname__}: "
                             "models cross the store only as registered instances")
        elif hasattr(v, "__dict__") and _legacy_class(t, self.allowed) and not isinstance(v, type):
            if getattr(t, "__slots__", None):
                raise CodecError(f"{t.__qualname__} uses __slots__")
            if self._memo(v):
                return
            start = len(self.parts)
            self.w(b"C"); self.s(t.__module__); self.s(t.__qualname__)
            if _resolve_class(t.__module__, t.__qualname__) is not t:
                raise CodecError(f"{t.__module__}.{t.__qualname__} is not resolvable by name (local or shadowed class)")
            state = vars(v)
            self.u(len(state))
            for k, x in state.items():
                self.s(k); self.enc(x)
            self.object_digests.append(hashlib.sha256(b"".join(self.parts[start:])).hexdigest())
        else:
            raise CodecError(f"type {t.__module__}.{t.__qualname__} cannot cross the store")

    def _tensor(self, v: torch.Tensor) -> None:
        if self._memo(v):
            return
        if v.device.type != "cpu" or v.layout != torch.strided or v.is_quantized:
            raise CodecError("only CPU strided non-quantized tensors cross the store")
        if type(v) is not torch.Tensor and type(v) is not torch.nn.Parameter:
            raise CodecError(f"tensor subclass {type(v).__qualname__} cannot cross the store")
        if v.grad_fn is not None:
            raise CodecError("a non-leaf tensor (autograd history) cannot cross the store")
        if v.grad is not None:
            raise CodecError("a tensor carrying .grad cannot cross the store")
        if v.is_conj() or v.is_neg():
            raise CodecError("tensors with conj/neg bits cannot cross the store")
        if v.numel() > 0:
            ptr = v.untyped_storage().data_ptr()
            other = self.storages.get(ptr)
            if other is not None and other is not v:
                raise CodecError("two distinct tensors sharing storage (views) cannot cross the store faithfully")
            self.storages[ptr] = v
        dt = str(v.dtype)
        if dt not in _TORCH_DTYPES:
            raise CodecError(f"tensor dtype {dt} cannot cross the store")
        self.w(b"X"); self.s(dt); self.u(v.dim())
        for n in v.shape:
            self.u(int(n))
        self.w(b"1" if v.requires_grad else b"0")
        self.w(b"1" if isinstance(v, torch.nn.Parameter) else b"0")
        raw = _tensor_bytes(v)
        self.u(len(raw)); self.w(raw)
        if informative(v):
            self.tensor_digests.append(tensor_digest(v))

    def _ndarray(self, v: np.ndarray) -> None:
        if self._memo(v):
            return
        if v.dtype.kind in "OV" or v.dtype.fields is not None:
            raise CodecError("numpy object/void/structured arrays do not cross the store")
        self.w(b"A"); self.s(v.dtype.str); self.u(v.ndim)
        for n in v.shape:
            self.u(int(n))
        raw = np.ascontiguousarray(v).tobytes()
        self.w(b"1" if v.flags.writeable else b"0")
        self.u(len(raw)); self.w(raw)


class _Reader:
    def __init__(self, data: bytes, instances: InstanceTable | None, allowed: Callable[[str], bool]):
        if data[:4] != MAGIC:
            raise CodecError("not a codec payload")
        self.d = memoryview(data); self.i = 4
        self.memo: list[Any] = []
        self.instances = instances
        self.allowed = allowed
        self.tensor_digests: list[str] = []
        self.object_digests: list[str] = []

    def r(self, n: int) -> bytes:
        b = bytes(self.d[self.i:self.i + n])
        if len(b) != n:
            raise CodecError("truncated payload")
        self.i += n
        return b

    def u(self) -> int:
        return struct.unpack("<Q", self.r(8))[0]

    def s(self) -> str:
        return self.r(self.u()).decode("utf-8")

    def dec(self) -> Any:
        tag = self.r(1)
        if tag == b"N":
            return None
        if tag == b"T":
            return True
        if tag == b"F":
            return False
        if tag == b"I":
            return int(self.s())
        if tag == b"D":
            return struct.unpack("<d", self.r(8))[0]
        if tag == b"J":
            re_, im = struct.unpack("<dd", self.r(16)); return complex(re_, im)
        if tag == b"S":
            return self.s()
        if tag == b"B":
            return self.r(self.u())
        if tag == b"R":
            return range(int(self.s()), int(self.s()), int(self.s()))
        if tag == b"Z":
            return torch.Size([int(self.s()) for _ in range(self.u())])
        if tag == b"U":
            return tuple(self.dec() for _ in range(self.u()))
        if tag == b"L":
            out: list = []
            self.memo.append(out)
            n = self.u()
            for _ in range(n):
                out.append(self.dec())
            return out
        if tag == b"M":
            out_d: dict = {}
            self.memo.append(out_d)
            n = self.u()
            for _ in range(n):
                k = self.dec(); out_d[k] = self.dec()
            return out_d
        if tag == b"P":
            return self.memo[self.u()]
        if tag == b"K":
            if self.instances is None:
                raise CodecError("instance token without an instance table")
            return self.instances.get(self.s())
        if tag == b"X":
            return self._tensor()
        if tag == b"A":
            return self._ndarray()
        if tag == b"G":
            dt = np.dtype(self.s()); b = self.r(self.u())
            return np.frombuffer(b, dtype=dt)[0]
        if tag == b"E":
            cls = self._cls(self.s(), self.s())
            return cls[self.s()]
        if tag == b"C":
            start = self.i - 1
            cls = self._cls(self.s(), self.s())
            obj = cls.__new__(cls)
            self.memo.append(obj)
            n = self.u()
            state = {}
            for _ in range(n):
                k = self.s(); state[k] = self.dec()
            obj.__dict__.update(state)
            self.object_digests.append(hashlib.sha256(bytes(self.d[start:self.i])).hexdigest())
            return obj
        raise CodecError(f"unknown tag {tag!r}")

    def _cls(self, module: str, qualname: str) -> type:
        if not self.allowed(module):
            raise CodecError(f"{module} is not an allowed legacy module")
        mod = _EXTRA_MODULES.get(module) or sys.modules.get(module) or importlib.import_module(module)
        obj: Any = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj

    def _tensor(self) -> torch.Tensor:
        idx = len(self.memo)
        self.memo.append(None)
        dt = _TORCH_DTYPES[self.s()]
        nd = self.u()
        shape = tuple(self.u() for _ in range(nd))
        req = self.r(1) == b"1"
        is_param = self.r(1) == b"1"
        raw = self.r(self.u())
        if len(raw) == 0:
            t = torch.empty(shape, dtype=dt)
        else:
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape).clone()
        if is_param:
            t = torch.nn.Parameter(t, requires_grad=req)
        elif req:
            t.requires_grad_(True)
        self.memo[idx] = t
        if informative(t):
            self.tensor_digests.append(tensor_digest(t))
        return t

    def _ndarray(self) -> np.ndarray:
        idx = len(self.memo)
        self.memo.append(None)
        dt = np.dtype(self.s()); nd = self.u()
        shape = tuple(self.u() for _ in range(nd))
        writeable = self.r(1) == b"1"
        raw = self.r(self.u())
        a = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
        a.setflags(write=writeable)
        self.memo[idx] = a
        return a


@dataclasses.dataclass
class Encoded:
    data: bytes
    tensor_digests: list[str]
    object_digests: list[str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def encode(value: Any, instances: InstanceTable | None = None,
           allowed: Callable[[str], bool] = default_allowed_module) -> Encoded:
    w = _Writer(instances, allowed)
    try:
        w.enc(value)
    except CodecError:
        raise
    except Exception as e:      # fail closed: any unexpected failure is a codec failure, never a legacy error
        raise CodecError(f"encode failed: {type(e).__name__}: {e}") from e
    return Encoded(b"".join(w.parts), w.tensor_digests, w.object_digests)


@dataclasses.dataclass
class Decoded:
    value: Any
    tensor_digests: list[str]
    object_digests: list[str]


def decode_with_digests(data: bytes, instances: InstanceTable | None = None,
                        allowed: Callable[[str], bool] = default_allowed_module) -> Decoded:
    try:
        r = _Reader(bytes(data), instances, allowed)
        v = r.dec()
        if r.i != len(r.d):
            raise CodecError("trailing bytes after payload")
    except CodecError:
        raise
    except Exception as e:
        raise CodecError(f"decode failed: {type(e).__name__}: {e}") from e
    return Decoded(v, r.tensor_digests, r.object_digests)


def decode(data: bytes, instances: InstanceTable | None = None,
           allowed: Callable[[str], bool] = default_allowed_module) -> Any:
    return decode_with_digests(data, instances, allowed).value


def equal_encoding(a: Any, b: Any, instances: InstanceTable | None = None) -> bool:
    return encode(a, instances).data == encode(b, instances).data


def float_bits_equal(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (math.isnan(a) and math.isnan(b))
