"""Store-mediated mechanism boundaries.

A frozen legacy evaluator runs unmodified as the runtime's task driver. Every
top-level call it (or any legacy helper) makes to a hosted mechanism API
becomes one runtime tick:

  1. the call (API name, arguments) is encoded and written by the driver to
     the blob channel driver.<slot>.request
  2. the runtime ticks: every bound host steps in scheduler order; the host
     owning <slot> reads its request from the store, decodes it (fresh
     objects, no Python references), executes the original legacy callable,
     and writes the encoded result, any in-place argument mutation, and its
     canonical channels
  3. the driver reads <slot>.response from the store, applies the argument
     post-state in place (preserving the caller's object identities), and
     returns the decoded result

What is and is not mediated (measured, reported in every attestation)
  * mediated: top-level calls to the spec'd mechanism APIs (spec.SPECS)
  * nested: a hosted API that internally calls another spec'd API runs that
    call directly inside the host's tick (a tick cannot nest); counted, and
    refused if the callee's slot is not bound
  * driver-direct computation: an evaluator may call a runtime-owned model
    object directly (model(...), model.step(...)). That computation is not a
    spec'd API and does not cross the store. Every forward of every submodule
    of every registered model instance is counted, split by host vs driver,
    so the attestation shows exactly how much of a route's model computation
    the store mediates.

Shared instances
  * frozen models (spec loaders): one runtime-owned instance per pinned
    checkpoint, loaded at install with the global RNG states saved and
    restored around it. The evaluator's first load of that checkpoint
    returns the shared instance; the original loader is still run once at
    that moment and its result is checked bit-exactly against the shared
    instance (then discarded), so loader side effects (file reads, RNG) are
    the same as in G0. Later loads are genuine fresh loads (reload probes),
    registered and reported as private instances.
  * stateless constructors (controller, goal generator): instances are not
    shared (sharing would require overriding __new__, which also captures
    copy/pickle reconstruction); every constructed instance is registered as
    a token, and its configuration is checked against the canonical default.
"""

from __future__ import annotations

import builtins
import collections
import functools
import hashlib
import importlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..channels import ChannelDecl
from ..components import Component, ComponentProvenance, ProvenanceKind, TickContext
from ..slots import DRIVER_OWNER
from .codec import (CodecError, InstanceTable, clear_extra_modules, decode, decode_with_digests, encode,
                    register_extra_module)
from .spec import PROJECTIONS, MechanismSpec, spec_digest


class RouteBlocked(RuntimeError):
    """The composition route cannot run faithfully (unbound host, unencodable value, ...)."""


def request_channel(slot: str) -> str:
    return f"{DRIVER_OWNER}.{slot}.request"


def response_channel(slot: str) -> str:
    return f"{slot}.response"


def _resolve(module: str, qualname: str) -> Any:
    mod = sys.modules.get(module) or importlib.import_module(module)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


_IMMUTABLE = (int, float, bool, str, bytes, complex, type(None), range)


def _identity_bearing(v: Any) -> bool:
    if isinstance(v, _IMMUTABLE) or isinstance(v, torch.nn.Module) or isinstance(v, type):
        return False
    return isinstance(v, (list, dict, torch.Tensor, np.ndarray)) or hasattr(v, "__dict__")


def _contains_object(value: Any, targets: set[int], seen: set[int] | None = None) -> bool:
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if id(value) in targets:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_object(v, targets, seen) for v in value)
    if isinstance(value, dict):
        return any(_contains_object(k, targets, seen) or _contains_object(v, targets, seen) for k, v in value.items())
    if hasattr(value, "__dict__") and not isinstance(value, (type, torch.nn.Module)):
        return any(_contains_object(v, targets, seen) for v in vars(value).values())
    return False


def _mergeable(old: Any, new: Any) -> bool:
    if type(old) is not type(new):
        return False
    if isinstance(old, (list, dict, torch.Tensor, np.ndarray)):
        return True
    return hasattr(old, "__dict__") and not isinstance(old, (type, torch.nn.Module))


class _Merger:
    """Apply a decoded post-state onto the caller's objects in place, keeping nested identities.

    Objects the caller already holds are updated in place (recursively); a decoded object that
    appears twice (an alias inside the post-state) maps to the same caller object both times."""

    def __init__(self):
        self.memo: dict[int, Any] = {}
        self.keep: list = []

    def adopt(self, new: Any) -> Any:
        return self.memo.get(id(new), new)

    def merge(self, target: Any, new: Any) -> Any:
        if id(new) in self.memo:
            return self.memo[id(new)]
        if not _mergeable(target, new):
            raise RouteBlocked(f"post-state type changed: {type(target).__qualname__} -> {type(new).__qualname__}")
        self.memo[id(new)] = target
        self.keep.append(new)
        if isinstance(target, torch.Tensor):
            if target.shape != new.shape or target.dtype != new.dtype:
                raise RouteBlocked("tensor argument changed shape/dtype in place")
            if target.requires_grad != new.requires_grad:
                raise RouteBlocked("tensor argument changed requires_grad in place")
            with torch.no_grad():
                target.copy_(new)
        elif isinstance(target, np.ndarray):
            if target.shape != new.shape or target.dtype != new.dtype:
                raise RouteBlocked("array argument changed shape/dtype in place")
            target[...] = new
        elif isinstance(target, list):
            items = [self.merge(target[i], nv) if i < len(target) and _mergeable(target[i], nv) else self.adopt(nv)
                     for i, nv in enumerate(new)]
            target[:] = items
        elif isinstance(target, dict):
            items = {k: (self.merge(target[k], nv) if k in target and _mergeable(target[k], nv) else self.adopt(nv))
                     for k, nv in new.items()}
            target.clear(); target.update(items)
        else:
            old = vars(target)
            items = {k: (self.merge(old[k], nv) if k in old and _mergeable(old[k], nv) else self.adopt(nv))
                     for k, nv in vars(new).items()}
            target.__dict__.clear(); target.__dict__.update(items)
        return target


@dataclass
class HostStats:
    requests: int = 0
    legacy_errors: int = 0
    blocked: int = 0
    idle_ticks: int = 0
    apis: collections.Counter = field(default_factory=collections.Counter)
    api_response_shas: dict = field(default_factory=dict)       # api -> set of response hashes
    api_ticks: dict = field(default_factory=dict)               # api -> [(tick, response sha256)] (first 64)
    api_first_response: dict = field(default_factory=dict)      # api -> (tick, bytes)
    mutated_args: int = 0

    def as_dict(self) -> dict:
        return {"requests": self.requests, "legacy_errors": self.legacy_errors, "blocked": self.blocked,
                "idle_ticks": self.idle_ticks, "apis": dict(sorted(self.apis.items())),
                "distinct_responses_by_api": {k: len(v) for k, v in sorted(self.api_response_shas.items())},
                "mutated_argument_calls": self.mutated_args}


class MechanismHost(Component):
    """Runtime component that serves one legacy mechanism's API through the canonical store."""

    def __init__(self, spec: MechanismSpec, boundary: "Boundary"):
        self.spec = spec
        self.name = f"legacy-host:{spec.mechanism_id}"
        self.slot = spec.slot
        self.reads = (request_channel(spec.slot),)
        self.writes = (ChannelDecl(response_channel(spec.slot), spec.slot, None, "uint8", intervenable=True,
                                   description=f"encoded responses of {spec.mechanism_id}"),
                       *(ChannelDecl(c.name, spec.slot, c.shape, "float64", neutral=-1.0,
                                     description=f"canonical projection held by {spec.mechanism_id}")
                         for c in spec.canonical))
        self.boundary = boundary
        self.originals: dict[str, Callable] = {}
        self.stats = HostStats()
        self._held = {c.name: np.full(c.shape, -1.0) for c in spec.canonical}

    def provenance(self) -> ComponentProvenance:
        return ComponentProvenance(ProvenanceKind.TEST_FIXTURE, {
            "mechanism_id": self.spec.mechanism_id, "spec_digest": spec_digest((self.spec,)),
            "engineering_only": True,
            "note": "legacy code executed unmodified from the hash-pinned legacy root; binding through the "
                    "verified-manifest path is required before any scientific use"})

    def state_dict(self) -> dict:
        return {"held": {k: np.array(v) for k, v in self._held.items()}}

    def load_state_dict(self, state: dict) -> None:
        held = state.get("held", {})
        if set(held) != set(self._held):
            raise ValueError(f"{self.name}: checkpoint holds canonical channels {sorted(held)}")
        self._held = {k: np.array(v, dtype=np.float64) for k, v in held.items()}

    def _blocked(self, message: str) -> bytes:
        self.stats.blocked += 1
        return encode({"status": "blocked", "message": message}).data

    def _serve(self, raw: bytes) -> tuple[bytes, str | None]:
        b = self.boundary
        try:
            call = decode(raw, b.instances)
            qual, args, kwargs = call["api"], list(call["args"]), dict(call["kwargs"])
        except (CodecError, KeyError, TypeError) as e:
            return self._blocked(f"undecodable request: {e}"), None
        fn = self.originals.get(qual)
        if fn is None:
            return self._blocked(f"{qual!r} is not an API of {self.spec.mechanism_id}"), qual
        self.stats.apis[qual] += 1
        try:
            pre_a = [encode(a, b.instances).data for a in args]
            pre_k = {k: encode(v, b.instances).data for k, v in kwargs.items()}
        except CodecError as e:
            return self._blocked(f"{qual}: {e}"), qual
        try:
            result = fn(*args, **kwargs)
        except RouteBlocked as e:   # e.g. a nested call to an unbound mechanism
            return self._blocked(f"{qual}: {e}"), qual
        except Exception as e:      # legacy exceptions travel back and are re-raised to the caller
            self.stats.legacy_errors += 1
            try:
                ex_args = list(e.args); encode(ex_args, b.instances)
            except CodecError:
                ex_args = [str(e)]
            return encode({"status": "error", "exc_module": type(e).__module__, "exc_type": type(e).__qualname__,
                           "exc_args": ex_args}, b.instances).data, qual
        try:
            post_a = {i: a for i, a in enumerate(args) if encode(a, b.instances).data != pre_a[i]}
            post_k = {k: v for k, v in kwargs.items() if encode(v, b.instances).data != pre_k[k]}
            if post_a or post_k:
                self.stats.mutated_args += 1
            result_ref = None
            if _identity_bearing(result):
                result_ref = next((["arg", i] for i, a in enumerate(args) if result is a), None) or \
                    next((["kwarg", k] for k, v in kwargs.items() if result is v), None)
            if result_ref is None:
                arg_ids = {id(a) for a in (*args, *kwargs.values()) if _identity_bearing(a)}
                arg_ids -= {id(o) for o in b.instances.objects()}
                if arg_ids and _contains_object(result, arg_ids):
                    raise CodecError(f"{qual} returned an object aliasing one of its arguments")
            payload = {"status": "ok", "result": None if result_ref is not None else result,
                       "result_ref": result_ref, "post_args": post_a, "post_kwargs": post_k}
            data = encode(payload, b.instances).data
        except CodecError as e:
            return self._blocked(f"{qual}: {e}"), qual
        for k, v in PROJECTIONS[self.slot](qual, tuple(args), result).items():
            self._held[k] = np.asarray(v, dtype=np.float64)
        return data, qual

    def step(self, ctx: TickContext):
        req = ctx.inputs[request_channel(self.slot)]
        if req.size == 0:
            self.stats.idle_ticks += 1
            return {response_channel(self.slot): b"", **{k: v for k, v in self._held.items()}}
        self.stats.requests += 1
        b = self.boundary
        b.in_host = self.slot
        try:
            data, qual = self._serve(req.tobytes())
        finally:
            b.in_host = None
        if qual is not None:
            sha = hashlib.sha256(data).hexdigest()
            self.stats.api_response_shas.setdefault(qual, set()).add(sha)
            ticks = self.stats.api_ticks.setdefault(qual, [])
            if len(ticks) < 64:
                ticks.append((ctx.tick, sha))
            self.stats.api_first_response.setdefault(qual, (ctx.tick, data))
        return {response_channel(self.slot): data, **{k: v for k, v in self._held.items()}}


class Boundary:
    """Driver side: turns hosted API calls into store round-trips, one runtime tick each."""

    def __init__(self):
        self.runtime = None
        self.instances = InstanceTable()
        self.hosts: dict[str, MechanismHost] = {}
        self.in_host: str | None = None
        self.nested = collections.Counter()
        self.unbound_calls = collections.Counter()
        self.content_flows = collections.Counter()
        self._emitted: dict[str, str] = {}
        self.calls = 0
        self.first_responses: dict[str, tuple[int, bytes]] = {}
        self.forwards = collections.Counter()           # "instance_key|host" / "instance_key|driver"
        self.after_tick: Callable[[int], None] | None = None   # driver hook, called after every tick
        self._hook_handles: list = []
        self._restore: list[Callable[[], None]] = []

    def uninstall(self) -> None:
        """Undo every redirect install() made (module globals, class attributes, torch.load, hooks)."""
        while self._restore:
            self._restore.pop()()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        clear_extra_modules()

    def attach(self, runtime, hosts: dict[str, MechanismHost]) -> None:
        self.runtime = runtime
        self.hosts = hosts

    def watch_model(self, key: str, model: torch.nn.Module) -> None:
        """Count every submodule forward of a registered model, split by host vs driver."""
        def hook(_mod, _inp, key=key):
            self.forwards[f"{key}|{'host:' + self.in_host if self.in_host else 'driver'}"] += 1
        for sub in model.modules():
            self._hook_handles.append(sub.register_forward_pre_hook(hook))

    def call(self, slot: str, qual: str, original: Callable, args: tuple, kwargs: dict) -> Any:
        host = self.hosts.get(slot)
        if host is None:
            self.unbound_calls[f"{slot}:{qual}"] += 1
            raise RouteBlocked(f"{qual} belongs to slot {slot!r}, which is not bound in this composition")
        if self.in_host is not None:
            self.nested[f"{self.in_host}->{slot}:{qual}"] += 1
            return original(*args, **kwargs)
        try:
            enc = encode({"api": qual, "args": list(args), "kwargs": dict(kwargs)}, self.instances)
        except CodecError as e:
            raise RouteBlocked(f"cannot encode call {qual}: {e}") from e
        for d in set(enc.tensor_digests + enc.object_digests):
            src = self._emitted.get(d)
            if src is not None and src != slot:
                self.content_flows[f"{src}->{slot}"] += 1
        drive = {request_channel(s): b"" for s in self.hosts}
        drive[request_channel(slot)] = enc.data
        self.calls += 1
        self.runtime.tick_once(driver=drive)
        if self.after_tick is not None:
            self.after_tick(self.runtime.tick)
        raw = self.runtime.store.read(response_channel(slot), reader=DRIVER_OWNER).tobytes()
        self.first_responses.setdefault(slot, (self.runtime.tick, raw))
        try:
            dec = decode_with_digests(raw, self.instances)
            resp = dec.value
            status = resp["status"]
        except (CodecError, KeyError, TypeError) as e:
            raise RouteBlocked(f"{qual}: response channel does not hold a valid payload ({e})") from e
        if status == "blocked":
            raise RouteBlocked(f"{qual}: {resp.get('message')}")
        required = {"ok": ("result", "result_ref", "post_args", "post_kwargs"),
                    "error": ("exc_module", "exc_type", "exc_args")}.get(status)
        if required is None or not isinstance(resp, dict) or any(k not in resp for k in required):
            raise RouteBlocked(f"{qual}: malformed response payload (status {status!r})")
        if status == "error":
            exc_t = getattr(builtins, resp["exc_type"], None) if resp["exc_module"] == "builtins" else None
            if not (isinstance(exc_t, type) and issubclass(exc_t, BaseException)):
                try:
                    exc_t = _resolve(resp["exc_module"], resp["exc_type"])
                except Exception:
                    exc_t = RuntimeError
            raise exc_t(*resp["exc_args"])
        # provenance of content: digests of what the driver actually read (after any store intervention)
        for d in dec.tensor_digests + dec.object_digests:
            self._emitted.setdefault(d, slot)
        merger = _Merger()
        for i, v in resp["post_args"].items():
            merger.merge(args[i], v)
        for k, v in resp["post_kwargs"].items():
            merger.merge(kwargs[k], v)
        ref = resp["result_ref"]
        if ref is not None:
            return args[ref[1]] if ref[0] == "arg" else kwargs[ref[1]]
        return resp["result"]


@dataclass
class Census:
    shared_acquisitions: collections.Counter = field(default_factory=collections.Counter)
    acquisition_verifications: list = field(default_factory=list)
    private_instances: list = field(default_factory=list)
    binding_loads: list = field(default_factory=list)
    constructed_instances: list = field(default_factory=list)
    torch_loads: collections.Counter = field(default_factory=collections.Counter)
    patched: list = field(default_factory=list)

    def as_dict(self) -> dict:
        kinds = collections.Counter(p[2].split(":", 1)[0] for p in self.patched)
        return {"shared_acquisitions": dict(sorted(self.shared_acquisitions.items())),
                "acquisition_verifications": list(self.acquisition_verifications),
                "private_instances": list(self.private_instances), "binding_loads": list(self.binding_loads),
                "constructed_instances": list(self.constructed_instances),
                "torch_load_paths": dict(sorted(self.torch_loads.items())),
                "patched_bindings_by_kind": dict(sorted(kinds.items()))}


_EXTRA_MODULES: list = []     # modules deliberately kept out of sys.modules (the frozen evaluator)


def _replace_everywhere(original: Any, replacement: Any, census: Census, label: str,
                        restore: list | None = None) -> None:
    """Rebind every module-global name that refers to `original` (covers `from x import f`).

    The frozen evaluator is loaded exactly as in G0 (not registered in sys.modules, so pickling
    behaviour is unchanged); it is passed in through _EXTRA_MODULES."""
    for mname, mod in [*list(sys.modules.items()), *[(getattr(m, "__name__", "?"), m) for m in _EXTRA_MODULES]]:
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for k, v in list(d.items()):
            if v is original:
                d[k] = replacement
                census.patched.append((mname, k, label))
                if restore is not None:
                    restore.append(lambda d=d, k=k, v=v: d.__setitem__(k, v))


def _rng_states() -> tuple:
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def _set_rng_states(states: tuple) -> None:
    torch.set_rng_state(states[0]); np.random.set_state(states[1]); random.setstate(states[2])


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    if list(sa) != list(sb):
        return False
    return all(sa[k].dtype == sb[k].dtype and sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]) for k in sa)


def install(boundary: Boundary, specs: tuple[MechanismSpec, ...], bound: set[str], legacy_root: Path,
            census: Census, extra_modules: tuple = ()) -> dict[str, MechanismHost]:
    """Build hosts for `bound` mechanisms, load their shared model instances, and redirect every spec'd API.

    APIs of mechanisms that are not bound are redirected too, so a call to them is reported
    (RouteBlocked) instead of silently running a private copy."""
    hosts: dict[str, MechanismHost] = {}
    _EXTRA_MODULES[:] = list(extra_modules)
    for m in extra_modules:
        register_extra_module(m)
    R = boundary._restore
    real_torch_load = torch.load

    @functools.wraps(real_torch_load)
    def counting_load(f, *a, **k):
        try:
            key = str(Path(f).resolve().relative_to(legacy_root)) if isinstance(f, (str, Path)) else "<stream>"
        except ValueError:
            key = str(f)
        census.torch_loads[key] += 1
        return real_torch_load(f, *a, **k)
    torch.load = counting_load
    R.append(lambda: setattr(torch, "load", real_torch_load))
    _replace_everywhere(real_torch_load, counting_load, census, "torch.load", R)

    for spec in specs:
        host = MechanismHost(spec, boundary) if spec.mechanism_id in bound else None
        if host is not None:
            hosts[spec.slot] = host
        for ld in spec.loaders:
            orig = _resolve(ld.module, ld.name)
            shared: dict[tuple, Any] = {}
            if host is not None:
                saved = _rng_states()
                for rel in ld.paths:
                    out = orig(rel, "cpu")
                    model = out[0] if isinstance(out, tuple) else out
                    key = f"{ld.instance_prefix}:shared"
                    boundary.instances.register(key, model)
                    boundary.watch_model(key, model)
                    shared[(str((legacy_root / rel).resolve()), "cpu")] = out
                    census.binding_loads.append({"slot": spec.slot, "path": rel, "instance": key})
                _set_rng_states(saved)        # binding loads leave no trace in the evaluator's RNG streams
            acquired: set = set()

            def make_loader(orig=orig, shared=shared, acquired=acquired, ld=ld, spec=spec, host=host):
                @functools.wraps(orig)
                def loader(path, device="cpu", *a, **k):
                    key = (str(Path(path).resolve()), str(device))
                    out = orig(path, device, *a, **k)            # same side effects as the G0 route
                    model = out[0] if isinstance(out, tuple) else out
                    if key in shared and key not in acquired and not a and not k:
                        acquired.add(key)
                        sh = shared[key]
                        sh_model = sh[0] if isinstance(sh, tuple) else sh
                        same = _same_weights(sh_model, model)
                        census.acquisition_verifications.append({"slot": spec.slot, "loader": ld.name,
                                                                 "path": str(path), "bit_identical": same})
                        if not same:
                            raise RouteBlocked(f"{ld.name}({path}): fresh load differs from the shared instance")
                        census.shared_acquisitions[f"{spec.slot}:{ld.name}"] += 1
                        return (sh_model, *out[1:]) if isinstance(out, tuple) else sh_model
                    reason = ("unhosted_mechanism" if host is None else
                              "reload_probe" if key in acquired else "other_path")
                    n = len(census.private_instances)
                    census.private_instances.append({"slot": spec.slot, "loader": ld.name, "path": str(path),
                                                     "reason": reason})
                    if host is not None:
                        pkey = f"{ld.instance_prefix}:private{n}"
                        boundary.instances.register(pkey, model)
                        boundary.watch_model(pkey, model)
                    return out
                return loader
            _replace_everywhere(orig, make_loader(), census, f"loader:{ld.name}", R)
        for ct in spec.constructors:
            cls = _resolve(ct.module, ct.qualname)
            orig_init = cls.__init__
            reference = vars(cls()) if host is not None else None     # canonical default configuration

            def make_init(cls=cls, orig_init=orig_init, ct=ct, spec=spec, host=host, reference=reference):
                @functools.wraps(orig_init)
                def __init__(self, *a, **k):
                    orig_init(self, *a, **k)
                    if host is None or type(self) is not cls:
                        return
                    n = sum(1 for c in census.constructed_instances if c["constructor"] == ct.qualname)
                    key = f"{ct.instance_key}:{n}"
                    boundary.instances.register(key, self)
                    census.constructed_instances.append({"slot": spec.slot, "constructor": ct.qualname,
                                                         "instance": key, "default_config": vars(self) == reference})
                return __init__
            R.append(lambda cls=cls, orig_init=orig_init: setattr(cls, "__init__", orig_init))
            cls.__init__ = make_init()
        for api in spec.apis:
            orig = _resolve(api.module, api.qualname)
            if host is not None:
                host.originals[api.qualname] = orig
            if api.is_method:
                cls_name, meth = api.qualname.split(".", 1)
                cls = _resolve(api.module, cls_name)

                def make_method(orig=orig, slot=spec.slot, qual=api.qualname):
                    @functools.wraps(orig)
                    def stub(self, *a, **k):
                        return boundary.call(slot, qual, orig, (self, *a), k)
                    del stub.__wrapped__
                    return stub
                prev = cls.__dict__[meth]
                R.append(lambda cls=cls, meth=meth, prev=prev: setattr(cls, meth, prev))
                setattr(cls, meth, make_method())
                census.patched.append((api.module, api.qualname, "api-method"))
            else:
                def make_fn(orig=orig, slot=spec.slot, qual=api.qualname):
                    @functools.wraps(orig)
                    def stub(*a, **k):
                        return boundary.call(slot, qual, orig, a, k)
                    del stub.__wrapped__
                    return stub
                _replace_everywhere(orig, make_fn(), census, f"api:{api.qualname}", R)
    return hosts
