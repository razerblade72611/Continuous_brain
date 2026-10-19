"""The Unified Individual runtime: composition engine for the fifteen required subsystems.

What it does
  * owns the clock: fixed dt, integer ticks; nothing else advances time
  * holds the channel store (single writer per channel, write log) and the
    per-tick DAG schedule of bound components
  * binds a component into a mechanism slot only if its provenance passes the
    binding policy; every other mechanism slot is reported UNBOUND with a
    reason. It never substitutes a stand-in (C1)
  * provides the three infrastructure slots itself: persistence (checkpoints
    through the never-overwrite SnapshotStore), provenance (config hash,
    binding report, logs), and the coprocessor interface (disabled at boot)
  * applies declared interventions through the InterventionManager
  * gives evaluators a read-only view

Binding policies
  VERIFIED_ONLY  (default) LEGACY_VERIFIED and EXPERIMENTAL_PREREG components
  TEST_FIXTURES  additionally accepts TEST_FIXTURE components, and marks the
                 runtime `scientific = False` in every report and checkpoint
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from continuous_brain_protocols import StreamSpec, sha256_json, state_digest
from continuous_brain_snn.provenance import code_revision, environment
from continuous_brain_snn.snapshot import SnapshotStore

from . import __version__
from .channels import ChannelDecl, ChannelStore, value_sha256
from .components import Component, ProvenanceKind, TickContext
from .coprocessor import Coprocessor, CoprocessorClient
from .interventions import InterventionManager
from .scheduler import schedule
from .slots import ENVIRONMENT_OWNER, REQUIRED_SLOTS, SLOTS_BY_NAME, SlotKind, slot
from .legacy.adapters import issued_verification
from .workspace import registered_prereg

RUNTIME_FORMAT = "cb-unified-runtime/1"


class BindingPolicy(str, Enum):
    VERIFIED_ONLY = "verified_only"
    TEST_FIXTURES = "test_fixtures"


class BindingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    dt_s: float = 0.01
    seed: int = 0
    binding_policy: BindingPolicy = BindingPolicy.VERIFIED_ONLY
    coprocessor_enabled: bool = False
    max_coprocessor_calls_per_tick: int = 1
    observation: StreamSpec | None = None      # physical channels written by the environment each tick

    def config_hash(self) -> str:
        d = {"dt_s": self.dt_s, "seed": self.seed, "binding_policy": self.binding_policy.value,
             "coprocessor_enabled": self.coprocessor_enabled,
             "max_coprocessor_calls_per_tick": self.max_coprocessor_calls_per_tick,
             "observation": None if self.observation is None else self.observation.spec_hash()}
        return sha256_json(d)


@dataclass
class SlotReport:
    slot: str
    kind: str
    status: str                 # "bound" | "unbound" | "infrastructure"
    component: str | None = None
    provenance: dict | None = None
    reason: str = ""


@dataclass
class TickResult:
    tick: int
    time_s: float
    order: list[str]
    writes: dict[str, str] = field(default_factory=dict)       # channel -> sha256 after interventions
    interventions: int = 0


class ReadOnlyRuntimeView:
    """What evaluators get: reads, logs, and reports; no write or intervention methods.

    The runtime is captured in closures rather than stored as an attribute. Python has no real
    access control, so this guards against accidental writes, not against deliberate subversion.
    """

    __slots__ = ("_get",)

    def __init__(self, runtime: "UnifiedRuntime"):
        store, interventions, cop = runtime.store, runtime.interventions, runtime.coprocessor
        getters = {
            "tick": lambda: runtime.tick, "time_s": lambda: runtime.time_s, "scientific": lambda: runtime.scientific,
            "read": lambda ch, delayed=False: store.read(ch, delayed), "channels": lambda: store.names(),
            "write_log": lambda: [dict(e) for e in store.log], "binding_report": lambda: runtime.binding_report(),
            "intervention_records": lambda: [asdict(r) for r in interventions.records],
            "coprocessor_log": lambda: [dict(e) for e in cop.log.entries]}
        object.__setattr__(self, "_get", getters.__getitem__)

    def __setattr__(self, key, value):
        raise AttributeError("the evaluator view is read-only")

    @property
    def tick(self) -> int:
        return self._get("tick")()

    @property
    def time_s(self) -> float:
        return self._get("time_s")()

    @property
    def scientific(self) -> bool:
        return self._get("scientific")()

    def read(self, channel: str, delayed: bool = False) -> np.ndarray:
        return self._get("read")(channel, delayed)

    def channels(self) -> tuple[str, ...]:
        return self._get("channels")()

    def write_log(self) -> list[dict]:
        return self._get("write_log")()

    def binding_report(self) -> list[SlotReport]:
        return self._get("binding_report")()

    def intervention_records(self) -> list[dict]:
        return self._get("intervention_records")()

    def coprocessor_log(self) -> list[dict]:
        return self._get("coprocessor_log")()


class UnifiedRuntime:
    def __init__(self, config: RuntimeConfig | None = None, coprocessor: Coprocessor | None = None,
                 log_dir: str | os.PathLike | None = None):
        self.config = config or RuntimeConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        self.store = ChannelStore(self.log_dir / "channel_writes.jsonl" if self.log_dir else None)
        self.interventions = InterventionManager(self.store)
        self.coprocessor = CoprocessorClient(coprocessor, enabled=self.config.coprocessor_enabled,
                                             max_calls_per_tick=self.config.max_coprocessor_calls_per_tick,
                                             log_path=self.log_dir / "coprocessor.jsonl" if self.log_dir else None)
        self.components: dict[str, Component] = {}           # slot -> component
        self._unbound_reasons: dict[str, str] = {s.name: "no verified mechanism bound (DAV-74 pending)"
                                                 for s in REQUIRED_SLOTS if s.kind == SlotKind.MECHANISM}
        self._order: list[Component] = []
        self._observation_channels: set[str] = set()
        self.tick = -1
        self.dev_overrides: list[dict] = []
        if self.config.observation is not None:
            for c in self.config.observation.channels:
                name = f"{ENVIRONMENT_OWNER}.{c.name}"
                self.store.declare(ChannelDecl(name, ENVIRONMENT_OWNER, tuple(c.shape), "float64",
                                               description=f"{c.units} ({c.kind.value})"))
                self._observation_channels.add(name)

    # ------------------------------------------------------------------ status
    @property
    def time_s(self) -> float:
        return max(self.tick, 0) * self.config.dt_s

    @property
    def scientific(self) -> bool:
        if self.config.binding_policy != BindingPolicy.VERIFIED_ONLY or self.dev_overrides:
            return False
        return all(c.provenance().kind != ProvenanceKind.TEST_FIXTURE for c in self.components.values())

    def view(self) -> ReadOnlyRuntimeView:
        return ReadOnlyRuntimeView(self)

    # ------------------------------------------------------------------ binding
    def _check_provenance(self, component: Component) -> dict | None:
        """Raise unless the component's provenance passes the policy; return a dev-override record, if any."""
        prov = component.provenance()
        if prov.kind == ProvenanceKind.TEST_FIXTURE:
            if self.config.binding_policy != BindingPolicy.TEST_FIXTURES:
                raise BindingError(f"{component.name} is a test fixture; this runtime only binds verified mechanisms")
            return None
        if prov.kind == ProvenanceKind.LEGACY_VERIFIED:
            issued = issued_verification(component)
            if issued is None:
                raise BindingError(f"{component.name} claims legacy provenance but was not built by "
                                   f"build_legacy_components from a verified manifest entry")
            digest, result = issued
            if prov.detail.get("entry_digest") != digest or result.entry_digest != digest:
                raise BindingError(f"{component.name}: provenance does not match the verified manifest entry")
            if result.overridden:
                return {"component": component.name, "mechanism_id": result.mechanism_id}
            return None
        if prov.kind == ProvenanceKind.EXPERIMENTAL_PREREG:
            prereg = registered_prereg(type(component))
            if prereg is None or prereg.id != prov.detail.get("prereg_id"):
                raise BindingError(f"{component.name} is not registered under preregistration "
                                   f"{prov.detail.get('prereg_id')!r}")
            if not prereg.unlocks or component.slot not in prereg.allowed_slots:
                raise BindingError(f"{prereg.id} does not admit {component.name} into slot {component.slot}")
            return None
        raise BindingError(f"unknown provenance kind {prov.kind}")

    def bind(self, component: Component) -> None:
        self.bind_many([component])

    def bind_many(self, components: list[Component]) -> None:
        """Bind a set atomically: provenance, slot rules, and the schedule are checked for the whole set,
        so components that read each other through delayed edges can be bound together."""
        if self.tick >= 0:
            raise BindingError("components are bound before the first tick")
        seen: set[str] = set()
        overrides: list[dict] = []
        for component in components:
            spec = slot(component.slot)
            if spec.kind != SlotKind.MECHANISM:
                raise BindingError(f"{spec.name} is an infrastructure slot provided by the runtime")
            if spec.name in self.components or spec.name in seen:
                bound = self.components.get(spec.name)
                raise BindingError(f"slot {spec.name} already bound" + (f" to {bound.name}" if bound else ""))
            seen.add(spec.name)
            override = self._check_provenance(component)
            if override:
                overrides.append(override)
            written = {d.name for d in component.writes}
            missing = [c for c in spec.canonical_writes if c not in written]
            if missing:
                raise BindingError(f"{component.name} must write the canonical channels {missing}")
            for d in component.writes:
                if d.owner != spec.name:
                    raise BindingError(f"{component.name} declares {d.name} with owner {d.owner!r}, not {spec.name!r}")
        declared: list[str] = []
        try:
            for component in components:
                for d in component.writes:
                    self.store.declare(d)
                declared.append(component.slot)
            order = schedule(list(self.components.values()) + list(components), self._observation_channels)
        except Exception:
            for owner in declared:
                self.store.undeclare_owner(owner)
            raise
        for component in components:
            self.components[component.slot] = component
            self._unbound_reasons.pop(component.slot, None)
        self._order = order
        self.dev_overrides.extend(overrides)

    def mark_unbound(self, slot_name: str, reason: str) -> None:
        if slot_name in self.components:
            raise BindingError(f"{slot_name} is bound")
        if SLOTS_BY_NAME[slot_name].kind != SlotKind.MECHANISM:
            raise BindingError(f"{slot_name} is not a mechanism slot")
        self._unbound_reasons[slot_name] = reason

    def binding_report(self) -> list[SlotReport]:
        out = []
        for s in REQUIRED_SLOTS:
            if s.kind == SlotKind.INFRASTRUCTURE:
                detail = {"persistence": "SnapshotStore (never-overwrite, SHA-256 checksums)",
                          "provenance": "config hash, binding report, write log, run registry",
                          "coprocessor": f"{self.coprocessor.provider.name} "
                                         f"({'enabled' if self.coprocessor.enabled else 'disabled'})"}[s.name]
                out.append(SlotReport(s.name, s.kind.value, "infrastructure", "runtime", None, detail))
            elif s.name in self.components:
                c = self.components[s.name]
                out.append(SlotReport(s.name, s.kind.value, "bound", c.name, c.provenance().as_dict()))
            else:
                out.append(SlotReport(s.name, s.kind.value, "unbound", None, None, self._unbound_reasons[s.name]))
        return out

    # ------------------------------------------------------------------ running
    def tick_once(self, observation: Mapping[str, np.ndarray] | None = None) -> TickResult:
        if self.tick < 0:
            self.interventions.lock()
        tick = self.tick + 1
        self.store.begin_tick(tick)
        self.coprocessor.begin_tick(tick)
        self.tick = tick
        result = TickResult(tick, self.time_s, [c.name for c in self._order])
        n_before = len(self.interventions.records)
        spec = self.config.observation
        if spec is not None:
            if observation is None:
                raise ValueError("this runtime declares observation channels; pass an observation each tick")
            problems = spec.validate(observation)
            if problems:
                raise ValueError("invalid observation: " + "; ".join(problems))
            for c in spec.channels:
                name = f"{ENVIRONMENT_OWNER}.{c.name}"
                self.store.write(name, observation[c.name], ENVIRONMENT_OWNER)
                self.interventions.apply(name, tick)
                result.writes[name] = value_sha256(self.store.read(name))
        elif observation:
            raise ValueError("this runtime has no observation channels")
        for comp in self._order:
            ctx = TickContext(tick, self.time_s, self.config.dt_s,
                              {ch: self.store.read(ch) for ch in comp.reads},
                              {ch: self.store.read(ch, delayed=True) for ch in comp.reads_delayed},
                              self.coprocessor.handle() if comp.uses_coprocessor else None)
            outputs = dict(comp.step(ctx))
            declared = {d.name for d in comp.writes}
            if set(outputs) != declared:
                raise BindingError(f"{comp.name} returned {sorted(outputs)}, declared {sorted(declared)}")
            for d in comp.writes:
                self.store.write(d.name, outputs[d.name], comp.slot)
                self.interventions.apply(d.name, tick)
                result.writes[d.name] = value_sha256(self.store.read(d.name))
        result.interventions = len(self.interventions.records) - n_before
        return result

    def run(self, n_ticks: int, observations=None) -> list[TickResult]:
        """observations: None, or a callable tick -> observation mapping."""
        return [self.tick_once(None if observations is None else observations(self.tick + 1))
                for _ in range(n_ticks)]

    # ------------------------------------------------------------------ persistence (infrastructure slot)
    def identity(self) -> dict:
        return {"format": RUNTIME_FORMAT, "version": __version__, "config_hash": self.config.config_hash(),
                "binding": [asdict(r) for r in self.binding_report()],
                "channels": self.store.names(), "scientific": self.scientific,
                "interventions": [s.as_dict() for s in self.interventions.specs],
                "intervention_payloads": self.interventions.payload_digest()}

    def state_dict(self) -> dict:
        return {"format": RUNTIME_FORMAT, "tick": self.tick, "identity": self.identity(),
                "store": self.store.state_dict(),
                "components": {s: c.state_dict() for s, c in self.components.items()},
                "interventions": self.interventions.state_dict(),
                "coprocessor": self.coprocessor.state_dict(), "dev_overrides": list(self.dev_overrides)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("format") != RUNTIME_FORMAT:
            raise ValueError(f"not a {RUNTIME_FORMAT} state")
        mine, theirs = self.identity(), state["identity"]
        for key in ("config_hash", "binding", "channels", "interventions", "intervention_payloads"):
            if mine[key] != theirs[key]:
                raise BindingError(f"checkpoint {key} differs from this runtime's; refusing to restore")
        self.store.load_state_dict(state["store"])
        for s, c in self.components.items():
            c.load_state_dict(state["components"][s])
        self.interventions.load_state_dict(state["interventions"])
        self.coprocessor.load_state_dict(state["coprocessor"])
        self.dev_overrides = list(state["dev_overrides"])
        self.tick = int(state["tick"])

    def state_hash(self) -> str:
        return state_digest({"tick": self.tick, "store": self.store.state_dict(),
                             "components": {s: c.state_dict() for s, c in self.components.items()},
                             "interventions": self.interventions.state_dict(),
                             "coprocessor": self.coprocessor.state_dict()})

    def save(self, store: SnapshotStore, label: str = "unified") -> Path:
        meta = {"tick": self.tick, "config_hash": self.config.config_hash(), "scientific": self.scientific,
                "state_hash": self.state_hash()}
        manifest = {**self.identity(), "tick": self.tick, "state_hash": meta["state_hash"],
                    "code": code_revision(), "env": environment()}
        return store.save_state({"runtime": self.state_dict(), "manifest": manifest}, label=label, meta=meta)

    def restore(self, store: SnapshotStore, snapshot: str | os.PathLike) -> None:
        """Restore from a verified snapshot; on any failure the runtime is left exactly as it was."""
        manifest = store.load_state(snapshot, "manifest")
        state = store.load_state(snapshot, "runtime")
        backup = self.state_dict()
        try:
            self.load_state_dict(state)
            if self.state_hash() != manifest["state_hash"]:
                raise BindingError("restored runtime state does not reproduce its recorded hash")
        except Exception:
            self.load_state_dict(backup)
            raise
