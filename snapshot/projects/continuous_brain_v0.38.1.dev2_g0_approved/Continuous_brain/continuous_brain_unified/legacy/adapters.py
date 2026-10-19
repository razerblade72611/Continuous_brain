"""Binding verified legacy mechanisms into slots.

A legacy mechanism is bound only when three things exist:
  1. a VERIFIED manifest entry whose files re-hash correctly under the root,
  2. a *step contract* registered for exactly that mechanism id and module
     SHA-256: the audited translation between the runtime's channels and the
     legacy object's API (which inputs it reads, what it writes each tick),
  3. the legacy module imports from the verified root.

Step contracts are written as part of the DAV-74 audit, one mechanism at a
time, against the canonical tree. v0.38.0.dev0 registers none: the audit has
not produced a VERIFIED entry yet, and the source mirror has no checkpoints to
load. So every mechanism slot reports UNBOUND with the reason, which is the
honest state (Joint Contract §1: "must not claim that the historical
mechanisms have already been integrated").
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..components import Component, ComponentProvenance, ProvenanceKind
from .loader import import_verified_module
from .manifest import LegacyManifest, ManifestEntry, VerificationResult, root_for_entry, verify_entry

StepContractFactory = Callable[[ManifestEntry, object, Path, VerificationResult], Component]


@dataclass(frozen=True)
class StepContract:
    mechanism_id: str
    module_sha256: str
    factory: StepContractFactory
    audit_ref: str


_CONTRACTS: dict[tuple[str, str], StepContract] = {}
# Components built by build_legacy_components, keyed weakly by object: the runtime accepts LEGACY_VERIFIED
# provenance only from a component issued here, so provenance cannot be claimed by constructing a dict.
_ISSUED: "weakref.WeakKeyDictionary[Component, tuple[str, VerificationResult]]" = weakref.WeakKeyDictionary()


def issued_verification(component: Component) -> tuple[str, VerificationResult] | None:
    """(entry digest, verification) for a component this module built, else None."""
    return _ISSUED.get(component)


def register_step_contract(contract: StepContract) -> None:
    key = (contract.mechanism_id, contract.module_sha256)
    if key in _CONTRACTS:
        raise ValueError(f"step contract already registered for {key}")
    if not contract.audit_ref:
        raise ValueError("a step contract needs its DAV-74 audit reference")
    _CONTRACTS[key] = contract


def registered_contracts() -> tuple[StepContract, ...]:
    return tuple(_CONTRACTS.values())


def legacy_provenance(entry: ManifestEntry, result: VerificationResult, contract: StepContract) -> ComponentProvenance:
    return ComponentProvenance(ProvenanceKind.LEGACY_VERIFIED, {
        "mechanism_id": entry.mechanism_id, "entry_digest": entry.digest(), "lineage": entry.lineage_version,
        "module_sha256": entry.file("module").sha256, "overridden": result.overridden,
        "audit_ref": entry.audit_ref, "contract_audit_ref": contract.audit_ref})


@dataclass
class BindingOutcome:
    mechanism_id: str
    slot: str
    bound: bool
    reason: str
    component: Component | None = None
    verification: VerificationResult | None = None


def build_legacy_components(manifest: LegacyManifest, root, dev_override: bool = False,
                            override_log: str | os.PathLike | None = None) -> list[BindingOutcome]:
    outcomes = []
    for entry in manifest.entries:
        entry_root = root_for_entry(entry, root)
        result = verify_entry(entry, entry_root, dev_override, override_log)
        contract = _CONTRACTS.get((entry.mechanism_id, entry.file("module").sha256))
        if contract is None:
            outcomes.append(BindingOutcome(entry.mechanism_id, entry.slot, False,
                                           f"no audited step contract for {entry.mechanism_id}@"
                                           f"{entry.file('module').sha256[:12]}", None, result))
            continue
        module = import_verified_module(entry, entry_root)
        component = contract.factory(entry, module, entry_root, result)
        if component.slot != entry.slot:
            raise ValueError(f"step contract for {entry.mechanism_id} built a component for slot {component.slot}")
        _ISSUED[component] = (entry.digest(), result)
        outcomes.append(BindingOutcome(entry.mechanism_id, entry.slot, True, "verified", component, result))
    return outcomes
