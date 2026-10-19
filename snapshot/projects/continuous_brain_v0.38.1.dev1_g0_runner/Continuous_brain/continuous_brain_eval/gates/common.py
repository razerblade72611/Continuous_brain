"""Shared pieces of the Unified Individual integration-gate harness (G1-G5).

The harness is first-class code (Joint Contract §5), but the gate itself is a
DRAFT (UI-GATE-v0): thresholds, margins, the G2 sign matrix, and run lengths
are not frozen. Until they are, and until DAV-74 binds verified mechanisms,
every gate result is BLOCKED (nothing to test) or ENGINEERING_ONLY.

Evaluators are read-only: they receive a ReadOnlyRuntimeView, and may change
the individual only through interventions declared before the run.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol

import numpy as np

from continuous_brain_protocols import ResultStatus
from continuous_brain_snn.provenance import code_revision, environment
from continuous_brain_unified import ReadOnlyRuntimeView, UnifiedRuntime
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry

from ..results import ResultRecord

RuntimeFactory = Callable[[int], UnifiedRuntime]           # seed -> fresh, fully bound runtime
ObservationFn = Callable[[int, int], Mapping[str, np.ndarray] | None]   # (seed, tick) -> observation
MetricFn = Callable[[ReadOnlyRuntimeView], float]


class MechanismEvaluator(Protocol):
    """A mechanism's own frozen evaluator, re-hosted inside the unified runtime (G1)."""
    mechanism_id: str
    slot: str

    def evaluate(self, view: ReadOnlyRuntimeView, seed: int) -> dict[str, float]: ...


def prereg_unlocks(prereg_id: str | None, item: str | None = None) -> bool:
    """True iff the registry says prereg_id is APPROVED/FROZEN and covers this result item."""
    if not prereg_id:
        return False
    try:
        PreregistrationRegistry.load().require(prereg_id, item=item)
        return True
    except PreregistrationError:
        return False


def propose_verdict(rec: ResultRecord, scientific: bool, prereg_id: str | None, passed: bool) -> None:
    """Record a gate verdict. It becomes the status (PASS/FAIL) only in RunStore.finalize, after the run's
    primary artifacts are attached and the record is validated against the preregistration registry."""
    if scientific and prereg_unlocks(prereg_id, rec.item):
        rec.status = ResultStatus.UNVERIFIED
        rec.metrics["verdict"] = "PASS" if passed else "FAIL"
        rec.notes.append("verdict pending: becomes the status only when finalized with primary artifacts")


def check_seeds(seeds, registry=None) -> None:
    from ..seeds import SeedRegistry
    (registry or SeedRegistry.load()).check_seeds(list(seeds))


def gate_record(item: str, runtime: UnifiedRuntime | None, prereg_id: str | None, **metrics) -> ResultRecord:
    prov = {"code_revision": code_revision(), "environment": environment(), "artifacts": {}}
    scientific = False
    if runtime is not None:
        prov["runtime"] = runtime.identity()
        scientific = runtime.scientific
    unlocked = prereg_unlocks(prereg_id, item)
    rec = ResultRecord(item=item, status=ResultStatus.ENGINEERING_ONLY,
                       metric_status=ResultStatus.UNREGISTERED_METRIC if not unlocked else ResultStatus.UNVERIFIED,
                       substrate="unified-runtime", metrics=dict(metrics), provenance=prov, prereg_id=prereg_id,
                       scientific_context=scientific)
    if not scientific:
        rec.notes.append("runtime is non-scientific (test fixtures or developer overrides): engineering validation only")
    if not unlocked:
        rec.notes.append(f"preregistration {prereg_id or '(none)'} is not APPROVED: no PASS/FAIL is emitted")
    return rec


def bound_mechanisms(runtime: UnifiedRuntime) -> list[str]:
    return sorted(runtime.components)


def run_ticks(runtime: UnifiedRuntime, n_ticks: int, observe: ObservationFn | None, seed: int) -> None:
    runtime.run(n_ticks, None if observe is None else (lambda t: observe(seed, t)))
