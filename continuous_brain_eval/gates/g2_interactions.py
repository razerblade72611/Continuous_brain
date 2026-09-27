"""G2 Predicted interactions: an intervention x mechanism matrix with predicted signs.

The matrix is filled and frozen (hashed) before any run; editing it afterwards
raises. Each cell compares paired control/intervened runs over seeds:

  "+"       the metric must increase: CI lower bound > 0
  "-"       the metric must decrease: CI upper bound < 0
  "change"  any significant change: CI excludes 0
  "0"       non-interference: CI inside [-margin, +margin] (equivalence)

The six interventions named in the integration-gate draft target canonical
slot channels (STANDARD_INTERVENTIONS). A cell whose target channel does not
exist in the runtime (its slot is unbound) is reported BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from continuous_brain_protocols import InterventionKind, InterventionSpec, ResultStatus, sha256_json

from ..stats import paired_difference_ci
from .common import MetricFn, ObservationFn, RuntimeFactory, check_seeds, gate_record, propose_verdict, run_ticks

SIGNS = ("+", "-", "change", "0")


@dataclass(frozen=True)
class NamedIntervention:
    name: str
    kind: InterventionKind
    target: str
    payload_ref: str | None = None


STANDARD_INTERVENTIONS = (
    NamedIntervention("erase_autobiography", InterventionKind.ERASE, "autobiography.record"),
    NamedIntervention("donor_self_state", InterventionKind.DONOR, "self_state.vector", "donor:self_state.vector"),
    NamedIntervention("frozen_autobiography", InterventionKind.FREEZE, "autobiography.record"),
    NamedIntervention("stale_plan", InterventionKind.FREEZE, "planner.plan"),
    NamedIntervention("zeroed_workspace", InterventionKind.ERASE, "workspace.state"),
    NamedIntervention("erased_episodic_slot", InterventionKind.ERASE, "episodic_memory.slots"),
)


@dataclass
class InteractionMatrix:
    interventions: tuple[NamedIntervention, ...]
    metrics: dict[str, MetricFn]
    cells: dict[tuple[str, str], str]              # (intervention name, metric name) -> sign
    equivalence_margin: float
    start_tick: int
    frozen_hash: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        names = {i.name for i in self.interventions}
        for (iv, m), sign in self.cells.items():
            if iv not in names or m not in self.metrics:
                raise ValueError(f"cell ({iv}, {m}) names an unknown intervention or metric")
            if sign not in SIGNS:
                raise ValueError(f"cell ({iv}, {m}): sign must be one of {SIGNS}")

    def describe(self) -> dict:
        return {"interventions": [(i.name, i.kind.value, i.target, i.payload_ref) for i in self.interventions],
                "metrics": {k: _source_digest(f) for k, f in sorted(self.metrics.items())}, "cells": sorted((k[0], k[1], v) for k, v in self.cells.items()),
                "equivalence_margin": self.equivalence_margin, "start_tick": self.start_tick}

    def freeze(self) -> str:
        self.frozen_hash = sha256_json(self.describe())
        return self.frozen_hash

    def check_frozen(self) -> None:
        if self.frozen_hash is None:
            raise RuntimeError("freeze the matrix before running G2")
        if sha256_json(self.describe()) != self.frozen_hash:
            raise RuntimeError("the matrix changed after it was frozen")


def _source_digest(fn) -> str:
    """Hash of a metric's source (and its bound constants), so a metric cannot be swapped after freezing."""
    import inspect
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    consts = getattr(getattr(fn, "__code__", None), "co_consts", ())
    return sha256_json({"source": src, "consts": [repr(c) for c in consts]})


def _verdict(sign: str, ci, margin: float) -> bool:
    if sign == "+":
        return ci.low > 0
    if sign == "-":
        return ci.high < 0
    if sign == "change":
        return ci.excludes_zero()
    return ci.within(margin)


def run_g2(factory: RuntimeFactory, matrix: InteractionMatrix, seeds: list[int], ticks: int,
           observe: ObservationFn | None = None, payloads: Callable[[int], dict] | None = None,
           prereg_id: str | None = None, n_resamples: int = 10_000, seed_registry=None):
    matrix.check_frozen()
    check_seeds(seeds, seed_registry)
    probe = factory(seeds[0])
    control: dict[str, list[float]] = {m: [] for m in matrix.metrics}
    for s in seeds:
        rt = factory(s)
        run_ticks(rt, ticks, observe, s)
        for m, fn in matrix.metrics.items():
            control[m].append(float(fn(rt.view())))
    cells = []
    for iv in matrix.interventions:
        if iv.target not in probe.store.names():
            for (name, m), sign in matrix.cells.items():
                if name == iv.name:
                    cells.append({"intervention": name, "metric": m, "sign": sign, "status": "BLOCKED",
                                  "reason": f"{iv.target} does not exist (slot unbound)"})
            continue
        treated: dict[str, list[float]] = {m: [] for m in matrix.metrics}
        for s in seeds:
            rt = factory(s)
            if payloads is not None:
                for ref, value in payloads(s).items():
                    rt.interventions.add_payload(ref, value)
            rt.interventions.register(InterventionSpec(iv.kind, iv.target, matrix.start_tick, None, iv.payload_ref,
                                                       prereg_id, f"G2:{iv.name}"))
            run_ticks(rt, ticks, observe, s)
            for m, fn in matrix.metrics.items():
                treated[m].append(float(fn(rt.view())))
        for (name, m), sign in matrix.cells.items():
            if name != iv.name:
                continue
            ci = paired_difference_ci(treated[m], control[m], n_resamples)
            cells.append({"intervention": name, "metric": m, "sign": sign, "status": "RUN", "difference": ci.as_dict(),
                          "consistent_with_prediction": _verdict(sign, ci, matrix.equivalence_margin)})
    rec = gate_record("G2", probe, prereg_id, matrix_hash=matrix.frozen_hash, matrix=matrix.describe(), cells=cells,
                      seeds=len(seeds))
    ran = [c for c in cells if c["status"] == "RUN"]
    if not ran:
        rec.status = ResultStatus.BLOCKED
        rec.notes.append("no cell could run: every targeted slot is unbound")
    elif len(ran) == len(cells):
        propose_verdict(rec, probe.scientific, prereg_id, all(c["consistent_with_prediction"] for c in ran))
    return rec
