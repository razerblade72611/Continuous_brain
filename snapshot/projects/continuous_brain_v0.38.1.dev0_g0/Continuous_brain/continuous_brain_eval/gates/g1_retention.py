"""G1 Retention: each admitted mechanism's frozen evaluator, re-run inside the
unified runtime on a fresh seed family, against its original admission thresholds.

Refuses a seed family that overlaps any registered range (spent sealed TEST
ranges can never be reused). With no bound mechanism the result is BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuous_brain_protocols import ResultStatus, SeedFamily, SeedPurpose

from ..seeds import SeedRegistry
from .common import MechanismEvaluator, ObservationFn, RuntimeFactory, gate_record, propose_verdict, run_ticks


@dataclass(frozen=True)
class RetentionThreshold:
    mechanism_id: str
    metric: str
    minimum: float
    source: str            # the original protocol / result that set it


def run_g1(factory: RuntimeFactory, evaluators: list[MechanismEvaluator], family: SeedFamily,
           thresholds: list[RetentionThreshold], ticks: int, observe: ObservationFn | None = None,
           prereg_id: str | None = None, seeds: SeedRegistry | None = None, max_seeds: int | None = None,
           required_mechanisms: set[str] | None = None):
    registry = seeds or SeedRegistry.load()
    registry.check_usable(family)
    probe = factory(family.start)
    if not probe.components:
        rec = gate_record("G1", probe, prereg_id, bound=[])
        rec.status = ResultStatus.BLOCKED
        rec.notes.append("no mechanism is bound (DAV-74 has not verified any legacy entry)")
        return rec
    # G1 is a retention gate over a declared mechanism set, never over whichever
    # evaluators happened to be supplied. Scientific runtimes infer legacy ids
    # from component provenance; engineering fixtures may pass an explicit set.
    inferred = set()
    for comp in probe.components.values():
        detail = dict(comp.provenance().detail)
        mid = detail.get("mechanism_id")
        if mid:
            inferred.add(str(mid))
    required = set(required_mechanisms if required_mechanisms is not None else (inferred or {e.mechanism_id for e in evaluators}))
    evaluator_ids = {e.mechanism_id for e in evaluators if e.slot in probe.components}
    threshold_ids = {t.mechanism_id for t in thresholds}
    missing_evaluators = sorted(required - evaluator_ids)
    missing_thresholds = sorted(required - threshold_ids)
    if missing_evaluators or missing_thresholds:
        rec = gate_record("G1", probe, prereg_id, family=family.as_dict(), bound=sorted(probe.components),
                          required_mechanisms=sorted(required), evaluator_coverage=sorted(evaluator_ids),
                          threshold_coverage=sorted(threshold_ids), missing_evaluators=missing_evaluators,
                          missing_thresholds=missing_thresholds)
        rec.status = ResultStatus.BLOCKED
        rec.notes.append("G1 coverage is incomplete: every required mechanism needs its frozen evaluator and threshold")
        return rec

    per_mech: dict[str, dict[str, list[float]]] = {}
    seed_list = list(family.seeds())[:max_seeds] if max_seeds else list(family.seeds())
    for seed in seed_list:
        rt = factory(seed)
        run_ticks(rt, ticks, observe, seed)
        view = rt.view()
        for ev in evaluators:
            if ev.slot not in rt.components:
                continue
            for k, v in ev.evaluate(view, seed).items():
                per_mech.setdefault(ev.mechanism_id, {}).setdefault(k, []).append(float(v))
    summary = {m: {k: sum(v) / len(v) for k, v in ks.items()} for m, ks in per_mech.items()}
    checks = []
    for th in thresholds:
        value = summary.get(th.mechanism_id, {}).get(th.metric)
        checks.append({"mechanism_id": th.mechanism_id, "metric": th.metric, "minimum": th.minimum,
                       "value": value, "meets": value is not None and value >= th.minimum, "source": th.source})
    missing_values = sorted({c["mechanism_id"] for c in checks if c["value"] is None})
    rec = gate_record("G1", probe, prereg_id, family=family.as_dict(), seeds_run=len(seed_list), summary=summary,
                      threshold_checks=checks, bound=sorted(probe.components),
                      required_mechanisms=sorted(required), evaluator_coverage=sorted(evaluator_ids),
                      threshold_coverage=sorted(threshold_ids), missing_metric_values=missing_values)
    if missing_values:
        rec.status = ResultStatus.BLOCKED
        rec.notes.append("one or more required threshold metrics were not emitted by their evaluator")
    else:
        propose_verdict(rec, probe.scientific and bool(checks), prereg_id, all(c["meets"] for c in checks))
    if family.purpose == SeedPurpose.TEST:
        registry.spend(family, source="G1")          # a TEST family is opened once
    return rec
