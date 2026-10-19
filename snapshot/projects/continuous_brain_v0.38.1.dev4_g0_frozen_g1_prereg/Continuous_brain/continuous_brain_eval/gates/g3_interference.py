"""G3 Interference (proposed by ChatGPT): adversarial schedules that make
individually valid mechanisms collide, scored as degradation of G1 metrics
against the isolated-package baseline.

The four schedules from the integration-gate draft are expressed as data: the
slots they need, and the interventions (on canonical channels) that create the
pressure. A schedule whose slots are unbound is BLOCKED rather than run on
stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from continuous_brain_protocols import InterventionKind, InterventionSpec, ResultStatus

from .common import MetricFn, ObservationFn, RuntimeFactory, check_seeds, gate_record, propose_verdict, run_ticks


@dataclass(frozen=True)
class InterferenceSchedule:
    name: str
    description: str
    requires_slots: tuple[str, ...]
    build: Callable[[int, int], list[tuple[InterventionSpec, dict]]]    # (seed, ticks) -> [(spec, payloads)]


def _churn(target: str, period: int, kind=InterventionKind.REPLACE):
    def build(seed: int, ticks: int):
        out = []
        for i, start in enumerate(range(period, ticks, 2 * period)):
            ref = None if kind == InterventionKind.FREEZE else f"churn:{target}:{i % 2}"
            out.append((InterventionSpec(kind, target, start, min(start + period, ticks), ref, note="G3"), {"ref": ref}))
        return out
    return build


STANDARD_SCHEDULES = (
    InterferenceSchedule("working_memory_load_during_goal",
                         "fill working memory while an endogenous goal is being pursued",
                         ("working_memory", "goal_genesis"), _churn("working_memory.contents", 5)),
    InterferenceSchedule("replan_during_autobiography_update",
                         "force stale plans exactly while the autobiography is being updated",
                         ("planner", "autobiography"), _churn("planner.plan", 3, InterventionKind.FREEZE)),
    InterferenceSchedule("rapid_goal_churn", "replace the active goal every few ticks",
                         ("goal_genesis",), _churn("goal_genesis.active_goal", 2)),
    InterferenceSchedule("long_run_shared_state", "run long enough that one mechanism's updates can "
                         "overwrite shared state; no intervention, length is the pressure",
                         ("self_state", "autobiography"), lambda seed, ticks: []),
)


def run_g3(factory: RuntimeFactory, schedules: tuple[InterferenceSchedule, ...], metrics: dict[str, MetricFn],
           seeds: list[int], ticks: int, margin: float | None = None, observe: ObservationFn | None = None,
           payload_values: Callable[[str, int], object] | None = None, prereg_id: str | None = None,
           seed_registry=None):
    check_seeds(seeds, seed_registry)
    probe = factory(seeds[0])
    baseline = {m: [] for m in metrics}
    for s in seeds:
        rt = factory(s)
        run_ticks(rt, ticks, observe, s)
        for m, fn in metrics.items():
            baseline[m].append(float(fn(rt.view())))
    base_mean = {m: sum(v) / len(v) for m, v in baseline.items()}
    results = []
    for sch in schedules:
        missing = [s for s in sch.requires_slots if s not in probe.components]
        if missing:
            results.append({"schedule": sch.name, "status": "BLOCKED", "unbound": missing})
            continue
        loaded = {m: [] for m in metrics}
        for s in seeds:
            rt = factory(s)
            for spec, pay in sch.build(s, ticks):
                if spec.payload_ref and spec.payload_ref not in rt.interventions.payloads:
                    if payload_values is None:
                        raise ValueError(f"schedule {sch.name} needs payload values for {spec.payload_ref}")
                    rt.interventions.add_payload(spec.payload_ref, payload_values(spec.payload_ref, s))
                rt.interventions.register(spec)
            run_ticks(rt, ticks, observe, s)
            for m, fn in metrics.items():
                loaded[m].append(float(fn(rt.view())))
        degr = {m: base_mean[m] - sum(v) / len(v) for m, v in loaded.items()}
        entry = {"schedule": sch.name, "status": "RUN", "degradation": degr}
        if margin is not None:
            entry["within_margin"] = all(abs(d) <= margin for d in degr.values())
        results.append(entry)
    rec = gate_record("G3", probe, prereg_id, baseline=base_mean, schedules=results, margin=margin, seeds=len(seeds))
    ran = [r for r in results if r["status"] == "RUN"]
    if not ran:
        rec.status = ResultStatus.BLOCKED
    elif margin is not None and len(ran) == len(results):
        propose_verdict(rec, probe.scientific, prereg_id, all(r["within_margin"] for r in ran))
    return rec
