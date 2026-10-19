"""G5 Coprocessor ablation: reported separately, never gated.

The same metrics are measured with the configured coprocessor and with it
ablated (NullCoprocessor). What survives ablation is what the individual
itself knows and causes. This result never carries PASS/FAIL and is never
used to hide a native deficit.
"""

from __future__ import annotations

from continuous_brain_protocols import ResultStatus

from .common import MetricFn, ObservationFn, RuntimeFactory, check_seeds, gate_record, run_ticks


def run_g5(factory: RuntimeFactory, metrics: dict[str, MetricFn], seeds: list[int], ticks: int,
           observe: ObservationFn | None = None, prereg_id: str | None = None, seed_registry=None):
    check_seeds(seeds, seed_registry)
    with_cp = {m: [] for m in metrics}
    ablated = {m: [] for m in metrics}
    calls = {"with": 0, "ablated": 0}
    for s in seeds:
        for arm, store in (("with", with_cp), ("ablated", ablated)):
            rt = factory(s)
            if arm == "ablated":
                rt.coprocessor.ablate()
            run_ticks(rt, ticks, observe, s)
            calls[arm] += rt.coprocessor.total_calls
            for m, fn in metrics.items():
                store[m].append(float(fn(rt.view())))
    probe = factory(seeds[0])
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    rec = gate_record("G5", probe, prereg_id, gated=False,
                      with_coprocessor={m: mean(v) for m, v in with_cp.items()},
                      coprocessor_ablated={m: mean(v) for m, v in ablated.items()},
                      borrowed={m: mean(with_cp[m]) - mean(ablated[m]) for m in metrics},
                      coprocessor_calls=calls, provider=probe.coprocessor.provider.name,
                      enabled=probe.coprocessor.enabled)
    rec.status = ResultStatus.ENGINEERING_ONLY
    rec.notes.append("G5 is reported, not gated (integration-gate draft)")
    return rec
