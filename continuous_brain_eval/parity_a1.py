"""Power-aware cross-backend parity (Amendment A1.2) and its pilot utilities.

PROPOSED, not approved. The current rule is DAV-32-v1 (fixed +-5 % rate and
+-0.05 ISI CV on every population, KS < 0.1), implemented in
`continuous_brain_snn.parity.compare` and unchanged.

A1.2, per population p with n neurons over T seconds:
  rate   SE = sqrt((r_a + r_b) / (n T))  (Poisson counts)
         tolerance = max(5 % of r_a, z * SE), z = 3.57 (Bonferroni, family
         alpha = 0.05 over 140 populations, two-sided)
  CV     SE from a bootstrap over the population's neurons (both runs)
         tolerance = max(0.05, z * SE)
  KS     unchanged (< 0.1)
Populations where z * SE exceeds the fixed tolerance are reported as
underpowered, so reduced sensitivity is visible.

Open question for A1 (not decided here): with a rate test *and* a CV test per
population the family is 280 tests, for which the Bonferroni z is 3.75, not
3.57. `bonferroni_z(n_tests)` computes either; the default follows A1's text.

Acceptance before freezing (A1.2): identical-process pilot false-fail rate
<= 5 % over >= 100 pairs, and detection of a 10 % rate shift in every
population with >= 5,000 expected spikes. `pilot_false_fail` and
`pilot_sensitivity` run exactly those checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

from continuous_brain_snn.parity import KS_MAX, ks_statistic, statistics_from_events

Z_BONFERRONI_140 = 3.57
RATE_REL_FLOOR = 0.05
CV_ABS_FLOOR = 0.05


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided z for family alpha over n tests (bisection on the normal CDF)."""
    target = 1 - alpha / (2 * n_tests)
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class PopulationArrays:
    """Per-neuron spike counts and ISI CVs, grouped by population, plus the correlation sample."""
    duration_s: float
    counts: dict[str, np.ndarray]
    cvs: dict[str, np.ndarray]
    correlations: np.ndarray


def population_arrays(anatomy, dt: float, steps: np.ndarray, neurons: np.ndarray, n_steps: int,
                      sample_size: int = 200, bin_s: float = 0.05, seed: int = 0) -> PopulationArrays:
    duration = n_steps * dt
    order = np.lexsort((steps, neurons))
    st, nr = steps[order], neurons[order]
    bounds = np.searchsorted(nr, np.arange(anatomy.n + 1))
    cv = np.full(anatomy.n, np.nan)
    for i in range(anatomy.n):
        s = st[bounds[i]:bounds[i + 1]]
        if s.size >= 3:
            isi = np.diff(s).astype(np.float64)
            cv[i] = isi.std() / isi.mean()
    counts = np.bincount(neurons, minlength=anatomy.n)
    base = statistics_from_events(anatomy, dt, steps, neurons, n_steps, sample_size, bin_s, seed)
    return PopulationArrays(duration, {p.name: counts[p.index].astype(np.float64) for p in anatomy.populations},
                            {p.name: cv[p.index] for p in anatomy.populations}, base.correlations)


def _boot_se_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("inf")
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(means.std(ddof=1))


@dataclass
class A1Result:
    passed: bool
    failures: list[str]
    underpowered: list[str]
    ks: float
    per_population: dict[str, dict] = field(default_factory=dict)


def compare_a1(a: PopulationArrays, b: PopulationArrays, z: float = Z_BONFERRONI_140, min_rate_hz: float = 0.5,
               n_boot: int = 500, seed: int = 0) -> A1Result:
    if abs(a.duration_s - b.duration_s) > 1e-9:
        raise ValueError("A1 compares runs of equal duration")
    rng = np.random.default_rng(seed)
    failures, under, per = [], [], {}
    for name, ca in a.counts.items():
        cb = b.counts[name]
        n = ca.size
        ra, rb = ca.sum() / (n * a.duration_s), cb.sum() / (n * b.duration_s)
        if max(ra, rb) < min_rate_hz:
            continue
        se = sqrt((ra + rb) / (n * a.duration_s))
        tol = max(RATE_REL_FLOOR * ra, z * se)
        entry = {"rate_a": ra, "rate_b": rb, "rate_tol_hz": tol, "rate_se_hz": se}
        if z * se > RATE_REL_FLOOR * ra:
            under.append(f"rate {name}")
        if abs(ra - rb) > tol:
            failures.append(f"rate {name}: {ra:.3f} vs {rb:.3f} Hz (tol {tol:.3f})")
        va, vb = a.cvs[name], b.cvs[name]
        if np.isfinite(va).sum() >= 2 and np.isfinite(vb).sum() >= 2:
            ma, mb = float(np.nanmean(va)), float(np.nanmean(vb))
            se_cv = sqrt(_boot_se_mean(va, n_boot, rng) ** 2 + _boot_se_mean(vb, n_boot, rng) ** 2)
            tol_cv = max(CV_ABS_FLOOR, z * se_cv)
            entry.update({"cv_a": ma, "cv_b": mb, "cv_tol": tol_cv})
            if z * se_cv > CV_ABS_FLOOR:
                under.append(f"cv {name}")
            if abs(ma - mb) > tol_cv:
                failures.append(f"ISI CV {name}: {ma:.3f} vs {mb:.3f} (tol {tol_cv:.3f})")
        per[name] = entry
    ks = ks_statistic(a.correlations, b.correlations)
    if ks >= KS_MAX:
        failures.append(f"correlation KS {ks:.3f} >= {KS_MAX}")
    return A1Result(not failures, failures, under, ks, per)


def poisson_events(anatomy, rates_hz: dict[str, float] | float, duration_s: float, dt: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Independent Poisson spike trains per neuron at population rates (pilot generator)."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration_s / dt))
    rate = np.zeros(anatomy.n)
    for p in anatomy.populations:
        rate[p.index] = rates_hz if isinstance(rates_hz, (int, float)) else rates_hz[p.name]
    steps_l, neur_l = [], []
    chunk = max(1, 200_000_000 // max(anatomy.n, 1) // 8)
    for start in range(0, n_steps, chunk):
        m = rng.random((min(chunk, n_steps - start), anatomy.n)) < rate * dt
        s, nidx = np.nonzero(m)
        steps_l.append(s + start + 1)
        neur_l.append(nidx)
    return np.concatenate(steps_l), np.concatenate(neur_l), n_steps


def pilot_false_fail(anatomy, rates_hz, n_pairs: int = 100, duration_s: float = 60.0, dt: float = 0.001,
                     seed: int = 0, rule: str = "A1") -> dict:
    """Fraction of statistically identical pairs that the rule rejects (A1 acceptance 1: <= 5 %)."""
    from continuous_brain_snn.parity import compare as compare_v1
    fails = 0
    for i in range(n_pairs):
        runs = []
        for j in (0, 1):
            s, nr, n_steps = poisson_events(anatomy, rates_hz, duration_s, dt, seed + 2 * i + j)
            runs.append((s, nr, n_steps))
        if rule == "A1":
            res = compare_a1(population_arrays(anatomy, dt, *runs[0]), population_arrays(anatomy, dt, *runs[1]),
                             seed=seed + i)
            fails += not res.passed
        else:
            res = compare_v1(statistics_from_events(anatomy, dt, *runs[0]), statistics_from_events(anatomy, dt, *runs[1]))
            fails += not res.passed
    return {"rule": rule, "pairs": n_pairs, "false_fails": fails, "false_fail_rate": fails / n_pairs,
            "criterion": "<= 0.05", "meets_criterion": fails / n_pairs <= 0.05}


def pilot_sensitivity(anatomy, rates_hz: dict[str, float], shift: float = 0.10, duration_s: float = 60.0,
                      dt: float = 0.001, seed: int = 0, min_expected_spikes: int = 5000) -> dict:
    """Shift each eligible population's rate by `shift` alone; the rule must flag every one (A1 acceptance 2)."""
    s0, n0, steps = poisson_events(anatomy, rates_hz, duration_s, dt, seed)
    base = population_arrays(anatomy, dt, s0, n0, steps)
    eligible = [p.name for p in anatomy.populations if rates_hz[p.name] * p.index.size * duration_s >= min_expected_spikes]
    missed = []
    for k, name in enumerate(eligible):
        shifted = dict(rates_hz)
        shifted[name] = rates_hz[name] * (1 + shift)
        s1, n1, _ = poisson_events(anatomy, shifted, duration_s, dt, seed + 1 + k)
        res = compare_a1(base, population_arrays(anatomy, dt, s1, n1, steps), seed=seed + k)
        if not any(f.startswith(f"rate {name}:") for f in res.failures):
            missed.append(name)
    return {"shift": shift, "eligible_populations": len(eligible), "missed": missed,
            "meets_criterion": not missed}
