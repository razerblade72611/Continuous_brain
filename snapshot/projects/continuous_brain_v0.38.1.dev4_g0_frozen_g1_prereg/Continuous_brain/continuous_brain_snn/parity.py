"""Statistical cross-backend parity (DAV-32).

Recurrent spiking networks amplify rounding differences: one flipped spike
eventually diverges the trajectory. So CPU / CUDA / DirectML runs are compared
statistically, over >= 60 s of simulated time with plasticity off:

  * per-population mean firing rate within +-5 %
  * per-population ISI coefficient of variation within +-0.05
  * two-sample KS statistic < 0.1 between the distributions of pairwise
    spike-count correlations (50 ms bins, fixed neuron sample)

Time to first spike mismatch is recorded as a diagnostic, never a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .engine import SNNEngine

RATE_REL_TOL = 0.05
CV_ABS_TOL = 0.05
KS_MAX = 0.1


class SpikeRecorder:
    """Records (step, neuron) spike events for batch row 0, plus a raster hash."""

    def __init__(self, sample: np.ndarray | None = None):
        self.steps: list[np.ndarray] = []
        self.neurons: list[np.ndarray] = []
        self.sample = sample

    def __call__(self, engine: SNNEngine, spikes: torch.Tensor) -> None:
        idx = torch.nonzero(spikes[0]).flatten().cpu().numpy()
        if idx.size:
            self.neurons.append(idx)
            self.steps.append(np.full(idx.size, engine.steps, dtype=np.int64))

    def events(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.steps:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(self.steps), np.concatenate(self.neurons)


@dataclass
class SpikeStatistics:
    duration_s: float
    rate_hz: dict[str, float]
    isi_cv: dict[str, float]
    correlations: np.ndarray
    first_spike_steps: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_json(self) -> dict:
        return {"duration_s": self.duration_s, "rate_hz": self.rate_hz, "isi_cv": self.isi_cv,
                "correlations": self.correlations.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "SpikeStatistics":
        return cls(d["duration_s"], d["rate_hz"], d["isi_cv"], np.asarray(d["correlations"]))


def compute_statistics(engine: SNNEngine, recorder: SpikeRecorder, n_steps: int,
                       sample_size: int = 200, bin_s: float = 0.05, seed: int = 0) -> SpikeStatistics:
    steps, neurons = recorder.events()
    return statistics_from_events(engine.anatomy, engine.dt, steps, neurons, n_steps,
                                  sample_size, bin_s, seed)


def statistics_from_events(a, dt: float, steps: np.ndarray, neurons: np.ndarray, n_steps: int,
                           sample_size: int = 200, bin_s: float = 0.05, seed: int = 0) -> SpikeStatistics:
    duration = n_steps * dt
    rate, cv = {}, {}
    order = np.lexsort((steps, neurons))
    steps_s, neurons_s = steps[order], neurons[order]
    bounds = np.searchsorted(neurons_s, np.arange(a.n + 1))
    cv_per_neuron = np.full(a.n, np.nan)
    for i in range(a.n):
        st = steps_s[bounds[i]:bounds[i + 1]]
        if st.size >= 3:
            isi = np.diff(st).astype(np.float64)
            cv_per_neuron[i] = isi.std() / isi.mean()
    counts = np.bincount(neurons, minlength=a.n)
    for p in a.populations:
        rate[p.name] = float(counts[p.index].mean() / duration)
        vals = cv_per_neuron[p.index]
        cv[p.name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    rng = np.random.default_rng(seed)
    cortex_exc = np.nonzero((a.region == 3) & a.is_exc)[0]
    sample = np.sort(rng.choice(cortex_exc, size=min(sample_size, cortex_exc.size), replace=False))
    n_bins = max(int(duration / bin_s), 2)
    bins_per = int(round(bin_s / dt))
    mat = np.zeros((sample.size, n_bins))
    in_sample = np.isin(neurons, sample)
    rows = np.searchsorted(sample, neurons[in_sample])
    cols = np.minimum((steps[in_sample] - 1) // bins_per, n_bins - 1)
    np.add.at(mat, (rows, cols), 1)
    active = mat.std(axis=1) > 0
    m = mat[active]
    if m.shape[0] >= 2:
        c = np.corrcoef(m)
        corr = c[np.triu_indices_from(c, k=1)]
    else:
        corr = np.zeros(0)
    return SpikeStatistics(duration, rate, cv, corr)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(fa - fb).max())


@dataclass
class ParityResult:
    passed: bool
    failures: list[str]
    ks: float
    worst_rate_rel: float
    worst_cv_abs: float


def compare(a: SpikeStatistics, b: SpikeStatistics, rate_rel_tol: float = RATE_REL_TOL,
            cv_abs_tol: float = CV_ABS_TOL, ks_max: float = KS_MAX, min_rate_hz: float = 0.5) -> ParityResult:
    failures, worst_rate, worst_cv = [], 0.0, 0.0
    for name, ra in a.rate_hz.items():
        rb = b.rate_hz[name]
        if max(ra, rb) < min_rate_hz:
            continue   # near-silent populations: relative error is meaningless
        rel = abs(ra - rb) / max(ra, 1e-9)
        worst_rate = max(worst_rate, rel)
        if rel > rate_rel_tol:
            failures.append(f"rate {name}: {ra:.3f} vs {rb:.3f} Hz ({rel:.1%})")
        ca, cb = a.isi_cv[name], b.isi_cv[name]
        if np.isfinite(ca) and np.isfinite(cb):
            worst_cv = max(worst_cv, abs(ca - cb))
            if abs(ca - cb) > cv_abs_tol:
                failures.append(f"ISI CV {name}: {ca:.3f} vs {cb:.3f}")
    ks = ks_statistic(a.correlations, b.correlations)
    if ks >= ks_max:
        failures.append(f"correlation KS {ks:.3f} >= {ks_max}")
    return ParityResult(not failures, failures, ks, worst_rate, worst_cv)
