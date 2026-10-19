"""Resampling statistics shared by the battery and the gate harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float
    n: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def within(self, margin: float) -> bool:
        return -margin <= self.low and self.high <= margin

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": self.level, "n": self.n}


def bootstrap_mean_ci(values, n_resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> Interval:
    """Percentile bootstrap of the mean (the E2 prereg uses 10,000 resamples over replicates)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    a = (1 - level) / 2
    return Interval(float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), level, int(v.size))


def paired_difference_ci(a, b, n_resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> Interval:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal shape")
    return bootstrap_mean_ci(a - b, n_resamples, level, seed)


def moving_block_bootstrap(values, block: int, n_resamples: int = 2000, level: float = 0.95,
                           seed: int = 0) -> Interval:
    """CI of the mean of an autocorrelated series, resampling contiguous blocks."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    block = max(1, min(block, n))
    starts = np.arange(n - block + 1)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = (rng.choice(starts, size=k)[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        means[i] = v[idx].mean()
    a = (1 - level) / 2
    return Interval(float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), level, n)


def wilcoxon_signed_rank_one_sided(diffs) -> float:
    """Exact one-sided p (H1: median > 0) for small n; normal approximation above 25."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 1.0
    ranks = _rankdata(np.abs(d))
    w = float(ranks[d > 0].sum())
    if n <= 25:
        # exact distribution of W+ by dynamic programming over integer-scaled ranks
        scale = 2
        r = np.round(ranks * scale).astype(int)
        total = int(r.sum())
        counts = np.zeros(total + 1)
        counts[0] = 1.0
        for x in r:                       # every rank is >= 1, so x >= 2
            new = counts.copy()
            new[x:] += counts[:total + 1 - x]
            counts = new
        counts /= 2.0 ** n
        return float(counts[int(round(w * scale)):].sum())
    mu = n * (n + 1) / 4
    _, tie_counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(((tie_counts ** 3) - tie_counts).sum()) / 48.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term)
    from math import erf, sqrt
    z = (w - mu - 0.5) / sigma
    return float(0.5 * (1 - erf(z / sqrt(2))))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def linear_trend(values, dt: float = 1.0) -> tuple[float | None, float | None]:
    """Least-squares slope per unit time and its standard error (None, None with fewer than 3 points)."""
    y = np.asarray(values, dtype=np.float64)
    t = np.arange(y.size) * dt
    if y.size < 3:
        return None, None
    tc = t - t.mean()
    slope = float((tc * (y - y.mean())).sum() / (tc ** 2).sum())
    resid = y - y.mean() - slope * tc
    se = float(np.sqrt((resid ** 2).sum() / (y.size - 2) / (tc ** 2).sum()))
    return slope, se
