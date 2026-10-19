"""Ridge probes with blocked cross-validation (E2 prereg procedure).

  * features are standardized with statistics from the training rows only
  * the regularization is chosen by blocked k-fold CV on the probe-training
    rows only (contiguous validation blocks, a gap of `gap` rows removed on
    both sides of each block so autocorrelation cannot leak)
  * train and test are contiguous time blocks separated by a gap of at least
    horizon + history rows
  * R^2 is computed per target column against the test-set mean

The solve uses one SVD per fit, so a whole lambda grid costs one decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_LAMBDAS = tuple(10.0 ** np.arange(-3, 6))


@dataclass
class RidgeFit:
    lam: float
    coef: np.ndarray            # [F, T]
    intercept: np.ndarray       # [T]
    mean: np.ndarray
    scale: np.ndarray
    cv_r2: dict[float, float] = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept


SCALE_FLOOR_REL = 1e-3


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score with training statistics.

    Columns constant in the training rows get an infinite scale (they contribute
    nothing). Other scales are floored at 1e-3 x the median positive scale: a
    feature that is almost silent during training (a decaying spike-trace tail,
    say) would otherwise be multiplied by ~1e9 on the test rows.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    live = std > 1e-12
    floor = SCALE_FLOOR_REL * float(np.median(std[live])) if live.any() else 1.0
    scale = np.where(live, np.maximum(std, floor), np.inf)
    return (x - mean) / scale, mean, scale


def _solve_grid(xs: np.ndarray, yc: np.ndarray, lambdas) -> dict[float, np.ndarray]:
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    uty = u.T @ yc
    return {lam: vt.T @ ((s / (s ** 2 + lam))[:, None] * uty) for lam in lambdas}


def r2_score(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = y.reshape(len(y), -1)
    pred = pred.reshape(len(pred), -1)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    sse = ((y - pred) ** 2).sum(axis=0)
    return np.where(sst > 1e-12, 1.0 - sse / np.where(sst > 1e-12, sst, 1.0), 0.0)


def blocked_folds(n: int, k: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = np.linspace(0, n, k + 1).astype(int)
    folds = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        val = np.arange(lo, hi)
        train = np.concatenate([np.arange(0, max(lo - gap, 0)), np.arange(min(hi + gap, n), n)])
        if train.size and val.size:
            folds.append((train, val))
    return folds


def blocked_split(n: int, test_fraction: float = 0.2, gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n_test = max(1, int(round(n * test_fraction)))
    test_start = n - n_test
    train = np.arange(0, max(test_start - gap, 0))
    if train.size < 2:
        raise ValueError("not enough rows for a training block after the gap")
    return train, np.arange(test_start, n)


def fit_ridge(x: np.ndarray, y: np.ndarray, lambdas=DEFAULT_LAMBDAS, k: int = 5, gap: int = 0) -> RidgeFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(len(x), -1)
    cv = {lam: [] for lam in lambdas}
    for tr, va in blocked_folds(len(x), k, gap):
        xs, mean, scale = _standardize(x[tr])
        ym = y[tr].mean(axis=0)
        coefs = _solve_grid(xs, y[tr] - ym, lambdas)
        xv = (x[va] - mean) / scale
        for lam, c in coefs.items():
            cv[lam].append(float(np.mean(r2_score(y[va], xv @ c + ym))))
    cv_mean = {lam: float(np.mean(v)) if v else -np.inf for lam, v in cv.items()}
    best = max(cv_mean, key=lambda lam: (cv_mean[lam], lam))    # ties -> stronger regularization
    xs, mean, scale = _standardize(x)
    ym = y.mean(axis=0)
    coef = _solve_grid(xs, y - ym, [best])[best]
    return RidgeFit(best, coef, ym, mean, scale, cv_mean)


@dataclass
class ProbeResult:
    r2: np.ndarray                  # per target column, on the test block
    mean_r2: float
    lam: float
    n_train: int
    n_test: int
    gap: int
    predictions: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {"r2": self.r2.tolist(), "mean_r2": self.mean_r2, "lambda": self.lam, "n_train": self.n_train,
                "n_test": self.n_test, "gap": self.gap}


def probe(x: np.ndarray, y: np.ndarray, gap: int, test_fraction: float = 0.2, k: int = 5,
          lambdas=DEFAULT_LAMBDAS, keep_predictions: bool = False) -> ProbeResult:
    tr, te = blocked_split(len(x), test_fraction, gap)
    fit = fit_ridge(x[tr], y[tr], lambdas, k, gap)
    pred = fit.predict(x[te])
    r2 = r2_score(np.asarray(y, dtype=np.float64).reshape(len(x), -1)[te], pred)
    return ProbeResult(r2, float(np.mean(r2)), fit.lam, len(tr), len(te), gap, pred if keep_predictions else None)


def classify(x: np.ndarray, labels: np.ndarray, n_classes: int, gap: int = 0, test_fraction: float = 0.3,
             k: int = 5, lambdas=DEFAULT_LAMBDAS) -> dict:
    """Ridge on one-hot targets, argmax decision. Returns accuracy and chance level."""
    labels = np.asarray(labels, dtype=int)
    onehot = np.eye(n_classes)[labels]
    tr, te = blocked_split(len(x), test_fraction, gap)
    fit = fit_ridge(x[tr], onehot[tr], lambdas, k, gap)
    acc = float(np.mean(np.argmax(fit.predict(x[te]), axis=1) == labels[te]))
    return {"accuracy": acc, "chance": 1.0 / n_classes, "n_train": int(len(tr)), "n_test": int(len(te)),
            "lambda": fit.lam}
