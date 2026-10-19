"""Telemetry schema and loggers (DAV-37).

`WindowStats` accumulates spikes and drive over a measurement window;
`collect` turns it into one flat record whose keys are fixed by
REQUIRED_KEYS, so every smoke run yields a complete, dashboard-ready record.

Backends:
  JsonlLogger   local, offline-safe, append-only (always available)
  WandbLogger   Weights & Biases; imported lazily, defaults to offline mode
  MultiLogger   fan-out to several backends
Neuromodulator fields are reserved (null) until E3 introduces them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .anatomy import REGIONS
from .config import SCHEMA_VERSION
from .engine import SNNEngine

REQUIRED_KEYS = (
    "schema_version", "stage", "step", "time_s", "window_s",
    "rate_hz/mean", "rate_hz/exc", "rate_hz/inh",
    *(f"rate_hz/region/{r}" for r in REGIONS),
    "rate_hz/population_min", "rate_hz/population_max", "silent_fraction",
    "threshold/mean", "threshold/std", "threshold/p05", "threshold/p95",
    "drive/g_exc_mean", "drive/g_inh_mean", "drive/ei_ratio",
    *(f"trace/tau_{int(round(t * 1000))}ms_mean" for t in (0.005, 0.025, 0.1, 0.5, 2.0)),
    "sparsity/active_slot_fraction", "sparsity/connection_density",
    "sparsity/fanin_mean", "sparsity/fanout_max",
    "hubs/top1pct_outdegree_share", "hubs/outdegree_gini",
    "timing/ms_per_step", "memory/rss_mb", "memory/device_mb",
    "health/nonfinite", "neuromodulators",
)


def _rss_mb() -> float:
    try:
        import resource  # POSIX (Linux / WSL)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:  # native Windows
        try:
            import psutil
            return psutil.Process().memory_info().rss / 2 ** 20
        except ImportError:
            return float("nan")


def _gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    if x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


class WindowStats:
    """Accumulates per-neuron spike counts and E/I drive between collections."""

    def __init__(self, engine: SNNEngine):
        self.e = engine
        self.reset()

    def reset(self) -> None:
        self.counts = torch.zeros(self.e.n, device=self.e.device)
        self.g_e = torch.zeros((), device=self.e.device)
        self.g_i = torch.zeros((), device=self.e.device)
        self.n_steps = 0
        self.t_wall = time.perf_counter()

    def __call__(self, engine: SNNEngine, spikes: torch.Tensor) -> None:
        self.counts += spikes.to(torch.float32).sum(dim=0) / engine.batch_size
        self.g_e += engine.g_e.mean()
        self.g_i += engine.g_i.mean()
        self.n_steps += 1


def collect(engine: SNNEngine, window: WindowStats) -> dict:
    if window.n_steps == 0:
        raise ValueError("empty telemetry window")
    a = engine.anatomy
    dur = window.n_steps * engine.dt
    rates = (window.counts / dur).cpu().numpy()
    exc = a.is_exc
    pop_rates = [float(rates[p.index].mean()) for p in a.populations]
    thr = engine.threshold.detach().cpu().numpy().ravel()
    traces = engine.traces.x.detach().cpu().numpy().mean(axis=(0, 1))
    slots = engine.slots
    fanout = slots.fanout().cpu().numpy()
    k = max(1, int(round(0.01 * engine.n)))
    g_e = float(window.g_e / window.n_steps)
    g_i = float(window.g_i / window.n_steps)
    wall = time.perf_counter() - window.t_wall
    device_mb = 0.0
    if engine.device.type == "cuda":
        device_mb = torch.cuda.max_memory_allocated(engine.device) / 2 ** 20
    nonfinite = 0
    for t in (engine.v, engine.g_e, engine.g_i, engine.threshold, engine.traces.x):
        nonfinite += int((~torch.isfinite(t)).sum())

    rec = {
        "schema_version": SCHEMA_VERSION, "stage": engine.cfg.stage,
        "step": engine.steps, "time_s": engine.time_s, "window_s": dur,
        "rate_hz/mean": float(rates.mean()), "rate_hz/exc": float(rates[exc].mean()),
        "rate_hz/inh": float(rates[~exc].mean()) if (~exc).any() else 0.0,
        **{f"rate_hz/region/{r}": float(rates[a.region == i].mean()) for i, r in enumerate(REGIONS)},
        "rate_hz/population_min": min(pop_rates), "rate_hz/population_max": max(pop_rates),
        "silent_fraction": float((rates == 0).mean()),
        "threshold/mean": float(thr.mean()), "threshold/std": float(thr.std()),
        "threshold/p05": float(np.percentile(thr, 5)), "threshold/p95": float(np.percentile(thr, 95)),
        "drive/g_exc_mean": g_e, "drive/g_inh_mean": g_i,
        "drive/ei_ratio": g_e / g_i if g_i > 0 else float("inf"),
        **{f"trace/tau_{int(round(t * 1000))}ms_mean": float(v)
           for t, v in zip(engine.cfg.traces.taus_s, traces)},
        "sparsity/active_slot_fraction": float(slots.active.float().mean()),
        "sparsity/connection_density": float(slots.active.sum()) / engine.n ** 2,
        "sparsity/fanin_mean": float(slots.fanin().float().mean()),
        "sparsity/fanout_max": int(fanout.max()),
        "hubs/top1pct_outdegree_share": float(np.sort(fanout)[-k:].sum() / max(fanout.sum(), 1)),
        "hubs/outdegree_gini": _gini(fanout),
        "timing/ms_per_step": 1000.0 * wall / window.n_steps,
        "memory/rss_mb": _rss_mb(),
        "memory/device_mb": device_mb,
        "health/nonfinite": nonfinite,
        "neuromodulators": None,
        "population_rates_hz": {p.name: r for p, r in zip(a.populations, pop_rates)},
    }
    missing = [key for key in REQUIRED_KEYS if key not in rec]
    if missing:
        raise AssertionError(f"telemetry record missing keys: {missing}")
    return rec


class JsonlLogger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        pass


class WandbLogger:
    def __init__(self, project: str = "continuous-brain", run_name: str | None = None,
                 config: dict | None = None, mode: str = "offline", tags: list[str] | None = None):
        import wandb  # lazy: the local logger must work without it
        self._wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode, tags=tags)

    def log(self, record: dict) -> None:
        flat = {k: v for k, v in record.items() if isinstance(v, (int, float))}
        self._wandb.log(flat, step=int(record["step"]))

    def close(self) -> None:
        self.run.finish()


class MultiLogger:
    def __init__(self, *loggers):
        self.loggers = loggers

    def log(self, record: dict) -> None:
        for lg in self.loggers:
            lg.log(record)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()
