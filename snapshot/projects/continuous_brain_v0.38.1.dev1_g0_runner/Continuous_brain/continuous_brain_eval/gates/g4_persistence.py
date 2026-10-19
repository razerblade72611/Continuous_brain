"""G4 Persistence.

  save/restore  run A continuously; run B saves mid-task to the never-overwrite
                SnapshotStore, restores into a *fresh* runtime, and continues.
                Channels whose owner declared an exact-preservation contract
                must match bit for bit at every later tick; the others report
                their maximum absolute difference.
  drift         a metric sampled over consecutive windows of one long run; the
                slope and its standard error are reported against a margin.
"""

from __future__ import annotations

import tempfile

import numpy as np

from continuous_brain_protocols import ResultStatus
from continuous_brain_snn.snapshot import SnapshotStore

from ..stats import linear_trend
from .common import MetricFn, ObservationFn, RuntimeFactory, check_seeds, gate_record, propose_verdict


def _trace(rt, n, observe, seed):
    rows = []
    for _ in range(n):
        tick = rt.tick + 1
        rt.tick_once(None if observe is None else observe(seed, tick))
        rows.append({ch: np.array(rt.store.read(ch)) for ch in rt.store.names()})
    return rows


def run_g4_save_restore(factory: RuntimeFactory, seed: int, save_at: int, continue_ticks: int,
                        observe: ObservationFn | None = None, store_root: str | None = None,
                        prereg_id: str | None = None, seed_registry=None):
    check_seeds([seed], seed_registry)
    a = factory(seed)
    _trace(a, save_at, observe, seed)
    trace_a = _trace(a, continue_ticks, observe, seed)
    b = factory(seed)
    _trace(b, save_at, observe, seed)
    tmp = None
    if store_root is None:
        tmp = tempfile.TemporaryDirectory()
        store_root = tmp.name
    store = SnapshotStore(store_root)
    snap = b.save(store, label="g4")
    c = factory(seed)
    c.restore(store, snap)
    trace_c = _trace(c, continue_ticks, observe, seed)
    exact_channels = [n for n in a.store.names() if a.store.decl(n).exact_contract]
    exact_ok, max_diff = True, {}
    for ra, rc in zip(trace_a, trace_c):
        for ch in ra:
            d = float(np.max(np.abs(ra[ch] - rc[ch]))) if ra[ch].size else 0.0
            max_diff[ch] = max(max_diff.get(ch, 0.0), d)
            if ch in exact_channels and not np.array_equal(ra[ch], rc[ch]):
                exact_ok = False
    rec = gate_record("G4", a, prereg_id, save_at=save_at, continue_ticks=continue_ticks, snapshot=str(snap.name),
                      snapshot_verified=store.verify(snap), exact_channels=exact_channels,
                      exact_channels_preserved=exact_ok, max_abs_difference=max_diff,
                      final_state_equal=a.state_hash() == c.state_hash())
    if tmp is not None:
        tmp.cleanup()
    if not a.components:
        rec.notes.append("no mechanism bound: only environment channels were checked")
    if not exact_channels:
        rec.notes.append("no channel declares an exact-preservation contract: no verdict (it would be vacuous)")
    elif a.components:
        propose_verdict(rec, a.scientific, prereg_id, exact_ok)
    return rec


def run_g4_drift(factory: RuntimeFactory, seed: int, window_ticks: int, windows: int, metric: MetricFn,
                 observe: ObservationFn | None = None, margin_per_s: float | None = None,
                 prereg_id: str | None = None, seed_registry=None):
    check_seeds([seed], seed_registry)
    rt = factory(seed)
    values = []
    for _ in range(windows):
        rt.run(window_ticks, None if observe is None else (lambda t: observe(seed, t)))
        values.append(float(metric(rt.view())))
    slope, se = linear_trend(values, window_ticks * rt.config.dt_s)
    rec = gate_record("G4-drift", rt, prereg_id, window_values=values, slope_per_s=slope, slope_se=se,
                      simulated_s=rt.time_s, margin_per_s=margin_per_s)
    if slope is None:
        rec.notes.append("fewer than 3 windows: no trend estimate")
    elif margin_per_s is not None and rt.components:
        propose_verdict(rec, rt.scientific, prereg_id, abs(slope) + 2 * se <= margin_per_s)
    return rec
