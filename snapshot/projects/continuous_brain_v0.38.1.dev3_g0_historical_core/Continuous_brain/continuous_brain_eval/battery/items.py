"""Battery items B1-B6 (H1/H2 shared battery), engineering-only in v0.38.0.dev0.

Each item takes a substrate factory (seed -> fresh substrate) and a stream
configuration, runs identically for either track, and returns a ResultRecord.
No item emits PASS/FAIL; metrics are labelled UNREGISTERED_METRIC.

  B1  sensorimotor prediction      BLOCKED: requires the E2 MuJoCo body (C5)
  B2  continual-learning retention a frozen probe fitted after task 1 is re-scored
                                   on task 1's evaluation stream after tasks 2..N
  B3  long-run stability, restore  save mid-run, restore into a fresh substrate,
                                   continue; bitwise comparison; drift of a frozen
                                   probe across evaluation windows
  B4  delayed working memory       cue flash at one of K positions; decode the
                                   position from state D frames after the cue
  B5  temporal structure           rhythmic flashes; decode time-to-next-flash
  B6  lesion recovery              damage 10 % of recurrent synapses / weights;
                                   probe before, just after, and after recovery
"""

from __future__ import annotations

import copy

import numpy as np

from continuous_brain_protocols import Capability, LesionSpec, ResultStatus

from ..driver import displacement_targets, run_stream
from ..results import ResultRecord
from ..ridge import blocked_split, classify, fit_ridge, probe, r2_score
from ..stats import linear_trend
from ..streams import StreamConfig, SyntheticBodyStream
from .common import BatteryConfig, SubstrateFactory, base_record, check_engineering_seed


def _adaptive(sub) -> list[str]:
    return [c.value for c in (Capability.LOCAL_PLASTICITY, Capability.GRADIENT_TRAINING, Capability.HOMEOSTASIS)
            if sub.has(c)]


def _develop(sub, stream_cfg: StreamConfig, seed: int, frames: int) -> dict:
    stream = SyntheticBodyStream(stream_cfg)
    first = stream.reset(seed)
    rec = run_stream(sub, stream, frames, learn=True, record=False, first_frame=first)
    return rec.learning


def _record(sub, stream_cfg: StreamConfig, seed: int, frames: int, cfg: BatteryConfig, raw: bool = False):
    stream = SyntheticBodyStream(stream_cfg)
    first = stream.reset(seed)
    was_frozen = sub.frozen
    sub.freeze()
    try:
        rec = run_stream(sub, stream, frames, learn=False, record=True, raw_lags=cfg.raw_lags if raw else 0,
                         feature_cap=cfg.feature_cap, feature_seed=cfg.seed, first_frame=first)
    finally:
        if not was_frozen:
            sub.unfreeze()
    w = cfg.washout_frames
    rec.features = rec.features[w:]
    rec.targets = {k: v[w:] for k, v in rec.targets.items()}
    if rec.raw is not None:
        rec.raw = rec.raw[w:]
    return rec


def _dq_probe(x: np.ndarray, q: np.ndarray, cfg: BatteryConfig):
    y, idx = displacement_targets(q, cfg.horizon_frames)
    return probe(x[idx], y, gap=cfg.horizon_frames + cfg.raw_lags)


# ---------------------------------------------------------------------- B1
def b1_sensorimotor(factory: SubstrateFactory, cfg: BatteryConfig) -> ResultRecord:
    sub = factory(cfg.seed)
    rec = base_record("B1", sub, cfg, stream_hash="none")
    rec.status = ResultStatus.BLOCKED
    rec.metric_status = ResultStatus.BLOCKED
    rec.notes.append("requires the E2 MuJoCo body (DAV-4x); not run on a stand-in (C5)")
    return rec


# ---------------------------------------------------------------------- B2
def b2_continual(factory: SubstrateFactory, stream_cfg: StreamConfig, cfg: BatteryConfig) -> ResultRecord:
    check_engineering_seed(cfg)
    sub = factory(cfg.seed)
    tasks = [stream_cfg.with_vision(grating_speed_hz=stream_cfg.vision.grating_speed_hz * (k + 1))
             .with_babble(onset_rate_hz=stream_cfg.babble.onset_rate_hz * (1 + 0.5 * k)) for k in range(cfg.b2_tasks)]
    eval_seed = cfg.seed + 1000
    learning, scores = [], []
    fit = split = None
    for k, task in enumerate(tasks):
        learning.append(_develop(sub, task, cfg.seed + k, cfg.develop_frames))
        r = _record(sub, tasks[0], eval_seed, cfg.eval_frames, cfg)
        y, idx = displacement_targets(r.targets["q"], cfg.horizon_frames)
        x = r.features[idx]
        if fit is None:
            split = blocked_split(len(x), 0.2, cfg.horizon_frames)
            fit = fit_ridge(x[split[0]], y[split[0]], gap=cfg.horizon_frames)
        frozen_r2 = float(np.mean(r2_score(y[split[1]], fit.predict(x[split[1]]))))
        refit = _dq_probe(r.features, r.targets["q"], cfg)
        scores.append({"after_task": k + 1, "frozen_probe_r2": frozen_r2, "refit_r2": refit.mean_r2})
    rec = base_record("B2", sub, cfg, tasks[0].config_hash(), {"task_config_hashes": [t.config_hash() for t in tasks]})
    first = scores[0]["frozen_probe_r2"]
    rec.metrics = {"scores": scores, "learning": learning, "adaptive_processes": _adaptive(sub),
                   "retention_ratio_last": (scores[-1]["frozen_probe_r2"] / first) if abs(first) > 1e-9 else None}
    if not (sub.has(Capability.LOCAL_PLASTICITY) or sub.has(Capability.GRADIENT_TRAINING)):
        rec.status = ResultStatus.CAPABILITY_ABSENT
        rec.notes.append("substrate has no learning rule (only homeostasis, if any): retention measures drift, "
                         "not resistance to forgetting")
    return rec


# ---------------------------------------------------------------------- B3
def b3_stability_restore(factory: SubstrateFactory, stream_cfg: StreamConfig, cfg: BatteryConfig) -> ResultRecord:
    check_engineering_seed(cfg)
    sub_a = factory(cfg.seed)
    stream_a = SyntheticBodyStream(stream_cfg)
    first = stream_a.reset(cfg.seed)
    run_stream(sub_a, stream_a, cfg.develop_frames, learn=True, record=False, first_frame=first)
    saved_sub = copy.deepcopy(sub_a.state_dict())
    saved_stream = copy.deepcopy(stream_a.state_dict())
    hash_at_save = sub_a.state_hash()
    cont_a = run_stream(sub_a, stream_a, cfg.b3_continue_frames, learn=True, record=True, feature_cap=cfg.feature_cap,
                        feature_seed=cfg.seed)
    end_a = sub_a.state_hash()

    sub_b = factory(cfg.seed)
    sub_b.load_state_dict(saved_sub)
    stream_b = SyntheticBodyStream(stream_cfg)
    stream_b.reset(cfg.seed)
    stream_b.load_state_dict(saved_stream)
    restored_hash = sub_b.state_hash()
    cont_b = run_stream(sub_b, stream_b, cfg.b3_continue_frames, learn=True, record=True, feature_cap=cfg.feature_cap,
                        feature_seed=cfg.seed)
    end_b = sub_b.state_hash()

    # drift: frozen probe fitted on the first window, scored on later windows of one continuous run
    windows = []
    sub_a.freeze()
    stream_d = SyntheticBodyStream(stream_cfg)
    fr = stream_d.reset(cfg.seed + 1500)
    fit = split = None
    for w in range(cfg.b3_drift_windows):
        r = run_stream(sub_a, stream_d, cfg.eval_frames, record=True, feature_cap=cfg.feature_cap,
                       feature_seed=cfg.seed, first_frame=fr)
        fr = None
        y, idx = displacement_targets(r.targets["q"], cfg.horizon_frames)
        x = r.features[idx]
        if fit is None:
            split = blocked_split(len(x), 0.2, cfg.horizon_frames)
            fit = fit_ridge(x[split[0]], y[split[0]], gap=cfg.horizon_frames)
            windows.append(float(np.mean(r2_score(y[split[1]], fit.predict(x[split[1]])))))
        else:
            windows.append(float(np.mean(r2_score(y, fit.predict(x)))))
    slope, se = linear_trend(windows, cfg.eval_frames * stream_cfg.dt_s)
    rec = base_record("B3", sub_a, cfg, stream_cfg.config_hash())
    rec.metrics = {
        "restore_state_hash_equal": restored_hash == hash_at_save,
        "continuation_features_bit_equal": bool(np.array_equal(cont_a.features, cont_b.features)),
        "continuation_end_state_equal": end_a == end_b,
        "max_abs_feature_difference": float(np.max(np.abs(cont_a.features - cont_b.features))) if cont_a.features.size else 0.0,
        "drift_window_r2": windows, "drift_slope_per_s": slope, "drift_slope_se": se,
        "learning_during_continuation": cont_a.learning}
    return rec


# ---------------------------------------------------------------------- B4
def b4_delayed_memory(factory: SubstrateFactory, stream_cfg: StreamConfig, cfg: BatteryConfig) -> ResultRecord:
    check_engineering_seed(cfg)
    lo, hi = cfg.b4_cue_interval_s
    sc = stream_cfg.with_vision(cue_positions=cfg.b4_positions, cue_min_interval_s=lo, cue_max_interval_s=hi)
    sub = factory(cfg.seed)
    learning = _develop(sub, sc, cfg.seed, cfg.develop_frames)
    r = _record(sub, sc, cfg.seed + 1000, cfg.b4_eval_frames, cfg, raw=True)
    onsets = np.nonzero(r.targets["cue_onset"] > 0.5)[0]
    cue_frames = int(round(sc.vision.cue_duration_s / sc.dt_s))
    per_delay = {}
    for d in cfg.b4_delays_frames:
        rows, labels = [], []
        for i, t0 in enumerate(onsets):
            t = t0 + cue_frames + d
            nxt = onsets[i + 1] if i + 1 < len(onsets) else len(r.features)
            if t < nxt and t < len(r.features):
                rows.append(t)
                labels.append(int(r.targets["last_cue_pos"][t0]))
        entry = {"trials": len(rows), "delay_s": d * sc.dt_s}
        if len(rows) >= 10 and len(set(labels)) > 1:
            rows_a, lab = np.asarray(rows), np.asarray(labels)
            entry["substrate"] = classify(r.features[rows_a], lab, cfg.b4_positions)
            entry["raw_history"] = classify(r.raw[rows_a], lab, cfg.b4_positions)
        else:
            entry["note"] = "too few trials for a decode"
        per_delay[str(d)] = entry
    rec = base_record("B4", sub, cfg, sc.config_hash())
    rec.metrics = {"per_delay_frames": per_delay, "cues": int(len(onsets)), "learning": learning,
                   "raw_history_lags": cfg.raw_lags}
    return rec


# ---------------------------------------------------------------------- B5
def b5_temporal(factory: SubstrateFactory, stream_cfg: StreamConfig, cfg: BatteryConfig) -> ResultRecord:
    check_engineering_seed(cfg)
    sc = stream_cfg.with_vision(rhythm_period_s=cfg.b5_period_s)
    sub = factory(cfg.seed)
    learning = _develop(sub, sc, cfg.seed, cfg.develop_frames)
    r = _record(sub, sc, cfg.seed + 1000, cfg.eval_frames, cfg, raw=True)
    out = {}
    for target in ("time_to_flash_s", "time_since_flash_s"):
        y = r.targets[target]
        out[target] = {"substrate_r2": probe(r.features, y, gap=cfg.raw_lags).mean_r2,
                       "raw_history_r2": probe(r.raw, y, gap=cfg.raw_lags).mean_r2}
    rec = base_record("B5", sub, cfg, sc.config_hash())
    rec.metrics = {"period_s": cfg.b5_period_s, "targets": out, "learning": learning}
    return rec


# ---------------------------------------------------------------------- B6
def b6_lesion(factory: SubstrateFactory, stream_cfg: StreamConfig, cfg: BatteryConfig) -> ResultRecord:
    check_engineering_seed(cfg)
    sub = factory(cfg.seed)
    learning = _develop(sub, stream_cfg, cfg.seed, cfg.develop_frames)
    eval_seed = cfg.seed + 1000
    pre = _record(sub, stream_cfg, eval_seed, cfg.eval_frames, cfg, raw=True)
    pre_r2 = _dq_probe(pre.features, pre.targets["q"], cfg).mean_r2
    raw_r2 = _dq_probe(pre.raw, pre.targets["q"], cfg).mean_r2
    target = cfg.lesion_target(sub.metadata.track)
    lesion = sub.lesion(LesionSpec(target, cfg.b6_fraction, cfg.seed))
    post = _record(sub, stream_cfg, eval_seed, cfg.eval_frames, cfg)
    post_r2 = _dq_probe(post.features, post.targets["q"], cfg).mean_r2
    can_learn = sub.has(Capability.LOCAL_PLASTICITY) or sub.has(Capability.GRADIENT_TRAINING)
    rec_learning = _develop(sub, stream_cfg, cfg.seed + 7, cfg.b6_recovery_frames)
    after = _record(sub, stream_cfg, eval_seed, cfg.eval_frames, cfg)
    after_r2 = _dq_probe(after.features, after.targets["q"], cfg).mean_r2
    rec = base_record("B6", sub, cfg, stream_cfg.config_hash())
    rec.metrics = {"lesion": {"target": lesion.target, "fraction": lesion.requested_fraction, "removed": lesion.removed,
                              "of_total": lesion.of_total, "detail": lesion.detail},
                   "r2_pre": pre_r2, "r2_post_lesion": post_r2, "r2_after_recovery": after_r2,
                   "raw_history_r2": raw_r2, "learning_development": learning, "learning_recovery": rec_learning,
                   "adaptive_processes": _adaptive(sub)}
    if not can_learn:
        rec.status = ResultStatus.CAPABILITY_ABSENT
        rec.notes.append("no learning rule: the recovery phase ran with homeostasis only; recovery through "
                         "plasticity is not measurable until E1 (DAV-40)")
    return rec


ITEMS = {"B1": b1_sensorimotor, "B2": b2_continual, "B3": b3_stability_restore, "B4": b4_delayed_memory,
         "B5": b5_temporal, "B6": b6_lesion}
