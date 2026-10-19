"""Track-neutral driver: step a substrate through a stream and record what probes need.

Learning is dispatched by capability and never merged (C2):
  GRADIENT_TRAINING  train_step() every `train_every` frames on the causal window
  LOCAL_PLASTICITY   plasticity_step() every frame
  neither            nothing (homeostasis, if the substrate has it, runs inside step)
The record says which of these actually ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from continuous_brain_protocols import Capability, Substrate

from .features import RawHistoryFeatures, SubsetFeatures


@dataclass
class Recording:
    features: np.ndarray                       # [T, F]
    targets: dict[str, np.ndarray]             # name -> [T, ...]
    raw: np.ndarray | None                     # [T, R] raw-history features when requested
    frames: int
    learning: dict = field(default_factory=dict)
    wall_s: float = 0.0
    feature_index: np.ndarray | None = None


def run_stream(sub: Substrate, stream, n_frames: int, *, learn: bool = False, train_every: int | None = None,
               record: bool = True, raw_lags: int = 0, feature_cap: int = 2000, feature_seed: int = 0,
               first_frame=None) -> Recording:
    """Consume n_frames from `stream` (which must already be reset). Returns the recording.

    `first_frame`: the frame returned by stream.reset(), consumed first if given.
    """
    feats = SubsetFeatures(sub.features(), feature_cap, feature_seed) if record else None
    raw = RawHistoryFeatures(stream.spec, raw_lags) if (record and raw_lags) else None
    can_grad = sub.has(Capability.GRADIENT_TRAINING)
    can_local = sub.has(Capability.LOCAL_PLASTICITY)
    if train_every is None:
        train_every = getattr(getattr(sub, "cfg", None), "tbptt_frames", 20)
    rows, raws, targets = [], [], {}
    counts = {"gradient_updates": 0, "plasticity_updates": 0, "learning_requested": learn,
              "gradient_training": bool(learn and can_grad), "local_plasticity": bool(learn and can_local)}
    t0 = time.perf_counter()
    frame = first_frame
    for i in range(n_frames):
        if frame is None:
            frame = stream.next()
        sub.step(frame.values, stream.spec.dt_s)
        if learn and can_local:
            sub.plasticity_step({})
            counts["plasticity_updates"] += 1
        if learn and can_grad and (i + 1) % train_every == 0:
            if sub.train_step().updated:
                counts["gradient_updates"] += 1
        if record:
            rows.append(feats.read_features())
            if raw is not None:
                raw.push(frame.values)
                raws.append(raw.read_features())
            for k, v in stream.last_targets.items():
                targets.setdefault(k, []).append(np.array(v, dtype=np.float64))
        frame = None
    if learn and can_grad:
        while True:                         # flush the tail of the causal window
            rep = sub.train_step()
            if not rep.updated:
                break
            counts["gradient_updates"] += 1
    return Recording(np.asarray(rows) if record else np.zeros((0, 0)),
                     {k: np.asarray(v) for k, v in targets.items()},
                     np.asarray(raws) if raws else None, n_frames, counts, time.perf_counter() - t0,
                     feats.index if feats is not None else None)


def displacement_targets(q: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Delta q = q(t+h) - q(t); returns (targets [T-h, J], valid row index [T-h])."""
    idx = np.arange(len(q) - horizon)
    return q[idx + horizon] - q[idx], idx
