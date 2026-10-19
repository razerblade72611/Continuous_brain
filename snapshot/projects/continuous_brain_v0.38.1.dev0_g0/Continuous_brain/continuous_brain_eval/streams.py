"""Synthetic physical experience streams for engineering validation of B2-B6.

NOT the E2 body. B1 (sensorimotor prediction) needs the MuJoCo body and stays
BLOCKED (C5). These streams exist so the battery harness, determinism,
save/restore, interventions and provenance can be exercised on both
substrates with physically meaningful signals and no semantics:

  body       two hinge joints, I*qdd = g*(u_plus - u_minus) - b*qd - k(u_co)*q
             + external torque, with soft joint limits; integrated at 1 ms
  motor      a network-independent babbling generator: Poisson-onset bursts per
             actuator channel, smoothed (the stream plays these open-loop and
             reports them as the efference copy; E2 prereg "replay, don't babble")
  vision     a drifting luminance grating, a bright blob at a position set by
             the joint angles (the body is visible), optional cue flashes at
             one of K fixed positions (B4) and optional rhythmic full-field
             flashes (B5), plus pixel noise

Probe targets (true joint state, cue position, flash timing, torque pulses)
are kept in `last_targets`, which evaluators read and substrates never see.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping

import numpy as np

from continuous_brain_protocols import Frame, StreamSpec, sha256_json
from continuous_brain_substrates.encoders import EncoderConfig, sensory_stream_spec


@dataclass(frozen=True)
class BodyConfig:
    inertia: float = 0.02
    gain: float = 1.2
    damping: float = 0.15
    stiffness_base: float = 0.3
    stiffness_co: float = 2.0
    limit_rad: float = 1.4
    limit_stiffness: float = 40.0
    external_torque_rate_hz: float = 0.2
    external_torque_amp: float = 0.6
    external_torque_dur_s: float = 0.05


@dataclass(frozen=True)
class BabbleConfig:
    onset_rate_hz: float = 1.5
    min_dur_s: float = 0.1
    max_dur_s: float = 0.4
    min_amp: float = 0.2
    max_amp: float = 1.0
    smoothing_tau_s: float = 0.05


@dataclass(frozen=True)
class VisionConfig:
    grating_period_px: float = 8.0
    grating_speed_hz: float = 0.5
    grating_contrast: float = 0.3
    blob_sigma_px: float = 1.2
    blob_amp: float = 0.6
    pixel_noise: float = 0.05
    cue_positions: int = 0                    # B4: K > 0 enables cue flashes
    cue_duration_s: float = 0.05
    cue_amp: float = 0.8
    cue_min_interval_s: float = 1.5
    cue_max_interval_s: float = 2.5
    rhythm_period_s: float | None = None      # B5: full-field flash period
    rhythm_flash_s: float = 0.03
    rhythm_amp: float = 0.4


@dataclass(frozen=True)
class StreamConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dt_s: float = 0.01
    body: BodyConfig = field(default_factory=BodyConfig)
    babble: BabbleConfig = field(default_factory=BabbleConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    physics_substeps: int = 10

    def config_hash(self) -> str:
        return sha256_json(asdict(self))

    def with_vision(self, **kw) -> "StreamConfig":
        return replace(self, vision=replace(self.vision, **kw))

    def with_babble(self, **kw) -> "StreamConfig":
        return replace(self, babble=replace(self.babble, **kw))


class SyntheticBodyStream:
    """An EmbodiedStream. Deterministic given (config, seed)."""

    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self._spec = sensory_stream_spec(cfg.encoder, cfg.dt_s)
        side = cfg.encoder.retina_side
        yy, xx = np.meshgrid(np.arange(side, dtype=np.float64), np.arange(side, dtype=np.float64), indexing="ij")
        self._yy, self._xx = yy, xx
        k = cfg.vision.cue_positions
        if k:
            ang = 2 * np.pi * np.arange(k) / k
            r = side * 0.3
            self._cue_xy = np.stack([side / 2 + r * np.cos(ang), side / 2 + r * np.sin(ang)], axis=1)
        self.last_targets: dict[str, np.ndarray] = {}

    @property
    def spec(self) -> StreamSpec:
        return self._spec

    # ------------------------------------------------------------------ state
    def reset(self, seed: int) -> Frame:
        c = self.cfg
        j = c.encoder.n_joints
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.index = -1
        self.q = np.zeros(j)
        self.qd = np.zeros(j)
        self.u = np.zeros((j, c.encoder.motor_channels))
        self.burst_left = np.zeros((j, c.encoder.motor_channels))
        self.burst_amp = np.zeros((j, c.encoder.motor_channels))
        self.torque_left = np.zeros(j)
        self.torque_amp = np.zeros(j)
        self.cue_left = 0.0
        self.cue_pos = -1
        self.next_cue_s = self._draw_cue_interval() if c.vision.cue_positions else math.inf
        self.last_cue_onset_s = -math.inf
        self.last_cue_pos = -1
        return self.next()

    def _draw_cue_interval(self) -> float:
        v = self.cfg.vision
        return float(self.rng.uniform(v.cue_min_interval_s, v.cue_max_interval_s))

    def state_dict(self) -> dict:
        return {"seed": self.seed, "rng": self.rng.bit_generator.state, "index": self.index,
                "q": self.q.copy(), "qd": self.qd.copy(), "u": self.u.copy(), "burst_left": self.burst_left.copy(),
                "burst_amp": self.burst_amp.copy(), "torque_left": self.torque_left.copy(),
                "torque_amp": self.torque_amp.copy(), "cue_left": self.cue_left, "cue_pos": self.cue_pos,
                "next_cue_s": self.next_cue_s, "last_cue_onset_s": self.last_cue_onset_s,
                "last_cue_pos": self.last_cue_pos, "config_hash": self.cfg.config_hash()}

    def load_state_dict(self, s: dict) -> None:
        if s["config_hash"] != self.cfg.config_hash():
            raise ValueError("stream state belongs to a different stream configuration")
        self.seed = s["seed"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = s["rng"]
        self.index = s["index"]
        for k in ("q", "qd", "u", "burst_left", "burst_amp", "torque_left", "torque_amp"):
            setattr(self, k, np.array(s[k]))
        for k in ("cue_left", "cue_pos", "next_cue_s", "last_cue_onset_s", "last_cue_pos"):
            setattr(self, k, s[k])

    # ------------------------------------------------------------------ dynamics
    def _babble(self, dt: float) -> None:
        b = self.cfg.babble
        start = (self.burst_left <= 0) & (self.rng.random(self.u.shape) < b.onset_rate_hz * dt)
        dur = self.rng.uniform(b.min_dur_s, b.max_dur_s, self.u.shape)
        amp = self.rng.uniform(b.min_amp, b.max_amp, self.u.shape)
        self.burst_left = np.where(start, dur, self.burst_left - dt)
        self.burst_amp = np.where(start, amp, self.burst_amp)
        drive = np.where(self.burst_left > 0, self.burst_amp, 0.0)
        a = dt / b.smoothing_tau_s
        self.u = np.clip(self.u + a * (drive - self.u), 0.0, 1.0)

    def _torque(self, dt: float) -> None:
        b = self.cfg.body
        start = (self.torque_left <= 0) & (self.rng.random(self.q.shape) < b.external_torque_rate_hz * dt)
        sign = np.where(self.rng.random(self.q.shape) < 0.5, -1.0, 1.0)
        self.torque_left = np.where(start, b.external_torque_dur_s, self.torque_left - dt)
        self.torque_amp = np.where(start, sign * b.external_torque_amp, self.torque_amp)

    def _physics(self, dt: float) -> None:
        b = self.cfg.body
        n = self.cfg.physics_substeps
        h = dt / n
        ext = np.where(self.torque_left > 0, self.torque_amp, 0.0)
        for _ in range(n):
            k = b.stiffness_base + b.stiffness_co * self.u[:, 2]
            over = np.maximum(np.abs(self.q) - b.limit_rad, 0.0) * np.sign(self.q)
            torque = (b.gain * (self.u[:, 0] - self.u[:, 1]) - b.damping * self.qd - k * self.q
                      - b.limit_stiffness * over + ext)
            self.qd = self.qd + h * torque / b.inertia
            vlo, vhi = self.cfg.encoder.velocity_range_rad_s
            self.qd = np.clip(self.qd, vlo, vhi)
            self.q = self.q + h * self.qd
            alo, ahi = self.cfg.encoder.angle_range_rad
            self.q = np.clip(self.q, alo, ahi)

    def _render(self, t: float) -> np.ndarray:
        v = self.cfg.vision
        side = self.cfg.encoder.retina_side
        lum = 0.5 + v.grating_contrast * np.sin(2 * np.pi * (self._xx / v.grating_period_px - v.grating_speed_hz * t))
        # the body is visible: joint 0 moves the blob horizontally, joint 1 vertically
        span = self.cfg.encoder.angle_range_rad[1]
        bx = side / 2 + (self.q[0] / span) * side * 0.4
        by = side / 2 + (self.q[1 % len(self.q)] / span) * side * 0.4
        lum = lum + v.blob_amp * np.exp(-((self._xx - bx) ** 2 + (self._yy - by) ** 2) / (2 * v.blob_sigma_px ** 2))
        if self.cue_left > 0 and self.cue_pos >= 0:
            cx, cy = self._cue_xy[self.cue_pos]
            lum = lum + v.cue_amp * np.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2 * 1.0 ** 2))
        if v.rhythm_period_s:
            if (t % v.rhythm_period_s) < v.rhythm_flash_s:
                lum = lum + v.rhythm_amp
        return lum + self.rng.normal(0.0, v.pixel_noise, lum.shape)

    def next(self, motor: Mapping[str, np.ndarray] | None = None) -> Frame:
        """Advance one frame. `motor` is ignored: these streams are open-loop by design."""
        c = self.cfg
        dt = c.dt_s
        if self.index >= 0:
            self._babble(dt)
            self._torque(dt)
            self._physics(dt)
        self.index += 1
        t = self.index * dt
        # cue schedule (B4)
        if c.vision.cue_positions:
            self.cue_left -= dt
            if t >= self.next_cue_s:
                self.cue_pos = int(self.rng.integers(c.vision.cue_positions))
                self.cue_left = c.vision.cue_duration_s
                self.last_cue_onset_s, self.last_cue_pos = t, self.cue_pos
                self.next_cue_s = t + self._draw_cue_interval()
        lum = self._render(t)
        values = {"luminance": lum, "joint_angle": self.q.copy(), "joint_velocity": self.qd.copy(),
                  "motor_efference": self.u.copy()}
        period = c.vision.rhythm_period_s
        self.last_targets = {
            "q": self.q.copy(), "qd": self.qd.copy(),
            "external_torque": np.where(self.torque_left > 0, self.torque_amp, 0.0),
            "cue_onset": np.asarray(1.0 if self.last_cue_onset_s == t else 0.0),
            "last_cue_pos": np.asarray(float(self.last_cue_pos)),
            "time_since_cue_s": np.asarray(t - self.last_cue_onset_s if math.isfinite(self.last_cue_onset_s) else -1.0),
            "time_to_flash_s": np.asarray((period - (t % period)) % period if period else -1.0),
            "time_since_flash_s": np.asarray((t % period) if period else -1.0),
        }
        return Frame(self.index, t, values)
