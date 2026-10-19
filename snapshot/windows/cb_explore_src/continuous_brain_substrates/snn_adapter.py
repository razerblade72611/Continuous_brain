"""Substrate-API adapter over the existing E0 SNN engine (C6: wrap, do not modify).

The engine in `continuous_brain_snn` is used exactly as E0 built and gated it.
This adapter adds only what the common API needs:

  * transduction   physical frame -> shared encoder -> input currents on the
                   matching sensory populations (retina ON/OFF, proprioception);
                   the efference copy drives the matching motor populations
  * substeps       step(frame, dt) holds the frame constant for dt / engine.dt
                   Euler steps (1 ms each by default)
  * motor output   motor population rates over the frame, divided by a fixed
                   physical scale, clipped to [0, 1]
  * features       per-neuron spike trains filtered at tau = 25 ms (the E0
                   trace bank's 25 ms component), as the E2 prereg specifies
  * freeze         homeostatic threshold adaptation (the only adaptive process
                   in E0) is undone after every step, so frozen dynamics are exact
  * lesion         synapse or neuron removal through the slot masks; lesioned
                   neurons are also silenced
  * gradient firewall (Charter invariant 3, C2): `train_step` always raises
                   GradientFirewallError, and every engine tensor is checked
                   for requires_grad at construction, load, and on demand

Local plasticity does not exist in E0. `plasticity_step` raises
CapabilityUnavailable until E1 (DAV-40) adds it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

import continuous_brain_snn
from continuous_brain_protocols import (BudgetReport, Capability, CapabilityUnavailable, FeatureSpec,
                                        GradientFirewallError, LesionRecord, LesionSpec, PlasticityReport,
                                        StreamSpec, Substrate, SubstrateMetadata, TrainReport, read_only,
                                        sha256_json, state_digest)
from continuous_brain_snn.anatomy import REGIONS
from continuous_brain_snn.engine import SNNEngine

from .encoders import EncoderConfig, SensoryEncoder, motor_stream_spec, sensory_stream_spec
from .memory import peak_rss_bytes

ADAPTER_VERSION = "snn-adapter/1"
_MOTOR_CHANNELS = ("plus", "minus", "co")


@dataclass(frozen=True)
class SNNAdapterConfig:
    retina_gain: float = 6.0             # same gains as the E0 gate's SyntheticDrive
    proprio_gain: float = 1.2
    efference_gain: float = 1.5
    motor_rate_scale_hz: float = 50.0    # rate that maps to a full (1.0) command
    feature_tau_s: float = 0.025
    silence_current: float = -50.0       # clamps lesioned neurons far below threshold

    def config_hash(self) -> str:
        return sha256_json(asdict(self))


class SpikeTraceFeatures:
    """Per-neuron spike trains filtered at feature_tau_s (a FeatureSource)."""

    def __init__(self, engine: SNNEngine, tau_s: float):
        taus = np.asarray(engine.traces.taus_s)
        idx = int(np.argmin(np.abs(taus - tau_s)))
        if abs(taus[idx] - tau_s) > 1e-9:
            raise ValueError(f"no trace with tau {tau_s} s (have {taus.tolist()})")
        self.engine, self.tau_index, self.tau_s = engine, idx, float(taus[idx])

    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec("snn_spike_trace", self.engine.n,
                           f"per-neuron spike train filtered with tau={self.tau_s * 1e3:.0f} ms",
                           filter_tau_s=self.tau_s)

    def read_features(self) -> np.ndarray:
        return self.engine.traces.x[0, :, self.tau_index].detach().cpu().numpy().astype(np.float64)


class SNNSubstrate(Substrate):
    def __init__(self, engine: SNNEngine, cfg: SNNAdapterConfig | None = None, frame_dt_s: float = 0.01):
        if engine.batch_size != 1:
            raise ValueError("the substrate adapter models one individual: engine batch_size must be 1")
        self.engine = engine
        self.cfg = cfg or SNNAdapterConfig()
        self.encoder_cfg = EncoderConfig.from_anatomy_config(engine.cfg.anatomy)
        self.encoder = SensoryEncoder(self.encoder_cfg)
        self.frame_dt_s = frame_dt_s
        self._in_spec = sensory_stream_spec(self.encoder_cfg, frame_dt_s)
        self._out_spec = motor_stream_spec(self.encoder_cfg, frame_dt_s)
        self._check_anatomy_matches_encoder()
        dev = engine.device
        a = engine.anatomy
        self._block_index = {name: torch.from_numpy(a.pop(name).index).to(dev)
                             for name, _ in self.encoder.blocks() if name != "efference"}
        self._motor_index = [[torch.from_numpy(a.pop(f"motor_{ch}_j{j}").index).to(dev)
                              for ch in _MOTOR_CHANNELS] for j in range(self.encoder_cfg.n_joints)]
        self._features = SpikeTraceFeatures(engine, self.cfg.feature_tau_s)
        self._frozen_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._silenced = torch.zeros(engine.n, dtype=torch.bool, device=dev)
        self._lesion_log: list[dict] = []
        self._frames = 0
        self._wall_s = 0.0
        self.assert_gradient_firewall()

    # ------------------------------------------------------------------ identity
    def _check_anatomy_matches_encoder(self) -> None:
        a = self.engine.anatomy
        for j in range(self.encoder_cfg.n_joints):
            for q, centres in (("angle", self.encoder.angle_centres), ("velocity", self.encoder.velocity_centres)):
                pop = a.pop(f"proprio_{q}_j{j}")
                if not np.allclose(a.preferred[pop.index], centres, atol=1e-5):
                    raise ValueError(f"anatomy tuning centres for {pop.name} differ from the shared encoder")
        for name, size in self.encoder.blocks():
            if name != "efference" and a.pop(name).index.size != size:
                raise ValueError(f"population {name} size differs from the encoder block")

    @property
    def metadata(self) -> SubstrateMetadata:
        caps = frozenset({Capability.HOMEOSTASIS, Capability.LESION, Capability.FREEZE,
                          Capability.SPONTANEOUS_ACTIVITY, Capability.EXACT_RESTORE})
        return SubstrateMetadata(
            kind="snn-e0", track="H1", seed=self.engine.cfg.seed,
            config_hash=sha256_json({"engine": self.engine.cfg.config_hash(), "adapter": self.cfg.config_hash(),
                                     "encoder": self.encoder_cfg.config_hash(), "frame_dt_s": self.frame_dt_s}),
            input_spec_hash=self._in_spec.spec_hash(),
            version=f"{continuous_brain_snn.__version__}+{ADAPTER_VERSION}", capabilities=caps)

    def input_spec(self) -> StreamSpec:
        return self._in_spec

    def output_spec(self) -> StreamSpec:
        return self._out_spec

    # ------------------------------------------------------------------ dynamics
    def reset(self, seed: int | None = None) -> None:
        self.engine.reset_state()
        if seed is not None:
            self.engine.noise_gen.manual_seed(seed + 104729)
        if self._frozen_state is not None:
            self._frozen_state = (self.engine.threshold.clone(), self.engine.rate_est.clone())

    def _substeps(self, dt: float) -> int:
        n = int(round(dt / self.engine.dt))
        if n < 1 or abs(n * self.engine.dt - dt) > 1e-9:
            raise ValueError(f"dt={dt} must be a positive multiple of the engine step {self.engine.dt}")
        return n

    def _current(self, frame: Mapping[str, np.ndarray]) -> torch.Tensor:
        enc = self.encoder.encode(frame)
        e = self.engine
        cur = torch.zeros(1, e.n, device=e.device)
        for name, idx in self._block_index.items():
            gain = self.cfg.retina_gain if name.startswith("retina") else self.cfg.proprio_gain
            cur[0, idx] = torch.as_tensor(gain * enc[name], dtype=torch.float32, device=e.device)
        eff = enc["efference"].reshape(self.encoder_cfg.n_joints, len(_MOTOR_CHANNELS))
        for j, chans in enumerate(self._motor_index):
            for c, idx in enumerate(chans):
                cur[0, idx] = cur[0, idx] + float(self.cfg.efference_gain * eff[j, c])
        if bool(self._silenced.any()):
            cur[0, self._silenced] = self.cfg.silence_current
        return cur

    def step(self, frame: Mapping[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        problems = self._in_spec.validate(frame)
        if problems:
            raise ValueError("invalid frame: " + "; ".join(problems))
        t0 = time.perf_counter()
        n_sub = self._substeps(dt)
        cur = self._current(frame)
        counts = torch.zeros(self.encoder_cfg.n_joints, len(_MOTOR_CHANNELS), device=self.engine.device)
        for _ in range(n_sub):
            spikes = self.engine.step(cur)[0]
            for j, chans in enumerate(self._motor_index):
                for c, idx in enumerate(chans):
                    counts[j, c] += spikes[idx].sum()
            if self._frozen_state is not None:
                self.engine.threshold.copy_(self._frozen_state[0])
                self.engine.rate_est.copy_(self._frozen_state[1])
        sizes = torch.tensor([[idx.numel() for idx in chans] for chans in self._motor_index],
                             dtype=torch.float32, device=self.engine.device)
        rate_hz = counts / (sizes * n_sub * self.engine.dt)
        command = (rate_hz / self.cfg.motor_rate_scale_hz).clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
        self._frames += 1
        self._wall_s += time.perf_counter() - t0
        return {"motor_command": command}

    def expose_state(self) -> Mapping[str, np.ndarray]:
        e = self.engine
        grab = lambda t: read_only(t.detach().cpu().numpy())  # noqa: E731
        return {"v": grab(e.v[0]), "g_e": grab(e.g_e[0]), "g_i": grab(e.g_i[0]),
                "threshold": grab(e.threshold[0]), "rate_est": grab(e.rate_est[0]),
                "spikes": grab(e.spikes[0]), "refractory": grab(e.refractory[0]),
                "traces": grab(e.traces.x[0]), "step": read_only(np.asarray(e.steps))}

    def features(self) -> SpikeTraceFeatures:
        return self._features

    # ------------------------------------------------------------------ learning (C2)
    def plasticity_step(self, modulators: Mapping[str, float] | None = None) -> PlasticityReport:
        raise CapabilityUnavailable(Capability.LOCAL_PLASTICITY,
                                    "E0 has no local plasticity; STDP arrives with E1 (DAV-40)")

    def train_step(self, batch: Any = None) -> TrainReport:
        self.assert_gradient_firewall()
        raise GradientFirewallError("the SNN never receives gradient updates (Charter invariant 3)")

    def freeze(self) -> None:
        self._frozen_state = (self.engine.threshold.clone(), self.engine.rate_est.clone())

    def unfreeze(self) -> None:
        self._frozen_state = None

    @property
    def frozen(self) -> bool:
        return self._frozen_state is not None

    def assert_gradient_firewall(self) -> None:
        offenders = []
        for owner, obj in (("engine", self.engine), ("slots", self.engine.slots), ("traces", self.engine.traces)):
            for name, val in vars(obj).items():
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    offenders.append(f"{owner}.{name}")
        if offenders:
            raise GradientFirewallError(f"SNN tensors require grad: {offenders}")

    # ------------------------------------------------------------------ damage, budget, persistence
    def lesion(self, spec: LesionSpec) -> LesionRecord:
        what, region = spec.target.split(":", 1)
        if region not in REGIONS:
            raise ValueError(f"unknown region {region!r}")
        e = self.engine
        rng = np.random.default_rng(spec.seed)
        in_region = torch.from_numpy(e.anatomy.region == REGIONS.index(region)).to(e.device)
        slots = e.slots
        if what == "synapses":
            post, slot = torch.nonzero(slots.active & in_region[:, None], as_tuple=True)
            total = int(post.numel())
            k = int(round(spec.fraction * total))
            pick = torch.from_numpy(np.sort(rng.choice(total, size=k, replace=False))).to(e.device)
            if k:
                slots.set_active(post[pick], slot[pick], False)
            removed, detail = k, {"synapses_removed": k}
        elif what == "neurons":
            members = torch.nonzero(in_region).flatten()
            total = int(members.numel())
            k = int(round(spec.fraction * total))
            chosen = members[torch.from_numpy(np.sort(rng.choice(total, size=k, replace=False))).to(e.device)]
            dead = torch.zeros(e.n, dtype=torch.bool, device=e.device)
            dead[chosen] = True
            hit = slots.active & (dead[:, None] | dead[slots.src])
            post, slot = torch.nonzero(hit, as_tuple=True)
            if post.numel():
                slots.set_active(post, slot, False)
            self._silenced |= dead
            removed, detail = k, {"neurons_removed": k, "synapses_removed": int(post.numel())}
        else:
            raise ValueError(f"SNN lesions target 'synapses:<region>' or 'neurons:<region>', not {what!r}")
        rec = LesionRecord(spec.spec_hash(), spec.target, spec.fraction, removed, total, detail)
        self._lesion_log.append(asdict(rec))
        return rec

    def budget(self) -> BudgetReport:
        active_mask = self.engine.slots.active
        active = int(active_mask.sum())
        cortex_region = REGIONS.index("cortex")
        region = torch.as_tensor(self.engine.anatomy.region, device=self.engine.device)
        post_cortex = region == cortex_region
        pre_cortex = region[self.engine.slots.src] == cortex_region
        cortex_to_cortex = int((active_mask & post_cortex[:, None] & pre_cortex).sum())
        return BudgetReport(
            kind="snn-e0", learned_scalars=active, active_synapses=active, parameter_count=active,
            frames=self._frames, simulated_s=self.engine.time_s, wall_s=self._wall_s,
            peak_memory_bytes=peak_rss_bytes(self.engine.device), device=str(self.engine.device),
            notes={"trained_parameters_in_E0": 0, "homeostatic_thresholds": self.engine.n,
                   "neurons": self.engine.n, "slot_capacity": self.engine.slots.n * self.engine.slots.capacity,
                   "cortex_to_cortex_active_synapses": cortex_to_cortex})

    def _adapter_state(self) -> dict:
        return {"frozen": self.frozen,
                "frozen_threshold": None if self._frozen_state is None else self._frozen_state[0].cpu().clone(),
                "frozen_rate_est": None if self._frozen_state is None else self._frozen_state[1].cpu().clone(),
                "silenced": self._silenced.cpu().clone(), "frames": self._frames,
                "lesion_log": [dict(x) for x in self._lesion_log]}

    def state_dict(self) -> dict:
        return {"format": ADAPTER_VERSION, "metadata": self.metadata.as_dict(),
                "engine": self.engine.state_dict(), "adapter": self._adapter_state()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("format") != ADAPTER_VERSION:
            raise ValueError(f"not an {ADAPTER_VERSION} state")
        if state["metadata"]["config_hash"] != self.metadata.config_hash:
            raise ValueError("state was saved from a differently configured SNN substrate")
        self.engine.load_state_dict(state["engine"])
        a = state["adapter"]
        dev = self.engine.device
        self._frozen_state = None if not a["frozen"] else (a["frozen_threshold"].to(dev).clone(),
                                                           a["frozen_rate_est"].to(dev).clone())
        self._silenced = a["silenced"].to(dev).clone()
        self._frames = int(a["frames"])
        self._lesion_log = [dict(x) for x in a["lesion_log"]]
        self._features = SpikeTraceFeatures(self.engine, self.cfg.feature_tau_s)
        self.assert_gradient_firewall()

    def state_hash(self) -> str:
        a = self._adapter_state()
        # engine.state_hash() omits the noise generator and the last spike vector; both shape the next step
        return sha256_json({"engine": self.engine.state_hash(),
                            "connectivity": self.engine.slots.connectivity_hash(),
                            "noise_gen": state_digest(self.engine.noise_gen.get_state()),
                            "spikes": state_digest(self.engine.spikes),
                            "adapter": state_digest(a)})
