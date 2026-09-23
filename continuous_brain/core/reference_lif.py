"""Reference spiking core: a stand-in for InfantBrain, NOT InfantBrain.

This exists so every other component can be built and tested before the
canonical InfantBrain v0.11.3 is ported from the home machine. Do not train it
and treat the result as project history.

Mechanism (all real, none simplified away):
  * Current-based leaky integrate-and-fire neurons, Euler-integrated at dt.
  * Fixed sparse random recurrent topology obeying Dale's law
    (excitatory / inhibitory presynaptic identity).
  * Absolute refractory period, background drive plus independent noise.
  * Reward-modulated STDP with eligibility traces (Izhikevich 2007):
      pair-based pre/post traces write a decaying eligibility E_ij; weights
      change only when a neuromodulator is applied: dW = lr * m * E.
    Only excitatory synapses are plastic; weights are bounded to [0, w_max].
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..transducer.events import SpikeEvents
from .interface import SpikingCore


@dataclass
class ReferenceLIFConfig:
    n_neurons: int = 512
    n_input: int = 64
    n_projection: int = 128
    frac_excitatory: float = 0.8
    connection_prob: float = 0.1
    dt: float = 0.001
    tau_mem: float = 0.020
    tau_syn: float = 0.005
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_s: float = 0.002
    w_exc: float = 0.05
    w_inh_ratio: float = 4.0
    w_max: float = 0.25
    background_mean: float = 0.85
    noise_std: float = 0.35
    input_gain: float = 1.0
    tau_plus: float = 0.020
    tau_minus: float = 0.020
    a_plus: float = 0.010
    a_minus: float = 0.012
    tau_eligibility: float = 1.0
    learning_rate: float = 1.0
    seed: int = 0


class ReferenceLIFCore(SpikingCore):
    def __init__(self, config: ReferenceLIFConfig | None = None, batch_size: int = 1,
                 device: torch.device | str = "cpu"):
        self.cfg = config or ReferenceLIFConfig()
        c = self.cfg
        self.n_exc = int(round(c.n_neurons * c.frac_excitatory))
        if c.n_input + c.n_projection > self.n_exc:
            raise ValueError("input and projection neurons must fit inside the excitatory pool")
        self.n_input = c.n_input
        self.n_projection = c.n_projection
        self.batch_size = batch_size
        self.dt = c.dt
        self.device = torch.device(device)
        self.input_idx = torch.arange(0, c.n_input, device=self.device)
        self.proj_idx = torch.arange(c.n_input, c.n_input + c.n_projection, device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(c.seed)

        n = c.n_neurons
        is_exc = torch.zeros(n, dtype=torch.bool, device=self.device)
        is_exc[: self.n_exc] = True
        self.sign = torch.where(is_exc, 1.0, -1.0)                      # per presynaptic neuron
        mask = torch.rand(n, n, generator=self.gen, device=self.device) < c.connection_prob
        mask.fill_diagonal_(False)
        self.mask = mask                                                # [post, pre]
        self.plastic = mask & is_exc[None, :]
        jitter = torch.exp(0.3 * torch.randn(n, n, generator=self.gen, device=self.device))
        scale = torch.where(is_exc, c.w_exc, c.w_exc * c.w_inh_ratio)
        self.w = (jitter * scale[None, :] * mask).clamp(max=c.w_max * c.w_inh_ratio)
        self._steps = 0
        self.tracking = False
        self.reset_state()

    # ------------------------------------------------------------------ state

    @property
    def time_s(self) -> float:
        return self._steps * self.dt

    def reset_state(self) -> None:
        b, n = self.batch_size, self.cfg.n_neurons
        z = lambda: torch.zeros(b, n, device=self.device)  # noqa: E731
        self.v, self.i_syn, self.x_pre, self.x_post = z(), z(), z(), z()
        self.last_spikes = z()
        self.refractory = torch.zeros(b, n, dtype=torch.long, device=self.device)
        self.eligibility = torch.zeros(n, n, device=self.device)

    def set_plasticity_tracking(self, enabled: bool) -> None:
        self.tracking = bool(enabled)

    def reseed(self, seed: int) -> None:
        self.gen.manual_seed(int(seed))

    def effective_weights(self) -> torch.Tensor:
        return self.w * self.sign[None, :]

    # ------------------------------------------------------------------ dynamics

    def step(self, input_current: torch.Tensor, duration_s: float) -> SpikeEvents:
        c = self.cfg
        if input_current.shape != (self.batch_size, self.n_input):
            raise ValueError(f"expected input [{self.batch_size}, {self.n_input}], "
                             f"got {tuple(input_current.shape)}")
        n_steps = int(round(duration_s / self.dt))
        if n_steps < 1:
            raise ValueError("duration shorter than one integration step")
        ext = torch.zeros(self.batch_size, c.n_neurons, device=self.device)
        ext[:, self.input_idx] = input_current.to(self.device, torch.float32) * c.input_gain
        w_eff_t = self.effective_weights().T                            # [pre, post]
        d_syn = math.exp(-self.dt / c.tau_syn)
        d_plus = math.exp(-self.dt / c.tau_plus)
        d_minus = math.exp(-self.dt / c.tau_minus)
        d_elig = math.exp(-self.dt / c.tau_eligibility)
        ref_steps = int(round(c.refractory_s / self.dt))
        rasters = []
        for _ in range(n_steps):
            self.i_syn = self.i_syn * d_syn + self.last_spikes @ w_eff_t
            noise = torch.randn(self.batch_size, c.n_neurons, generator=self.gen,
                                device=self.device) * c.noise_std
            drive = self.i_syn + ext + c.background_mean + noise
            active = self.refractory == 0
            self.v = torch.where(active, self.v + (self.dt / c.tau_mem) * (drive - self.v),
                                 torch.full_like(self.v, c.v_reset))
            spikes = (self.v >= c.v_threshold) & active
            s = spikes.to(torch.float32)
            self.v = torch.where(spikes, torch.full_like(self.v, c.v_reset), self.v)
            self.refractory = torch.where(spikes, torch.full_like(self.refractory, ref_steps),
                                          (self.refractory - 1).clamp(min=0))
            if self.tracking:
                # LTP: post spike i pairs with prior pre activity x_pre_j.
                # LTD: pre spike j pairs with prior post activity x_post_i.
                ltp = s.T @ self.x_pre
                ltd = self.x_post.T @ s
                self.eligibility = (self.eligibility * d_elig
                                    + (c.a_plus * ltp - c.a_minus * ltd) * self.plastic)
            self.x_pre = self.x_pre * d_plus + s
            self.x_post = self.x_post * d_minus + s
            self.last_spikes = s
            self._steps += 1
            rasters.append(s[:, self.proj_idx])
        t0 = (self._steps - n_steps) * self.dt
        return SpikeEvents.from_dense(torch.stack(rasters, dim=1), t0, self.dt)

    # ------------------------------------------------------------------ learning

    def apply_plasticity(self, modulator: float) -> dict[str, float]:
        c = self.cfg
        dw = c.learning_rate * float(modulator) * self.eligibility * self.plastic
        before = self.w.clone()
        self.w = torch.where(self.plastic, (self.w + dw).clamp(0.0, c.w_max), self.w)
        change = (self.w - before).abs()
        self.eligibility.zero_()  # consumed: never applied twice
        n_plastic = max(int(self.plastic.sum()), 1)
        return {"mean_abs_dw": float(change.sum() / n_plastic),
                "frac_changed": float((change > 0).sum() / n_plastic),
                "mean_exc_w": float(self.w[self.plastic].mean())}

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "ReferenceLIFCore", "config": asdict(self.cfg),
                "batch_size": self.batch_size, "w": self.w.clone(), "mask": self.mask.clone(),
                "steps": self._steps, "gen": self.gen.get_state(),
                "v": self.v.clone(), "i_syn": self.i_syn.clone(), "x_pre": self.x_pre.clone(),
                "x_post": self.x_post.clone(), "last_spikes": self.last_spikes.clone(),
                "refractory": self.refractory.clone(), "eligibility": self.eligibility.clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "ReferenceLIFCore":
            raise ValueError("state is not from a ReferenceLIFCore")
        if state["config"] != asdict(self.cfg) or state["batch_size"] != self.batch_size:
            raise ValueError("state was saved from a differently-configured core")
        if not torch.equal(state["mask"].to(self.device), self.mask):
            raise ValueError("state topology does not match this core")
        for key in ("w", "v", "i_syn", "x_pre", "x_post", "last_spikes", "refractory", "eligibility"):
            setattr(self, key, state[key].to(self.device).clone())
        self._steps = int(state["steps"])
        self.gen.set_state(state["gen"])
