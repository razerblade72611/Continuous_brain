"""Continuous-time spiking state engine (DAV-32).

Current-based leaky integrate-and-fire neurons, forward-Euler at dt (1 ms):

    g_e <- g_e * exp(-dt/tau_se) + sum_k w_ek * s_src(k)      excitatory drive
    g_i <- g_i * exp(-dt/tau_si) + sum_k w_ik * s_src(k)      inhibitory drive
    v   <- v + dt/tau_m * (v_rest - v + g_e - g_i + I_bg + I_ext + noise)
    spike if v >= theta and not refractory; then v <- v_reset, refractory for t_ref

Excitatory and inhibitory drive are integrated and reported separately, so
high simultaneous E and I activity never collapses into a low net signal.

The homeostatic threshold theta drifts slowly toward a target rate:

    r     <- r + dt/tau_h * (s/dt - r)
    theta <- clamp(theta + gain * dt * (r - r_target), bounds)

Determinism: the anatomy, candidate edges, and weights are generated on CPU
from the seed. Spike propagation gathers each neuron's own K sources (no
atomic adds). Noise is drawn from a seeded CPU generator by default, so every
backend receives identical noise. On one device and backend a seed therefore
reproduces bit-exactly; across backends, parity is statistical (see parity.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .anatomy import Anatomy, build_anatomy
from .config import E0Config
from .connectivity import SparseSlots
from .traces import MultiscaleTraces


class NonFiniteStateError(FloatingPointError):
    pass


class SNNEngine:
    def __init__(self, cfg: E0Config, device: torch.device | str = "cpu", batch_size: int = 1,
                 anatomy: Anatomy | None = None, slots: SparseSlots | None = None):
        self.cfg = cfg
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.anatomy = anatomy or build_anatomy(cfg)
        self.slots = (slots or SparseSlots.build(self.anatomy, cfg.connectivity, cfg.seed)).to(self.device)
        nc = cfg.neuron
        self.dt = nc.dt_s
        n = self.anatomy.n

        is_exc = torch.from_numpy(self.anatomy.is_exc).to(self.device)
        self.tau_m = torch.where(is_exc, nc.tau_mem_exc_s, nc.tau_mem_inh_s).to(torch.float32)
        self.alpha = (self.dt / self.tau_m).to(torch.float32)             # [N]
        self.d_se = math.exp(-self.dt / nc.tau_syn_exc_s)
        self.d_si = math.exp(-self.dt / nc.tau_syn_inh_s)
        self.ref_steps = int(round(nc.refractory_s / self.dt))
        self.pop_mean = self.anatomy.population_mean_matrix().to(self.device)

        noise_dev = torch.device("cpu") if nc.noise_on_cpu else self.device
        self.noise_gen = torch.Generator(device=noise_dev).manual_seed(cfg.seed + 104729)
        self._noise_dev = noise_dev
        self.traces = MultiscaleTraces(n, batch_size, cfg.traces.taus_s, self.device)
        self._w_version = -1
        self.reset_state()

    # ------------------------------------------------------------------ state

    @property
    def n(self) -> int:
        return self.anatomy.n

    @property
    def time_s(self) -> float:
        return self.steps * self.dt

    def reset_state(self) -> None:
        b, n, nc = self.batch_size, self.n, self.cfg.neuron
        z = lambda: torch.zeros(b, n, device=self.device)  # noqa: E731
        self.v, self.g_e, self.g_i = z(), z(), z()
        self.spikes = torch.zeros(b, n, dtype=torch.bool, device=self.device)
        self.refractory = torch.zeros(b, n, dtype=torch.int32, device=self.device)
        self.threshold = torch.full((b, n), nc.threshold_base, device=self.device)
        self.rate_est = torch.full((b, n), nc.target_rate_hz, device=self.device)
        self.traces.x.zero_()
        self.steps = 0

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_version != self.slots.version:
            self._w_e, self._w_i = self.slots.effective()
            self._w_version = self.slots.version
        return self._w_e, self._w_i

    def current_from_populations(self, drive: dict[str, torch.Tensor | float]) -> torch.Tensor:
        """Build an external current [B, N] from {population name: value per member}."""
        cur = torch.zeros(self.batch_size, self.n, device=self.device)
        for name, value in drive.items():
            idx = torch.from_numpy(self.anatomy.pop(name).index).to(self.device)
            cur[:, idx] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return cur

    # ------------------------------------------------------------------ dynamics

    @torch.no_grad()
    def step(self, external: torch.Tensor | None = None) -> torch.Tensor:
        nc = self.cfg.neuron
        w_e, w_i = self._weights()
        pre = self.spikes.to(torch.float32)[:, self.slots.src]            # [B, N, K] gather
        self.g_e = self.g_e * self.d_se + (pre * w_e).sum(dim=-1)
        self.g_i = self.g_i * self.d_si + (pre * w_i).sum(dim=-1)

        noise = torch.randn(self.batch_size, self.n, generator=self.noise_gen,
                            device=self._noise_dev).to(self.device) * nc.noise_std
        drive = self.g_e - self.g_i + nc.background_mean + noise
        if external is not None:
            drive = drive + external
        ready = self.refractory == 0
        v_new = self.v + self.alpha * (nc.v_rest - self.v + drive)
        self.v = torch.where(ready, v_new, torch.full_like(self.v, nc.v_reset))
        spikes = ready & (self.v >= self.threshold)
        self.v = torch.where(spikes, torch.full_like(self.v, nc.v_reset), self.v)
        self.refractory = torch.where(spikes, torch.full_like(self.refractory, self.ref_steps),
                                      (self.refractory - 1).clamp(min=0))
        s = spikes.to(torch.float32)
        if nc.homeostasis:
            self.rate_est = self.rate_est + (self.dt / nc.tau_homeostasis_s) * (s / self.dt - self.rate_est)
            lo, hi = nc.threshold_bounds
            self.threshold = (self.threshold + nc.homeostasis_gain * self.dt
                              * (self.rate_est - nc.target_rate_hz)).clamp(lo, hi)
        self.traces.update(s, self.dt)
        self.spikes = spikes
        self.steps += 1
        return spikes

    def run(self, n_steps: int, input_fn=None, recorder=None) -> None:
        """input_fn(engine) -> external current or None; recorder(engine, spikes)."""
        for _ in range(n_steps):
            ext = input_fn(self) if input_fn is not None else None
            spikes = self.step(ext)
            if recorder is not None:
                recorder(self, spikes)

    def check_finite(self) -> None:
        for name in ("v", "g_e", "g_i", "threshold", "rate_est"):
            t = getattr(self, name)
            if not bool(torch.isfinite(t).all()):
                raise NonFiniteStateError(f"non-finite values in {name} at step {self.steps}")
        if not bool(torch.isfinite(self.traces.x).all()):
            raise NonFiniteStateError(f"non-finite values in traces at step {self.steps}")

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict[str, Any]:
        cpu = lambda t: t.detach().cpu().clone()  # noqa: E731
        return {"steps": self.steps, "batch_size": self.batch_size,
                "v": cpu(self.v), "g_e": cpu(self.g_e), "g_i": cpu(self.g_i),
                "spikes": cpu(self.spikes), "refractory": cpu(self.refractory),
                "threshold": cpu(self.threshold), "rate_est": cpu(self.rate_est),
                "traces": self.traces.state_dict(), "slots": self.slots.state_dict(),
                "noise_gen": self.noise_gen.get_state()}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d["batch_size"] != self.batch_size:
            raise ValueError("checkpoint batch size differs")
        for k in ("v", "g_e", "g_i", "spikes", "refractory", "threshold", "rate_est"):
            setattr(self, k, d[k].to(self.device).clone())
        self.traces.load_state_dict(d["traces"])
        self.slots = SparseSlots.from_state_dict(d["slots"]).to(self.device)
        self._w_version = -1
        self.noise_gen.set_state(d["noise_gen"])
        self.steps = int(d["steps"])

    def state_hash(self) -> str:
        """SHA-256 of the dynamic state; equal hashes mean bit-identical state."""
        import hashlib
        h = hashlib.sha256()
        for k in ("v", "g_e", "g_i", "refractory", "threshold", "rate_est"):
            h.update(np.ascontiguousarray(getattr(self, k).detach().cpu().numpy()).tobytes())
        h.update(np.ascontiguousarray(self.traces.x.detach().cpu().numpy()).tobytes())
        h.update(str(self.steps).encode())
        return h.hexdigest()
