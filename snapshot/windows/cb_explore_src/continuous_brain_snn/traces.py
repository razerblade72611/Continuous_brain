"""Multiscale synaptic traces (DAV-33).

For each neuron i and time constant tau_m in {5, 25, 100, 500, 2000} ms:

    x_{i,m}(t + dt) = x_{i,m}(t) * exp(-dt / tau_m) + s_i(t + dt)

The decay factor is recomputed from the actual dt of every update, so the
trace is exact for any dt schedule: decaying through dt1 then dt2 equals
decaying through dt1 + dt2 (the semigroup property the tests check). A spike
arriving at the end of an interval adds exactly 1.

Per-population summaries are means over members, reported separately for the
excitatory and inhibitory subsets so E/I activity never collapses into one
net number.
"""

from __future__ import annotations

import math

import torch


class MultiscaleTraces:
    def __init__(self, n_neurons: int, batch_size: int, taus_s: tuple[float, ...],
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        if any(t <= 0 for t in taus_s):
            raise ValueError("time constants must be positive")
        self.taus_s = tuple(float(t) for t in taus_s)
        self.taus = torch.tensor(self.taus_s, dtype=torch.float64, device=device)
        self.n, self.batch_size, self.dtype = n_neurons, batch_size, dtype
        self.x = torch.zeros(batch_size, n_neurons, len(taus_s), device=device, dtype=dtype)
        self._decay_cache: dict[float, torch.Tensor] = {}

    def _decay(self, dt: float) -> torch.Tensor:
        d = self._decay_cache.get(dt)
        if d is None:
            d = torch.exp(-dt / self.taus).to(self.dtype)       # computed in float64
            if len(self._decay_cache) < 64:
                self._decay_cache[dt] = d
        return d

    def decay(self, dt: float) -> None:
        """Advance by dt with no spikes."""
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if dt > 0:
            self.x.mul_(self._decay(dt))

    def update(self, spikes: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance by dt, then add spikes [B, N] (0/1) that occurred at the end of the step."""
        self.decay(dt)
        self.x.add_(spikes.to(self.dtype)[..., None])
        return self.x

    def population_summary(self, pop_mean: torch.Tensor) -> torch.Tensor:
        """[B, N, M] traces -> [B, P, M] means, via a sparse [P, N] averaging matrix."""
        b, n, m = self.x.shape
        flat = self.x.permute(1, 0, 2).reshape(n, b * m).to(torch.float32)
        out = torch.sparse.mm(pop_mean.to(flat.device), flat)          # [P, B*M]
        return out.reshape(-1, b, m).permute(1, 0, 2)

    @staticmethod
    def steady_state_rate_hz(trace_value: float, tau_s: float) -> float:
        """Invert the mean trace of a Poisson train: E[x] = rate * tau (dt << tau)."""
        return trace_value / tau_s

    def state_dict(self) -> dict:
        return {"taus_s": self.taus_s, "x": self.x.detach().cpu().clone()}

    def load_state_dict(self, d: dict) -> None:
        if tuple(d["taus_s"]) != self.taus_s:
            raise ValueError("trace time constants differ from this instance")
        self.x = d["x"].to(self.x.device, self.dtype).clone()

    @staticmethod
    def analytic_single_spike(t_since_s: float, tau_s: float) -> float:
        return math.exp(-t_since_s / tau_s)
