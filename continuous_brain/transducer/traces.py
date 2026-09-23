"""Multiscale synaptic traces: the spike -> continuous interface.

For each projection neuron i and time constant tau_m:

    x_{i,m}(k) = x_{i,m}(k-1) * exp(-bin/tau_m) + s_i(k)

This is the spike-response-model featurization (a bank of leaky integrators).
It is a compressed multiscale summary of recency and burst structure, not a
lossless record of spike times; timing is quantized to one integration bin.

The recurrence is linear, so a whole cognitive-cycle window is integrated in
one parallel step with a lower-triangular decay kernel instead of a
sequential per-event loop (which does not parallelize on a GPU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .events import SpikeEvents


@dataclass
class TraceWindow:
    t_start: float
    t_end: float
    state: torch.Tensor          # [B, N, M] traces at t_end
    history_fast: torch.Tensor   # [B, T, N] fastest trace at each bin (for synchrony)
    recency: torch.Tensor        # [B, N] exp(-(t_end - t_last_spike)/tau_r), 0 if never spiked
    spike_counts: torch.Tensor   # [B, N] spikes in this window


class MultiscaleTraceBank:
    def __init__(self, n_neurons: int, batch_size: int, taus_s: tuple[float, ...],
                 bin_s: float, recency_tau_s: float, device: torch.device | str = "cpu"):
        if bin_s <= 0 or any(t <= 0 for t in taus_s):
            raise ValueError("bin and time constants must be positive")
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.bin_s = bin_s
        self.recency_tau_s = recency_tau_s
        self.device = torch.device(device)
        self.taus = torch.tensor(taus_s, dtype=torch.float32, device=self.device)
        self.decay = torch.exp(-bin_s / self.taus)                  # [M]
        self._kernels: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reset()

    @property
    def n_timescales(self) -> int:
        return int(self.taus.shape[0])

    def reset(self, t: float = 0.0) -> None:
        self.state = torch.zeros(self.batch_size, self.n_neurons, self.n_timescales, device=self.device)
        self.last_spike = torch.full((self.batch_size, self.n_neurons), -math.inf,
                                     dtype=torch.float64, device=self.device)
        self.t = float(t)

    def _kernel(self, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n_bins not in self._kernels:
            s = torch.arange(n_bins, device=self.device)
            lag = (s[:, None] - s[None, :]).to(torch.float32)          # [T, T]  s - k
            causal = (lag >= 0).to(torch.float32)
            log_d = torch.log(self.decay)                               # [M]
            # L[m, s, k] = d_m^(s-k) for k <= s
            L = torch.exp(log_d[:, None, None] * lag.clamp(min=0)[None]) * causal[None]
            # carry[m, s] = d_m^(s+1): contribution of the state entering the window
            carry = torch.exp(log_d[:, None] * (s[None, :].to(torch.float32) + 1.0))
            self._kernels[n_bins] = (L, carry)
        return self._kernels[n_bins]

    def advance(self, events: SpikeEvents, t_end: float) -> TraceWindow:
        if events.n_neurons != self.n_neurons or events.batch_size != self.batch_size:
            raise ValueError("events do not match this trace bank's shape")
        n_bins = int(round((t_end - self.t) / self.bin_s))
        if n_bins < 1:
            raise ValueError("window must span at least one integration bin")
        spikes = events.to(self.device).bin(self.t, n_bins, self.bin_s)         # [B, T, N]
        L, carry = self._kernel(n_bins)
        hist = torch.einsum("bkn,msk->bsnm", spikes, L)                           # [B, T, N, M]
        hist = hist + self.state[:, None, :, :] * carry.T[None, :, None, :]
        self.state = hist[:, -1].contiguous()

        if events.n_events:
            flat = events.batch.to(self.device) * self.n_neurons + events.neuron.to(self.device)
            latest = self.last_spike.flatten().scatter_reduce(
                0, flat, events.time.to(self.device), reduce="amax", include_self=True)
            self.last_spike = latest.view(self.batch_size, self.n_neurons)
        age = (t_end - self.last_spike).clamp(min=0.0)
        # exp(-inf) = 0 for neurons that have never spiked.
        recency = torch.exp(-age / self.recency_tau_s).to(torch.float32)

        window = TraceWindow(self.t, t_end, self.state, hist[..., 0].contiguous(),
                             recency, spikes.sum(dim=1))
        self.t = float(t_end)
        return window
