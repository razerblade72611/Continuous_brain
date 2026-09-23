"""Sparse, event-native spike representation.

Spikes are kept as (batch, neuron, time) triples rather than a dense [B, T, N]
tensor. Densification happens only inside one short cognitive-cycle window,
in `SpikeEvents.bin`, just before trace integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpikeEvents:
    batch: torch.Tensor   # [E] long
    neuron: torch.Tensor  # [E] long, index into the core's projection neurons
    time: torch.Tensor    # [E] float64, absolute simulation time in seconds
    value: torch.Tensor   # [E] float, normally 1.0
    batch_size: int
    n_neurons: int

    def __post_init__(self) -> None:
        # Absolute time must be float64: a continuously running core passes
        # 86,400 s per day, where float32 resolution (~8 ms) is coarser than a
        # 1 ms integration bin.
        self.time = self.time.to(torch.float64)
        n = self.batch.shape[0]
        if not (self.neuron.shape[0] == self.time.shape[0] == self.value.shape[0] == n):
            raise ValueError("SpikeEvents fields must all have the same length")
        if n:
            if int(self.batch.min()) < 0 or int(self.batch.max()) >= self.batch_size:
                raise ValueError("batch index out of range")
            if int(self.neuron.min()) < 0 or int(self.neuron.max()) >= self.n_neurons:
                raise ValueError("neuron index out of range")

    @classmethod
    def empty(cls, batch_size: int, n_neurons: int,
              device: torch.device | str = "cpu") -> "SpikeEvents":
        z = torch.zeros(0, dtype=torch.long, device=device)
        return cls(z, z.clone(), torch.zeros(0, dtype=torch.float64, device=device),
                   torch.zeros(0, device=device), batch_size, n_neurons)

    @classmethod
    def from_dense(cls, spikes: torch.Tensor, t0: float, dt: float) -> "SpikeEvents":
        """Build events from a dense [B, T, N] 0/1 raster whose step k ends at t0+(k+1)*dt."""
        b, k, n = torch.nonzero(spikes > 0, as_tuple=True)
        t = t0 + (k.to(torch.float64) + 1.0) * dt
        return cls(b, n, t, spikes[b, k, n].to(torch.float32),
                   spikes.shape[0], spikes.shape[2])

    @property
    def n_events(self) -> int:
        return int(self.batch.shape[0])

    @property
    def device(self) -> torch.device:
        return self.batch.device

    def to(self, device: torch.device | str) -> "SpikeEvents":
        return SpikeEvents(self.batch.to(device), self.neuron.to(device), self.time.to(device),
                           self.value.to(device), self.batch_size, self.n_neurons)

    @staticmethod
    def cat(parts: list["SpikeEvents"]) -> "SpikeEvents":
        if not parts:
            raise ValueError("nothing to concatenate")
        b, n = parts[0].batch_size, parts[0].n_neurons
        if any(p.batch_size != b or p.n_neurons != n for p in parts):
            raise ValueError("cannot concatenate events from differently-shaped sources")
        return SpikeEvents(torch.cat([p.batch for p in parts]), torch.cat([p.neuron for p in parts]),
                           torch.cat([p.time for p in parts]), torch.cat([p.value for p in parts]), b, n)

    def bin(self, t0: float, n_bins: int, bin_s: float) -> torch.Tensor:
        """Densify into [B, n_bins, N]; bin k covers (t0 + k*bin_s, t0 + (k+1)*bin_s].

        Events more than half a bin outside the window are a caller bug and raise.
        """
        out = torch.zeros(self.batch_size, n_bins, self.n_neurons, device=self.device)
        if self.n_events == 0:
            return out
        rel = (self.time - t0) / bin_s
        # A spike exactly on a bin's end edge belongs to that bin; the 1e-3-bin
        # tolerance absorbs float64 rounding in accumulated simulation time.
        idx = torch.ceil(rel - 1e-3).long() - 1
        if bool(((rel < -0.5) | (rel > n_bins + 0.5)).any()):
            raise ValueError("spike events fall outside the requested window")
        idx = idx.clamp(0, n_bins - 1)
        out.index_put_((self.batch, idx, self.neuron), self.value.to(out.dtype), accumulate=True)
        return out
