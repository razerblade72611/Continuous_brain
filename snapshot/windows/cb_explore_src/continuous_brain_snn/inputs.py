"""Synthetic physical inputs for E0 stability runs.

Only physically observable quantities: luminance on the retinal sheet and
joint angle/velocity for proprioception. No labels, no object identities.

  retina         a drifting luminance grating plus pixel noise, passed through
                 a difference-of-Gaussians centre-surround filter; ON cells take
                 the positive part, OFF cells the negative part
  proprioception Gaussian tuning curves over each cell's preferred angle or
                 velocity, for sinusoidal joint trajectories
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .engine import SNNEngine


def _dog_kernel(sigma_c: float = 0.8, sigma_s: float = 2.0, radius: int = 5) -> torch.Tensor:
    r = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    d2 = xx ** 2 + yy ** 2
    c = torch.exp(-d2 / (2 * sigma_c ** 2)); c /= c.sum()
    s = torch.exp(-d2 / (2 * sigma_s ** 2)); s /= s.sum()
    return (c - s)[None, None]


class SyntheticDrive:
    def __init__(self, engine: SNNEngine, seed: int = 0, retina_gain: float = 6.0,
                 proprio_gain: float = 1.2, grating_period_cells: float = 8.0,
                 grating_speed_hz: float = 1.0, pixel_noise: float = 0.1):
        self.e = engine
        a = engine.anatomy
        self.side = a.config.retina_side
        self.on = torch.from_numpy(a.pop("retina_on").index).to(engine.device)
        self.off = torch.from_numpy(a.pop("retina_off").index).to(engine.device)
        self.kernel = _dog_kernel().to(engine.device)
        self.gen = torch.Generator().manual_seed(seed + 2718)
        self.retina_gain, self.proprio_gain = retina_gain, proprio_gain
        self.period, self.speed, self.pixel_noise = grating_period_cells, grating_speed_hz, pixel_noise
        self.proprio = []
        for p in a.populations:
            if p.region == "proprioception":
                pref = torch.from_numpy(a.preferred[p.index]).to(engine.device)
                lo, hi = p.meta["range"]
                width = (hi - lo) / max(p.index.size - 1, 1) * 1.5
                self.proprio.append((torch.from_numpy(p.index).to(engine.device), pref, width,
                                     p.meta["joint"], p.meta["quantity"]))
        yy, xx = torch.meshgrid(torch.arange(self.side), torch.arange(self.side), indexing="ij")
        self.xx = xx.to(engine.device, torch.float32)

    def luminance(self, t: float) -> torch.Tensor:
        phase = 2 * math.pi * (self.xx / self.period - self.speed * t)
        noise = torch.randn(self.side, self.side, generator=self.gen).to(self.e.device) * self.pixel_noise
        return 0.5 + 0.5 * torch.sin(phase) + noise

    def joint_state(self, t: float, joint: int) -> tuple[float, float]:
        w = 2 * math.pi * (0.5 + 0.2 * joint)
        return 1.0 * math.sin(w * t), 1.0 * w * math.cos(w * t)

    def __call__(self, engine: SNNEngine) -> torch.Tensor:
        t = engine.time_s
        cur = torch.zeros(engine.batch_size, engine.n, device=engine.device)
        lum = self.luminance(t)[None, None]
        dog = torch.nn.functional.conv2d(lum, self.kernel, padding=self.kernel.shape[-1] // 2)[0, 0]
        # anatomy lays the sheet out with grid x as the slow index: flatten matches
        dog = dog.T.reshape(-1)
        cur[:, self.on] = self.retina_gain * torch.relu(dog)
        cur[:, self.off] = self.retina_gain * torch.relu(-dog)
        for idx, pref, width, joint, quantity in self.proprio:
            angle, vel = self.joint_state(t, joint)
            value = angle if quantity == "angle" else vel
            cur[:, idx] = self.proprio_gain * torch.exp(-0.5 * ((pref - value) / width) ** 2)
        return cur
