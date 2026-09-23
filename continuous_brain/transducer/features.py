"""Per-neuron temporal features and population synchrony.

Per projection neuron, per cognitive cycle:

    F = [X_1..X_M, dX_1..dX_M, d2X_1..d2X_M, recency, burst]

  X      multiscale traces at the end of the cycle
  dX     change since the previous cycle ("becoming active" vs "active")
  d2X    change of the change
  recency exp(-(t - t_last_spike) / tau_r)
  burst  relu(X_fast - 1): the fast trace exceeds one spike's worth only when
         two or more spikes land within ~tau_fast of each other

All channels are passed through a signed log1p to tame their ranges without
normalizing amplitude away.

Population synchrony uses the Golomb-Rinzel measure over the cycle window:

    chi^2 = Var_t(mean_i V_i(t)) / mean_i Var_t(V_i(t))

with V_i the fast trace. chi^2 -> 1 for fully synchronous populations and
-> ~1/n_members for asynchronous ones.
"""

from __future__ import annotations

import torch

from .traces import TraceWindow


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class FeatureAssembler:
    def __init__(self, n_timescales: int):
        self.n_timescales = n_timescales
        self.reset()

    @property
    def n_channels(self) -> int:
        return 3 * self.n_timescales + 2

    def reset(self) -> None:
        self._prev_x: torch.Tensor | None = None
        self._prev_dx: torch.Tensor | None = None

    def __call__(self, window: TraceWindow) -> torch.Tensor:
        x = window.state
        dx = x - self._prev_x if self._prev_x is not None else torch.zeros_like(x)
        d2x = dx - self._prev_dx if self._prev_dx is not None else torch.zeros_like(x)
        burst = torch.relu(x[..., 0] - 1.0)
        feats = torch.cat([x, dx, d2x, window.recency[..., None], burst[..., None]], dim=-1)
        self._prev_x, self._prev_dx = x, dx
        return signed_log1p(feats)


def population_synchrony(history_fast: torch.Tensor, pool_mean) -> torch.Tensor:
    """Golomb-Rinzel chi^2 per population. history_fast: [B, T, N] -> [B, P].

    `pool_mean(x)` averages the last dimension over each population's members.
    """
    if history_fast.shape[1] < 2:
        return torch.zeros(history_fast.shape[0], pool_mean.n_populations,
                           device=history_fast.device)
    v_pop = pool_mean(history_fast)                               # [B, T, P]
    var_pop = v_pop.var(dim=1, unbiased=False)                    # [B, P]
    mean_var = pool_mean(history_fast.var(dim=1, unbiased=False))  # [B, P]
    chi2 = torch.where(mean_var > 1e-12, var_pop / mean_var.clamp(min=1e-12),
                       torch.zeros_like(var_pop))
    return chi2.clamp(0.0, 1.0)
