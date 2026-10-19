from __future__ import annotations

"""DirectML-safe recurrent cells for Continuous Brain v0.11.2.

PyTorch's fused GRUCell path may dispatch ``aten::_thnn_fused_gru_cell`` on
PrivateUse1/DirectML.  Some torch-directml builds attempt a CPU fallback for
that operator and then fail because the fused CPU kernel is unavailable.  This
module preserves the standard ``nn.GRUCell`` parameters/state-dict schema while
using the algebraically equivalent unfused GRU equations only on DirectML.
"""

import torch
from torch import nn
from torch.nn import functional as F


_DML_TYPES = {"privateuseone", "privateuse1"}


def is_directml_tensor(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor) and x.device.type in _DML_TYPES


def unfused_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
) -> torch.Tensor:
    """Algebraically equivalent implementation of ``torch.nn.GRUCell``.

    PyTorch stores gates in reset/update/new order.  Keeping the same ordering
    and parameter tensors makes this function checkpoint-compatible with
    ``nn.GRUCell``.
    """
    gi = F.linear(input, weight_ih, bias_ih)
    gh = F.linear(hx, weight_hh, bias_hh)

    # v0.12.1: use plain slices rather than Tensor.chunk on the DirectML-safe
    # algebraic path.  Forward values are identical, while this avoids depending
    # on split/chunk backward kernels in PrivateUse1.
    h = int(hx.shape[-1])
    if gi.shape[-1] != 3 * h or gh.shape[-1] != 3 * h:
        raise RuntimeError(
            f"GRU gate shape mismatch: gi={tuple(gi.shape)} gh={tuple(gh.shape)} hidden={h}"
        )
    i_r, i_z, i_n = gi[..., :h], gi[..., h:2 * h], gi[..., 2 * h:3 * h]
    h_r, h_z, h_n = gh[..., :h], gh[..., h:2 * h], gh[..., 2 * h:3 * h]
    reset = torch.sigmoid(i_r + h_r)
    update = torch.sigmoid(i_z + h_z)
    new = torch.tanh(i_n + reset * h_n)
    return new + update * (hx - new)


class DMLSafeGRUCell(nn.GRUCell):
    """Drop-in GRUCell with an unfused DirectML forward path.

    CPU/CUDA behavior remains the native PyTorch implementation.  Parameter
    names, shapes and initialization are inherited unchanged from nn.GRUCell.
    """

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> torch.Tensor:
        if not is_directml_tensor(input):
            return super().forward(input, hx)

        if input.dim() not in (1, 2):
            raise ValueError(f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        elif hx.dim() not in (1, 2):
            raise ValueError(f"GRUCell: Expected hx to be 1D or 2D, got {hx.dim()}D instead")
        elif not is_batched:
            hx = hx.unsqueeze(0)

        if hx.device != input.device:
            raise RuntimeError(f"GRUCell input/hx device mismatch: {input.device} vs {hx.device}")

        out = unfused_gru_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih if self.bias else None,
            self.bias_hh if self.bias else None,
        )
        return out if is_batched else out.squeeze(0)
