from __future__ import annotations

"""Small DirectML-safe tensor helpers introduced in v0.12.1.

These helpers preserve the historical CPU/CUDA implementation whenever possible
and only avoid operators with observed PrivateUse1/DirectML incompatibilities on
the user's local stack.
"""

import torch
from torch.nn import functional as F

from .dml_recurrent_v112 import is_directml_tensor


def safe_one_hot(indices: torch.Tensor, num_classes: int, *, dtype=torch.float32) -> torch.Tensor:
    """One-hot encode valid integer indices without DirectML scatter.

    CPU/CUDA retain ``F.one_hot`` exactly.  On DirectML, equality against a class
    range is algebraically identical for valid indices and avoids the scatter
    kernel that fails on affected torch-directml builds.
    """
    idx = indices.long()
    n = int(num_classes)
    if n <= 0:
        raise ValueError("num_classes must be positive")
    if is_directml_tensor(idx):
        classes = torch.arange(n, device=idx.device, dtype=idx.dtype)
        return (idx.unsqueeze(-1) == classes).to(dtype=dtype)
    return F.one_hot(idx, num_classes=n).to(dtype=dtype)
