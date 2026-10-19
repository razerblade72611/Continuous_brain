from __future__ import annotations

"""DirectML-safe Transformer helpers for Continuous Brain v0.11.3.

Some torch-directml/PyTorch combinations represent boolean attention masks via
an internal uint8 path. PyTorch's automatic causal-mask detection may then try
to create ``-inf`` in that dtype and fail before attention executes. DirectML
also lacks several fused Transformer fast paths and may attempt slow or broken
CPU fallbacks.

These helpers do not introduce parameters and do not alter checkpoint schemas.
CPU/CUDA callers keep the ordinary PyTorch path. DirectML callers use an
explicit floating additive causal mask, provide the causal hint, and temporarily
disable MHA/Transformer fast paths so execution is decomposed into ordinary
PyTorch operators supported by the backend.
"""

from contextlib import contextmanager
import torch

from .dml_recurrent_v112 import is_directml_tensor


def directml_causal_mask(length: int, reference: torch.Tensor) -> torch.Tensor:
    """Return an additive upper-triangular causal mask on reference.device.

    The mask deliberately uses the reference floating dtype and a large finite
    negative value rather than a bool/uint8 mask or ``-inf``.  Finite values are
    sufficient to suppress future-token softmax probability at the dtypes used
    by this project while avoiding DirectML's problematic uint8 overflow path.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if not reference.dtype.is_floating_point:
        raise TypeError("reference tensor for causal mask must be floating point")
    if length == 0:
        return torch.empty((0, 0), dtype=reference.dtype, device=reference.device)
    # -1e4 is safely representable in float16/bfloat16/float32 and drives
    # softmax probability of masked positions effectively to zero.
    return torch.triu(
        torch.full((length, length), -1.0e4, dtype=reference.dtype, device=reference.device),
        diagonal=1,
    )


@contextmanager
def directml_transformer_safe_path(reference: torch.Tensor):
    """Disable PyTorch MHA/Transformer fused fast paths only for DirectML.

    The setting is restored exactly after the call, including on exceptions.
    This is process-global inside PyTorch, so the context is intentionally kept
    narrow around individual Transformer calls.
    """
    if not is_directml_tensor(reference):
        yield
        return

    mha_backend = getattr(torch.backends, "mha", None)
    get_enabled = getattr(mha_backend, "get_fastpath_enabled", None) if mha_backend else None
    set_enabled = getattr(mha_backend, "set_fastpath_enabled", None) if mha_backend else None
    if get_enabled is None or set_enabled is None:
        yield
        return

    previous = bool(get_enabled())
    try:
        set_enabled(False)
        yield
    finally:
        set_enabled(previous)
