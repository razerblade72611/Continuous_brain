"""Peak-memory accounting that works on Linux/WSL, Windows, and CUDA."""

from __future__ import annotations

import sys

import torch


def peak_rss_bytes(device: torch.device | str | None = None) -> int | None:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(dev))
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)   # Linux reports KiB
    except (ImportError, OSError):
        pass
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except ImportError:
        return None
