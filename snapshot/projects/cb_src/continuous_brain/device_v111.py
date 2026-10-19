from __future__ import annotations

"""Device resolution for Continuous Brain v0.11.1.

DirectML is an optional backend.  It is imported only when explicitly requested
(or when ``auto`` is asked to consider it), so CPU/CUDA installations do not
acquire a hard torch-directml dependency.
"""

import importlib
from typing import Any

import torch


_DML_ALIASES = {"dml", "directml", "privateuseone", "privateuse1"}


def _parse_dml_index(text: str) -> int:
    value = text.strip().lower()
    if ":" not in value:
        return 0
    _, raw = value.rsplit(":", 1)
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid DirectML device index in {text!r}") from exc
    if index < 0:
        raise ValueError("DirectML device index must be >= 0")
    return index


def is_directml_spec(spec: Any) -> bool:
    if isinstance(spec, torch.device):
        return spec.type in {"privateuseone", "privateuse1"}
    text = str(spec).strip().lower()
    base = text.split(":", 1)[0]
    return base in _DML_ALIASES


def _import_directml():
    try:
        return importlib.import_module("torch_directml")
    except Exception as exc:
        raise RuntimeError(
            "DirectML was requested but torch-directml could not be imported. "
            "Install torch-directml in a compatible, separate Python environment."
        ) from exc


def resolve_device(spec: Any = "cpu", *, auto_directml: bool = False) -> torch.device:
    """Resolve CPU, CUDA, or DirectML device syntax to a torch device.

    Accepted DirectML forms: ``dml``, ``dml:0``, ``directml``,
    ``privateuseone`` and ``privateuseone:0``.  ``auto`` prefers CUDA; it only
    tries DirectML when ``auto_directml=True`` so historical behavior stays
    deterministic.
    """
    if isinstance(spec, torch.device):
        if spec.type in {"privateuseone", "privateuse1"}:
            # Importing torch_directml registers the PrivateUse1 backend.
            _import_directml()
        return spec

    text = str(spec).strip().lower()
    if not text:
        raise ValueError("device may not be empty")
    if text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if auto_directml:
            try:
                dml = _import_directml()
                if bool(dml.is_available()) and int(dml.device_count()) > 0:
                    return dml.device(0)
            except Exception:
                pass
        return torch.device("cpu")

    if is_directml_spec(text):
        dml = _import_directml()
        if not bool(dml.is_available()):
            raise RuntimeError("torch-directml imported, but DirectML is not available")
        count = int(dml.device_count())
        index = _parse_dml_index(text)
        if count <= index:
            raise RuntimeError(f"DirectML device index {index} requested, but only {count} device(s) are available")
        return dml.device(index)

    return torch.device(text)


def checkpoint_map_location(device: Any) -> Any:
    """Use CPU deserialization for DirectML/custom backends, then move the model."""
    resolved = resolve_device(device) if not isinstance(device, torch.device) else device
    if resolved.type in {"cpu", "cuda"}:
        return resolved
    return torch.device("cpu")


def device_report(spec: Any = "cpu", *, compute_test: bool = False, matrix_size: int = 256) -> dict:
    requested = str(spec)
    resolved = resolve_device(spec)
    report = {
        "format": "continuous_brain.device_report.v111",
        "requested": requested,
        "resolved": str(resolved),
        "backend": resolved.type,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "directml": resolved.type in {"privateuseone", "privateuse1"},
    }
    if report["directml"]:
        dml = _import_directml()
        idx = resolved.index or 0
        report["directml_device_count"] = int(dml.device_count())
        try:
            report["device_name"] = str(dml.device_name(idx))
        except Exception as exc:
            report["device_name_error"] = f"{type(exc).__name__}: {exc}"
    elif resolved.type == "cuda":
        idx = resolved.index or 0
        report["device_name"] = str(torch.cuda.get_device_name(idx))

    if compute_test:
        n = max(8, int(matrix_size))
        a = torch.randn(n, n, device=resolved)
        b = torch.randn(n, n, device=resolved)
        c = a @ b
        report["compute_test"] = {
            "passed": True,
            "matrix_size": n,
            "result_device": str(c.device),
            "mean": float(c.mean().detach().cpu().item()),
        }
    return report
