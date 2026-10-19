#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
from _codecs import encode as codecs_encode

import numpy as np
import torch

try:
    from torch._utils import _rebuild_device_tensor_from_numpy
except ImportError as exc:
    raise RuntimeError(
        "This PyTorch build does not expose "
        "torch._utils._rebuild_device_tensor_from_numpy"
    ) from exc

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:
    from numpy.core.multiarray import _reconstruct


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_cpu_tree(obj, path="root"):
    if torch.is_tensor(obj):
        if obj.device.type != "cpu":
            raise AssertionError(f"non-CPU tensor at {path}: {obj.device}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_cpu_tree(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_cpu_tree(v, f"{path}[{i}]")


def enable_known_directml_numpy_globals():
    add_fn = getattr(torch.serialization, "add_safe_globals", None)
    if add_fn is None:
        raise RuntimeError(
            "This PyTorch build does not expose "
            "torch.serialization.add_safe_globals"
        )

    allowed = [
        _rebuild_device_tensor_from_numpy,
        _reconstruct,
        np.ndarray,
        np.dtype,
        codecs_encode,
    ]

    dtype_samples = [
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.bool_),
    ]

    for dt in dtype_samples:
        cls = type(dt)
        if cls not in allowed:
            allowed.append(cls)

    add_fn(allowed)

    print("restricted_allowlist:")
    for item in allowed:
        print(
            " ",
            getattr(item, "__module__", "?"),
            getattr(item, "__qualname__", repr(item)),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    args = ap.parse_args()

    src = Path(args.source)
    dst = Path(args.destination)

    if not src.is_file():
        raise FileNotFoundError(src)

    if dst.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {dst}"
        )

    enable_known_directml_numpy_globals()

    payload = torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload is not a dict")

    required = {
        "base_checkpoint_sha256",
        "condition",
        "competition",
        "recurrent_residual",
        "motor_residual",
        "world_residual",
    }

    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"checkpoint missing required keys: {missing}")

    portable = cpu_tree(payload)
    portable["portable_cpu_checkpoint"] = True
    portable["migrated_from_sha256"] = sha256_file(src)

    assert_cpu_tree(portable)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, dst)

    print("source:", src)
    print("destination:", dst)
    print("condition:", portable["condition"])
    print("base_checkpoint_sha256:", portable["base_checkpoint_sha256"])
    print("source_sha256:", portable["migrated_from_sha256"])
    print("portable_sha256:", sha256_file(dst))
    print("portable_cpu_checkpoint: true")
    print("MIGRATION_WRITE_OK")


if __name__ == "__main__":
    main()
