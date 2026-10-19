#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import torch

try:
    from torch._utils import _rebuild_device_tensor_from_numpy
except ImportError as exc:
    raise RuntimeError(
        "This PyTorch build does not expose "
        "torch._utils._rebuild_device_tensor_from_numpy"
    ) from exc


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


def enable_directml_safe_global():
    """
    PyTorch 2.4.x exposes add_safe_globals(), while newer releases also
    provide the safe_globals() context manager.  This process is dedicated
    to one migration, so a process-local allowlist is sufficient.
    """
    add_fn = getattr(torch.serialization, "add_safe_globals", None)
    if add_fn is None:
        raise RuntimeError(
            "This PyTorch build does not expose "
            "torch.serialization.add_safe_globals"
        )
    add_fn([_rebuild_device_tensor_from_numpy])


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

    enable_directml_safe_global()

    # These checkpoints were created locally by our own trainer.
    # Keep restricted weights_only=True and allowlist only the exact
    # DirectML tensor reconstruction global identified by PyTorch.
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

    # Important: launch a fresh Python process for strict verification so
    # the process-local safe-global allowlist above cannot hide a problem.
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
