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

    # These checkpoints were created locally by our own trainer.  We still
    # keep weights_only=True and allowlist only the one DirectML tensor
    # reconstruction global identified by PyTorch's error message.
    with torch.serialization.safe_globals(
        [_rebuild_device_tensor_from_numpy]
    ):
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

    # Critical verification: the migrated checkpoint must load with ordinary
    # restricted weights_only=True and no allowlist at all.
    verified = torch.load(
        dst,
        map_location="cpu",
        weights_only=True,
    )
    assert_cpu_tree(verified)

    if verified["base_checkpoint_sha256"] != payload["base_checkpoint_sha256"]:
        raise AssertionError("base checkpoint SHA changed during migration")
    if verified["condition"] != payload["condition"]:
        raise AssertionError("condition changed during migration")

    print("source:", src)
    print("destination:", dst)
    print("condition:", verified["condition"])
    print("base_checkpoint_sha256:", verified["base_checkpoint_sha256"])
    print("source_sha256:", portable["migrated_from_sha256"])
    print("portable_sha256:", sha256_file(dst))
    print("portable_cpu_checkpoint:", verified["portable_cpu_checkpoint"])
    print("weights_only_reload: true")
    print("CHECKPOINT_MIGRATION_OK")


if __name__ == "__main__":
    main()
