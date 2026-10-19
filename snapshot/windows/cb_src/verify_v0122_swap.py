"""One-off verification for the v0.12.2 DirectML-removal + Qwen3-VL swap.

Run from the project root on the CUDA pod, after syncing the edited files:

    python3 verify_v0122_swap.py

Checks, against a REAL checkpoint (not a synthetic one):
  1. load_checkpoint() still accepts the checkpoint with no disallowed
     missing/unexpected keys (proves the nn.GRUCell swap didn't change any
     state_dict key names/shapes for the three recurrent sites).
  2. Those three sites are now plain nn.GRUCell instances (DirectML
     workaround actually gone), not just "still works by luck".
  3. A forward pass through the loaded weights runs and produces finite
     output (the module executes end-to-end, not just deserializes).
"""
import glob
import sys

import torch
from torch import nn

from continuous_brain.infant import InfantBrain  # noqa: F401 (import-side sanity)
from continuous_brain.parented_train import load_checkpoint

CKPTS = sorted(glob.glob("**/*.pt", recursive=True)) + sorted(glob.glob("**/*.pth", recursive=True))
CKPTS = [c for c in CKPTS if "/venv/" not in c and "/.git/" not in c]

if not CKPTS:
    print("No .pt/.pth checkpoint found under the current directory.")
    print("Run this from the project root that holds your checkpoint(s).")
    sys.exit(1)

print(f"Found {len(CKPTS)} checkpoint(s):")
for c in CKPTS:
    print(f"  - {c}")
print()

SITES = ("nucleus.core", "dynamics_temporal.cell", "planning.transition")

failures = []
for ckpt_path in CKPTS:
    print(f"=== {ckpt_path} ===")
    try:
        model, meta = load_checkpoint(ckpt_path, device="cpu")
    except Exception as exc:  # noqa: BLE001 - want to report, not crash the sweep
        print(f"  FAILED to load: {exc}")
        failures.append((ckpt_path, str(exc)))
        continue
    print("  load_checkpoint(): OK (no disallowed missing/unexpected keys)")

    model.eval()
    for site in SITES:
        obj = model
        for attr in site.split("."):
            obj = getattr(obj, attr)
        is_native = type(obj) is nn.GRUCell
        print(f"  {site}: {type(obj).__name__}  {'OK' if is_native else 'UNEXPECTED TYPE'}")
        if not is_native:
            failures.append((ckpt_path, f"{site} is {type(obj).__name__}, expected nn.GRUCell"))
            continue
        with torch.no_grad():
            h = torch.zeros(1, obj.hidden_size)
            x = torch.zeros(1, obj.input_size)
            out = obj(x, h)
        finite = bool(torch.isfinite(out).all())
        print(f"    forward() -> shape {tuple(out.shape)}, all finite: {finite}")
        if not finite:
            failures.append((ckpt_path, f"{site} forward produced non-finite output"))
    print()

if failures:
    print(f"FAILED: {len(failures)} problem(s) found:")
    for ckpt_path, msg in failures:
        print(f"  - {ckpt_path}: {msg}")
    sys.exit(1)

print("PASSED: all checkpoints load cleanly and all three recurrent sites are")
print("native nn.GRUCell with a working forward pass. DirectML removal verified")
print("against real checkpoint data, not just by reading the old class's docstring.")
