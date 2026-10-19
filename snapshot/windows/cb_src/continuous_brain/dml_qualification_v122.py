from __future__ import annotations

"""DirectML qualification gates for the v0.12.2 DirectML hardening release.

Every mode is intentionally runnable in a fresh Python process.  A DirectML
DXGI device-removal error poisons the current process, so the shell qualification
script invokes these modes separately and stops at the first real failure.
"""

import argparse
import json
from pathlib import Path
import tempfile
import platform
from importlib import metadata
from typing import Any

import torch
from torch.nn import functional as F

from .dml_ops_v121 import safe_one_hot
from .dml_recurrent_v112 import DMLSafeGRUCell


def resolve_device(name: str | Any):
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    if key in {"directml", "dml"}:
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DirectML requested but torch-directml is not installed") from exc
        return torch_directml.device()
    return torch.device(name)


def _max_grad(p: torch.Tensor | None):
    if p is None:
        return None
    return float(p.detach().abs().max().cpu().item())




def environment_info():
    try:
        dml_version = metadata.version("torch-directml")
    except metadata.PackageNotFoundError:
        dml_version = None
    return {
        "mode": "environment",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_directml": dml_version,
        "note": "DirectML PyTorch remains public preview; qualification is empirical on this exact runtime.",
    }


def basic(device_spec: str):
    device = resolve_device(device_spec)
    a = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
    oh = safe_one_hot(a, 5)
    expected = torch.tensor(
        [[1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 1.]],
        dtype=torch.float32,
    )
    if not torch.equal(oh.detach().cpu(), expected):
        raise AssertionError("safe_one_hot produced incorrect values")
    x = torch.randn(8, 32, device=device, requires_grad=True)
    w = torch.randn(32, 16, device=device, requires_grad=True)
    loss = (x @ w).square().mean()
    loss.backward()
    return {
        "mode": "basic",
        "device": str(device),
        "one_hot_ok": True,
        "matmul_backward_ok": True,
        "x_grad_max": _max_grad(x.grad),
        "w_grad_max": _max_grad(w.grad),
    }


def gru_chain(device_spec: str, steps: int):
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    device = resolve_device(device_spec)
    torch.manual_seed(12101 + int(steps))
    cell = DMLSafeGRUCell(512, 256).to(device)
    h0 = torch.randn(1, 256, device=device, requires_grad=True)
    h = h0
    for _ in range(int(steps)):
        x = torch.randn(1, 512, device=device)
        h = cell(x, h)
    loss = h.square().mean()
    loss.backward()
    return {
        "mode": "gru_chain",
        "device": str(device),
        "steps": int(steps),
        "loss": float(loss.detach().cpu().item()),
        "h0_grad_max": _max_grad(h0.grad),
        "weight_ih_grad_max": _max_grad(cell.weight_ih.grad),
        "weight_hh_grad_max": _max_grad(cell.weight_hh.grad),
    }


def infant_core(device_spec: str, checkpoint: str):
    from .evaluate_competitive_infant import _load_base_checkpoint

    device = resolve_device(device_spec)
    base, cfg, _, sha = _load_base_checkpoint(checkpoint)
    for p in base.parameters():
        p.requires_grad_(False)
    base = base.to(device).eval()
    x = torch.randn(1, int(cfg.bus_dim) * 4, device=device)
    h0 = torch.randn(1, int(cfg.nucleus_dim), device=device, requires_grad=True)
    h1 = base.nucleus.core(x, h0)
    loss = h1.square().mean()
    loss.backward()
    return {
        "mode": "infant_core",
        "device": str(device),
        "checkpoint_sha256": sha,
        "loss": float(loss.detach().cpu().item()),
        "hidden_grad_max": _max_grad(h0.grad),
    }


def competitive_one_step(device_spec: str, checkpoint: str):
    from .competitive_infant_brain import CompetitiveInfantBrain
    from .cued_detour_world import CuedDetourWorld
    from .evaluate_competitive_infant import (
        _context_tensor, _image_tensor, _load_base_checkpoint, _teacher_action,
    )

    device = resolve_device(device_spec)
    base, cfg, _, sha = _load_base_checkpoint(checkpoint)
    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).to(device).train(True)
    env = CuedDetourWorld(seed=81501, corridor_length=0.3, max_episode_steps=120)
    obs = env.reset()
    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    image = _image_tensor(env, cfg.image_size, device)
    context, _ = _context_tensor(obs, device, cfg)
    memory = torch.zeros(1, cfg.memory_dim, device=device)
    out = model.step(image, context, memory, prev_action, hidden, use_competition=True, compute_aux=False)
    target = torch.tensor([_teacher_action(env)], dtype=torch.long, device=device)
    loss = F.cross_entropy(out["logits"], target)
    loss.backward()
    return {
        "mode": "competitive_one_step",
        "device": str(device),
        "checkpoint_sha256": sha,
        "loss": float(loss.detach().cpu().item()),
        "recurrent_residual_grad_max": _max_grad(model.recurrent_residual.linear.weight.grad),
        "motor_residual_grad_max": _max_grad(model.motor_residual.linear.weight.grad),
        "selector_query_grad_max": _max_grad(model.competition.query.weight.grad),
        "note": "selector query may be exactly zero at zero-initialized consumer residuals",
    }


def competitive_episode(device_spec: str, checkpoint: str):
    from .evaluate_competitive_infant import train

    with tempfile.TemporaryDirectory(prefix="cb_v121_dml_gate_") as td:
        result = train(
            base_checkpoint=checkpoint,
            use_shared=True,
            updates=1,
            seed=81501,
            device=device_spec,
            state_dir=td,
            corridor_lengths=(0.3,),
            log_every=1,
        )
    return {"mode": "competitive_episode", "device": str(resolve_device(device_spec)), **result}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("environment", "basic", "gru", "infant-core", "competitive-one-step", "competitive-episode"))
    p.add_argument("--device", default="directml")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--checkpoint", default="state/child_v113_dml_smoke/infant_training.pt")
    args = p.parse_args(argv)
    if args.mode == "environment":
        result = environment_info()
    elif args.mode == "basic":
        result = basic(args.device)
    elif args.mode == "gru":
        result = gru_chain(args.device, args.steps)
    elif args.mode == "infant-core":
        result = infant_core(args.device, args.checkpoint)
    elif args.mode == "competitive-one-step":
        result = competitive_one_step(args.device, args.checkpoint)
    else:
        result = competitive_episode(args.device, args.checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
