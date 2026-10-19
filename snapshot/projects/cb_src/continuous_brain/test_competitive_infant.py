"""Executable preservation and gradient checks for competitive broadcast growth.

Run:
    python -m continuous_brain.test_competitive_infant
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .competitive_infant_brain import CompetitiveInfantBrain, IndependentSelectionTriple
from .infant import InfantBrain, InfantConfig


def _inputs(cfg, batch=2):
    torch.manual_seed(123)
    return {
        "images": torch.randn(batch, 3, cfg.image_size, cfg.image_size),
        "context": torch.randn(batch, cfg.context_dim),
        "memory": torch.randn(batch, cfg.memory_dim),
        "prev_action": torch.randint(0, cfg.action_dim + 1, (batch,)),
        "hidden": torch.randn(batch, cfg.nucleus_dim),
    }


def _plain_step(base: InfantBrain, x: dict):
    semantic = base.vision(x["images"])
    spatial = base.spatial(x["images"])
    fused = base.fuse_vision(semantic, spatial)
    nuc = base.nucleus(fused, x["context"], x["memory"], x["prev_action"], x["hidden"])
    return {
        "hidden": nuc["hidden"],
        "workspace": nuc["workspace"],
        "logits": base.motor(nuc["workspace"]),
        "value": base.values(nuc["workspace"]),
    }


def _assert_exact(a, b, name):
    if not torch.equal(a, b):
        diff = float((a - b).abs().max().item())
        raise AssertionError(f"{name} differs; max abs diff={diff}")


def test_immediate_preservation():
    cfg = InfantConfig()
    base = InfantBrain(cfg).eval()
    grown = CompetitiveInfantBrain.grow_from(base, use_shared=True).eval()
    x = _inputs(cfg)
    with torch.no_grad():
        plain = _plain_step(base, x)
        enabled = grown.step(**x, use_competition=True)
        disabled = grown.step(**x, use_competition=False)
    for name in ("hidden", "logits", "value"):
        _assert_exact(plain[name], enabled[name], f"enabled/{name}")
        _assert_exact(plain[name], disabled[name], f"disabled/{name}")
    print("PASS immediate growth is an exact no-op in both enabled and disabled modes")


def test_disabled_after_trainable_perturbation():
    cfg = InfantConfig()
    base = InfantBrain(cfg).eval()
    grown = CompetitiveInfantBrain.grow_from(base, use_shared=True).eval()
    x = _inputs(cfg)
    with torch.no_grad():
        for p in grown.trainable_parameters():
            p.add_(0.05 * torch.randn_like(p))
        plain = _plain_step(base, x)
        disabled = grown.step(**x, use_competition=False)
    for name in ("hidden", "logits", "value"):
        _assert_exact(plain[name], disabled[name], f"disabled-after-training/{name}")
    print("PASS disabling the learned branch still recovers the exact frozen base")


def test_save_restore_continuation():
    cfg = InfantConfig()
    base = InfantBrain(cfg).eval()
    grown = CompetitiveInfantBrain.grow_from(base, use_shared=True).eval()
    x = _inputs(cfg)
    with torch.no_grad():
        for p in grown.trainable_parameters():
            p.add_(0.02 * torch.randn_like(p))
        before = grown.step(**x, use_competition=True)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "grown.pt"
        torch.save(grown.state_dict(), path)
        restored = CompetitiveInfantBrain(cfg, use_shared=True).eval()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            after = restored.step(**x, use_competition=True)
    for name in ("hidden", "logits", "value", "workspace_for_world"):
        _assert_exact(before[name], after[name], f"save-restore/{name}")
    print("PASS save/restore preserves learned competitive behavior exactly")


def test_frozen_base_and_parameter_accounting():
    cfg = InfantConfig()
    model = CompetitiveInfantBrain(cfg, use_shared=True)
    assert all(not p.requires_grad for p in model.base.parameters())
    model.train(True)
    assert not model.base.training
    assert all(not m.training for m in model.base.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 98_944, trainable
    independent = CompetitiveInfantBrain(cfg, use_shared=False)
    independent_trainable = sum(p.numel() for p in independent.parameters() if p.requires_grad)
    assert independent_trainable == 164_736, independent_trainable
    print(f"PASS base frozen/eval; trainable params shared={trainable:,} independent={independent_trainable:,}")


def test_all_independent_selectors_receive_gradient():
    cfg = InfantConfig()
    model = CompetitiveInfantBrain(cfg, use_shared=False).train(True)
    # At exact zero-init, selector-query gradients are expected to be zero on
    # the first step because every consumer residual is zero.  Move only the
    # residuals off zero, then verify the actual recurrent/motor/world workspace
    # paths make all three selector queries differentiable.
    torch.manual_seed(321)
    with torch.no_grad():
        for residual in (model.recurrent_residual, model.motor_residual, model.world_residual):
            residual.linear.weight.normal_(0.0, 0.01)
            residual.linear.bias.normal_(0.0, 0.01)
    x = _inputs(cfg, batch=1)
    out = model.step(**x, use_competition=True)
    loss = out["hidden"].square().mean() + out["logits"].square().mean() + out["workspace_for_world"].square().mean()
    loss.backward()
    grads = {
        "recurrent": model.competition.recurrent_gate.query.weight.grad,
        "motor": model.competition.motor_gate.query.weight.grad,
        "world": model.competition.world_gate.query.weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None, f"{name} selector has no gradient"
        assert float(grad.abs().sum().item()) > 0.0, f"{name} selector gradient is all zero"
    print("PASS recurrent, motor, and world independent selectors all receive nonzero gradient")


if __name__ == "__main__":
    test_immediate_preservation()
    test_disabled_after_trainable_perturbation()
    test_save_restore_continuation()
    test_frozen_base_and_parameter_accounting()
    test_all_independent_selectors_receive_gradient()
    print("\nAll competitive-broadcast checks passed under real PyTorch.")
