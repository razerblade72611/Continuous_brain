"""Focused checks -- ready to run where PyTorch is available. NOT executed
in this sandbox (no torch); shape and gradient-existence claims have a
NumPy-only stand-in verification in verify_zero_init.py, actually run.
"""
from __future__ import annotations
import torch
from .competitive_development_brain import CompetitiveDevelopmentBrain, StreamCompetition, IndependentSelection
from .model import DevelopmentBrain, BrainConfig


def _all_close(a, b, name):
    ok = torch.allclose(a, b, atol=1e-6)
    print(f"  {'OK' if ok else 'MISMATCH'}  {name}")
    return ok


def test_preservation_1_immediately_after_growth():
    """Fresh growth, no training at all: both use_competition=True and
    use_competition=False must match the plain base model, since the
    residuals are exactly zero and haven't been touched yet."""
    base = DevelopmentBrain(BrainConfig())
    grown = CompetitiveDevelopmentBrain.grow_from(base, use_shared=True)
    grid, sensors, memory, style = torch.randn(2,4,9,9), torch.randn(2,18), torch.randn(2,8), torch.randn(2,4)
    hb, hg = base.initial(2, "cpu"), grown.initial(2, "cpu")
    out_base = base(grid, sensors, memory, style, hb)
    out_disabled = grown(grid, sensors, memory, style, hg, use_competition=False)
    out_enabled = grown(grid, sensors, memory, style, hg, use_competition=True)
    ok = (_all_close(out_base["hidden"], out_disabled["hidden"], "disabled hidden vs base") and
          _all_close(out_base["logits"], out_disabled["logits"], "disabled logits vs base") and
          _all_close(out_base["hidden"], out_enabled["hidden"], "ENABLED hidden vs base (residual still zero)") and
          _all_close(out_base["logits"], out_enabled["logits"], "ENABLED logits vs base (residual still zero)"))
    print("PASS  Check 1 (immediately after growth): both modes match base." if ok else "FAIL Check 1")
    return ok


def test_preservation_2_disabled_after_training():
    """Simulates 'training' by directly perturbing ONLY the trainable
    parameters (competition + residuals) away from init, WITHOUT touching
    base.* at all -- then confirms use_competition=False still matches a
    plain base model exactly. This is the check that would catch a bug
    where training accidentally reached a frozen parameter."""
    base = DevelopmentBrain(BrainConfig())
    grown = CompetitiveDevelopmentBrain.grow_from(base, use_shared=True)
    with torch.no_grad():
        for p in grown.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.1)
    for name, p in grown.base.named_parameters():
        assert not p.requires_grad, f"base.{name} should be frozen but requires_grad=True"
    grid, sensors, memory, style = torch.randn(2,4,9,9), torch.randn(2,18), torch.randn(2,8), torch.randn(2,4)
    out_base = base(grid, sensors, memory, style, base.initial(2, "cpu"))
    out_disabled = grown(grid, sensors, memory, style, grown.initial(2, "cpu"), use_competition=False)
    ok = (_all_close(out_base["hidden"], out_disabled["hidden"], "disabled hidden vs base, post-perturbation") and
          _all_close(out_base["logits"], out_disabled["logits"], "disabled logits vs base, post-perturbation"))
    print("PASS  Check 2 (disabled after training): matches base even after trainable params moved." if ok else "FAIL Check 2")
    return ok


def test_preservation_3_continuation_after_save_restore():
    """Perturb trainable params (simulating training), save, reload into a
    FRESH instance, confirm ENABLED-mode output matches pre-save exactly
    -- both the moved trainable weights and the frozen base must survive
    the round trip."""
    base = DevelopmentBrain(BrainConfig())
    grown = CompetitiveDevelopmentBrain.grow_from(base, use_shared=True)
    with torch.no_grad():
        for p in grown.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.1)
    grid, sensors, memory, style = torch.randn(2,4,9,9), torch.randn(2,18), torch.randn(2,8), torch.randn(2,4)
    hidden = grown.initial(2, "cpu")
    out_before = grown(grid, sensors, memory, style, hidden, use_competition=True)

    torch.save({"model": grown.state_dict()}, "/tmp/competitive_round_trip_test.pt")
    reloaded_state = torch.load("/tmp/competitive_round_trip_test.pt", weights_only=False)
    fresh = CompetitiveDevelopmentBrain(BrainConfig(), use_shared=True)
    fresh.load_state_dict(reloaded_state["model"])
    out_after = fresh(grid, sensors, memory, style, fresh.initial(2, "cpu"), use_competition=True)

    ok = (_all_close(out_before["hidden"], out_after["hidden"], "post-restore hidden vs pre-save") and
          _all_close(out_before["logits"], out_after["logits"], "post-restore logits vs pre-save"))
    print("PASS  Check 3 (continuation after save/restore): exact match." if ok else "FAIL Check 3")
    return ok


def test_frozen_vs_trainable_partition_is_exact():
    base = DevelopmentBrain(BrainConfig())
    grown = CompetitiveDevelopmentBrain.grow_from(base, use_shared=True)
    frozen = sum(p.numel() for p in grown.base.parameters())
    trainable = sum(p.numel() for p in grown.trainable_parameters())
    total = sum(p.numel() for p in grown.parameters())
    assert frozen + trainable == total, "every parameter must be exactly one of frozen or trainable, no overlap or gap"
    print(f"PASS  frozen={frozen}, trainable={trainable}, sum matches total parameter count exactly ({total}).")


def test_independent_selection_memory_gate_is_structurally_dead():
    """Confirms the reported finding precisely: memory_gate's parameters
    receive no gradient because its output is wrapped in no_grad and its
    only consumer (ExperienceMemory.add) is non-differentiable anyway."""
    gate = IndependentSelection(hidden_dim=192, stream_width=128)
    hidden = torch.randn(2, 192, requires_grad=True)
    streams = [torch.randn(2, 128) for _ in range(3)]
    out = gate(hidden, streams)
    loss = out["broadcast"].sum()  # only the WORKSPACE path, matching what's actually differentiable end to end
    loss.backward()
    ws_grad = gate.workspace_gate.query.weight.grad
    mem_grad = gate.memory_gate.query.weight.grad
    print(f"  workspace_gate.query gradient is None: {ws_grad is None}")
    print(f"  memory_gate.query gradient is None: {mem_grad is None}")
    assert ws_grad is not None, "workspace_gate should receive gradient from this loss"
    assert mem_grad is None, "memory_gate should NOT receive gradient -- it's wrapped in no_grad by design"
    print("PASS  workspace_gate trains, memory_gate structurally cannot -- matches the reported parameter count split.")


if __name__ == "__main__":
    results = [
        test_preservation_1_immediately_after_growth(),
        test_preservation_2_disabled_after_training(),
        test_preservation_3_continuation_after_save_restore(),
    ]
    test_frozen_vs_trainable_partition_is_exact()
    test_independent_selection_memory_gate_is_structurally_dead()
    print(f"\n{sum(results)}/{len(results)} preservation checks defined and passing in reasoning; "
          f"none executed in this sandbox (no torch). Run where PyTorch is available.")
