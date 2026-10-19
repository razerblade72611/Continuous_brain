from dataclasses import replace
from pathlib import Path

import torch

from continuous_brain import __version__
from continuous_brain.growth_v110 import ProtectedZeroResidualStack, is_growth_parameter
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_train import save_checkpoint, load_checkpoint
from continuous_brain.whole_brain_v110 import expand_capacity
from continuous_brain.protected_growth_v114 import (
    STAGE_PLAN, gate_report, protect_growth, set_protected_gates,
)


def test_version_is_v114():
    assert __version__ in ("0.11.4", "0.12.0", "0.12.1")


def test_protected_stack_explicit_shadow_gradient_and_closed_gate():
    torch.manual_seed(114)
    block = ProtectedZeroResidualStack(8, 16, 2, gate=0.0)
    # Make the branch non-zero so the gate is genuinely tested.
    with torch.no_grad():
        for b in block.blocks:
            b.out_proj.weight.normal_(0, 0.05)
            b.out_proj.bias.normal_(0, 0.05)
    x = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    y = block(x)
    assert torch.equal(y.detach(), x.detach())
    block.shadow_training = True
    y = block(x)
    assert not torch.equal(y.detach(), x.detach())
    loss = (y - target).square().mean()
    loss.backward()
    grads = [b.out_proj.weight.grad for b in block.blocks]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_protected_stack_no_grad_respects_visibility_gate():
    torch.manual_seed(115)
    block = ProtectedZeroResidualStack(6, 12, 1, gate=0.0)
    with torch.no_grad():
        block.blocks[0].out_proj.weight.normal_(0, 0.1)
        x = torch.randn(3, 6)
        closed = block(x)
        assert torch.equal(closed, x)
        block.set_gate(1.0)
        opened = block(x)
        assert not torch.equal(opened, x)


def test_protect_growth_migration_is_exact_and_adds_no_learned_parameters(tmp_path):
    src_base = tmp_path / "base"
    src_base.mkdir()
    base = InfantBrain(InfantConfig(), vision_parent_dims={"vision.test": 8})
    save_checkpoint(src_base / "infant_training.pt", base, updates=14000,
                    parent_id="vision.test", dataset_digest="x",
                    metadata={"planning_updates": 4000, "option_updates": 3000,
                              "option_cortex_version": "latent-temporal-options-v1"})
    expanded_dir = tmp_path / "expanded"
    expand_capacity(src_base, expanded_dir, device="cpu")
    expanded, _ = load_checkpoint(expanded_dir / "infant_training.pt", "cpu")
    before_params = sum(p.numel() for p in expanded.parameters())
    before_state = {k: v.clone() for k, v in expanded.state_dict().items()}

    result = protect_growth(expanded_dir, tmp_path / "protected", device="cpu")
    assert result["exact_function_preservation"] is True
    assert result["max_abs_difference"] == 0.0
    assert result["source_tensor_mismatches"] == 0
    assert result["learned_parameters_added"] == 0

    protected, saved = load_checkpoint(tmp_path / "protected" / "infant_training.pt", "cpu")
    after_params = sum(p.numel() for p in protected.parameters())
    assert before_params == after_params
    assert protected.cfg.protected_growth_enabled is True
    assert saved["metadata"]["protected_growth_version"] == "protected-shadow-gates-v1"
    current = protected.state_dict()
    for k, v in before_state.items():
        assert torch.equal(v, current[k])
    assert all(k.endswith(".gate") for k in saved["metadata"]["protected_growth_new_gate_keys"])


def test_default_protected_gate_roles_and_stage_plan(tmp_path):
    src = tmp_path / "base"; src.mkdir()
    model = InfantBrain(InfantConfig(), vision_parent_dims={"vision.test": 8})
    save_checkpoint(src/"infant_training.pt", model, updates=1,
                    parent_id="vision.test", dataset_digest="x")
    exp = tmp_path / "exp"
    expand_capacity(src, exp, device="cpu")
    dst = tmp_path / "protected"
    protect_growth(exp, dst, device="cpu")
    protected, _ = load_checkpoint(dst/"infant_training.pt", "cpu")
    gates = gate_report(protected)
    assert gates["planning.growth"] == 0.0
    assert gates["options.growth"] == 0.0
    assert gates["target_vision.growth"] == 0.0
    assert gates["target_spatial.growth"] == 0.0
    assert gates["vision.growth"] == 1.0
    assert STAGE_PLAN["shadow"]["planner_gate"] == 0.0
    assert STAGE_PLAN["micro"]["planner_gate"] == 0.02
    assert STAGE_PLAN["low"]["planner_gate"] == 0.05
    assert STAGE_PLAN["moderate"]["planner_gate"] == 0.10


def test_gate_changes_do_not_change_parameter_count():
    cfg = InfantConfig(growth_enabled=True, growth_profile="balanced-v110", protected_growth_enabled=True)
    model = InfantBrain(cfg)
    count = sum(p.numel() for p in model.parameters())
    set_protected_gates(model, general_gate=1.0, planner_gate=0.02, option_gate=0.02, target_gate=0.0)
    assert sum(p.numel() for p in model.parameters()) == count
    assert not any(n.endswith(".gate") for n, _ in model.named_parameters())
