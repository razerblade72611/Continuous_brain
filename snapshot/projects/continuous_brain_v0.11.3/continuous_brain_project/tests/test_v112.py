import torch
from torch import nn

from continuous_brain import __version__
from continuous_brain.dml_recurrent_v112 import DMLSafeGRUCell, unfused_gru_cell
from continuous_brain.infant import InfantBrain, InfantConfig


def test_version_is_v112_compatible():
    assert __version__.startswith("0.11.")


def test_unfused_gru_matches_native_cpu_forward_and_backward():
    torch.manual_seed(112)
    native = nn.GRUCell(7, 11)
    safe = DMLSafeGRUCell(7, 11)
    safe.load_state_dict(native.state_dict())

    x1 = torch.randn(5, 7, requires_grad=True)
    h1 = torch.randn(5, 11, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    h2 = h1.detach().clone().requires_grad_(True)

    ref = native(x1, h1)
    alt = unfused_gru_cell(x2, h2, safe.weight_ih, safe.weight_hh, safe.bias_ih, safe.bias_hh)
    assert torch.allclose(ref, alt, atol=1e-7, rtol=1e-6)

    ref.square().mean().backward()
    alt.square().mean().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=2e-7, rtol=2e-6)
    assert torch.allclose(h1.grad, h2.grad, atol=2e-7, rtol=2e-6)


def test_safe_gru_state_dict_schema_is_native_compatible():
    native = nn.GRUCell(3, 5)
    safe = DMLSafeGRUCell(3, 5)
    assert tuple(native.state_dict()) == tuple(safe.state_dict())
    safe.load_state_dict(native.state_dict(), strict=True)
    native.load_state_dict(safe.state_dict(), strict=True)


def test_current_child_recurrent_sites_are_dml_safe():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    assert isinstance(model.nucleus.core, DMLSafeGRUCell)
    assert isinstance(model.dynamics_temporal.cell, DMLSafeGRUCell)
    assert isinstance(model.planning.transition, DMLSafeGRUCell)


def test_growth_checkpoint_keys_unchanged_by_gru_wrapper():
    cfg = InfantConfig.for_profile("proof")
    cfg.growth_enabled = True
    cfg.growth_profile = "balanced-v110"
    model = InfantBrain(cfg)
    keys = set(model.state_dict())
    assert "nucleus.core.weight_ih" in keys
    assert "nucleus.core.weight_hh" in keys
    assert "dynamics_temporal.cell.weight_ih" in keys
    assert "planning.transition.weight_hh" in keys
    assert not any("dml_recurrent" in k for k in keys)
