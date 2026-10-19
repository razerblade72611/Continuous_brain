from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from continuous_brain import __version__
from continuous_brain.dml_ops_v121 import safe_one_hot
from continuous_brain.dml_recurrent_v112 import unfused_gru_cell
from continuous_brain.infant import CorticalNucleus, InfantConfig, one_hot_action


def test_version_v121():
    assert __version__ in {"0.12.1", "0.12.2"}


def test_safe_one_hot_matches_historical_cpu_path():
    x = torch.tensor([0, 4, 2, 1, 3])
    expected = F.one_hot(x, num_classes=5).float()
    assert torch.equal(safe_one_hot(x, 5), expected)
    assert torch.equal(one_hot_action(x, 5), expected)


def test_unfused_sliced_gru_matches_native_forward_and_gradients():
    torch.manual_seed(121)
    native = nn.GRUCell(7, 5).double()
    x1 = torch.randn(3, 7, dtype=torch.double, requires_grad=True)
    h1 = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    h2 = h1.detach().clone().requires_grad_(True)

    y_native = native(x1, h1)
    y_safe = unfused_gru_cell(
        x2, h2, native.weight_ih, native.weight_hh, native.bias_ih, native.bias_hh
    )
    assert torch.allclose(y_safe, y_native, atol=1e-12, rtol=1e-10)

    g = torch.randn_like(y_native)
    grads_native = torch.autograd.grad(
        y_native, (x1, h1, native.weight_ih, native.weight_hh, native.bias_ih, native.bias_hh),
        g, retain_graph=True,
    )
    grads_safe = torch.autograd.grad(
        y_safe, (x2, h2, native.weight_ih, native.weight_hh, native.bias_ih, native.bias_hh),
        g,
    )
    for a, b in zip(grads_safe, grads_native):
        assert torch.allclose(a, b, atol=1e-11, rtol=1e-9)


def test_unfused_sliced_gru_multistep_temporal_gradient_matches_native():
    torch.manual_seed(122)
    native = nn.GRUCell(6, 4).double()
    xs_native = [torch.randn(2, 6, dtype=torch.double, requires_grad=True) for _ in range(5)]
    xs_safe = [x.detach().clone().requires_grad_(True) for x in xs_native]
    h_native = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    h_safe = h_native.detach().clone().requires_grad_(True)
    h0_native, h0_safe = h_native, h_safe
    for xn, xs in zip(xs_native, xs_safe):
        h_native = native(xn, h_native)
        h_safe = unfused_gru_cell(xs, h_safe, native.weight_ih, native.weight_hh, native.bias_ih, native.bias_hh)
    assert torch.allclose(h_safe, h_native, atol=1e-12, rtol=1e-10)
    gn = torch.autograd.grad(h_native.square().mean(), h0_native)[0]
    gs = torch.autograd.grad(h_safe.square().mean(), h0_safe)[0]
    assert torch.allclose(gs, gn, atol=1e-11, rtol=1e-9)


def test_nucleus_stream_refactor_is_exact():
    torch.manual_seed(123)
    cfg = InfantConfig(growth_enabled=False)
    nucleus = CorticalNucleus(cfg).eval()
    vision = torch.randn(2, cfg.bus_dim)
    context = torch.randn(2, cfg.context_dim)
    memory = torch.randn(2, cfg.memory_dim)
    action = torch.randint(0, cfg.action_dim + 1, (2,))
    hidden = torch.randn(2, cfg.nucleus_dim)
    streams = nucleus.project_streams(vision, context, memory, action)
    a = nucleus.forward_from_streams(streams, hidden)
    b = nucleus(vision, context, memory, action, hidden)
    for key in ("hidden", "workspace", "router", "metacognition"):
        assert torch.equal(a[key], b[key])
