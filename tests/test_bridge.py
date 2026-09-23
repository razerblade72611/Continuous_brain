import math

import pytest
import torch
import torch.nn.functional as F

from continuous_brain.bridge import InvocationGate, SoftPrefixBridge, WritebackDecoder


def test_lm_loss_matches_manual_computation(backbone):
    prompt, target = backbone.encode("Q: hi"), backbone.encode(" A: ok")
    got = backbone.lm_loss(None, [prompt], [target])[0]
    ids = torch.cat([prompt, target])[None]
    out = backbone.forward_embeds(backbone.embed(ids), torch.ones_like(ids))
    logp = F.log_softmax(out.logits[0], -1)
    s = prompt.shape[0]
    manual = -torch.stack([logp[s - 1 + q, target[q]] for q in range(target.shape[0])]).mean()
    assert float(got) == pytest.approx(float(manual), rel=1e-5)


def test_lm_loss_ignores_batch_padding(backbone):
    p1, t1 = backbone.encode("short"), backbone.encode(" x")
    p2, t2 = backbone.encode("a much longer prompt here"), backbone.encode(" y")
    alone = backbone.lm_loss(None, [p1], [t1])[0]
    batched = backbone.lm_loss(None, [p1, p2], [t1, t2])[0]
    assert float(alone) == pytest.approx(float(batched), rel=1e-5)


def test_gradients_reach_bridge_not_backbone(backbone):
    bridge = SoftPrefixBridge(16, backbone.hidden_size, 32, backbone.embedding_rms())
    z = torch.randn(2, 3, 16)
    loss = backbone.lm_loss(bridge(z), [backbone.encode("a"), backbone.encode("b")],
                            [backbone.encode(" c"), backbone.encode(" d")]).sum()
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.mlp.parameters())
    assert all(p.grad is None for p in backbone.model.parameters())


def test_prefix_matches_embedding_scale(backbone):
    bridge = SoftPrefixBridge(16, backbone.hidden_size, 32, backbone.embedding_rms())
    p = bridge(torch.randn(1, 4, 16))
    rms = torch.sqrt((p * p).mean(-1))
    assert torch.allclose(rms, torch.full_like(rms, backbone.embedding_rms()), rtol=1e-3)


def test_generate_returns_hidden_per_token(backbone):
    g = backbone.generate(None, "hello", 5)
    assert g.hidden.shape[0] == g.token_ids.shape[0] or g.hidden.shape[0] == g.token_ids.shape[0] + 1
    assert g.hidden.shape[-1] == backbone.hidden_size


def test_writeback_starts_silent_and_log_prob_is_gaussian():
    wb = WritebackDecoder(hidden_size=8, n_input=4, hidden=16, max_current=2.0, noise_std=0.5)
    s = torch.randn(3, 8)
    assert torch.allclose(wb.mean(s), torch.zeros(3, 4))
    cur = torch.randn(3, 4)
    expected = -0.5 * ((cur / 0.5) ** 2 + math.log(2 * math.pi * 0.25)).sum(-1)
    assert torch.allclose(wb.log_prob(s, cur), expected, atol=1e-5)


def test_writeback_current_is_bounded():
    wb = WritebackDecoder(8, 4, 16, max_current=1.5, noise_std=0.1)
    torch.nn.init.normal_(wb.net[-1].weight, std=100.0)
    with torch.no_grad():
        assert float(wb.mean(torch.randn(5, 8)).abs().max()) <= 1.5 + 1e-6


def test_gate_threshold_refractory_and_force():
    gate = InvocationGate((0, 1), threshold=0.5, refractory_s=1.0)
    hot = torch.zeros(1, 3, 2)
    hot[0, :2, 0] = 1.0
    cold = torch.zeros(1, 3, 2)
    assert gate(hot, 0.0)
    assert not gate(hot, 0.5)          # refractory
    assert gate(hot, 1.5)
    assert not gate(cold, 5.0)         # below threshold
    assert gate(cold, 5.1, force=True)  # message forces it
