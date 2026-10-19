import torch

from continuous_brain.working_memory_infant import GatedWorkingMemory
from continuous_brain.working_memory_recall_arms_v2 import (
    LinearRecallBridge,
    GatedRecallBridge,
)


def test_gated_working_memory_is_exact_noop_from_zero_state():
    torch.manual_seed(1)

    writer = GatedWorkingMemory(
        nucleus_dim=32,
        memory_dim=64,
    )

    hidden = torch.randn(5, 32)
    memory = torch.zeros(5, 64)

    next_memory, aux = writer(hidden, memory)

    assert torch.equal(next_memory, memory)
    assert torch.equal(
        aux["candidate"],
        torch.zeros_like(aux["candidate"]),
    )

    expected_gate = torch.full_like(
        aux["write_gate"],
        0.5,
    )
    assert torch.equal(aux["write_gate"], expected_gate)


def test_gated_working_memory_can_become_load_bearing():
    writer = GatedWorkingMemory(
        nucleus_dim=8,
        memory_dim=4,
    )

    with torch.no_grad():
        writer.candidate.bias[0] = 0.5

    hidden = torch.zeros(2, 8)
    memory = torch.zeros(2, 4)

    next_memory, aux = writer(hidden, memory)

    assert next_memory[:, 0].abs().min().item() > 0.0

    # Untouched candidate dimensions remain exactly zero.
    assert torch.equal(
        next_memory[:, 1:],
        torch.zeros_like(next_memory[:, 1:]),
    )

    assert torch.equal(
        aux["write_gate"],
        torch.full_like(aux["write_gate"], 0.5),
    )


def test_linear_recall_bridge_is_exact_noop_for_nonzero_memory():
    torch.manual_seed(2)

    bridge = LinearRecallBridge(
        memory_dim=64,
        bus_dim=128,
    )

    hidden = torch.randn(4, 96)
    memory = torch.randn(4, 64)

    residual, aux = bridge(hidden, memory)

    assert torch.equal(
        residual,
        torch.zeros_like(residual),
    )
    assert torch.equal(
        aux["decoded"],
        torch.zeros_like(aux["decoded"]),
    )


def test_gated_recall_bridge_is_exact_noop_for_nonzero_memory():
    torch.manual_seed(3)

    bridge = GatedRecallBridge(
        nucleus_dim=96,
        memory_dim=64,
        bus_dim=128,
    )

    hidden = torch.randn(4, 96)
    memory = torch.randn(4, 64)

    residual, aux = bridge(hidden, memory)

    assert torch.equal(
        residual,
        torch.zeros_like(residual),
    )

    assert torch.equal(
        aux["decoded"],
        torch.zeros_like(aux["decoded"]),
    )

    assert torch.equal(
        aux["recall_gate"],
        torch.full_like(aux["recall_gate"], 0.5),
    )

    assert torch.equal(
        aux["recall_multiplier"],
        torch.ones_like(aux["recall_multiplier"]),
    )


def test_working_memory_state_dict_roundtrip():
    torch.manual_seed(4)

    original = GatedWorkingMemory(
        nucleus_dim=12,
        memory_dim=7,
    )

    with torch.no_grad():
        original.candidate.weight.normal_(0.0, 0.05)
        original.candidate.bias.normal_(0.0, 0.05)
        original.write_gate.weight.normal_(0.0, 0.05)
        original.write_gate.bias.normal_(0.0, 0.05)

    restored = GatedWorkingMemory(
        nucleus_dim=12,
        memory_dim=7,
    )

    restored.load_state_dict(
        original.state_dict()
    )

    hidden = torch.randn(3, 12)
    memory = torch.randn(3, 7)

    out_a, aux_a = original(hidden, memory)
    out_b, aux_b = restored(hidden, memory)

    assert torch.equal(out_a, out_b)
    assert torch.equal(
        aux_a["candidate"],
        aux_b["candidate"],
    )
    assert torch.equal(
        aux_a["write_gate"],
        aux_b["write_gate"],
    )


def test_working_memory_changes_across_steps_when_writer_is_active():
    writer = GatedWorkingMemory(
        nucleus_dim=6,
        memory_dim=3,
    )

    with torch.no_grad():
        writer.candidate.bias.copy_(
            torch.tensor([0.4, -0.2, 0.1])
        )

    hidden = torch.zeros(1, 6)
    memory0 = torch.zeros(1, 3)

    memory1, _ = writer(hidden, memory0)
    memory2, _ = writer(hidden, memory1)

    assert not torch.equal(memory1, memory0)
    assert not torch.equal(memory2, memory1)

    # Repeated writing should move toward the candidate rather
    # than behaving as a disconnected one-shot output.
    candidate = torch.tanh(
        writer.candidate(hidden)
    )

    error1 = (candidate - memory1).abs().sum()
    error2 = (candidate - memory2).abs().sum()

    assert error2 < error1
