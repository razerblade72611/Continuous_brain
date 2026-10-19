import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from being.memory import TitansMemory  # noqa: E402


def _write(mem, k, v, steps=200):
    for _ in range(steps):
        F.mse_loss(mem(k), v).backward()
        mem.titans_step(lr=0.05, momentum=0.9, forget=0.0)


def test_born_empty_and_writes_association():
    mem = TitansMemory(16, 64, 8, birth_seed=0)
    k, v = torch.randn(16), torch.randn(8)
    assert mem(k).abs().max() == 0
    _write(mem, k, v)
    assert F.cosine_similarity(mem(k), v, dim=0) > 0.99


def test_forget_gate_erases():
    mem = TitansMemory(16, 64, 8, birth_seed=0)
    k, v = torch.randn(16), torch.randn(8)
    _write(mem, k, v)
    before = mem(k).norm().item()
    for n, p in mem.trainable():
        mem.momentum_state[n].zero_()
    for _ in range(20):
        mem(k).sum().mul(0).backward()
        mem.titans_step(lr=0.0, momentum=0.0, forget=0.5)
    assert mem(k).norm().item() < 0.01 * before


def test_snapshot_restore_is_exact():
    mem = TitansMemory(16, 64, 8, birth_seed=0)
    k, v = torch.randn(16), torch.randn(8)
    snap = mem.snapshot()
    h0 = mem.weight_hash()
    _write(mem, k, v, steps=10)
    assert mem.weight_hash() != h0
    mem.restore(snap)
    assert mem.weight_hash() == h0
