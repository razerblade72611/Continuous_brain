"""The chooser never applies or acts on non-finite numbers (the first L1 GPU sleep's value head overflowed to NaN)."""
import math

import torch

from continuous_brain_cb2.chooser import Chooser, ChooserConfig, N_COMMANDS
from continuous_brain_cb2.world_model import N_ACTIONS


def _batch(B=4, T=20, F=32):
    return (torch.randn(B, T, F), torch.randint(0, N_COMMANDS, (B, T)), torch.randint(0, N_ACTIONS, (B, T)),
            torch.randn(B, T), torch.zeros(B, T, dtype=torch.bool), torch.zeros(B, T))


def test_nonfinite_update_is_skipped_and_params_unchanged():
    torch.manual_seed(0)
    ch = Chooser(ChooserConfig(), 32)
    f, c, a, r, d, s = _batch()
    r[0, 5] = float("nan")
    before = [p.clone() for p in ch.parameters()]
    m = ch.train_step(f, c, a, r, d, s)
    assert m["skipped"] == 1
    assert all(torch.equal(x, y) for x, y in zip(before, ch.parameters()))


def test_value_is_bounded_and_training_stays_finite():
    torch.manual_seed(0)
    ch = Chooser(ChooserConfig(), 32)
    with torch.no_grad():
        ch.value[-1].bias.fill_(500.0)
    f, c, a, r, d, s = _batch()
    assert torch.isfinite(ch.v(f)).all()
    m = ch.train_step(f, c, a, r, d, s)
    assert m["skipped"] == 0 and math.isfinite(m["value_loss"])
