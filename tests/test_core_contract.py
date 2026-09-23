"""Contract every SpikingCore must satisfy.

When InfantBrain is ported, add a factory for its adapter to CORE_FACTORIES
and this whole file runs against it unchanged.
"""

import pytest
import torch

from conftest import small_core

CORE_FACTORIES = {
    "reference_lif": lambda: small_core(batch_size=2),
}


@pytest.fixture(params=list(CORE_FACTORIES))
def any_core(request):
    return CORE_FACTORIES[request.param]()


def _drive(core, value=1.0):
    return torch.full((core.batch_size, core.n_input), value)


def test_step_returns_projection_events_in_window(any_core):
    t0 = any_core.time_s
    ev = any_core.step(_drive(any_core), 0.05)
    assert any_core.time_s == pytest.approx(t0 + 0.05)
    assert ev.n_neurons == any_core.n_projection and ev.batch_size == any_core.batch_size
    if ev.n_events:
        assert float(ev.time.min()) > t0 - 1e-9 and float(ev.time.max()) <= any_core.time_s + 1e-9


def test_rejects_wrong_input_shape(any_core):
    with pytest.raises(ValueError):
        any_core.step(torch.zeros(any_core.batch_size, any_core.n_input + 1), 0.01)


def test_reseed_makes_runs_reproducible(any_core):
    saved = any_core.state_dict()
    any_core.reset_state(); any_core.reseed(7)
    a = any_core.step(_drive(any_core), 0.05)
    any_core.load_state_dict(saved)
    any_core.reset_state(); any_core.reseed(7)
    b = any_core.step(_drive(any_core), 0.05)
    assert torch.equal(a.neuron, b.neuron) and torch.equal(a.time - a.time.min() if a.n_events else a.time,
                                                           b.time - b.time.min() if b.n_events else b.time)


def test_state_round_trip_is_exact(any_core):
    any_core.step(_drive(any_core), 0.03)
    saved = any_core.state_dict()
    a = any_core.step(_drive(any_core), 0.05)
    any_core.load_state_dict(saved)
    b = any_core.step(_drive(any_core), 0.05)
    assert torch.equal(a.neuron, b.neuron) and torch.equal(a.time, b.time)


def test_no_learning_without_apply(any_core):
    before = any_core.state_dict()
    any_core.set_plasticity_tracking(True)
    any_core.step(_drive(any_core), 0.1)
    any_core.set_plasticity_tracking(False)
    after = any_core.state_dict()
    for key in before:
        if key in ("w", "weights"):
            assert torch.equal(before[key], after[key])


def test_plasticity_is_modulated_and_consumed(any_core):
    saved = any_core.state_dict()
    any_core.set_plasticity_tracking(True)
    any_core.step(_drive(any_core), 0.1)
    stats = any_core.apply_plasticity(1.0)
    assert stats["mean_abs_dw"] > 0
    again = any_core.apply_plasticity(1.0)        # eligibility already consumed
    assert again["mean_abs_dw"] == 0
    any_core.load_state_dict(saved)
    any_core.set_plasticity_tracking(True)
    any_core.step(_drive(any_core), 0.1)
    assert any_core.apply_plasticity(0.0)["mean_abs_dw"] == 0   # zero modulator: no change


def test_reset_state_keeps_weights(any_core):
    any_core.set_plasticity_tracking(True)
    any_core.step(_drive(any_core), 0.1)
    any_core.apply_plasticity(1.0)
    w = any_core.state_dict()
    any_core.reset_state()
    w2 = any_core.state_dict()
    for key in ("w",):
        if key in w:
            assert torch.equal(w[key], w2[key])
