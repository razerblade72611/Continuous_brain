"""DAV-32: deterministic seeded run; no NaN/Inf; configurable counts; bit-identical on the same device."""

import numpy as np
import pytest
import torch

from continuous_brain_snn.config import small_config
from continuous_brain_snn.engine import SNNEngine
from continuous_brain_snn.inputs import SyntheticDrive
from continuous_brain_snn.parity import SpikeRecorder


def _run(seed, steps=300, n=1200):
    e = SNNEngine(small_config(seed, n))
    rec = SpikeRecorder()
    e.run(steps, input_fn=SyntheticDrive(e, seed=seed), recorder=rec)
    return e, rec.events()


def test_same_seed_is_bit_identical():
    a, (sa, na) = _run(1)
    b, (sb, nb) = _run(1)
    assert a.state_hash() == b.state_hash()
    assert np.array_equal(sa, sb) and np.array_equal(na, nb)


def test_different_seed_differs():
    a, _ = _run(1)
    b, _ = _run(2)
    assert a.state_hash() != b.state_hash()


@pytest.mark.parametrize("n", [800, 1600])
def test_neuron_count_is_configurable(n):
    e, (steps, _) = _run(0, steps=50, n=n)
    assert e.n == n and e.v.shape == (1, n)


def test_no_nonfinite_values_under_drive():
    e, _ = _run(0, steps=1000)
    e.check_finite()


def test_refractory_period_is_respected():
    e, (steps, neurons) = _run(0, steps=500)
    ref = e.ref_steps
    order = np.lexsort((steps, neurons))
    s, n = steps[order], neurons[order]
    same = n[1:] == n[:-1]
    assert (np.diff(s)[same] > ref).all()


def test_excitatory_and_inhibitory_drive_stay_separate_and_nonnegative():
    e, _ = _run(0, steps=300)
    assert bool((e.g_e >= 0).all()) and bool((e.g_i >= 0).all())
    assert float(e.g_i.mean()) > 0 and float(e.g_e.mean()) > 0


def test_homeostasis_raises_threshold_for_overactive_neurons():
    cfg = small_config(0)
    cfg.neuron.target_rate_hz = 0.5            # far below the operating rate
    e = SNNEngine(cfg)
    e.run(1000, input_fn=SyntheticDrive(e))
    assert float(e.threshold.mean()) > cfg.neuron.threshold_base


def test_homeostasis_off_keeps_threshold_fixed():
    cfg = small_config(0)
    cfg.neuron.homeostasis = False
    e = SNNEngine(cfg)
    e.run(200, input_fn=SyntheticDrive(e))
    assert bool((e.threshold == cfg.neuron.threshold_base).all())


def test_batch_instances_share_anatomy_but_get_independent_noise():
    e = SNNEngine(small_config(0), batch_size=2)
    e.run(200)
    assert not torch.equal(e.v[0], e.v[1])


def test_population_drive_targets_only_members(engine):
    cur = engine.current_from_populations({"motor_plus_j0": 1.0})
    idx = engine.anatomy.pop("motor_plus_j0").index
    assert float(cur[0, idx].min()) == 1.0 and float(cur.sum()) == idx.size
