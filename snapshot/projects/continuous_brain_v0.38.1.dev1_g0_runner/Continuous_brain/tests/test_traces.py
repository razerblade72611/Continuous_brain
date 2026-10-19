"""DAV-33: analytical decay tests pass and traces remain stable across variable dt."""

import math

import numpy as np
import pytest
import torch

from continuous_brain_snn.traces import MultiscaleTraces

TAUS = (0.005, 0.025, 0.1, 0.5, 2.0)


def _one_spike():
    t = MultiscaleTraces(1, 1, TAUS, dtype=torch.float64)
    t.update(torch.ones(1, 1), 0.0)
    return t


@pytest.mark.parametrize("elapsed", [0.001, 0.01, 0.1, 1.0, 5.0])
def test_single_spike_matches_analytic_decay(elapsed):
    t = _one_spike()
    t.decay(elapsed)
    expected = [math.exp(-elapsed / tau) for tau in TAUS]
    assert np.allclose(t.x[0, 0].numpy(), expected, rtol=1e-12)


def test_variable_dt_partitions_agree():
    rng = np.random.default_rng(0)
    for _ in range(10):
        total = 0.7
        cuts = np.sort(rng.random(20)) * total
        dts = np.diff(np.concatenate([[0.0], cuts, [total]]))
        a, b = _one_spike(), _one_spike()
        for dt in dts:
            a.decay(float(dt))
        b.decay(total)
        assert torch.allclose(a.x, b.x, rtol=1e-12)


def test_steady_state_under_regular_spiking_matches_geometric_series():
    t = MultiscaleTraces(1, 1, TAUS, dtype=torch.float64)
    dt = 0.01
    for _ in range(5000):
        t.update(torch.ones(1, 1), dt)
    expected = [1.0 / (1.0 - math.exp(-dt / tau)) for tau in TAUS]
    assert np.allclose(t.x[0, 0].numpy(), expected, rtol=1e-6)


def test_bounded_under_random_dt_and_spikes():
    t = MultiscaleTraces(50, 2, TAUS)
    g = torch.Generator().manual_seed(0)
    for _ in range(3000):
        dt = float(torch.rand(1, generator=g)) * 0.004 + 1e-4
        t.update((torch.rand(2, 50, generator=g) < 0.05).float(), dt)
    assert bool(torch.isfinite(t.x).all())
    min_dt = 1e-4
    bound = torch.tensor([1.0 / (1.0 - math.exp(-min_dt / tau)) for tau in TAUS])
    assert bool((t.x <= bound + 1e-3).all()) and bool((t.x >= 0).all())


def test_population_summary_is_member_mean(engine):
    engine.run(200)
    summary = engine.traces.population_summary(engine.pop_mean)
    p = 0
    idx = torch.from_numpy(engine.anatomy.populations[p].index)
    assert torch.allclose(summary[0, p], engine.traces.x[0, idx].mean(dim=0), atol=1e-5)
