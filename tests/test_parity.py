"""Statistical parity harness (DAV-32 note).

The harness must accept statistically identical runs and reject genuinely
different ones. Per-population tolerances need statistical power, so the
synthetic cases use high rates; see the README for the finding that the
current DAV-32 tolerances are underpowered for small populations at 60 s.
"""

import numpy as np
import pytest

from continuous_brain_snn.anatomy import build_anatomy
from continuous_brain_snn.config import small_config
from continuous_brain_snn.engine import SNNEngine
from continuous_brain_snn.parity import (SpikeRecorder, compare, compute_statistics, ks_statistic,
                                         statistics_from_events)

DT, T = 0.001, 60_000


def _poisson(anatomy, rate_hz, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((T, anatomy.n)) < rate_hz * DT
    s, n = np.nonzero(m)
    return statistics_from_events(anatomy, DT, s + 1, n, T)


@pytest.fixture(scope="module")
def anatomy():
    return build_anatomy(small_config(0))


def test_ks_statistic_basics():
    a = np.linspace(0, 1, 100)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 2.0) == 1.0


def test_identical_processes_pass_when_powered(anatomy):
    res = compare(_poisson(anatomy, 60.0, 1), _poisson(anatomy, 60.0, 2))
    assert res.passed, res.failures


def test_a_real_rate_shift_fails(anatomy):
    res = compare(_poisson(anatomy, 60.0, 1), _poisson(anatomy, 72.0, 2))
    assert not res.passed and any(f.startswith("rate") for f in res.failures)


def test_engine_noise_realisations_share_correlation_structure():
    stats = []
    for offset in (0, 1):
        cfg = small_config(0)
        cfg.neuron.homeostasis = False
        e = SNNEngine(cfg)
        e.noise_gen.manual_seed(1000 + offset)   # same body, different noise stream
        rec = SpikeRecorder()
        e.run(10_000, recorder=rec)
        stats.append(compute_statistics(e, rec, 10_000))
    assert compare(stats[0], stats[1]).ks < 0.1
