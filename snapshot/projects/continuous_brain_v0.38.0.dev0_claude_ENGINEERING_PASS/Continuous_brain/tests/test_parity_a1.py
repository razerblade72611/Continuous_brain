import numpy as np
import pytest

from continuous_brain_snn.anatomy import build_anatomy
from continuous_brain_snn.config import small_config
from continuous_brain_eval.parity_a1 import (Z_BONFERRONI_140, bonferroni_z, compare_a1, pilot_false_fail,
                                             pilot_sensitivity, poisson_events, population_arrays)

DT = 0.001


@pytest.fixture(scope="module")
def anatomy():
    return build_anatomy(small_config(0))


def arrays(anatomy, rates, seed, duration=20.0):
    s, n, steps = poisson_events(anatomy, rates, duration, DT, seed)
    return population_arrays(anatomy, DT, s, n, steps)


def test_bonferroni_z_matches_the_amendment():
    assert bonferroni_z(140) == pytest.approx(Z_BONFERRONI_140, abs=0.01)


def test_identical_processes_pass_and_small_populations_are_flagged_underpowered(anatomy):
    res = compare_a1(arrays(anatomy, 8.0, 1), arrays(anatomy, 8.0, 2))
    assert res.passed, res.failures
    assert res.underpowered                                   # 8-cell populations at 8 Hz for 20 s


def test_a_large_rate_shift_in_a_well_powered_population_fails(anatomy):
    rates = {p.name: 8.0 for p in anatomy.populations}
    big = max(anatomy.populations, key=lambda p: p.index.size).name
    shifted = dict(rates)
    shifted[big] = 8.0 * 1.3
    res = compare_a1(arrays(anatomy, rates, 1), arrays(anatomy, shifted, 2))
    assert any(f.startswith(f"rate {big}:") for f in res.failures)


def test_pilots_run_and_report_their_acceptance_criteria(anatomy):
    ff = pilot_false_fail(anatomy, 8.0, n_pairs=3, duration_s=5.0)
    assert ff["pairs"] == 3 and "meets_criterion" in ff
    rates = {p.name: 20.0 for p in anatomy.populations}
    sens = pilot_sensitivity(anatomy, rates, shift=0.5, duration_s=10.0, min_expected_spikes=5000)
    assert sens["eligible_populations"] > 0 and sens["meets_criterion"], sens["missed"]
