"""Battery items run end to end on both substrates (engineering validation only)."""

import numpy as np
import pytest

from continuous_brain_protocols import ResultStatus
from continuous_brain_snn.config import small_config
from continuous_brain_snn.engine import SNNEngine
from continuous_brain_eval.battery import ITEMS, BatteryConfig, check_engineering_seed
from continuous_brain_eval.streams import StreamConfig
from continuous_brain_substrates import DenseRNNConfig, DenseRNNSubstrate, EncoderConfig, SNNSubstrate

ENC = EncoderConfig.from_anatomy_config(small_config(0).anatomy)
STREAM = StreamConfig(encoder=ENC)
CFG = BatteryConfig(develop_frames=60, eval_frames=160, washout_frames=10, horizon_frames=5, raw_lags=5,
                    b2_tasks=2, b3_continue_frames=20, b3_drift_windows=2, b4_eval_frames=500,
                    b4_cue_interval_s=(0.3, 0.4), b4_delays_frames=(2,), b6_recovery_frames=40)


def snn(seed):
    return SNNSubstrate(SNNEngine(small_config(seed, n_neurons=600)))


_TARGET = snn(CFG.seed).budget().active_synapses


def dense(seed):
    return DenseRNNSubstrate(EncoderConfig.from_anatomy_config(small_config(seed, n_neurons=600).anatomy),
                             DenseRNNConfig(seed=seed, target_learned_scalars=_TARGET, tbptt_frames=10))


@pytest.mark.parametrize("factory", [snn, dense], ids=["snn", "dense"])
@pytest.mark.parametrize("item", ["B1", "B2", "B3", "B4", "B5", "B6"])
def test_item_runs_and_never_claims_a_result(factory, item):
    fn = ITEMS[item]
    rec = fn(factory, CFG) if item == "B1" else fn(factory, STREAM, CFG)
    assert rec.status in (ResultStatus.ENGINEERING_ONLY, ResultStatus.CAPABILITY_ABSENT, ResultStatus.BLOCKED)
    assert rec.metric_status in (ResultStatus.UNREGISTERED_METRIC, ResultStatus.BLOCKED)
    assert rec.provenance["config_hash"] and rec.provenance["budget"]["learned_scalars"] > 0
    rec.validate()
    if item == "B1":
        assert rec.status == ResultStatus.BLOCKED
    if item == "B3":
        assert rec.metrics["continuation_features_bit_equal"] and rec.metrics["continuation_end_state_equal"]
        assert rec.metrics["restore_state_hash_equal"]
    if item in ("B2", "B6") and factory is snn:
        assert rec.status == ResultStatus.CAPABILITY_ABSENT
    if item == "B6":
        assert rec.metrics["lesion"]["removed"] > 0
    for v in _numbers(rec.metrics):
        assert np.isfinite(v)


def _numbers(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)


def test_battery_refuses_non_engineering_seeds():
    with pytest.raises(ValueError):
        check_engineering_seed(BatteryConfig(seed=80000))
    with pytest.raises(ValueError):
        check_engineering_seed(BatteryConfig(seed=909500))     # seed + span would leave the family
    check_engineering_seed(CFG)
