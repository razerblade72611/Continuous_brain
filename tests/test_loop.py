import pytest
import torch

from conftest import small_config, small_core
from continuous_brain.agent import ContinuousBrain
from continuous_brain.config import BrainConfig
from continuous_brain.testing import TinyBackbone


@pytest.fixture
def brain(tmp_path):
    return ContinuousBrain(small_core(), TinyBackbone(hidden=64), tmp_path, small_config())


def test_core_runs_continuously_without_consulting(brain):
    results = brain.run(20, sensory=torch.full((32,), 0.8))
    assert brain.core.time_s == pytest.approx(20 * 0.02)
    assert not any(r.consulted for r in results)
    assert len(brain.store) == 0


def test_message_forces_consultation_and_logs_episode(brain):
    brain.run(5)
    r = brain.tick(message="hello")
    assert r.consulted and r.episode_id and r.response is not None
    ep = brain.store.get(r.episode_id)
    assert ep.query == "hello"
    assert ep.arrays["features"].shape == (16, brain.pooling.n_out_channels(brain.features.n_channels))
    n_rows = ep.arrays["core_inputs"].shape[0]
    assert n_rows == min(6, brain.cfg.loop.history_cycles) + brain.cfg.loop.aftermath_cycles


def test_recall_puts_related_memory_in_prompt(brain):
    r = brain.tick(message="What is my name?")
    brain.feedback(r.episode_id, target=" David.")
    brain.run(3)
    r2 = brain.tick(message="What is my name?")
    prompt = brain.store.get(r2.episode_id).prompt
    assert "Relevant memories:" in prompt and "David." in prompt


def test_checkpoint_round_trip(brain):
    brain.run(5)
    snap = brain.save_checkpoint("test")
    w = brain.core.w.clone()
    brain.core.w.add_(1.0)
    brain.load_checkpoint(snap)
    assert torch.equal(brain.core.w, w)


def test_bin_must_match_core_step(tmp_path):
    cfg = BrainConfig()
    cfg.transducer.bin_s = 0.002
    with pytest.raises(ValueError):
        ContinuousBrain(small_core(), TinyBackbone(hidden=64), tmp_path, cfg)


def test_single_individual_only(tmp_path):
    with pytest.raises(ValueError):
        ContinuousBrain(small_core(batch_size=2), TinyBackbone(hidden=64), tmp_path, small_config())
