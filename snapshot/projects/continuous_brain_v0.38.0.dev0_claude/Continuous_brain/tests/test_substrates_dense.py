import numpy as np
import pytest
import torch

from continuous_brain_protocols import (Capability, CapabilityUnavailable, LesionSpec, SubstrateFrozenError)
from continuous_brain_snn.config import small_config
from continuous_brain_substrates import (BudgetMatchError, DenseRNNConfig, DenseRNNSubstrate, EncoderConfig,
                                         SingleExposureViolation, build_matched_pair, gru_parameter_count,
                                         solve_hidden_size)
from continuous_brain_substrates.dense_rnn import TrainBatch

ENC = EncoderConfig.from_anatomy_config(small_config(0).anatomy)


def frame(i):
    rng = np.random.default_rng(i)
    return {"luminance": rng.random((ENC.retina_side, ENC.retina_side)),
            "joint_angle": rng.uniform(-1, 1, 2), "joint_velocity": rng.uniform(-3, 3, 2),
            "motor_efference": rng.uniform(0, 1, (2, 3))}


def make(seed=0, **kw):
    return DenseRNNSubstrate(ENC, DenseRNNConfig(seed=seed, hidden_size=kw.pop("hidden_size", 16), **kw))


def test_budget_matching_within_ten_percent():
    snn, gru = build_matched_pair(small_config(1))
    target = snn.budget().active_synapses
    assert abs(gru.parameter_count() - target) <= 0.10 * target
    assert gru.parameter_count() == gru_parameter_count(gru.hidden_size, gru.n_in, gru.n_pred, gru.n_motor)
    assert gru.metadata.track == "H2" and Capability.GRADIENT_TRAINING in gru.metadata.capabilities
    with pytest.raises(BudgetMatchError):
        solve_hidden_size(1000, 1000, 6, target=5000)


def test_objective_is_causal():
    """A prediction made at frame t cannot depend on any input after t (no future efference)."""
    g = make()
    for i in range(6):
        g.step(frame(i), 0.01)
    batch = g.causal_batch()
    x = batch.inputs.clone().requires_grad_(True)
    h = batch.h0
    preds = []
    for xt in x:
        h = g.model.cell(xt[None], h)
        preds.append(g.model.pred(h)[0])
    preds[1].sum().backward()
    assert torch.all(x.grad[2:] == 0) and torch.any(x.grad[:2] != 0)
    # targets are the *next* frame's sensory encoding
    assert torch.equal(batch.targets[0], g._buffer[batch.first_input + 1]["s"])


def test_single_exposure_ledger_and_replay_budget():
    g = make(tbptt_frames=4)
    for i in range(6):
        g.step(frame(i), 0.01)
    rep = g.train_step()
    assert rep.updated and rep.targets_used == 4
    replay = TrainBatch(0, torch.stack([g._buffer[i]["x"] for i in range(2)]),
                        torch.stack([g._buffer[i + 1]["s"] for i in range(2)]), torch.zeros(1, g.hidden_size))
    with pytest.raises(SingleExposureViolation):
        g.train_step(replay)
    g2 = make(tbptt_frames=4, replay_budget_targets=2)
    for i in range(6):
        g2.step(frame(i), 0.01)
    g2.train_step()
    assert g2.train_step(replay).detail["replayed_targets"] == 2


def test_freeze_and_capabilities():
    g = make()
    g.step(frame(0), 0.01)
    g.step(frame(1), 0.01)
    g.freeze()
    with pytest.raises(SubstrateFrozenError):
        g.train_step()
    with pytest.raises(CapabilityUnavailable):
        g.plasticity_step({})
    with pytest.raises(ValueError):
        g.step(frame(2), 0.02)


def test_unit_lesion_stays_dead_through_training():
    g = make(tbptt_frames=3)
    rec = g.lesion(LesionSpec("units:hidden", 0.25, 0))
    dead = torch.nonzero(g._unit_mask == 0).flatten()
    assert rec.removed == len(dead) == 4
    for i in range(12):
        g.step(frame(i), 0.01)
        if (i + 1) % 3 == 0:
            g.train_step()
    assert torch.all(g.h[0, dead] == 0)
    assert torch.all(g.model.cell.weight_hh[:, dead] == 0)
    w = g.lesion(LesionSpec("weights:recurrent", 0.1, 1))
    assert w.removed == round(0.1 * g.model.cell.weight_hh.numel())


def test_exact_restore_and_determinism():
    a, b = make(seed=4, tbptt_frames=3), make(seed=4, tbptt_frames=3)
    for i in range(7):
        for g in (a, b):
            g.step(frame(i), 0.01)
            if (i + 1) % 3 == 0:
                g.train_step()
    assert a.state_hash() == b.state_hash()
    c = make(seed=4, tbptt_frames=3)
    c.load_state_dict(a.state_dict())
    assert c.state_hash() == a.state_hash()
    for i in range(7, 12):
        for g in (a, c):
            g.step(frame(i), 0.01)
            if (i + 1) % 3 == 0:
                g.train_step()
    assert c.state_hash() == a.state_hash()
    np.testing.assert_array_equal(a.features().read_features(), c.features().read_features())


def test_reset_opens_a_new_episode_without_reusing_frame_ids():
    g = make(tbptt_frames=4)
    for i in range(6):
        g.step(frame(i), 0.01)
    g.train_step()
    g.reset()
    for i in range(6):
        g.step(frame(100 + i), 0.01)
    rep = g.train_step()
    assert rep.updated and rep.detail["first_target_input"] == 6


def test_explicit_batches_must_come_from_the_causal_buffer():
    g = make(tbptt_frames=4)
    for i in range(6):
        g.step(frame(i), 0.01)
    good = g.causal_batch()
    forged = TrainBatch(good.first_input, good.inputs, good.targets + 1.0, good.h0)       # e.g. probe targets
    with pytest.raises(ValueError, match="does not match"):
        g.train_step(forged)
    future = TrainBatch(10 ** 6, good.inputs, good.targets, good.h0)
    with pytest.raises(ValueError):
        g.train_step(future)
    assert g.train_step(good).updated


def test_explicit_hidden_size_is_still_budget_checked():
    with pytest.raises(BudgetMatchError):
        DenseRNNSubstrate(ENC, DenseRNNConfig(hidden_size=300, target_learned_scalars=10_000))
