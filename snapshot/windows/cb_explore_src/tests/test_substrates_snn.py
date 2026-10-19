import numpy as np
import pytest
import torch

from continuous_brain_protocols import (Capability, CapabilityUnavailable, GradientFirewallError, LesionSpec,
                                        Substrate)
from continuous_brain_snn.config import small_config
from continuous_brain_snn.engine import SNNEngine
from continuous_brain_snn.inputs import SyntheticDrive
from continuous_brain_substrates import SNNSubstrate


def frame(side, rng=None, eff=0.2):
    rng = rng or np.random.default_rng(0)
    return {"luminance": rng.random((side, side)), "joint_angle": np.array([0.3, -0.4]),
            "joint_velocity": np.array([1.5, -2.0]), "motor_efference": np.full((2, 3), eff)}


@pytest.fixture
def snn():
    return SNNSubstrate(SNNEngine(small_config(3)))


def test_is_a_substrate_with_h1_metadata(snn):
    assert isinstance(snn, Substrate)
    m = snn.metadata
    assert m.track == "H1" and m.kind == "snn-e0"
    assert Capability.GRADIENT_TRAINING not in m.capabilities
    assert Capability.LOCAL_PLASTICITY not in m.capabilities


def test_transduction_matches_the_e0_drive(snn):
    """The adapter's retina/proprio currents equal SyntheticDrive's for the same physical input."""
    side = snn.encoder_cfg.retina_side
    f = frame(side)
    cur = snn._current(f)[0]
    drive = SyntheticDrive(snn.engine)
    lum = torch.tensor(f["luminance"], dtype=torch.float32)[None, None]
    dog = torch.nn.functional.conv2d(lum, drive.kernel, padding=drive.kernel.shape[-1] // 2)[0, 0].T.reshape(-1)
    on = snn.engine.anatomy.pop("retina_on").index
    assert torch.allclose(cur[on], 6.0 * torch.relu(dog), atol=1e-5)
    idx, pref, width, joint, q = drive.proprio[0]
    expected = 1.2 * torch.exp(-0.5 * ((pref - float(f["joint_angle"][joint])) / width) ** 2)
    assert torch.allclose(cur[idx], expected, atol=1e-5)


def test_step_substeps_and_motor_output(snn):
    out = snn.step(frame(snn.encoder_cfg.retina_side), 0.01)
    assert snn.engine.steps == 10
    assert out["motor_command"].shape == (2, 3)
    assert np.all((out["motor_command"] >= 0) & (out["motor_command"] <= 1))
    with pytest.raises(ValueError):
        snn.step(frame(snn.encoder_cfg.retina_side), 0.0105)
    with pytest.raises(ValueError, match="invalid frame"):
        snn.step({"luminance": np.zeros((2, 2))}, 0.01)


def test_gradient_firewall(snn):
    with pytest.raises(GradientFirewallError):
        snn.train_step(None)
    snn.step(frame(snn.encoder_cfg.retina_side), 0.01)
    snn.assert_gradient_firewall()                       # nothing requires grad after stepping
    for obj in (snn.engine, snn.engine.slots, snn.engine.traces):
        for v in vars(obj).values():
            if isinstance(v, torch.Tensor):
                assert not v.requires_grad
    snn.engine.slots.weight.requires_grad_(True)
    with pytest.raises(GradientFirewallError, match="slots.weight"):
        snn.assert_gradient_firewall()


def test_no_local_plasticity_until_e1(snn):
    with pytest.raises(CapabilityUnavailable, match="DAV-40"):
        snn.plasticity_step({})


def test_freeze_holds_thresholds_exactly(snn):
    f = frame(snn.encoder_cfg.retina_side)
    snn.step(f, 0.01)
    snn.freeze()
    th = snn.engine.threshold.clone()
    for _ in range(5):
        snn.step(f, 0.01)
    assert torch.equal(th, snn.engine.threshold)
    snn.unfreeze()
    snn.step(f, 0.01)
    assert not torch.equal(th, snn.engine.threshold)


def test_determinism_and_exact_restore():
    a = SNNSubstrate(SNNEngine(small_config(5)))
    b = SNNSubstrate(SNNEngine(small_config(5)))
    side = a.encoder_cfg.retina_side
    for i in range(4):
        f = frame(side, np.random.default_rng(i))
        a.step(f, 0.01)
        b.step(f, 0.01)
    assert a.state_hash() == b.state_hash()
    saved = a.state_dict()
    c = SNNSubstrate(SNNEngine(small_config(5)))
    c.load_state_dict(saved)
    assert c.state_hash() == a.state_hash()
    for i in range(3):
        f = frame(side, np.random.default_rng(10 + i))
        a.step(f, 0.01)
        c.step(f, 0.01)
    assert c.state_hash() == a.state_hash()
    np.testing.assert_array_equal(a.features().read_features(), c.features().read_features())
    with pytest.raises(ValueError):
        SNNSubstrate(SNNEngine(small_config(6))).load_state_dict(saved)


def test_lesions(snn):
    before = int(snn.engine.slots.active.sum())
    rec = snn.lesion(LesionSpec("synapses:cortex", 0.1, 1))
    assert rec.removed == round(0.1 * rec.of_total)
    assert int(snn.engine.slots.active.sum()) == before - rec.removed
    rec2 = snn.lesion(LesionSpec("neurons:cortex", 0.05, 2))
    dead = snn._silenced
    assert int(dead.sum()) == rec2.removed
    slots = snn.engine.slots
    assert not bool((slots.active & (dead[:, None] | dead[slots.src])).any())
    snn.step(frame(snn.encoder_cfg.retina_side), 0.01)
    assert not bool(snn.engine.spikes[0, dead].any())
    assert snn.engine.slots.check_invariants(snn.engine.cfg.connectivity) == []
    with pytest.raises(ValueError):
        snn.lesion(LesionSpec("units:hidden", 0.1, 0))


def test_budget_and_features(snn):
    b = snn.budget()
    assert b.active_synapses == int(snn.engine.slots.active.sum()) == b.learned_scalars
    spec = snn.features().feature_spec()
    assert spec.size == snn.engine.n and spec.filter_tau_s == pytest.approx(0.025)
    state = snn.expose_state()
    with pytest.raises(ValueError):
        state["v"][0] = 5.0


def test_state_hash_covers_the_noise_generator(snn):
    h = snn.state_hash()
    snn.engine.noise_gen.manual_seed(12345)
    assert snn.state_hash() != h
