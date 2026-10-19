"""DAV-34: deterministic anatomy hash from seed; no semantic labels in sensory populations."""

import numpy as np
import pytest

from continuous_brain_snn.anatomy import REGIONS, Population, build_anatomy, validate_physical
from continuous_brain_snn.config import E0Config, small_config


def test_hash_is_deterministic_from_seed():
    assert build_anatomy(small_config(3)).anatomy_hash() == build_anatomy(small_config(3)).anatomy_hash()


def test_different_seed_different_anatomy():
    assert build_anatomy(small_config(3)).anatomy_hash() != build_anatomy(small_config(4)).anatomy_hash()


def test_full_size_organism_has_10k_neurons_and_all_regions():
    a = build_anatomy(E0Config())
    assert a.n == 10_000
    for r in REGIONS:
        assert (a.region == REGIONS.index(r)).any(), r


def test_motor_has_antagonist_and_cocontraction_per_joint(cfg):
    a = build_anatomy(cfg)
    for j in range(cfg.anatomy.n_joints):
        for ch in ("plus", "minus", "co"):
            assert a.pop(f"motor_{ch}_j{j}").index.size == cfg.anatomy.motor_cells_per_channel


def test_proprioception_tuning_spans_physical_range(cfg):
    a = build_anatomy(cfg)
    p = a.pop("proprio_angle_j0")
    pref = a.preferred[p.index]
    assert pref.min() == pytest.approx(cfg.anatomy.angle_range_rad[0], abs=1e-4)
    assert pref.max() == pytest.approx(cfg.anatomy.angle_range_rad[1], abs=1e-4)
    assert p.meta["units"] == "rad"


def test_cortex_is_mostly_excitatory_and_sensory_is_excitatory(cfg):
    a = build_anatomy(cfg)
    ctx = a.region == REGIONS.index("cortex")
    assert 0.7 < a.is_exc[ctx].mean() < 0.9
    assert a.is_exc[a.region != REGIONS.index("cortex")].all()


def test_populations_are_physical_only(cfg):
    assert validate_physical(build_anatomy(cfg)) == []


@pytest.mark.parametrize("bad_key", ["object_id", "label", "category", "semantic"])
def test_firewall_rejects_semantic_metadata(cfg, bad_key):
    a = build_anatomy(cfg)
    ret = a.pop("retina_on")
    a.populations.append(Population("retina_tagged", "retina", ret.index, {"polarity": "on", bad_key: "cup"}))
    assert any("non-physical" in p for p in validate_physical(a))


def test_retina_is_topographic_under_cortex(cfg):
    a = build_anatomy(cfg)
    on = a.coords_mm[a.pop("retina_on").index]
    assert np.allclose(on[:, 2], -0.5)
    ex = cfg.anatomy.cortex_extent_mm
    assert on[:, 0].min() > -ex[0] / 2 and on[:, 0].max() < ex[0] / 2
