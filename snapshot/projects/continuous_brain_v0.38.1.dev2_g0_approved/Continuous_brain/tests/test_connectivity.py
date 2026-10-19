"""DAV-35: activate/prune by masks without tensor shape mutation; fan-in/out and occupancy caps validated."""

import numpy as np
import pytest
import torch

from continuous_brain_snn.anatomy import REGIONS, build_anatomy
from continuous_brain_snn.connectivity import SLOT_FIELDS, SparseSlots


@pytest.fixture
def built(cfg):
    a = build_anatomy(cfg)
    return a, SparseSlots.build(a, cfg.connectivity, cfg.seed)


def test_genesis_satisfies_all_invariants(cfg, built):
    _, s = built
    assert s.check_invariants(cfg.connectivity) == []


def test_build_is_deterministic(cfg, built):
    a, s = built
    assert s.connectivity_hash() == SparseSlots.build(a, cfg.connectivity, cfg.seed).connectivity_hash()


def test_prune_and_activate_never_change_shapes(cfg, built):
    _, s = built
    shapes = {f: tuple(getattr(s, f).shape) for f in SLOT_FIELDS}
    act = torch.nonzero(s.active)[:50]
    s.set_active(act[:, 0], act[:, 1], False)
    dormant = torch.nonzero(s.candidate & ~s.active)
    g = torch.Generator().manual_seed(0)
    pick = dormant[torch.randperm(dormant.shape[0], generator=g)[:50]]   # spread across neurons
    s.set_active(pick[:, 0], pick[:, 1], True)
    assert {f: tuple(getattr(s, f).shape) for f in SLOT_FIELDS} == shapes
    assert s.check_invariants(cfg.connectivity) == []


def test_activation_beyond_occupancy_cap_is_refused(cfg, built):
    _, s = built
    row = int(torch.nonzero(s.candidate.sum(1) == s.capacity)[0])      # a neuron with full candidates
    dormant = torch.nonzero(s.candidate[row] & ~s.active[row]).flatten()
    before = s.active.clone()
    with pytest.raises(ValueError, match="occupancy"):
        s.set_active(torch.full_like(dormant, row), dormant, True)
    assert torch.equal(s.active, before)                                 # nothing changed


def test_cannot_activate_a_non_candidate_slot(built):
    _, s = built
    empty = torch.nonzero(~s.candidate)[:1]
    with pytest.raises(ValueError):
        s.set_active(empty[:, 0], empty[:, 1], True)


def test_occupancy_cap_violation_is_detected(cfg, built):
    _, s = built
    s.active |= s.candidate          # fill every candidate slot
    assert any("occupancy" in p for p in s.check_invariants(cfg.connectivity))


def test_fanout_cap_violation_is_detected(cfg, built):
    _, s = built
    cfg.connectivity.fanout_cap = 1
    assert any("fan-out" in p for p in s.check_invariants(cfg.connectivity))


def test_fanout_cap_enforced_at_genesis(cfg):
    cfg.connectivity.fanout_cap = 20
    a = build_anatomy(cfg)
    s = SparseSlots.build(a, cfg.connectivity, cfg.seed)
    assert int(s.fanout().max()) <= 20


def test_sensory_neurons_receive_no_recurrent_input(built):
    a, s = built
    sensory = np.isin(a.region, [REGIONS.index("retina"), REGIONS.index("proprioception")])
    assert int(s.candidate[torch.from_numpy(sensory)].sum()) == 0


def test_motor_inputs_come_only_from_cortex(built):
    a, s = built
    motor = torch.from_numpy(a.region == REGIONS.index("motor"))
    srcs = s.src[motor][s.candidate[motor]]
    assert np.all(a.region[srcs.numpy()] == REGIONS.index("cortex"))


def test_candidates_prefer_nearby_sources(built):
    a, s = built
    ctx = np.nonzero(a.region == REGIONS.index("cortex"))[0][:100]
    xyz = torch.from_numpy(a.coords_mm)
    chosen = torch.cat([(xyz[s.src[i][s.candidate[i]]] - xyz[i]).norm(dim=1) for i in ctx]).mean()
    rng = np.random.default_rng(0)
    random_d = (xyz[rng.integers(0, a.n, 5000)] - xyz[rng.integers(0, a.n, 5000)]).norm(dim=1).mean()
    assert chosen < 0.7 * random_d


def test_ei_split_is_exact(built):
    _, s = built
    w_e, w_i = s.effective()
    assert torch.allclose(w_e + w_i, s.weight * s.active)
    assert bool((w_e[~s.src_exc] == 0).all()) and bool((w_i[s.src_exc] == 0).all())
