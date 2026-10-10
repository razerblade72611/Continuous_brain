"""CB2-L (L-A): the remembered-map head of the predictive core."""
import numpy as np
import torch

from continuous_brain_cb2.world_model import MAP_CLASSES, CoreConfig, PredictiveCore


def _core(**kw):
    torch.manual_seed(0)
    return PredictiveCore(CoreConfig(deter=32, hidden=32, stoch=4, discrete=4, cnn_depth=4, units=32, **kw), "cpu")


def test_map_target_geometry_and_mask():
    core = _core(map_radius=2)
    crop = np.full((1, 1, 33, 33), -1, np.int8)
    crop[0, 0, 16, 16] = 5                                   # its own tile
    crop[0, 0, 14, 18] = 3                                   # 2 up, 2 right: inside the 5 x 5 target
    t = core.map_target(crop.reshape(1, 1, -1)).numpy().reshape(5, 5)   # stored flat, read back as a grid
    assert t[2, 2] == 6 and t[0, 4] == 4 and t[1, 1] == 0
    oov = core.out_of_view_mask().reshape(5, 5)
    assert not oov.any()                                     # a 5 x 5 square lies inside the 7 x 9 view
    big = _core(map_radius=8).out_of_view_mask().reshape(17, 17)
    assert big.sum() == 17 * 17 - 7 * 9 and not big[8, 8] and big[0, 0]


def test_head_is_added_to_a_saved_core_without_changing_it():
    old = _core()
    saved = old.state_dict_full()
    new = PredictiveCore.from_state(saved, "cpu", map_radius=3)
    assert "map" in new.heads
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k]), k
    feat = torch.zeros(2, 5, new.feat_size)
    assert new.map_logits(feat).shape == (2, 5, 49, MAP_CLASSES)
