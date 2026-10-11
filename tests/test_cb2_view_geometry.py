"""The out-of-view mask must match the remembered-map crop's [x, y] layout (Sol's review S-002: the old mask treated
the crop as image [y, x], so at radius 8 it counted 14 visible cells as out of view and dropped 14 hidden ones)."""
from types import SimpleNamespace

import numpy as np

from continuous_brain_cb2.record import map_crop
from continuous_brain_cb2.world_model import PredictiveCore, in_view_crop
from continuous_brain_individual import memory as mem


def test_in_view_cells_of_the_real_crop_are_exactly_the_mask():
    st = mem.empty()
    pos = (30, 40)
    for _, _, x, y in mem.view_cells(pos):                   # only what is in view right now is known
        st["map"][x, y] = 1
    crop = map_crop(st, pos)
    assert ((crop >= 0) == in_view_crop(33)).all()
    xs, ys = np.nonzero(in_view_crop(33))
    assert in_view_crop(33).sum() == 63 and np.ptp(xs) + 1 == 9 and np.ptp(ys) + 1 == 7   # 9 in x, 7 in y


def test_core_out_of_view_mask_is_the_crop_mask_at_its_radius():
    r = 8
    oov = PredictiveCore.out_of_view_mask(SimpleNamespace(cfg=SimpleNamespace(map_radius=r)))
    want = ~in_view_crop(33)[16 - r:16 + r + 1, 16 - r:16 + r + 1]
    assert (oov == want.reshape(-1)).all() and (~oov).sum() == 63
    old = np.ones((17, 17), bool)                             # the transposed mask L1 used
    old[r - 3:r + 4, r - 4:r + 5] = False
    leaked = (~old.reshape(-1)) & oov                         # hidden cells the old mask excluded
    wrong = old.reshape(-1) & ~oov                            # visible cells the old mask counted as out of view
    assert leaked.sum() == 14 and wrong.sum() == 14
