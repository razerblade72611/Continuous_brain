"""The viewer living the canonical CB2 individual (exploration lane): a life lived in the viewer, even one stopped
and continued mid-life, is the same life, archived the same way, as one lived headless by cb2_collect wake."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from continuous_brain_cb2.archive import Archive
from continuous_brain_individual.viewer import LiveSession

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"


def _vision():
    from continuous_brain_organs.vision import VisionOrgan
    return VisionOrgan(WEIGHTS, CARD)


def _wake(tmp, lives, ind="t-ind"):
    r = subprocess.run([sys.executable, str(REPO / "scripts/explore/cb2_collect.py"), "wake", "--lives", str(lives),
                        "--seed0", "75104", "--individual", ind, "--archive", str(tmp / "archive"),
                        "--individuals", str(tmp / "individuals")], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-3000:]


def test_a_life_in_the_viewer_is_the_life_lived_headless(tmp_path):
    if not WEIGHTS.exists():
        import pytest
        pytest.skip("vision weights missing")
    a, b = tmp_path / "a", tmp_path / "b"
    _wake(a, 1)                                                     # the individual's first life, headless
    shutil.copytree(a, b)
    _wake(a, 1)                                                     # A: its second life, headless
    s = LiveSession(_vision(), {"backend": "dev"}, b / "viewer")     # B: its second life, in the viewer ...
    s.live_as(b / "individuals/t-ind", b / "archive", seed0=75104)
    for _ in range(120):
        s.step()
        s.frame()
    s.save()                                                        # ... stopped mid-life
    meta = json.loads((b / "individuals/t-ind/individual.json").read_text())
    assert meta["in_life"]
    s2 = LiveSession(_vision(), {"backend": "dev"}, b / "viewer")    # ... and continued
    s2.live_as(b / "individuals/t-ind", b / "archive", seed0=75104)
    while s2.alive:
        s2.step()
    for s_ in (s, s2):
        for bad in (lambda: s_.erase("skills"), s_.mark, lambda: s_.new_life(1)):
            try:
                bad()
                raise AssertionError("allowed")
            except ValueError:
                pass
    ma = json.loads((a / "individuals/t-ind/individual.json").read_text())
    mb = json.loads((b / "individuals/t-ind/individual.json").read_text())
    assert mb["state_hash"] == ma["state_hash"] and mb["lives_lived"] == ma["lives_lived"] == 2
    assert not mb["in_life"]
    ta = [t for t in Archive(a / "archive").lived_rows("t-ind", columns=["tick", "action", "store_hash"])]
    tb = [t for t in Archive(b / "archive").lived_rows("t-ind", columns=["tick", "action", "store_hash"])]
    col = lambda ts, k: np.concatenate([t[k].to_numpy(zero_copy_only=False) for t in ts])
    for k in ("tick", "action", "store_hash"):
        assert np.array_equal(col(ta, k), col(tb, k)), k
    Archive(b / "archive").verify()
