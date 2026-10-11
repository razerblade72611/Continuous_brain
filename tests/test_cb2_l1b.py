"""CB2-L1b: the yoked control, the instrument check, the final-battery world plan and the preregistered verdict
(docs/explore/CB2_L1B_PREREG_DRAFT.md).  Pure functions only: no lives are run here."""
import argparse
from pathlib import Path

import pytest


def _l1():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("l1", Path(__file__).parents[1] / "scripts/explore/cb2_l1.py")
    sys.argv = ["x"]
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _plan(**kw):
    d = dict(cycles=10, probe_worlds=40, lives=20, holdout_lives=2, situation_worlds=24, max_steps=10000,
             final_worlds=1200, chooser_off_worlds=300)
    d.update(kw)
    return argparse.Namespace(**d)


def _homes(tmp_path):
    return {k: tmp_path / k for k in ("learning", "yoked", "frozen")}, \
        {k: tmp_path / f"archive-{k}" for k in ("learning", "yoked", "heldout")}


def test_yoked_experience_is_the_frozen_twins_lives_on_the_same_worlds(tmp_path):
    m = _l1()
    homes, arch = _homes(tmp_path)
    for c in (0, 3, 9):
        w = m.wake_specs(c, _plan(), homes, arch, tmp_path, "rid", {k: k + "-hash" for k in homes})
        L, Y, H = w["learning"], w["yoked_source"], w["heldout"]
        assert Y["home"] == homes["frozen"] and Y["archive"] == str(arch["yoked"]) and Y["state_hash"] == "frozen-hash"
        assert L["home"] == homes["learning"] and L["archive"] == str(arch["learning"])
        assert Y["seeds"] == L["seeds"]                      # same worlds, another author
        assert H["archive"] == str(arch["heldout"]) and not set(H["seeds"]) & set(L["seeds"])
        # the yoked twin's archive never receives a learning-twin life, and the learning archive never a frozen one
        assert all(x["home"] != homes["learning"] for x in w.values() if x["archive"] == str(arch["yoked"]))
        assert all(x["home"] == homes["learning"] for x in w.values() if x["archive"] == str(arch["learning"]))
        ids = [set(range(x["life0"], x["life0"] + len(x["seeds"]))) for x in w.values()]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_both_sleepers_get_their_own_archive_and_nothing_else_differs(tmp_path):
    m = _l1()
    homes, arch = _homes(tmp_path)
    sp = m.sleep_specs(homes, arch)
    assert sp == {"learning": (homes["learning"], arch["learning"]), "yoked": (homes["yoked"], arch["yoked"])}
    assert "frozen" not in sp                                 # frozen never sleeps


def test_instrument_check_flags_any_difference_between_arms_at_cycle_0():
    m = _l1()
    lives = [{"seed": s, "steps": 100 + s, "n_unlocked": 5} for s in range(3)]
    situ = [{"seed": 1, "situation": "hungry", "steps": 300}]
    same = {k: {"fresh": [dict(x) for x in lives], "situations": [dict(x) for x in situ]} for k in m.ARMS}
    assert m.instrument_check(same) == []
    same["yoked"]["fresh"][1]["steps"] += 1
    assert m.instrument_check(same) == [("yoked", "fresh", 1, None)]


def test_final_battery_worlds_are_disjoint_from_everything_and_from_sols_block():
    m = _l1()
    w = m._check_worlds(_plan())
    assert w["final-fresh"] == [84000, 85199] and w["final-transfer"] == [86000, 87199]
    for lo, hi in w.values():                                 # Sol's memory-diagnostic claim is never touched
        assert hi < 83090 or lo > 83195
    with pytest.raises(AssertionError):
        m._check_worlds(_plan(final_worlds=m.FINAL_BLOCK + 1))
    with pytest.raises(AssertionError):
        m._check_worlds(_plan(chooser_off_worlds=1201))
    m.FINAL_FRESH0 = 83000                                    # a block that would reach into the claim is refused
    with pytest.raises(AssertionError, match="claimed by Sol"):
        m._check_worlds(_plan())


def test_life_id_families_are_disjoint_including_final_and_yoked():
    m = _l1()
    blocks = []
    for k, b in m.LID.items():
        if k.startswith("final"):
            blocks.append((b, b + m.FINAL_BLOCK - 1))
        else:
            blocks += [(b + c * m.LIFE_STRIDE, b + (c + 1) * m.LIFE_STRIDE - 1) for c in range(11)]
    blocks.sort()
    assert all(x[1] < y[0] for x, y in zip(blocks, blocks[1:]))


def _prim(lf, uf, lt, ut, lyf, uyf, lyt, uyt):
    b = lambda lo, hi: {"lower95": lo, "upper95": hi}
    return {"vs_frozen": {"fresh": b(lf, uf), "transfer": b(lt, ut)},
            "vs_yoked": {"fresh": b(lyf, uyf), "transfer": b(lyt, uyt)}}


def test_verdict_follows_the_preregistered_rules():
    m = _l1()
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60))
    assert v["claim"] == "shown" and not v["direction_triggers"]
    v = m.verdict(_prim(5, 90, 3, 80, -40, 20, -30, 25))      # growth, but the yoked twin grew as much
    assert v["claim"] == "growth, not shown to be from its own experience"
    assert any("more data" in t for t in v["direction_triggers"])
    v = m.verdict(_prim(-40, 25, -50, 20, -40, 25, -30, 20))  # nothing of detectable size on fresh
    assert v["claim"] == "not shown" and any("futility" in t for t in v["direction_triggers"])
    v = m.verdict(_prim(10, 90, -60, 20, 5, 70, -30, 40))     # fresh only: memorisation-type
    assert v["claim"] == "not shown" and any("memorisation" in t for t in v["direction_triggers"])
    v = m.verdict(_prim(-5, 60, -10, 50, -5, 60, -10, 50))    # wide: inconclusive, no direction change
    assert v["claim"] == "not shown" and not v["direction_triggers"]
