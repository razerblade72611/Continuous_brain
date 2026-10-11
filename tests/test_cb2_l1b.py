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
             final_worlds=1200, chooser_off_worlds=300, run=11, rehearsal=False)
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


def _c0(m):
    lives = [{"seed": s, "steps": 100 + s, "n_unlocked": 5, "died": True, "fulfilled": 1, "failed": 0,
              "trace_sha": f"t{s}"} for s in range(3)]
    situ = [{"seed": 1, "situation": "hungry", "steps": 300, "died": False, "trace_sha": "s1"}]
    probes = {k: {"fresh": [dict(x) for x in lives], "situations": [dict(x) for x in situ]} for k in m.ARMS}
    expected = {"fresh": [(s, None) for s in range(3)], "situations": [(1, "hungry")]}
    return probes, expected


def test_instrument_check_flags_any_difference_between_arms_at_cycle_0():
    m = _l1()
    same, exp = _c0(m)
    assert m.instrument_check(same, exp) == []
    same["yoked"]["fresh"][1]["steps"] += 1
    assert m.instrument_check(same, exp) == [("yoked", "fresh", 1, None)]


def test_instrument_check_needs_every_row_and_the_same_life_not_just_the_same_totals():
    """Sol's counterexamples (S-009): an empty yoked arm, and lives with equal totals but different actions."""
    m = _l1()
    p, exp = _c0(m)
    p["yoked"]["fresh"] = []
    assert ("yoked", "fresh", "rows differ from the expected set", None) in m.instrument_check(p, exp)
    p, exp = _c0(m)
    p["learning"]["fresh"][0]["trace_sha"] = "other"           # same steps and unlocks, different life
    assert m.instrument_check(p, exp) == [("learning", "fresh", 0, None)]
    p, exp = _c0(m)
    p["learning"]["fresh"][2]["failed"] = 1                     # same steps and unlocks, different requests
    assert m.instrument_check(p, exp) == [("learning", "fresh", 2, None)]
    p, exp = _c0(m)
    del p["yoked"]
    assert ("yoked", "*", "missing arm", None) in m.instrument_check(p, exp)
    p, exp = _c0(m)
    p["frozen"]["fresh"].append(dict(p["frozen"]["fresh"][0]))  # a duplicated row
    assert ("frozen", "fresh", "rows differ from the expected set", None) in m.instrument_check(p, exp)


def test_final_battery_worlds_are_disjoint_from_everything_and_from_sols_block():
    m = _l1()
    w = m._check_worlds(_plan())
    assert w["final-fresh"] == [84100, 85299] and w["final-transfer"] == [86100, 87299]
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
            blocks.append((b, b + m.FINAL_BLOCK - 1))   # (final ids: base + index, no cycle part)
        else:
            blocks += [(b + c * m.LIFE_STRIDE, b + (c + 1) * m.LIFE_STRIDE - 1) for c in range(11)]
    blocks.sort()
    assert all(x[1] < y[0] for x, y in zip(blocks, blocks[1:]))


def _prim(lf, uf, lt, ut, lyf, uyf, lyt, uyt):
    b = lambda lo, hi: {"lower95": lo, "upper95": hi}
    return {"vs_frozen": {"fresh": b(lf, uf), "transfer": b(lt, ut)},
            "vs_yoked": {"fresh": b(lyf, uyf), "transfer": b(lyt, uyt)}}


SAFE_OK = {m: {f: {"lower95": -0.1} for f in ("fresh", "transfer")} for m in ("n_unlocked", "requests_net")}
ID_OK = {k: {"ok": True} for k in ("learning", "yoked", "frozen")}


def test_verdict_follows_the_preregistered_rules():
    m = _l1()
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60), SAFE_OK, ID_OK)
    assert v["claim"] == "shown" and not v["decisions_needed"]
    v = m.verdict(_prim(5, 90, 3, 80, -40, 20, -30, 25), SAFE_OK, ID_OK)   # growth; the yoked twin grew as much
    assert v["claim"].startswith("growth shown; own closed-loop adaptation") and v["decisions_needed"]
    assert all("cause unproven" in t for t in v["decisions_needed"])
    v = m.verdict(_prim(-40, 25, -50, 20, -40, 25, -30, 20), SAFE_OK, ID_OK)
    assert v["claim"] == "not shown" and "detectable size" in v["decisions_needed"][0]
    v = m.verdict(_prim(10, 90, -60, 20, 5, 70, -30, 40), SAFE_OK, ID_OK)   # fresh only
    assert v["claim"] == "not shown" and "transfer" in v["decisions_needed"][0]
    v = m.verdict(_prim(-5, 60, -10, 50, -5, 60, -10, 50), SAFE_OK, ID_OK)  # wide: inconclusive, nothing to decide
    assert v["claim"] == "not shown" and not v["decisions_needed"]


def test_survival_gain_bought_with_capability_or_identity_is_not_shown():
    m = _l1()
    bad = {**SAFE_OK, "n_unlocked": {"fresh": {"lower95": -0.1}, "transfer": {"lower95": -0.8}}}
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60), bad, ID_OK)
    assert v["claim"] == "not shown: safeguards failed (n_unlocked)"
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60), SAFE_OK, {**ID_OK, "yoked": {"ok": False}})
    assert v["claim"] == "not shown: safeguards failed (identity/provenance/rollback)"
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60), SAFE_OK, None)       # identity never checked: no claim
    assert v["claim"].startswith("not shown: safeguards failed")


def test_new_run_and_blocks_never_reuse_l1():
    m = _l1()
    for r in (10, 9, 6, 0):                                   # 10: touched by rehearsals
        m.BLOCKS["production"]["run"] = r
        with pytest.raises(AssertionError, match="used before"):
            m._check_worlds(_plan())
    m.BLOCKS["production"]["run"] = m.RUN
    m.FRESH0 = 78900                                          # inside L1's probe worlds
    with pytest.raises(AssertionError, match="L1 used"):
        m._check_worlds(_plan())


def test_production_blocks_avoid_every_world_a_rehearsal_touched_and_rehearsals_use_smoke_blocks():
    m = _l1()
    m._check_worlds(_plan())
    m.FINAL_FRESH0 = 84000                                    # where the CPU rehearsals ran
    with pytest.raises(AssertionError, match="rehearsal used"):
        m._check_worlds(_plan())
    m.FINAL_FRESH0 = 84100
    w = m._check_worlds(_plan(rehearsal=True, final_worlds=3, chooser_off_worlds=1, cycles=1, probe_worlds=2))
    assert w["final-fresh"][0] == 98500 and w["fresh-c0"][0] == 96000 and min(w["wake-c0"]) == 80_000_000 + 99 * 1_000_000


def test_a_rehearsal_never_makes_a_claim():
    m = _l1()
    v = m.verdict(_prim(5, 90, 3, 80, 2, 70, 1, 60), SAFE_OK, ID_OK, rehearsal=True)
    assert v["claim"] == "rehearsal: no claim" and not v["decisions_needed"]


def test_pairing_refuses_missing_duplicate_unexpected_or_non_finite_rows():
    """Sol's counterexample (S-009): a missing world and a duplicated one gave a positive bound from incomplete data."""
    m = _l1()
    L = [{"seed": s, "steps": 100.0} for s in range(4)]
    F = [{"seed": s, "steps": 90.0} for s in range(4)]
    assert m.paired(L, F, "steps", expect=range(4))["n"] == 4
    with pytest.raises(ValueError, match="different rows"):
        m.paired(L, F[:3], "steps")
    with pytest.raises(ValueError, match="duplicate"):
        m.paired(L + [dict(L[0])], F, "steps")
    with pytest.raises(ValueError, match="expected worlds"):
        m.paired(L, F, "steps", expect=range(5))
    with pytest.raises(ValueError, match="non-finite"):
        m.paired([*L[:3], {"seed": 3, "steps": float("nan")}], F, "steps")


def _lock(tmp_path, m, **over):
    import json
    out = tmp_path / "run"
    out.mkdir()
    g = m.git_rev()
    a = _plan(cycles=1, workers=2, out=str(out), twin="t", bootstrap="b", resume=False, max_steps=10000,
              core_updates=1500, anchor_updates=1000, chooser_updates=1500, batch=16, T=256, map_radius=8, tau=1.0,
              horizon=15)
    lock = {"format": "cb2-l1b/2", "run_id": "RID", **g, "rehearsal": True, "blocks": m._blocks(_plan()),
            "twin_state_hash": "S2B", "origin_manifest_sha256": "x", "bootstrap_sha256": "BOOT",
            "experiment": m._experiment(a), "args": {**vars(a), "final_worlds": 3, "chooser_off_worlds": 1}, **over}
    (out / "run.json").write_text(json.dumps(lock))
    return out, a, lock


def test_resume_refuses_a_changed_experiment_before_writing_anything(tmp_path, monkeypatch):
    """Sol's counterexample (S-009): resume accepted another revision and max_steps, rewrote the lock, kept the id."""
    import argparse
    import json
    m = _l1()
    out, a, lock = _lock(tmp_path, m)
    (out / "origin.json").write_text(json.dumps({"individual": {"consolidations": 2, "state_hash": "S2B"}}))
    monkeypatch.setattr(m, "_archive_sha", lambda p: "BOOT")
    tw = tmp_path / "t" / "cb2-ind-0001"
    tw.mkdir(parents=True)
    (tw / "individual.json").write_text(json.dumps({"state_hash": "S2B"}))
    a.twin = str(tmp_path / "t")
    changed = argparse.Namespace(**{**vars(a), "max_steps": 17, "workers": 5})
    probs = m.validate_resume(out, changed, lock)
    assert any("max_steps" in p for p in probs) and not any("workers" in p for p in probs)
    monkeypatch.setattr(m, "git_rev", lambda: {"git": "0" * 40, "git_dirty": False})
    assert any("is not the run's" in p for p in m.validate_resume(out, a, lock))
    # and run() refuses before touching the lock
    before = (out / "run.json").read_bytes()
    monkeypatch.setattr(m, "_check_worlds", lambda a: {})
    with pytest.raises(SystemExit, match="refusing to resume"):
        m.run(argparse.Namespace(**{**vars(changed), "resume": True}))
    assert (out / "run.json").read_bytes() == before


def _final_dir(tmp_path, m):
    import json
    out, a, lock = _lock(tmp_path, m)
    hashes = {k: f"H-{k}" for k in m.ARMS}
    for k in m.ARMS:
        h = out / f"twins/{k}/cb2-ind-0001"
        h.mkdir(parents=True)
        (h / "individual.json").write_text(json.dumps({"state_hash": hashes[k]}))
    (out / "cycles.jsonl").write_text(json.dumps({"cycle": 0, "twin_hashes": hashes}) + "\n" +
                                      json.dumps({"cycle": 1, "twin_hashes": hashes}) + "\n")
    (out / "final").mkdir()
    return out


def test_final_refuses_stale_partials_and_never_silently_repeats_an_interrupted_batch(tmp_path):
    import argparse
    import json
    m = _l1()
    out = _final_dir(tmp_path, m)
    (out / "final" / "frozen-fresh.json").write_text(json.dumps({"condition": {"run_id": "OTHER"}, "lives": [],
                                                                 "lives_sha256": m._canon_sha([])}))
    with pytest.raises(SystemExit, match="stale or corrupt"):
        m.final(argparse.Namespace(out=str(out), workers=1))
    (out / "final" / "frozen-fresh.json").unlink()
    (out / "final" / "yoked-transfer.started.json").write_text("{}")
    with pytest.raises(SystemExit, match="started and never recorded"):
        m.final(argparse.Namespace(out=str(out), workers=1))
