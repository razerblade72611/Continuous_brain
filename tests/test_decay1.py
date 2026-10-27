"""DECAY-1 pure parts (no model): background-day matching and the frozen analysis rules."""
import copy
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


dc = load("decay1_t", "scripts/decay1.py")
da = load("decay1_an_t", "scripts/decay1_analyze.py")


def test_background_day_matched_unique_and_rng_isolated():
    bg = [f"sentence {'w ' * (i % 17)} number {i}." for i in range(400)]
    targets = [f"Fact text {'x ' * (i % 9)}{i}." for i in range(40)]
    ntok = lambda t: len(t.split())
    state = random.getstate()
    day = dc.background_day(2, bg, targets, ntok)
    assert random.getstate() == state                                  # global/training RNG untouched
    assert len(day) == 40 and len(set(day)) == 40 and set(day) <= set(bg)
    assert all(abs(ntok(d) - ntok(t)) <= 1 for d, t in zip(day, targets))
    assert day == dc.background_day(2, bg, targets, ntok) != dc.background_day(3, bg, targets, ntok)


def fake(seed, dV, dB=None, repro=0.0, teacher_ok=True, acq=4, e_uptake=True):
    dB = dB or {a: 0.0 for a in dc.ARMS}

    def rows(V, B, fc=True, n_fc=4, uptake=True):
        return [{"lV_A": V, "lZ_A": V - B, "fc": i < n_fc, "counter": {"lZ_C": -1.0, "lV_C": -5.0},
                 "treat": {"U": {"fc": uptake}}} for i in range(4)]
    h = da.EXPECTED_BACKBONE
    pl = {"eval": 3.0, "modern": 4.0}
    arms = {a: {"targets": rows(-5.0 + dV[a], 10.0 + dB[a], uptake=(e_uptake or a != "E")), "ppl_loss": dict(pl),
                "teacher_hash_start": "t" if a == "E" else None,
                "teacher_hash_end": ("t" if teacher_ok else "u") if a == "E" else None} for a in dc.ARMS}
    tr = {"steps": 400, "schedule_sha256": "s"}
    return {"seed": seed, "status": "complete", "commit": "c", "prefix_check": {"fails": 0},
            "config": {"arm": "L", "lr": 1e-4, "kl_weight": 1.0, "night_passes": 4},
            "backbone_sha256": h, "backbone_start": h, "backbone_end": h, "base_ppl_loss": dict(pl),
            "checkpoint1": {"targets": rows(-5.0, 10.0, n_fc=acq), "ppl_loss": dict(pl)}, "arms_out": arms,
            "traces": {a: dict(tr) for a in dc.ARMS}, "rng_after": {a: "r" for a in dc.ARMS},
            "reproduction": {k: {"max_abs_diff": repro if k == "cp1" else 0.0, "n_mismatches": 0,
                                 "memory_hash_match": True} for k in ("cp1", "A_vs_U2")},
            "anchor_difference": {"max": 0.5}, "state_restore_test": {"ok": True}}


SEEDS = [2, 3, 4, 5]


def test_kl_is_the_culprit_pattern():
    dV = {"A": -30, "B": -8, "C": -25, "D": -2, "E": -10}
    res = da.analyse([fake(s, dV) for s in SEEDS])
    v, st = res["validity"], res["verdict"]
    assert v["V1"] and v["V2"] and v["V3"] and v["V3b"] and v["R"] and v["T"] and v["V5"]
    assert "frozen-backbone KL adds forgetting during new-fact learning" in st["dV"]
    assert "frozen-backbone KL adds forgetting without new facts" in st["dV"]
    assert "new-fact learning causes forgetting without KL" in st["dV"]
    assert "previous-self teacher preserves more: teacher choice matters" in st["dV"]
    assert not any("background cross-entropy" in x for x in st["dV"])


def test_background_dynamics_pattern_and_unresolved_e():
    dV = {"A": -30, "B": -29, "C": -25, "D": -24, "E": -30}
    st = da.analyse([fake(s, dV) for s in SEEDS])["verdict"]["dV"]
    assert any("background cross-entropy" in x for x in st)
    assert "E-A unresolved: no general conclusion about previous-self consolidation" in st
    assert not any("KL adds" in x for x in st)


def test_mixed_signs_not_robust():
    rs = [fake(s, {"A": a, "B": 0, "C": 0, "D": 0, "E": 0}) for s, a in zip(SEEDS, (-20, -15, 3, 4))]
    e = da.analyse(rs)["effects"]["dV"]["A-B"]
    assert e["mean"] == -7.0 and not e["robust_neg"] and not e["robust_pos"]


def test_gates():
    dV = {"A": -30, "B": -8, "C": -25, "D": -2, "E": -10}
    assert da.analyse([fake(s, dV, acq=1) for s in SEEDS])["verdict"]["status"].startswith("INVALID")
    bad = [fake(s, dV) for s in SEEDS]
    bad[2]["traces"]["C"]["schedule_sha256"] = "other"
    assert da.analyse(bad)["verdict"]["status"].startswith("INVALID")
    dup = [fake(s, dV) for s in (2, 3, 4, 4)]
    assert not da.analyse(dup)["validity"]["V1"]
    r = da.analyse([fake(s, dV, repro=0.5) for s in SEEDS])["verdict"]
    assert r["status"] == "VALID" and any("REPRODUCTION" in q for q in r["qualifiers"])
    t = da.analyse([fake(s, dV, teacher_ok=False) for s in SEEDS])["verdict"]
    assert any(x.startswith("E-A INVALID") for x in t["dV"])
    hot = [fake(s, dV) for s in SEEDS]
    hot[0]["arms_out"]["C"]["ppl_loss"]["modern"] = 4.2
    assert any("COMPETENCE" in q for q in da.analyse(hot)["verdict"]["qualifiers"])


def test_sol_revisions():
    dV = {"A": -30, "B": -8, "C": -25, "D": -2, "E": -10}
    res = da.analyse([fake(s, dV) for s in SEEDS])
    assert res["interaction"]["dV"]["per_seed"] == [1.0] * 4                    # (-30+8) - (-25+2) = 1
    st = da.analyse([fake(s, dV, e_uptake=False) for s in SEEDS])["verdict"]
    assert any("improved retention with insufficient new-fact acquisition" in x for x in st["dV"])
    assert any("uptake below standard" in q for q in st["qualifiers"])
    flat = [fake(s, dV) for s in SEEDS]
    flat[1]["anchor_difference"]["max"] = 0.0                                  # anchor indistinguishable -> E-A invalid
    assert any("E-A INVALID" in x for x in da.analyse(flat)["verdict"]["dV"])
    nores = [fake(s, dV) for s in SEEDS]
    nores[0]["state_restore_test"]["ok"] = False
    assert any("NOT RESTORABLE" in q for q in da.analyse(nores)["verdict"]["qualifiers"])
    mism = [fake(s, dV) for s in SEEDS]
    mism[3]["reproduction"]["A_vs_U2"]["n_mismatches"] = 1                     # a non-numeric field differs
    assert not da.analyse(mism)["validity"]["R"]


def test_compare_records():
    a = {"targets": [{"fc": True, "lp": -1.0, "cands": [{"v": "x", "lp": -2.0}]}], "memory_sha256": "h"}
    b = copy.deepcopy(a)
    assert dc.compare_records(a, b) == (0.0, [])
    b["targets"][0]["cands"][0]["lp"] = -2.0005
    d, bad = dc.compare_records(a, b)
    assert abs(d - 0.0005) < 1e-9 and bad == []
    b["targets"][0]["fc"] = False
    assert dc.compare_records(a, b)[1] == [".targets[0].fc"]
    b["targets"].append({})
    assert dc.compare_records(a, b)[1] == [".targets[len]"]
