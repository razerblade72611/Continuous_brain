"""WRITE-1 frozen analysis rules (no model)."""
import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


wa = load("write1_an_t", "scripts/write1_analyze.py")
EXACT = {"max_abs_diff": 0.0, "n_mismatches": 0, "memory_hash_match": True}


def rows(dV=0.0, fc=4, bind=4, uptake=4):
    return [{"lV_A": -5.0 + dV, "lZ_A": -15.0 if i < bind else 0.0, "lU_A": -20.0, "fc": i < fc,
             "counter": {"fc": True, "lZ_C": -1.0, "lV_C": -5.0}, "treat": {"U": {"fc": i < uptake}}} for i in range(4)]


def fake(seed, arms):
    """arms: {arm: dict(dV, fc, bind, uptake)}"""
    h = wa.EXPECTED_BACKBONE
    pl = {"eval": 3.0, "modern": 4.0}
    tr = {"steps": 400, "schedule_sha256": "s"}
    return {"seed": seed, "status": "complete", "commit": "c", "prefix_check": {"fails": 0},
            "config": {"arm": "L", "lr": 1e-4, "kl_weight": 1.0, "night_passes": 4},
            "backbone_sha256": h, "backbone_start": h, "backbone_end": h, "base_ppl_loss": dict(pl),
            "checkpoint1": {"targets": rows(), "ppl_loss": dict(pl)},
            "arms_out": {a: {"targets": rows(**k), "ppl_loss": dict(pl)} for a, k in arms.items()},
            "P_readouts": {"new": {"targets": rows(dV=-40, fc=0, bind=0), "ppl_loss": dict(pl)}},
            "traces": {a: dict(tr) for a in wa.ARMS}, "rng_after": {a: "r" for a in wa.ARMS},
            "trainable_params": {a: 100 for a in wa.ARMS},
            "reproduction": {k: dict(EXACT) for k in ("cp1", "A_vs_U2", "A_vs_DECAY1_A")},
            "isolation_integrity": dict(EXACT), "P_zero_init_check": dict(EXACT)}


BAD = dict(dV=-25, fc=0, bind=1)
GOOD = dict(dV=-1, fc=4, bind=4)
SEEDS = [2, 3, 4, 5]


def verdict(arms):
    return wa.analyse([fake(s, arms) for s in SEEDS])


def test_four_way_classification():
    assert verdict({"A": BAD, "R": BAD, "P": GOOD})["verdict"]["P_primary"] == "isolated incremental memory works for one life"
    assert "readout interference demonstrated" in verdict({"A": BAD, "R": BAD, "P": BAD})["verdict"]["P_primary"]
    assert "stability purchased" in verdict({"A": BAD, "R": BAD, "P": dict(GOOD, uptake=0)})["verdict"]["P_primary"]
    assert "fails both" in verdict({"A": BAD, "R": BAD, "P": dict(BAD, uptake=1)})["verdict"]["P_primary"]
    r = verdict({"A": BAD, "R": BAD, "P": GOOD})
    assert r["effects"]["dV"]["P-A"]["robust_pos"] and r["validity"]["V5"]


def test_preservation_requires_same_seed():
    # fc passes in seeds 2,3,4 and binding passes in seeds 3,4,5: only seeds 3,4 qualify -> not preserved
    rs = []
    for s in SEEDS:
        p = dict(dV=-1, fc=4 if s != 5 else 0, bind=4 if s != 2 else 0)
        rs.append(fake(s, {"A": BAD, "R": BAD, "P": p}))
    assert not wa.analyse(rs)["classification"]["P"]["preserved"]


def test_optimizer_reset_note():
    r = verdict({"A": BAD, "R": dict(BAD, dV=-15), "P": dict(BAD, dV=-15)})
    assert "optimizer reset deserves investigation" in r["verdict"]["notes"]


def test_gates():
    base = {"A": BAD, "R": BAD, "P": GOOD}
    rs = [fake(s, base) for s in SEEDS]
    rs[1]["reproduction"]["A_vs_DECAY1_A"]["n_mismatches"] = 2
    assert wa.analyse(rs)["verdict"]["status"].startswith("INVALID")
    rs = [fake(s, base) for s in SEEDS]
    rs[0]["isolation_integrity"]["max_abs_diff"] = 0.5
    assert wa.analyse(rs)["verdict"]["status"].startswith("INVALID")
    rs = [fake(s, base) for s in SEEDS]
    rs[2]["P_zero_init_check"]["n_mismatches"] = 1
    assert wa.analyse(rs)["verdict"]["status"].startswith("INVALID")
    rs = [fake(s, base) for s in SEEDS]
    rs[3]["trainable_params"]["P"] = 999
    assert wa.analyse(rs)["verdict"]["status"].startswith("INVALID")
    rs = [fake(s, base) for s in SEEDS]
    rs[0]["arms_out"]["P"]["ppl_loss"]["modern"] = 4.2
    assert "COMPETENCE BAND EXCEEDED" in wa.analyse(rs)["verdict"]["qualifiers"]
    dup = [fake(s, base) for s in (2, 3, 3, 5)]
    assert not wa.analyse(dup)["validity"]["V1"]


def test_single_adapter_arms_get_neutral_labels():
    c = verdict({"A": BAD, "R": GOOD, "P": BAD})["classification"]
    assert c["A"]["label"] == "old memory not preserved; new facts learned"
    assert c["R"]["label"] == "old memory preserved; new facts learned"
    assert "combining adapters" in c["P"]["label"]
