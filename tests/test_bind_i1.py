"""BIND-I1 pure parts (no model): the controlled stream's guarantees and the frozen analysis decision rules."""
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


lg = load("lifegen_i1t", "being/v3/lifegen.py")
i1 = load("bind_i1_t", "scripts/bind_i1.py")
an = load("bind_i1_an_t", "scripts/bind_i1_analyze.py")


def test_stream_guarantees_all_seeds_and_determinism():
    fake_ntok = lambda w: 1 + len(w) // 4
    for seed in sorted(i1.SEEDS):
        st = i1.build_stream(seed, lg, fake_ntok)
        assert i1.check_stream(st, lg)
        d = st["design"]
        assert d["ntok"]["U"] == d["ntok"]["W"]                          # token-count matched
        again = i1.build_stream(seed, lg, fake_ntok)
        assert again["design"] == d                                      # deterministic per seed
        for c in i1.CONDITIONS:
            assert [[f.train_texts for f in L] for L in again["lives"][c]] == \
                   [[f.train_texts for f in L] for L in st["lives"][c]]
        # each condition differs from the others only in the four treatment facts of life 2
        for a, b in (("S", "X"), ("S", "U"), ("X", "U")):
            diff = [(L, j) for L in range(i1.N_LIVES) for j in range(i1.FACTS_PER_LIFE)
                    if st["lives"][a][L][j].train_texts != st["lives"][b][L][j].train_texts]
            assert diff == [(1, j) for j in range(i1.N_TARGETS)]
        # life-2 day size is identical across conditions (equal RNG consumption)
        assert len({sum(len(f.train_texts) for f in st["lives"][c][1]) for c in i1.CONDITIONS}) == 1
        # B's query in S uses the same template frame as A's target query (only the entity differs)
        for i, (a, b) in enumerate(zip(st["targets"], st["treatment"]["S"])):
            assert lg.probes_for(a)["template"].replace(a.entity, "#") == lg.probes_for(b)["template"].replace(b.entity, "#")


def fake_result(seed, dDA_by_cond, treat_ok=True, acq=4):
    """Minimal I1-shaped result: D_A(1) = 10 for all targets, D_A(5) = 10 + dDA[cond]."""
    def row(i, DA, fc=True, treat=None):
        r = {"i": i, "fid": f"t{i}", "fc": fc, "lV_A": -5.0, "lW_A": -5.0 - DA, "lU_A": -20.0, "margin_best": DA,
             "lV_B": -15.0, "lW_B": -4.0, "lU_B": -20.0}
        if treat is not None:
            r["treat"] = treat
        return r
    tr = {"fc": treat_ok, "lp": -3.0, "margin_best": 4.0 if treat_ok else -4.0}
    cp1 = {"after_life": 1, "targets": [row(i, 10.0, fc=i < acq) for i in range(4)], "ppl_loss": {"eval": 3.0}}
    conds = {c: [{"after_life": L, "targets": [row(i, 10.0 + d * (L - 1) / 4, treat=tr) for i in range(4)],
                  "ppl_loss": {"eval": 3.0}} for L in (2, 3, 5)] for c, d in dDA_by_cond.items()}
    return {"seed": seed, "status": "complete", "commit": "c", "config": {"arm": "L"}, "backbone_sha256": "h",
            "backbone_start": "h", "backbone_end": "h", "base_ppl_loss": {"eval": 3.0},
            "wiped": {"targets": [row(i, -1.0) for i in range(4)]}, "checkpoint1": cp1, "conditions": conds,
            "steps": {c: [400, 400, 400, 400] for c in "SXU"}, "rng_after": {c: "r" for c in "SXU"}}


def run_analysis(rs):
    seeds = {r["seed"]: an.per_seed(r) for r in rs}
    eff = {f"{a}-{b}": an.effect(seeds, a, b) for a, b in (("S", "X"), ("S", "U"), ("X", "U"))}
    v = an.validity(rs, seeds)
    return an.interpret(eff, v), eff, v


def test_decision_rules():
    base = [2, 3, 4, 5]
    interp, eff, v = run_analysis([fake_result(s, {"S": -10, "X": -1, "U": 0}) for s in base])
    assert v["V1"] and v["V2"] and v["V3"] and all(v["V4"].values()) and v["V5"]
    assert abs(eff["S-X"]["mean"] - (-9)) < 1e-6 and interp.startswith("1 ")
    assert run_analysis([fake_result(s, {"S": -10, "X": -9, "U": 0}) for s in base])[0].startswith("2 ")
    assert run_analysis([fake_result(s, {"S": -5, "X": -5.5, "U": -4.5}) for s in base])[0].startswith("3 ")
    assert run_analysis([fake_result(s, {"S": 0, "X": -10, "U": 0}) for s in base])[0].startswith("4 ")
    # sign rule: a large mean carried by one seed is not "worse"
    mixed = [fake_result(2, {"S": -40, "X": 0, "U": 0})] + [fake_result(s, {"S": 1, "X": 0, "U": 0}) for s in (3, 4, 5)]
    assert not run_analysis(mixed)[1]["S-X"]["worse"]
    # B fails to learn W in S -> general learning failure, overrides interference
    bad = [fake_result(s, {"S": -10, "X": -1, "U": 0}, treat_ok=False) for s in base]
    assert run_analysis(bad)[0].startswith("general learning failure")
    # no life-1 acquisition -> invalid
    assert run_analysis([fake_result(s, {"S": -10, "X": -1, "U": 0}, acq=1) for s in base])[0].startswith("INVALID")
    # diverged branch -> invalid
    rs = [fake_result(s, {"S": -10, "X": -1, "U": 0}) for s in base]
    rs[0] = copy.deepcopy(rs[0])
    rs[0]["steps"]["X"] = [400, 401, 400, 400]
    assert run_analysis(rs)[0].startswith("INVALID")
