"""BIND-I1 pure parts (no model): the controlled stream's guarantees and the frozen analysis decision rules."""
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
        # conditions differ only in the four treatment facts of life 2
        for a, b in (("S", "X"), ("S", "U"), ("X", "U")):
            diff = [(L, j) for L in range(i1.N_LIVES) for j in range(i1.FACTS_PER_LIFE)
                    if st["lives"][a][L][j].train_texts != st["lives"][b][L][j].train_texts]
            assert diff == [(1, j) for j in range(i1.N_TARGETS)]
        # equal day sizes (equal RNG consumption) in every life
        for L in range(i1.N_LIVES):
            assert len({sum(len(f.train_texts) for f in st["lives"][c][L]) for c in i1.CONDITIONS}) == 1
        # life 1 is binding-sensitive: every target attribute has two entities with different values
        life1 = st["lives"]["S"][0]
        for t in d["T"]:
            fs = [f for f in life1 if f.attr == t]
            assert len(fs) == 2 and len({f.entity for f in fs}) == 2 and len({f.value for f in fs}) == 2
        # A, C and B queries share the T_i template frame (only the entity differs)
        for a, c, b in zip(st["targets"], st["counters"], st["treatment"]["S"]):
            frames = {lg.probes_for(f)["template"].replace(f.entity, "#") for f in (a, c, b)}
            assert len(frames) == 1
        assert [i1.treat_index(c) for c in i1.CONDITIONS] == [2, 2, 3]


def fake_result(seed, dDA, treat_ok=True, acq=4, bind=4):
    """Minimal I1-shaped result: D_A(1) = 10 for all targets, D_A(5) = 10 + dDA[cond] (W rises, V fixed, Z fixed)."""
    def row(i, DA, fc=True, bound=True, treat_ok=True):
        tr = {"fc": treat_ok, "lp": -3.0, "margin_best": 4.0 if treat_ok else -4.0}
        return {"i": i, "fid": f"t{i}", "fc": fc, "lV_A": -5.0, "lZ_A": -12.0, "lW_A": -5.0 - DA, "lU_A": -20.0,
                "margin_best": min(DA, 7.0),
                "counter": {"fc": bound, "lZ_C": -4.0 if bound else -9.0, "lV_C": -8.0},
                "lV_B": -15.0, "lZ_B": -15.0, "lW_B": -4.0, "lU_B": -20.0, "treat": {c: tr for c in "SXU"}}
    cp1 = {"after_life": 1, "targets": [row(i, 10.0, fc=i < acq, bound=i < bind) for i in range(4)],
           "ppl_loss": {"eval": 3.0}}
    conds = {c: [{"after_life": L, "targets": [row(i, 10.0 + d * (L - 1) / 4, treat_ok=treat_ok) for i in range(4)],
                  "ppl_loss": {"eval": 3.0}} for L in (2, 3, 5)] for c, d in dDA.items()}
    tr = [{"steps": 400, "bg_teacher_sha256": "x"}] * 4
    return {"seed": seed, "status": "complete", "commit": "c", "config": {"arm": "L"}, "backbone_sha256": "h",
            "backbone_start": "h", "backbone_end": "h", "base_ppl_loss": {"eval": 3.0},
            "wiped": {"targets": [row(i, -1.0) for i in range(4)]}, "checkpoint1": cp1, "conditions": conds,
            "traces": {c: copy.deepcopy(tr) for c in "SXU"}, "rng_after": {c: "r" for c in "SXU"}}


def run_analysis(rs):
    seeds = {r["seed"]: an.per_seed(r) for r in rs}
    eff = {f"{a}-{b}": an.effect(seeds, a, b) for a, b in (("S", "X"), ("S", "U"), ("X", "U"))}
    v = an.validity(rs, seeds)
    return an.interpret(eff, seeds, v), eff, v, seeds


SEEDS = [2, 3, 4, 5]


def test_decision_rules():
    interp, eff, v, seeds = run_analysis([fake_result(s, {"S": -12, "X": -1, "U": 0}) for s in SEEDS])
    assert v["V1"] and v["V2"] and v["V3"] and v["V3b"] and all(v["V4"].values()) and v["V5"]
    assert abs(eff["S-X"]["mean"] - (-11)) < 1e-6 and interp.startswith("1 ") and "within band" in interp
    assert seeds[2]["conditions"]["S"]["patterns5"] == {"intrusion_with_binding": 4}   # W > V > Z
    assert seeds[2]["conditions"]["U"]["patterns5"] == {"survives": 4}
    assert run_analysis([fake_result(s, {"S": -10, "X": -9, "U": 0}) for s in SEEDS])[0].startswith("2 ")
    i3 = run_analysis([fake_result(s, {"S": -5, "X": -5.5, "U": -4.5}) for s in SEEDS])[0]
    assert i3.startswith("3 ") and "general drift" in i3
    i3b = run_analysis([fake_result(s, {"S": -0.5, "X": 0, "U": 0}) for s in SEEDS])[0]
    assert i3b.startswith("3 ") and "no detectable impairment" in i3b
    assert run_analysis([fake_result(s, {"S": 0, "X": -10, "U": 0}) for s in SEEDS])[0].startswith("4 ")


def test_split_signs_large_mean_is_not_small_or_worse():
    # Sol's example: per-seed S-X = [-18, -15, +2, +3] -> mean -7, neither robust-worse nor small
    rs = [fake_result(2, {"S": -18, "X": 0, "U": 0}), fake_result(3, {"S": -15, "X": 0, "U": 0}),
          fake_result(4, {"S": 2, "X": 0, "U": 0}), fake_result(5, {"S": 3, "X": 0, "U": 0})]
    interp, eff, _, _ = run_analysis(rs)
    assert eff["S-X"]["mean"] == -7.0 and not eff["S-X"]["robust_worse"] and not eff["S-X"]["small"]
    assert interp.startswith("3 ")                                       # no robust difference -> not established


def test_gates():
    # treatment uptake failure -> inconclusive (symmetric across conditions)
    assert run_analysis([fake_result(s, {"S": -10, "X": -1, "U": 0}, treat_ok=False) for s in SEEDS])[0] \
        .startswith("INCONCLUSIVE DUE TO INADEQUATE TREATMENT UPTAKE")
    # no life-1 acquisition -> invalid
    assert run_analysis([fake_result(s, {"S": -10, "X": -1, "U": 0}, acq=1) for s in SEEDS])[0].startswith("INVALID")
    # acquisition without bidirectional binding -> verdict qualified
    i = run_analysis([fake_result(s, {"S": -10, "X": -1, "U": 0}, bind=1) for s in SEEDS])[0]
    assert i.startswith("1 ") and "INITIAL BINDING NOT ESTABLISHED" in i
    # diverged background/teacher sequence -> invalid
    rs = [fake_result(s, {"S": -10, "X": -1, "U": 0}) for s in SEEDS]
    rs[0]["traces"]["X"][1] = {"steps": 400, "bg_teacher_sha256": "different"}
    assert run_analysis(rs)[0].startswith("INVALID")
