"""BIND-D1 pure parts (no model): frozen pair selection and the tokenizer-prefix check."""
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


def test_pairs_frozen_from_stream_with_cohorts():
    lg = load("lifegen_t", "being/v3/lifegen.py")
    d1 = load("bind_d1_t", "scripts/bind_d1.py")
    st = lg.make_stream(0, n_lives=5)
    pairs, same = d1.pairs_for(st, 5)
    assert pairs and same >= 0
    seen = set()
    for a, b, cohort in pairs:
        assert a.attr == b.attr and a.entity != b.entity and a.value != b.value
        assert (a.life, a.fid) < (b.life, b.fid)                       # oriented by the stream: A is earlier
        assert cohort == ("primary" if a.life == 1 and b.life > 1 else "simultaneous" if a.life == 1 else "secondary")
        assert (a.fid, b.fid) not in seen and (b.fid, a.fid) not in seen  # no reversed duplicates
        seen.add((a.fid, b.fid))
    assert {"primary", "simultaneous", "secondary"} <= {c for _, _, c in pairs}
    again, _ = d1.pairs_for(lg.make_stream(0, n_lives=5), 5)
    assert [(a.fid, b.fid, c) for a, b, c in again] == [(a.fid, b.fid, c) for a, b, c in pairs]


class _Tok:
    """Whitespace tokenizer with one deliberate boundary merge: 'is a' + ' xy' -> merged token when value starts 'xy'."""
    def __call__(self, text):
        toks, out = text.split(" "), []
        for t in toks:
            if out and out[-1] == "a" and t.startswith("xy"):
                out[-1] = "a_" + t
            else:
                out.append(t)
        toks = out
        return type("E", (), {"input_ids": toks})()


def test_candidate_meta_flags_prefix_breaks():
    d1 = load("bind_d1_t2", "scripts/bind_d1.py")
    m = d1.candidate_meta(_Tok(), "Ann owns a", ["bell", "xylo pipe"])
    assert m[0] == {"ntok": 1, "prefix_ok": True}
    assert m[1]["prefix_ok"] is False


def _fake_result(lg, d1, seed=0, lives=5):
    """A D1-shaped result built from the real stream with synthetic scores. For each life-1 fact the strongest rival
    is a value taught for the same attribute in a LATER life (if one exists), so exposure timing can be checked."""
    st = lg.make_stream(seed, n_lives=lives)
    pairs, same = d1.pairs_for(st, lives)
    later = {}
    for life in st.lives[1:lives]:
        for f in life:
            later.setdefault(f.attr, f)
    def rows(L):
        out = []
        for f in st.lives[0]:
            pool = [v for v in st.pools[f.attr] if v != f.value][:49]
            lt = later.get(f.attr)
            if lt and lt.value in pool:
                pool.remove(lt.value)
                pool = [lt.value] + pool[:48]
            cands = [{"v": f.value, "lp": -10.0, "ntok": 3, "prefix_ok": True}] + \
                    [{"v": v, "lp": (-5.0 if i == 0 else -30.0 - i), "ntok": 1 + i % 6, "prefix_ok": True}
                     for i, v in enumerate(pool)]
            out.append({"fid": f.fid, "attr": f.attr, "answer": f.value, "template": True, "fc": False, "fc_ties": 0,
                        "margin_best": -5.0, "logprob": -10.0, "cands": cands, "prefix_fail": 0})
        return out
    paired = [{"a_fid": a.fid, "a_life": a.life, "b_fid": b.fid, "b_life": b.life, "attr": a.attr, "cohort": c,
               "V": a.value, "W": b.value, "lV_A": -1.0, "lW_A": -3.0, "lV_B": -4.0, "lW_B": -1.5} for a, b, c in pairs]
    return {"probe_seed": seed, "wiped": {"life1": rows(0), "paired": paired},
            "checkpoints": [{"after_life": L, "life1": rows(L), "paired": paired} for L in (1, 2, 3, 5)],
            "pairs_identical_value_excluded": same}, later


def test_analysis_no_future_leakage_and_contrasts():
    lg = load("lifegen_t3", "being/v3/lifegen.py")
    d1 = load("bind_d1_t3", "scripts/bind_d1.py")
    an = load("bind_d1_analyze_t", "scripts/bind_d1_analyze.py")
    r, later = _fake_result(lg, d1)
    out = an.analyse_seed(r, 5)
    for L, c in out["checkpoints"].items():
        for w in c["rival"]["winners"]:
            ex = w["winner_exposure"]
            if ex is not None:
                assert ex["latest"] <= L and ex["first"] <= L        # never classified by a future life
        q = c["acquisition"]
        assert q["shared_n"] + q["unique_n"] == q["all_n"] == 10
    lives_of_winner = {w["fid"]: later[w["attr"]].life for w in out["checkpoints"][5]["rival"]["winners"]
                       if w["attr"] in later and w["winner"] == later[w["attr"]].value}
    assert lives_of_winner                                            # the synthetic winners exist
    for fid, life in lives_of_winner.items():
        w2 = next(w for w in out["checkpoints"][2]["rival"]["winners"] if w["fid"] == fid)
        w5 = next(w for w in out["checkpoints"][5]["rival"]["winners"] if w["fid"] == fid)
        assert w2["winner_taught"] == (life <= 2) and w5["winner_taught"] == (life <= 5)
    c = out["checkpoints"][5]["contrasts"]
    k = next(k for k in c if k.endswith("|post"))
    assert c[k]["DA"] == 2.0 and c[k]["DB"] == 2.5 and c[k]["C"] == 4.5 and c[k]["dC"] == 0.0
    assert any(k.endswith("|pre") for k in out["checkpoints"][1]["contrasts"])   # later-B pairs are pre at life 1
