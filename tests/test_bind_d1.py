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


def test_pairs_are_same_attribute_other_entity():
    lg = load("lifegen_t", "being/v3/lifegen.py")
    d1 = load("bind_d1_t", "scripts/bind_d1.py")
    st = lg.make_stream(0, n_lives=5)
    pairs = d1.pairs_for(st, 5)
    assert set(pairs) == {f.fid for f in st.lives[0]}
    by = {f.fid: f for f in st.lives[0]}
    n = 0
    for fid, bs in pairs.items():
        for b in bs:
            assert b.attr == by[fid].attr and b.entity != by[fid].entity and 1 <= b.life <= 5
            n += 1
    assert n > 0
    # deterministic: the same stream gives the same frozen pairs
    assert {k: [b.fid for b in v] for k, v in d1.pairs_for(lg.make_stream(0, n_lives=5), 5).items()} == \
           {k: [b.fid for b in v] for k, v in pairs.items()}


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
