"""CLS-0C Phase B-strict: the only change is the VERIFY question line; the B1 logic equals Phase B's run_life; the
promotion rule is the frozen one; the baseline commitment count reproduces the reported 89/110 answered-stored figure."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BS = _load("cls0c_bs_t", "scripts/cls0c_bs.py")
B = BS.B


def test_only_the_question_line_changes():
    assert BS.VERIFY_STRICT.replace(BS.QS, B._Q) == B.VERIFY
    assert BS.VERIFY_STRICT.count(BS.QS) == 4 and B._Q not in BS.VERIFY_STRICT
    assert "Answer: yes" in BS.VERIFY_STRICT and BS.VERIFY_STRICT.count("Answer: no") == 2       # example answers kept
    p = BS.VERIFY_STRICT.format(mem=B.MEM.format(t=1, text="Ann works as a Plover."), cue="Ann works as a")
    assert p.endswith('Statement to complete: "Ann works as a ..."\n' + BS.QS) and "{" not in p
    md = (ROOT / "lab/CLS0C_B_STRICT_PROTOCOL.md").read_text()
    assert " ".join(BS.QS.split("\n")[0].split()) in " ".join(md.replace("`", "").split())          # frozen text


def test_b1_answer_matches_phase_b_run_life():
    texts = {"a": "Ann works as a Plover.", "b": "Ann lives in Zorn.", "c": "Ann works as a Heron."}
    times = {"a": 1, "b": 1, "c": 3}
    margins = {"a": 2.0, "b": -1.0, "c": 0.5}
    ext = {"a": "Plover", "c": "Heron"}
    o, ch = BS.b1_answer("Ann works as a", ["a", "b", "c"], lambda c, e: margins[e], lambda c, e: ext[e], texts, times)
    assert ch == "c" and o["answer"] == "Heron" and o["n_supported"] == 2                      # latest supported
    o, ch = BS.b1_answer("At life 1, Ann works as a", ["a", "b", "c"], lambda c, e: margins[e],
                         lambda c, e: ext[e], texts, times)
    assert ch == "a" and o["answer"] == "Plover"
    o, ch = BS.b1_answer("Ann has a pet", ["b"], lambda c, e: margins[e], lambda c, e: "", texts, times)
    assert ch is None and o["answer"] is None


def test_promotion_rule_is_the_frozen_one():
    good = {"template": 1.0, "para_entailed": 0.98, "near_unsupported": 0.025, "unseen_unsupported": 0.0,
            "temporal": 1.0, "provenance": 1.0}
    ne = {"rate": 33 / 110}
    assert BS.promote(good, ne)[0]
    assert not BS.promote(good, {"rate": 34 / 110})[0]
    for k, v in (("template", 0.985), ("para_entailed", 0.969), ("near_unsupported", 0.0625),
                 ("unseen_unsupported", 0.0125), ("temporal", 11 / 12), ("provenance", 0.94)):
        assert not BS.promote(dict(good, **{k: v}), ne)[0], k


def test_baseline_commitments_from_committed_phase_b():
    eps = {}
    rows = []
    for sd in (2, 3, 4, 5):
        d = json.loads((ROOT / f"results/cls0b_final/seed{sd}.json").read_text())
        eps.update({(sd, e["eid"]): e["attr"] for e in d["episodes"]})
        pb = json.loads((ROOT / f"results/cls0c_b/seed{sd}.json").read_text())
        rows += [dict(r, key=(r["key"], sd)) for r in pb["lives"][-1]["methods"]["B1_K1"]["rows"]]
    form = lambda ks: (eps[(ks[1], ks[0].split("|")[1])], int(ks[0].split("|")[2][-1]))
    ne = BS.not_entailed_commitments(rows, form)
    assert ne["n"] == 110 and ne["committed"] >= 89                       # 89 answered with the stored value
    assert sum(v["n"] for v in ne["per_form"].values()) == 110 and len(ne["per_form"]) == 4
