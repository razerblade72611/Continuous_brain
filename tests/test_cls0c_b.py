"""CLS-0C Phase B: pure parts + the full per-checkpoint pipeline on real committed CLS-0B data with fake readers."""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from being.v3 import lifegen  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B = _load("cls0c_b_t", "scripts/cls0c_b.py")
C0B = _load("cls0b_t2", "scripts/cls0b.py")
CA = _load("cls0c_a_t2", "scripts/cls0c_a.py")


def test_audit_matches_frozen_document_and_covers_every_paraphrase():
    md = (ROOT / "lab/CLS0C_PARAPHRASE_AUDIT.md").read_text()
    rows = re.findall(r"^\| (.+?) \| p(\d) \"(.+?)\" \| (ENTAILED|BORDERLINE|NOT ENTAILED) \|", md, re.M)
    assert len(rows) == 20
    for attr, i, text, label in rows:
        assert lifegen.PARAPHRASE[attr][int(i)] == text
        assert B.AUDIT[(attr, int(i))] == label.lower().replace(" ", "_")
    assert set(B.AUDIT) == {(a, i) for a in lifegen.PARAPHRASE for i in range(len(lifegen.PARAPHRASE[a]))}
    labels = list(B.AUDIT.values())
    assert (labels.count("entailed"), labels.count("borderline"), labels.count("not_entailed")) == (14, 2, 4)


def test_text_helpers():
    assert B.norm("  The Talsheiskpal. ") == "talsheiskpal" and B.norm("a Glimmer-stone") == "glimmer-stone"
    assert B.parse_fields("Mara Lind | plays instrument | cello\nmore", 3) == ["Mara Lind", "plays instrument", "cello"]
    assert B.parse_fields("Mara Lind | cello", 3) is None and B.parse_fields(" | x | y", 3) is None
    assert B.cut_value(" Talsheiskpal. Pyvnous Braen's allergy") == "Talsheiskpal"
    assert B.cut_value(' "Zorbel"\nnext') == "Zorbel"
    t = "Pyvnous Braen is allergic to Talsheiskpal. Pyvnous Braen's allergy is to Talsheiskpal."
    assert B.evidence(t, "Talsheiskpal") == "Pyvnous Braen is allergic to Talsheiskpal."
    assert B.evidence(t, "Talsheis") is None and B.evidence(t, "") is None          # whole words only
    assert B.evidence("Kai's favorite object is a lamp.", "a lamp") == "Kai's favorite object is a lamp."
    assert B.time_constraint("At life 3, Ann works as a") == 3 and B.time_constraint("Ann works as a") is None


def test_temporal_selection_and_contract():
    c = [{"eid": "a", "time": 1, "margin": 5.0}, {"eid": "b", "time": 3, "margin": 1.0}]
    assert B.select(c, None)["eid"] == "b" and B.select(c, 1)["eid"] == "a" and B.select(c, 2) is None
    assert B.select([], None) is None
    texts = {"e": "Ann works as a Plover. Ann's job is Plover."}
    assert B.commit("e", "Plover", texts)["span"] == "Ann works as a Plover."
    assert B.commit("e", "Heron", texts)["answer"] is None                          # not in the episode: fail closed
    assert B.commit(None, None, texts)["answer"] is None
    o = B.commit("e", "plover", texts)
    assert B.grade(o, "Plover", {"e"}, texts)["correct"]
    assert not B.grade(o, "Plover", {"other"}, {**texts, "other": "x"})["correct"]  # wrong provenance
    assert not B.grade({"answer": None, "eid": None, "span": None}, "Plover", {"e"}, texts)["answered"]


def test_summary_gates_and_frozen_choice():
    rows = ([{"cat": "template", "answered": True, "correct": True, "provenance_ok": True}] * 20 +
            [{"cat": "para_entailed", "answered": True, "correct": True, "provenance_ok": True}] * 10 +
            [{"cat": "near_miss", "answered": False, "correct": False, "provenance_ok": False}] * 10 +
            [{"cat": "unseen", "answered": False, "correct": False, "provenance_ok": False}] * 10)
    s = B.summarize(rows)
    assert s["passes"] and s["errors"] == 0
    bad = B.summarize(rows + [{"cat": "near_miss", "answered": True, "correct": False, "provenance_ok": False}] * 3)
    assert not bad["passes"] and bad["errors"] == 3
    P = dict(s, errors=4)
    F = dict(bad)
    assert B.choose({"B1_K1": P, "B2": P, "B2e": F}, {"B1_K1": 100, "B2": 60, "B2e": 1}, "B1_K1")[0] == "B1_K1"
    assert B.choose({"B1_K1": P, "B2": P, "B2e": F}, {"B1_K1": 100, "B2": 50, "B2e": 1}, "B1_K1")[0] == "B2"
    assert B.choose({"B1_K1": P, "B2": dict(P, errors=2), "B2e": F}, {"B1_K1": 1, "B2": 9, "B2e": 1},
                    "B1_K1")[0] == "B2"
    assert B.choose({"B1_K1K3": F, "B2": F, "B2e": P}, {"B1_K1K3": 1, "B2": 1, "B2e": 1}, "B1_K1K3")[0] == "B2e"
    assert B.choose({"B1_K1": F, "B2": F, "B2e": F}, {"B1_K1": 1, "B2": 1, "B2e": 1}, "B1_K1")[0] is None


def test_run_life_with_perfect_readers_is_perfect_and_b0_commits_without_verification():
    d = json.loads((ROOT / "results/cls0b_final/seed2.json").read_text())
    eps = [{k: v for k, v in e.items() if k != "text_sha256"} for e in d["episodes"]]
    cues, cal, neg, temporal = C0B.build_cues(2, eps, lifegen)
    attrs = sorted({e["attr"] for e in eps if e["kind"] != "temporal"})
    tmpl = lambda a, e: C0B.template(lifegen, a, e)
    info = {e["eid"]: e for e in eps}
    texts = {e["eid"]: e["text"] for e in eps}
    times = {e["eid"]: e["time"] for e in eps}
    attr_of = {e["eid"]: e["attr"] for e in eps}
    # evaluator-side truth about what each cue asks (fake readers may use it; the real ones never do)
    asks = {}
    for c in cues:
        asks[c["cue"]] = (info[c["eid"]]["entity"], info[c["eid"]]["attr"])
    for c in neg:
        asks[c["cue"]] = (c["entity"], CA.cue_attr(c["cue"], c["entity"], attrs, tmpl))
    tq = [e for e in eps if e["kind"] == "temporal"]
    for t in temporal:
        asks[t["cue"]] = (tq[0]["entity"], tq[0]["attr"])
    L = 5
    bank = [e["eid"] for e in eps if e["time"] <= L]
    supports = lambda cue, e: (info[e]["entity"], info[e]["attr"]) == asks[cue] and \
        (B.time_constraint(cue) is None or B.time_constraint(cue) == info[e]["time"])
    rank = lambda cue: sorted(bank, key=lambda e: (-(info[e]["entity"] == asks[cue][0]) - supports(cue, e), e))
    k1_route = lambda cue: rank(cue)[0] if info[rank(cue)[0]]["entity"] == asks[cue][0] else None   # entity-driven
    items = []
    for c in cues:
        items.append({"key": f"F|{c['eid']}|{c['form']}", "cue": c["cue"], "cat": B.cue_category(c, attr_of),
                      "gold_value": c["value"], "gold_eids": {c["eid"]}})
    for j, c in enumerate(neg):
        if c["form"] == "near_miss" and c["observed_since"] > L:
            continue
        items.append({"key": f"N|{j}", "cue": c["cue"], "cat": B.cue_category(c, attr_of)})
    tv = {e["value"]: e["eid"] for e in tq}
    for c in C0B.temporal_items(temporal, L):
        items.append({"key": f"T|{c['form']}", "cue": c["cue"], "cat": "temporal", "gold_value": c["expect"],
                      "gold_eids": {tv[c["expect"]]}})
    recs = {m: {e: (info[e]["entity"], info[e]["attr"], info[e]["value"]) for e in bank} for m in ("B2", "B2c")}
    qp = {m: {it["cue"]: asks[it["cue"]] for it in items} for m in ("B2", "B2c")}
    outs, inf = B.run_life(L, bank, items, texts, times, rank, rank, k1_route,
                           lambda c, e: 1.0 if supports(c, e) else -1.0, lambda c, e: info[e]["value"], recs, qp,
                           lambda x, y: 1.0 if x == y else 0.0)
    for m in ("B1_K1", "B1_K1K3", "B2", "B2e", "B2c"):
        rows = []
        for it in items:
            o = outs[m][it["key"]]
            g = B.grade(o, it["gold_value"], it["gold_eids"], texts) if "gold_value" in it else \
                {"answered": o["answer"] is not None, "correct": False, "provenance_ok": False}
            rows.append({"cat": it["cat"], **g})
        s = B.summarize(rows)
        assert s["template"] == 1.0 and s["para_entailed"] == 1.0 and s["temporal"] == 1.0, (m, s)
        assert s["near_unsupported"] == 0.0 and s["unseen_unsupported"] == 0.0 and s["passes"], (m, s)
    # B0 commits on entity similarity alone: every near-miss gets an (unsupported) answer
    nm = [it for it in items if it["cat"] == "near_miss"]
    assert nm and all(outs["B0"][it["key"]]["answer"] is not None for it in nm)


def test_every_prompt_formats_with_exactly_its_fields():
    mem = B.MEM.format(t=3, text="Ann works as a Plover.")
    v = B.VERIFY.format(mem=mem, cue="Ann works as a")
    assert v.endswith('Memory (from life 3): Ann works as a Plover.\nStatement to complete: "Ann works as a ..."\n'
                      "Does the memory above state exactly what completes this statement? Answer yes or no.\nAnswer:")
    assert v.count("Answer: yes") == 1 and v.count("Answer: no") == 2
    x = B.EXTRACT_VALUE.format(mem=mem, cue="Ann works as a")
    assert x.endswith("Memory (from life 3): Ann works as a Plover.\nUsing only the memory above, complete the "
                      "statement with the exact words from the memory.\nAnn works as a")
    r = B.EXTRACT_RECORD.format(text="Ann works as a Plover.", schema="")
    assert r.endswith("Text: Ann works as a Plover.\nWrite the one fact stated in the text in the form: "
                      "subject | relation | value\nFact:")
    q = B.PARSE_QUERY.format(cue="Ann works as a", schema=B.SCHEMA.format(rels="occupation, city"))
    assert q.endswith('"Ann works as a ..."\nWrite what the statement asks for in the form: subject | relation\n'
                      "Use one of these relations: occupation, city\nAsked:")
    for p_ in (v, x, r, q):
        assert "{" not in p_ and "}" not in p_
    # no workload content in the frozen few-shot examples
    words = {w.lower() for a in lifegen.TRAIN for t_ in lifegen.TRAIN[a] for w in re.findall(r"[A-Za-z]+", t_)}
    shots = B.VERIFY_SHOTS + B.EXTRACT_VALUE.split("{mem}")[0]
    assert "cello" not in words and "mara" not in words and "lind" not in words and "cello" in shots


def test_choose_zero_error_tie_keeps_b1_and_genuine_halving_switches():
    """Sol: with B1 at zero errors, 0 <= 0.5 * 0 must NOT let a B2 variant displace B1 on a tie."""
    rows = ([{"cat": "template", "answered": True, "correct": True, "provenance_ok": True}] * 20 +
            [{"cat": "para_entailed", "answered": True, "correct": True, "provenance_ok": True}] * 10 +
            [{"cat": "near_miss", "answered": False, "correct": False, "provenance_ok": False}] * 10 +
            [{"cat": "unseen", "answered": False, "correct": False, "provenance_ok": False}] * 10)
    P0 = B.summarize(rows)
    assert P0["passes"] and P0["errors"] == 0
    F = dict(P0, passes=False)
    secs = {"B1_K1": 100, "B2": 80, "B2e": 80}
    assert B.choose({"B1_K1": P0, "B2": P0, "B2e": P0}, secs, "B1_K1") == ("B1_K1", "B1 passes (preferred on a tie)")
    assert B.choose({"B1_K1": dict(P0, errors=6), "B2": dict(P0, errors=3), "B2e": F}, secs, "B1_K1")[0] == "B2"
    assert B.choose({"B1_K1": dict(P0, errors=6), "B2": dict(P0, errors=4), "B2e": F}, secs, "B1_K1")[0] == "B1_K1"
    assert B.choose({"B1_K1": P0, "B2": P0, "B2e": F}, {"B1_K1": 100, "B2": 50, "B2e": 1}, "B1_K1")[0] == "B2"
