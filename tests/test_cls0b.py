"""CLS-0B pure parts (no model): episodes, cues, lexical keys, null threshold, routing, context."""
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


lg = load("lifegen_c0b", "being/v3/lifegen.py")
i1 = load("bind_i1_c0b", "scripts/bind_i1.py")
cb = load("cls0b_t", "scripts/cls0b.py")
NTOK = lambda w: 1 + len(w) // 4


def test_episodes_and_cues():
    for seed in (2, 3, 4, 5):
        eps, st = cb.build_episodes(seed, lg, i1, NTOK)
        assert len(eps) == 52 and len({e["eid"] for e in eps}) == 52
        assert [e["time"] for e in eps] == sorted(e["time"] for e in eps)              # chronological
        kinds = [e["kind"] for e in eps]
        assert kinds.count("target") == kinds.count("counter") == kinds.count("treatment") == 4
        assert kinds.count("temporal") == 2 and kinds.count("filler") == 38
        for e in eps:                                                                   # text = observed phrasings
            if e["kind"] != "temporal":
                assert e["value"] in e["text"] and e["entity"] in e["text"]
        tq = [e for e in eps if e["kind"] == "temporal"]
        assert tq[0]["entity"] == tq[1]["entity"] and tq[0]["attr"] == tq[1]["attr"] and tq[0]["time"] == 1 \
            and tq[1]["time"] == 3 and tq[0]["value"] != tq[1]["value"]
        cues, cal, neg, temporal = cb.build_cues(seed, eps, lg)
        assert len(cues) == 150 and all(len(c["pool"]) == 50 and len(set(c["pool"])) == 50 for c in cues)
        assert all(c["pool"][0] == c["value"] for c in cues)
        import re
        stream_words = {w.lower() for e in eps for w in re.findall(r"[A-Za-z]+", e["text"])}
        for c in cal + [n for n in neg if n["form"] == "unseen_person"]:                # never-observed people
            assert c["entity"] in c["cue"] and not ({w.lower() for w in c["entity"].split()} & stream_words)
        assert not {c["cue"] for c in cal} & {n["cue"] for n in neg}                    # calibration != test
        nm = [n for n in neg if n["form"] == "near_miss"]
        assert len(cal) == 40 and len(neg) == 40 and len(nm) == 20
        for n in nm:                                                                    # attribute never observed
            assert not any(e["entity"] == n["entity"] and cb.template(lg, e["attr"], e["entity"]) == n["cue"]
                           for e in eps)
        assert [t["form"] for t in temporal] == ["life1", "life3", "latest"]
        assert temporal[0]["expect"] == tq[0]["value"] and temporal[2]["expect"] == tq[1]["value"]
        again = cb.build_cues(seed, cb.build_episodes(seed, lg, i1, NTOK)[0], lg)
        assert again[1] == cal and again[2] == neg                                     # deterministic


def test_tfidf_threshold_route_context():
    s = cb.tfidf_sims(["Ann works as a smith.", "Bob lives in Vale."], ["Ann works as a", "Who lives in Vale"])
    assert s[0][0] > s[0][1] and s[1][1] > s[1][0]
    cal = [0.1 * i for i in range(10)]                    # 0.0..0.9: 10% allowance -> exactly 1 may activate
    thr = cb.threshold(cal, 0.10)
    assert sum(x >= thr for x in cal) == 1 and thr > 0.8
    cal40 = [i / 40 for i in range(40)]                   # 40 negatives, 10% -> exactly 4 activate
    assert sum(x >= cb.threshold(cal40, 0.10) for x in cal40) == 4
    assert cb.route([0.2, 0.95, 0.5], ["a", "b", "c"], thr) == [("b", 0.95)]
    assert cb.route([0.2, 0.5, 0.4], ["a", "b", "c"], thr) == []
    assert cb.route([0.99, 0.95, 0.5], ["a", "b", "c"], thr, k=2) == [("a", 0.99), ("b", 0.95)]
    assert cb.context([(3, "later."), (1, "first.")]) == "Memory from life 1: first.\nMemory from life 3: later.\n"


ca = load("cls0b_an_t", "scripts/cls0b_analyze.py")


def fake(seed, route=1.0, irag=1.0, raw=0.5, para=1.0, falseact=0.0, cover=1.0, temporal=True, k4=0.0, n=40):
    eps = [{"eid": f"e{i}", "time": 1 + i % 5} for i in range(n)]
    writes = {e["eid"]: {"ok": i < cover * n} for i, e in enumerate(eps)}

    def facts(rr, ir, rw, pa):
        rows = []
        for i, e in enumerate(eps):
            for form, acc in (("template", ir), ("paraphrase0", pa)):
                ok = i < rr * n
                rows.append({"eid": e["eid"], "form": form, "kind": "filler", "time": e["time"], "routed": e["eid"] if ok else None,
                             "route_ok": ok, "irag": {"correct": i < acc * n}, "raw": {"correct": i < (rw if form == "template" else pa) * n}})
        return rows
    neg = [{"form": "unseen_person", "activated": i < falseact * 20} for i in range(20)]
    tm = [{"form": f, "correct": temporal, "routed": []} for f in ("life1", "life3", "latest")]
    meth = {m: {"threshold": 0.5, "facts": facts(k4 if m == "K4" else route, irag, raw, para), "negatives": neg, "temporal": tm}
            for m in ca.METHODS}
    cps = [{"after_life": L, "experienced": n, "stored": n, "references": [], "methods": meth} for L in range(1, 6)]
    h = "x"
    return {"seed": seed, "status": "complete", "test_mode": False, "design_check": "match", "config": dict(ca.FROZEN_CFG),
            "commit": "c", "model_manifest": {"revision": "r", "files": {"a": h}}, "prefix_failures": 0, "episodes": eps, "writes": writes,
            "reconstruction_exact": {e["eid"]: True for e in eps}, "checkpoints": cps}


def test_decision_rules():
    S = [2, 3, 4, 5]
    d = ca.decide([fake(s) for s in S])
    assert d["direction"].startswith("5 ") and not d["qualifiers"]
    assert ca.decide([fake(s, cover=0.9) for s in S])["direction"].startswith("1 ")
    assert ca.decide([fake(s, route=0.85) for s in S])["direction"].startswith("2 ")
    assert ca.decide([fake(s, falseact=0.2) for s in S])["direction"].startswith("2 ")
    assert ca.decide([fake(s, irag=0.7, raw=0.5) for s in S])["direction"].startswith("3 ")
    assert ca.decide([fake(s, irag=0.85, raw=0.99) for s in S])["direction"].startswith("4 ")
    d = ca.decide([fake(s, raw=1.0) for s in S])                         # raw == iRAG is expected, not a direction
    assert d["direction"].startswith("5 ") and any("reproduces the usable information" in n for n in d["notes"])
    q = ca.decide([fake(s, para=0.7) for s in S])["qualifiers"]
    assert any("paraphrase" in x for x in q)
    assert any("temporal" in x for x in ca.decide([fake(s, temporal=False) for s in S])["qualifiers"])
    assert any("permuted" in x for x in ca.decide([fake(s, k4=0.9) for s in S])["qualifiers"])
    v = ca.validity([fake(s) for s in S])
    assert v["valid"]
    bad = [fake(s) for s in S]
    bad[0]["config"]["init_mask"] = 45
    assert not ca.validity(bad)["valid"]


def test_chronology_k4_and_temporal():
    keys = {f"e{i}": i for i in range(6)}
    early = cb.permuted_keys(2, 1, ["e0", "e1", "e2"], keys)
    assert set(early) == {"e0", "e1", "e2"} and set(early.values()) <= {0, 1, 2}     # never a future memory's key
    assert early == cb.permuted_keys(2, 1, ["e0", "e1", "e2"], keys)                 # deterministic per seed/life
    eps, _ = cb.build_episodes(2, lg, i1, NTOK)
    *_, temporal = cb.build_cues(2, eps, lg)
    q1, q3 = temporal[0]["expect"], temporal[1]["expect"]
    for L in (1, 2):
        items = cb.temporal_items(temporal, L)
        assert [t["form"] for t in items] == ["life1", "latest_pre"] and all(t["expect"] == q1 for t in items)
        assert all(q3 not in t["cue"] for t in items)
    assert [t["expect"] for t in cb.temporal_items(temporal, 3)] == [q1, q3, q3]
