"""Tests for scripts/abst0.py (lab/ABST0_PROTOCOL.md)."""
import importlib.util, json, math, random, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("abst0", ROOT / "scripts/abst0.py")
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
from being.v3 import lifegen as LG


def test_streams_deterministic_and_well_formed():
    for s in (12, 13, 14, 15, 16):
        a, b = X.make_stream(s, LG), X.make_stream(s, LG)
        assert a == b and X.check_stream(a)
        lives = [sum(f["life"] == L for f in a["facts"]) for L in range(1, 6)]
        assert lives == [10, 10, 10, 13, 10], lives
        tr = {tuple(x) for x in a["transitions"]}
        assert sum(e in a["A"] for e, _ in tr) == 4 and sum(e in a["H"] for e, _ in tr) == 4
        assert sum(a["intro"][e] == 1 for e, _ in tr) == 4                       # 2 A + 2 H on life-1 entities
        for L in (1, 2, 3):
            assert not (tr & set(X.known(a, L)))                                    # transitions unknown before life 4
        assert tr <= set(X.known(a, 4))
        for e, at in a["updates"]:
            assert X.known(a, 3)[(e, at)] != X.known(a, 4)[(e, at)]                  # the update changes the value
    assert X.make_stream(12, LG) != X.make_stream(13, LG)


def test_memory_index_exact_and_complete():
    st = X.make_stream(12, LG)
    e = st["A"][0]
    m1 = X.memory(st, e, 1)
    assert all(f["text"] in m1 for f in st["facts"] if f["entity"] == e and f["life"] <= 1)
    assert not any(f["text"] in m1 for f in st["facts"] if f["entity"] == e and f["life"] > 1)
    assert X.memory(st, e, 5).count(e) >= X.memory(st, e, 1).count(e)


def test_probes_categories():
    st = X.make_stream(12, LG)
    upd = {f"{e}|{a}" for e, a in st["updates"]}
    for L in range(1, 6):
        P = X.probes(st, L, LG)
        assert P == X.probes(st, L, LG)
        cats = {p["cat"] for p in P}
        assert {"POS", "RET", "HNM", "ANM", "FRESH"} <= cats
        assert not any(p["key"] in upd for p in P if p["cat"] in ("POS", "RET", "FIRST"))   # updates descriptive only
        kn = X.known(st, L)
        planned = {tuple(x) for x in st["transitions"]} | {tuple(x) for x in st["updates"]}
        for p in P:
            e, a = p["key"].split("|")
            if p["cat"] in ("HNM", "ANM"):
                assert (e, a) not in kn and (e, a) not in planned and st["intro"][e] <= L
                assert e in (st["H"] if p["cat"] == "HNM" else st["A"])
            if p["cat"] == "FRESH":
                assert e in st["V"]
            if p["cat"] in ("POS", "FIRST", "RET", "TPOST"):
                assert p["value"] == kn[(e, a)]
            assert p["item"].startswith("Statement: ")                              # evaluation construction family
        assert ("TPRE" in cats) == (L <= 3) and ("TPOST" in cats) == (L >= 4)
        assert ("TPRE_UNFAM" in cats) == (L == 1)                                   # life-2 entities unfamiliar at sleep 1
        assert sum(p["cat"] == "HNM" for p in P) <= X.N_HNM


def test_outcome_contract():
    assert X.outcome("Leston", "Leston") == "correct"
    assert X.outcome("UNKNOWN", "Leston") == "unknown" and X.outcome("unknown.", None) == "unknown"
    assert X.outcome("UNKNOWN: Leston", None) == "substantive"
    assert X.outcome("", None) == "empty" and X.outcome("  ", "x") == "empty"
    assert X.outcome("Paris", "Leston") == "substantive"


def test_negative_filters():
    st = X.make_stream(12, LG)
    w = X.C.eval_wordings(LG)
    a, h = st["A"][0], st["H"][0]
    others = [n for n in st["A"] + st["H"] + st["transient"] if n != a]
    assert X.neg_filters(f"What is {a}'s favourite song?", a, others, w) is None
    assert X.neg_filters("What is her favourite song?", a, others, w) == "no_target"
    assert X.neg_filters(f"Does {a} know {h.split()[0]}?", a, others, w) == "other_name"   # any H name word rejects
    stem = LG.PARAPHRASE["city"][1].format(e=a)
    assert X.neg_filters(stem + "?", a, others, w) == "leak"
    assert X.parse_q("Q: Where was Ann born?\nQ: no question mark\nQ: Who?\nnoise") == ["Where was Ann born?"]


def test_third_slot_and_assemble():
    nsl = {"A_accepted": [{"q": f"A{i}?", "swap": f"G{i}?"} for i in range(3)],
           "T_accepted": [{"q": f"T{i}?"} for i in range(2)]}
    U, G, log = X.third_slot(nsl, 7, 12, 2)
    assert log["nA"] == 4 and log["nT"] == 3 and len(U) == len(G) == 7
    assert [x["q"] for x in U[4:]] == [x["q"] for x in G[4:]]                         # T half identical
    assert all(g["q"] == u["q"].replace("A", "G") for u, g in zip(U[:4], G[:4]))       # A half name-swapped
    assert sum(x["exposures"] for x in log["A_exposure"]) == 4
    for bad in ({"A_accepted": [], "T_accepted": [{"q": "T0?"}]}, {"A_accepted": nsl["A_accepted"], "T_accepted": []}):
        with pytest.raises(SystemExit):                                          # no silent A->T substitution (Sol)
            X.third_slot(bad, 4, 12, 2)
    assert all(x["src"] == "A" for x in U[:4]) and all(x["src"] == "G" for x in G[:4])
    base = [{"id": f"b{i}"} for i in range(5)]
    sch = X.assemble(base, [{"id": "u"}] * 3, [{"id": "g"}] * 3, [{"id": "p"}] * 3, 12, 1)
    pos = lambda arm, tag: [i for i, x in enumerate(sch[arm]) if x["id"] == tag]
    assert pos("U", "u") == pos("G", "g") == pos("P", "p")                             # identical positions
    assert [x for x in sch["U"] if x["id"].startswith("b")] == [x for x in sch["P"] if x["id"].startswith("b")]


def _M(**over):
    base = {"POS_correct": 0.8, "FIRST_correct": 0.8, "RET_correct": 0.9, "TPOST_correct": 0.7,
            "HNM_substantive": 0.05, "HNM_unknown": 0.95, "FRESH_substantive": 0.0, "FRESH_unknown": 1.0,
            "TPRE_unknown": 0.9, "TPRE_unknown_A": 0.9, "TPRE_unknown_H": 0.9, "TPOST_correct_A": 0.7,
            "TPOST_correct_H": 0.7}
    p = dict(base, HNM_substantive=1.0, HNM_unknown=0.0, FRESH_substantive=1.0, FRESH_unknown=0.0)
    g = dict(base, HNM_substantive=0.6, HNM_unknown=0.4)
    arms = {"U": dict(base, **over.get("U", {})), "P": dict(p, **over.get("P", {})), "G": dict(g, **over.get("G", {}))}
    M = {a: {s: {L: dict(v) for L in range(1, 6)} for s in (12, 13, 14, 15)} for a, v in arms.items()}
    ppl = {a: {s: {L: {"eval": 0.95, "modern": 0.97} for L in range(1, 6)} for s in (12, 13, 14, 15)} for a in arms}
    return M, ppl


def test_criteria_and_verdicts():
    S = [12, 13, 14, 15]
    M, ppl = _M(); d = X.criteria(M, S, ppl)
    assert d["functional"] and d["mechanism"] and d["verdict"] == "ABST0_FUNCTIONAL_AND_GROUNDED"
    M, ppl = _M(G={"HNM_substantive": 0.2}); d = X.criteria(M, S, ppl)          # G also good -> no mechanism claim
    assert d["functional"] and not d["mechanism"] and d["verdict"] == "ABST0_FUNCTIONAL"
    M, ppl = _M(U={"HNM_unknown": 0.80, "HNM_substantive": 0.0}); d = X.criteria(M, S, ppl)   # silence/empty not rejection
    assert not d["D2"] and d["verdict"] == "ABST0_NO_ADVANCE"
    M, ppl = _M(U={"POS_correct": 0.6}, P={"POS_correct": 0.8}); assert not X.criteria(M, S, ppl)["D1"]   # > P - 0.10
    M, ppl = _M(U={"TPOST_correct_A": 0.3}); assert not X.criteria(M, S, ppl)["D3"]              # A-cells must pass
    M, ppl = _M(U={"TPRE_unknown_H": 0.5}); assert not X.criteria(M, S, ppl)["D3"]
    M, ppl = _M(P={"POS_correct": 0.0, "TPOST_correct_A": 0.0, "TPOST_correct_H": 0.0, "RET_correct": 0.0})
    assert X.criteria(M, S, ppl)["functional"]                                   # P == 0.0 is a value, not missing
    M, ppl = _M(); M["P"][13][3]["POS_correct"] = None
    assert not X.criteria(M, S, ppl)["D1"]                                       # a missing measure never passes
    M, ppl = _M(); ppl["U"][14][2]["eval"] = 1.06; assert not X.criteria(M, S, ppl)["D4"]       # per-seed band
    M, ppl = _M()
    for s in S:
        for L in range(1, 6):
            del ppl["U"][s][L]["modern"]
    assert not X.criteria(M, S, ppl)["D4"]                                       # a dropped monitor never passes (Sol)
    M, ppl = _M(); ppl["U"][12][3]["eval"] = float("nan"); assert not X.criteria(M, S, ppl)["D4"]
    M, ppl = _M(); M["U"][15][5]["RET_correct"] = 0.4; M["U"][14][5]["RET_correct"] = 0.4
    assert not X.criteria(M, S, ppl)["D4"]


def test_integrity():
    rows = [{"cat": "POS", "key": "a|b", "kind": "template", "item": "Statement: x ...", "answer": "v", "outcome": "correct"}]
    ev = {"rows": rows, "ppl_ratio": {"eval": 1.0, "modern": 1.0}}
    assert X.integrity(ev, json.loads(json.dumps(ev)))["ok"]
    bad = json.loads(json.dumps(ev)); bad["rows"][0]["answer"] = "w"
    assert not X.integrity(ev, bad)["ok"]
    bad = json.loads(json.dumps(ev)); bad["ppl_ratio"]["eval"] = 1.0002
    assert not X.integrity(ev, bad)["ok"]
    bad = json.loads(json.dumps(ev)); bad["ppl_ratio"]["eval"] = float("nan")
    assert not X.integrity(ev, bad)["ok"]


@pytest.mark.slow
def test_end_to_end_tiny(tmp_path):
    pytest.importorskip("torch")
    import tiny_hf_model
    model_dir = tiny_hf_model.build(tmp_path / "m")
    py = sys.executable
    run = lambda *a: subprocess.run([py, str(ROOT / "scripts/abst0.py"), *map(str, a)], check=True, capture_output=True,
                                    text=True, timeout=3000)
    r = subprocess.run([py, str(ROOT / "scripts/abst0.py"), "prep", "--seed", "12", "--model", str(model_dir), "--out",
                        str(tmp_path / "strict.json"), "--model-manifest-sha", "x"], capture_output=True, text=True, timeout=3000)
    assert r.returncode != 0 and "NEGATIVE CORPUS STOP" in r.stdout + r.stderr       # zero eligible negatives -> STOP
    run("prep", "--seed", 12, "--model", model_dir, "--out", tmp_path / "prep_seed12.json", "--test-mode")
    prep = json.loads((tmp_path / "prep_seed12.json").read_text())
    assert prep["test_mode_corpus_stops"]
    assert (tmp_path / "audit_sample_seed12.json").exists() and set(prep["neg"]) == {"1", "2", "3", "4", "5"}
    r = run("prep", "--seed", 12, "--model", model_dir, "--out", tmp_path / "prep_seed12.json", "--test-mode")
    assert "up to date - skipped" in r.stdout                                    # provenance-bound resume
    # the tiny model writes no usable QA/negatives: inject a plumbing corpus (pure selection logic is tested above)
    st = prep["stream"]
    prep["qa"]["accepted"] = {f["fid"]: [{"q": f"What about {f['entity']}?", "a": f["value"]}] for f in st["facts"]}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    n = lambda q: len(X.C.qa_ids(tok, q, "UNKNOWN")[0])
    swap = {e: next(f"What does {t} eat?" for t in st["T"] if n(f"What does {t} eat?") == n(f"What does {e} eat?"))
            for e in st["A"]}                                                    # token-length-matched, as prep does
    for L in map(str, range(1, 6)):
        prep["neg"][L]["A_accepted"] = [{"entity": e, "q": f"What does {e} eat?", "swap": swap[e]} for e in st["A"]]
        prep["neg"][L]["T_accepted"] = [{"q": f"Where was {t} born?"} for t in st["T"][-3:]]
    (tmp_path / "prep2.json").write_text(json.dumps(prep))
    out = tmp_path / "run"
    r = subprocess.run([py, str(ROOT / "scripts/abst0.py"), "train", "--prep", str(tmp_path / "prep2.json"), "--model",
                        str(model_dir), "--passes", "1", "--out", str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "production training needs" in r.stdout + r.stderr       # no silent production run
    run("train", "--prep", tmp_path / "prep2.json", "--model", model_dir, "--passes", 1, "--out", out, "--test-mode")
    r = run("train", "--prep", tmp_path / "prep2.json", "--model", model_dir, "--passes", 1, "--out", out, "--test-mode")
    assert "hashes verified) - skipped" in r.stdout
    (out / "cortex_G_s3.pt").write_bytes((out / "cortex_G_s3.pt").read_bytes() + b"x")   # a changed checkpoint retrains
    r = run("train", "--prep", tmp_path / "prep2.json", "--model", model_dir, "--passes", 1, "--out", out, "--test-mode")
    assert "retraining from scratch" in r.stdout
    tr = json.loads((out / "train_seed12.json").read_text())
    for L, lg in tr["sleeps"].items():
        assert lg["n_third"] == math.ceil(0.5 * lg["n_new"]) and lg["budget"][0] == 2 * lg["n_new"] + lg["n_third"]
    for arm in ("P", "U", "G"):
        for L in range(1, 6):
            assert (out / f"cortex_{arm}_s{L}.pt").exists()
            assert tr["arms"][arm][str(L)]["items"] == tr["sleeps"][str(L)]["budget"][0]
    assert tr["arms"]["U"]["5"]["ckpt_sha256"] != tr["arms"]["P"]["5"]["ckpt_sha256"]          # arms differ
    ex = json.loads((out / "exposure_seed12.json").read_text())["cells"]
    st2 = json.loads((tmp_path / "prep2.json").read_text())["stream"]
    assert set(ex) == {f"{e}|{a}" for e, a in st2["transitions"] if e in st2["A"]}
    for k, v in ex.items():                                  # report == exactly what the selector delivered, sleeps 1-3
        e = k.split("|")[0]
        exp_n = sum(x["exposures"] for L in ("1", "2", "3") for x in tr["sleeps"][L]["third"]["A_exposure"]
                    if x["entity"] == e)
        assert v["delivered_exposures_about_entity"] == exp_n
    d = tmp_path / "evals" / "seed12"; d.mkdir(parents=True)
    (d / "prep_seed12.json").write_text((tmp_path / "prep2.json").read_text())
    (d / "exposure_seed12.json").write_text((out / "exposure_seed12.json").read_text())
    (d / "train_seed12.json").write_text((out / "train_seed12.json").read_text())
    for L in range(1, 6):
        run("evalck", "--prep", tmp_path / "prep2.json", "--model", model_dir, "--sleep", L, "--out", d / f"eval_N_s{L}.json",
            "--test-mode")
        for arm in ("P", "U", "G"):
            for mode in ("merged", "unmerged"):
                run("evalck", "--prep", tmp_path / "prep2.json", "--model", model_dir, "--sleep", L, "--ckpt",
                    out / f"cortex_{arm}_s{L}.pt", "--mode", mode, "--out", d / f"eval_{arm}_s{L}_{mode}.json", "--test-mode")
    ev = json.loads((d / "eval_U_s5_merged.json").read_text())
    assert ev["env"]["tf32"] is False and "TPOST_correct_A" in ev["measures"]
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "an.json", "--test-mode")
    an = json.loads((tmp_path / "an.json").read_text())
    assert all(v["ok"] for v in an["integrity"].values()), an["integrity"]              # fp32 merged == unmerged
    assert an["verdict"].startswith("ABST0_") and not an["invalid_arms"] and an["errors"] == []
    assert an["outcome_tables"] and any(k.startswith("U|5|HNM|template") for k in an["outcome_tables"])
    r = subprocess.run([py, str(ROOT / "scripts/abst0.py"), "analyze", "--dir", str(tmp_path / "evals"), "--seeds", "12",
                        "--out", str(tmp_path / "anprod.json")], capture_output=True, text=True, timeout=600)
    assert json.loads((tmp_path / "anprod.json").read_text())["verdict"] == "ABST0_INVALID"   # production rejects test runs
    # provenance: an evaluation of a checkpoint other than the one trained for that arm/sleep invalidates the run
    good = (d / "eval_U_s2_merged.json").read_text()
    ev2 = json.loads(good); ev2["provenance"]["ckpt_sha256"] = "0" * 64
    (d / "eval_U_s2_merged.json").write_text(json.dumps(ev2))
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "anp.json", "--test-mode")
    anp = json.loads((tmp_path / "anp.json").read_text())
    assert anp["verdict"] == "ABST0_INVALID" and "U" in anp["invalid_arms"]
    assert any("ckpt_sha256" in e for e in anp["integrity"]["12:U:2"]["validation_errors"])
    (d / "eval_U_s2_merged.json").write_text(good)
    # both twins carry the same forged gold labels: twin agreement must not rescue them
    keep = {m: (d / f"eval_U_s4_{m}.json").read_text() for m in ("merged", "unmerged")}
    for m in ("merged", "unmerged"):
        ev4 = json.loads(keep[m])
        for r in ev4["rows"]:
            if r["cat"] == "POS":
                r.update(value="Zzzforged", answer="Zzzforged", outcome="correct")
        (d / f"eval_U_s4_{m}.json").write_text(json.dumps(ev4))
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "anf.json", "--test-mode")
    anf = json.loads((tmp_path / "anf.json").read_text())
    assert anf["verdict"] == "ABST0_INVALID" and "U" in anf["invalid_arms"] and anf["integrity"]["12:U:4"]["ok"]
    for m in ("merged", "unmerged"):
        (d / f"eval_U_s4_{m}.json").write_text(keep[m])
    # a train log from different code is rejected upstream
    tlog = (d / "train_seed12.json").read_text()
    tl = json.loads(tlog); tl["provenance"]["code"] = "0" * 64
    (d / "train_seed12.json").write_text(json.dumps(tl))
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "anc.json", "--test-mode")
    anc = json.loads((tmp_path / "anc.json").read_text())
    assert anc["verdict"] == "ABST0_INVALID" and any("frozen code/model" in e for e in anc["errors"])
    (d / "train_seed12.json").write_text(tlog)
    # an unmerged evaluation that disagrees invalidates that arm
    orig_p3 = (d / "eval_P_s3_unmerged.json").read_text()
    bad = json.loads(orig_p3); bad["rows"][0]["answer"] = "zzz"
    (d / "eval_P_s3_unmerged.json").write_text(json.dumps(bad))
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "an2.json", "--test-mode")
    an2 = json.loads((tmp_path / "an2.json").read_text())
    assert an2["verdict"] == "ABST0_INVALID" and "P" in an2["invalid_arms"]
    (d / "eval_P_s3_unmerged.json").write_text(orig_p3)
    bad = json.loads((d / "eval_G_s5_unmerged.json").read_text()); bad["ppl_ratio"]["eval"] += 1e-3
    (d / "eval_G_s5_unmerged.json").write_text(json.dumps(bad))
    run("analyze", "--dir", tmp_path / "evals", "--seeds", "12", "--out", tmp_path / "an3.json", "--test-mode")
    an3 = json.loads((tmp_path / "an3.json").read_text())                       # invalid G: mechanism not assessable
    assert an3["verdict"].endswith("MECHANISM_NOT_ASSESSABLE") and an3["criteria"]["mechanism"] == "NOT_ASSESSABLE"
    assert an3["criteria"]["verdict"] == an3["verdict"]                          # nested verdict consistent
    assert "TPOST_correct_Aexp" in an3["pooled"]["U"]["5"] or "TPOST_correct_Aexp" in an3["pooled"]["U"][5]


def test_length_guard():
    ok = {"U": [{"ids": [0] * 256}], "P": [{"ids": [0] * 10}]}
    assert X.over_length(ok) == [] and X.over_length({"G": [{"ids": [0] * 257}]}) == [257]


def test_decide3_ties_never_accept():
    d = X.decide3
    assert d({"SUPPORTED": -2, "UNSUPPORTED": -1, "AMBIGUOUS": -1}) == "TIE"          # Sol's reproduction
    assert d({"SUPPORTED": -3, "UNSUPPORTED": -1, "AMBIGUOUS": -2}) == "UNSUPPORTED"
    assert d({"SUPPORTED": -1, "UNSUPPORTED": -1, "AMBIGUOUS": -5}) == "TIE"          # tied SUPPORTED is no exposure
    assert d({"SUPPORTED": -1, "UNSUPPORTED": float("nan"), "AMBIGUOUS": -5}) == "TIE"
    assert d({"SUPPORTED": float("-inf"), "UNSUPPORTED": -1, "AMBIGUOUS": -5}) == "TIE"


def test_duplicates_dedupe_cap_and_conserved_exposure():
    c = [{"entity": "Ann Bo", "q": q, "reject": None} for q in ("Where?", "where", "Who?", "Why?", "When?")]
    X.dedupe(c, lambda x: (x["entity"], X.C.norm(x["q"])))
    assert [x["reject"] for x in c] == [None, "duplicate", None, None, None]
    acc = X.cap(c, lambda x: x["entity"], 3)
    assert [x["q"] for x in acc] == ["Where?", "Who?", "Why?"] and c[4]["reject"] == "cap"
    t = [{"name": "T", "q": f"q{i}?", "reject": None} for i in range(5)]
    assert len(X.cap(t, lambda x: x["name"], 3)) == 3                                 # the cap applies to T too
    # four delivered A-items over a 2-item pool: exposures are summed per item, never overwritten
    nsl = {"A_accepted": [{"q": "a?", "entity": "E1", "swap": "g?"}, {"q": "b?", "entity": "E2", "swap": "h?"}],
           "T_accepted": [{"q": "t?"}]}
    U, G, log = X.third_slot(nsl, 8, 12, 1)
    assert sum(x["exposures"] for x in log["A_exposure"]) == 4 == sum(u["src"] == "A" for u in U)
    assert len({x["q"] for x in log["A_exposure"]}) == len(log["A_exposure"])


def test_audit_binding_and_stats():
    sample = {"items": [{"sleep": 1, "entity": "E1", "q": "a?", "memory": "m"}, {"sleep": 1, "entity": "E2", "q": "b?",
                                                                                  "memory": "m"}]}
    labels = {"prep_sha256": "P", "labels": [{"sleep": 1, "entity": "E1", "q": "a?", "label": "unsupported"},
                                             {"sleep": 1, "entity": "E2", "q": "b?", "label": "supported"}]}
    assert X.validate_audit("P", sample, labels) == []
    assert X.validate_audit("Q", sample, labels)                                      # bound to another prep
    assert X.validate_audit("P", sample, dict(labels, labels=labels["labels"][:1]))   # incomplete
    bad = json.loads(json.dumps(labels)); bad["labels"][1]["label"] = "maybe"
    assert X.validate_audit("P", sample, bad)
    bad = json.loads(json.dumps(labels)); bad["labels"][1]["q"] = "c?"
    assert X.validate_audit("P", sample, bad)                                         # not the sampled item
    tr = {"sleeps": {"1": {"third": {"A_exposure": [{"q": "a?", "entity": "E1", "exposures": 3},
                                                    {"q": "b?", "entity": "E2", "exposures": 1}]}}}}
    st = X.audit_stats(sample, labels, tr)[1]
    assert st["n_audited"] == 2 and st["error_rate"] == 0.5 and st["delivered_bad_exposure_mass"] == 0.25


def test_validate_eval():
    st = X.make_stream(12, LG)
    P = X.probes(st, 2, LG)
    rows = [dict(p, answer="UNKNOWN", outcome=X.outcome("UNKNOWN", p["value"])) for p in P]
    exp = {"seed": 12, "prep_sha256": "P", "sleep": 2, "mode": "merged", "ckpt_sha256": "C", "model_manifest": "M",
           "test_mode": False, "code": "K"}
    ev = {"provenance": dict(exp), "rows": rows, "ppl_ratio": {"eval": 1.0, "modern": 1.0}}
    assert X.validate_eval(ev, st, LG, exp) == []
    for k, v in (("ckpt_sha256", "X"), ("mode", "base"), ("test_mode", True), ("model_manifest", ""), ("sleep", 3)):
        assert X.validate_eval(dict(ev, provenance=dict(exp, **{k: v})), st, LG, exp), k
    assert X.validate_eval(dict(ev, ppl_ratio={"eval": 1.0}), st, LG, exp)            # one monitor only
    assert X.validate_eval(dict(ev, ppl_ratio={"eval": 1.0, "modern": float("inf")}), st, LG, exp)
    assert X.validate_eval(dict(ev, rows=rows[1:]), st, LG, exp)                      # not the expected inventory
    wrong = [dict(r) for r in rows]; wrong[0]["outcome"] = "correct"
    assert X.validate_eval(dict(ev, rows=wrong), st, LG, exp)                         # outcome must follow the answer
    # Sol's reproduction: invented gold values + matching answers + "correct" outcomes must not validate
    forged = [dict(r, value="Zzzforged", answer="Zzzforged", outcome="correct") if r["cat"] == "POS" else dict(r)
              for r in rows]
    assert X.validate_eval(dict(ev, rows=forged), st, LG, exp)
    assert all(r["outcome"] != "correct" for r in X.regrade(forged, st, 2, LG) if r["cat"] == "POS")


@pytest.mark.slow
def test_calibration_tiny(tmp_path):
    """Calibration path: train P sleep 1 per E, merged + unmerged + N fp32 evaluations, validated calchoose; the tiny
    model acquires nothing, so calibration must STOP with exit 4 (and record no validation errors)."""
    pytest.importorskip("torch")
    import tiny_hf_model
    model_dir = tiny_hf_model.build(tmp_path / "m")
    py = sys.executable
    run = lambda *a: subprocess.run([py, str(ROOT / "scripts/abst0.py"), *map(str, a)], check=True, capture_output=True,
                                    text=True, timeout=3000)
    cal = tmp_path / "cal"; cal.mkdir()
    run("prep", "--seed", 16, "--model", model_dir, "--out", cal / "prep_tmp.json", "--test-mode")
    prep = json.loads((cal / "prep_tmp.json").read_text()); st = prep["stream"]
    prep["qa"]["accepted"] = {f["fid"]: [{"q": f"What about {f['entity']}?", "a": f["value"]}] for f in st["facts"]}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    n = lambda q: len(X.C.qa_ids(tok, q, "UNKNOWN")[0])
    sw = {e: next(f"What does {t} eat?" for t in st["T"] if n(f"What does {t} eat?") == n(f"What does {e} eat?")) for e in st["A"]}
    prep["neg"]["1"]["A_accepted"] = [{"entity": e, "q": f"What does {e} eat?", "swap": sw[e]} for e in st["A"]]
    prep["neg"]["1"]["T_accepted"] = [{"q": f"Where was {t} born?"} for t in st["T"][-3:]]
    (cal / "prep_seed16.json").write_text(json.dumps(prep))
    P = cal / "prep_seed16.json"
    run("evalck", "--prep", P, "--model", model_dir, "--sleep", 1, "--out", cal / "eval_N_s1.json", "--test-mode")
    for E in (4, 8, 16):
        run("train", "--prep", P, "--model", model_dir, "--passes", E, "--arms", "P", "--sleeps", 1, "--out",
            cal / f"E{E}", "--test-mode", "--calibration")
        for mode in ("merged", "unmerged"):
            run("evalck", "--prep", P, "--model", model_dir, "--sleep", 1, "--ckpt", cal / f"E{E}" / "cortex_P_s1.pt",
                "--mode", mode, "--out", cal / f"E{E}" / f"eval_P_s1_{mode}.json", "--test-mode")
    r = subprocess.run([py, str(ROOT / "scripts/abst0.py"), "calchoose", "--dir", str(cal), "--out", str(cal / "c.json"),
                        "--test-mode"], capture_output=True, text=True, timeout=600)
    c = json.loads((cal / "c.json").read_text())
    assert r.returncode == 4 and c["stop"] and c["errors"] == [], (r.returncode, c["errors"])
    assert all(v["integrity"]["ok"] for v in c["passes"].values())
    (cal / "E8" / "eval_P_s1_unmerged.json").unlink()                           # a missing twin is an error, not a pass
    r = subprocess.run([py, str(ROOT / "scripts/abst0.py"), "calchoose", "--dir", str(cal), "--out", str(cal / "c2.json"),
                        "--test-mode"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 4 and json.loads((cal / "c2.json").read_text())["errors"]
