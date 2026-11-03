"""PROV-1 prep invariants (lab/PROV1_PROTOCOL.md §2-§4, §9). Claude's own checks; Sol reviews independently.
Runs prep in test mode with the tiny tokenizer."""
import collections, json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT))
import prov1 as V
import support1 as S


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    import tiny_hf_model
    from transformers import AutoTokenizer
    d = tiny_hf_model.build(tmp_path_factory.mktemp("tiny") / "m")
    return AutoTokenizer.from_pretrained(d)


@pytest.fixture(scope="module")
def P(tok):
    return V.prep(20, tok, "", test_mode=True)


@pytest.fixture(scope="module")
def P1(tok):
    return S.prep(20, tok, "", test_mode=True)


def test_deterministic(tok, P):
    assert S.canon(V.prep(20, tok, "", test_mode=True)) == S.canon(P)


def test_support1_code_untouched():
    assert S.code_hash() == "38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae"


def test_probe_inventory_is_support1(P, P1):
    assert P["support1_probes_sha256"] == S.chash(P1["probes"]) and P["stream_sha256"] == P1["stream_sha256"]
    for h, d in P1["probes"].items():
        for L, rows in d.items():
            absent = [r for r in P["probes"][h][L] if r["condition"] == "ABSENT"]
            assert [(r["base_probe_id"], r["item"], r["gold_truth"]) for r in absent] == \
                   [(q["probe_id"], q["item"], q["gold_current"]) for q in rows]


def test_record_is_support_only_and_faithful(P):
    st, fork = P["stream"], P["fork"]
    for h, d in P["probes"].items():
        facts = S.history_facts(st, fork, h)
        for L, rows in d.items():
            for r in rows:
                if r["condition"] == "FAITHFUL":
                    rs = V.record_set(facts, r["cell"][0], int(L))
                    assert r["record"] == V.render(r["cell"][0], rs)
                    assert r["supported"] == (r["cell"][1] in rs)
                if r["record"] and r["condition"] != "FULL":
                    assert not V.leak_words(r["record"], r["cell"][0], facts + fork["facts"])
                for c in V.CONDITIONS:                       # condition names never in prompts (Sol H4)
                    assert c not in r["item"]


def test_truth_never_changed_by_corruption(P):
    for d in P["probes"].values():
        for rows in d.values():
            by = collections.defaultdict(dict)
            for r in rows:
                by[r["base_probe_id"]][r["condition"]] = r
            for conds in by.values():
                truths = {r["gold_truth"] for r in conds.values()}
                assert len(truths) == 1                      # one truth label per base probe, every condition
                if "DEL-Q" in conds:
                    assert conds["DEL-Q"]["gold_compliance"] == "UNKNOWN" and conds["FAITHFUL"]["supported"]
                    assert conds["DEL-Q"]["cell"][1] not in conds["DEL-Q"]["record"].split("experienced:")[-1].split(".")[0] \
                        or conds["DEL-Q"]["record"].endswith("no experiences.")
                if "DEL-X" in conds:
                    x = conds["DEL-X"]
                    assert x["gold_compliance"] == x["gold_truth"] and x["delx_attr"] != x["cell"][1]
                    assert f"{x['cell'][1]} (life" in x["record"] and f"{x['delx_attr']} (life" not in x["record"]
                if "ADD" in conds:
                    assert conds["ADD"]["gold_compliance"] is None and not conds["FAITHFUL"]["supported"]
                    assert conds["ADD"]["gold_truth"] == conds["FAITHFUL"]["gold_truth"]


def test_delx_draw_shared_across_constructions(P):
    for d in P["probes"].values():
        for L, rows in d.items():
            pick = collections.defaultdict(set)
            for r in rows:
                if r["condition"] == "DEL-X":
                    pick[tuple(r["cell"])].add(r["delx_attr"])
            assert all(len(v) == 1 for v in pick.values())


def test_budgets_equal_and_slots(P):
    for L in map(str, range(1, 6)):
        assert len({tuple(P["budget"][a][L]) for a in V.ARMS}) == 1
        k = {a: collections.Counter(x["kind"] for x in P["schedule"][a][L]) for a in V.ARMS}
        new = {a: [(x["item_id"], x["input_ids"]) for x in P["schedule"][a][L] if x["slot"] == "new"] for a in V.ARMS}
        rep = {a: sorted(tuple(x["input_ids"]) for x in P["schedule"][a][L] if x["slot"] == "replay") for a in V.ARMS}
        assert new["RP"] == new["RU"] == new["RC"] and rep["RP"] == rep["RU"] == rep["RC"]
        assert k["RP"]["neg_a"] == k["RP"]["neg_t"] == k["RP"]["cf"] == 0
        assert k["RU"]["neg_a"] + k["RU"]["neg_t"] == k["RC"]["neg_a"] + k["RC"]["neg_t"] and k["RU"]["cf"] == 0
        neg = {a: sorted(tuple(x["input_ids"]) for x in P["schedule"][a][L] if x["kind"] in ("neg_a", "neg_t")) for a in ("RU", "RC")}
        assert neg["RU"] == neg["RC"]
        n = len([x for x in P["schedule"]["RC"][L] if x["slot"] == "new"]) // S.E_PASSES
        third = [x for x in P["schedule"]["RC"][L] if x["slot"] == "third"]
        assert len(third) == 8 * n and k["RC"]["cf"] == 2 * n and k["RC"]["neg_a"] + k["RC"]["neg_t"] == 6 * n


def test_counterfactuals(P):
    st = P["stream"]
    rev = {tuple(c) for c in P["strata"]["rev"]}
    upd = {tuple(c) for c in st["updates"]}
    for L in map(str, range(1, 6)):
        orc = S.oracle(st["facts"], int(L))
        for x in P["schedule"]["RC"][L]:
            if x["kind"] == "cf":
                c = tuple(x["cell"])
                assert c[0] in st["A"] and c not in rev and c not in upd and c in orc   # A, supported, never REV/update
                assert x["label"] == "UNKNOWN" and x["rec_mode"] == "delq" and f"{c[1]} (life" not in x["record"]
            if x["kind"] in ("neg_a", "neg_t", "cf"):
                assert x["cell"][0] not in st["H"]                                       # H never negatively supervised


def test_negative_block_is_support1_base_block_x3(P, P1):
    for L in map(str, range(1, 6)):
        b = P1["base_blocks"][L]
        want = collections.Counter([(tuple(c), k) for c, k in b["A"]] + [(tuple(c), k) for c, k in b["T"]])
        got = collections.Counter((tuple(x["cell"]), x["wording"]) for x in P["schedule"]["RU"][L] if x["kind"] in ("neg_a", "neg_t"))
        assert got == collections.Counter({k: v * V.NEG_REPS for k, v in want.items()})


def test_fork_branch(P):
    st, fork = P["stream"], P["fork"]
    tg = {tuple(c) for c in fork["targets"]}
    wh = S.history_facts(st, fork, "withheld")
    for arm in V.FORK_ARMS:
        rows, obs = P["fork_schedule"][arm], P["schedule"][arm]["4"]
        assert len(rows) == len(obs) and [x["pos"] for x in rows] == [x["pos"] for x in obs]
        assert not any(tuple(x["cell"]) in tg for x in rows if x["kind"] != "bg")      # no target supervision
        for x in rows:
            if x.get("record"):
                assert x["record"] == V.render(x["cell"][0], {k: v for k, v in V.record_set(wh, x["cell"][0], 4).items()
                                                              if not (x["rec_mode"] == "delq" and k == x["cell"][1])})


def test_rendered_ids_round_trip(tok, P):
    x = next(x for x in P["schedule"]["RC"]["2"] if x["kind"] == "pos")
    ids, nl = V.rq_ids(tok, x["record"], S.QT[x["cell"][1]][x["wording"]].format(e=x["cell"][0]), x["label"])
    assert ids == x["input_ids"] and nl == x["n_loss"]
    text = tok.decode(x["input_ids"])
    assert x["record"] in text and text.index(x["record"]) < text.index("Question:")


def test_end_to_end_tiny(tmp_path):
    """Plumbing on CPU with the tiny model: prep -> train (RP/RU/RC, RU/RC fork) -> evalck (merged/unmerged, base,
    withheld, SUPPORT-1 checkpoint as phase-A source). Scores are meaningless; identities, files and twins are checked."""
    import subprocess, tiny_hf_model
    m = tiny_hf_model.build(tmp_path / "m")
    run = lambda script, *x: subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, x)], check=True,
                                            capture_output=True, text=True).stdout
    run("prov1.py", "prep", "--seeds", "24", "--tokenizer", m, "--out", tmp_path, "--test-mode")
    prep = tmp_path / "prov1_prep_seed24.json"
    P = json.loads(prep.read_text())
    run("prov1.py", "train", "--prep", prep, "--model", m, "--out", tmp_path / "tr", "--sleeps", 4, "--fork", "--test-mode")
    tr = json.loads((tmp_path / "tr/prov1_train_seed24.json").read_text())
    assert set(tr["arms"]) == set(V.ARMS) and set(tr["fork"]["withheld"]) == set(V.FORK_ARMS)
    assert tr["arms"]["RC"]["3"]["full_state_hash"] == tr["fork"]["parent_full_state_hash"]["RC"]
    assert len({tr["arms"][a]["2"]["ckpt_sha256"] for a in V.ARMS}) == 3                    # arms differ in content
    ev = lambda name, *x: run("prov1.py", "evalck", "--prep", prep, "--model", m, "--out", tmp_path / name, "--test-mode", *x)
    ev("m.json", "--sleep", 2, "--arm", "RC", "--ckpt", tmp_path / "tr/cortex_RC_s2.pt", "--mode", "merged", "--monitors")
    ev("u.json", "--sleep", 2, "--arm", "RC", "--ckpt", tmp_path / "tr/cortex_RC_s2.pt", "--mode", "unmerged", "--monitors")
    ev("n.json", "--sleep", 2, "--arm", "N", "--monitors")
    ev("w.json", "--sleep", 4, "--arm", "RC", "--ckpt", tmp_path / "tr/cortex_RC_s4_withheld.pt", "--history", "withheld")
    mj, uj, nj, wj = (json.loads((tmp_path / f).read_text()) for f in ("m.json", "u.json", "n.json", "w.json"))
    want = [r["probe_id"] for r in P["probes"]["observed"]["2"] if r["condition"] in V.PHASE_B_CONDITIONS]
    assert [r["probe_id"] for r in mj["rows"]] == want == [r["probe_id"] for r in uj["rows"]]
    assert [r["answer"] for r in mj["rows"]] == [r["answer"] for r in uj["rows"]]             # fp32 merged == unmerged
    assert mj["base_denominator_sha256"] == uj["base_denominator_sha256"] == nj["base_denominator_sha256"]
    assert all(abs(mj["ppl_ratio"][k] - uj["ppl_ratio"][k]) < 1e-4 for k in mj["ppl_ratio"])
    by = {r["probe_id"]: r for r in P["probes"]["observed"]["2"]}
    for x in mj["rows"]:
        r = by[x["probe_id"]]
        assert x["outcome_truth"] == S.outcome(x["answer"], r["gold_truth"])
        assert x["outcome_compliance"] == (None if r["gold_compliance"] is None else S.outcome(x["answer"], r["gold_compliance"]))
    assert all(r["probe_id"].startswith("24-withheld-4-") for r in wj["rows"]) and wj["ppl_ratio"] is None
    # caches (Sol's HOLD at b88ca93): an identical request is reused; a different one is recomputed
    assert "up to date" in ev("m.json", "--sleep", 2, "--arm", "RC", "--ckpt", tmp_path / "tr/cortex_RC_s2.pt",
                              "--mode", "merged", "--monitors")
    assert "up to date" not in ev("m.json", "--sleep", 2, "--arm", "RC", "--ckpt", tmp_path / "tr/cortex_RC_s2.pt",
                                  "--mode", "unmerged", "--monitors")
    assert json.loads((tmp_path / "m.json").read_text())["mode"] == "unmerged"
    tr_again = lambda: run("prov1.py", "train", "--prep", prep, "--model", m, "--out", tmp_path / "tr", "--sleeps", 4,
                           "--fork", "--test-mode")
    assert "skipped" in tr_again()
    (tmp_path / "tr/cortex_RU_s4_withheld.pt").write_bytes(b"corrupt")
    assert "skipped" not in tr_again()
    assert "skipped" in tr_again()
    # phase A: a SUPPORT-1 checkpoint evaluated on PROV-1's record conditions
    run("support1.py", "prep", "--seeds", "24", "--tokenizer", m, "--out", tmp_path / "s1", "--test-mode")
    run("support1.py", "train", "--prep", tmp_path / "s1/prep_seed24.json", "--model", m, "--out", tmp_path / "s1tr",
        "--arms", "P", "--sleeps", 1, "--test-mode")
    ev("a.json", "--sleep", 1, "--arm", "P", "--ckpt", tmp_path / "s1tr/cortex_P_s1.pt", "--mode", "merged",
       "--conditions", ",".join(V.PHASE_A_CONDITIONS), "--source", "support1")
    aj = json.loads((tmp_path / "a.json").read_text())
    assert {by_c for by_c in (r["probe_id"].rsplit("-", 1)[-1] for r in aj["rows"])} <= {"FAITHFUL", "Q", "X", "ADD", "FULL"}
    assert aj["provenance"]["source"] == "support1" and aj["provenance"]["ckpt_sha256"] == \
        json.loads((tmp_path / "s1tr/train_seed24.json").read_text())["arms"]["P"]["1"]["ckpt_sha256"]


def test_calibration_and_projection_tiny(tmp_path):
    """Phase-1 plumbing: RP,RC sleep-1 calibration -> calcheck (no errors on genuine files; tampering is an error,
    exit 4) -> projection from a timings file over the actual inventories; the production train guard refuses a dev
    seed without validator and calibration bindings."""
    import subprocess, tiny_hf_model
    m = tiny_hf_model.build(tmp_path / "m")
    run = lambda script, *x, check=True: subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, x)],
                                                       check=check, capture_output=True, text=True)
    run("prov1.py", "prep", "--seeds", "24", "--tokenizer", m, "--out", tmp_path, "--test-mode")
    prep = tmp_path / "prov1_prep_seed24.json"
    run("prov1.py", "train", "--prep", prep, "--model", m, "--out", tmp_path / "cal", "--arms", "RP,RC", "--sleeps", 1, "--test-mode")
    ev = lambda name, *x: run("prov1.py", "evalck", "--prep", prep, "--model", m, "--out", tmp_path / name, "--test-mode",
                              "--sleep", 1, "--monitors", *x)
    for arm in ("RP", "RC"):
        for mode in ("merged", "unmerged"):
            ev(f"{arm}_{mode}.json", "--arm", arm, "--ckpt", tmp_path / f"cal/cortex_{arm}_s1.pt", "--mode", mode)
    ev("n.json", "--arm", "N")

    def calcheck():
        p = run("prov1.py", "calcheck", "--prep", prep, "--train", tmp_path / "cal/prov1_train_seed24.json",
                "--rp-merged", tmp_path / "RP_merged.json", "--rp-unmerged", tmp_path / "RP_unmerged.json",
                "--rc-merged", tmp_path / "RC_merged.json", "--rc-unmerged", tmp_path / "RC_unmerged.json",
                "--base", tmp_path / "n.json", "--out", tmp_path / "calcheck.json", check=False)
        return p.returncode, json.loads((tmp_path / "calcheck.json").read_text())
    rc, cal = calcheck()
    assert cal["errors"] == [] and rc == (0 if cal["pass"] else 4) and cal["test_mode"] is True
    j = json.loads((tmp_path / "RC_unmerged.json").read_text())
    j["rows"][0]["answer"] = "tampered " + j["rows"][0]["answer"]
    (tmp_path / "RC_unmerged.json").write_text(json.dumps(j))
    rc, cal = calcheck()
    assert rc == 4 and cal["pass"] is False and cal["errors"]
    run("support1.py", "prep", "--seeds", "24", "--tokenizer", m, "--out", tmp_path / "s1", "--test-mode")
    (tmp_path / "t.json").write_text(json.dumps({"train_items": 100, "train_seconds": 10.0, "eval_rows": 100, "eval_seconds": 40.0}))
    p = run("prov1.py", "project", "--timings", tmp_path / "t.json", "--preps", prep, "--support1-preps",
            tmp_path / "s1/prep_seed24.json", "--out", tmp_path / "proj.json", check=False)
    pj = json.loads((tmp_path / "proj.json").read_text())
    assert pj["phase_A"]["eval_rows"] > 0 and pj["phase_B"]["train_items"] > 0 and p.returncode == (0 if pj["phase_A_affordable"] else 5)
    # production guard: a non-test-mode train of a test prep is refused before any model load
    g = run("prov1.py", "train", "--prep", prep, "--model", m, "--out", tmp_path / "g", "--model-manifest-sha", "x" * 64, check=False)
    assert g.returncode != 0 and "production" in (g.stderr + g.stdout)
