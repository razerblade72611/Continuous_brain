"""SUPPORT-1 prep invariants (protocol v2.1 §2-§5, §7, §9). Claude's own checks; Sol's validator is independent.
Runs prep in test mode with the tiny tokenizer (G token matching is relaxed there by design; checked in production)."""
import collections, json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT))
import support1 as S


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    import tiny_hf_model
    from transformers import AutoTokenizer
    d = tiny_hf_model.build(tmp_path_factory.mktemp("tiny") / "m")
    return AutoTokenizer.from_pretrained(d)


@pytest.fixture(scope="module")
def P(tok):
    return S.prep(20, tok, "", test_mode=True)


def cells(rows, kinds):
    return [tuple(x["cell"]) for x in rows if x["kind"] in kinds]


def test_deterministic(tok, P):
    assert S.canon(S.prep(20, tok, "", test_mode=True)) == S.canon(P)


def test_stream_is_abst0_construction_with_new_key():
    import abst0
    from being.v3 import lifegen
    st = S.make_stream(20)
    assert st != abst0.make_stream(20, lifegen)            # different key -> different stream
    abst0.check_stream(st)


def test_strata_and_split(P):
    st, spl, strat = P["stream"], P["split"], P["strata"]
    sel, how = {tuple(c) for c in spl["sel"]}, {tuple(c) for c in spl["how"]}
    assert not sel & how and sel | how == {tuple(c) for c in strat["pm"] if c[0] in st["A"]}
    for a in S.ATTRS9:
        assert abs(sum(c[1] == a for c in sel) - sum(c[1] == a for c in how)) <= 1
    for e in st["A"]:
        ns, nh = sum(c[0] == e for c in sel), sum(c[0] == e for c in how)
        assert ns >= 1 and nh >= 1 and abs(ns - nh) <= 1
    have = {(f["entity"], f["attr"]) for f in st["facts"]}
    assert not {tuple(c) for c in strat["pm"]} & have
    assert {tuple(c) for c in strat["rev"]} == {tuple(c) for c in st["transitions"] if c[0] in st["A"]}


def test_t_partition_global(P):
    tn, ts = set(P["tpart"]["t_neg"]), set(P["tpart"]["t_swap"])
    assert not tn & ts and tn | ts == set(P["stream"]["T"])
    for arm in ("U1", "U4"):                               # no T_swap name ever reaches a U arm
        for L, rows in P["schedule"][arm].items():
            assert not {x["cell"][0] for x in rows if x["cell"]} & ts


def test_nested_dose_and_g_mapping(P):
    for L in map(str, range(1, 6)):
        third = lambda arm: collections.Counter(
            (x["kind"], tuple(x["cell"]), x["wording"], tuple(x["src_cell"] or ())) for x in P["schedule"][arm][L]
            if x["slot"] == "third" and x["kind"] != "bg")
        u1, u4, g1, g4 = third("U1"), third("U4"), third("G1"), third("G4")
        assert u4 == collections.Counter({k: 4 * v for k, v in u1.items()})
        assert g4 == collections.Counter({k: 4 * v for k, v in g1.items()})
        assert not third("P")
        sw = {(tuple(m["src_cell"]), m["wording"]): m["t_name"] for m in P["swap_map"][L]}
        assert sum(v for k, v in g1.items() if k[0] == "neg_swap") == sum(v for k, v in u1.items() if k[0] == "neg_a")
        for (kind, cell, k, src), _ in g1.items():
            if kind == "neg_swap":
                assert sw[(src, k)] == cell[0]


def test_no_unknown_on_forbidden_cells(P):
    st = P["stream"]
    sel, rev = {tuple(c) for c in P["split"]["sel"]}, {tuple(c) for c in P["strata"]["rev"]}
    for arm in S.ARMS:
        for L, rows in P["schedule"][arm].items():
            L = int(L)
            for x in rows:
                if x["kind"] == "neg_a":
                    c = tuple(x["cell"])
                    assert c in sel | rev and c[0] in st["A"] and st["intro"][c[0]] <= L
                    if c in rev:
                        assert L <= 3
                if x["kind"] in ("pos", "replay"):
                    assert x["label"] != "UNKNOWN"


def test_positives_and_replay_match_oracle(P):
    facts = P["stream"]["facts"]
    for arm in S.ARMS:
        for L, rows in P["schedule"][arm].items():
            orc = S.oracle(facts, int(L))
            for x in rows:
                if x["kind"] in ("pos", "replay"):
                    assert orc[tuple(x["cell"])] == x["label"], (arm, L, x["cell"])
    upd = [tuple(c) for c in P["stream"]["updates"]]           # the superseded value is never replayed after sleep 4
    new = {c: next(f["value"] for f in facts if (f["entity"], f["attr"]) == c and f["role"] == "update_new") for c in upd}
    for L in ("4", "5"):
        for x in P["schedule"]["P"][L]:
            if x["kind"] == "replay" and tuple(x["cell"]) in new:
                assert x["label"] == new[tuple(x["cell"])]


def test_rev_quota(P):
    st, rev = P["stream"], [tuple(c) for c in P["strata"]["rev"]]
    for L in (1, 2, 3):
        for arm, q in (("U1", 2), ("U4", 8)):
            cnt = collections.Counter(tuple(x["cell"]) for x in P["schedule"][arm][str(L)] if x["kind"] == "neg_a")
            for c in rev:
                assert cnt[c] == (q if st["intro"][c[0]] <= L else 0)
    for L in (4, 5):
        assert not [x for x in P["schedule"]["U4"][str(L)] if x["kind"] == "neg_a" and tuple(x["cell"]) in set(rev)]


def test_budget_and_positions(P):
    for L in map(str, range(1, 6)):
        rows = {arm: P["schedule"][arm][L] for arm in S.ARMS}
        assert len({len(r) for r in rows.values()}) == 1
        for arm in ("P", "U1", "U4"):                       # G is length-matched only with the real tokenizer
            assert S.budget(rows[arm]) == S.budget(rows["P"])
        for i in range(len(rows["P"])):                     # positives identical at identical positions in every arm
            ref = rows["P"][i]
            for arm in S.ARMS:
                x = rows[arm][i]
                assert x["pos"] == i
                if ref["slot"] in ("new", "replay"):
                    assert x == ref


def test_fork_withheld_has_no_target_supervision(P):
    tg = {tuple(c) for c in P["fork"]["targets"]}
    assert len(tg) == 4 and all(c[0] in P["stream"]["H"] for c in tg)
    for arm in ("U4", "G4"):
        rows = P["fork_schedule"][arm]
        assert not [x for x in rows if x["cell"] and tuple(x["cell"]) in tg]
        main = P["schedule"][arm]["4"]
        replaced = [i for i, (a, b) in enumerate(zip(main, rows)) if a != b]
        assert len(replaced) == 32 and all(tuple(main[i]["cell"]) in tg for i in replaced)
        assert {rows[i]["cell"][0] for i in replaced} == set(P["fork"]["entities"])
    stream_words = {w.lower() for e in P["stream"]["A"] + P["stream"]["H"] + P["stream"]["T"] + P["stream"]["V"]
                    for w in e.split()}
    assert not {w.lower() for e in P["fork"]["entities"] for w in e.split()} & stream_words


def test_probes(P):
    for h, d in P["probes"].items():
        ids = [r["probe_id"] for L in d for r in d[L]]
        assert len(ids) == len(set(ids))
    for L in map(str, range(1, 6)):
        rows = P["probes"]["observed"][L]
        by = {r["probe_id"]: r for r in rows}
        for r in rows:
            assert r["cell"] is not None
            if r["pair_probe_id"]:
                assert by[r["pair_probe_id"]]["cell"] == r["cell"] and by[r["pair_probe_id"]]["construction"] == "statement"
        delivered = {(tuple(x["cell"]), x["wording"]) for Lp in range(1, int(L) + 1)
                     for x in P["schedule"]["U1"][str(Lp)] if x["kind"] == "neg_a"}
        for r in rows:
            if r["cat"] == "SEL-Q":
                assert (tuple(r["cell"]), int(r["kind"][2])) in delivered and r["delivered_eligible"] is True
            if r["cat"] == "GTQ":
                assert r["cell"][0] in set(P["tpart"]["t_swap"])
        c = collections.Counter(r["cat"] for r in rows)
        assert c["REV-Q"] == 4 and c["SEL-Q"] <= 12 and c["GTQ"] <= 16
    w4 = P["probes"]["withheld"]["4"]
    tg = {tuple(c) for c in P["fork"]["targets"]}
    for r in w4:
        if r["cat"] in ("FORK-H", "FORK-H-Q"):
            assert r["gold_current"] == "UNKNOWN" and tuple(r["cell"]) in tg
    obs = {r["probe_id"].split("-", 2)[2]: r for r in P["probes"]["observed"]["4"] if r["cat"] == "FORK-H"}
    assert all(r["gold_current"] != "UNKNOWN" for r in obs.values())


def test_norm_vectors():
    vec = json.loads((ROOT / "lab/support1_norm_vectors.json").read_text())
    assert len(vec) == 15 and all(S.C.B.norm(a) == b for a, b in vec)


@pytest.mark.slow
def test_end_to_end_tiny(tmp_path):
    """Plumbing on CPU with the tiny model: prep -> train (all arms, fork) -> restoretest -> evalck (merged/unmerged/base,
    withheld) -> calcheck. Scores are meaningless; identities, files and integrity are what is checked."""
    import subprocess, tiny_hf_model
    m = tiny_hf_model.build(tmp_path / "m")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/support1.py"), *map(str, x)], check=True,
                                    capture_output=True, text=True).stdout
    run("prep", "--seeds", "24", "--tokenizer", m, "--out", tmp_path, "--test-mode")
    prep = tmp_path / "prep_seed24.json"
    run("train", "--prep", prep, "--model", m, "--out", tmp_path / "tr", "--sleeps", 4, "--fork", "--test-mode")
    tr = json.loads((tmp_path / "tr/train_seed24.json").read_text())
    assert set(tr["arms"]) == set(S.ARMS) and set(tr["fork"]["withheld"]) == {"U4", "G4"}
    assert tr["arms"]["U4"]["3"]["full_state_hash"] == tr["fork"]["parent_full_state_hash"]["U4"]
    assert tr["fork"]["withheld"]["U4"]["ckpt_sha256"] != tr["arms"]["U4"]["4"]["ckpt_sha256"]
    tgt = {tuple(c) for c in json.loads(prep.read_text())["fork"]["targets"]}
    assert not [d for d in tr["fork"]["withheld"]["U4"]["delivered"] if tuple(d["cell"]) in tgt]
    out = run("restoretest", "--prep", prep, "--model", m, "--out", tmp_path / "rt.json")
    assert json.loads((tmp_path / "rt.json").read_text())["admitted"] is True, out
    ev = lambda name, *x: run("evalck", "--prep", prep, "--model", m, "--out", tmp_path / name, "--test-mode", *x)
    ev("m.json", "--sleep", 1, "--arm", "P", "--ckpt", tmp_path / "tr/cortex_P_s1.pt", "--mode", "merged")
    ev("u.json", "--sleep", 1, "--arm", "P", "--ckpt", tmp_path / "tr/cortex_P_s1.pt", "--mode", "unmerged")
    ev("n.json", "--sleep", 1)
    ev("w.json", "--sleep", 4, "--arm", "U4", "--ckpt", tmp_path / "tr/cortex_U4_s4_withheld.pt", "--history", "withheld",
       "--qdiag", 32)
    mj, uj, wj = (json.loads((tmp_path / f).read_text()) for f in ("m.json", "u.json", "w.json"))
    ids = [r["probe_id"] for r in json.loads(prep.read_text())["probes"]["observed"]["1"]]
    assert [r["probe_id"] for r in mj["rows"]] == ids == [r["probe_id"] for r in uj["rows"]]
    assert [r["answer"] for r in mj["rows"]] == [r["answer"] for r in uj["rows"]]           # fp32 merged == unmerged
    assert sum(r["probe_id"].split("-")[-2] == "Q" or "-Q-" in r["probe_id"] or "GTQ" in r["probe_id"] for r in wj["rows"]) <= 32 + 4
    assert wj["history"] == "withheld" and all(r["probe_id"].startswith("24-withheld-4-") for r in wj["rows"])
    p = subprocess.run([sys.executable, str(ROOT / "scripts/support1.py"), "calcheck", "--prep", prep, "--merged",
                        tmp_path / "m.json", "--unmerged", tmp_path / "u.json", "--base", tmp_path / "n.json", "--out",
                        tmp_path / "cal.json"], capture_output=True, text=True)
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["errors"] == [] and p.returncode == (0 if cal["pass"] else 4)
