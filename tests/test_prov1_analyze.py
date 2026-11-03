"""Claude's PROV-1 analyzer core on synthetic answer sets (no model): a perfect ledger-follower passes K1-K7; each
targeted failure mode fails the criterion meant to catch it (§5, Sol's design review)."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT))
import prov1 as V
import prov1_analyze as PA
import support1 as S


@pytest.fixture(scope="module")
def prep():
    import tiny_hf_model, tempfile
    from transformers import AutoTokenizer
    d = tiny_hf_model.build(Path(tempfile.mkdtemp()) / "m")
    return V.prep(20, AutoTokenizer.from_pretrained(d), "", test_mode=True)


def setup(prep):
    st = prep["stream"]
    meta = {(h, int(L)): rows for h, dd in prep["probes"].items() for L, rows in dd.items()}
    known = lambda r: r["gold_truth"] != "UNKNOWN"
    pref = {L: {r["base_probe_id"]: ("correct" if known(r) else "substantive")      # P: knows facts, confabulates gaps
                for r in meta[("observed", L)] if r["condition"] == "ABSENT"} for L in range(1, 6)}
    return st, meta, pref


def follower(meta, L, history="observed", delq="unknown", delx="correct", faithful=None):
    """Answers exactly what the shown record allows (the intended behaviour), with optional deviations."""
    out = {}
    for r in meta[(history, L)]:
        k = (r["condition"], r["base_probe_id"])
        if r["condition"] in ("FAITHFUL", "FULL"):
            out[k] = faithful(r) if faithful else ("correct" if r["supported"] else "unknown")
        elif r["condition"] == "ABSENT":
            out[k] = "correct" if r["gold_truth"] != "UNKNOWN" else "substantive"
        elif r["condition"] == "DEL-Q":
            out[k] = delq
        elif r["condition"] == "DEL-X":
            out[k] = delx
        elif r["condition"] == "ADD":
            out[k] = "unknown"
    return out


def run(prep, **kw):
    st, meta, pref = setup(prep)
    cand = {20: {L: follower(meta, L, **kw) for L in range(1, 6)}}
    fork = {20: {"obs": follower(meta, 4), "wit": follower(meta, 4, history="withheld")}}
    ppl = {20: {L: {"eval": 0.9, "modern": 0.9} for L in range(1, 6)}}
    return PA.analyze([20], {20: st}, {20: meta}, cand, {20: pref}, ppl=ppl, fork=fork)


def test_perfect_follower_passes_everything(prep):
    r = run(prep)
    assert r["pass"] == {k: True for k in ("K1", "K2", "K3", "K4", "K5", "K6", "K7")}, r["pass"]
    assert r["checks"]["K5"]["A"]["K5a"] == 1.0 and r["checks"]["K5"]["H"]["K5b"] == 1.0
    assert all(v > 0 for v in r["checks"]["K5"]["H"]["per_seed"][20]["eligible_cell_sleeps"].values())


def test_record_ignorer_fails_k5a_only(prep):
    r = run(prep, delq="correct")                     # answers from weights even when the record lacks the entry
    assert not r["pass"]["K5"] and r["checks"]["K5"]["A"]["K5a"] == 0.0
    assert all(r["pass"][k] for k in ("K1", "K2", "K3", "K4", "K6", "K7"))


def test_damage_detector_fails_k5b_only(prep):
    r = run(prep, delx="unknown")                     # any damaged-looking record -> UNKNOWN
    assert not r["pass"]["K5"] and r["checks"]["K5"]["H"]["K5b"] == 0.0
    assert all(r["pass"][k] for k in ("K1", "K2", "K3", "K4", "K6", "K7"))


def test_wrong_after_irrelevant_deletion_is_not_preservation(prep):
    r = run(prep, delx="substantive")                 # Sol: a wrong answer is not "preserved"
    assert r["checks"]["K5"]["A"]["K5b"] == 0.0 and not r["pass"]["K5"]


def test_suppressor_fails_acquisition_and_k7(prep):
    r = run(prep, faithful=lambda r: "unknown")       # abstains on everything
    assert not r["pass"]["K1"] and not r["pass"]["K7"] and r["checks"]["K7"]["POS-H"]["pass"] is False
    assert r["pass"]["K2"] and r["pass"]["K3"]


def test_confabulating_on_held_out_fails_k2(prep):
    r = run(prep, faithful=lambda r: "correct" if r["supported"] else ("substantive" if r["cat"] == "HNM" else "unknown"))
    assert not r["pass"]["K2"] and not r["pass"]["K7"] and r["checks"]["K7"]["HNM"]["candidate"] == 0.0


def test_empty_eligible_set_fails_not_drops(prep):
    st, meta, pref = setup(prep)
    meta2 = {k: [r for r in v if r["condition"] != "DEL-X"] for k, v in meta.items()}
    cand = {20: {L: follower(meta2, L) for L in range(1, 6)}}
    r = PA.analyze([20], {20: st}, {20: meta2}, cand, {20: pref}, ppl=None, fork=None)
    assert r["checks"]["K5"]["A"]["K5a"] is None and not r["pass"]["K5"]
    assert not r["pass"]["K6"]                        # no fork supplied -> not assessable -> no pass


def test_fork_requires_both_constructions(prep):
    st, meta, pref = setup(prep)
    cand = {20: {L: follower(meta, L) for L in range(1, 6)}}
    wit = follower(meta, 4, history="withheld")
    for r in meta[("withheld", 4)]:
        if r["cat"] == "FORK-H-Q" and r["condition"] == "FAITHFUL":
            wit[("FAITHFUL", r["base_probe_id"])] = "substantive"     # question form ignores the changed history
    r = PA.analyze([20], {20: st}, {20: meta}, cand, {20: pref}, ppl=None, fork={20: {"obs": follower(meta, 4), "wit": wit}})
    assert r["checks"]["K6"]["per_seed"][20]["FORK-H"]["history_sensitive"] is True
    assert r["checks"]["K6"]["per_seed"][20]["FORK-H-Q"]["history_sensitive"] is False and not r["pass"]["K6"]


def test_p_gate_meets_k1_k4_by_construction(prep):
    st, meta, pref = setup(prep)
    g, _ = PA.gate_k1_k4([20], {20: st}, {20: meta}, {20: pref})
    assert g == {"K1": True, "K2": True, "K3": True, "K4": True}


def _answer(r, how):
    truth = r["gold_truth"]
    if how == "follow":
        if r["condition"] in ("FAITHFUL", "FULL", "DEL-X"):
            return truth if r["supported"] else "UNKNOWN"
        if r["condition"] in ("DEL-Q", "ADD"):
            return "UNKNOWN"
        return truth if truth != "UNKNOWN" else "Zorvane"          # ABSENT: weights only, confabulates gaps
    return truth if truth != "UNKNOWN" else "Zorvane"               # P (SUPPORT-1 reference)


def test_cli_end_to_end_synthetic(tmp_path):
    """Full file path: PROV-1 phase-B files + SUPPORT-1 P reference on disk -> CLI -> PROV1_ADVANCE; one tampered
    outcome -> PROV1_INVALID."""
    import json, subprocess, tiny_hf_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_hf_model.build(tmp_path / "m"))
    pv, p1 = V.prep(20, tok, "", test_mode=True), S.prep(20, tok, "", test_mode=True)
    d, s1 = tmp_path / "b" / "seed20", tmp_path / "s1" / "seed20"
    d.mkdir(parents=True); s1.mkdir(parents=True)
    (d / "prov1_prep_seed20.json").write_text(json.dumps(pv)); (s1 / "prep_seed20.json").write_text(json.dumps(p1))
    psha, p1sha = V.sha((d / "prov1_prep_seed20.json").read_bytes()), S.sha((s1 / "prep_seed20.json").read_bytes())
    hx = lambda *k: V.sha("|".join(map(str, k)))
    den = hx("den")
    env = {"torch": "x", "tf32": False, "gpu": "cpu"}
    # SUPPORT-1 reference
    t1 = {"arms": {"P": {str(L): {"ckpt_sha256": hx("s1P", L)} for L in range(1, 6)}}}
    (s1 / "train_seed20.json").write_text(json.dumps(t1))
    for L in range(1, 6):
        rows = []
        for q in p1["probes"]["observed"][str(L)]:
            a = _answer({"gold_truth": q["gold_current"]}, "p")
            rows.append({"probe_id": q["probe_id"], "answer": a, "outcome": S.outcome(a, q["gold_current"])})
        ev = {"schema": S.SCHEMA, "experiment": S.EXPERIMENT, "seed": 20, "arm": "P", "sleep": L, "mode": "merged",
              "history": "observed", "provenance": {"code_hash": S.code_hash(), "prep_sha256": p1sha, "ckpt_sha256": hx("s1P", L),
                                                   "model_manifest": "", "test_mode": True, "qdiag": 64},
              "ppl_ratio": {"eval": 0.9, "modern": 0.9}, "base_denominator_sha256": den, "rows": rows, "env": env}
        (s1 / f"eval_P_s{L}_merged.json").write_text(json.dumps(ev))
    # PROV-1 train log + evaluations
    tr = {"schema": V.SCHEMA, "experiment": V.EXPERIMENT, "seed": 20,
          "provenance": {"code_hash": V.code_hash(), "prep_sha256": psha, "model_manifest": "", "test_mode": True,
                         "gpu": "cpu", "arms": list(V.ARMS), "sleeps": 5, "fork": True},
          "arms": {a: {str(L): {"ckpt_sha256": hx(a, L), "full_state_hash": hx("fs", a, L)} for L in range(1, 6)} for a in V.ARMS},
          "fork": {"withheld": {a: {"ckpt_sha256": hx(a, "w")} for a in V.FORK_ARMS},
                   "parent_full_state_hash": {a: hx("fs", a, 3) for a in V.FORK_ARMS}}}
    (d / "prov1_train_seed20.json").write_text(json.dumps(tr))

    def write(name, arm, L, mode, hist, ck, monitors=True):
        inv = [r for r in pv["probes"][hist][str(L)] if r["condition"] in V.PHASE_B_CONDITIONS]
        rows = []
        for r in inv:
            a = "UNKNOWN" if arm == "N" else _answer(r, "follow")
            rows.append({"probe_id": r["probe_id"], "answer": a, "outcome_truth": V.outcome(a, r["gold_truth"]),
                         "outcome_compliance": V.outcome(a, r["gold_compliance"])})
        prov = {"code_hash": V.code_hash(), "prep_sha256": psha, "ckpt_sha256": ck, "model_manifest": "", "test_mode": True,
                "conditions": list(V.PHASE_B_CONDITIONS), "monitors": monitors, "source": "prov1", "arm": arm, "sleep": L,
                "history": hist, "mode": mode}
        ev = {"schema": V.SCHEMA, "experiment": V.EXPERIMENT, "seed": 20, "arm": arm, "sleep": L, "mode": mode,
              "history": hist, "provenance": prov, "ppl_ratio": {"eval": 0.9, "modern": 0.9},
              "base_denominator_sha256": den, "rows": rows, "env": env}
        (d / name).write_text(json.dumps(ev))
    for L in range(1, 6):
        write(f"eval_N_s{L}.json", "N", L, "base", "observed", None)
        for a in V.ARMS:
            for mode in ("merged", "unmerged"):
                write(f"eval_{a}_s{L}_{mode}.json", a, L, mode, "observed", hx(a, L))
    for a in V.FORK_ARMS:
        for mode in ("merged", "unmerged"):
            write(f"eval_{a}_s4_withheld_{mode}.json", a, 4, mode, "withheld", hx(a, "w"))
    cli = lambda: subprocess.run([sys.executable, str(ROOT / "scripts/prov1_analyze.py"), "--phase", "B", "--dir",
                                  tmp_path / "b", "--support1-dir", tmp_path / "s1", "--seeds", "20", "--test-mode",
                                  "--out", tmp_path / "a.json"], capture_output=True, text=True)
    p = cli()
    res = json.loads((tmp_path / "a.json").read_text())
    assert res["verdict"] == "PROV1_ADVANCE", (p.stdout[-2000:], p.stderr[-2000:], res.get("errors"))
    assert all(v["ok"] for v in res["integrity"].values()) and len(res["integrity"]) == 3 * 5 + 2
    good_tr = (d / "prov1_train_seed20.json").read_text()
    for mut in (lambda t: t.update(seed=21), lambda t: t["provenance"].update(arms=["RP", "RC"]),
                lambda t: t["provenance"].update(fork=False), lambda t: t["fork"]["parent_full_state_hash"].update(RC=hx("x"))):
        t = json.loads(good_tr); mut(t); (d / "prov1_train_seed20.json").write_text(json.dumps(t)); cli()
        assert json.loads((tmp_path / "a.json").read_text())["verdict"] == "PROV1_INVALID"     # Sol: foreign train identity
    (d / "prov1_train_seed20.json").write_text(good_tr)
    f = d / "eval_RC_s3_unmerged.json"
    ev = json.loads(f.read_text()); ev["rows"][0]["outcome_truth"] = "bogus"; f.write_text(json.dumps(ev))
    cli()
    bad = json.loads((tmp_path / "a.json").read_text())
    assert bad["verdict"] == "PROV1_INVALID" and any("independent grading" in e for e in bad["errors"])
