"""CONS-0A: pure-logic tests + an end-to-end CPU run of every stage on a tiny random Qwen3 (scores meaningless)."""
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("cons0a", ROOT / "scripts/cons0a.py")
C = importlib.util.module_from_spec(spec)
sys.modules["cons0a"] = C
spec.loader.exec_module(C)
from being.v3 import lifegen  # noqa: E402


def test_stems_and_paraphrases():
    for a in lifegen.TRAIN:
        s = C.template_stem(lifegen, a, "Ann Bo")
        assert "{" not in s and "Ann Bo" in s and not s.endswith(".")
        for k, p in C.paraphrase_stems(lifegen, a, "Ann Bo"):
            assert C.B.AUDIT[(a, k)] == "entailed" and "Ann Bo" in p
    assert C.paraphrase_stems(lifegen, "allergy", "X") == []          # no entailed allergy paraphrase (audit)


def test_contract():
    assert C.first_line(" Paris.\nStatement: x") == "Paris"
    g = C.grade("the Leston", "Leston")
    assert g["correct"] and not g["unknown"]
    assert C.grade("UNKNOWN", "Leston")["unknown"] and not C.grade("UNKNOWN", "Leston")["correct"]
    assert C.grade("", "Leston")["empty"]
    for ok in ("UNKNOWN", "Unknown.", "unknown!", " UNKNOWN "):                    # exact abstention only (Sol)
        assert C.is_unknown(ok)
    for bad in ("UNKNOWN because I never met them", "UNKNOWN: Leston", "unknown, maybe Leston", "Leston"):
        assert not C.is_unknown(bad)
        assert not C.grade(bad, "Pell")["unknown"] and not C.grade(bad, "Pell")["empty"]   # counts as an answer
    p = C.frame(C.statement("Ann lives in"))
    assert p.endswith("Statement: Ann lives in ...\nAnswer:") and "UNKNOWN" in p


def test_qa_filters():
    qa = C.parse_qa("Q: Where does Ann Bo live? | A: Leston.\njunk\nQ: Who? | A: Ann Bo")
    assert qa == [("Where does Ann Bo live?", "Leston"), ("Who?", "Ann Bo")]
    assert C.span_ok("Leston", "Ann Bo lives in Leston.") and not C.span_ok("Lest", "Ann Bo lives in Leston.")
    assert not C.span_ok("a b c d e", "a b c d e")
    w = C.eval_wordings(lifegen)
    assert C.leaks("The name of the city where Ann Bo resides is", "Ann Bo", w)   # a held-out paraphrase verbatim
    assert not C.leaks("Which town did Ann Bo grow pears in?", "Ann Bo", w)


def _eps():
    C0B = C.C0B
    i1 = C._load("bind_i1_t", "scripts/bind_i1.py")
    return C0B.build_episodes(2, lifegen, i1, None)


def test_negatives_follow_checkpoint():
    eps, st = _eps()
    for L in (1, 3, 5):
        seen = {(e["entity"], e["attr"]) for e in eps if e["time"] <= L}
        ents = {e["entity"] for e in eps if e["time"] <= L}
        nm = C.negatives(eps, L, 2, lifegen)
        assert nm and all(x not in seen and x[0] in ents for x in nm)
    names = {e["entity"] for e in eps}
    assert all(n not in names for n, _ in C.unfamiliar(eps, 2, lifegen))
    P = C.probe_set(eps, st, 5, 2, lifegen)
    assert {p["kind"] for p in P} == {"template", "paraphrase", "nearmiss", "unfamiliar", "temporal"}
    assert not any(p["kind"] == "temporal" for p in C.probe_set(eps, st, 2, 2, lifegen))
    it = C.intrusion_items(st, lifegen)
    assert len(it) == 4 and all(x["V"] != x["W"] != x["Z"] for x in it)


def test_schedule_budget_matching():
    tl = lambda s: len(s.split())
    chunk = lambda i, n: [7] * n
    new = [{"text": f"new {i} a b", "loss_tail": None, "life": 3} for i in range(5)]
    old = [{"text": f"old {i} " + "x " * i, "loss_tail": 2 if i % 2 else None, "life": 1 + i % 2} for i in range(7)]
    I, S, exp_ = C.schedule(new, old, 4, random.Random(0), tl, chunk)
    assert len(I) == len(S) == 40
    assert sum(1 for x in I if x["text"] and x["text"].startswith("new")) == \
        sum(1 for x in S if x["text"] and x["text"].startswith("new")) == 20
    i_slot = [x for x in I if not (x["text"] or "").startswith("new")]
    s_slot = [x for x in S if x.get("background")]
    assert sorted(tl(x["text"]) for x in i_slot) == sorted(len(x["ids"]) for x in s_slot)       # identical lengths
    assert sorted(str(x["loss_tail"]) for x in i_slot) == sorted(str(x["loss_tail"]) for x in s_slot)
    assert sum(exp_.values()) == 20 and max(exp_.values()) - min(exp_.values()) <= 3          # balanced cycling
    I1, S1, e1 = C.schedule(new, [], 2, random.Random(0), tl, chunk)
    assert e1 == {} and [(x["text"], x.get("ids")) for x in I1] == [(x["text"], x.get("ids")) for x in S1]


def _summ(new_para, life1=0.5, conf=0.0):
    return {"new_para": new_para, "life1_para": life1, "new_tmpl": 1.0, "nearmiss_conf": conf, "unfamiliar_conf": 0}


def _per_seed(i_new=0.8, n_new=0.1, i_l1=0.7, s_l1=0.2, conf=0.0, ppl=1.0):
    ps = {}
    for s in (2, 3, 4, 5):
        d = {"N": {L: _summ(n_new) for L in range(1, 6)}}
        for arm in ("I-V", "S-V", "I-QA", "S-QA"):
            l1 = i_l1 if arm.startswith("I") else s_l1
            d[arm] = {L: _summ(i_new, l1, conf) for L in range(1, 6)}
            d[arm]["ppl"] = {L: {"eval": ppl, "modern": ppl} for L in range(1, 6)}
        ps[s] = d
    return ps


def test_criteria():
    crit, adv = C.criteria(_per_seed())
    assert adv == "I-V" and all(crit["I-V"][k] for k in ("A1", "A2", "A3", "A4", "A5"))
    assert C.criteria(_per_seed(i_new=0.45))[1] is None                       # A1
    assert C.criteria(_per_seed(i_new=0.55, n_new=0.3))[1] is None            # A2 gain < 0.30
    assert C.criteria(_per_seed(i_l1=0.2, s_l1=0.2))[1] is None               # A3 no retention advantage
    assert C.criteria(_per_seed(conf=0.11))[1] is None                        # A4
    assert C.criteria(_per_seed(ppl=1.03))[1] is None                         # A5
    ps = _per_seed()
    ps[2]["I-V"][3]["nearmiss_conf"] = None
    assert not C.criteria(ps)[0]["I-V"]["A4"]                                 # missing measure never passes


def test_merge_check():
    row = lambda i, ans: {"kind": "template", "eid": f"e{i}", "item": f"Statement: s{i} ...", "answer": ans}
    a = {"rows": [row(i, "x") for i in range(100)], "summary": {"new_para": 0.5, "life1_para": 0.5},
         "ppl_ratio": {"eval": 1.000, "modern": 1.000}}
    b = {"rows": [row(i, "x" if i < 97 else "y") for i in range(100)], "summary": {"new_para": 0.51, "life1_para": 0.5},
         "ppl_ratio": {"eval": 1.005, "modern": 0.999}}
    assert C.merge_check(a, b)["ok"]
    b2 = dict(b, rows=[row(0, "z")] + b["rows"][1:])
    assert not C.merge_check(a, b2)["ok"]                                       # 96% agreement
    assert not C.merge_check(a, dict(b, ppl_ratio={"eval": 1.02, "modern": 1.0}))["ok"]       # competence differs
    assert not C.merge_check(a, dict(b, rows=b["rows"][:99] + [dict(row(99, "y"), item="other")]))["ok"]  # probes


def test_replay_errors_and_budget():
    eps = [{"eid": f"e{i}", "text": f"Ann {i} lives in Leston."} for i in range(4)]
    good = {e["eid"]: e["text"] for e in eps}
    rec = dict(good)
    assert C.replay_errors(good, eps, rec, True, True) == []
    assert C.replay_errors({"e0": eps[0]["text"]}, eps, rec, False, False) == []            # partial bank, test mode
    assert any("lacks" in x for x in C.replay_errors({"e0": eps[0]["text"]}, eps, rec, True, False))
    assert any("differ" in x for x in C.replay_errors(dict(good, e2="Ann 2 lives in Paris."), eps, rec, True, True))
    assert any("not exact" in x for x in C.replay_errors(good, eps, dict(rec, e1="Ann 1 lives in"), True, True))
    assert any("not in the stream" in x for x in C.replay_errors(dict(good, zz="x"), eps, rec, False, False))
    items = [{"ids": [1, 2, 3], "loss_tail": None}, {"ids": [4, 5, 6, 7], "loss_tail": 2}]
    assert C.budget(items) == (2, 7, 5)


@pytest.mark.slow
def test_end_to_end_tiny(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import tiny_hf_model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from being.cls.episodic import EpisodicMemory
    model_dir = tiny_hf_model.build(tmp_path / "m")
    eps, st, _ = C._stream(2, model_dir, test_mode=True)          # the stream exactly as every stage builds it
    cfg = {"layer": 1, "key_layer": 1, "r": 16, "init_mask": 64, "max_epochs": 300, "check_every": 50, "retries": 2}
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    mem = EpisodicMemory(m, tok, seed=2, **(dict(C.C0B.CFG) | cfg))
    e0 = eps[0]
    mem.write(e0["eid"], "The quick brown fox works as a Pemyiasnind by trade.", e0["time"], e0["source"])  # tiny model: short text
    mem.episodes[e0["eid"]].text = e0["text"]                 # the bank records the stream text (as CLS-0B's did)
    mem.save(tmp_path / "bank.pt")
    mem.episodes[e0["eid"]].text = e0["text"] + " Tampered."
    mem.save(tmp_path / "bank_bad.pt")
    py = sys.executable
    run = lambda *x: subprocess.run([py, str(ROOT / "scripts/cons0a.py"), *map(str, x)], check=True,
                                    capture_output=True, text=True, timeout=1800)
    stk = ["--stream-tokenizer", model_dir]
    bad = subprocess.run([py, str(ROOT / "scripts/cons0a.py"), "prep", "--seed", "2", "--bank", str(tmp_path / "bank_bad.pt"),
                          "--model", str(model_dir), "--out", str(tmp_path / "bad.json"), "--test-mode", "--test-cfg",
                          json.dumps(cfg), *map(str, stk)], capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "PREP FAILED" in (bad.stderr + bad.stdout)        # wrong history fails closed
    run("prep", "--seed", 2, "--bank", tmp_path / "bank.pt", "--model", model_dir, "--out", tmp_path / "rep.json",
        "--test-mode", "--test-cfg", json.dumps(cfg), *stk)
    R = json.loads((tmp_path / "rep.json").read_text())
    assert R["n"] == len(eps) and sum(e["recon"] is not None for e in R["episodes"]) == 1
    run("qa", "--replay", tmp_path / "rep.json", "--model", model_dir, "--out", tmp_path / "qa.json")
    Q = json.loads((tmp_path / "qa.json").read_text())
    assert set(Q["rejects"]) == {"span", "answer_in_question", "leak", "entailment", "cap"}
    # the tiny model produces no usable recon/QA: give run a stream-text replay and a hand-made QA set (plumbing only)
    for e in R["episodes"]:
        e["recon"] = e["stream_text"]
    (tmp_path / "rep2.json").write_text(json.dumps(R))
    Q["accepted"] = {e["eid"]: [{"q": f"What about {e['entity']}?", "a": e["value"]}] for e in R["episodes"]}
    (tmp_path / "qa2.json").write_text(json.dumps(Q))
    out = tmp_path / "run"
    # QA supervision boundary: the loss tail is exactly the answer's ids, at the end of the sequence
    items = C._items(R, Q, "QA", tok, life=1)
    for x in items[:5]:
        assert tok.decode(x["ids"][-x["loss_tail"]:]) == " " + x["answer"]
        assert x["ids"][:-x["loss_tail"]] == tok(C.frame(C.question(x["q"])), add_special_tokens=False).input_ids
    r = run("run", "--replay", tmp_path / "rep2.json", "--qa", tmp_path / "qa2.json", "--model", model_dir,
            "--passes", 1, "--out", out, "--test-mode", *stk)
    assert "RUN seed 2 DONE" in r.stdout
    res = json.loads((out / "run_seed2.json").read_text())
    assert set(res["arms"]) == {"S-V", "I-V", "S-QA", "I-QA"}
    for arm in res["arms"]:
        tr = res["arms"][arm]["train"]
        assert set(tr) == {"1", "2", "3", "4", "5"}
    for L in ("2", "3", "4", "5"):                            # budget matching S vs I, per family
        for fam in ("V", "QA"):
            a, b = res["arms"][f"S-{fam}"]["train"][L], res["arms"][f"I-{fam}"]["train"][L]
            assert a["items"] == b["items"] and a["updates"] == b["updates"]
            assert a["tokens"] == b["tokens"] and a["loss_tokens"] == b["loss_tokens"] and a["budget"] == b["budget"]
        assert res["arms"]["I-V"]["train"][L]["exposure"] and not res["arms"]["S-V"]["train"][L]["exposure"]
    assert res["arms"]["S-V"]["eval"]["1"] == res["arms"]["I-V"]["eval"]["1"]       # shared sleep 1
    # non-trivial: the trained cortex is non-zero and changes the model's outputs relative to the base N
    st5 = torch.load(out / "cortex_I-V_s5.pt")
    assert sum(float(B_.abs().sum()) for _, B_ in st5.values()) > 0
    nV = [x["V"] for x in res["N"]["5"]["intrusion"]]
    for arm in ("I-V", "S-V", "I-QA"):
        aV = [x["V"] for x in res["arms"][arm]["eval"]["5"]["intrusion"]]
        assert max(abs(x - y) for x, y in zip(aV, nV)) > 1e-3, arm
    assert res["arms"]["I-V"]["eval"]["5"]["intrusion"] != res["arms"]["S-V"]["eval"]["5"]["intrusion"]
    for arm in ("S-V", "I-V", "S-QA", "I-QA"):
        assert (out / f"cortex_{arm}_s5.pt").exists()
        for K, lora in ((1, out / f"cortex_{arm.split('-')[1]}_shared_s1.pt"), (5, out / f"cortex_{arm}_s5.pt")):
            run("fresh", "--replay", tmp_path / "rep2.json", "--model", model_dir, "--lora", lora, "--sleep", K,
                "--out", out / f"fresh_seed2_{arm}_s{K}.json", "--test-mode", *stk)
    r = run("analyze", out, "--out", tmp_path / "an.json")
    an = json.loads((tmp_path / "an.json").read_text())
    assert an["status"].startswith("CONS0A_") and set(an["criteria"]) == {"I-V", "I-QA"}
    assert all("agree" in v for v in an["merge_checks"].values())
    # merged fresh evaluation reproduces the in-process LoRA evaluation (fp32 tiny model: near exact)
    assert all(v["agree"] >= 0.97 for v in an["merge_checks"].values()), an["merge_checks"]
    r = run("calibrate", "--model", model_dir, "--out", tmp_path / "cal.json", "--test-mode", *stk)
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["chosen_E"] in C.PASSES and set(cal["passes"]) == {"4", "8", "16"}
    del torch
