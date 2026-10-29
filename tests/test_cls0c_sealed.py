"""CLS-0C sealed harness on PUBLIC fixtures only (no sealed item exists when these tests are written)."""
import importlib.util
import json
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


H = _load("cls0c_sealed_t", "scripts/cls0c_sealed.py")


def fixture(n_worlds=1):
    """A public toy dataset in the author schema (invented content, not sealed material)."""
    worlds = []
    for k in range(n_worlds):
        w = f"F{k:02d}"
        xs = [{"xid": f"{w}-X1", "life": 1, "text": "Tamsin Orrel led the orchard survey as chief counter. She left a "
                                                    "blue lantern by the gate."},
              {"xid": f"{w}-X2", "life": 3, "text": "By the third winter Tamsin Orrel had become the ferry warden."},
              {"xid": f"{w}-X3", "life": 2, "text": "Dov Marsh said he might move to Ellery someday."}]
        items = []
        for i in range(10):
            items.append({"iid": f"{w}-S{i}", "label": "SUPPORTED", "stem": "During the orchard survey Tamsin Orrel "
                          "worked as", "answer": "chief counter", "xids": [f"{w}-X1"],
                          "evidence": "Tamsin Orrel led the orchard survey as chief counter.",
                          "question": "What was Tamsin Orrel's job during the orchard survey?"})
            items.append({"iid": f"{w}-I{i}", "label": "IMPLICATION", "stem": "Tamsin Orrel's favourite colour is",
                          "plausible": "blue", "why_not": "a lantern colour is not a preference"})
        for i in range(5):
            items.append({"iid": f"{w}-N{i}", "label": "NEAR-MISS", "stem": "Dov Marsh's sister is called"})
            items.append({"iid": f"{w}-U{i}", "label": "UNFAMILIAR", "stem": "Quill Haverly works as a"})
        for i, (stem, ans, x, cp) in enumerate([("At life 1, Tamsin Orrel's role was", "chief counter", "X1", 5),
                                               ("Tamsin Orrel's current role is", "ferry warden", "X2", 3),
                                               ("At life 3, Tamsin Orrel's role was", "ferry warden", "X2", 4),
                                               ("Tamsin Orrel's present duty is", "ferry warden", "X2", 5)]):
            items.append({"iid": f"{w}-T{i}", "label": "TEMPORAL", "stem": stem, "answer": ans, "xids": [f"{w}-{x}"],
                          "evidence": "", "checkpoint": cp})
        worlds.append({"world_id": w, "experiences": xs, "items": items})
    tpl = {a: {"supported": [f"Fixture wording {j} asks {{e}} {a} exactly:" .replace(":", "") for j in range(4)],
               "implication": [f"Fixture implied {j} of {{e}} about {a}" for j in range(3)]} for a in lifegen.TRAIN}
    return {"worlds": worlds, "attribute_templates": tpl}


def test_fixture_validates_and_defects_are_caught():
    ds = fixture(10)
    E, W = H.validate(ds, lifegen)
    assert E == [], E
    bad = json.loads(json.dumps(ds))
    bad["worlds"][0]["items"][0]["answer"] = "chief count"                 # not whole words
    bad["worlds"][0]["items"][1]["stem"] += "?"
    bad["worlds"][1]["items"].pop()                                       # wrong count
    bad["attribute_templates"]["city"]["supported"][0] = "{e} lives in"   # development wording
    bad["worlds"][2]["items"][-1]["checkpoint"] = 2                       # evidence from life 3 > checkpoint
    E, _ = H.validate(bad, lifegen)
    joined = " ".join(E)
    for frag in ("not verbatim", "unfinished sentence", "label counts", "duplicates a development wording",
                 "bad checkpoint"):
        assert frag in joined, (frag, E)


def test_dev_wordings_cover_every_development_probe():
    dev = H.dev_wordings(lifegen)
    for a in lifegen.TRAIN:
        for t in lifegen.PARAPHRASE[a]:
            assert H.B.norm(t.replace("{e}", "E")) in dev
        assert H.B.norm(lifegen.TRAIN[a][0].split("{v}")[0].replace("{e}", "E")) in dev


def test_blind_package_hides_labels_and_order():
    ds = fixture(10)
    pkg, mapping = H.blind(ds, "salt")
    txt = json.dumps(pkg)
    for leak in ("SUPPORTED", "IMPLICATION", "NEAR-MISS", "UNFAMILIAR", "TEMPORAL", "plausible", "why_not", "answer",
                 "evidence", "chief counter\"", "-S0", "-I0", "supported", "implication", "F00"):
        assert leak not in txt, leak
    st = pkg["worlds"][0]["statements"]
    assert len(st) == 34 and all(s["id"].startswith("S") and len(s) == 2 for s in st)
    assert [mapping[s["id"]] for s in st] != [it["iid"] for it in ds["worlds"][0]["items"]]   # shuffled
    assert len(pkg["templates"]) == 70 and all(set(t) == {"id", "attribute", "template"} for t in pkg["templates"])


def test_reconcile_flags_disagreements_and_ambiguity():
    ds = fixture(1)
    pkg, mapping = H.blind(ds, "salt")
    inv = {v: k for k, v in mapping.items() if isinstance(v, str)}
    author = {it["iid"]: it for it in ds["worlds"][0]["items"]}
    adj = {"statements": {}, "templates": {}}
    for iid, it in author.items():
        adj["statements"][inv[iid]] = {"label": it["label"], "answer": it.get("answer"),
                                       "refs": [inv[x] for x in it.get("xids", [])]}
    adj["statements"][inv["F00-I0"]] = {"label": "SUPPORTED", "answer": "blue", "refs": [inv["F00-X1"]]}
    adj["statements"][inv["F00-S1"]] = {"label": "AMBIGUOUS"}
    adj["statements"][inv["F00-T1"]] = {"label": "TEMPORAL", "answer": "chief counter", "refs": [inv["F00-X1"]]}
    for t in pkg["templates"]:
        m = mapping[t["id"]]
        adj["templates"][t["id"]] = "SUPPORTED" if m["kind"] == "supported" else "IMPLICATION"
    first = pkg["templates"][0]["id"]
    adj["templates"][first] = "AMBIGUOUS"
    ag, dis, miss = H.reconcile(ds, adj, mapping)
    ids = {d.get("iid") or d.get("template") for d in dis}
    assert {"F00-I0", "F00-S1", "F00-T1"} <= ids and len(dis) == 4 and miss == []
    assert ag["F00-S0"] == "SUPPORTED" and "F00-I0" not in ag


def test_seal_is_hash_consistent_and_applies_exclusions(tmp_path):
    ds = fixture(10)
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(ds))
    fl = tmp_path / "fl.json"
    fl.write_text(json.dumps({"exclude_items": ["F00-S1"], "exclude_templates": [
        "city|supported|" + ds["attribute_templates"]["city"]["supported"][0]]}))
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts/cls0c_sealed.py"), "seal", str(p), str(fl),
                        str(tmp_path / "sealed")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = (tmp_path / "sealed/sealed.json").read_bytes()
    assert (tmp_path / "sealed/SEALED.sha256").read_text().split()[0] == H.sha(s)
    d = json.loads(s)
    assert "F00-S1" not in {it["iid"] for it in d["worlds"][0]["items"]} and len(d["templates"]["city"]["supported"]) == 3


def test_synthetic_sampling_is_deterministic_balanced_and_blind():
    tpl = {a: {"supported": [f"S{j} {{e}} {a}" for j in range(4)],
               "implication": [f"I{j} {{e}} {a}" for j in range(3)]} for a in lifegen.TRAIN}
    tot = {"para_entailed": 0, "implication": 0}
    for seed in H.SEALED_SEEDS:
        eps, cal, items = H.synthetic_items(seed, lifegen, tpl)
        eps2, _, items2 = H.synthetic_items(seed, lifegen, tpl)
        assert [i["key"] for i in items] == [i["key"] for i in items2]
        for c in tot:
            tot[c] += sum(i["cat"] == c for i in items)
        assert sum(i["cat"] == "template" for i in items) == 50 and len(cal) == 40
        assert sum(i["cat"] in ("near_miss", "unfamiliar") for i in items) == 40
        dev_eids = {e["eid"] for s in (2, 3, 4, 5) for e in json.loads(
            (ROOT / f"results/cls0b_final/seed{s}.json").read_text())["episodes"]}
        assert not ({e["eid"] for e in eps} & dev_eids)                       # fresh streams
    assert tot == {"para_entailed": 200, "implication": 150}
    assert H.split_counts(150, 4) == [38, 38, 37, 37]


def test_gates_apply_per_format_without_averaging():
    summ = {"synthetic": {"template": 1.0, "para_entailed": 0.95, "implication_commit": 0.05, "near_unsupported": 0.0,
                          "unfamiliar_unsupported": 0.0, "temporal": 1.0, "provenance": 1.0},
            "narrative": {"narrative_supported": 0.95, "implication_commit": 0.30, "near_unsupported": 0.0,
                          "unfamiliar_unsupported": 0.0, "temporal": 0.9, "provenance": 1.0}}
    ok, rep = H.gate_report(summ)
    assert not ok and not rep["narrative:implication_commit"]["pass"] and rep["synthetic:implication_commit"]["pass"]
    summ["narrative"]["implication_commit"] = 0.1
    assert H.gate_report(summ)[0]
    rows = [{"fmt": "narrative", "cat": "implication", "answered": True, "correct": False, "provenance_ok": False},
            {"fmt": "narrative", "cat": "implication", "answered": False, "correct": False, "provenance_ok": False}]
    assert H.score_rows(rows)["narrative"]["implication_commit"] == 0.5
