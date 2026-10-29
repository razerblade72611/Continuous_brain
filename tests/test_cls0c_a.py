"""CLS-0C Phase A diagnostic: pure parts, exercised on the real committed CLS-0B results (no models needed).
The full pipeline (cue rebuild -> analyze_life -> reproduction) is checked against the RECORDED K3 (TF-IDF) decisions,
which are deterministic and need no embedder: they must reproduce exactly at every checkpoint of every seed."""
import importlib.util
import json
import random
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


A = _load("cls0c_a_t", "scripts/cls0c_a.py")
B = _load("cls0b_t", "scripts/cls0b.py")
template = lambda attr, ent: B.template(lifegen, attr, ent)
fresh_name = lambda rng, used: B.fresh_name(rng, lifegen, used)


def seed_data(seed):
    d = json.loads((ROOT / f"results/cls0b_final/seed{seed}.json").read_text())
    eps = [{k: v for k, v in e.items() if k != "text_sha256"} for e in d["episodes"]]
    return d, eps


def test_rebuilt_cues_match_recorded_counts_and_k3_decisions_reproduce_exactly():
    for seed in (2, 3, 4, 5):
        d, eps = seed_data(seed)
        cues, cal, neg, temporal = B.build_cues(seed, eps, lifegen)
        assert d["n_cues"] == {"fact": len(cues), "calibration": len(cal), "test_negative": len(neg),
                               "temporal": len(temporal)}
        attrs = sorted({e["attr"] for e in eps if e["kind"] != "temporal"})
        for L in range(1, 6):
            bank = [e for e in eps if e["time"] <= L]
            texts = {e["eid"]: e["text"] for e in bank}
            eids = list(texts)
            cache = {}

            def sim(c, e_):
                if c not in cache:
                    cache[c] = dict(zip(eids, B.tfidf_sims([texts[x] for x in eids], [c])[0]))
                return cache[c][e_]
            lo = A.analyze_life(L, eps, cues, cal, neg, B.temporal_items(temporal, L), sim, B.threshold, attrs,
                                template)
            r = A.reproduction(d["checkpoints"][L - 1], lo, method="K3")
            assert r["agreement"] == 1.0 and r["threshold_abs_diff"] < 1e-9, (seed, L, r)
            # temporal routing reproduces too
            rec_t = [t["routed"] for t in d["checkpoints"][L - 1]["methods"]["K3"]["temporal"]]
            assert [t["routed"] for t in lo["temporal"]] == rec_t, (seed, L)


def test_near_miss_relations_and_attributes():
    d, eps = seed_data(3)
    cues, cal, neg, temporal = B.build_cues(3, eps, lifegen)
    attrs = sorted({e["attr"] for e in eps if e["kind"] != "temporal"})
    observed = {}
    for e in eps:
        if e["kind"] != "temporal":
            observed.setdefault(e["entity"], set()).add(e["attr"])
    for c in neg:
        a = A.cue_attr(c["cue"], c["entity"], attrs, template)
        if c["form"] == "near_miss":
            assert a not in observed[c["entity"]]
    # a similarity that only sees the entity name: every near-miss top-1 must be the same entity
    ent_of = {e["eid"]: e["entity"] for e in eps}
    sim = lambda c, e_: 1.0 if ent_of[e_] in c else 0.0
    lo = A.analyze_life(5, eps, cues, cal, neg, B.temporal_items(temporal, 5), sim, B.threshold, attrs, template)
    nm = [n for n in lo["negatives"] if n["form"] == "near_miss"]
    assert nm and all(n["top1_relation"] == "same_entity" for n in nm)


def test_probes_and_entity_dominance():
    d, eps = seed_data(4)
    probes = A.build_probes(4, eps, template, fresh_name)
    facts = [e for e in eps if e["kind"] != "temporal"]
    assert sum(p["kind"] == "entity_swap" for p in probes) == len(facts)
    assert probes == A.build_probes(4, eps, template, fresh_name)                     # deterministic
    used = {w.lower() for e in eps for w in __import__("re").findall(r"[A-Za-z]+", e["text"])}
    assert all(p["entity"].split()[0].lower() not in used for p in probes if p["kind"] == "entity_swap")
    info = {e["eid"]: e for e in facts}
    own = {e["eid"]: template(e["attr"], e["entity"]) for e in facts}
    # entity-only similarity -> entity swap costs everything, attribute swap nothing -> entity dominates
    s_ent = lambda c, e_: 1.0 if info[e_]["entity"] in c else 0.0
    q = A.probe_summary(probes, s_ent, own, 0.5)
    assert q["entity_dominates"] and q["frac_attr_swap_above_thr"] == 1.0 and q["frac_entity_swap_above_thr"] == 0.0
    # attribute-only similarity -> the reverse
    s_att = lambda c, e_: 1.0 if c == template(info[e_]["attr"], c.split("'")[0].split(" is")[0]) or \
        any(c == template(info[e_]["attr"], x) for x in [info[e_]["entity"]] + [p.get("entity", "") for p in probes]) \
        else 0.0
    q2 = A.probe_summary(probes, s_att, own, 0.5)
    assert not q2["entity_dominates"]


def test_sweep_and_auroc():
    recs = [{"form": "template", "top1_ok": True, "top1_sim": s} for s in (0.9, 0.8, 0.7)] + \
           [{"form": "paraphrase0", "top1_ok": True, "top1_sim": 0.75}] + \
           [{"form": "near_miss", "top1_sim": s} for s in (0.6, 0.65)] + [{"form": "unseen_person", "top1_sim": 0.3}]
    s = A.sweep(recs)
    assert s["feasible_template"] and s["feasible_template_and_paraphrase"]
    recs.append({"form": "near_miss", "top1_sim": 0.95})            # 1/3 near-miss above every template
    recs += [{"form": "near_miss", "top1_sim": 0.85}]               # 2/4 above the lowest template
    s = A.sweep(recs)
    assert not s["feasible_template"] and s["best_at_template_target"]["near_miss"] == 0.5
    assert A.auroc([2, 3], [1, 1]) == 1.0 and A.auroc([1], [1]) == 0.5 and A.auroc([0], [1]) == 0.0
    r = random.Random(0)
    assert 0.0 <= A.auroc([r.random() for _ in range(9)], [r.random() for _ in range(9)]) <= 1.0
