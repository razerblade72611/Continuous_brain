"""CLS-0C Phase A: read-only diagnosis of K1 addressing on the four CLS-0B results (Claude + Sol, 2026-10-29).
Protocol: lab/CLS0C_A_PROTOCOL.md. Nothing is trained or written to the CLS-0B results; inputs are the committed
results/cls0b_final/seed{2..5}.json (episode texts + recorded decisions) and the same Qwen3-Embedding-0.6B.

Per seed and checkpoint (life 1-5; life 5 primary) it rebuilds the CLS-0B cues exactly (scripts/cls0b.py build_cues
from the recorded episodes), re-embeds episodes and cues with being/cls/keys.py, and records:
  R   reproduction: recomputed K1 thresholds and route/activation decisions vs the recorded CLS-0B result
  Q1  for every negative cue: top-5 candidates (eid, entity, attribute, time, similarity) and the relation of the
      top-1 candidate to the cue (same entity / same attribute / other)
  Q2  top-1 similarity distributions (template, paraphrase, near-miss, unseen, calibration) and AUROC
  Q3  threshold feasibility: is there ONE K1 threshold with template recall >= 0.95 and near-miss activation <= 0.10
      (also with paraphrase recall >= 0.90)? per seed and pooled across seeds
  Q4  entity-swap vs attribute-swap probes: similarity of each stored episode to its own template cue, to the same
      entity asked about an attribute never observed for it, and to an unseen person asked about the same attribute
  T   temporal: top-2 candidates and similarities for the life-1 / life-3 / latest probes

  python scripts/cls0c_a.py --results results/cls0b_final --external /workspace/hf/Qwen3-Embedding-0.6B \\
      --out results/cls0c_a
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (2, 3, 4, 5)
TARGET_TEMPLATE, TARGET_PARA, TARGET_NEAR, TARGET_UNSEEN = 0.95, 0.90, 0.10, 0.10   # frozen Q3 criteria
REPRO_DECISION_AGREEMENT, REPRO_THR_TOL = 0.99, 2e-3               # frozen reproduction gate (R)
REJECT_ALL = 2.0                     # finite reject-all threshold for the sweep (similarities are <= 1; JSON-safe)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- pure parts (unit-tested) -------------------------
def cue_attr(cue, entity, attrs, template):
    """The attribute a template cue asks about (negatives record only the entity)."""
    hits = [a for a in attrs if template(a, entity) == cue]
    assert len(hits) == 1, f"cue does not identify one attribute: {cue!r}"
    return hits[0]


def build_probes(seed, eps, template, fresh_name):
    """Q4 probes per stored fact (entity E, attribute A):
    attr_swap   - E asked about every attribute never observed for E (the near-miss construction)
    entity_swap - one unseen person asked about A"""
    facts = [e for e in eps if e["kind"] != "temporal"]
    attrs = sorted({e["attr"] for e in facts})
    observed = {}
    for e in facts:
        observed.setdefault(e["entity"], set()).add(e["attr"])
    used = {w.lower() for e in eps for w in re.findall(r"[A-Za-z]+", e["text"])}
    rng = random.Random(f"CLS0C-A-probes-{seed}")
    probes = []
    for e in facts:
        for a2 in attrs:
            if a2 not in observed[e["entity"]]:
                probes.append({"eid": e["eid"], "kind": "attr_swap", "cue": template(a2, e["entity"]), "attr": a2})
        ent = fresh_name(rng, used)
        probes.append({"eid": e["eid"], "kind": "entity_swap", "cue": template(e["attr"], ent), "entity": ent})
    return probes


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def sweep(records):
    """records: dicts {form, top1_ok (positives only), top1_sim}. For every candidate threshold t (activation: sim >= t)
    template/paraphrase recall = fraction of that form whose top-1 is the correct episode AND >= t; near/unseen =
    fraction activated. Returns the curve, Q3 feasibility, and the best near-miss rate attainable at template >= 0.95."""
    by = lambda f: [r for r in records if r["form"] == f]
    tm, pa = by("template"), [r for r in records if r["form"].startswith("paraphrase")]
    nm, un = by("near_miss"), by("unseen_person")
    ts = sorted({r["top1_sim"] for r in records}) + [REJECT_ALL]
    frac = lambda rs, f: sum(map(f, rs)) / len(rs) if rs else None
    curve = []
    for t in ts:
        curve.append({"t": t,
                      "template": frac(tm, lambda r: r["top1_ok"] and r["top1_sim"] >= t),
                      "paraphrase": frac(pa, lambda r: r["top1_ok"] and r["top1_sim"] >= t),
                      "near_miss": frac(nm, lambda r: r["top1_sim"] >= t),
                      "unseen": frac(un, lambda r: r["top1_sim"] >= t)})
    ok_t = [c for c in curve if c["template"] is not None and c["template"] >= TARGET_TEMPLATE]
    best = min(ok_t, key=lambda c: (c["near_miss"], -c["template"])) if ok_t else None
    feas = [c for c in ok_t if c["near_miss"] is not None and c["near_miss"] <= TARGET_NEAR]
    feas_p = [c for c in feas if c["paraphrase"] is not None and c["paraphrase"] >= TARGET_PARA]
    feas_full = [c for c in feas_p if c["unseen"] is None or c["unseen"] <= TARGET_UNSEEN]
    return {"feasible_template": bool(feas), "feasible_template_and_paraphrase": bool(feas_p),
            "feasible_full": bool(feas_full),
            "best_at_template_target": best, "example_feasible": (feas_full or feas_p or feas or [None])[0],
            "max_template_recall": max(c["template"] for c in curve) if tm else None,
            "curve": curve}


def analyze_life(life, eps, cues, cal, neg, temporal, sim, threshold, attrs, template, k=5):
    """One checkpoint. sim(cue_text, eid) -> float (K1 cosine). Mirrors CLS-0B: bank = episodes experienced by `life`,
    threshold from calibration negatives, top-1 routing, near-misses only once the person has been met."""
    bank = [e for e in eps if e["time"] <= life]
    eids = [e["eid"] for e in bank]
    info = {e["eid"]: e for e in bank}

    def top(c):
        r = [sim(c, x) for x in eids]
        o = sorted(range(len(eids)), key=lambda j: -r[j])
        return [(eids[j], r[j]) for j in o[:k]], max(r)
    thr = threshold([top(c["cue"])[1] for c in cal])
    cand = lambda t: [{"eid": e_, "entity": info[e_]["entity"], "attr": info[e_]["attr"], "time": info[e_]["time"],
                       "sim": s_} for e_, s_ in t]
    out = {"life": life, "threshold": thr, "positives": [], "negatives": [], "temporal": []}
    for c in cues:
        if c["time"] > life:
            continue
        t, _ = top(c["cue"])
        out["positives"].append({"eid": c["eid"], "form": c["form"], "top1": t[0][0], "top1_sim": t[0][1],
                                 "top1_ok": t[0][0] == c["eid"], "own_sim": sim(c["cue"], c["eid"]),
                                 "routed": t[0][0] if t[0][1] >= thr else None,
                                 "route_ok": t[0][0] == c["eid"] and t[0][1] >= thr})
    for c in neg:
        if c["form"] == "near_miss" and c["observed_since"] > life:
            continue
        t, _ = top(c["cue"])
        a = cue_attr(c["cue"], c["entity"], attrs, template)
        e1 = info[t[0][0]]
        rel = "same_entity" if e1["entity"] == c["entity"] else ("same_attribute" if e1["attr"] == a else "other")
        out["negatives"].append({"form": c["form"], "entity": c["entity"], "attr": a, "top1_sim": t[0][1],
                                 "activated": t[0][1] >= thr, "top1_relation": rel, "top5": cand(t)})
    for c in temporal:
        t, _ = top(c["cue"])
        out["temporal"].append({"form": c["form"], "top2": cand(t[:2]), "routed": [x for x, s in t[:2] if s >= thr]})
    return out


def probe_summary(probes, sim, own_cue, thr):
    """Q4 at the final checkpoint: similarity of each episode to its own template cue vs swapped cues."""
    rows = []
    for p in probes:
        s_own = sim(own_cue[p["eid"]], p["eid"])
        rows.append({**p, "sim": sim(p["cue"], p["eid"]), "own": s_own})
    by = lambda k: [r for r in rows if r["kind"] == k]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    at, es = by("attr_swap"), by("entity_swap")
    return {"n_attr_swap": len(at), "n_entity_swap": len(es),
            "mean_own": mean([r["own"] for r in es]),
            "mean_attr_swap": mean([r["sim"] for r in at]), "mean_entity_swap": mean([r["sim"] for r in es]),
            "mean_drop_attr_swap": mean([r["own"] - r["sim"] for r in at]),
            "mean_drop_entity_swap": mean([r["own"] - r["sim"] for r in es]),
            "frac_attr_swap_above_thr": mean([r["sim"] >= thr for r in at]),
            "frac_entity_swap_above_thr": mean([r["sim"] >= thr for r in es]),
            # relative sensitivity: supports (does not prove) entity dominance; the swaps change different tokens
            "entity_swap_drop_exceeds_attr_swap": (mean([r["own"] - r["sim"] for r in es]) or 0) >
                                                  (mean([r["own"] - r["sim"] for r in at]) or 0),
            "rows": rows}


def reproduction(recorded_cp, life_out, method="K1"):
    """R: recomputed vs recorded decisions of one retrieval method at one checkpoint."""
    rec = recorded_cp["methods"][method]
    thr_diff = abs(rec["threshold"] - life_out["threshold"])
    rf = [(f["eid"], f["form"], f["routed"]) for f in rec["facts"]]          # the routed episode id, None = declined
    cf = [(f["eid"], f["form"], f["routed"]) for f in life_out["positives"]]
    rn = [(n["form"], n["activated"]) for n in rec["negatives"]]
    cn = [(n["form"], n["activated"]) for n in life_out["negatives"]]
    rt = [(t["form"], t["routed"]) for t in rec["temporal"]]
    ct = [(t["form"], t["routed"]) for t in life_out["temporal"]]
    assert [x[:2] for x in rf] == [x[:2] for x in cf] and [x[0] for x in rn] == [x[0] for x in cn] and \
        [x[0] for x in rt] == [x[0] for x in ct], "cue set differs from the recorded CLS-0B result"
    agree = sum(a == b for a, b in zip(rf, cf)) + sum(a == b for a, b in zip(rn, cn)) + \
        sum(a == b for a, b in zip(rt, ct))
    n = len(rf) + len(rn) + len(rt)
    return {"threshold_recorded": rec["threshold"], "threshold_recomputed": life_out["threshold"],
            "threshold_abs_diff": thr_diff, "decisions": n, "agree": agree, "agreement": agree / n if n else None,
            "ok": thr_diff <= REPRO_THR_TOL and (agree / n if n else 1) >= REPRO_DECISION_AGREEMENT}


# ---------------------------------------------------------------- run ------------------------------------------------
def main():
    import torch
    from being.cls.keys import ExternalKeys
    from being.v3 import lifegen
    c0b = _load("cls0b_for_cls0c_a", "scripts/cls0b.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)   # CPU plumbing test with a tiny model
    a = ap.parse_args()
    rdir, out = ROOT / a.results, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    ek = ExternalKeys(a.external, "cuda" if cuda else "cpu", torch.bfloat16 if cuda else torch.float32)
    template = lambda attr, ent: c0b.template(lifegen, attr, ent)
    fresh_name = lambda rng, used: c0b.fresh_name(rng, lifegen, used)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {"which": "CLS-0C-A", "commit": commit, "gpu": torch.cuda.get_device_name(0) if cuda else None,
               "targets": {"template": TARGET_TEMPLATE, "paraphrase": TARGET_PARA, "near_miss": TARGET_NEAR,
                           "unseen": TARGET_UNSEEN},
               "inputs": {}, "seeds": {}}
    pooled = []
    for seed in a.seeds:
        p = rdir / f"seed{seed}.json"
        summary["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = json.loads(p.read_text())
        assert d["status"] == "complete" and d["seed"] == seed
        if d["external_manifest"]["files"] != c0b_manifest(a.external)["files"] and not a.test_mode:
            sys.exit(f"seed {seed}: embedder files differ from the ones CLS-0B used")
        eps = [{k: v for k, v in e.items() if k != "text_sha256"} for e in d["episodes"]]
        for e, r in zip(eps, d["episodes"]):
            assert hashlib.sha256(e["text"].encode()).hexdigest() == r["text_sha256"]
        stored = [e for e in eps if d["writes"].get(e["eid"], {}).get("ok")]
        cues, cal, neg, temporal = c0b.build_cues(seed, eps, lifegen)
        assert d["n_cues"] == {"fact": len(cues), "calibration": len(cal), "test_negative": len(neg),
                               "temporal": len(temporal)}, "rebuilt cue counts differ from CLS-0B"
        attrs = sorted({e["attr"] for e in eps if e["kind"] != "temporal"})
        probes = build_probes(seed, eps, template, fresh_name)
        texts = [e["text"] for e in stored]
        qtexts = sorted({c["cue"] for c in cues + cal + neg + temporal + probes})
        ev = dict(zip([e["eid"] for e in stored], ek.embed(texts)))
        qv = dict(zip(qtexts, ek.embed(qtexts, query=True)))
        sim = lambda c, e_: float(qv[c] @ ev[e_])
        lives, repro = [], []
        for L in range(1, 6):
            lo = analyze_life(L, stored, cues, cal, neg, c0b.temporal_items(temporal, L), sim, c0b.threshold, attrs,
                              template)
            repro.append({"life": L, **reproduction(d["checkpoints"][L - 1], lo)})
            lives.append(lo)
        last = lives[-1]
        own_cue = {c["eid"]: c["cue"] for c in cues if c["form"] == "template"}
        q4 = probe_summary(probes, sim, own_cue, last["threshold"])
        recs = last["positives"] + last["negatives"]
        nm = [n for n in last["negatives"] if n["form"] == "near_miss"]
        nma = [n for n in nm if n["activated"]]
        rel = lambda xs: {k: sum(n["top1_relation"] == k for n in xs) for k in ("same_entity", "same_attribute", "other")}
        s = {"reproduction": repro, "reproduction_ok": all(r["ok"] for r in repro),
             "q1_near_miss_top1_relation_all": rel(nm), "q1_near_miss_top1_relation_activated": rel(nma),
             "q1_n_near_miss": len(nm), "q1_n_activated": len(nma),
             "q2_auroc_template_vs_near_miss": auroc([r["top1_sim"] for r in last["positives"]
                                                      if r["form"] == "template" and r["top1_ok"]],
                                                     [n["top1_sim"] for n in nm]),
             "q3": {k: v for k, v in sweep(recs).items() if k != "curve"},
             "q4": {k: v for k, v in q4.items() if k != "rows"}}
        summary["seeds"][seed] = s
        pooled += recs
        (out / f"seed{seed}.json").write_text(json.dumps({"seed": seed, "lives": lives, "probes": q4["rows"],
                                                          "sweep_life5": sweep(recs)["curve"], "summary": s},
                                                         indent=1, allow_nan=False))
        print(f"seed {seed}: reproduction_ok={s['reproduction_ok']} near-miss activated {len(nma)}/{len(nm)} "
              f"{s['q1_near_miss_top1_relation_activated']} AUROC {s['q2_auroc_template_vs_near_miss']:.3f} "
              f"Q3 narrow/para/full {s['q3']['feasible_template']}/{s['q3']['feasible_template_and_paraphrase']}/"
              f"{s['q3']['feasible_full']} entity-swap drop > attr-swap drop "
              f"{s['q4']['entity_swap_drop_exceeds_attr_swap']}", flush=True)
    ps = sweep(pooled)
    summary["pooled_q3"] = {k: v for k, v in ps.items() if k != "curve"}
    summary["test_mode"] = a.test_mode
    summary["valid"] = all(s["reproduction_ok"] for s in summary["seeds"].values()) and not a.test_mode
    summary["q3_decision"] = ("NOT DECIDED: the diagnostic did not reproduce CLS-0B (or ran in test mode)"
                              if not summary["valid"] else
                              "single global K1 threshold FULLY feasible (template, paraphrase, near-miss, unseen): "
                              "threshold-only calibration remains a serious Phase B candidate (development data only)"
                              if ps["feasible_full"] else
                              "single global K1 threshold feasible for template + near-miss only, NOT for the full "
                              "requirements: reported, not treated as a solution"
                              if ps["feasible_template"] else
                              "no single K1 threshold meets template >= 0.95 with near-miss <= 0.10: global-threshold "
                              "tuning is eliminated as the primary remedy")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False))
    print("CLS-0C-A", json.dumps({k: summary[k] for k in ("valid", "q3_decision", "pooled_q3")}, indent=1))
    print("CLS-0C-A DONE valid=%s" % summary["valid"])
    sys.exit(0 if summary["valid"] else 3)


def c0b_manifest(d):
    """Same file manifest as CLS-0B records (weights/config/tokenizer files)."""
    d = Path(d)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix in (".safetensors", ".json", ".txt"))
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()
    return {"files": {p.name: sha(p) for p in files}}


if __name__ == "__main__":
    main()
