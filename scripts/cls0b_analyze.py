"""CLS-0B frozen analysis (written before any CLS-0B data; Claude + Sol, 2026-10-28). Read-only, CPU only.
Development evidence (seeds 2-5 were used throughout BIND-I1/DECAY-1/WRITE-1). A two-seed run is labelled PILOT and is
operational evidence only; the scientific reading needs all four seeds.

Validity (any failure -> INVALID): every seed complete, not test-mode, design_check == "match", the frozen config, one
commit, the same model files, zero tokenizer-prefix failures.

Measured per method m in {K1, K2, K3, K4} at each checkpoint, pooled over seeds (per-seed values also reported):
  coverage            exact-reconstructable stored episodes / experienced episodes (failures and censored writes
                      count in the denominator); reconstruction exactness of stored episodes
  addressing          route accuracy on template cues and on paraphrase cues (declined or wrong = miss);
                      false activation on test negatives (unseen people; near-miss = familiar person, unobserved attribute)
  answers             forced-choice accuracy under iRAG (primary), raw-text and K1 direct; references none and oracle
  binding             per seed, target/counter pairs with BOTH answers correct (iRAG, template)
  retention by age    life-1 fact accuracy (iRAG, template) at checkpoints 1..5
  failure split       none retrieved | wrong episode | right episode wrong answer (template cues)
  temporal            life-1, life-3 and latest questions (top-2 retrieval), reported separately, never in the primary
  cost                write seconds/episode, masks, params/token, epochs/s
Frozen decision (K1 primary, life-5 checkpoint, pooled over seeds; statements in this order, the first is the direction):
  1 coverage < 0.95                                         -> storage failures: reconsider episode representation,
                                                               masking policy or model scale
  2 template route accuracy < 0.90 or false activation > 0.15 -> addressing becomes the priority
  3 iRAG template accuracy < 0.80                          -> reconstruction-to-reasoning transfer becomes the priority
  4 raw-text accuracy >= iRAG accuracy - 0.02               -> raw-text episodic memory matches the adapters: prefer
                                                               direct episodic storage for fast memory; neural
                                                               consolidation becomes a separate research track
  5 otherwise                                               -> continue toward online consolidation
Qualifiers (always evaluated):
  paraphrase iRAG accuracy < template iRAG accuracy - 0.20  -> strong only on templates: require paraphrase/semantic
                                                               generalisation before claiming progress
  temporal: life-1 and life-3 questions not both correct in >= 3/4 seeds -> add explicit temporal addressing before
                                                               consolidation
  K2 vs K1 and K3 vs K1 differences in route accuracy are reported as descriptive comparisons; K4 must be worse than
  K1 by >= 0.20 in route accuracy, otherwise "K1 addressing not distinguishable from a permuted control" is added.

  python scripts/cls0b_analyze.py results/cls0b/seed*.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("K1", "K2", "K3", "K4")
FROZEN_CFG = {"layer": 10, "key_layer": 24, "r": 2, "init_mask": 78, "growth": 1.2, "retries": 10, "lr": 0.05,
              "max_epochs": 1000, "check_every": 100}


def rate(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def per_checkpoint(r, cp):
    exper = cp["experienced"]
    writes = r["writes"]
    eids_upto = [e["eid"] for e in r["episodes"] if e["time"] <= cp["after_life"]]
    stored_exact = [e for e in eids_upto if writes.get(e, {}).get("ok") and r["reconstruction_exact"].get(e)]
    out = {"coverage": len(stored_exact) / exper if exper else None, "methods": {}}
    refs = cp["references"]
    out["none_acc"] = rate(x["none"]["correct"] for x in refs if x["form"] == "template")
    out["oracle_acc"] = rate(x["oracle"]["correct"] for x in refs if x["form"] == "template")
    for m in METHODS:
        mm = cp["methods"].get(m)
        if not mm:
            continue
        t = [x for x in mm["facts"] if x["form"] == "template"]
        p = [x for x in mm["facts"] if x["form"] != "template"]
        res = {"route_template": rate(x["route_ok"] for x in t), "route_paraphrase": rate(x["route_ok"] for x in p),
               "false_activation": rate(n["activated"] for n in mm["negatives"]),
               "false_activation_near_miss": rate(n["activated"] for n in mm["negatives"] if n["form"] == "near_miss"),
               "irag_template": rate(x["irag"]["correct"] for x in t), "irag_paraphrase": rate(x["irag"]["correct"] for x in p),
               "raw_template": rate(x["raw"]["correct"] for x in t), "raw_paraphrase": rate(x["raw"]["correct"] for x in p),
               "life1_irag": rate(x["irag"]["correct"] for x in t if x["time"] == 1),
               "fail_none": rate(x["routed"] is None for x in t),
               "fail_wrong_episode": rate(x["routed"] is not None and not x["route_ok"] for x in t),
               "fail_right_episode_wrong_answer": rate(x["route_ok"] and not x["irag"]["correct"] for x in t)}
        if m == "K1":
            res["direct_template"] = rate(x.get("direct", {}).get("correct", False) for x in t if x["route_ok"])
        tg = {x["eid"]: x["irag"]["correct"] for x in t if x["kind"] in ("target", "counter")}
        pairs = [(f"I{r['seed']}L01T{i}", f"I{r['seed']}L01C{i}") for i in range(4)]
        res["binding_pairs"] = sum(tg.get(a_, False) and tg.get(c_, False) for a_, c_ in pairs)
        res["temporal"] = {x["form"]: x["correct"] for x in mm["temporal"]}
        out["methods"][m] = res
    return out


def pooled(results, L, m, key):
    """Pool a rate over seeds by re-counting from the rows (not averaging rates)."""
    num = den = 0
    for r in results:
        cp = next(c for c in r["checkpoints"] if c["after_life"] == L)
        mm = cp["methods"].get(m)
        if not mm:
            continue
        t = [x for x in mm["facts"] if x["form"] == "template"]
        p = [x for x in mm["facts"] if x["form"] != "template"]
        rows = {"route_template": [x["route_ok"] for x in t], "irag_template": [x["irag"]["correct"] for x in t],
                "raw_template": [x["raw"]["correct"] for x in t], "irag_paraphrase": [x["irag"]["correct"] for x in p],
                "false_activation": [n["activated"] for n in mm["negatives"]]}[key]
        num += sum(rows)
        den += len(rows)
    return num / den if den else None


def validity(results):
    v = {"complete": all(r.get("status") == "complete" for r in results),
         "not_test": all(not r.get("test_mode") for r in results),
         "design": all(r.get("design_check") == "match" for r in results),
         "config": all(r.get("config") == FROZEN_CFG for r in results),
         "one_commit": len({r.get("commit") for r in results}) == 1,
         "model_files": len({json.dumps(r.get("model_files"), sort_keys=True) for r in results}) == 1,
         "prefix": all(r.get("prefix_failures", 1) == 0 for r in results),
         "seeds_unique": len({r["seed"] for r in results}) == len(results)}
    v["valid"] = all(v.values())
    return v


def decide(results):
    L = 5
    cov_num = sum(sum(1 for e in r["episodes"] if r["writes"].get(e["eid"], {}).get("ok")
                      and r["reconstruction_exact"].get(e["eid"])) for r in results)
    cov_den = sum(len(r["episodes"]) for r in results)
    k = {key: pooled(results, L, "K1", key) for key in ("route_template", "irag_template", "raw_template",
                                                        "irag_paraphrase", "false_activation")}
    k["coverage"] = cov_num / cov_den
    statements = []
    if k["coverage"] < 0.95:
        statements.append("1 storage failures: reconsider episode representation, masking policy or model scale")
    if k["route_template"] < 0.90 or k["false_activation"] > 0.15:
        statements.append("2 addressing becomes the priority")
    if k["irag_template"] < 0.80:
        statements.append("3 reconstruction-to-reasoning transfer becomes the priority")
    if k["raw_template"] >= k["irag_template"] - 0.02:
        statements.append("4 raw-text episodic memory matches the adapters: prefer direct episodic storage for fast "
                          "memory; neural consolidation becomes a separate research track")
    if not statements:
        statements.append("5 continue toward online consolidation")
    qual = []
    if k["irag_paraphrase"] < k["irag_template"] - 0.20:
        qual.append("strong only on templates: require paraphrase/semantic generalisation before claiming progress")
    temporal_ok = 0
    for r in results:
        cp = next(c for c in r["checkpoints"] if c["after_life"] == L)
        tm = {x["form"]: x["correct"] for x in cp["methods"]["K1"]["temporal"]}
        temporal_ok += bool(tm.get("life1") and tm.get("life3"))
    if temporal_ok < max(1, round(0.75 * len(results))):
        qual.append("add explicit temporal addressing before consolidation")
    k4 = pooled(results, L, "K4", "route_template")
    if k4 is not None and k["route_template"] - k4 < 0.20:
        qual.append("K1 addressing not distinguishable from a permuted control")
    return {"direction": statements[0], "statements": statements, "qualifiers": qual, "k1_life5": k,
            "k2_route_template": pooled(results, L, "K2", "route_template"),
            "k3_route_template": pooled(results, L, "K3", "route_template"), "k4_route_template": k4}


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/cls0b").glob("seed[0-9].json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["seed"])
    v = validity(results)
    pilot = sorted(r["seed"] for r in results) != [2, 3, 4, 5]
    per = {r["seed"]: {cp["after_life"]: per_checkpoint(r, cp) for cp in r["checkpoints"]} for r in results}
    cost = {r["seed"]: {"write_seconds": r.get("write_seconds"), "throughput": r.get("throughput"),
                        "masks": [w.get("mask") for w in r["writes"].values() if w.get("ok")],
                        "failed_writes": [k for k, w in r["writes"].items() if not w.get("ok")]} for r in results}
    out = {"which": "CLS-0B analysis", "pilot": pilot, "validity": v, "per_seed": per, "cost": cost}
    out["decision"] = decide(results) if v["valid"] else {"direction": "INVALID - descriptive only"}
    if pilot:
        out["decision"]["label"] = "PILOT (operational evidence only; scientific reading requires seeds 2-5)"
    (ROOT / "results/cls0b").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/cls0b/analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print("validity:", v)
    for s, cps in per.items():
        print(f"-- seed {s}  cost: {cost[s]['write_seconds']} s, masks {sorted(set(cost[s]['masks']))}, "
              f"failed {cost[s]['failed_writes']}")
        for L, c in cps.items():
            k1 = c["methods"].get("K1") or {}
            print(f"   life {L}: coverage {c['coverage']:.2f}  none {c['none_acc']}  oracle {c['oracle_acc']}  "
                  f"K1 route {k1.get('route_template')} irag {k1.get('irag_template')} raw {k1.get('raw_template')} "
                  f"para {k1.get('irag_paraphrase')} falseact {k1.get('false_activation')} life1 {k1.get('life1_irag')} "
                  f"binding {k1.get('binding_pairs')}/4")
    print("DECISION:", json.dumps(out["decision"], indent=1, default=str))


if __name__ == "__main__":
    main()
