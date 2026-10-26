"""BIND-I1 frozen analysis (written before any I1 data exists; Claude + Sol, 2026-10-26). Read-only, CPU only.

Validity gates (checked first; a failed gate makes the run INVALID for interpretation, results still reported):
  V1 status complete for all four seeds, identical config/commit/backbone, backbone unchanged.
  V2 branch integrity: identical step counts and RNG end state across conditions (asserted by the runner, re-checked).
  V3 acquisition: at checkpoint 1, targets won forced choice in >= 3 of 4 targets in >= 3 of 4 seeds.
  V4 treatment acquisition at checkpoint 2, per condition: >= 3 of 4 treatment facts in >= 3 of 4 seeds.
     (A failure in S alone is reported as "B fails to acquire W" -> general learning failure, not interference.)
  V5 competence: eval/modern loss ratio vs base, mean <= 1.02 and every seed <= 1.05, at every checkpoint.

Primary outcome per target i: dDA = D_A(5) - D_A(1), D_A = l(V|A) - l(W|A) (summed log-prob, T_i template).
Per seed: condition mean over the 4 targets. Effects: E(S-X), E(S-U), E(X-U) = mean over seeds of per-seed differences.
"Worse" (frozen): mean effect <= -2.0 nats AND the per-seed difference is negative in >= 3 of 4 seeds.
"Similar": |mean effect| < 2.0 nats, or the per-seed signs split 2/2.
Interpretations (frozen, first match):
  1 S worse than X and S worse than U                      -> same-attribute associative interference
  2 S similar to X, and both S and X worse than U          -> value exposure alone is sufficient
  3 all three pairwise similar                            -> general drift (no condition-specific damage)
  4 anything else                                         -> unclassified; report descriptively
  Qualifier: if 1 holds and V4 holds for S (B learns W while A is damaged) -> stability/plasticity conflict.
  Qualifier: if V4 fails for S -> general learning failure (overrides 1-3).
Secondary (descriptive): life-5 forced-choice counts; dl(V|A), dl(W|A), dl(U|A) from checkpoint 1 to 5; margin_best
minus wiped margin_best; D_B = l(W|B) - l(V|B); treatment-fact margins; competence ratios; checkpoints 2 and 3.
Four seeds, five lives: development evidence, not a sealed test.

  python scripts/bind_i1_analyze.py results/dev3/bind_i1/seed*.json
Writes results/dev3/bind_i1_analysis.json and prints the tables.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("S", "X", "U")
WORSE, NSEEDS_SIGN = -2.0, 3


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def cp(res, cond, L):
    if L == 1:
        return res["checkpoint1"]
    return next(c for c in res["conditions"][cond] if c["after_life"] == L)


def DA(row):
    return row["lV_A"] - row["lW_A"]


def per_seed(res):
    c1 = res["checkpoint1"]["targets"]
    wiped = {r["i"]: r for r in res["wiped"]["targets"]}
    out = {"acq_cp1": sum(bool(r["fc"]) for r in c1), "conditions": {}}
    for cond in CONDITIONS:
        c5 = cp(res, cond, 5)["targets"]
        rows = []
        for a, b in zip(c1, c5):
            rows.append({"i": a["i"], "dDA": DA(b) - DA(a), "DA1": DA(a), "DA5": DA(b),
                         "dlV": b["lV_A"] - a["lV_A"], "dlW": b["lW_A"] - a["lW_A"], "dlU": b["lU_A"] - a["lU_A"],
                         "DB5": b["lW_B"] - b["lV_B"], "fc5": bool(b["fc"]),
                         "margin5_vs_wiped": b["margin_best"] - wiped[a["i"]]["margin_best"]})
        traj = {L: mean([DA(r) for r in cp(res, cond, L)["targets"]]) for L in (1, 2, 3, 5)}
        treat2 = [r["treat"] for r in cp(res, cond, 2)["targets"]]
        treat5 = [r["treat"] for r in cp(res, cond, 5)["targets"]]
        out["conditions"][cond] = {
            "dDA": mean([r["dDA"] for r in rows]), "DA_traj": traj,
            "dlV": mean([r["dlV"] for r in rows]), "dlW": mean([r["dlW"] for r in rows]),
            "dlU": mean([r["dlU"] for r in rows]), "DB5": mean([r["DB5"] for r in rows]),
            "fc5": sum(r["fc5"] for r in rows), "margin5_vs_wiped": mean([r["margin5_vs_wiped"] for r in rows]),
            "treat_fc2": sum(bool(t["fc"]) for t in treat2), "treat_fc5": sum(bool(t["fc"]) for t in treat5),
            "treat_margin2": mean([t["margin_best"] for t in treat2]),
            "treat_margin5": mean([t["margin_best"] for t in treat5]), "targets": rows}
    comp = {}
    for cond in CONDITIONS:
        for L in (2, 3, 5):
            pl = cp(res, cond, L).get("ppl_loss") or {}
            comp[f"{cond}{L}"] = {k: round(v / res["base_ppl_loss"][k], 4) for k, v in pl.items()}
    pl1 = res["checkpoint1"].get("ppl_loss") or {}
    comp["cp1"] = {k: round(v / res["base_ppl_loss"][k], 4) for k, v in pl1.items()}
    out["competence"] = comp
    return out


def effect(seeds, a, b):
    d = [s["conditions"][a]["dDA"] - s["conditions"][b]["dDA"] for s in seeds.values()]
    m = mean(d)
    neg = sum(x < 0 for x in d)
    pos = sum(x > 0 for x in d)
    worse = m is not None and m <= WORSE and neg >= NSEEDS_SIGN
    similar = m is not None and (abs(m) < abs(WORSE) or (neg == 2 and pos == 2))
    return {"per_seed": [round(x, 3) for x in d], "mean": m, "neg_seeds": neg, "worse": worse, "similar": similar}


def validity(results, seeds):
    v = {}
    v["V1"] = (len(results) == 4 and all(r.get("status") == "complete" for r in results)
               and len({r["commit"] for r in results}) == 1 and len({json.dumps(r["config"], sort_keys=True) for r in results}) == 1
               and all(r["backbone_start"] == r["backbone_end"] == r["backbone_sha256"] for r in results)
               and len({r["backbone_sha256"] for r in results}) == 1)
    v["V2"] = all(len({tuple(x) for x in r["steps"].values()}) == 1 and len(set(r["rng_after"].values())) == 1
                  for r in results)
    v["V3"] = sum(s["acq_cp1"] >= 3 for s in seeds.values()) >= 3
    v["V4"] = {c: sum(s["conditions"][c]["treat_fc2"] >= 3 for s in seeds.values()) >= 3 for c in CONDITIONS}
    ratios = [[x for k, x in s["competence"].items() for x in x.values()] for s in seeds.values()]
    flat = [x for r in ratios for x in r]
    v["V5"] = bool(flat) and mean(flat) <= 1.02 and max(flat) <= 1.05
    v["V5_detail"] = {"mean": mean(flat), "max": max(flat) if flat else None}
    return v


def interpret(eff, v):
    if not (v["V1"] and v["V2"] and v["V3"]):
        return "INVALID (validity gate V1/V2/V3 failed) - descriptive only"
    if not v["V4"]["S"]:
        return "general learning failure (B fails to acquire W in S)"
    sx, su, xu = eff["S-X"], eff["S-U"], eff["X-U"]
    if sx["worse"] and su["worse"]:
        out = "1 same-attribute associative interference; stability/plasticity conflict (B acquires W while A is damaged)"
    elif sx["similar"] and su["worse"] and xu["worse"]:
        out = "2 value exposure alone is sufficient"
    elif sx["similar"] and su["similar"] and xu["similar"]:
        out = "3 general drift (no condition-specific damage)"
    else:
        out = "4 unclassified - report descriptively"
    if not v["V5"]:
        out += " [competence band exceeded]"
    if not all(v["V4"].values()):
        out += f" [treatment acquisition failed in {[c for c, ok in v['V4'].items() if not ok]}]"
    return out


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/dev3/bind_i1").glob("seed*.json"))
    results = [json.loads(p.read_text()) for p in paths]
    results = sorted(results, key=lambda r: r["seed"])
    seeds = {r["seed"]: per_seed(r) for r in results if r.get("status") == "complete"}
    eff = {f"{a}-{b}": effect(seeds, a, b) for a, b in (("S", "X"), ("S", "U"), ("X", "U"))} if seeds else {}
    v = validity(results, seeds) if seeds else {}
    res = {"which": "BIND-I1 analysis", "validity": v, "effects_dDA": eff,
           "interpretation": interpret(eff, v) if seeds else "no data", "seeds": seeds}
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/bind_i1_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print("validity:", json.dumps(v, default=str))
    for s, sd in seeds.items():
        print(f"-- seed {s}: life-1 targets acquired {sd['acq_cp1']}/4")
        print(f"   {'cond':4} {'dDA':>7} {'DA 1/2/3/5':>26} {'dlV':>7} {'dlW':>7} {'dlU':>7} {'DB5':>7} fc5 "
              f"treat fc2/fc5")
        for c, x in sd["conditions"].items():
            tr = "/".join(f"{x['DA_traj'][L]:+.1f}" for L in (1, 2, 3, 5))
            print(f"   {c:4} {x['dDA']:+7.2f} {tr:>26} {x['dlV']:+7.2f} {x['dlW']:+7.2f} {x['dlU']:+7.2f} "
                  f"{x['DB5']:+7.2f} {x['fc5']}/4 {x['treat_fc2']}/4 {x['treat_fc5']}/4")
    for k, e in eff.items():
        print(f"E({k}) = {e['mean']:+.2f}  per seed {e['per_seed']}  worse={e['worse']} similar={e['similar']}")
    print("INTERPRETATION:", res["interpretation"])


if __name__ == "__main__":
    main()
