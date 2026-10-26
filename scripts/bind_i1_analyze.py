"""BIND-I1 frozen analysis (written before any I1 data exists; Claude + Sol, 2026-10-26; revised after Sol's review of
b7c87a8, still before data). Read-only, CPU only. Four seeds, five lives: development evidence, not a sealed test.

Validity gates (checked first):
  V1 status complete for all four seeds; identical commit, config and backbone; backbone unchanged.
  V2 branch integrity: identical step counts AND identical background/teacher text sequences per life across S/X/U,
     identical RNG end state.                                                  (V1 or V2 fail -> INVALID)
  V3 target acquisition: at checkpoint 1, >= 3 of 4 targets win forced choice, in >= 3 of 4 seeds.   (fail -> INVALID)
  V3b initial binding: at checkpoint 1, >= 3 of 4 target/counter pairs correct in BOTH directions
     (l(V|A) > l(Z|A) and l(Z|C) > l(V|C)), in >= 3 of 4 seeds. Fail -> verdict carries
     "INITIAL BINDING NOT ESTABLISHED": differences may reflect attribute-level interference only.
  V4 treatment uptake per condition, at checkpoint 2, under the attribute it was taught with, over the same pool
     (named values + neutrals): >= 3 of 4 correct, in >= 3 of 4 seeds. Any failing condition makes every contrast
     involving it INCONCLUSIVE DUE TO INADEQUATE TREATMENT UPTAKE.
  V5 competence: eval/modern loss ratio vs base, mean <= 1.02 and max <= 1.05 over all checkpoints; reported on the verdict.

Primary outcome per target i: dDA = D_A(5) - D_A(1), D_A = l(V|A) - l(W|A). Per seed: mean over the 4 targets
(correlated measurements, not replications). Effects: E(S-X), E(S-U), E(X-U) = mean over seeds of per-seed differences.
  robust-worse: mean <= -2.0 nats AND negative in >= 3 of 4 seeds.
  small:        |mean| < 2.0 nats.  (Neither establishes equivalence with four seeds.)
Interpretation (frozen, first match):
  1 S robust-worse than X and than U                       -> same-attribute interference supported
  2 S and X robust-worse than U, and S-X small             -> value-exposure mechanism compatible; attribute specificity
                                                              unresolved
  3 no pairwise robust-worse difference                     -> no differential effect established; called general drift
                                                              ONLY if U's own dDA is robust-worse than 0 (mean <= -2,
                                                              negative in >= 3/4 seeds)
  4 otherwise                                               -> mixed / unclassified
Secondary (frozen, descriptive): D_A^VZ = l(V|A) - l(Z|A) trajectory; per-target life-5 pattern:
  intrusion-with-binding (V > Z but W > V) | deterioration (V below both Z and W) | survives (V above Z and W) | other;
life-5 forced choice; dl(V|A), dl(Z|A), dl(W|A), dl(U|A) from checkpoint 1; wiped and checkpoint-1 scores of W and U
under A; margin vs wiped; counter (C) binding over time; B transfer scores; treatment margins; training tokens;
checkpoints 2 and 3; competence.

  python scripts/bind_i1_analyze.py results/dev3/bind_i1/seed*.json
Writes results/dev3/bind_i1_analysis.json and prints the tables.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("S", "X", "U")
WORSE, MIN_SEEDS = -2.0, 3


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def cp(res, cond, L):
    if L == 1:
        return res["checkpoint1"]
    return next(c for c in res["conditions"][cond] if c["after_life"] == L)


def DA(r):
    return r["lV_A"] - r["lW_A"]


def DAVZ(r):
    return r["lV_A"] - r["lZ_A"]


def pattern(r):
    v, z, w = r["lV_A"], r["lZ_A"], r["lW_A"]
    if v > z and v > w:
        return "survives"
    if v > z and w > v:
        return "intrusion_with_binding"
    if v < z and v < w:
        return "deterioration"
    return "other"


def per_seed(res):
    c1 = res["checkpoint1"]["targets"]
    wiped = {r["i"]: r for r in res["wiped"]["targets"]}
    bind = sum((r["lV_A"] > r["lZ_A"]) and (r["counter"]["lZ_C"] > r["counter"]["lV_C"]) for r in c1)
    out = {"acq_cp1": sum(bool(r["fc"]) for r in c1), "binding_cp1": bind,
           "counter_fc_cp1": sum(bool(r["counter"]["fc"]) for r in c1), "conditions": {},
           "wiped_vs_cp1": [{"i": r["i"], "lW_A_wiped": wiped[r["i"]]["lW_A"], "lU_A_wiped": wiped[r["i"]]["lU_A"],
                             "lW_A_cp1": r["lW_A"], "lU_A_cp1": r["lU_A"]} for r in c1],
           "train_tokens": res.get("train_tokens")}
    for cond in CONDITIONS:
        c5 = cp(res, cond, 5)["targets"]
        rows = []
        for a, b in zip(c1, c5):
            rows.append({"i": a["i"], "dDA": DA(b) - DA(a), "DA1": DA(a), "DA5": DA(b), "DAVZ1": DAVZ(a), "DAVZ5": DAVZ(b),
                         "dlV": b["lV_A"] - a["lV_A"], "dlZ": b["lZ_A"] - a["lZ_A"], "dlW": b["lW_A"] - a["lW_A"],
                         "dlU": b["lU_A"] - a["lU_A"], "fc5": bool(b["fc"]), "pattern5": pattern(b),
                         "counter_bound5": b["counter"]["lZ_C"] > b["counter"]["lV_C"],
                         "B_transfer5": {k: b[k] for k in ("lV_B", "lZ_B", "lW_B", "lU_B")},
                         "margin5_vs_wiped": b["margin_best"] - wiped[a["i"]]["margin_best"]})
        traj = {L: mean([DA(r) for r in cp(res, cond, L)["targets"]]) for L in (1, 2, 3, 5)}
        trajvz = {L: mean([DAVZ(r) for r in cp(res, cond, L)["targets"]]) for L in (1, 2, 3, 5)}
        treat = {L: [r["treat"][cond] for r in cp(res, cond, L)["targets"]] for L in (2, 3, 5)}
        pats = {}
        for r in rows:
            pats[r["pattern5"]] = pats.get(r["pattern5"], 0) + 1
        out["conditions"][cond] = {
            "dDA": mean([r["dDA"] for r in rows]), "DA_traj": traj, "DAVZ_traj": trajvz,
            **{k: mean([r[k] for r in rows]) for k in ("dlV", "dlZ", "dlW", "dlU", "margin5_vs_wiped")},
            "fc5": sum(r["fc5"] for r in rows), "patterns5": pats,
            "counter_bound5": sum(r["counter_bound5"] for r in rows),
            "treat_fc": {L: sum(bool(t["fc"]) for t in ts) for L, ts in treat.items()},
            "treat_margin": {L: mean([t["margin_best"] for t in ts]) for L, ts in treat.items()},
            "targets": rows}
    comp = {"cp1": res["checkpoint1"].get("ppl_loss") or {}}
    for cond in CONDITIONS:
        for L in (2, 3, 5):
            comp[f"{cond}{L}"] = cp(res, cond, L).get("ppl_loss") or {}
    out["competence"] = {k: {m: round(v / res["base_ppl_loss"][m], 4) for m, v in x.items()} for k, x in comp.items()}
    return out


def robust(diffs):
    m = mean(diffs)
    neg = sum(x < 0 for x in diffs)
    return {"per_seed": [round(x, 3) for x in diffs], "mean": m, "neg_seeds": neg,
            "robust_worse": m is not None and m <= WORSE and neg >= MIN_SEEDS,
            "small": m is not None and abs(m) < abs(WORSE)}


def effect(seeds, a, b):
    return robust([s["conditions"][a]["dDA"] - s["conditions"][b]["dDA"] for s in seeds.values()])


def validity(results, seeds):
    v = {}
    v["V1"] = (len(results) == 4 and all(r.get("status") == "complete" for r in results)
               and len({r["commit"] for r in results}) == 1
               and len({json.dumps(r["config"], sort_keys=True) for r in results}) == 1
               and all(r["backbone_start"] == r["backbone_end"] == r["backbone_sha256"] for r in results)
               and len({r["backbone_sha256"] for r in results}) == 1)
    v["V2"] = all(len({json.dumps(t, sort_keys=True) for t in r["traces"].values()}) == 1
                  and len(set(r["rng_after"].values())) == 1 for r in results)
    v["V3"] = sum(s["acq_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    v["V3b"] = sum(s["binding_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    v["V4"] = {c: sum(s["conditions"][c]["treat_fc"][2] >= 3 for s in seeds.values()) >= MIN_SEEDS for c in CONDITIONS}
    flat = [x for s in seeds.values() for k in s["competence"].values() for x in k.values()]
    v["V5"] = bool(flat) and mean(flat) <= 1.02 and max(flat) <= 1.05
    v["V5_detail"] = {"mean": mean(flat), "max": max(flat) if flat else None}
    return v


def interpret(eff, seeds, v):
    if not (v["V1"] and v["V2"] and v["V3"]):
        return "INVALID (validity gate V1/V2/V3 failed) - descriptive only"
    failed = [c for c, ok in v["V4"].items() if not ok]
    if failed:
        out = f"INCONCLUSIVE DUE TO INADEQUATE TREATMENT UPTAKE in {failed}"
    else:
        sx, su, xu = eff["S-X"], eff["S-U"], eff["X-U"]
        if sx["robust_worse"] and su["robust_worse"]:
            out = "1 same-attribute interference supported"
        elif su["robust_worse"] and xu["robust_worse"] and sx["small"]:
            out = "2 value-exposure mechanism compatible; attribute specificity unresolved"
        elif not (sx["robust_worse"] or su["robust_worse"] or xu["robust_worse"]):
            u = robust([s["conditions"]["U"]["dDA"] for s in seeds.values()])
            out = ("3 no differential effect established; general drift (U itself declines)" if u["robust_worse"]
                   else "3 no differential effect established; no detectable impairment in U")
        else:
            out = "4 mixed / unclassified - report descriptively"
    if not v["V3b"]:
        out += " [INITIAL BINDING NOT ESTABLISHED: attribute-level interference only]"
    out += " [competence within band]" if v["V5"] else " [COMPETENCE BAND EXCEEDED]"
    return out


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/dev3/bind_i1").glob("seed*.json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["seed"])
    seeds = {r["seed"]: per_seed(r) for r in results if r.get("status") == "complete"}
    if not seeds:
        print("no complete results")
        return
    eff = {f"{a}-{b}": effect(seeds, a, b) for a, b in (("S", "X"), ("S", "U"), ("X", "U"))}
    v = validity(results, seeds)
    res = {"which": "BIND-I1 analysis", "validity": v, "effects_dDA": eff,
           "interpretation": interpret(eff, seeds, v), "seeds": seeds}
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/bind_i1_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print("validity:", json.dumps(v, default=str))
    for s, sd in seeds.items():
        print(f"-- seed {s}: life-1 targets {sd['acq_cp1']}/4, counters {sd['counter_fc_cp1']}/4, "
              f"bidirectional binding {sd['binding_cp1']}/4")
        print(f"   {'cond':4} {'dDA':>7} {'D_A(VW) 1/2/3/5':>26} {'D_A(VZ) 1/2/3/5':>26} {'dlV':>6} {'dlW':>6} "
              f"fc5 treat2 patterns@5")
        for c, x in sd["conditions"].items():
            tr = "/".join(f"{x['DA_traj'][L]:+.1f}" for L in (1, 2, 3, 5))
            tz = "/".join(f"{x['DAVZ_traj'][L]:+.1f}" for L in (1, 2, 3, 5))
            print(f"   {c:4} {x['dDA']:+7.2f} {tr:>26} {tz:>26} {x['dlV']:+6.1f} {x['dlW']:+6.1f} "
                  f"{x['fc5']}/4 {x['treat_fc'][2]}/4  {x['patterns5']}")
    for k, e in eff.items():
        print(f"E({k}) = {e['mean']:+.2f}  per seed {e['per_seed']}  robust_worse={e['robust_worse']} small={e['small']}")
    print("INTERPRETATION:", res["interpretation"])


if __name__ == "__main__":
    main()
