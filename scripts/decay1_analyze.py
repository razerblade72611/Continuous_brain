"""DECAY-1 frozen analysis (written before any DECAY-1 data exists; Claude + Sol, 2026-10-27). Read-only, CPU only.
Development evidence on the BIND-I1 dev streams (seeds 2-5): a mechanistic follow-up, not independent confirmation.

Validity gates:
  V1 exactly seeds {2,3,4,5}, each once, complete; one commit and config; config = L lr 1e-4 KL 1.0 night 4; backbone =
     locked Qwen3-0.6B hash, unchanged; tokenizer-prefix preflight clean.                       (fail -> INVALID)
  V2 per seed, all five arms have identical step counts, schedule hash and RNG end state.         (fail -> INVALID)
  V3 life-1 acquisition (>= 3/4 targets FC in >= 3/4 seeds) and V3b bidirectional binding (>= 3/4 pairs in >= 3/4 seeds)
     at checkpoint 1.                                                    (V3 fail -> INVALID; V3b fail -> qualifier)
  R  reproduction of the archived BIND-I1 run: checkpoint 1 and arm A vs condition U at life 2, over the FULL probe
     record (generation, forced choice, every candidate/counter/treatment score, competence): numbers within 1e-3,
     everything else identical, memory hashes equal, every seed. Failure is a qualifier: the contrasts stay internally
     valid but are exploratory until the cause is investigated, and an R-failed result may not authorise an
     architectural change without a separate engineering review.
  T  arm E's teacher logits identical before and after the life, and the checkpoint-1 anchor measurably different
     from the frozen backbone (max per-token KL > 1e-6), every seed.             (fail -> the E-A contrast is INVALID)
  S  saved extension states reload to identical memory hash, optimiser state and RNG states, every seed. (qualifier)
  U  new-fact uptake in A, B, E at life 2: >= 3/4 U facts correct in >= 3/4 seeds. A retention gain in B (A-B) or E (E-A)
     counts as a stability-plasticity result only if that arm meets this; otherwise "improved retention with
     insufficient new-fact acquisition".
  V5 competence (perplexity ratio exp(L - L_base), eval and modern separately, checkpoint 1 and every arm): reported;
     mean <= 1.02 and max <= 1.05, else a qualifier.

Co-primary measures per seed (mean over the 4 targets), checkpoint 1 -> checkpoint 2:
  dV = l(V|A)(2) - l(V|A)(1)                                 accessibility of the true answer
  dB = [l(V|A) - l(Z|A)](2) - [l(V|A) - l(Z|A)](1)           entity-binding discrimination
Contrasts X-Y paired by seed: A-B, C-D, A-C, B-D, E-A (negative = X lost more).
  robust_neg: mean <= -2 and negative in >= 3 of 4 seeds;  robust_pos: mean >= +2 and positive in >= 3 of 4 seeds;
  otherwise unresolved (never "equivalent").
Frozen statements, evaluated separately for dV and dB:
  A-B robust_neg -> frozen-backbone KL adds forgetting during new-fact learning   (robust_pos -> it protects)
  C-D robust_neg -> frozen-backbone KL adds forgetting without new facts          (robust_pos -> it protects)
  B-D robust_neg -> new-fact learning causes forgetting without KL
  A-C robust_neg -> new-fact learning adds forgetting with KL
  C and D both mean dV <= -10 -> background cross-entropy / inherited optimiser dynamics implicated (dV only)
  E-A robust_pos -> the previous-self teacher preserves more: teacher choice matters
  E-A robust_neg -> the previous-self teacher preserves less
  E-A unresolved -> no general conclusion about previous-self consolidation
Also reported (descriptive): the factorial interaction (A-B)-(C-D) per seed and mean, for dV and dB; per-arm dV, dB, life-2 forced choice, uptake of the U facts in A/B/E, the gradient diagnostic,
token counts.

  python scripts/decay1_analyze.py results/dev3/decay1/seed*.json
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "D", "E")
EXPECTED_SEEDS = {2, 3, 4, 5}
EXPECTED_CONFIG = {"arm": "L", "lr": 1e-4, "kl_weight": 1.0, "night_passes": 4}
EXPECTED_BACKBONE = "c75424fb5370f0d514a05d7c640e6c2e1e4b504564e4d48cb2ddedb71609e0a0"
MONITORS = ("eval", "modern")
THRESH, MIN_SEEDS, REPRO_TOL = 2.0, 3, 1e-3
CONTRASTS = (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("E", "A"))


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def per_seed(r):
    c1 = r["checkpoint1"]["targets"]
    V1 = sum(x["lV_A"] for x in c1) / 4
    B1 = sum(x["lV_A"] - x["lZ_A"] for x in c1) / 4
    out = {"acq_cp1": sum(bool(x["fc"]) for x in c1),
           "binding_cp1": sum((x["lV_A"] > x["lZ_A"]) and (x["counter"]["lZ_C"] > x["counter"]["lV_C"]) for x in c1),
           "arms": {}}
    for arm in ARMS:
        t = r["arms_out"][arm]["targets"]
        out["arms"][arm] = {"dV": sum(x["lV_A"] for x in t) / 4 - V1,
                            "dB": sum(x["lV_A"] - x["lZ_A"] for x in t) / 4 - B1,
                            "fc2": sum(bool(x["fc"]) for x in t),
                            "binding2": sum((x["lV_A"] > x["lZ_A"]) and (x["counter"]["lZ_C"] > x["counter"]["lV_C"])
                                            for x in t),
                            "u_uptake": sum(bool(x["treat"]["U"]["fc"]) for x in t)}

    def ratio(pl, m):
        try:
            v = math.exp(pl[m] - r["base_ppl_loss"][m])
        except (KeyError, TypeError, OverflowError):
            return None
        return v if math.isfinite(v) and v > 0 else None
    comp = {"cp1": r["checkpoint1"].get("ppl_loss") or {}}
    comp.update({arm: r["arms_out"][arm].get("ppl_loss") or {} for arm in ARMS})
    out["competence"] = {k: {m: ratio(v, m) for m in MONITORS} for k, v in comp.items()}
    return out


def contrast(seeds, x, y, key):
    d = [s["arms"][x][key] - s["arms"][y][key] for s in seeds.values()]
    m = mean(d)
    neg, pos = sum(v < 0 for v in d), sum(v > 0 for v in d)
    return {"per_seed": [round(v, 3) for v in d], "mean": m,
            "robust_neg": m is not None and m <= -THRESH and neg >= MIN_SEEDS,
            "robust_pos": m is not None and m >= THRESH and pos >= MIN_SEEDS}


def validity(results, seeds):
    v = {}
    ids = [r.get("seed") for r in results]
    v["V1"] = (len(results) == 4 and sorted(ids) == sorted(EXPECTED_SEEDS) and len(set(ids)) == 4
               and all(r.get("status") == "complete" for r in results)
               and len({r["commit"] for r in results}) == 1
               and len({json.dumps(r["config"], sort_keys=True) for r in results}) == 1
               and all(r["config"].get(k) == x for r in results for k, x in EXPECTED_CONFIG.items())
               and all(r["backbone_start"] == r["backbone_end"] == r["backbone_sha256"] == EXPECTED_BACKBONE
                       for r in results)
               and all(r.get("prefix_check", {}).get("fails", 1) == 0 for r in results))
    v["V2"] = all(len({json.dumps(t, sort_keys=True) for t in r["traces"].values()}) == 1
                  and len(set(r["rng_after"].values())) == 1 and set(r["traces"]) == set(ARMS) for r in results)
    v["V3"] = sum(s["acq_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    v["V3b"] = sum(s["binding_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    rep = [r.get("reproduction") or {} for r in results]
    v["R"] = all(x.get(k, {}).get("max_abs_diff", 1) <= REPRO_TOL and x.get(k, {}).get("n_mismatches", 1) == 0
                 and x.get(k, {}).get("memory_hash_match") for x in rep for k in ("cp1", "A_vs_U2"))
    v["R_detail"] = rep
    e = [r["arms_out"]["E"] for r in results]
    v["T"] = (all(x.get("teacher_hash_start") and x.get("teacher_hash_start") == x.get("teacher_hash_end") for x in e)
              and all((r.get("anchor_difference") or {}).get("max", 0) > 1e-6 for r in results))
    v["S"] = all((r.get("state_restore_test") or {}).get("ok") for r in results)
    v["U_uptake"] = {a: sum(s["arms"][a]["u_uptake"] >= 3 for s in seeds.values()) >= MIN_SEEDS for a in ("A", "B", "E")}
    detail, ok = {}, len(seeds) == 4
    for m in MONITORS:
        for k in ("cp1",) + ARMS:
            xs = [s["competence"].get(k, {}).get(m) for s in seeds.values()]
            good = len(xs) == 4 and all(x is not None for x in xs)
            detail[f"{m}|{k}"] = {"mean": round(sum(xs) / 4, 5) if good else None, "max": max(xs) if good else None}
            ok = ok and good and sum(xs) / 4 <= 1.02 and max(xs) <= 1.05
    v["V5"], v["V5_detail"] = ok, detail
    return v


def statements(eff, seeds, key, v):
    out = []
    ab, cd, ac, bd, ea = (eff[f"{x}-{y}"] for x, y in CONTRASTS)
    if ab["robust_neg"]:
        out.append("frozen-backbone KL adds forgetting during new-fact learning"
                   + ("" if v["U_uptake"]["B"] else " - B: improved retention with insufficient new-fact acquisition"))
    if ab["robust_pos"]:
        out.append("frozen-backbone KL protects during new-fact learning")
    if cd["robust_neg"]:
        out.append("frozen-backbone KL adds forgetting without new facts")
    if cd["robust_pos"]:
        out.append("frozen-backbone KL protects without new facts")
    if bd["robust_neg"]:
        out.append("new-fact learning causes forgetting without KL")
    if ac["robust_neg"]:
        out.append("new-fact learning adds forgetting with KL")
    if key == "dV":
        c = mean([s["arms"]["C"]["dV"] for s in seeds.values()])
        d = mean([s["arms"]["D"]["dV"] for s in seeds.values()])
        if c is not None and d is not None and c <= -10 and d <= -10:
            out.append("background cross-entropy / inherited optimiser dynamics implicated (C and D both dV <= -10)")
    if not v["T"]:
        out.append("E-A INVALID (teacher logits changed, or the previous-self anchor is indistinguishable from frozen)")
    elif ea["robust_pos"]:
        out.append("previous-self teacher preserves more: teacher choice matters"
                   + ("" if v["U_uptake"]["E"] else " - improved retention with insufficient new-fact acquisition"))
    elif ea["robust_neg"]:
        out.append("previous-self teacher preserves less")
    else:
        out.append("E-A unresolved: no general conclusion about previous-self consolidation")
    return out


def analyse(results):
    seeds = {r["seed"]: per_seed(r) for r in results if r.get("status") == "complete"}
    v = validity(results, seeds)
    eff = {key: {f"{x}-{y}": contrast(seeds, x, y, key) for x, y in CONTRASTS} for key in ("dV", "dB")}
    inter = {}
    for key in ("dV", "dB"):                                   # descriptive factorial interaction (A-B) - (C-D)
        d = [(s["arms"]["A"][key] - s["arms"]["B"][key]) - (s["arms"]["C"][key] - s["arms"]["D"][key])
             for s in seeds.values()]
        inter[key] = {"per_seed": [round(x, 3) for x in d], "mean": mean(d)}
    if not (v["V1"] and v["V2"] and v["V3"]):
        verdict = {"status": "INVALID (V1/V2/V3 failed) - descriptive only"}
    else:
        verdict = {"status": "VALID", "dV": statements(eff["dV"], seeds, "dV", v),
                   "dB": statements(eff["dB"], seeds, "dB", v), "qualifiers": []}
        if not v["V3b"]:
            verdict["qualifiers"].append("INITIAL BINDING NOT ESTABLISHED")
        if not v["R"]:
            verdict["qualifiers"].append("REPRODUCTION OF BIND-I1 FAILED (contrasts internally valid)")
        if not v["V5"]:
            verdict["qualifiers"].append("COMPETENCE BAND EXCEEDED")
        if not v["S"]:
            verdict["qualifiers"].append("EXTENSION STATES NOT RESTORABLE")
        if not all(v["U_uptake"].values()):
            verdict["qualifiers"].append(f"new-fact uptake below standard in {[a for a, ok in v['U_uptake'].items() if not ok]}")
    return {"which": "DECAY-1 analysis", "validity": v, "effects": eff, "interaction": inter, "verdict": verdict,
            "seeds": seeds}


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/dev3/decay1").glob("seed[0-9].json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["seed"])
    if len({r["seed"] for r in results}) != len(results):
        sys.exit("duplicate seeds in the inputs - refusing to analyse")
    res = analyse(results)
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/decay1_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    v = res["validity"]
    print("validity:", {k: v[k] for k in ("V1", "V2", "V3", "V3b", "R", "T", "S", "U_uptake", "V5")})
    print("interaction (A-B)-(C-D):", json.dumps(res["interaction"]))
    print("reproduction:", json.dumps(v["R_detail"]))
    print(f"{'seed':>4} | " + " | ".join(f"{a}: dV dB fc bind" for a in ARMS))
    for s, sd in res["seeds"].items():
        print(f"{s:>4} | " + " | ".join(f"{a}: {x['dV']:+6.1f} {x['dB']:+6.1f} {x['fc2']}/4 {x['binding2']}/4"
                                         for a, x in sd["arms"].items()))
    for key in ("dV", "dB"):
        for k, e in res["effects"][key].items():
            print(f"{key} {k}: mean {e['mean']:+.2f} per seed {e['per_seed']} robust_neg={e['robust_neg']} "
                  f"robust_pos={e['robust_pos']}")
    print("VERDICT:", json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
