"""WRITE-1 frozen analysis (written before any WRITE-1 data exists; Claude + Sol, 2026-10-27). Read-only, CPU only.
Development evidence on the BIND-I1 dev streams (seeds 2-5); one continuation life.

Validity gates (any failure -> INVALID, descriptive only):
  V1 exactly seeds {2,3,4,5}, complete, one commit/config, config = L lr 1e-4 KL 1.0 night 4, locked backbone unchanged,
     tokenizer preflight clean.
  V2 per seed, A/R/P have identical step counts, schedule hash and RNG end state; equal trainable-parameter counts.
  V3 life-1 acquisition (>= 3/4 targets FC in >= 3/4 seeds).
  R  exact reproduction, every seed: checkpoint 1 vs BIND-I1; arm A vs BIND-I1 condition U life 2; arm A vs DECAY-1
     arm A (full probe records within 1e-3, zero non-numeric mismatches, equal memory hashes).
  I  P integrity, every seed: the new adapter starts at zero output (combined readout before life 2 == checkpoint 1)
     and the old-only readout after life 2 == checkpoint 1 (the life-1 adapter is untouched).
  V3b (initial binding) and V5 (competence of each arm's primary readout) are qualifiers.

A seed QUALIFIES as preserved for an arm when, at life 2, old-target forced choice >= 3/4 AND bidirectional binding
>= 3/4 in that same seed. Preserved: >= 3 of 4 seeds qualify. Learned: U-fact uptake >= 3/4 in >= 3 of 4 seeds.
Primary classification (P, combined readout, no routing):
  preserved & learned         -> "isolated incremental memory works for one life"
  not preserved & learned     -> "frozen old knowledge is disrupted by combining adapters: readout interference demonstrated"
  preserved & not learned     -> "stability purchased at the expense of plasticity"
  not preserved & not learned -> "combined mechanism fails both requirements; cause unresolved"
Contrasts paired by seed on dV = d l(V|A) and dB = d [l(V|A) - l(Z|A)] (checkpoint 1 -> 2): P-A, R-A, P-R;
robust_pos/neg = |mean| >= 2 with the same sign in >= 3/4 seeds, else unresolved.
  R-A robust_pos (dV) and P-R not robust_pos -> "optimizer reset deserves investigation".
Secondary (descriptive): counter-entity (C -> Z) forced choice, A and R classified the same way, P new-only readout,
value intrusion d l(U|A).
Pre-registered qualitative predictions (not decision rules): Claude - P combined beats A modestly but fails
preservation; Sol - isolation protects the old adapter internally but the combined output stays vulnerable.

  python scripts/write1_analyze.py results/dev3/write1/seed*.json
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "R", "P")
EXPECTED_SEEDS = {2, 3, 4, 5}
EXPECTED_CONFIG = {"arm": "L", "lr": 1e-4, "kl_weight": 1.0, "night_passes": 4}
EXPECTED_BACKBONE = "c75424fb5370f0d514a05d7c640e6c2e1e4b504564e4d48cb2ddedb71609e0a0"
MONITORS = ("eval", "modern")
THRESH, MIN_SEEDS, TOL = 2.0, 3, 1e-3
CONTRASTS = (("P", "A"), ("R", "A"), ("P", "R"))


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def exact(rep):
    return bool(rep) and rep.get("max_abs_diff", 1) <= TOL and rep.get("n_mismatches", 1) == 0 \
        and rep.get("memory_hash_match", False)


def summarise(cp, c1):
    t = cp["targets"]
    return {"dV": sum(x["lV_A"] for x in t) / 4 - sum(x["lV_A"] for x in c1) / 4,
            "dB": sum(x["lV_A"] - x["lZ_A"] for x in t) / 4 - sum(x["lV_A"] - x["lZ_A"] for x in c1) / 4,
            "dU": sum(x["lU_A"] for x in t) / 4 - sum(x["lU_A"] for x in c1) / 4,
            "fc2": sum(bool(x["fc"]) for x in t),
            "binding2": sum((x["lV_A"] > x["lZ_A"]) and (x["counter"]["lZ_C"] > x["counter"]["lV_C"]) for x in t),
            "counter_fc2": sum(bool(x["counter"]["fc"]) for x in t),
            "uptake": sum(bool(x["treat"]["U"]["fc"]) for x in t)}


def per_seed(r):
    c1 = r["checkpoint1"]["targets"]
    out = {"acq_cp1": sum(bool(x["fc"]) for x in c1),
           "binding_cp1": sum((x["lV_A"] > x["lZ_A"]) and (x["counter"]["lZ_C"] > x["counter"]["lV_C"]) for x in c1),
           "arms": {a: summarise(r["arms_out"][a], c1) for a in ARMS},
           "P_new_only": summarise(r["P_readouts"]["new"], c1)}
    for a, x in out["arms"].items():
        x["qualifies_preserved"] = x["fc2"] >= 3 and x["binding2"] >= 3

    def ratio(pl, m):
        try:
            v = math.exp(pl[m] - r["base_ppl_loss"][m])
        except (KeyError, TypeError, OverflowError):
            return None
        return v if math.isfinite(v) and v > 0 else None
    comp = {"cp1": r["checkpoint1"].get("ppl_loss") or {}}
    comp.update({a: r["arms_out"][a].get("ppl_loss") or {} for a in ARMS})
    out["competence"] = {k: {m: ratio(v, m) for m in MONITORS} for k, v in comp.items()}
    return out


def contrast(seeds, x, y, key):
    d = [s["arms"][x][key] - s["arms"][y][key] for s in seeds.values()]
    m = mean(d)
    return {"per_seed": [round(v, 3) for v in d], "mean": m,
            "robust_neg": m is not None and m <= -THRESH and sum(v < 0 for v in d) >= MIN_SEEDS,
            "robust_pos": m is not None and m >= THRESH and sum(v > 0 for v in d) >= MIN_SEEDS}


def classify(seeds, arm):
    preserved = sum(s["arms"][arm]["qualifies_preserved"] for s in seeds.values()) >= MIN_SEEDS
    learned = sum(s["arms"][arm]["uptake"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    if arm == "P":                                             # mechanism-specific wording applies only to P
        label = {(True, True): "isolated incremental memory works for one life",
                 (False, True): "frozen old knowledge is disrupted by combining adapters: readout interference demonstrated",
                 (True, False): "stability purchased at the expense of plasticity",
                 (False, False): "combined mechanism fails both requirements; cause unresolved"}[(preserved, learned)]
    else:
        label = (f"old memory {'preserved' if preserved else 'not preserved'}; "
                 f"new facts {'learned' if learned else 'not learned'}")
    return {"preserved": preserved, "learned": learned, "label": label}


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
                  and len(set(r["rng_after"].values())) == 1 and set(r["traces"]) == set(ARMS)
                  and len(set(r["trainable_params"].values())) == 1 for r in results)
    v["V3"] = sum(s["acq_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    v["V3b"] = sum(s["binding_cp1"] >= 3 for s in seeds.values()) >= MIN_SEEDS
    v["R"] = all(exact(r["reproduction"].get(k)) for r in results for k in ("cp1", "A_vs_U2", "A_vs_DECAY1_A"))
    v["I"] = all(exact(r.get("isolation_integrity")) and (r.get("P_zero_init_check") or {}).get("n_mismatches", 1) == 0
                 and (r.get("P_zero_init_check") or {}).get("max_abs_diff", 1) <= TOL for r in results)
    ok, detail = len(seeds) == 4, {}
    for m in MONITORS:
        for k in ("cp1",) + ARMS:
            xs = [s["competence"].get(k, {}).get(m) for s in seeds.values()]
            good = len(xs) == 4 and all(x is not None for x in xs)
            detail[f"{m}|{k}"] = {"mean": round(sum(xs) / 4, 5) if good else None, "max": max(xs) if good else None}
            ok = ok and good and sum(xs) / 4 <= 1.02 and max(xs) <= 1.05
    v["V5"], v["V5_detail"] = ok, detail
    return v


def analyse(results):
    seeds = {r["seed"]: per_seed(r) for r in results if r.get("status") == "complete"}
    v = validity(results, seeds)
    eff = {key: {f"{x}-{y}": contrast(seeds, x, y, key) for x, y in CONTRASTS} for key in ("dV", "dB")}
    cls = {a: classify(seeds, a) for a in ARMS}
    if not all(v[k] for k in ("V1", "V2", "V3", "R", "I")):
        verdict = {"status": "INVALID (V1/V2/V3/R/I failed) - descriptive only"}
    else:
        notes = []
        if eff["dV"]["R-A"]["robust_pos"] and not eff["dV"]["P-R"]["robust_pos"]:
            notes.append("optimizer reset deserves investigation")
        q = []
        if not v["V3b"]:
            q.append("INITIAL BINDING NOT ESTABLISHED")
        if not v["V5"]:
            q.append("COMPETENCE BAND EXCEEDED")
        verdict = {"status": "VALID", "P_primary": cls["P"]["label"], "notes": notes, "qualifiers": q}
    return {"which": "WRITE-1 analysis", "validity": v, "effects": eff, "classification": cls, "verdict": verdict,
            "seeds": seeds}


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/dev3/write1").glob("seed[0-9].json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["seed"])
    if len({r["seed"] for r in results}) != len(results):
        sys.exit("duplicate seeds in the inputs - refusing to analyse")
    res = analyse(results)
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/write1_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    v = res["validity"]
    print("validity:", {k: v[k] for k in ("V1", "V2", "V3", "V3b", "R", "I", "V5")})
    print(f"{'seed':>4} | " + " | ".join(f"{a}: dV dB dU fc bind cfc upt" for a in ARMS + ("Pnew",)))
    for s, sd in res["seeds"].items():
        cols = [(a, sd["arms"][a]) for a in ARMS] + [("Pnew", sd["P_new_only"])]
        print(f"{s:>4} | " + " | ".join(f"{a}: {x['dV']:+6.1f} {x['dB']:+6.1f} {x['dU']:+6.1f} {x['fc2']}/4 "
                                         f"{x['binding2']}/4 {x['counter_fc2']}/4 {x['uptake']}/4" for a, x in cols))
    for key in ("dV", "dB"):
        for k, e in res["effects"][key].items():
            print(f"{key} {k}: mean {e['mean']:+.2f} per seed {e['per_seed']} robust_neg={e['robust_neg']} "
                  f"robust_pos={e['robust_pos']}")
    print("classification:", json.dumps({a: (c["preserved"], c["learned"]) for a, c in res["classification"].items()}))
    print("VERDICT:", json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
