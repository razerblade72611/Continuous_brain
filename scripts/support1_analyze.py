"""SUPPORT-1 analysis (Claude's; protocol §5-§6). A separate file so the frozen CODE_HASH of the training/evaluation code
(scripts/support1.py, 38b8e705...) is untouched; it imports support1 read-only and verifies that the code it imports is
the frozen code before analysing. Sol's analyzer (scripts/support1_sol_analyze.py) is independent.
  python3 scripts/support1_analyze.py --dir <phase-2 dir with seed20..seed23/> --out analysis.json"""
import argparse, json, math, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import support1 as S
from support1 import (A0, ARMS, DEV_SEEDS, EXPERIMENT, SCHEMA, SLEEPS, code_hash, outcome, sha)
FROZEN_CODE_HASH = "38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae"

# ------------------------------------------------------------------ analysis (§5, §6) -------------------------------------
TH = dict(A0.TH)                                          # ABST-0 thresholds, unchanged (§6)
POSCATS = ("POS", "FIRST", "RET", "TPOST-A", "TPOST-H")


def _hex64(x):
    return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None


def load_eval(path, prep, psha, arm, sleep, mode, history, ckpt, test_mode, qdiag=64):
    """Read one evaluation and validate it against its expected identity; returns (ev, errors). Outcomes are recomputed
    from answers and prep gold; the stored outcome must agree."""
    p = Path(path)
    if not p.exists():
        return None, [f"{p.name}: missing"]
    try:
        ev = json.loads(p.read_text())
    except Exception as e:                                # noqa: BLE001
        return None, [f"{p.name}: unreadable ({e})"]
    errs = []
    want_prov = {"code_hash": code_hash(), "prep_sha256": psha, "ckpt_sha256": ckpt, "model_manifest": prep["model_manifest"],
                 "test_mode": test_mode, "qdiag": qdiag}
    if {k: (ev.get("provenance") or {}).get(k) for k in want_prov} != want_prov:
        errs.append(f"{p.name}: provenance mismatch")
    if (ev.get("schema"), ev.get("experiment"), ev.get("seed"), ev.get("arm"), ev.get("sleep"), ev.get("mode"),
            ev.get("history")) != (SCHEMA, EXPERIMENT, prep["seed"], arm, sleep, mode, history):
        errs.append(f"{p.name}: identity fields wrong")
    inv = prep["probes"][history][str(sleep)]
    if [r.get("probe_id") for r in ev.get("rows", [])] != [r["probe_id"] for r in inv]:
        errs.append(f"{p.name}: rows are not the complete probe inventory in prep order")
    else:
        for r, g in zip(ev["rows"], inv):
            if outcome(r.get("answer"), g["gold_current"]) != r.get("outcome"):
                errs.append(f"{p.name}: stored outcome disagrees with independent grading ({r['probe_id']})"); break
    ppl = ev.get("ppl_ratio") or {}
    if set(ppl) != {"eval", "modern"} or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                and math.isfinite(v) and v > 0 for v in ppl.values()):
        errs.append(f"{p.name}: competence monitors missing, non-finite or non-positive")
    if not _hex64(ev.get("base_denominator_sha256")):
        errs.append(f"{p.name}: missing base-denominator identity")
    if (ev.get("env") or {}).get("tf32") is not False and not test_mode:
        errs.append(f"{p.name}: TF32 not recorded as disabled")
    return ev, errs


def twins(m, u):
    """fp32 merged vs unmerged: identical probes, 0 answer/outcome mismatches, |dppl| <= 1e-4 on both monitors."""
    if [r["probe_id"] for r in m["rows"]] != [r["probe_id"] for r in u["rows"]]:
        return {"ok": False, "reason": "probe identities differ"}
    am = sum(x["answer"] != y["answer"] for x, y in zip(m["rows"], u["rows"]))
    om = sum(x["outcome"] != y["outcome"] for x, y in zip(m["rows"], u["rows"]))
    dp = {k: abs(m["ppl_ratio"][k] - u["ppl_ratio"][k]) for k in ("eval", "modern")}
    return {"ok": am == 0 and om == 0 and all(v <= 1e-4 for v in dp.values()), "answer_mismatches": am,
            "outcome_mismatches": om, "ppl_diff": dp}


def rates(ev, prep, history, sleep):
    """{(cat, kindclass): {class: count, 'n': n}}; kindclass = 'para' (positive primary), 'template', 'question'."""
    inv = {r["probe_id"]: r for r in prep["probes"][history][str(sleep)]}
    out = {}
    for r in ev["rows"]:
        g = inv[r["probe_id"]]
        kc = "question" if g["construction"] == "question" else ("para" if g["kind"].startswith("para") else "template")
        d = out.setdefault((g["cat"], kc), {"correct": 0, "unknown": 0, "empty": 0, "substantive": 0, "n": 0})
        d[r["outcome"]] += 1; d["n"] += 1
    return out


def frac(R, cat, kc, cls):
    d = R.get((cat, kc))
    return {"num": d[cls], "den": d["n"], "value": d[cls] / d["n"]} if d and d["n"] else {"num": 0, "den": 0, "value": None}


def pooled(vals):
    """Equal-weight mean over seeds; any missing seed value -> missing (a criterion needing it fails)."""
    return None if not vals or any(v is None for v in vals) else sum(vals) / len(vals)


def ge(a, b):
    """a >= b; a missing value on EITHER side fails (never coerced; 0.0 is a value)."""
    return a is not None and b is not None and a >= b - 1e-12


def le(a, b):
    return a is not None and b is not None and a <= b + 1e-12


def minus(x, d):
    return None if x is None else x - d


def stage_analyze(a):
    d = Path(a.dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    errs, warn = [], []
    if not a.test_mode and seeds != list(DEV_SEEDS):
        errs.append(f"production analysis needs exactly seeds {DEV_SEEDS}")
    arms_t = list(ARMS)
    R, PPL, integ, invalid, fork = {}, {}, {}, set(), {}
    for s in seeds:
        sd = d / f"seed{s}"
        pf, tf = sd / f"prep_seed{s}.json", sd / f"train_seed{s}.json"
        if not (pf.exists() and tf.exists()):
            errs.append(f"seed {s}: prep or train log missing"); continue
        pb = pf.read_bytes(); prep = json.loads(pb); psha = sha(pb); tr = json.loads(tf.read_text())
        if prep["seed"] != s or prep["code_hash"] != code_hash() or bool(prep["test_mode"]) != a.test_mode:
            errs.append(f"seed {s}: prep identity/code/test-mode mismatch")
        tp = tr.get("provenance", {})
        if tr.get("seed") != s or tp.get("prep_sha256") != psha or tp.get("code_hash") != code_hash():
            errs.append(f"seed {s}: train log does not belong to this prep/code")
        R[s], PPL[s] = {}, {}
        for L in range(1, SLEEPS + 1):
            ev, e_ = load_eval(sd / f"eval_N_s{L}.json", prep, psha, "N", L, "base", "observed", None, a.test_mode)
            if e_:
                errs += [f"seed {s} N s{L}: {x}" for x in e_]
            else:
                R[s][("N", L)], PPL[s][("N", L)] = rates(ev, prep, "observed", L), ev["ppl_ratio"]
        for arm in arms_t:
            for L in range(1, SLEEPS + 1):
                ck = (tr.get("arms", {}).get(arm, {}).get(str(L)) or {}).get("ckpt_sha256")
                if not _hex64(ck):
                    integ[f"{s}:{arm}:{L}"] = {"ok": False, "reason": "no checkpoint identity"}; invalid.add(arm); continue
                m, em = load_eval(sd / f"eval_{arm}_s{L}_merged.json", prep, psha, arm, L, "merged", "observed", ck, a.test_mode)
                u, eu = load_eval(sd / f"eval_{arm}_s{L}_unmerged.json", prep, psha, arm, L, "unmerged", "observed", ck, a.test_mode)
                t = twins(m, u) if (m and u and not em and not eu) else {"ok": False}
                integ[f"{s}:{arm}:{L}"] = t | {"validation_errors": em + eu}
                if em or eu or not t["ok"]:
                    invalid.add(arm); continue
                R[s][(arm, L)], PPL[s][(arm, L)] = rates(m, prep, "observed", L), m["ppl_ratio"]
        if tr.get("fork", {}).get("admitted"):
            for arm in ("U4", "G4"):
                ck = (tr["fork"].get("withheld", {}).get(arm) or {}).get("ckpt_sha256")
                m, em = load_eval(sd / f"eval_{arm}_s4_withheld_merged.json", prep, psha, arm, 4, "merged", "withheld", ck, a.test_mode)
                u, eu = load_eval(sd / f"eval_{arm}_s4_withheld_unmerged.json", prep, psha, arm, 4, "unmerged", "withheld", ck, a.test_mode)
                t = twins(m, u) if (m and u and not em and not eu and _hex64(ck)) else {"ok": False}
                integ[f"{s}:{arm}:4:withheld"] = t | {"validation_errors": em + eu}
                if t["ok"]:
                    fork.setdefault(arm, {})[s] = rates(m, prep, "withheld", 4)
                else:
                    warn.append(f"seed {s} {arm} withheld fork evaluation invalid (fork reading not assessable)")
    if errs:
        res = {"which": "SUPPORT-1 analyze", "verdict": "SUPPORT1_INVALID", "errors": errs, "warnings": warn}
        Path(a.out).write_text(json.dumps(res, indent=1)); print("verdict=SUPPORT1_INVALID", errs[:5]); return

    def M(arm, L, cat, kc, cls):                         # per-seed values and pooled
        per = {s: frac(R[s][(arm, L)], cat, kc, cls) if (arm, L) in R[s] else {"num": 0, "den": 0, "value": None}
               for s in seeds}
        return {"pooled": pooled([v["value"] for v in per.values()]), "per_seed": per}
    P = lambda arm, L, cat, kc, cls: M(arm, L, cat, kc, cls)["pooled"]
    # ---- primary functional claim: U4 vs P, D1-D4 (§6; ABST-0 thresholds)
    crit = None
    if not ({"U4", "P"} & invalid):
        D1 = all(ge(P("U4", L, "POS", "para", "correct"), TH["pos"]) and
                 ge(P("U4", L, "POS", "para", "correct"), minus(P("P", L, "POS", "para", "correct"), TH["pos_vs_p"]))
                 for L in range(1, 6)) and all(ge(P("U4", L, "FIRST", "para", "correct"), TH["first"]) for L in range(2, 6))
        D2 = all(le(P("U4", L, c, "template", "substantive"), TH["conf"]) and ge(P("U4", L, c, "template", "unknown"), TH["unk"])
                 for L in range(1, 6) for c in ("HNM", "FRESH"))
        D3 = all(ge(P("U4", L, c, "template", "unknown"), TH["tpre_unk"]) for L in (2, 3) for c in ("TPRE-A", "TPRE-H")) and \
            all(ge(P("U4", L, c, "para", "correct"), TH["tpost"]) and
                ge(P("U4", L, c, "para", "correct"), minus(P("P", L, c, "para", "correct"), TH["tpost_vs_p"]))
                for L in (4, 5) for c in ("TPOST-A", "TPOST-H"))
        comp = all(ge(TH["ppl_mean"], pooled([PPL[s][("U4", L)][k] for s in seeds])) and
                   all(le(PPL[s][("U4", L)][k], TH["ppl_seed"]) for s in seeds) for L in range(1, 6) for k in ("eval", "modern"))
        D4 = ge(P("U4", 5, "RET", "para", "correct"), minus(P("P", 5, "RET", "para", "correct"), TH["ret_vs_p"])) and comp
        crit = {"D1": D1, "D2": D2, "D3": D3, "D4": D4, "functional": D1 and D2 and D3 and D4}
    # ---- mechanism: U4 vs G4 HNM confabulation at sleep 5
    mech = None
    if crit is not None and "G4" not in invalid:
        u = M("U4", 5, "HNM", "template", "substantive")["per_seed"]; g = M("G4", 5, "HNM", "template", "substantive")["per_seed"]
        diff = pooled([g[s]["value"] - u[s]["value"] if None not in (g[s]["value"], u[s]["value"]) else None for s in seeds])
        lower = sum(1 for s in seeds if None not in (g[s]["value"], u[s]["value"]) and u[s]["value"] < g[s]["value"])
        mech = {"G4_minus_U4": diff, "seeds_U4_lower": lower, "pass": ge(diff, TH["mech"]) and lower >= TH["seeds"]}
    if crit is None:
        verdict = "SUPPORT1_INVALID"
    elif mech is None:
        verdict = ("SUPPORT1_FUNCTIONAL" if crit["functional"] else "SUPPORT1_NO_ADVANCE") + "_MECHANISM_NOT_ASSESSABLE"
    else:
        verdict = ("SUPPORT1_FUNCTIONAL_AND_GROUNDED" if mech["pass"] else "SUPPORT1_FUNCTIONAL") if crit["functional"] \
            else "SUPPORT1_NO_ADVANCE"
    # ---- descriptive readings (§6), sleep 5 unless stated; each with gains and guard status
    def guard(arm):
        if arm in invalid:
            return "invalid"
        d1 = all(ge(P(arm, L, "POS", "para", "correct"), TH["pos"]) and
                 ge(P(arm, L, "POS", "para", "correct"), minus(P("P", L, "POS", "para", "correct"), TH["pos_vs_p"]))
                 for L in range(1, 6))
        d4 = ge(P(arm, 5, "RET", "para", "correct"), minus(P("P", 5, "RET", "para", "correct"), TH["ret_vs_p"])) and all(
            ge(TH["ppl_mean"], pooled([PPL[s][(arm, L)][k] for s in seeds])) for L in range(1, 6) for k in ("eval", "modern"))
        return "ok" if (d1 and d4) else "guard failed"
    readings = {}
    for arm in ("U1", "U4", "G1", "G4"):
        if arm in invalid or "P" in invalid:
            readings[arm] = {"assessable": False}; continue
        unk = lambda cat, kc="template", L=5: M(arm, L, cat, kc, "unknown")
        gain = lambda cat: {s: (unk(cat)["per_seed"][s]["value"] - M("P", 5, cat, "template", "unknown")["per_seed"][s]["value"])
                            if None not in (unk(cat)["per_seed"][s]["value"], M("P", 5, cat, "template", "unknown")["per_seed"][s]["value"])
                            else None for s in seeds}
        gs, gh, gn = gain("SEL-D"), gain("HOW"), gain("HNM")
        def gr(num):
            den = pooled(list(gs.values()))
            ok = den is not None and den >= 0.10 and sum(1 for v in gs.values() if v is not None and v >= 0.05) >= 3
            return {"pooled": (pooled(list(num.values())) / den) if ok and pooled(list(num.values())) is not None else None,
                    "per_seed": {s: (num[s] / gs[s]) if (gs[s] is not None and gs[s] >= 0.05 and num[s] is not None) else None
                                 for s in seeds}, "defined": ok}
        selq = unk("SEL-Q", "question")["pooled"]
        pair = {}
        for s in seeds:                                    # paired construction contrast on the same cells (SEL-Q vs SEL-D)
            inv = {r["probe_id"]: r for r in json.loads((d / f"seed{s}/prep_seed{s}.json").read_text())["probes"]["observed"]["5"]}
            ev = json.loads((d / f"seed{s}/eval_{arm}_s5_merged.json").read_text())
            out = {r["probe_id"]: r["outcome"] for r in ev["rows"]}
            pq = [(out[r["probe_id"]] == "unknown", out[r["pair_probe_id"]] == "unknown") for r in inv.values()
                  if r["cat"] == "SEL-Q" and r["pair_probe_id"]]
            pair[s] = (sum(q for q, _ in pq) - sum(t for _, t in pq)) / len(pq) if pq else None
        gw, ga = gr(gh), gr(gn)
        how_u, how_c = unk("HOW")["pooled"], M(arm, 5, "HOW", "template", "substantive")["pooled"]
        rev3 = {s: frac(R[s][(arm, 3)], "REV-Q", "question", "unknown") for s in seeds}
        acq_rev = sum(1 for v in rev3.values() if v["den"] and v["num"] >= 3) >= 3
        revc = {L: M(arm, L, "REV-Q", "question", "correct")["pooled"] for L in (4, 5)}
        prevc = {L: M("P", L, "REV-Q", "question", "correct")["pooled"] for L in (4, 5)}
        readings[arm] = {
            "guard": guard(arm), "SEL_Q_unknown": selq, "SEL_D_unknown": unk("SEL-D")["pooled"], "HOW_unknown": how_u,
            "HOW_confab": how_c, "HNM_unknown": unk("HNM")["pooled"], "gains": {"SEL-D": gs, "HOW": gh, "HNM": gn},
            "GR_within": gw, "GR_across": ga, "construction_gap_paired": pooled(list(pair.values())),
            "ACQ": ge(selq, 0.80),
            "CONSTRUCTION_GAP": ge(selq, 0.80) and ge(pooled(list(pair.values())), 0.30),
            "WITHIN_TRANSFER": ge(how_u, 0.70) and le(how_c, 0.30) and ge(pooled(list(gh.values())), 0.40),
            "ACROSS_TRANSFER": ge(unk("HNM")["pooled"], 0.50) and ge(pooled(list(gn.values())), 0.30),
            "MEMORIZATION": ("unresolved" if not gw["defined"] else
                             bool(ge(selq, 0.80) and ge(unk("SEL-D")["pooled"], 0.80) and le(gw["pooled"], 0.25))),
            "REVERSAL": {"pre_acquisition_per_seed": {s: (v["num"], v["den"]) for s, v in rev3.items()},
                         "acquired": acq_rev,
                         "success": acq_rev and all(ge(revc[L], 0.50) and ge(revc[L], minus(prevc[L], 0.15)) for L in (4, 5)),
                         "REV_Q_correct": revc, "P_REV_Q_correct": prevc,
                         "TPOST_A_correct": {L: P(arm, L, "TPOST-A", "para", "correct") for L in (4, 5)}},
            "GTQ_unknown": unk("GTQ", "question")["pooled"]}
    dose = {}
    for lo, hi in (("U1", "U4"), ("G1", "G4")):
        if {lo, hi} & invalid:
            dose[f"{hi}-{lo}"] = "NOT_ASSESSABLE"; continue
        dd = {k: (readings[hi][k] - readings[lo][k]) if None not in (readings[hi][k], readings[lo][k]) else None
              for k in ("SEL_Q_unknown", "SEL_D_unknown", "HOW_unknown", "HNM_unknown")}
        dose[f"{hi}-{lo}"] = dd | {"dose_responsive": ge(dd["SEL_Q_unknown"], 0.20)}
    ground = None
    if not {"U4", "G4"} & invalid:
        g = (readings["U4"]["HOW_unknown"] - readings["G4"]["HOW_unknown"]) if None not in (
            readings["U4"]["HOW_unknown"], readings["G4"]["HOW_unknown"]) else None
        ground = {"U4_minus_G4_HOW_unknown": g, "demonstrated": ge(g, 0.20)}
    forkr = {}
    for arm, per in fork.items():
        hs = {}
        for s, Rw in per.items():
            obs = frac(R[s][(arm, 4)], "FORK-H", "template", "correct")["value"] if (arm, 4) in R[s] else None
            wit = frac(Rw, "FORK-H", "template", "unknown")["value"]
            hs[s] = {"observed_correct": obs, "withheld_unknown": wit,
                     "history_sensitive": ge(obs, 0.50) and ge(wit, 0.80)}
        forkr[arm] = hs
    keys = [(c, kc) for c, kc in sorted({k for s in seeds for v in R[s].values() for k in v})]
    table = {f"{arm}|{L}|{c}|{kc}": {cls: M(arm, L, c, kc, cls) for cls in ("correct", "unknown", "empty", "substantive")}
             for arm in arms_t + ["N"] for L in range(1, 6) for (c, kc) in keys
             if all((arm, L) in R[s] for s in seeds) and any((c, kc) in R[s][(arm, L)] for s in seeds)}
    res = {"which": "SUPPORT-1 analyze", "schema": SCHEMA, "seeds": seeds, "code_hash": code_hash(), "errors": [],
           "warnings": warn, "invalid_arms": sorted(invalid), "integrity": integ, "criteria": crit, "mechanism": mech,
           "verdict": verdict, "readings": readings, "dose": dose, "grounding_HOW": ground, "fork": forkr,
           "competence": {f"{arm}|{L}": {k: pooled([PPL[s][(arm, L)][k] for s in seeds]) for k in ("eval", "modern")}
                          for arm in arms_t + ["N"] for L in range(1, 6) if all((arm, L) in PPL[s] for s in seeds)},
           "table": table}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"verdict": verdict, "criteria": crit, "mechanism": mech, "invalid_arms": sorted(invalid),
                      "warnings": warn}, indent=1, default=str))
    print(f"verdict={verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS))); ap.add_argument("--test-mode", action="store_true")
    a = ap.parse_args()
    if not a.test_mode and code_hash() != FROZEN_CODE_HASH:
        sys.exit(f"ANALYZE: support1.py code hash {code_hash()} is not the frozen {FROZEN_CODE_HASH}")
    stage_analyze(a)


if __name__ == "__main__":
    main()
