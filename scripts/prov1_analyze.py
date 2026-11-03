"""PROV-1 analysis (Claude's; lab/PROV1_PROTOCOL.md §5, §9). Outside CODE_FILES. Sol's analyzer is independent.

  phase B: python3 scripts/prov1_analyze.py --phase B --dir <dir with seed20..23/> --support1-dir <SUPPORT-1 phase-2 dir> --out a.json
  phase A: python3 scripts/prov1_analyze.py --phase A --dir <dir with seed20..23/> --support1-dir <...> --out a.json  (admission: exact file sha256 only)

Layout of --dir/seed{S}/: prov1_prep_seed{S}.json; phase B: prov1_train_seed{S}.json, eval_{RP,RU,RC}_s{L}_{merged,unmerged}.json,
eval_N_s{L}.json, eval_{RU,RC}_s4_withheld_{merged,unmerged}.json; phase A: eval_{P,U4}_s{L}_{merged,unmerged}.json and
eval_U4_s4_withheld_{merged,unmerged}.json (source support1).
--support1-dir/seed{S}/: SUPPORT-1's prep_seed{S}.json, train_seed{S}.json and eval_*.json (the P reference, P-GATE, and
phase A's ABSENT condition).

The core (`analyze`) is pure: it takes per-row outcomes and returns criteria, so tests can drive it with synthetic answers.
"""
import argparse, json, math, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prov1 as V
import support1 as S
import support1_analyze as SA

TH = dict(S.A0.TH)                                        # ABST-0 thresholds (pos 0.5, pos_vs_p 0.1, first 0.5, ret_vs_p 0.1, ...)
K = {"k2_hnm": 0.90, "k2_hnm_seed": 0.80, "k2_fresh": 0.90, "k3": 0.90, "k5a": 0.80, "k5b": 0.90, "k7": 0.05,
     "fork_obs": 0.50, "fork_wit": 0.80, "fork_seeds": 3}
SLEEPS_GATE = (2, 3, 4, 5)
K7_COHORTS = (("POS-A", "correct"), ("POS-H", "correct"), ("RET", "correct"), ("SEL-D", "unknown"), ("HOW", "unknown"),
              ("HNM", "unknown"), ("FRESH", "unknown"))


# ------------------------------------------------------------------ pure core ------------------------------------------------
def kc(meta):
    return "question" if meta["construction"] == "question" else ("para" if meta["kind"].startswith("para") else "template")


def cohort(meta, st):
    """Disjoint cohorts used by the criteria (§2): POS split by entity group; others by category."""
    c = meta["cat"]
    if c == "POS":
        e = meta["cell"][0]
        return "POS-A" if e in st["A"] else ("POS-H" if e in st["H"] else "POS-T")
    return c


def cohort_kc(name):
    return "para" if name in ("POS-A", "POS-H", "POS-T", "POS", "RET", "FIRST", "TPOST-A", "TPOST-H") else "template"


def mean(xs):
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def ge(a, b):
    return a is not None and b is not None and a >= b - 1e-12


def rate(outs, cls):
    return (sum(o == cls for o in outs) / len(outs)) if outs else None


def cohort_rate(meta_rows, outs, st, coh, cls, cond="FAITHFUL"):
    """Rate of class `cls` (truth outcome) over the base probes of cohort `coh` in its primary construction class."""
    xs = [outs[(cond, r["base_probe_id"])] for r in meta_rows
          if r["condition"] == cond and cohort(r, st) == coh and kc(r) == cohort_kc(coh) and (cond, r["base_probe_id"]) in outs]
    return rate(xs, cls)


def cat_rate(meta_rows, outs, cat, kcl, cls, cond="FAITHFUL"):
    xs = [outs[(cond, r["base_probe_id"])] for r in meta_rows
          if r["condition"] == cond and r["cat"] == cat and kc(r) == kcl and (cond, r["base_probe_id"]) in outs]
    return rate(xs, cls)


def p_outs(p_rows, meta_rows):
    """SUPPORT-1 P's ABSENT outcomes keyed like candidate outcomes, under condition 'FAITHFUL' (for reuse of the rate
    helpers) - P is the reference cortex, read with no record."""
    return {("FAITHFUL", pid): o for pid, o in p_rows.items()}


def gate_outs(src_rows, meta_rows):
    """GATE: the source's ABSENT answer where the FAITHFUL record lists the probed attribute; UNKNOWN (outcome 'unknown'
    against the truth label, which is UNKNOWN for every unsupported cell) where it does not."""
    out = {}
    for r in meta_rows:
        if r["condition"] != "FAITHFUL" or r["base_probe_id"] not in src_rows:
            continue
        out[("FAITHFUL", r["base_probe_id"])] = src_rows[r["base_probe_id"]] if r["supported"] else S.outcome("UNKNOWN", r["gold_truth"])
    return out


def k5(meta_rows, outs, st, group):
    """Paired reliance on eligible matched probes of `group` (A or H) positives (POS, POS-Q): supported, correct under
    FAITHFUL, with both DEL-Q and DEL-X rows. Constructions averaged within cell; returns per-cell values."""
    by = {}
    for r in meta_rows:
        by.setdefault(r["base_probe_id"], {})[r["condition"]] = r
    cells = {}
    for bid, cs in by.items():
        f = cs.get("FAITHFUL")
        if not f or f["cat"] not in ("POS", "POS-Q") or "DEL-Q" not in cs or "DEL-X" not in cs:
            continue
        e = f["cell"][0]
        if e not in st[group] or f["n_entries"] < 2 or not f["supported"]:
            continue
        if outs.get(("FAITHFUL", bid)) != "correct" or ("DEL-Q", bid) not in outs or ("DEL-X", bid) not in outs:
            continue
        a = 1.0 if outs[("DEL-Q", bid)] == "unknown" else 0.0
        b = 1.0 if outs[("DEL-X", bid)] == "correct" else 0.0
        cells.setdefault(tuple(f["cell"]), {}).setdefault(f["construction"], {}).setdefault(f["item"], (a, b, bid))
    # §9 (Sol): identical items deduplicated; variants averaged within construction, then constructions within cell
    res = {}
    for c, cons in cells.items():
        per = [(sum(x[0] for x in v.values()) / len(v), sum(x[1] for x in v.values()) / len(v)) for v in cons.values()]
        res[c] = (sum(x[0] for x in per) / len(per), sum(x[1] for x in per) / len(per),
                  sorted(x[2] for v in cons.values() for x in v.values()))
    return res


def analyze(seeds, st_of, meta_of, cand, pref, ppl=None, fork=None, gate_src=None, phase="B"):
    """seeds: list; st_of[s]: stream; meta_of[s][(history, L)]: expanded probe rows; cand[s][L]: {(cond, base_id): outcome};
    pref[s][L]: SUPPORT-1 P ABSENT {base_id: outcome}; ppl[s][L]: {'eval','modern'} or None (phase A: inherited);
    fork[s]: {'obs': outs at sleep 4 observed, 'wit': outs at sleep 4 withheld} or None;
    gate_src[s][L]: {base_id: outcome} for the main P-GATE (default: pref)."""
    gate_src = gate_src or pref
    M = lambda s, L: meta_of[s][("observed", L)]
    out = {"checks": {}, "per_seed": {}}

    def per_seed_pooled(fn):
        vals = {s: fn(s) for s in seeds}
        return mean(list(vals.values())), vals
    # K1: D1 exactly as frozen (POS para; FIRST para), plus POS-A and POS-H separately vs P, at every sleep 1-5
    k1 = []
    for L in range(1, 6):
        c, _ = per_seed_pooled(lambda s: cat_rate(M(s, L), cand[s][L], "POS", "para", "correct"))
        p, _ = per_seed_pooled(lambda s: cat_rate(M(s, L), p_outs(pref[s][L], M(s, L)), "POS", "para", "correct"))
        k1 += [("POS minimum", L, c, TH["pos"]), ("POS vs P", L, c, None if p is None else p - TH["pos_vs_p"])]
        for coh in ("POS-A", "POS-H"):
            cc, _ = per_seed_pooled(lambda s: cohort_rate(M(s, L), cand[s][L], st_of[s], coh, "correct"))
            pp, _ = per_seed_pooled(lambda s: cohort_rate(M(s, L), p_outs(pref[s][L], M(s, L)), st_of[s], coh, "correct"))
            k1.append((f"{coh} vs P", L, cc, None if pp is None else pp - TH["pos_vs_p"]))
        if L >= 2:
            fv, _ = per_seed_pooled(lambda s: cat_rate(M(s, L), cand[s][L], "FIRST", "para", "correct"))
            k1.append(("FIRST", L, fv, TH["first"]))
    out["checks"]["K1"] = [{"requirement": n, "sleep": L, "actual": a, "threshold": t, "pass": ge(a, t)} for n, L, a, t in k1]
    # K2: HNM unknown >= .90 pooled and >= .80 per seed at every sleep 2-5; FRESH >= .90 pooled
    k2 = []
    for L in SLEEPS_GATE:
        h, hs = per_seed_pooled(lambda s: cat_rate(M(s, L), cand[s][L], "HNM", "template", "unknown"))
        k2.append({"requirement": "HNM unknown pooled", "sleep": L, "actual": h, "threshold": K["k2_hnm"], "pass": ge(h, K["k2_hnm"])})
        k2.append({"requirement": "HNM unknown per seed", "sleep": L, "actual": hs, "threshold": K["k2_hnm_seed"],
                   "pass": all(ge(v, K["k2_hnm_seed"]) for v in hs.values())})
        f, _ = per_seed_pooled(lambda s: cat_rate(M(s, L), cand[s][L], "FRESH", "template", "unknown"))
        k2.append({"requirement": "FRESH unknown pooled", "sleep": L, "actual": f, "threshold": K["k2_fresh"], "pass": ge(f, K["k2_fresh"])})
    out["checks"]["K2"] = k2
    # K3: SEL-D and HOW unknown >= .90 at every sleep 2-5
    k3 = []
    for L in SLEEPS_GATE:
        for cat in ("SEL-D", "HOW"):
            v, _ = per_seed_pooled(lambda s: cat_rate(M(s, L), cand[s][L], cat, "template", "unknown"))
            k3.append({"requirement": f"{cat} unknown", "sleep": L, "actual": v, "threshold": K["k3"], "pass": ge(v, K["k3"])})
    out["checks"]["K3"] = k3
    # K4: RET para correct at sleep 5 >= P - .10; competence (phase B) pooled <= 1.02 and per seed <= 1.05, sleeps 1-5
    r, _ = per_seed_pooled(lambda s: cat_rate(M(s, 5), cand[s][5], "RET", "para", "correct"))
    pr, _ = per_seed_pooled(lambda s: cat_rate(M(s, 5), p_outs(pref[s][5], M(s, 5)), "RET", "para", "correct"))
    k4 = [{"requirement": "RET vs P", "sleep": 5, "actual": r, "threshold": None if pr is None else pr - TH["ret_vs_p"],
           "pass": ge(r, None if pr is None else pr - TH["ret_vs_p"])}]
    if ppl is None:
        k4.append({"requirement": "competence", "actual": "inherited (phase A: SUPPORT-1 weights, unchanged)", "pass": True})
    else:
        for L in range(1, 6):
            for k in ("eval", "modern"):
                vals = [ppl[s][L][k] if ppl[s].get(L) else None for s in seeds]
                pm = mean(vals)
                k4.append({"requirement": f"competence {k}", "sleep": L, "actual": {"pooled": pm, "per_seed": vals},
                           "pass": pm is not None and pm <= TH["ppl_mean"] + 1e-12 and all(v is not None and v <= TH["ppl_seed"] + 1e-12 for v in vals)})
    out["checks"]["K4"] = k4
    # K5: paired reliance, A and H separately; equal weight per eligible cell x sleep within seed, then seeds
    k5r = {}
    for g in ("A", "H"):
        per = {}
        for s in seeds:
            pts = []
            counts = {}
            for L in SLEEPS_GATE:
                cl = k5(M(s, L), cand[s][L], st_of[s], g)
                counts[L] = len(cl)
                pts += [(v[0], v[1]) for v in cl.values()]
            per[s] = {"K5a": mean([p[0] for p in pts]) if pts else None, "K5b": mean([p[1] for p in pts]) if pts else None,
                      "eligible_cell_sleeps": counts}
        a_ = mean([per[s]["K5a"] for s in seeds]); b_ = mean([per[s]["K5b"] for s in seeds])
        k5r[g] = {"K5a": a_, "K5b": b_, "per_seed": per, "pass": ge(a_, K["k5a"]) and ge(b_, K["k5b"])}
    out["checks"]["K5"] = k5r
    # K6: fork history-sensitive for BOTH FORK-H (statement) and FORK-H-Q (question) in >= 3/4 seeds
    if fork is None:
        out["checks"]["K6"] = {"assessable": False, "pass": False}
    else:
        hs = {}
        for s in seeds:
            f = fork.get(s)
            if not f:
                hs[s] = {"history_sensitive": False, "missing": True}; continue
            mo, mw = meta_of[s][("observed", 4)], meta_of[s][("withheld", 4)]
            vals = {}
            for cat, kcl in (("FORK-H", "template"), ("FORK-H-Q", "question")):
                vals[cat] = {"observed_correct": cat_rate(mo, f["obs"], cat, kcl, "correct"),
                             "withheld_unknown": cat_rate(mw, f["wit"], cat, kcl, "unknown")}
                vals[cat]["history_sensitive"] = ge(vals[cat]["observed_correct"], K["fork_obs"]) and ge(vals[cat]["withheld_unknown"], K["fork_wit"])
            hs[s] = vals | {"history_sensitive": all(v["history_sensitive"] for v in vals.values())}
        n = sum(1 for v in hs.values() if v["history_sensitive"])
        need = math.ceil(0.75 * len(seeds))                # = 3 of the 4 production seeds (K["fork_seeds"])
        assert len(seeds) != 4 or need == K["fork_seeds"]
        out["checks"]["K6"] = {"assessable": True, "per_seed": hs, "seeds_sensitive": n, "required": need, "pass": n >= need}
    # K7: noninferiority to P-GATE per cohort; within seed equal sleeps 2-5, then equal seeds
    k7 = {}
    for coh, cls in K7_COHORTS:
        def wsl(outs_fn, s):
            vs = [cohort_rate(M(s, L), outs_fn(s, L), st_of[s], coh, cls) if coh in ("POS-A", "POS-H")
                  else cat_rate(M(s, L), outs_fn(s, L), coh, cohort_kc(coh), cls) for L in SLEEPS_GATE]
            return mean(vs)
        c_ = mean([wsl(lambda s, L: cand[s][L], s) for s in seeds])
        g_ = mean([wsl(lambda s, L: gate_outs(gate_src[s][L], M(s, L)), s) for s in seeds])
        k7[coh] = {"candidate": c_, "p_gate": g_, "pass": ge(c_, None if g_ is None else g_ - K["k7"])}
    out["checks"]["K7"] = k7
    passed = {
        "K1": all(x["pass"] for x in out["checks"]["K1"]), "K2": all(x["pass"] for x in k2), "K3": all(x["pass"] for x in k3),
        "K4": all(x["pass"] for x in k4), "K5": all(v["pass"] for v in k5r.values()),
        "K6": out["checks"]["K6"]["pass"], "K7": all(v["pass"] for v in k7.values())}
    out["pass"] = passed
    return out


def gate_k1_k4(seeds, st_of, meta_of, pref, ppl_p=None):
    """GATE_ONLY test: P-GATE meets K1-K4 (P's supported answers unchanged; unsupported cells UNKNOWN)."""
    gcand = {s: {L: gate_outs(pref[s][L], meta_of[s][("observed", L)]) for L in range(1, 6)} for s in seeds}
    r = analyze(seeds, st_of, meta_of, gcand, pref, ppl=ppl_p, fork=None, phase="gate")
    return {k: r["pass"][k] for k in ("K1", "K2", "K3", "K4")}, r


# ------------------------------------------------------------------ loading + integrity ------------------------------------
GPU_NAME = "NVIDIA RTX 6000 Ada Generation"


def load_prov1_eval(path, prep, psha, arm, L, mode, history, conds, monitors, source, ckpt_ok, test_mode, gpu=GPU_NAME):
    p = Path(path)
    if not p.exists():
        return None, [f"{p.name}: missing"]
    try:
        ev = json.loads(p.read_text())
    except Exception as e:                                # noqa: BLE001
        return None, [f"{p.name}: unreadable ({e})"]
    errs = []
    pv = ev.get("provenance") or {}
    want = {"code_hash": V.code_hash(), "prep_sha256": psha, "model_manifest": prep["model_manifest"], "test_mode": test_mode,
            "conditions": list(conds), "monitors": monitors, "source": source, "arm": arm, "sleep": L, "history": history,
            "mode": mode}
    if {k: pv.get(k) for k in want} != want:
        errs.append(f"{p.name}: provenance mismatch")
    if not ckpt_ok(pv.get("ckpt_sha256")):
        errs.append(f"{p.name}: checkpoint identity not bound to the train log / admission record")
    if (ev.get("schema"), ev.get("experiment"), ev.get("seed"), ev.get("arm"), ev.get("sleep"), ev.get("mode"),
            ev.get("history")) != (V.SCHEMA, V.EXPERIMENT, prep["seed"], arm, L, mode, history):
        errs.append(f"{p.name}: identity fields wrong")
    inv = [r for r in prep["probes"][history][str(L)] if r["condition"] in conds]
    if [r.get("probe_id") for r in ev.get("rows", [])] != [r["probe_id"] for r in inv]:
        errs.append(f"{p.name}: rows are not the complete probe inventory in prep order")
    else:
        for r, g in zip(ev["rows"], inv):
            if (V.outcome(r.get("answer"), g["gold_truth"]) != r.get("outcome_truth")
                    or V.outcome(r.get("answer"), g["gold_compliance"]) != r.get("outcome_compliance")):
                errs.append(f"{p.name}: stored outcome disagrees with independent grading ({r['probe_id']})"); break
    if monitors:
        ppl = ev.get("ppl_ratio") or {}
        if set(ppl) != {"eval", "modern"} or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                    and math.isfinite(v) and v > 0 for v in ppl.values()):
            errs.append(f"{p.name}: competence monitors missing, non-finite or non-positive")
        if not SA._hex64(ev.get("base_denominator_sha256")):
            errs.append(f"{p.name}: missing base-denominator identity")
    if (ev.get("env") or {}).get("tf32") is not False and not test_mode:
        errs.append(f"{p.name}: TF32 not recorded as disabled")
    if not test_mode and (ev.get("env") or {}).get("gpu") != gpu:
        errs.append(f"{p.name}: evaluated on {(ev.get('env') or {}).get('gpu')!r}, not {gpu!r}")
    return ev, errs


def twins(m, u, monitors):
    if [r["probe_id"] for r in m["rows"]] != [r["probe_id"] for r in u["rows"]]:
        return {"ok": False, "reason": "probe identities differ"}
    am = sum(x["answer"] != y["answer"] for x, y in zip(m["rows"], u["rows"]))
    dp = {k: abs(m["ppl_ratio"][k] - u["ppl_ratio"][k]) for k in ("eval", "modern")} if monitors else {}
    ok = am == 0 and all(v <= 1e-4 for v in dp.values()) and (not monitors or m["base_denominator_sha256"] == u["base_denominator_sha256"])
    return {"ok": ok, "answer_mismatches": am, "ppl_diff": dp}


def to_outs(ev, prep, history, L):
    inv = {r["probe_id"]: r for r in prep["probes"][history][str(L)]}
    return {(inv[r["probe_id"]]["condition"], inv[r["probe_id"]]["base_probe_id"]): r["outcome_truth"] for r in ev["rows"]}


def stage_analyze(a):
    d, s1d = Path(a.dir), Path(a.support1_dir)
    seeds = [int(x) for x in a.seeds.split(",")]
    errs, integ = [], {}
    if not a.test_mode and seeds != list(V.DEV_SEEDS):
        errs.append(f"production analysis needs exactly seeds {V.DEV_SEEDS}")
    if S.code_hash() != SA.FROZEN_CODE_HASH:
        errs.append("support1.py is not the frozen SUPPORT-1 code")
    gpu = getattr(a, "gpu", GPU_NAME)
    errs += bind_frozen(a, seeds, d) if not a.test_mode else []
    st_of, meta_of, pref, cands, ppl, forks, phaseA_absent = {}, {}, {}, {}, {}, {}, {}
    arms = ("RP", "RU", "RC") if a.phase == "B" else ("P", "U4")
    for s in seeds:
        pp = d / f"seed{s}" / f"prov1_prep_seed{s}.json"
        prep = json.loads(pp.read_text()); psha = V.sha(pp.read_bytes())
        if not a.test_mode and (prep["code_hash"] != V.code_hash() or prep["test_mode"]):
            errs.append(f"seed {s}: prep not built by this code")
        st_of[s] = prep["stream"]
        meta_of[s] = {(h, int(L)): rows for h, dd in prep["probes"].items() for L, rows in dd.items()}
        # SUPPORT-1 reference: its prep, train log and P evals (validated by SUPPORT-1's own loader)
        p1p = s1d / f"seed{s}" / f"prep_seed{s}.json"
        p1 = json.loads(p1p.read_text()); p1sha = S.sha(p1p.read_bytes())
        if prep["support1_probes_sha256"] != S.chash(p1["probes"]):
            errs.append(f"seed {s}: PROV-1 probe inventory is not SUPPORT-1's")
        t1 = json.loads((s1d / f"seed{s}" / f"train_seed{s}.json").read_text())
        pref[s] = {}
        for L in range(1, 6):
            ev, e = SA.load_eval(s1d / f"seed{s}" / f"eval_P_s{L}_merged.json", p1, p1sha, "P", L, "merged", "observed",
                                 t1["arms"]["P"][str(L)]["ckpt_sha256"], a.test_mode)
            errs += [f"seed {s} SUPPORT-1 P: {x}" for x in e]
            if ev:
                pref[s][L] = {r["probe_id"]: r["outcome"] for r in ev["rows"]}
        if a.phase == "A":
            phaseA_absent[s] = {}
            for arm in arms:
                phaseA_absent[s][arm] = {}
                for L in range(1, 6):
                    mode0 = "merged"
                    f = s1d / f"seed{s}" / f"eval_{arm}_s{L}_{mode0}.json"
                    if not f.exists():
                        mode0 = "unmerged"; f = s1d / f"seed{s}" / f"eval_{arm}_s{L}_{mode0}.json"
                    ev, e = SA.load_eval(f, p1, p1sha, arm, L, mode0, "observed", t1["arms"][arm][str(L)]["ckpt_sha256"], a.test_mode)
                    errs += [f"seed {s} SUPPORT-1 {arm}: {x}" for x in e]
                    if ev:
                        phaseA_absent[s][arm][L] = {r["probe_id"]: r["outcome"] for r in ev["rows"]}
            logged = {(arm, str(L)): t1["arms"][arm][str(L)]["ckpt_sha256"] for arm in arms for L in range(1, 6)}
            logged[("U4", "4w")] = t1["fork"]["withheld"]["U4"]["ckpt_sha256"]
            conds, monitors, source = V.PHASE_A_CONDITIONS, False, "support1"
        else:
            tr = json.loads((d / f"seed{s}" / f"prov1_train_seed{s}.json").read_text())
            tp = tr.get("provenance") or {}
            want_tp = {"prep_sha256": psha, "code_hash": V.code_hash(), "model_manifest": prep["model_manifest"],
                       "test_mode": a.test_mode, "arms": list(V.ARMS), "sleeps": 5, "fork": True}
            if (tr.get("schema"), tr.get("experiment"), tr.get("seed")) != (V.SCHEMA, V.EXPERIMENT, s) or \
                    {k: tp.get(k) for k in want_tp} != want_tp or (not a.test_mode and tp.get("gpu") != gpu):
                errs.append(f"seed {s}: train log is not RP/RU/RC x 5 sleeps + fork of this prep/code/model/GPU")
            elif any(not SA._hex64(tr["arms"][arm][str(L)].get("ckpt_sha256")) for arm in arms for L in range(1, 6)) or \
                    any(not SA._hex64(tr["fork"]["withheld"][arm].get("ckpt_sha256")) or
                        tr["fork"]["parent_full_state_hash"].get(arm) != tr["arms"][arm]["3"].get("full_state_hash")
                        for arm in V.FORK_ARMS):
                errs.append(f"seed {s}: train log checkpoint or fork-parent identities invalid")
            elif any(not SA._hex64(tr["arms"][arm][str(L)].get("full_state_hash")) or
                     _bag(tr["arms"][arm][str(L)].get("delivered")) != _bag(S._delivered(prep["schedule"][arm][str(L)])) or
                     list(tr["arms"][arm][str(L)].get("budget") or []) != list(V.budget(prep["schedule"][arm][str(L)]))
                     for arm in arms for L in range(1, 6)) or \
                    any(_bag(tr["fork"]["withheld"][arm].get("delivered")) != _bag(S._delivered(prep["fork_schedule"][arm]))
                        for arm in V.FORK_ARMS):
                errs.append(f"seed {s}: train log exposures (delivered items, budgets) or state hashes do not match the prep")
            logged = {(arm, str(L)): tr["arms"][arm][str(L)]["ckpt_sha256"] for arm in arms for L in range(1, 6)}
            for arm in V.FORK_ARMS:
                logged[(arm, "4w")] = tr["fork"]["withheld"][arm]["ckpt_sha256"]
            conds, monitors, source = V.PHASE_B_CONDITIONS, True, "prov1"

        def ckpt_ok_for(key):
            def ok(h):                                    # exact file sha256 only (the declared phase-A admission route)
                return SA._hex64(h) and h == logged.get(key)
            return ok
        cands[s], ppl[s], forks[s] = {arm: {} for arm in arms}, {arm: {} for arm in arms}, {}
        base_ref = None
        if a.phase == "B":
            dens = set()
            for L in range(1, 6):
                ev, e = load_prov1_eval(d / f"seed{s}" / f"eval_N_s{L}.json", prep, psha, "N", L, "base", "observed", conds,
                                        True, source, lambda h: h is None, a.test_mode, gpu)
                errs += [f"seed {s}: {x}" for x in e]
                if ev:
                    dens.add(ev.get("base_denominator_sha256"))
            if len(dens) != 1:
                errs.append(f"seed {s}: base denominators differ across N evaluations")
            base_ref = next(iter(dens)) if len(dens) == 1 else None
        for arm in arms:
            for L in list(range(1, 6)) + ["4w"]:
                if L == "4w" and arm not in (V.FORK_ARMS if a.phase == "B" else ("U4",)):
                    continue
                Lx, hist = (4, "withheld") if L == "4w" else (L, "observed")
                evs = {}
                for mode in ("merged", "unmerged"):
                    f = d / f"seed{s}" / (f"eval_{arm}_s4_withheld_{mode}.json" if L == "4w" else f"eval_{arm}_s{L}_{mode}.json")
                    ev, e = load_prov1_eval(f, prep, psha, arm, Lx, mode, hist, conds, monitors, source,
                                            ckpt_ok_for((arm, str(L))), a.test_mode, gpu)
                    errs += [f"seed {s}: {x}" for x in e]
                    evs[mode] = ev
                if evs["merged"] and evs["unmerged"]:
                    tw = twins(evs["merged"], evs["unmerged"], monitors)
                    if monitors and base_ref and evs["merged"]["base_denominator_sha256"] != base_ref:
                        tw["ok"] = False; tw["reason"] = "base denominator differs from N reference"
                    integ[f"{s}:{arm}:{L}"] = tw
                    if not tw["ok"]:
                        errs.append(f"seed {s} {arm} {L}: twin integrity failed")
                    outs = to_outs(evs["merged"], prep, hist, Lx)
                    if L == "4w":
                        forks[s].setdefault(arm, {})["wit"] = outs
                    else:
                        cands[s][arm][L] = outs
                        if monitors:
                            ppl[s][arm][L] = evs["merged"]["ppl_ratio"]
                        if L == 4:
                            forks[s].setdefault(arm, {})["obs"] = outs
    if errs:
        res = {"which": "PROV-1 analyze", "phase": a.phase, "verdict": "PROV1_INVALID", "errors": errs}
        Path(a.out).write_text(json.dumps(res, indent=1)); print("verdict=PROV1_INVALID", errs[:5]); return
    if a.phase == "A":
        for s in seeds:                                   # ABSENT for phase-A readings = SUPPORT-1's archived evaluation
            for arm in arms:
                for L in range(1, 6):
                    for bid, o in phaseA_absent[s][arm][L].items():
                        cands[s][arm][L][("ABSENT", bid)] = o
    res = {"which": "PROV-1 analyze", "phase": a.phase, "seeds": seeds, "code_hash": V.code_hash(), "errors": [],
           "integrity": integ, "arms": {}}
    for arm in arms:
        fk = {s: forks[s].get(arm) for s in seeds} if all(forks[s].get(arm, {}).keys() >= {"obs", "wit"} for s in seeds) else None
        res["arms"][arm] = analyze(seeds, st_of, meta_of, {s: cands[s][arm] for s in seeds}, pref,
                                   ppl=({s: ppl[s][arm] for s in seeds} if a.phase == "B" else None), fork=fk)
        res["arms"][arm]["own_gate_K7"] = analyze(
            seeds, st_of, meta_of, {s: cands[s][arm] for s in seeds}, pref, ppl=None, fork=None,
            gate_src={s: {L: {bid: o for (c, bid), o in cands[s][arm][L].items() if c == "ABSENT"} for L in range(1, 6)}
                      for s in seeds})["checks"]["K7"]
        res["arms"][arm]["value_of_information"] = {
            coh: mean([mean([cohort_rate(meta_of[s][("observed", L)], cands[s][arm][L], st_of[s], coh, cls, "FAITHFUL")
                             - cohort_rate(meta_of[s][("observed", L)], cands[s][arm][L], st_of[s], coh, cls, "ABSENT")
                             if None not in (cohort_rate(meta_of[s][("observed", L)], cands[s][arm][L], st_of[s], coh, cls, "FAITHFUL"),
                                             cohort_rate(meta_of[s][("observed", L)], cands[s][arm][L], st_of[s], coh, cls, "ABSENT"))
                             else None for L in SLEEPS_GATE]) for s in seeds])
            for coh, cls in K7_COHORTS}
        res["arms"][arm]["HNM_unknown_by_sleep"] = {L: mean([cat_rate(meta_of[s][("observed", L)], cands[s][arm][L], "HNM",
                                                                       "template", "unknown") for s in seeds]) for L in range(1, 6)}
        res["arms"][arm]["ADD_confabulation"] = mean([mean([cat_rate_any(meta_of[s][("observed", L)], cands[s][arm][L], "ADD",
                                                                         "substantive") for L in SLEEPS_GATE]) for s in seeds])
    gate_pass, gate_detail = gate_k1_k4(seeds, st_of, meta_of, pref)
    res["p_gate_K1_K4"] = gate_pass
    if a.phase == "B":
        rc = res["arms"]["RC"]["pass"]
        res["verdict"] = "PROV1_ADVANCE" if all(rc.values()) else ("PROV1_GATE_ONLY" if all(gate_pass.values()) else "PROV1_NO_ADVANCE")
        res["RC_minus_RU_K5_common"] = common_k5(seeds, st_of, meta_of, cands, "RC", "RU")
    else:
        ok = [arm for arm in arms if all(res["arms"][arm]["pass"][k] for k in ("K1", "K2", "K3", "K4", "K5", "K7"))]
        res["verdict"] = "PROV1A_ZERO_SHOT_PASS" if ok else "PROV1A_ZERO_SHOT_FAIL"
        res["zero_shot_passing_arms"] = ok
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"verdict": res["verdict"], "pass": {arm: res["arms"][arm]["pass"] for arm in arms},
                      "p_gate_K1_K4": gate_pass}, indent=1, default=str))
    print(f"verdict={res['verdict']}")


def cat_rate_any(meta_rows, outs, cond, cls):
    xs = [outs[(cond, r["base_probe_id"])] for r in meta_rows if r["condition"] == cond and (cond, r["base_probe_id"]) in outs]
    return rate(xs, cls)


def _k5_items(meta_rows, outs, st, group):
    """Eligible probe variants for K5 (same rules as k5): {(cell, construction, item): (K5a, K5b)}."""
    by = {}
    for r in meta_rows:
        by.setdefault(r["base_probe_id"], {})[r["condition"]] = r
    out = {}
    for bid, cs in by.items():
        f = cs.get("FAITHFUL")
        if not f or f["cat"] not in ("POS", "POS-Q") or "DEL-Q" not in cs or "DEL-X" not in cs:
            continue
        if f["cell"][0] not in st[group] or f["n_entries"] < 2 or not f["supported"]:
            continue
        if outs.get(("FAITHFUL", bid)) != "correct" or ("DEL-Q", bid) not in outs or ("DEL-X", bid) not in outs:
            continue
        out.setdefault((tuple(f["cell"]), f["construction"], f["item"]),
                       (1.0 if outs[("DEL-Q", bid)] == "unknown" else 0.0, 1.0 if outs[("DEL-X", bid)] == "correct" else 0.0))
    return out


def common_k5(seeds, st_of, meta_of, cands, x, y):
    """RC - RU on JOINTLY eligible probe variants (Sol): intersect variants first, then average within construction,
    constructions within cell-sleep, cell-sleeps within seed, then seeds. Empty common support -> None (not assessable)."""
    out = {}
    for g in ("A", "H"):
        per_seed_a, per_seed_b = [], []
        for s in seeds:
            pts = []
            for L in SLEEPS_GATE:
                ix = _k5_items(meta_of[s][("observed", L)], cands[s][x][L], st_of[s], g)
                iy = _k5_items(meta_of[s][("observed", L)], cands[s][y][L], st_of[s], g)
                common = set(ix) & set(iy)
                cells = {}
                for k in common:
                    cells.setdefault(k[0], {}).setdefault(k[1], []).append((ix[k][0] - iy[k][0], ix[k][1] - iy[k][1]))
                for cons in cells.values():
                    per = [(sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs)) for vs in cons.values()]
                    pts.append((sum(p[0] for p in per) / len(per), sum(p[1] for p in per) / len(per)))
            per_seed_a.append(sum(p[0] for p in pts) / len(pts) if pts else None)
            per_seed_b.append(sum(p[1] for p in pts) / len(pts) if pts else None)
        out[g] = {"K5a_diff": mean(per_seed_a), "K5b_diff": mean(per_seed_b)}
    return out


def _bag(xs):
    """Order-insensitive canonical form of a delivered-items list (order carries no meaning)."""
    return None if not isinstance(xs, list) else sorted(json.dumps(x, sort_keys=True) for x in xs)


def bind_frozen(a, seeds, d):
    """Production: every analysed prep is a frozen phase-0 prep, and phase 1's calibration passed and is bound."""
    errs = []
    p0 = Path(getattr(a, "phase0_dir", None) or Path(__file__).resolve().parents[1] / "results/prov1/phase0")
    p1 = Path(getattr(a, "phase1_dir", None) or Path(__file__).resolve().parents[1] / "results/prov1/phase1")
    try:
        frozen = {ln.split()[1]: ln.split()[0] for ln in (p0 / "PHASE0.sha256").read_text().splitlines()
                  if ln.endswith(".json") and len(ln.split()) == 2}
        for s in seeds:
            f = d / f"seed{s}" / f"prov1_prep_seed{s}.json"
            if frozen.get(f.name) != V.sha(f.read_bytes()):
                errs.append(f"seed {s}: prep is not the frozen phase-0 prep")
        c = json.loads((p1 / "cal" / "calcheck.json").read_text())
        if c.get("pass") is not True or c.get("errors") or c.get("code_hash") != V.code_hash() or c.get("test_mode"):
            errs.append("phase-1 calibration did not pass for this code")
        if c.get("prep_sha256") != frozen.get("prov1_prep_seed24.json"):
            errs.append("phase-1 calibration is not bound to the frozen seed-24 prep")
    except (OSError, ValueError, KeyError) as e:
        errs.append(f"frozen phase-0/phase-1 evidence unreadable: {e}")
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("A", "B"), required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--support1-dir", required=True); ap.add_argument("--seeds", default="20,21,22,23")
    ap.add_argument("--phase0-dir"); ap.add_argument("--phase1-dir")
    ap.add_argument("--out", required=True); ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--gpu", default=GPU_NAME)
    stage_analyze(ap.parse_args())


if __name__ == "__main__":
    main()
