"""BIND-D1 frozen analysis (written before any D1 data exists; Claude + Sol, 2026-10-26). Read-only, CPU only.

Summaries, per arm and seed, at checkpoints 1, 2, 3, 5 (and the wiped backbone where relevant):
  R  reproduction vs the archived A0 run, tokenizer-prefix failures, ties.
  1  acquisition split: life-1 forced choice on SHARED-attribute facts (another life-1 entity has the same attribute;
     competition present) vs UNIQUE-attribute facts (binding-blind), with denominators; historical all-10 kept.
  2  rival attribution: the strongest rival per life-1 fact; whether it was taught to another entity (same attribute)
     BY THAT CHECKPOINT (no future leakage), first/latest life, exposure count, age; compared with the prevalence of
     taught values among all 49 eligible distractors. Distinct winners per attribute (a single global winner across
     entities suggests attribute-level answer preference).
  3  candidate shift: trained minus wiped log-prob of every distractor, matched by fact and value, stratified by
     exposure status at the checkpoint and by candidate token count.
  4  paired-entity contrasts per cohort (primary: A life 1, B later; simultaneous: both life 1; secondary: A later):
     D_A = l(V|A) - l(W|A), D_B = l(W|B) - l(V|B), C = D_A + D_B, each also minus the wiped value for the same pair.
     A pair counts as post-exposure from the first checkpoint at/after B's life; earlier checkpoints are pre-exposure
     controls (reported separately, never as binding evidence).
Descriptive only: two seeds, five lives. No significance threshold decides anything.

  python scripts/bind_d1_analyze.py results/dev3/bind_d1/*.json
Writes results/dev3/bind_d1_analysis.json and prints the tables.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lifegen_d1a", ROOT / "being/v3/lifegen.py")
lifegen = importlib.util.module_from_spec(spec)
sys.modules["lifegen_d1a"] = lifegen
spec.loader.exec_module(lifegen)


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def tok_bin(n):
    return "1-2" if n <= 2 else "3-4" if n <= 4 else "5+"


def taught_by(stream, n_lives):
    """value -> list of facts teaching it, per attribute, across lives 1..n_lives (for checkpoint-limited lookups)."""
    out = {}
    for life in stream.lives[:n_lives]:
        for f in life:
            out.setdefault((f.attr, f.value), []).append(f)
    return out


def analyse_seed(r, lives):
    st = lifegen.make_stream(r["probe_seed"], n_lives=lives)
    l1 = st.lives[0]
    by_fid = {f.fid: f for f in l1}
    attr_count = Counter(f.attr for f in l1)
    taught = taught_by(st, lives)
    wiped = {x["fid"]: x for x in r["wiped"]["life1"]}
    wiped_cand = {fid: {c["v"]: c["lp"] for c in x.get("cands", [])} for fid, x in wiped.items()}
    wiped_pair = {(p["a_fid"], p["b_fid"]): p for p in r["wiped"].get("paired", [])}
    out = {"checkpoints": {}}
    for cp in r["checkpoints"]:
        L = cp["after_life"]
        rows = cp["life1"]
        # 1 acquisition split
        sh = [x for x in rows if attr_count[x["attr"]] > 1]
        un = [x for x in rows if attr_count[x["attr"]] == 1]
        acq = {"all_fc": sum(bool(x["fc"]) for x in rows), "all_n": len(rows),
               "shared_fc": sum(bool(x["fc"]) for x in sh), "shared_n": len(sh),
               "unique_fc": sum(bool(x["fc"]) for x in un), "unique_n": len(un),
               "template_all": sum(bool(x["template"]) for x in rows)}
        # 2 rival attribution + 3 candidate shift
        winners, prevalence, winner_taught, shifts = [], [], 0, {}
        per_attr_winners = {}
        for x in rows:
            a = by_fid[x["fid"]]
            cands = x.get("cands", [])
            if not cands:
                continue
            rest = cands[1:]
            top = max(rest, key=lambda c: c["lp"])
            def exposure(v, L=L, a=a):
                fs = [f for f in taught.get((a.attr, v), []) if f.life <= L and f.entity != a.entity]
                if not fs:
                    return None
                return {"first": min(f.life for f in fs), "latest": max(f.life for f in fs), "count": len(fs),
                        "age": L - max(f.life for f in fs), "entities": sorted({f.entity for f in fs})}
            ex = exposure(top["v"])
            winners.append({"fid": a.fid, "attr": a.attr, "winner": top["v"], "winner_lp": top["lp"],
                            "true_lp": cands[0]["lp"], "ties_with_true": x.get("fc_ties", 0),
                            "winner_taught": ex is not None, "winner_exposure": ex,
                            "winner_ntok": top.get("ntok")})
            winner_taught += ex is not None
            n_taught = sum(exposure(c["v"]) is not None for c in rest)
            prevalence.append(n_taught / len(rest))
            per_attr_winners.setdefault(a.attr, []).append(top["v"])
            wc = wiped_cand.get(a.fid, {})
            for c in rest:
                if c["v"] in wc:
                    key = ("taught" if exposure(c["v"]) else "untaught") + "|" + tok_bin(c.get("ntok", 0))
                    shifts.setdefault(key, []).append(c["lp"] - wc[c["v"]])
        rival = {"n": len(winners), "winner_taught": winner_taught,
                 "expected_if_random": round(sum(prevalence), 2),
                 "mean_taught_prevalence": mean(prevalence),
                 "distinct_winners_per_attr": {k: f"{len(set(v))}/{len(v)}" for k, v in per_attr_winners.items()},
                 "winners": winners}
        shift = {k: {"n": len(v), "mean_dlp": mean(v)} for k, v in sorted(shifts.items())}
        # 4 paired contrasts
        coh = {}
        for p in cp.get("paired", []):
            w = wiped_pair.get((p["a_fid"], p["b_fid"]))
            phase = "post" if p["b_life"] <= L and p["a_life"] <= L else "pre"
            DA, DB = p["lV_A"] - p["lW_A"], p["lW_B"] - p["lV_B"]
            rec = {"DA": DA, "DB": DB, "C": DA + DB}
            if w:
                wDA, wDB = w["lV_A"] - w["lW_A"], w["lW_B"] - w["lV_B"]
                rec.update({"dDA": DA - wDA, "dDB": DB - wDB, "dC": (DA + DB) - (wDA + wDB)})
            coh.setdefault(f"{p['cohort']}|{phase}", []).append(rec)
        contrasts = {k: {"n": len(v), **{m: mean([x.get(m) for x in v]) for m in ("DA", "DB", "C", "dDA", "dDB", "dC")},
                         "DA_pos": sum(x["DA"] > 0 for x in v)} for k, v in sorted(coh.items())}
        out["checkpoints"][L] = {"acquisition": acq, "rival": rival, "candidate_shift": shift, "contrasts": contrasts}
    return out


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "results/dev3/bind_d1").glob("*.json"))
    res = {}
    for p in paths:
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            print("skip (not complete):", p.name)
            continue
        pf = sum(x.get("prefix_fail", 0) for r in d["per_seed"] for c in r["checkpoints"] for x in c["life1"])
        res[p.stem] = {"reproduction": d.get("reproduction"), "prefix_fail": pf,
                       "identical_value_pairs_excluded": {r["probe_seed"]: r.get("pairs_identical_value_excluded")
                                                          for r in d["per_seed"]},
                       "seeds": {r["probe_seed"]: analyse_seed(r, d["lives"]) for r in d["per_seed"]}}
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/bind_d1_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    for run, a in res.items():
        print(f"\n===== {run}   prefix failures: {a['prefix_fail']}   identical-value pairs excluded: "
              f"{a['identical_value_pairs_excluded']}")
        rep = a["reproduction"] or {}
        for s, v in (rep.get("seeds") or {}).items():
            print(f"  reproduction seed {s}: {v}")
        for s, sd in a["seeds"].items():
            print(f"  -- seed {s}")
            print(f"     {'life':>4} | FC all  shared  unique | rival taught (expected) | contrasts post-exposure "
                  f"[cohort: n DA DB C | dC vs wiped]")
            for L, c in sd["checkpoints"].items():
                q, rv = c["acquisition"], c["rival"]
                con = "; ".join(f"{k.split('|')[0]}: {v['n']} {v['DA']:+.1f} {v['DB']:+.1f} {v['C']:+.1f} | "
                                f"{(v['dC'] if v['dC'] is not None else float('nan')):+.1f}"
                                for k, v in c["contrasts"].items() if k.endswith("|post"))
                print(f"     {L:>4} | {q['all_fc']:>2}/{q['all_n']:<2} {q['shared_fc']:>2}/{q['shared_n']:<2} "
                      f"{q['unique_fc']:>2}/{q['unique_n']:<2} | {rv['winner_taught']:>2}/{rv['n']} "
                      f"({rv['expected_if_random']:>4}) | {con}")


if __name__ == "__main__":
    main()
