"""Bridge/revisit ledger on archived 25-life data (BIND-D0b, read-only, no GPU; Claude + Sol 2026-10-26).

For every life-1 target fact, regenerate the deterministic stream and record whether its entity reappears in later
lives (when, which attribute, bridge flag) and whether its value is later taught to another entity. Then compare the
change in true-answer log-probability between checkpoints 10 and 25, grouped by whether the entity was revisited in
lives 11-25. Correlational only: revisits are not randomized.

  python scripts/bind_bridge.py [results/dev3/stageA2_A1run2]
Also (Sol, 2026-10-26: commit the exact derivations) --
  value-reuse cut: change in true logP (vs wiped) from life 10 to 25 for facts whose VALUE is later taught to another
    entity in lives 11-25, vs not re-taught; reported per config AND per original fact (the independent unit);
  shared/unique cut (A0 archive, checkpoint 1): life-1 forced choice for facts whose attribute appears for another
    entity in life 1 (binding-sensitive) vs unique (binding-blind), with denominators per seed.
Writes results/dev3/bind_bridge.json (+ bind_bridge_cuts.json) and prints a summary.

  python scripts/bind_bridge.py [results/dev3/stageA2_A1run2] [results/dev3/stageA2_A0]
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lifegen", ROOT / "being/v3/lifegen.py")
lifegen = importlib.util.module_from_spec(spec)
sys.modules["lifegen"] = lifegen          # dataclasses need the module registered (loaded by path: no torch import)
spec.loader.exec_module(lifegen)


def ledger(seed, n_lives):
    st = lifegen.make_stream(seed, n_lives=n_lives)
    out = {}
    for f in st.lives[0]:
        rev = [{"life": g.life, "attr": g.attr, "value": g.value, "bridge": g.bridge}
               for life in st.lives[1:] for g in life if g.entity == f.entity]
        reuse = [{"life": g.life, "entity": g.entity, "attr": g.attr}
                 for life in st.lives[1:] for g in life if g.value == f.value and g.entity != f.entity]
        out[f.fid] = {"entity": f.entity, "attr": f.attr, "value": f.value, "revisits": rev, "value_reused": reuse}
    return st, out


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/dev3/stageA2_A1run2"
    res, groups = {}, {"revisited_11_25": [], "not_revisited_11_25": []}
    led = {}
    for p in sorted(dd.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            continue
        for r in d["per_seed"]:
            s = r["probe_seed"]
            if s not in led:
                st, led[s] = ledger(s, d["lives"])
            cp = {c["after_life"]: {x["fid"]: x for x in c["life1"]} for c in r["checkpoints"]}
            wiped = {x["fid"]: x for x in r["wiped"]["life1"]}
            mism = [fid for fid, x in cp[1].items() if fid not in led[s] or x["answer"] != led[s][fid]["value"]]
            if mism:
                raise SystemExit(f"{p.name} seed {s}: regenerated stream does not match archived facts {mism}")
            for fid, e in led[s].items():
                late = [v for v in e["revisits"] if 10 < v["life"] <= 25]
                d10 = cp[10][fid]["logprob"] - wiped[fid]["logprob"]
                d25 = cp[25][fid]["logprob"] - wiped[fid]["logprob"]
                key = "revisited_11_25" if late else "not_revisited_11_25"
                row = {"run": p.stem, "seed": s, "fid": fid, "late_revisits": len(late), "dtrue_10": round(d10, 2),
                       "dtrue_25": round(d25, 2), "change_10_25": round(d25 - d10, 2)}
                groups[key].append(row)
                res.setdefault(p.stem, []).append(row)
    out = {"ledger": {str(k): v for k, v in led.items()}, "rows": res}
    (ROOT / "results/dev3/bind_bridge.json").write_text(json.dumps(out, indent=1))
    print("streams regenerated and verified against archived life-1 facts for seeds", sorted(led))
    for s, L in led.items():
        rv = sum(1 for e in L.values() if any(10 < v["life"] <= 25 for v in e["revisits"]))
        ru = sum(1 for e in L.values() if e["value_reused"])
        print(f"seed {s}: life-1 entities revisited in lives 11-25: {rv}/10; life-1 values later taught to another entity: {ru}/10")
    print(f"\n{'run':48s} {'revisited: n  mean d(10->25)':>32s} {'not revisited: n  mean d(10->25)':>36s}")
    for run, rows in res.items():
        a = [x["change_10_25"] for x in rows if x["late_revisits"]]
        b = [x["change_10_25"] for x in rows if not x["late_revisits"]]
        print(f"{run[:48]:48s} {len(a):>14} {mean(a):>16.2f} {len(b):>18} {mean(b):>16.2f}")
    a = [x["change_10_25"] for x in groups["revisited_11_25"]]
    b = [x["change_10_25"] for x in groups["not_revisited_11_25"]]
    print(f"{'ALL (fact x config x seed)':48s} {len(a):>14} {mean(a):>16.2f} {len(b):>18} {mean(b):>16.2f}")


def cuts(a1_dir, a0_dir):
    from collections import Counter
    led = {s: ledger(s, 25)[1] for s in (0, 1)}
    by_cfg, by_fact = {}, {}
    for p in sorted(Path(a1_dir).glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            continue
        for r in d["per_seed"]:
            s = r["probe_seed"]
            cp = {c["after_life"]: {x["fid"]: x for x in c["life1"]} for c in r["checkpoints"]}
            w = {x["fid"]: x for x in r["wiped"]["life1"]}
            for fid, e in led[s].items():
                ch = (cp[25][fid]["logprob"] - w[fid]["logprob"]) - (cp[10][fid]["logprob"] - w[fid]["logprob"])
                g = "retaught_11_25" if any(10 < u["life"] <= 25 for u in e["value_reused"]) else "not_retaught"
                by_cfg.setdefault(p.stem, {}).setdefault(g, []).append(ch)
                by_fact.setdefault((s, fid, g), []).append(ch)
    cfg_summary = {k: {g: {"n": len(v), "mean": round(mean(v), 2)} for g, v in gs.items()} for k, gs in by_cfg.items()}
    facts = {g: [mean(v) for (s, f, gg), v in by_fact.items() if gg == g] for g in ("retaught_11_25", "not_retaught")}
    fact_summary = {g: {"n_facts": len(v), "mean_of_fact_means": round(mean(v), 2),
                        "facts_positive": sum(x > 0 for x in v)} for g, v in facts.items()}
    shared = {}
    for s in (0, 1):
        l1 = lifegen.make_stream(s, n_lives=5).lives[0]
        c = Counter(f.attr for f in l1)
        shared[s] = {f.fid: c[f.attr] > 1 for f in l1}
    su = {}
    for p in sorted(Path(a0_dir).glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            continue
        row = {}
        for r in d["per_seed"]:
            s = r["probe_seed"]
            x1 = {c["after_life"]: c for c in r["checkpoints"]}[1]["life1"]
            sh = [x for x in x1 if shared[s][x["fid"]]]
            un = [x for x in x1 if not shared[s][x["fid"]]]
            row[s] = {"shared_fc": sum(bool(x["fc"]) for x in sh), "shared_n": len(sh),
                      "unique_fc": sum(bool(x["fc"]) for x in un), "unique_n": len(un),
                      "shared_margin_best": round(mean([x["margin_best"] for x in sh]), 2)}
        su[p.stem] = row
    return {"value_reuse_by_config": cfg_summary, "value_reuse_by_fact": fact_summary, "shared_unique_life1": su}


if __name__ == "__main__":
    main()
    a1 = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/dev3/stageA2_A1run2"
    a0 = sys.argv[2] if len(sys.argv) > 2 else ROOT / "results/dev3/stageA2_A0"
    c = cuts(a1, a0)
    (ROOT / "results/dev3/bind_bridge_cuts.json").write_text(json.dumps(c, indent=1))
    print("\nvalue-reuse cut, per ORIGINAL FACT (independent unit; change in dlogP(true), life 10 -> 25, mean over configs):")
    for g, v in c["value_reuse_by_fact"].items():
        print(f"   {g:16s} facts {v['n_facts']:>2}  mean {v['mean_of_fact_means']:>6.2f}  positive {v['facts_positive']}/{v['n_facts']}")
    print("shared/unique life-1 forced choice at checkpoint 1 (A0), per seed: shared fc/n | unique fc/n")
    for k, row in c["shared_unique_life1"].items():
        print("   " + k[:46].ljust(46) + "  " + "  ".join(
            f"s{s}: {v['shared_fc']}/{v['shared_n']} | {v['unique_fc']}/{v['unique_n']}" for s, v in row.items()))
