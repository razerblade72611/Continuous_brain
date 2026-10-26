"""Bridge/revisit ledger on archived 25-life data (BIND-D0b, read-only, no GPU; Claude + Sol 2026-10-26).

For every life-1 target fact, regenerate the deterministic stream and record whether its entity reappears in later
lives (when, which attribute, bridge flag) and whether its value is later taught to another entity. Then compare the
change in true-answer log-probability between checkpoints 10 and 25, grouped by whether the entity was revisited in
lives 11-25. Correlational only: revisits are not randomized.

  python scripts/bind_bridge.py [results/dev3/stageA2_A1run2]
Writes results/dev3/bind_bridge.json and prints a summary.
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


if __name__ == "__main__":
    main()
