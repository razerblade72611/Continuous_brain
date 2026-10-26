"""BIND-D0: read-only decomposition of archived A0 / A2-Phase-1 results (Claude + Sol, 2026-10-26). No GPU, no rerun.

For each life-1 fact, the probes saved the true answer's absolute log-probability (`logprob`) and the margin over the
strongest same-pool rival (`margin_best`). So the strongest rival's log-probability is logprob - margin_best (its
IDENTITY was not saved). Relative to the wiped (no-memory) backbone, matched by fact id and seed:

    dM = dlogP(true) - dlogP(strongest rival)

This separates "the true answer lost probability" from "a rival gained probability". Caveat (recorded): the strongest
rival may be a different candidate in the wiped and trained models, so this is not rival attribution -- BIND-D1 does that.

  python scripts/bind_d0.py results/dev3/stageA2_A0 results/dev3/stageA2_A1run2
Writes results/dev3/bind_d0.json and prints a table.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def analyse(path):
    d = json.loads(path.read_text())
    rows = []
    for r in d["per_seed"]:
        wiped = {x["fid"]: x for x in r["wiped"]["life1"]}
        for c in r["checkpoints"]:
            facts = [x for x in c["life1"] if x["fid"] in wiped and "margin_best" in x and "logprob" in x
                     and "margin_best" in wiped[x["fid"]] and "logprob" in wiped[x["fid"]]]
            if not facts:
                continue
            dt, dr, dm, true_gain, rival_gain = [], [], [], 0, 0
            for x in facts:
                w = wiped[x["fid"]]
                d_true = x["logprob"] - w["logprob"]
                d_rival = (x["logprob"] - x["margin_best"]) - (w["logprob"] - w["margin_best"])
                dt.append(d_true); dr.append(d_rival); dm.append(d_true - d_rival)
                true_gain += d_true > 0
                rival_gain += d_rival > 0
            rows.append({"seed": r["probe_seed"], "life": c["after_life"], "n": len(facts),
                         "fc": sum(bool(x.get("fc")) for x in facts),
                         "dlogp_true": round(mean(dt), 2), "dlogp_rival": round(mean(dr), 2), "dM": round(mean(dm), 2),
                         "facts_true_up": true_gain, "facts_rival_up": rival_gain,
                         "abs_logp_true": round(mean([x["logprob"] for x in facts]), 2),
                         "abs_logp_rival": round(mean([x["logprob"] - x["margin_best"] for x in facts]), 2)})
    return rows


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "results/dev3/stageA2_A0", ROOT / "results/dev3/stageA2_A1run2"]
    out = {}
    for dd in dirs:
        for p in sorted(dd.glob("*.json")):
            if ".interrupted." in p.name or ".mismatch." in p.name:
                continue
            d = json.loads(p.read_text())
            if d.get("status") != "complete":
                continue
            out[f"{dd.name}/{p.stem}"] = analyse(p)
    (ROOT / "results/dev3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/dev3/bind_d0.json").write_text(json.dumps(out, indent=1))
    print(f"{'run/config':64s} sd life  FC | dlogP(true) dlogP(rival)    dM | true_up rival_up /n | logP true  rival")
    for name, rows in out.items():
        for r in rows:
            print(f"{name[:64]:64s} {r['seed']:>2} {r['life']:>4} {r['fc']:>3} | {r['dlogp_true']:>11.2f} {r['dlogp_rival']:>12.2f}"
                  f" {r['dM']:>6.2f} | {r['facts_true_up']:>7} {r['facts_rival_up']:>8} /{r['n']:<2}|"
                  f" {r['abs_logp_true']:>9.2f} {r['abs_logp_rival']:>6.2f}")
    print("\nd* = trained minus wiped, matched by fact and seed (nats). The strongest rival's identity was not saved; it may"
          " differ between wiped and trained models.")


if __name__ == "__main__":
    main()
