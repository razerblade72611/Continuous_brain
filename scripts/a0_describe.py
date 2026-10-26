"""Descriptive (non-decision) analysis of A2-A0 results, planned before Phase 1 (Claude + Sol, 2026-10-25).

For every config and seed: wiped (no-memory backbone) forced choice and mean strongest-rival margin on the life-1 facts,
trained forced choice and mean margin at checkpoints 1, 2, 3, 5, and the trained-minus-wiped margin change.
Also lists, per seed, which life-1 facts the trained model gets right at life 5 and whether the wiped model already
got them right (answer-prior check). This never changes the A0 verdict.

  python scripts/a0_describe.py [--dir results/dev3/stageA2_A0]
Writes results/dev3/stageA2_A0_describe.json and prints a table.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPS = (1, 2, 3, 5)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def describe(path):
    d = json.loads(path.read_text())
    out = {}
    for r in d["per_seed"]:
        wiped = r["wiped"]["life1"]
        cp = {c["after_life"]: c["life1"] for c in r["checkpoints"]}
        w_fc = [bool(x.get("fc")) for x in wiped]
        w_m = mean([x["margin_best"] for x in wiped if "margin_best" in x])
        row = {"wiped_fc": sum(w_fc), "wiped_margin": round(w_m, 3), "template_l1": sum(bool(x["template"]) for x in cp[1])}
        for L in CPS:
            m = mean([x["margin_best"] for x in cp[L]])
            row[f"fc_l{L}"] = sum(bool(x["fc"]) for x in cp[L])
            row[f"margin_l{L}"] = round(m, 3)
            row[f"dmargin_l{L}"] = round(m - w_m, 3)
        right5 = [x["fid"] for x in cp[5] if x["fc"]]
        wiped_right = {x["fid"] for x in wiped if x.get("fc")}
        row["life5_correct"] = right5
        row["life5_correct_also_wiped"] = [f for f in right5 if f in wiped_right]
        out[r["probe_seed"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/dev3/stageA2_A0")
    a = ap.parse_args()
    res = {}
    for p in sorted((ROOT / a.dir).glob("*.json")):
        if ".interrupted." in p.name or ".mismatch." in p.name:
            continue
        if json.loads(p.read_text()).get("status") != "complete":
            continue
        res[p.stem] = describe(p)
    (ROOT / "results/dev3/stageA2_A0_describe.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'config':52s} sd  wFC wMarg | tmpl1 | FC@1 FC@2 FC@3 FC@5 | dM@1   dM@2   dM@3   dM@5  | @5 right (also wiped)"
    print(hdr)
    print("-" * len(hdr))
    for n, seeds in res.items():
        for sd, r in seeds.items():
            print(f"{n[:52]:52s} {sd:>2}  {r['wiped_fc']:>3} {r['wiped_margin']:>5.2f} | {r['template_l1']:>5} | "
                  + " ".join(f"{r[f'fc_l{L}']:>4}" for L in CPS) + " | "
                  + " ".join(f"{r[f'dmargin_l{L}']:>6.2f}" for L in CPS)
                  + f" | {len(r['life5_correct'])} ({len(r['life5_correct_also_wiped'])})")
    print("\nwFC/wMarg = wiped backbone forced choice (/10) and mean strongest-rival margin; dM = trained minus wiped margin;"
          " chance FC = 0.2/10 nominal.")


if __name__ == "__main__":
    main()
