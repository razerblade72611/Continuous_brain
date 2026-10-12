"""CB2-L1b power calculation, from L1's own probe records (docs/explore/CB2_L1B_PREREG.md section 4).

Reads L1's cycles.jsonl (pooled cycles 6-10, paired by world) and reports:
  - SD of the paired survival difference and the correlation between paired arms (how much world-pairing helps);
  - the standardised effect with survival capped at several values (whether a restricted mean would help);
  - the worlds needed per family for 80% power at a one-sided 95% bound, for several true effects;
  - power of the single comparison and of the joint four-way claim (L-F and L-Y, fresh and transfer, all lower95 > 0)
    at several battery sizes, by simulation with the correlation between L-F and L-Y measured in L1.

  python scripts/explore/cb2_power.py --cycles <L1 run>/cycles.jsonl [--out evidence/explore/cb2/l1b-power.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

Z95, Z80 = norm.ppf(0.95), norm.ppf(0.80)


def pairs(C, a, b, fam, cycles=range(6, 11)):
    X, Y = [], []
    for c in cycles:
        for x, y in zip(C[c]["probes"][a][fam], C[c]["probes"][b][fam]):
            assert x["seed"] == y["seed"]
            X.append(x["steps"])
            Y.append(y["steps"])
    return np.asarray(X, float), np.asarray(Y, float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", required=True)
    ap.add_argument("--out")
    ap.add_argument("--sd", type=float, default=430.0, help="design SD of the paired difference (steps)")
    a = ap.parse_args()
    C = [json.loads(l) for l in open(a.cycles)]
    ctrl = "yoked" if "yoked" in C[0]["probes"] else "reference"
    rep = {"source": str(a.cycles), "cycles_pooled": [6, 10], "variance": {}, "capped": {}, "n_for_80pct": {}, "power": {}}
    for fam in ("fresh", "transfer"):
        for o in ("frozen", ctrl):
            X, Y = pairs(C, "learning", o, fam)
            d = X - Y
            rep["variance"][f"{fam} learning-{o}"] = {"n": len(d), "mean": round(d.mean(), 1), "sd_diff": round(d.std(ddof=1), 1),
                                                     "corr_arms": round(float(np.corrcoef(X, Y)[0, 1]), 3)}
            for cap in (None, 1500, 1000, 750):
                dc = (np.minimum(X, cap) - np.minimum(Y, cap)) if cap else d
                rep["capped"].setdefault(str(cap), {})[f"{fam} learning-{o}"] = {
                    "mean": round(dc.mean(), 1), "sd": round(dc.std(ddof=1), 1), "standardised": round(dc.mean() / dc.std(ddof=1), 3)}
        A, _ = pairs(C, "learning", "frozen", fam)
        _, F = pairs(C, "learning", "frozen", fam)
        _, R = pairs(C, "learning", ctrl, fam)
        rep["variance"][f"{fam} corr(L-F, L-{ctrl[0].upper()})"] = round(float(np.corrcoef(A - F, A - R)[0, 1]), 3)
    rho = float(np.mean([rep["variance"][f"{f} corr(L-F, L-{ctrl[0].upper()})"] for f in ("fresh", "transfer")]))
    for sd in (300, 350, 420, a.sd):
        rep["n_for_80pct"][f"sd{sd:.0f}"] = {f"+{e}": int(np.ceil(((Z95 + Z80) * sd / e) ** 2)) for e in (30, 40, 50)}
    rng = np.random.default_rng(20261011)
    for n in (700, 1000, 1200, 1500):
        se = a.sd / np.sqrt(n)
        row = {"mde80": round((Z95 + Z80) * se, 1)}
        for e in (30, 40, 50):
            ok = np.ones(200_000, bool)
            for _ in ("fresh", "transfer"):          # families independent (different worlds)
                x = rng.multivariate_normal([e, e], np.array([[1, rho], [rho, 1]]) * se ** 2, 200_000)
                ok &= (x[:, 0] - Z95 * se > 0) & (x[:, 1] - Z95 * se > 0)
            row[f"+{e}"] = {"single": round(float(norm.cdf(e / se - Z95)), 3), "joint4": round(float(ok.mean()), 3)}
        rep["power"][f"N{n}"] = row
    rep["assumptions"] = {"sd": a.sd, "rho_LF_LY": round(rho, 3), "one_sided": 0.05, "worlds_independent": True,
                          "note": "normal approximation for the bootstrap bound; equal true effects in every comparison"}
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
