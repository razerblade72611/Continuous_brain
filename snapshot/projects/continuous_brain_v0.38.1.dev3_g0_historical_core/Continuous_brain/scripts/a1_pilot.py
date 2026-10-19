#!/usr/bin/env python3
"""Amendment A1.2 acceptance pilots on the full E0 anatomy. Report only: approving A1 is David's decision.

  1. false-fail rate of each rule on statistically identical 60 s Poisson pairs
     (A1 criterion: <= 5 % over >= 100 pairs; DAV-32-v1 is run for comparison)
  2. A1 sensitivity: a 10 % rate shift must be detected in every population
     with >= 5,000 expected spikes

Rates: per-population operating rates from an E0 gate parity file
(--rates-from runs/e0_gate/<run_id>/parity_stats.json), or a uniform rate.

Example (inside WSL; ~1-3 h on the 5600X for 100 pairs):
  python scripts/a1_pilot.py --rates-from runs/e0_gate/<cpu_run_id>/parity_stats.json --pairs 100 --out runs/a1_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuous_brain_eval.parity_a1 import pilot_false_fail, pilot_sensitivity  # noqa: E402
from continuous_brain_protocols import sha256_file  # noqa: E402
from continuous_brain_snn.anatomy import build_anatomy  # noqa: E402
from continuous_brain_snn.config import E0Config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates-from", default=None)
    ap.add_argument("--uniform-rate", type=float, default=5.0)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=901000)
    ap.add_argument("--out", default="runs/a1_pilot")
    args = ap.parse_args(argv)
    anatomy = build_anatomy(E0Config(seed=0))
    if args.rates_from:
        rates = json.loads(Path(args.rates_from).read_text())["rate_hz"]
        rates = {p.name: max(float(rates[p.name]), 0.1) for p in anatomy.populations}
        source = {"path": args.rates_from, "sha256": sha256_file(args.rates_from)}
    else:
        rates = {p.name: args.uniform_rate for p in anatomy.populations}
        source = {"uniform_rate_hz": args.uniform_rate}
    report = {"status": "ENGINEERING_ONLY", "note": "pilot for proposed Amendment A1; not an approval",
              "rates_source": source, "duration_s": args.duration,
              "a1_false_fail": pilot_false_fail(anatomy, rates, args.pairs, args.duration, seed=args.seed, rule="A1"),
              "dav32_v1_false_fail": pilot_false_fail(anatomy, rates, args.pairs, args.duration, seed=args.seed,
                                                      rule="DAV-32-v1"),
              "a1_sensitivity": pilot_sensitivity(anatomy, rates, 0.10, args.duration, seed=args.seed + 5000)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "a1_pilot_report.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rates_source"}, indent=1))
    print(f"report: {path}  sha256 {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
