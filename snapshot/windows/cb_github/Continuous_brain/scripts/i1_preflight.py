#!/usr/bin/env python3
"""I1 engineering preflight (run on the machine that will run the science): registers and spends the
single-use family engineering_preflight_proposed (i1-v1-preflight-eng-b), runs every arm, shows validity
only (values withheld from display, committed by SHA-256, not encrypted),
and writes the manifest whose SHA-256 the preregistration must cite before approval."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain_unified.integration.runner import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--legacy-root", required=True)
ap.add_argument("--historical-core-archive", required=True)
ap.add_argument("--label", default="i1_v1_preflight")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
d = run("preflight", legacy_root=a.legacy_root, historical_core_archive=a.historical_core_archive, label=a.label,
        threads=a.threads)
print("manifest:", d / "manifest.json")
