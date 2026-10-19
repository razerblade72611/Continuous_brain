#!/usr/bin/env python3
"""ENGINEERING_ONLY I1 development run on a registered ENGINEERING family (full report)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continuous_brain_unified.integration.runner import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--legacy-root", required=True)
ap.add_argument("--historical-core-archive", required=True)
ap.add_argument("--family", default="cloud-dev-claude-eng")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
run("dev", legacy_root=a.legacy_root, historical_core_archive=a.historical_core_archive, label=a.label,
    threads=a.threads, family=a.family, start=a.start, n=a.n)
