#!/usr/bin/env python3
"""Scientific I1 run. Refuses unless I1-INTEGRATED-INDIVIDUAL-v1 is APPROVED by David with
approved_design_sha256 equal to the current design digest, the manifest equals the frozen preflight
manifest, and --confirm-open I1-INTEGRATED-INDIVIDUAL-v1 is given. Spends i1-v1-dev before launch;
the verdict is PASS, FAIL or BLOCKED.

--print-design-digest prints the design digest an approval must name, and exits (runs nothing)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain_unified.integration.runner import PREREG_FILE, REPO, design_digest, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--print-design-digest", action="store_true")
ap.add_argument("--legacy-root")
ap.add_argument("--historical-core-archive")
ap.add_argument("--confirm-open")
ap.add_argument("--label", default="i1_v1_science")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
if a.print_design_digest:
    print(design_digest(json.loads((REPO / PREREG_FILE).read_text())))
    raise SystemExit(0)
if not (a.legacy_root and a.historical_core_archive and a.confirm_open):
    ap.error("--legacy-root, --historical-core-archive and --confirm-open are required for a run")
run("science", legacy_root=a.legacy_root, historical_core_archive=a.historical_core_archive, label=a.label,
    threads=a.threads, confirm_open=a.confirm_open)
