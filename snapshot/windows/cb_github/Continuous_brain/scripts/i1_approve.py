#!/usr/bin/env python3
"""David's approval of I1-INTEGRATED-INDIVIDUAL-v1 (run it yourself, after the review and the preflight).

It refuses unless the preflight directory holds a valid, spent preflight of this preregistration's
preflight family and this machine reproduces the preflight's manifest exactly. It then writes, in one
step, the approval into registry/i1_integration_v1.json (status, approved_by, approved_utc,
preflight_manifest_sha256, approved_design_sha256) and sets the I1 entry of registry/preregistrations.json
to APPROVED. Nothing is run and no seed family is touched."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain_unified.integration.runner import approve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preflight-dir", required=True)
ap.add_argument("--approver", required=True)
ap.add_argument("--i-approve", required=True, help="type the preregistration id: I1-INTEGRATED-INDIVIDUAL-v1")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
print(json.dumps(approve(preflight_dir=a.preflight_dir, approver=a.approver, confirm=a.i_approve, threads=a.threads),
                 indent=2))
