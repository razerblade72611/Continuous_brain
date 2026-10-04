#!/usr/bin/env python3
"""David's approval of I1-INTEGRATED-INDIVIDUAL-v1 (run it yourself, after the review and the preflight).

It refuses unless runner.verify_preflight_chain() accepts the preflight directory: the canonical preflight
family was really spent in this checkout's ledger, once, by this run; the run was opened under the PROPOSED
preregistration whose design and both registry files are unchanged since; this machine reproduces the
preflight's manifest; exactly the 14 validity checks are all true; exactly the 12 arms are bound; the sealed
report exists, matches its commitment and agrees with the visible view. It then writes, as one journaled act,
the approval into registry/i1_integration_v1.json (status, approved_by, approved_utc,
preflight_manifest_sha256, approved_design_sha256, preflight_run, approval_basis) and sets the I1 entry of
registry/preregistrations.json to APPROVED. Nothing is run and no seed family is touched.

--recover completes an approval that was interrupted between the two file writes (from its journal)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain_unified.integration.runner import approve, recover_approval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--recover", action="store_true", help="complete an interrupted approval from its journal")
ap.add_argument("--preflight-dir")
ap.add_argument("--approver")
ap.add_argument("--i-approve", help="type the preregistration id: I1-INTEGRATED-INDIVIDUAL-v1")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
if a.recover:
    print(recover_approval())
    raise SystemExit(0)
if not (a.preflight_dir and a.approver and a.i_approve):
    ap.error("--preflight-dir, --approver and --i-approve are required")
print(json.dumps(approve(preflight_dir=a.preflight_dir, approver=a.approver, confirm=a.i_approve, threads=a.threads),
                 indent=2))
