#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from continuous_brain_unified.legacy.g0 import G0Plan


def main():
    ap=argparse.ArgumentParser(description="Verify the v0.38.1 G0 evidence plan without executing any historical TEST.")
    ap.add_argument("--plan", default="legacy/g0_plan.json")
    ap.add_argument("--legacy-root", required=True, help="root label wsl-canonical")
    ap.add_argument("--mechanism")
    ap.add_argument("--json", action="store_true")
    args=ap.parse_args()
    plan=G0Plan.load(args.plan)
    ver={v.mechanism_id:v for v in plan.verify({"wsl-canonical":args.legacy_root}, args.mechanism)}
    rows=[]
    entries=[plan.get(args.mechanism)] if args.mechanism else plan.entries
    for e in entries:
        v=ver[e.mechanism_id]
        rows.append({"mechanism_id":e.mechanism_id,"disposition":e.disposition.value,
                     "evidence_hashes_match":v.ok,"files_checked":v.files_checked,
                     "missing":list(v.missing),"mismatches":list(v.mismatches),"reason":e.reason})
    if args.json:
        print(json.dumps({"source_bundle_sha256":plan.source_bundle_sha256,"rows":rows},indent=2,sort_keys=True))
    else:
        for r in rows:
            print(f"{r['mechanism_id']:<44} {r['disposition']:<31} hashes={'OK' if r['evidence_hashes_match'] else 'BLOCKED'} files={r['files_checked']}")
            if r['missing']: print("  missing:", *r['missing'][:3], sep="\n    ")
            if r['mismatches']: print("  mismatches:", *[m['relpath'] for m in r['mismatches'][:3]], sep="\n    ")
    return 0 if all(r['evidence_hashes_match'] or r['disposition'] in ('SUPPORT_ONLY','CONTROL_ONLY','NEEDS_ADJUDICATION','NEEDS_REFRESHED_RESULT_PARSE') for r in rows) else 2

if __name__ == '__main__':
    raise SystemExit(main())
