#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from continuous_brain_unified.legacy.g0_runner import prepare, run, runtime_preflight

def main():
    ap=argparse.ArgumentParser(description='Prepare, engineering-preflight, or execute G0 isolated legacy requalification.')
    ap.add_argument('--legacy-root',required=True)
    ap.add_argument('--mechanism',required=True)
    ap.add_argument('--registry-dir',default=None)
    ap.add_argument('--historical-core-archive',default=None,help='Original continuous_brain_v0.12.2_unified_workspace.zip; exact SHA-256 is pinned.')
    ap.add_argument('--runtime-preflight',action='store_true',help='Spend only the registered ENGINEERING preflight family; never opens scientific DEV.')
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--confirm-open',default=None)
    ap.add_argument('--out',default='runs/g0_requalification')
    ap.add_argument('--preflight-attestations',default='evidence/g0_runtime_preflight')
    a=ap.parse_args()
    if a.run and a.runtime_preflight: ap.error('--run and --runtime-preflight are mutually exclusive')
    if a.runtime_preflight:
        print(json.dumps(runtime_preflight(a.mechanism,a.legacy_root,a.out,registry_dir=a.registry_dir,
            historical_core_archive=a.historical_core_archive,attestation_dir=a.preflight_attestations),indent=2,sort_keys=True))
    elif a.run:
        print(json.dumps(run(a.mechanism,a.legacy_root,a.out,registry_dir=a.registry_dir,
            confirm_open=a.confirm_open,historical_core_archive=a.historical_core_archive,
            attestation_dir=a.preflight_attestations),indent=2,sort_keys=True))
    else:
        p=prepare(a.mechanism,a.legacy_root,registry_dir=a.registry_dir,historical_core_archive=a.historical_core_archive)
        print(json.dumps({'mechanism_id':p.mechanism_id,'evidence_ok':p.evidence_ok,'prereg_status':p.prereg_status,
          'recipe':None if p.recipe is None else p.recipe.__dict__,'seed_plan':p.seed_plan,'blocked_reasons':list(p.blocked_reasons)},indent=2,sort_keys=True))
if __name__=='__main__': main()
