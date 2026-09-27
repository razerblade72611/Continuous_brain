#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from continuous_brain_unified.legacy.g0_runner import prepare, run

def main():
    ap=argparse.ArgumentParser(description='Prepare or execute G0 isolated legacy requalification. Scientific run is locked until prereg approval.')
    ap.add_argument('--legacy-root',required=True); ap.add_argument('--mechanism',required=True); ap.add_argument('--registry-dir',default=None)
    ap.add_argument('--run',action='store_true'); ap.add_argument('--out',default='runs/g0_requalification'); a=ap.parse_args()
    if a.run: print(json.dumps(run(a.mechanism,a.legacy_root,a.out,registry_dir=a.registry_dir),indent=2,sort_keys=True))
    else:
        p=prepare(a.mechanism,a.legacy_root,registry_dir=a.registry_dir)
        print(json.dumps({'mechanism_id':p.mechanism_id,'evidence_ok':p.evidence_ok,'prereg_status':p.prereg_status,
          'recipe':None if p.recipe is None else p.recipe.__dict__,'seed_plan':p.seed_plan,'blocked_reasons':list(p.blocked_reasons)},indent=2,sort_keys=True))
if __name__=='__main__': main()
