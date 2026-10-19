#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description='Check v0.11.0 growth-first post-training gates.')
    p.add_argument('--expansion-result',required=True)
    p.add_argument('--training-result',required=True)
    p.add_argument('--bootstrap-result',required=True)
    p.add_argument('--world-retention-result',required=True)
    p.add_argument('--p4b-regression-result',required=True)
    p.add_argument('--generalized-result',required=True)
    args=p.parse_args()
    e,t,b,w,p4,g=map(load,[args.expansion_result,args.training_result,args.bootstrap_result,args.world_retention_result,args.p4b_regression_result,args.generalized_result])
    gates={
      'exact_function_preserving_expansion': bool(e.get('equivalence',{}).get('exact_function_preservation')) and int(e.get('legacy_tensor_mismatches',1))==0,
      'expanded_capacity_ge_50m': int(e.get('expanded_trainable_parameters',0))>=50_000_000,
      'growth_first_kept_legacy_tensors_exact': int(t.get('legacy_tensors_changed',1))==0 and t.get('plasticity')=='growth-only',
      'bootstrap_retention': bool(b.get('passed')),
      'foraging_detour_world_retention': bool(w.get('world_consolidation_passed')),
      'p4b_hierarchical_regression': bool(p4.get('p4b_passed')),
      'generalized_randomized_detour_development': float(g.get('completion_rate',0.0))>=0.20,
    }
    result={
      'format':'continuous_brain.v110_growth_readiness',
      'gates':gates,
      'growth_phase_passed':all(gates.values()),
      'meaning':'Readiness to consider controlled legacy-weight plasticity after growth-first training; not evidence of consciousness or AGI.',
    }
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
