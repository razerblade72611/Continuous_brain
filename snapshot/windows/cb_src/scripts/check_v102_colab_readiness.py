#!/usr/bin/env python3
"""Check the post-v0.10.2 gates for the first major whole-brain Colab run."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load_json(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-v101-checkpoint', required=True)
    p.add_argument('--current-checkpoint', required=True)
    p.add_argument('--planning-result', required=True)
    p.add_argument('--p3-result', required=True)
    p.add_argument('--retention-result', required=True)
    a=p.parse_args()
    source=torch.load(a.source_v101_checkpoint,map_location='cpu',weights_only=False)
    current=torch.load(a.current_checkpoint,map_location='cpu',weights_only=False)
    changed=[]; missing=[]
    for key,before in source['model'].items():
        if key.startswith('options.'): continue
        if key not in current['model']: missing.append(key)
        elif not torch.equal(before,current['model'][key]): changed.append(key)
    lineage_ok=not changed and not missing
    planning=load_json(a.planning_result); p3=load_json(a.p3_result); retention=load_json(a.retention_result)
    meta=dict(current.get('metadata') or {})
    caps=retention.get('capability_scores') or {}; metrics=retention.get('metrics') or {}
    hist=metrics.get('vision_retention_min')
    retention_ok=(bool(retention.get('passed')) and float(caps.get('vision',0))>=.90 and float(caps.get('language',0))>=.85 and (hist is None or float(hist)>=.90))
    gates={
      'p4b_blind_temporal_option_planning': bool(planning.get('p4b_passed')),
      'p3_world_model_retained': bool(p3.get('passed')) and bool(p3.get('world_model_passed')),
      'bootstrap_retention': retention_ok,
      'pre_v102_tensors_bit_exact': lineage_ok,
      'option_cortex_trained': meta.get('option_cortex_version')=='latent-temporal-options-v1' and int(meta.get('option_updates',0))>0,
    }
    result={
      'format':'continuous_brain.colab_readiness.v102',
      'source_v101_sha256':sha(a.source_v101_checkpoint),
      'current_sha256':sha(a.current_checkpoint),
      'changed_pre_v102_tensor_count':len(changed),
      'missing_pre_v102_tensor_count':len(missing),
      'gates':gates,
      'colab_ready':all(gates.values()),
      'meaning':'Engineering readiness for major whole-brain training; not evidence of consciousness or AGI.',
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['colab_ready'] else 2)

if __name__=='__main__': main()
