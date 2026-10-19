#!/usr/bin/env python3
"""Check v0.10.4 readiness for the first major whole-brain Colab phase."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-v103-checkpoint', required=True,
                   help='Frozen P4b-passing v0.10.3 checkpoint before world consolidation')
    p.add_argument('--current-checkpoint', required=True)
    p.add_argument('--p4b-result', required=True,
                   help='Frozen 100-episode v0.10.3 blind P4b result')
    p.add_argument('--world-retention-result', required=True,
                   help='Fresh sealed v0.10.4 foraging+detour world retention result')
    p.add_argument('--bootstrap-retention-result', required=True)
    a = p.parse_args()

    source = torch.load(a.source_v103_checkpoint, map_location='cpu', weights_only=False)
    current = torch.load(a.current_checkpoint, map_location='cpu', weights_only=False)
    changed_non_world, missing = [], []
    changed_world = []
    for key, before in source['model'].items():
        if key not in current['model']:
            missing.append(key)
            continue
        if not torch.equal(before, current['model'][key]):
            if key.startswith('world.'):
                changed_world.append(key)
            else:
                changed_non_world.append(key)
    extra = sorted(set(current['model']) - set(source['model']))

    p4b = load_json(a.p4b_result)
    world = load_json(a.world_retention_result)
    bootstrap = load_json(a.bootstrap_retention_result)
    meta = dict(current.get('metadata') or {})
    caps = bootstrap.get('capability_scores') or {}
    metrics = bootstrap.get('metrics') or {}
    hist = metrics.get('vision_retention_min')
    bootstrap_ok = (
        bool(bootstrap.get('passed'))
        and float(caps.get('vision', 0.0)) >= .90
        and float(caps.get('language', 0.0)) >= .85
        and (hist is None or float(hist) >= .90)
    )
    domains = world.get('domains') or {}
    world_ok = (
        bool(world.get('world_consolidation_passed'))
        and bool((domains.get('foraging') or {}).get('world_model_passed'))
        and bool((domains.get('detour') or {}).get('world_model_passed'))
    )
    frozen_non_world_ok = not changed_non_world and not missing and not extra and bool(changed_world)
    gates = {
        'p4b_blind_hierarchical_option_planning': bool(p4b.get('p4b_passed')),
        'fresh_foraging_and_detour_world_retention': world_ok,
        'bootstrap_vision_language_retention': bootstrap_ok,
        'v103_to_v104_only_world_tensors_changed': frozen_non_world_ok,
        'world_consolidation_metadata': (
            meta.get('world_consolidation_version') == 'balanced-multidomain-action-rehearsal-v1'
            and int(meta.get('world_consolidation_updates', 0)) > 0
        ),
        'hierarchical_option_planner_preserved': (
            meta.get('option_cortex_version') == 'latent-temporal-options-v1'
            and int(meta.get('option_updates', 0)) > 0
        ),
    }
    result = {
        'format': 'continuous_brain.colab_readiness.v104',
        'source_v103_sha256': sha(a.source_v103_checkpoint),
        'current_sha256': sha(a.current_checkpoint),
        'changed_world_tensor_count': len(changed_world),
        'changed_non_world_tensor_count': len(changed_non_world),
        'missing_tensor_count': len(missing),
        'extra_tensor_count': len(extra),
        'gates': gates,
        'colab_ready': all(gates.values()),
        'meaning': 'Engineering readiness for major whole-brain training; not evidence of consciousness or AGI.',
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['colab_ready'] else 2)


if __name__ == '__main__':
    main()
