#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

p=argparse.ArgumentParser(description='Verify Continuous Brain state/snapshots without modifying them.')
p.add_argument('state_dir', nargs='?', default='state/embodied_brain_v050_spatial')
a=p.parse_args(); root=Path(a.state_dir)
print(f'state: {root}')
for name in ('brain.pt','brain.previous.pt','replay.pt','status.json'):
    q=root/name
    print(f'{name:20s}', 'OK' if q.exists() else 'missing', q.stat().st_size if q.exists() else '')
if (root/'brain.pt').exists():
    d=torch.load(root/'brain.pt',map_location='cpu',weights_only=False)
    print('checkpoint:', {'version':d.get('version'),'format_version':d.get('format_version'),'steps':d.get('steps'),'mastered_levels':d.get('mastered_levels')})
    env=d.get('environment_state') or {}
    sp=env.get('spatial') if isinstance(env,dict) else None
    if isinstance(sp,dict):
        arr=sp.get('log_odds')
        print('spatial:', {'scene':sp.get('scene'),'resolution':sp.get('resolution'),'shape':getattr(arr,'shape',None),'landmarks':len(sp.get('landmarks',{})),'map_updates':sp.get('map_updates'),'plans':sp.get('plans'),'plan_failures':sp.get('plan_failures')})
skills=root/'skills'
if skills.exists():
    for q in sorted(skills.glob('L*_mastered.pt')):
        try:
            d=torch.load(q,map_location='cpu',weights_only=False)
            print('skill:', q.name, {'version':d.get('version'),'level':d.get('skill_level'),'steps':d.get('steps')})
        except Exception as e:
            print('skill:', q.name, 'ERROR', e)
