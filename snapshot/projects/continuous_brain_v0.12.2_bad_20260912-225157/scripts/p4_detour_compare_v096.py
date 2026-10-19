from __future__ import annotations
import argparse, json
from continuous_brain.planning_v096 import compare_detour

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--episodes', type=int, default=100)
p.add_argument('--steps', type=int, default=120)
p.add_argument('--seed-start', type=int, default=9001)
p.add_argument('--horizons', type=int, nargs='+', default=[1,3,6])
p.add_argument('--beam-width', type=int, default=16)
p.add_argument('--device', default='cpu')
p.add_argument('--output', required=True)
a = p.parse_args()
r = compare_detour(a.checkpoint, episodes=a.episodes, steps=a.steps, seed_start=a.seed_start,
                   horizons=tuple(a.horizons), beam_width=a.beam_width,
                   device=a.device, output=a.output)
print(json.dumps({
    'reactive': r['reactive'],
    'horizons': r['horizons'],
    'p4_passed': r['p4_passed'],
    'full_result': a.output,
}, indent=2, sort_keys=True))
