#!/usr/bin/env python3
"""Live AI2-THOR manipulation smoke test for v0.8.0.

Tests one actually visible pickupable object, drops it, and opportunistically tests
one visible openable/toggleable object. It never uses forceAction=True.
"""
from __future__ import annotations
import argparse, math
from agent_core.models import Action
from agent_core.thor_adapter import ThorAdapter, ThorConfig
from agent_core.grounding import SemanticGrounder
from agent_core.executor import ActionExecutor
from agent_core.memory import EpisodicMemory
from agent_core.recovery import RecoveryManager


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scene',default='FloorPlan1'); ap.add_argument('--thor-executable',default=None); args=ap.parse_args()
    cfg=ThorConfig(scene=args.scene, width=320, height=240, render_instance_segmentation=True, local_executable_path=args.thor_executable, step_delay_seconds=0)
    thor=ThorAdapter(cfg); thor.start()
    try:
        mem=EpisodicMemory('/tmp/author-v080-interaction-smoke.jsonl',load_existing=False)
        ex=ActionExecutor(thor,mem,RecoveryManager(),max_steps=200,logger_path='/tmp/author-v080-actions.jsonl')
        g=SemanticGrounder(interaction_distance=cfg.interaction_distance)
        w=thor.observe()
        candidates=[o for o in w.objects.values() if o.pickupable and o.visible and o.distance is not None]
        if not candidates: raise RuntimeError('No visible pickupable object at initial pose; rotate/move or use another scene.')
        obj=min(candidates,key=lambda o:o.distance)
        if obj.distance>cfg.interaction_distance+0.15:
            out=ex.execute(g.ground('Approach',obj.object_type,{},w).action); assert out.success, out.error; w=thor.observe()
        out=ex.execute(g.ground('PickupObject',obj.object_type,{},w).action); assert out.success, out.error
        assert thor.observe().held_object_id==obj.object_id
        out=ex.execute(g.ground('DropObject',None,{},thor.observe()).action); assert out.success, out.error
        assert thor.observe().held_object_id is None
        print(f'PASS pickup/drop: {obj.object_type}')
        w=thor.observe()
        for purpose, pred in [('OpenObject',lambda o:o.openable and o.is_open is False),('ToggleObjectOn',lambda o:o.toggleable and o.is_toggled is False)]:
            choices=[o for o in w.objects.values() if o.visible and o.distance is not None and pred(o)]
            if not choices: continue
            target=min(choices,key=lambda o:o.distance)
            if target.distance>cfg.interaction_distance+0.15:
                try: ex.execute(g.ground('Approach',target.object_type,{},w).action); w=thor.observe()
                except Exception: continue
            try:
                out=ex.execute(g.ground(purpose,target.object_type,{},w).action)
                print(('PASS' if out.success else 'FAIL'), purpose, target.object_type, out.error or '')
            except Exception as e: print('SKIP',purpose,target.object_type,e)
    finally: thor.close()

if __name__=='__main__': main()
