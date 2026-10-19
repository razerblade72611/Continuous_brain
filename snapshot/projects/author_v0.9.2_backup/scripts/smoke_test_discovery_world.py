#!/usr/bin/env python3
"""Mechanical smoke test for the v0.9.2 connected Discovery World.

This is a test harness, not agent behavior. It may use forceAction to open the doors
from the test process so geometry/reachability can be verified independently of LLM policy.
"""
from __future__ import annotations
import argparse
from ai2thor.controller import Controller

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scene',default='FloorPlan1Discovery_physics')
    ap.add_argument('--thor-executable',required=True)
    a=ap.parse_args()
    c=Controller(scene=a.scene,local_executable_path=a.thor_executable,width=320,height=240)
    try:
        e=c.last_event
        doors=[o for o in e.metadata.get('objects',[]) if o.get('objectType')=='Door']
        print('Door objects:',[(o['objectId'],o.get('openable'),o.get('isOpen')) for o in doors])
        if len(doors)<2: raise RuntimeError(f'Expected >=2 discovery Door objects, found {len(doors)}')
        for d in doors[:2]:
            ev=c.step(action='OpenObject',objectId=d['objectId'],forceAction=True)
            if not ev.metadata.get('lastActionSuccess'):
                raise RuntimeError('OpenObject failed for '+d['objectId']+': '+ev.metadata.get('errorMessage',''))
        ev=c.step(action='GetReachablePositions')
        pts=ev.metadata.get('actionReturn') or []
        north=sum(1 for p in pts if p['z']>3.2)
        east=sum(1 for p in pts if p['x']>3.2)
        print('Reachable positions:',len(pts),'north-extension:',north,'east-extension:',east)
        if pts:
            print('Reachability extents: max_x=', max(float(p['x']) for p in pts), 'max_z=', max(float(p['z']) for p in pts))
        if north==0 or east==0:
            raise RuntimeError('AI2-THOR reachability flood-fill does not reach both Discovery Rooms after doors open.')
        types={o.get('objectType') for o in c.last_event.metadata.get('objects',[])}
        interesting=sorted(types & {'Laptop','Book','CellPhone','KeyChain','Statue','Box','DeskLamp','RemoteControl','AlarmClock','CD'})
        print('Discovery interactive object types:',interesting)
        if len(interesting)<4: raise RuntimeError('Too few interactive Discovery Room object types present.')
        print('PASS: two openable exits, both room reachability extensions, and interactive content are present.')
    finally:
        c.stop()
if __name__=='__main__': main()
