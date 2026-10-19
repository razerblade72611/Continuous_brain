#!/usr/bin/env python3
"""Mechanical smoke test for the v0.10.9 runtime-sanitized maneuver-clear Discovery World.

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
            current=next((o for o in ev.metadata.get('objects',[]) if o.get('objectId')==d['objectId']),None)
            print('Opened:', d['objectId'], 'isOpen=', None if current is None else current.get('isOpen'))
        # Give transforms/physics one additional command boundary before reachability.
        c.step(action='Pass')
        ev=c.step(action='GetReachablePositions')
        pts=ev.metadata.get('actionReturn') or []
        north=sum(1 for p in pts if p['z']>3.2)
        east=sum(1 for p in pts if p['x']>3.2)
        print('Reachable positions:',len(pts),'north-extension:',north,'east-extension:',east)
        if pts:
            print('Reachability extents: max_x=', max(float(p['x']) for p in pts), 'max_z=', max(float(p['z']) for p in pts))
        if north==0 or east==0:
            raise RuntimeError('AI2-THOR reachability flood-fill does not reach both Discovery Rooms after doors open.')
        objects=c.last_event.metadata.get('objects',[])
        stool_rows=[o for o in objects if 'stool' in str(o.get('objectType','')).lower() or 'stool' in str(o.get('objectId','')).lower() or 'stool' in str(o.get('assetId','')).lower()]
        print('Stool objects:',[(o.get('objectType'),o.get('objectId')) for o in stool_rows])
        if stool_rows:
            raise RuntimeError('v0.10.9 requires all stools to be removed from the Discovery scene.')

        north_turn=[p for p in pts if abs(float(p['x']) + 0.2) <= 1.10 and abs(float(p['z']) - 5.4) <= 1.10]
        east_turn=[p for p in pts if abs(float(p['x']) - 5.3) <= 1.10 and abs(float(p['z']) - 0.4) <= 1.10]
        north_corridor=[p for p in pts if abs(float(p['x']) + 0.2) <= 0.625 and 2.75 <= float(p['z']) <= 5.4]
        east_corridor=[p for p in pts if 2.70 <= float(p['x']) <= 5.3 and abs(float(p['z']) - 0.4) <= 0.625]
        print('Maneuver cells: north_turn=',len(north_turn),'east_turn=',len(east_turn),'north_corridor=',len(north_corridor),'east_corridor=',len(east_corridor))
        if len(north_turn) < 25 or len(east_turn) < 25:
            raise RuntimeError('Discovery Room turnaround zone does not contain enough reachable grid cells.')
        if len(north_corridor) < 8 or len(east_corridor) < 8:
            raise RuntimeError('Discovery Room doorway corridor is not sufficiently reachable.')

        types={o.get('objectType') for o in objects}
        interesting=sorted(types & {'Laptop','Book','CellPhone','KeyChain','Statue','Box','DeskLamp','RemoteControl','AlarmClock','CD'})
        print('Discovery interactive object types:',interesting)
        if len(interesting)<4: raise RuntimeError('Too few interactive Discovery Room object types present.')
        print('PASS: exits open, both rooms are reachable, stools are absent, maneuver zones are open, and interactive content is present.')
    finally:
        c.stop()
if __name__=='__main__': main()
