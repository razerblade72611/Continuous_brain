#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_image_tensor, _env_context, _plan_action


def mean(xs):
    xs=list(xs); return float(statistics.fmean(xs)) if xs else 0.0


def tail_value(model, z, mode, device):
    if mode == 'value':
        return float(model.planning.value(z).item())
    if mode == 'qmax':
        ad = int(model.cfg.action_dim)
        zrep = z.expand(ad, -1)
        all_a = one_hot_action(torch.arange(ad, device=device), ad)
        return float(model.planning.action_value(zrep, all_a).max().item())
    if mode == 'none':
        return 0.0
    raise ValueError(mode)


def diverse_prune(expanded, beam_width, action_dim):
    groups = defaultdict(list)
    for node in expanded:
        groups[int(node['seq'][0])].append(node)
    for g in groups.values():
        g.sort(key=lambda x: x['rank'], reverse=True)
    quota = max(1, int(beam_width) // int(action_dim))
    kept=[]; used=set()
    for a in range(action_dim):
        for node in groups.get(a, [])[:quota]:
            kept.append(node); used.add(id(node))
    if len(kept) < beam_width:
        rest=[n for n in expanded if id(n) not in used]
        rest.sort(key=lambda x: x['rank'], reverse=True)
        kept.extend(rest[:beam_width-len(kept)])
    kept.sort(key=lambda x: x['rank'], reverse=True)
    return kept[:beam_width]


@torch.no_grad()
def plan_variant(model, spatial, context7, *, horizon, beam_width, discount, device, tail_mode, root_diverse):
    planner=model.planning; ad=int(model.cfg.action_dim)
    root=planner.encode_state(spatial, context7)
    beam=[{'latent':root,'score':0.0,'seq':[]}]
    for depth in range(int(horizon)):
        parent_indices=[]; actions_list=[]
        for pi in range(len(beam)):
            for a in range(ad):
                parent_indices.append(pi); actions_list.append(a)
        actions=torch.tensor(actions_list,dtype=torch.long,device=device)
        aoh=one_hot_action(actions,ad)
        latents=torch.cat([beam[i]['latent'] for i in parent_indices],dim=0)
        pred=planner.step(latents,aoh)
        expanded=[]
        for j,(pi,a) in enumerate(zip(parent_indices,actions_list)):
            parent=beam[pi]
            score=parent['score']+(float(discount)**depth)*float(pred['reward'][j].item())
            node={'latent':pred['latent'][j:j+1], 'score':score, 'seq':parent['seq']+[int(a)]}
            tail=tail_value(model,node['latent'],tail_mode,device)
            node['rank']=score+(float(discount)**(depth+1))*tail
            expanded.append(node)
        expanded.sort(key=lambda x:x['rank'], reverse=True)
        if root_diverse and depth+1 < int(horizon):
            beam=diverse_prune(expanded, int(beam_width), ad)
        else:
            beam=expanded[:max(1,int(beam_width))]
    best=max(beam,key=lambda x:x['rank'])
    return int(best['seq'][0]), float(best['rank']), list(best['seq'])


@torch.no_grad()
def run_episode(model, seed, *, mode, horizon, beam_width, max_steps, discount, device, tail_mode='qmax', root_diverse=False):
    env=DetourWorld(seed=int(seed),max_episode_steps=int(max_steps)); obs=env.reset()
    hidden=model.nucleus.initial_state(1,device)
    memory=torch.zeros(1,model.cfg.memory_dim,dtype=torch.float32,device=device)
    previous_action=torch.zeros(1,dtype=torch.long,device=device)
    total_reward=0.0; overrides=0; completed=False; steps_used=max_steps; chosen=Counter()
    for step in range(1,int(max_steps)+1):
        image=_env_image_tensor(env,model.cfg.image_size,device)
        context,context7=_env_context(obs,device)
        semantic=model.vision(image); spatial_online=model.spatial(image); fused=model.fuse_vision(semantic,spatial_online)
        nuc=model.nucleus(fused,context,memory,previous_action,hidden); hidden=nuc['hidden']
        reactive=int(torch.argmax(model.motor(nuc['workspace']),dim=-1)[0].item())
        if mode=='reactive':
            action=reactive
        elif mode=='original_q':
            stable=model.target_spatial_bus(image)
            action,_,_=_plan_action(model,stable,context7,horizon=int(horizon),beam_width=int(beam_width),discount=float(discount),device=device)
        else:
            stable=model.target_spatial_bus(image)
            action,_,_=plan_variant(model,stable,context7,horizon=int(horizon),beam_width=int(beam_width),discount=float(discount),device=device,tail_mode=tail_mode,root_diverse=root_diverse)
        chosen[action]+=1; overrides += int(action!=reactive)
        obs,reward,done,info=env.step(action); total_reward += float(reward)
        previous_action=torch.tensor([action+1],dtype=torch.long,device=device)
        if bool(info.get('reached')):
            completed=True; steps_used=step
        if done: break
    return {'seed':int(seed),'mode':mode,'horizon':int(horizon if mode!='reactive' else 0),'reward':float(total_reward),'completed':bool(completed),'steps':int(steps_used),'collisions':int(env.collisions),'final_energy':float(env.energy),'planner_overrides':int(overrides),'chosen_action_counts':{str(k):int(v) for k,v in sorted(chosen.items())}}


def summary(rows,max_steps):
    ac=Counter()
    for r in rows:
        ac.update({int(k):v for k,v in r['chosen_action_counts'].items()})
    return {'episodes':len(rows),'completion_rate':mean(float(r['completed']) for r in rows),'mean_reward':mean(r['reward'] for r in rows),'mean_collisions':mean(r['collisions'] for r in rows),'mean_planner_overrides':mean(r['planner_overrides'] for r in rows),'mean_steps_penalized':mean(r['steps'] if r['completed'] else max_steps+1 for r in rows),'chosen_action_counts':{str(k):int(v) for k,v in sorted(ac.items())}}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--episodes',type=int,default=10); ap.add_argument('--steps',type=int,default=120); ap.add_argument('--seed-start',type=int,default=9001)
    ap.add_argument('--horizons',type=int,nargs='+',default=[3,6,12]); ap.add_argument('--beam-width',type=int,default=32); ap.add_argument('--discount',type=float,default=0.95); ap.add_argument('--device',default='cpu'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); model,saved=load_checkpoint(args.checkpoint,args.device); model.eval()
    variants={
        'original_q':('qmax',False),
        'value':('value',False),
        'diverse_q':('qmax',True),
        'diverse_value':('value',True),
    }
    raw={'reactive':[]}; results={}
    for h in args.horizons:
        for name in variants: raw[f'{name}_h{h}']=[]
    for i in range(args.episodes):
        seed=args.seed_start+i
        raw['reactive'].append(run_episode(model,seed,mode='reactive',horizon=1,beam_width=args.beam_width,max_steps=args.steps,discount=args.discount,device=args.device))
        for h in args.horizons:
            for name,(tail,diverse) in variants.items():
                raw[f'{name}_h{h}'].append(run_episode(model,seed,mode=name,horizon=h,beam_width=args.beam_width,max_steps=args.steps,discount=args.discount,device=args.device,tail_mode=tail,root_diverse=diverse))
        print(f'search_probe_episodes={i+1}/{args.episodes}')
    r=summary(raw['reactive'],args.steps)
    for key,rows in raw.items():
        if key=='reactive': continue
        s=summary(rows,args.steps)
        s['completion_advantage_pp']=100*(s['completion_rate']-r['completion_rate'])
        s['collisions_reduced']=r['mean_collisions']-s['mean_collisions']
        s['reward_absolute']=s['mean_reward']-r['mean_reward']
        results[key]=s
    out={'format':'continuous_brain.v100_search_probe.v1','checkpoint':str(Path(args.checkpoint).resolve()),'experiment':{'episodes':args.episodes,'seed_start':args.seed_start,'seed_end':args.seed_start+args.episodes-1,'horizons':args.horizons,'beam_width':args.beam_width,'discount':args.discount,'training_performed':False,'parents_used':False,'blind_evidence':False,'purpose':'diagnose terminal heuristic and beam collapse on already-used development seeds'},'reactive':r,'variants':results,'episodes':raw}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True))
    print(json.dumps({'full_result':str(p),'reactive':r,'variants':results},indent=2,sort_keys=True))

if __name__=='__main__': main()
