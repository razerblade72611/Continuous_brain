#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_image_tensor, _env_context
from continuous_brain.planning_v102 import _resolve_option


def mean(xs):
    xs = list(xs)
    return float(statistics.fmean(xs)) if xs else 0.0


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summary(rows, max_steps):
    return {
        'episodes': len(rows),
        'completion_rate': mean(float(r['completed']) for r in rows),
        'mean_reward': mean(r['reward'] for r in rows),
        'mean_collisions': mean(r['collisions'] for r in rows),
        'mean_steps_penalized': mean(r['steps'] if r['completed'] else max_steps+1 for r in rows),
        'mean_option_switches': mean(r['option_switches'] for r in rows),
        'mean_planner_overrides': mean(r['planner_overrides'] for r in rows),
    }


@torch.no_grad()
def topk_option_plan_action(model, spatial, context7, current_option, option_age, *, horizon, beam_width,
                            discount, min_commit, terminate_threshold, proposal_topk, device):
    planner, options = model.planning, model.options
    action_dim = int(model.cfg.action_dim)
    root = planner.encode_state(spatial, context7)
    root_option, root_age, switched = _resolve_option(
        model, root, current_option, option_age, min_commit, terminate_threshold
    )
    beam = [{'latent': root, 'score': 0.0, 'seq': [], 'option': root_option, 'age': root_age}]
    k = max(1, min(int(proposal_topk), action_dim))

    for depth in range(int(horizon)):
        expanded = []
        for node in beam:
            z = node['latent']
            option = int(node['option'])
            o = torch.tensor([option], dtype=torch.long, device=device)
            logits = options.action_logits(z, o)[0]
            candidate_actions = torch.topk(logits, k=k, dim=-1).indices.tolist()
            actions = torch.tensor(candidate_actions, dtype=torch.long, device=device)
            aoh = one_hot_action(actions, action_dim)
            zrep = z.expand(k, -1)
            pred = planner.step(zrep, aoh)
            for j, action in enumerate(candidate_actions):
                nz = pred['latent'][j:j+1]
                nage = int(node['age']) + 1
                noption, nage2, _ = _resolve_option(
                    model, nz, option, nage, min_commit, terminate_threshold
                )
                score = float(node['score']) + (float(discount) ** depth) * float(pred['reward'][j].item())
                expanded.append({
                    'latent': nz,
                    'score': score,
                    'seq': node['seq'] + [int(action)],
                    'option': int(noption),
                    'age': int(nage2),
                })
        for node in expanded:
            z = node['latent']
            option = int(node['option'])
            o = torch.tensor([option], dtype=torch.long, device=device)
            logits = options.action_logits(z, o)[0]
            tail_actions = torch.topk(logits, k=k, dim=-1).indices
            orepeat = torch.full((k,), option, dtype=torch.long, device=device)
            zrep = z.expand(k, -1)
            aoh = one_hot_action(tail_actions, action_dim)
            tail = options.action_value(zrep, orepeat, aoh).max().item()
            node['rank'] = float(node['score']) + (float(discount) ** (depth + 1)) * float(tail)
        expanded.sort(key=lambda x: x['rank'], reverse=True)
        beam = expanded[:max(1, int(beam_width))]

    best = beam[0]
    return int(best['seq'][0]), int(root_option), int(root_age), bool(switched), list(best['seq'])


@torch.no_grad()
def run_episode(model, seed, *, proposal_topk, horizon, beam_width, max_steps, discount,
                min_commit, terminate_threshold, device):
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    current_option = None
    option_age = 0
    option_switches = 0
    planner_overrides = 0
    action_counts = {str(i): 0 for i in range(model.cfg.action_dim)}
    option_counts = {str(i): 0 for i in range(model.cfg.option_count)}
    total_reward = 0.0
    completed = False
    steps_used = int(max_steps)

    for step in range(1, int(max_steps)+1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        context, context7 = _env_context(obs, device)
        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc['hidden']
        reactive = int(torch.argmax(model.motor(nuc['workspace']), dim=-1)[0].item())
        stable_spatial = model.target_spatial_bus(image)
        action, current_option, option_age, switched, _ = topk_option_plan_action(
            model, stable_spatial, context7, current_option, option_age,
            horizon=horizon, beam_width=beam_width, discount=discount,
            min_commit=min_commit, terminate_threshold=terminate_threshold,
            proposal_topk=proposal_topk, device=device,
        )
        option_counts[str(int(current_option))] += 1
        option_switches += int(switched)
        planner_overrides += int(action != reactive)
        option_age += 1
        action_counts[str(action)] += 1
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor([action+1], dtype=torch.long, device=device)
        if bool(info.get('reached')):
            completed = True
            steps_used = step
        if done:
            break

    return {
        'seed': int(seed), 'proposal_topk': int(proposal_topk), 'horizon': int(horizon),
        'completed': bool(completed), 'steps': int(steps_used), 'collisions': int(env.collisions),
        'reward': float(total_reward), 'option_switches': int(option_switches),
        'planner_overrides': int(planner_overrides), 'action_counts': action_counts,
        'option_counts': option_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--seed-start', type=int, default=9001)
    ap.add_argument('--horizons', type=int, nargs='+', default=[3,6,12])
    ap.add_argument('--proposal-topk', type=int, nargs='+', default=[1,2,3])
    ap.add_argument('--beam-width', type=int, default=32)
    ap.add_argument('--discount', type=float, default=0.95)
    ap.add_argument('--min-commit', type=int, default=3)
    ap.add_argument('--terminate-threshold', type=float, default=0.50)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    model, saved = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = dict(saved.get('metadata') or {})
    if metadata.get('option_cortex_version') != 'latent-temporal-options-v1':
        raise RuntimeError('checkpoint has not completed v0.10.2 option training')

    variants = {}
    all_rows = {}
    for k in args.proposal_topk:
        for h in args.horizons:
            key = f'top{k}_h{h}'
            rows = []
            for i in range(args.episodes):
                seed = args.seed_start + i
                rows.append(run_episode(
                    model, seed, proposal_topk=k, horizon=h, beam_width=args.beam_width,
                    max_steps=args.steps, discount=args.discount, min_commit=args.min_commit,
                    terminate_threshold=args.terminate_threshold, device=args.device,
                ))
            variants[key] = summary(rows, args.steps)
            all_rows[key] = rows
            print(f'option_proposal_probe={key} completion={variants[key]["completion_rate"]:.3f} collisions={variants[key]["mean_collisions"]:.2f}')

    result = {
        'format': 'continuous_brain.v102_option_policy_proposal_probe.v1',
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha(args.checkpoint),
        'blind_evidence': False,
        'training_performed': False,
        'parents_used': False,
        'memory_written': False,
        'seed_start': args.seed_start,
        'seed_end': args.seed_start + args.episodes - 1,
        'episodes': args.episodes,
        'proposal_rule': 'At every imagined node, expand only the top-k primitive actions from the learned option action policy; recurrent imagination ranks those proposals by predicted discounted reward plus option-conditioned terminal value.',
        'no_gap_wall_labels': True,
        'variants': variants,
        'episode_rows': all_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({'full_result': str(out), 'variants': variants}, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
