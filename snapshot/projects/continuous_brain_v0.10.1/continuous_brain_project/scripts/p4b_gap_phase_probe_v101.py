#!/usr/bin/env python3
"""Development-only localization probe for v0.10.1 detour planning.

No training, no parents, no memory writes. Uses already-spent seeds only.
The privileged control uses known DetourWorld gap geometry and therefore MUST
NOT be treated as planning evidence or qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_image_tensor, _env_context


def all_q(model, z):
    ad = int(model.cfg.action_dim)
    zrep = z.expand(ad, -1)
    aoh = one_hot_action(torch.arange(ad, device=z.device), ad)
    return model.planning.action_value(zrep, aoh)


def phase(env: DetourWorld):
    x, y = float(env.agent[0]), float(env.agent[1])
    lo = float(env.gap_center_y - env.gap_half_height)
    hi = float(env.gap_center_y + env.gap_half_height)
    if x > float(env.wall_half_width):
        return "right_of_wall"
    if x >= -float(env.wall_half_width):
        return "wall_band"
    if y < lo:
        return "left_below_gap"
    if y > hi:
        return "left_above_gap"
    return "left_in_gap"


@torch.no_grad()
def run_episode(model, seed, steps, device, privileged_gap_cross=False):
    env = DetourWorld(seed=int(seed), max_episode_steps=int(steps))
    obs = env.reset()  # Match evaluate-planning's second deterministic reset.
    total_reward = 0.0
    counts = Counter()
    phase_counts = defaultdict(Counter)
    q_sums = defaultdict(lambda: [0.0] * int(model.cfg.action_dim))
    q_ns = Counter()
    reached_gap = False
    crossed = False
    first_gap_step = None
    first_cross_step = None
    max_y = float(env.agent[1])
    max_x = float(env.agent[0])
    completed = False
    overrides = 0

    for step in range(1, int(steps) + 1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        _, ctx7 = _env_context(obs, device)
        spatial = model.target_spatial_bus(image)
        z = model.planning.encode_state(spatial, ctx7)
        q = all_q(model, z)
        direct = int(torch.argmax(q).item())
        ph = phase(env)
        phase_counts[ph][direct] += 1
        q_ns[ph] += 1
        for a, val in enumerate(q.detach().cpu().tolist()):
            q_sums[ph][a] += float(val)

        action = direct
        if privileged_gap_cross and ph == "left_in_gap":
            # Diagnostic oracle: geometry is supplied by the evaluator, not learned.
            action = 4
            overrides += int(action != direct)

        counts[action] += 1
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        max_y = max(max_y, float(env.agent[1]))
        max_x = max(max_x, float(env.agent[0]))

        in_gap_now = abs(float(env.agent[1]) - float(env.gap_center_y)) <= float(env.gap_half_height)
        if in_gap_now:
            reached_gap = True
            if first_gap_step is None:
                first_gap_step = step
        if float(env.agent[0]) > float(env.wall_half_width):
            crossed = True
            if first_cross_step is None:
                first_cross_step = step
        if bool(info.get("reached")):
            completed = True
        if done:
            break

    phase_out = {}
    for ph, ctr in phase_counts.items():
        n = int(q_ns[ph])
        phase_out[ph] = {
            "states": n,
            "direct_q_action_counts": {str(a): int(ctr.get(a, 0)) for a in range(int(model.cfg.action_dim))},
            "mean_q": {str(a): (q_sums[ph][a] / n if n else 0.0) for a in range(int(model.cfg.action_dim))},
        }

    return {
        "seed": int(seed),
        "mode": "privileged_gap_cross" if privileged_gap_cross else "direct_q",
        "completed": bool(completed),
        "reward": float(total_reward),
        "collisions": int(env.collisions),
        "chosen_action_counts": {str(a): int(counts.get(a, 0)) for a in range(int(model.cfg.action_dim))},
        "privileged_overrides": int(overrides),
        "reached_gap_band": bool(reached_gap),
        "crossed_wall": bool(crossed),
        "first_gap_step": first_gap_step,
        "first_cross_step": first_cross_step,
        "max_y": float(max_y),
        "max_x": float(max_x),
        "final_x": float(env.agent[0]),
        "final_y": float(env.agent[1]),
        "phase_q": phase_out,
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows):
    actions = Counter()
    phase_actions = defaultdict(Counter)
    phase_states = Counter()
    phase_q_sums = defaultdict(lambda: [0.0] * 5)
    phase_q_weight = Counter()
    for r in rows:
        for a, n in r["chosen_action_counts"].items():
            actions[a] += int(n)
        for ph, p in r["phase_q"].items():
            n = int(p["states"])
            phase_states[ph] += n
            for a, c in p["direct_q_action_counts"].items():
                phase_actions[ph][a] += int(c)
            for a, q in p["mean_q"].items():
                phase_q_sums[ph][int(a)] += float(q) * n
            phase_q_weight[ph] += n
    phase_summary = {}
    for ph in sorted(phase_states):
        n = int(phase_states[ph])
        phase_summary[ph] = {
            "states": n,
            "direct_q_action_counts": {str(a): int(phase_actions[ph].get(str(a), 0)) for a in range(5)},
            "mean_q": {str(a): (phase_q_sums[ph][a] / n if n else 0.0) for a in range(5)},
        }
    return {
        "episodes": len(rows),
        "completion_rate": mean(float(r["completed"]) for r in rows),
        "episodes_reaching_gap_band": sum(int(r["reached_gap_band"]) for r in rows),
        "episodes_crossing_wall": sum(int(r["crossed_wall"]) for r in rows),
        "mean_reward": mean(r["reward"] for r in rows),
        "mean_collisions": mean(r["collisions"] for r in rows),
        "mean_max_y": mean(r["max_y"] for r in rows),
        "mean_max_x": mean(r["max_x"] for r in rows),
        "mean_final_x": mean(r["final_x"] for r in rows),
        "mean_final_y": mean(r["final_y"] for r in rows),
        "mean_privileged_overrides": mean(r["privileged_overrides"] for r in rows),
        "chosen_action_counts": {str(a): int(actions.get(str(a), 0)) for a in range(5)},
        "phase_q": phase_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed-start", type=int, default=9001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model, saved = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    direct, privileged = [], []
    for i in range(args.episodes):
        seed = args.seed_start + i
        direct.append(run_episode(model, seed, args.steps, args.device, False))
        privileged.append(run_episode(model, seed, args.steps, args.device, True))
        print(f"gap_phase_probe={i+1}/{args.episodes}")

    result = {
        "format": "continuous_brain.v101_gap_phase_probe.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "episodes": args.episodes,
        "steps": args.steps,
        "action_mapping": {"0":"idle","1":"up","2":"down","3":"left","4":"right"},
        "gap": {"center_y": 0.58, "half_height": 0.16},
        "training_performed": False,
        "parents_used": False,
        "memory_written": False,
        "blind_evidence": False,
        "privileged_control_is_qualification_evidence": False,
        "diagnostic_question": "When direct Q reaches the gap on the left, does it choose RIGHT and cross?",
        "summaries": {
            "direct_q": summarize(direct),
            "privileged_gap_cross": summarize(privileged),
        },
        "episodes_detail": {
            "direct_q": direct,
            "privileged_gap_cross": privileged,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "full_result": str(out),
        "direct_q": result["summaries"]["direct_q"],
        "privileged_gap_cross": result["summaries"]["privileged_gap_cross"],
    }, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
