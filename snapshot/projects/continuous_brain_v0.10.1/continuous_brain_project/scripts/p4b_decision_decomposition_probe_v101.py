#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import statistics

import numpy as np
import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v101 import _all_action_values

ACTION_NAMES = {0: "idle", 1: "up", 2: "down", 3: "left", 4: "right"}


def _mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


@torch.no_grad()
def _state_latent(model, env: DetourWorld, obs: np.ndarray, device: str):
    image = _env_image_tensor(env, model.cfg.image_size, device)
    _, context7 = _env_context(obs, device)
    spatial = model.target_spatial_bus(image)
    return model.planning.encode_state(spatial, context7)


@torch.no_grad()
def _scores(model, env: DetourWorld, obs: np.ndarray, *, gamma: float, device: str):
    """Return direct-Q, predicted-Bellman, and true-next-Bellman scores.

    true_next_bellman is diagnostic-only and uses simulator clones.  It is never
    used as evidence of child-only planning; it only localizes whether errors come
    from the learned Q function or the imagined one-step latent/reward model.
    """
    planner = model.planning
    ad = int(model.cfg.action_dim)
    root = _state_latent(model, env, obs, device)
    actions = torch.arange(ad, dtype=torch.long, device=device)
    roots = root.expand(ad, -1)
    aoh = one_hot_action(actions, ad)

    direct_q = planner.action_value(roots, aoh)
    pred = planner.step(roots, aoh)
    pred_next_q = _all_action_values(planner, pred["latent"]).max(dim=1).values
    predicted_bellman = pred["reward"] + float(gamma) * pred_next_q

    true_next_bellman = []
    true_reward = []
    true_collision = []
    true_next_y = []
    for a in range(ad):
        clone = copy.deepcopy(env)
        next_obs, reward, _, info = clone.step(a)
        next_z = _state_latent(model, clone, next_obs, device)
        next_q = float(_all_action_values(planner, next_z).max(dim=1).values[0].item())
        true_next_bellman.append(float(reward) + float(gamma) * next_q)
        true_reward.append(float(reward))
        true_collision.append(bool(info.get("collision", False)))
        true_next_y.append(float(next_obs[1]))

    return {
        "direct_q": [float(x) for x in direct_q.cpu().tolist()],
        "predicted_bellman": [float(x) for x in predicted_bellman.cpu().tolist()],
        "true_next_bellman": true_next_bellman,
        "predicted_reward": [float(x) for x in pred["reward"].cpu().tolist()],
        "true_reward": true_reward,
        "predicted_next_y": [float(x) for x in pred["context"][:, 1].cpu().tolist()],
        "true_next_y": true_next_y,
        "true_collision": true_collision,
    }


def _argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


@torch.no_grad()
def _run_episode(model, seed: int, mode: str, *, steps: int, gamma: float, device: str):
    env = DetourWorld(seed=int(seed), max_episode_steps=int(steps))
    obs = env.reset()
    total_reward = 0.0
    completed = False
    actions = Counter()
    y_values = [float(obs[1])]
    x_values = [float(obs[0])]
    first_cross_step = None
    first_gap_band_step = None
    score_snapshots = []

    for step in range(1, int(steps) + 1):
        scores = _scores(model, env, obs, gamma=gamma, device=device)
        if mode == "direct_q":
            action = _argmax(scores["direct_q"])
        elif mode == "predicted_bellman":
            action = _argmax(scores["predicted_bellman"])
        elif mode == "true_next_bellman":
            action = _argmax(scores["true_next_bellman"])
        else:
            raise ValueError(mode)

        if step <= 8 or (first_gap_band_step is None and float(obs[1]) >= env.gap_center_y - env.gap_half_height - 0.08):
            score_snapshots.append({
                "step": step,
                "agent_x": float(obs[0]),
                "agent_y": float(obs[1]),
                "chosen_action": int(action),
                "chosen_action_name": ACTION_NAMES[int(action)],
                "direct_q": scores["direct_q"],
                "predicted_bellman": scores["predicted_bellman"],
                "true_next_bellman": scores["true_next_bellman"],
            })

        prev_x = float(obs[0])
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        actions[int(action)] += 1
        y_values.append(float(obs[1]))
        x_values.append(float(obs[0]))

        if first_gap_band_step is None and env._in_gap(float(obs[1])):
            first_gap_band_step = step
        if first_cross_step is None and prev_x < -env.wall_half_width and float(obs[0]) > env.wall_half_width:
            first_cross_step = step
        if bool(info.get("reached", False)):
            completed = True
        if done:
            break

    return {
        "seed": int(seed),
        "mode": mode,
        "completed": bool(completed),
        "reward": float(total_reward),
        "collisions": int(env.collisions),
        "steps_executed": int(env.steps),
        "action_counts": {str(a): int(actions.get(a, 0)) for a in range(env.action_dim)},
        "action_names": ACTION_NAMES,
        "max_y": float(max(y_values)),
        "min_y": float(min(y_values)),
        "max_x": float(max(x_values)),
        "first_gap_band_step": first_gap_band_step,
        "first_cross_step": first_cross_step,
        "final_x": float(obs[0]),
        "final_y": float(obs[1]),
        "score_snapshots": score_snapshots,
    }


def _summary(rows):
    action_totals = Counter()
    for r in rows:
        for a, n in r["action_counts"].items():
            action_totals[int(a)] += int(n)
    return {
        "episodes": len(rows),
        "completion_rate": _mean(float(r["completed"]) for r in rows),
        "mean_reward": _mean(r["reward"] for r in rows),
        "mean_collisions": _mean(r["collisions"] for r in rows),
        "mean_max_y": _mean(r["max_y"] for r in rows),
        "mean_final_y": _mean(r["final_y"] for r in rows),
        "episodes_reaching_gap_band": sum(r["first_gap_band_step"] is not None for r in rows),
        "episodes_crossing_wall": sum(r["first_cross_step"] is not None for r in rows),
        "chosen_action_counts": {str(a): int(action_totals.get(a, 0)) for a in range(5)},
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed-start", type=int, default=9001)
    ap.add_argument("--discount", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model, saved = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})

    modes = ("direct_q", "predicted_bellman", "true_next_bellman")
    raw = {m: [] for m in modes}
    for i in range(int(args.episodes)):
        seed = int(args.seed_start) + i
        for mode in modes:
            raw[mode].append(_run_episode(
                model, seed, mode, steps=args.steps, gamma=args.discount, device=args.device,
            ))
        print(f"decision_probe_episodes={i+1}/{args.episodes}")

    result = {
        "format": "continuous_brain.v101_decision_decomposition_probe.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "planning_value_version": metadata.get("planning_value_version"),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.episodes) - 1,
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "discount": float(args.discount),
        "training_performed": False,
        "parents_used": False,
        "memory_written": False,
        "blind_evidence": False,
        "diagnostic_note": "true_next_bellman uses privileged simulator clones only to localize error; it is not a child planning result",
        "summaries": {m: _summary(raw[m]) for m in modes},
        "episode_details": raw,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "episode_details"}, indent=2, sort_keys=True))
    print(f"full_result={p}")


if __name__ == "__main__":
    main()
