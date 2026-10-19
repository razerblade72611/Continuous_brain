#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v101 import _all_action_values

ACTION_NAMES = {0: "idle", 1: "up", 2: "down", 3: "left", 4: "right"}


def _state_table(model, env, obs, device, gamma):
    image = _env_image_tensor(env, model.cfg.image_size, device)
    _, context7 = _env_context(obs, device)
    spatial = model.target_spatial_bus(image)
    root = model.planning.encode_state(spatial, context7)
    ad = model.cfg.action_dim
    actions = torch.arange(ad, dtype=torch.long, device=device)
    roots = root.expand(ad, -1)
    pred = model.planning.step(roots, one_hot_action(actions, ad))
    q_root = model.planning.action_value(roots, one_hot_action(actions, ad))
    q_next = _all_action_values(model.planning, pred["latent"]).max(dim=1).values
    model_bellman = pred["reward"] + float(gamma) * q_next
    rows = []
    for a in range(ad):
        clone = copy.deepcopy(env)
        true_next, true_reward, _, info = clone.step(a)
        rows.append({
            "action": a,
            "action_name": ACTION_NAMES[a],
            "q": float(q_root[a]),
            "predicted_reward": float(pred["reward"][a]),
            "predicted_next_max_q": float(q_next[a]),
            "model_bellman_score": float(model_bellman[a]),
            "predicted_next_y": float(pred["context"][a, 1]),
            "true_next_y": float(true_next[1]),
            "true_reward": float(true_reward),
            "true_collision": bool(info.get("collision", False)),
        })
    return {
        "agent_x": float(obs[0]), "agent_y": float(obs[1]),
        "best_q": ACTION_NAMES[max(rows, key=lambda r: r["q"])["action"]],
        "best_model_bellman": ACTION_NAMES[max(rows, key=lambda r: r["model_bellman_score"])["action"]],
        "rows": rows,
    }


def _prewall(seed):
    env = DetourWorld(seed=seed, max_episode_steps=120)
    obs = env.reset()
    for _ in range(40):
        probe = copy.deepcopy(env)
        _, _, _, info = probe.step(4)
        if info.get("collision", False):
            return env, obs
        obs, _, done, _ = env.step(4)
        if done:
            break
    return env, obs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed-start", type=int, default=9001)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--discount", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model, saved = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})
    records = []
    counts = {"initial_q": 0, "initial_model_bellman": 0, "prewall_q": 0, "prewall_model_bellman": 0}
    for i in range(args.seeds):
        seed = args.seed_start + i
        e0 = DetourWorld(seed=seed, max_episode_steps=120); o0 = e0.reset()
        initial = _state_table(model, e0, o0, args.device, args.discount)
        ew, ow = _prewall(seed)
        prewall = _state_table(model, ew, ow, args.device, args.discount)
        records.append({"seed": seed, "initial": initial, "prewall": prewall})
        counts["initial_q"] += int(initial["best_q"] == "up")
        counts["initial_model_bellman"] += int(initial["best_model_bellman"] == "up")
        counts["prewall_q"] += int(prewall["best_q"] == "up")
        counts["prewall_model_bellman"] += int(prewall["best_model_bellman"] == "up")
        print(f"bellman_q_probe_seed={seed} initial={initial['best_model_bellman']} prewall={prewall['best_model_bellman']}")

    result = {
        "format": "continuous_brain.v101_bellman_q_probe.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "planning_value_version": metadata.get("planning_value_version"),
        "seed_start": args.seed_start, "seed_end": args.seed_start + args.seeds - 1,
        "seeds": args.seeds, "discount": args.discount,
        "training_performed": False, "parents_used": False, "blind_evidence": False,
        "up_ranked_best_counts": counts,
        "records": records,
    }
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, sort_keys=True))
    print(f"full_result={p}")


if __name__ == "__main__":
    main()
