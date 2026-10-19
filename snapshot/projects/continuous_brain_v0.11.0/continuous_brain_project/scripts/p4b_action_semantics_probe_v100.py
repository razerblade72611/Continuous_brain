#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor

ACTION_NAMES = {0: "idle", 1: "up", 2: "down", 3: "left", 4: "right"}


def _round(x, n=6):
    return round(float(x), n)


def _model_action_table(model, env: DetourWorld, obs, device: str, discount: float):
    image = _env_image_tensor(env, model.cfg.image_size, device)
    _, context7 = _env_context(obs, device)
    spatial = model.target_spatial_bus(image)
    root = model.planning.encode_state(spatial, context7)
    ad = int(model.cfg.action_dim)
    actions = torch.arange(ad, dtype=torch.long, device=device)
    aoh = one_hot_action(actions, ad)
    roots = root.expand(ad, -1)
    pred = model.planning.step(roots, aoh)
    q = model.planning.action_value(roots, aoh)

    rows = []
    for a in range(ad):
        clone = copy.deepcopy(env)
        true_next, true_reward, _, info = clone.step(a)
        pred_ctx = pred["context"][a]
        row = {
            "action": a,
            "action_name": ACTION_NAMES.get(a, str(a)),
            "pred": {
                "reward": _round(pred["reward"][a]),
                "collision_probability": _round(torch.sigmoid(pred["collision_logit"][a])),
                "q_current_action": _round(q[a]),
                "next_value": _round(pred["value"][a]),
                "next_goal_value": _round(pred["goal_value"][a]),
                "bellman_value_score": _round(pred["reward"][a] + float(discount) * pred["value"][a]),
                "next_context": [_round(v) for v in pred_ctx.tolist()],
            },
            "true": {
                "reward": _round(true_reward),
                "collision": bool(info.get("collision", False)),
                "reached": bool(info.get("reached", False)),
                "next_obs": [_round(v) for v in true_next.tolist()],
            },
        }
        row["errors"] = {
            "reward_abs": _round(abs(row["pred"]["reward"] - row["true"]["reward"])),
            "next_x_abs": _round(abs(row["pred"]["next_context"][0] - row["true"]["next_obs"][0])),
            "next_y_abs": _round(abs(row["pred"]["next_context"][1] - row["true"]["next_obs"][1])),
        }
        rows.append(row)

    def best(field, nested=False):
        if nested:
            return max(rows, key=lambda r: r["pred"][field])["action"]
        return max(rows, key=lambda r: r[field])["action"]

    # Is action semantics directionally correct in the learned transition?
    pred_y = {r["action"]: r["pred"]["next_context"][1] for r in rows}
    true_y = {r["action"]: r["true"]["next_obs"][1] for r in rows}
    directional = {
        "pred_up_above_idle": pred_y[1] > pred_y[0],
        "pred_down_below_idle": pred_y[2] < pred_y[0],
        "true_up_above_idle": true_y[1] > true_y[0],
        "true_down_below_idle": true_y[2] < true_y[0],
    }
    return {
        "state": {
            "agent_x": _round(obs[0]), "agent_y": _round(obs[1]),
            "target_dx": _round(obs[2]), "target_dy": _round(obs[3]),
        },
        "directional_action_semantics": directional,
        "best_actions": {
            "predicted_immediate_reward": ACTION_NAMES[best("reward", True)],
            "predicted_q_current_action": ACTION_NAMES[best("q_current_action", True)],
            "predicted_next_value": ACTION_NAMES[best("next_value", True)],
            "predicted_next_goal_value": ACTION_NAMES[best("next_goal_value", True)],
            "predicted_bellman_value_score": ACTION_NAMES[best("bellman_value_score", True)],
            "true_immediate_reward": ACTION_NAMES[max(rows, key=lambda r: r["true"]["reward"])["action"]],
        },
        "actions": rows,
    }


def _prewall_state(seed: int, max_steps: int = 120):
    env = DetourWorld(seed=seed, max_episode_steps=max_steps)
    obs = env.reset()
    # March right until the NEXT right action would collide; do not execute collision.
    for _ in range(40):
        probe = copy.deepcopy(env)
        _, _, _, info = probe.step(4)
        if bool(info.get("collision", False)):
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
    records = []
    semantics_ok = 0
    up_best_counts = {
        "initial_q": 0, "initial_value": 0, "initial_goal": 0, "initial_bellman": 0,
        "prewall_q": 0, "prewall_value": 0, "prewall_goal": 0, "prewall_bellman": 0,
    }
    for i in range(args.seeds):
        seed = args.seed_start + i
        env0 = DetourWorld(seed=seed, max_episode_steps=120)
        obs0 = env0.reset()
        initial = _model_action_table(model, env0, obs0, args.device, args.discount)
        envw, obsw = _prewall_state(seed)
        prewall = _model_action_table(model, envw, obsw, args.device, args.discount)
        records.append({"seed": seed, "initial": initial, "prewall": prewall})
        semantics_ok += int(all(initial["directional_action_semantics"].values()) and all(prewall["directional_action_semantics"].values()))
        for prefix, tab in (("initial", initial), ("prewall", prewall)):
            up_best_counts[f"{prefix}_q"] += int(tab["best_actions"]["predicted_q_current_action"] == "up")
            up_best_counts[f"{prefix}_value"] += int(tab["best_actions"]["predicted_next_value"] == "up")
            up_best_counts[f"{prefix}_goal"] += int(tab["best_actions"]["predicted_next_goal_value"] == "up")
            up_best_counts[f"{prefix}_bellman"] += int(tab["best_actions"]["predicted_bellman_value_score"] == "up")
        print(f"action_semantics_seed={seed} semantics_ok={all(initial['directional_action_semantics'].values()) and all(prewall['directional_action_semantics'].values())}")

    summary = {
        "format": "continuous_brain.v100_action_semantics_probe.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.seeds - 1,
        "seeds": args.seeds,
        "discount": args.discount,
        "training_performed": False,
        "parents_used": False,
        "blind_evidence": False,
        "action_mapping": ACTION_NAMES,
        "gap_center_y": DetourWorld.gap_center_y,
        "directional_semantics_correct_both_states": semantics_ok,
        "up_ranked_best_counts": up_best_counts,
        "records": records,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, sort_keys=True))
    print(f"full_result={p}")


if __name__ == "__main__":
    main()
