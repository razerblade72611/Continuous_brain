from __future__ import annotations

import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .environment import DetourWorld
from .infant import one_hot_action
from .parented_train import load_checkpoint


def _image_tensor(env: DetourWorld, size: int, device: str) -> torch.Tensor:
    arr = env.render_rgb(size).astype(np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2, 0, 1))[None], dtype=torch.float32, device=device)


def _context_tensor(obs: np.ndarray, device: str) -> torch.Tensor:
    context = np.zeros(48, dtype=np.float32)
    context[:6] = obs[:6]
    context[6] = float(obs[4])
    context[30] = 1.0
    return torch.tensor(context[None], dtype=torch.float32, device=device)


def _mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def _summary(rows, max_steps):
    return {
        "episodes": len(rows),
        "completion_rate": _mean([float(r["completed"]) for r in rows]),
        "mean_reward": _mean([r["reward"] for r in rows]),
        "mean_steps_penalized": _mean([
            r["steps"] if r["completed"] else max_steps + 1 for r in rows
        ]),
        "mean_collisions": _mean([r["collisions"] for r in rows]),
        "mean_final_energy": _mean([r["final_energy"] for r in rows]),
        "mean_planner_overrides": _mean([r["planner_overrides"] for r in rows]),
    }


@torch.no_grad()
def _imagine_action(model, *, semantic_bus, workspace, hidden, base_spatial, context,
                    horizon: int, beam_width: int, discount: float,
                    success_bonus: float, uncertainty_weight: float, device: str):
    """Batched beam-search internal rollout. No simulator state is consulted here."""
    action_dim = int(model.cfg.action_dim)
    start = {
        "score": 0.0,
        "seq": [],
        "workspace": workspace,
        "hidden": hidden,
        "spatial": base_spatial,
        "context7": context[:, :model.cfg.dynamics_context_dim],
    }
    beam = [start]
    for depth in range(int(horizon)):
        parent_indices = []
        action_values = []
        for pi in range(len(beam)):
            for action in range(action_dim):
                parent_indices.append(pi)
                action_values.append(action)
        n = len(action_values)
        actions = torch.tensor(action_values, dtype=torch.long, device=device)
        ws = torch.cat([beam[i]["workspace"] for i in parent_indices], dim=0)
        sp = torch.cat([beam[i]["spatial"] for i in parent_indices], dim=0)
        ctx7 = torch.cat([beam[i]["context7"] for i in parent_indices], dim=0)
        hid = torch.cat([beam[i]["hidden"] for i in parent_indices], dim=0)
        pred = model.world(ws, one_hot_action(actions, action_dim), base_bus=sp, base_context=ctx7)
        success = torch.sigmoid(pred["success_logit"])
        immediate = (
            pred["reward"]
            + float(success_bonus) * success
            - float(uncertainty_weight) * pred["uncertainty"]
        )
        full_context = torch.zeros(n, model.cfg.context_dim, dtype=torch.float32, device=device)
        full_context[:, :model.cfg.dynamics_context_dim] = pred["next_context"]
        full_context[:, 30] = 1.0
        semantic = semantic_bus.expand(n, -1)
        fused = model.fuse_vision(semantic, pred["next_bus"])
        memory = torch.zeros(n, model.cfg.memory_dim, dtype=torch.float32, device=device)
        nuc = model.nucleus(fused, full_context, memory, actions + 1, hid)

        expanded = []
        for j, (pi, action) in enumerate(zip(parent_indices, action_values)):
            node = beam[pi]
            score = node["score"] + (float(discount) ** depth) * float(immediate[j].item())
            expanded.append({
                "score": score,
                "seq": node["seq"] + [int(action)],
                "workspace": nuc["workspace"][j:j+1],
                "hidden": nuc["hidden"][j:j+1],
                "spatial": pred["next_bus"][j:j+1],
                "context7": pred["next_context"][j:j+1],
            })
        expanded.sort(key=lambda x: x["score"], reverse=True)
        beam = expanded[:max(1, int(beam_width))]
    best = beam[0]
    return int(best["seq"][0]), float(best["score"]), best["seq"]


@torch.no_grad()
def run_detour_episode(model, seed: int, mode: str, *, horizon: int, beam_width: int,
                       max_steps: int, device: str, discount=0.95,
                       success_bonus=0.50, uncertainty_weight=1.0):
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    total_reward = 0.0
    planner_overrides = 0
    completed = False
    steps_used = max_steps

    for step in range(1, int(max_steps) + 1):
        image = _image_tensor(env, model.cfg.image_size, device)
        context = _context_tensor(obs, device)
        semantic = model.vision(image)
        spatial = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]
        workspace = nuc["workspace"]
        reactive_action = int(torch.argmax(model.motor(workspace), dim=-1)[0].item())

        if mode == "reactive":
            action = reactive_action
        elif mode == "imagination":
            base_spatial = model.target_spatial_bus(image)
            action, _, _ = _imagine_action(
                model, semantic_bus=semantic, workspace=workspace, hidden=hidden,
                base_spatial=base_spatial, context=context, horizon=horizon,
                beam_width=beam_width, discount=discount, success_bonus=success_bonus,
                uncertainty_weight=uncertainty_weight, device=device,
            )
            planner_overrides += int(action != reactive_action)
        else:
            raise ValueError(mode)

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor([action + 1], dtype=torch.long, device=device)
        if bool(info.get("reached")):
            completed = True
            steps_used = step
        if done:
            break

    return {
        "seed": int(seed),
        "mode": mode,
        "horizon": int(horizon if mode == "imagination" else 0),
        "reward": float(total_reward),
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "final_energy": float(env.energy),
        "planner_overrides": int(planner_overrides),
    }


def compare_detour(checkpoint: str | Path, *, episodes=100, steps=120, seed_start=9001,
                   horizons=(1, 3, 6), beam_width=16, device="cpu", output=None) -> dict:
    model, saved = load_checkpoint(checkpoint, device)
    model.eval()
    reactive = []
    imagined = {int(h): [] for h in horizons}
    for i in range(int(episodes)):
        seed = int(seed_start) + i
        reactive.append(run_detour_episode(
            model, seed, "reactive", horizon=1, beam_width=beam_width,
            max_steps=steps, device=device,
        ))
        for h in horizons:
            imagined[int(h)].append(run_detour_episode(
                model, seed, "imagination", horizon=int(h), beam_width=beam_width,
                max_steps=steps, device=device,
            ))
        if (i + 1) % 10 == 0:
            print(f"paired_detour_episodes={i+1}/{episodes}")

    r = _summary(reactive, steps)
    results = {}
    any_pass = False
    for h, rows in imagined.items():
        m = _summary(rows, steps)
        completion_adv = m["completion_rate"] - r["completion_rate"]
        reward_abs = m["mean_reward"] - r["mean_reward"]
        reward_rel = reward_abs / max(abs(r["mean_reward"]), 1e-8)
        no_regress = completion_adv >= -0.02
        gate = completion_adv > 0.05 or reward_rel > 0.10
        passed = bool(gate and no_regress)
        any_pass = any_pass or passed
        results[str(h)] = {
            "imagination": m,
            "advantages": {
                "completion_percentage_points": 100.0 * completion_adv,
                "reward_absolute": reward_abs,
                "reward_relative": reward_rel,
                "collisions_reduced": r["mean_collisions"] - m["mean_collisions"],
                "steps_saved_penalized": r["mean_steps_penalized"] - m["mean_steps_penalized"],
            },
            "performance_gate": bool(gate),
            "no_material_completion_regression": bool(no_regress),
            "p4_passed": passed,
        }

    result = {
        "format": "continuous_brain.p4_detour_planning_gate.v096",
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "experiment": {
            "paired_episodes": int(episodes),
            "steps_per_episode": int(steps),
            "seed_start": int(seed_start),
            "seed_end": int(seed_start) + int(episodes) - 1,
            "horizons": [int(h) for h in horizons],
            "beam_width": int(beam_width),
            "parents_used": False,
            "training_performed_during_evaluation": False,
            "memory_written": False,
            "planner_score": "discounted(predicted_reward + 0.50*predicted_success - uncertainty)",
        },
        "reactive": r,
        "horizons": results,
        "thresholds": {
            "completion_advantage": 0.05,
            "reward_relative_advantage": 0.10,
            "max_completion_regression": 0.02,
        },
        "p4_passed": bool(any_pass),
        "episodes": {"reactive": reactive, **{f"h{h}": rows for h, rows in imagined.items()}},
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
