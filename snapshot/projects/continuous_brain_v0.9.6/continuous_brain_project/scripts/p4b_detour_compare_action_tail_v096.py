from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
from torch import nn

from continuous_brain.environment import DetourWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v096 import _image_tensor, _context_tensor


class ActionTailValue(nn.Module):
    def __init__(self, input_dim: int = 12, hidden: int = 128):
        super().__init__()
        inner = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, inner), nn.SiLU(),
            nn.Linear(inner, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActionTailBundle:
    def __init__(self, path: str | Path, device: str):
        saved = torch.load(path, map_location=device, weights_only=False)
        self.gamma = float(saved["gamma"])
        self.action_dim = int(saved["action_dim"])
        self.state_mean = saved["state_mean"].to(device)
        self.state_std = saved["state_std"].to(device)
        self.y_mean = saved["y_mean"].to(device)
        self.y_std = saved["y_std"].to(device)
        self.net = ActionTailValue(int(saved["input_dim"]), int(saved["hidden"])).to(device)
        self.net.load_state_dict(saved["critic"])
        self.net.eval()
        self.metadata = saved.get("metadata", {})

    @torch.no_grad()
    def value(self, next_context7: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = (next_context7 - self.state_mean) / self.state_std
        z = torch.clamp(z, -5.0, 5.0)
        oh = torch.nn.functional.one_hot(actions.long(), self.action_dim).float()
        x = torch.cat([z, oh], dim=-1)
        return self.net(x) * self.y_std + self.y_mean


def mean(xs):
    return float(statistics.fmean(xs)) if xs else 0.0


def summary(rows, max_steps):
    return {
        "episodes": len(rows),
        "completion_rate": mean([float(r["completed"]) for r in rows]),
        "mean_reward": mean([r["reward"] for r in rows]),
        "mean_steps_penalized": mean([r["steps"] if r["completed"] else max_steps + 1 for r in rows]),
        "mean_collisions": mean([r["collisions"] for r in rows]),
        "mean_final_energy": mean([r["final_energy"] for r in rows]),
        "mean_planner_overrides": mean([r["planner_overrides"] for r in rows]),
    }


@torch.no_grad()
def imagine_one_step(model, critic: ActionTailBundle, *, workspace, base_spatial, context,
                     uncertainty_weight: float, device: str):
    action_dim = int(model.cfg.action_dim)
    if action_dim != critic.action_dim:
        raise ValueError(f"critic action_dim={critic.action_dim} != model action_dim={action_dim}")
    actions = torch.arange(action_dim, dtype=torch.long, device=device)
    ws = workspace.expand(action_dim, -1)
    sp = base_spatial.expand(action_dim, -1)
    ctx7 = context[:, :model.cfg.dynamics_context_dim].expand(action_dim, -1)
    pred = model.world(
        ws,
        one_hot_action(actions, action_dim),
        base_bus=sp,
        base_context=ctx7,
    )
    immediate = pred["reward"] - float(uncertainty_weight) * pred["uncertainty"]
    tail = critic.value(pred["next_context"], actions)
    scores = immediate + float(critic.gamma) * tail
    best = int(torch.argmax(scores).item())
    return best, {
        "scores": scores.detach().cpu().tolist(),
        "predicted_reward_minus_uncertainty": immediate.detach().cpu().tolist(),
        "predicted_future_tail": tail.detach().cpu().tolist(),
    }


@torch.no_grad()
def run_episode(model, critic, seed, mode, *, max_steps, device, uncertainty_weight=1.0):
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    total_reward = 0.0
    planner_overrides = 0
    completed = False
    steps_used = max_steps

    for step in range(1, max_steps + 1):
        image = _image_tensor(env, model.cfg.image_size, device)
        context = _context_tensor(obs, device)
        semantic = model.vision(image)
        spatial = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]
        workspace = nuc["workspace"]
        reactive = int(torch.argmax(model.motor(workspace), -1)[0].item())

        if mode == "reactive":
            action = reactive
        elif mode == "one_step_action_tail":
            action, _ = imagine_one_step(
                model, critic,
                workspace=workspace,
                base_spatial=model.target_spatial_bus(image),
                context=context,
                uncertainty_weight=uncertainty_weight,
                device=device,
            )
            planner_overrides += int(action != reactive)
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
        "reward": float(total_reward),
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "final_energy": float(env.energy),
        "planner_overrides": int(planner_overrides),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--critic", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed-start", type=int, default=9001)
    p.add_argument("--uncertainty-weight", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    model, saved = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    critic = ActionTailBundle(a.critic, a.device)

    reactive = []
    imagined = []
    for i in range(a.episodes):
        seed = a.seed_start + i
        reactive.append(run_episode(
            model, critic, seed, "reactive", max_steps=a.steps, device=a.device,
            uncertainty_weight=a.uncertainty_weight,
        ))
        imagined.append(run_episode(
            model, critic, seed, "one_step_action_tail", max_steps=a.steps, device=a.device,
            uncertainty_weight=a.uncertainty_weight,
        ))
        if (i + 1) % 10 == 0:
            print(f"paired_p4b_action_tail_episodes={i+1}/{a.episodes}")

    r = summary(reactive, a.steps)
    m = summary(imagined, a.steps)
    completion_adv = m["completion_rate"] - r["completion_rate"]
    collision_regression = m["mean_collisions"] - r["mean_collisions"]
    passed = bool(
        m["completion_rate"] >= 0.20
        and completion_adv > 0.05
        and collision_regression <= 2.0
    )

    result = {
        "format": "continuous_brain.p4b_one_step_action_tail_gate.v096.v4",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "critic": str(Path(a.critic).resolve()),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "experiment": {
            "paired_episodes": a.episodes,
            "steps_per_episode": a.steps,
            "seed_start": a.seed_start,
            "seed_end": a.seed_start + a.episodes - 1,
            "planning_horizon": 1,
            "recursive_world_rollout": False,
            "uncertainty_weight": a.uncertainty_weight,
            "critic_gamma": critic.gamma,
            "parents_used": False,
            "training_performed_during_evaluation": False,
            "memory_written": False,
            "critic_input": "world-predicted next_context7 + candidate action",
            "planner_score": "predicted_reward - uncertainty + gamma * learned action-conditioned future-return tail",
        },
        "reactive": r,
        "imagination": m,
        "advantages": {
            "completion_percentage_points": 100.0 * completion_adv,
            "reward_absolute": m["mean_reward"] - r["mean_reward"],
            "collisions_reduced": r["mean_collisions"] - m["mean_collisions"],
            "steps_saved_penalized": r["mean_steps_penalized"] - m["mean_steps_penalized"],
        },
        "thresholds": {
            "absolute_completion_rate": 0.20,
            "completion_advantage": 0.05,
            "max_collision_regression_per_episode": 2.0,
        },
        "p4b_passed": passed,
        "episodes": {"reactive": reactive, "imagination": imagined},
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "reactive": r,
        "imagination": m,
        "advantages": result["advantages"],
        "p4b_passed": passed,
        "full_result": str(out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
