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


class ReturnValue(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        inner = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, inner), nn.SiLU(),
            nn.Linear(inner, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReturnCriticBundle:
    def __init__(self, path: str | Path, device: str):
        saved = torch.load(path, map_location=device, weights_only=False)
        self.gamma = float(saved["gamma"])
        self.x_mean = saved["x_mean"].to(device)
        self.x_std = saved["x_std"].to(device)
        self.y_mean = saved["y_mean"].to(device)
        self.y_std = saved["y_std"].to(device)
        self.net = ReturnValue(int(saved["input_dim"]), int(saved["hidden"])).to(device)
        self.net.load_state_dict(saved["critic"])
        self.net.eval()
        self.metadata = saved.get("metadata", {})

    @torch.no_grad()
    def value(self, context7: torch.Tensor):
        z = (context7 - self.x_mean) / self.x_std
        z = torch.clamp(z, -5.0, 5.0)
        return self.net(z) * self.y_std + self.y_mean


def mean(xs):
    return float(statistics.fmean(xs)) if xs else 0.0


def summary(rows, max_steps):
    return {
        "episodes": len(rows),
        "completion_rate": mean([float(r["completed"]) for r in rows]),
        "mean_reward": mean([r["reward"] for r in rows]),
        "mean_steps_penalized": mean([
            r["steps"] if r["completed"] else max_steps + 1 for r in rows
        ]),
        "mean_collisions": mean([r["collisions"] for r in rows]),
        "mean_final_energy": mean([r["final_energy"] for r in rows]),
        "mean_planner_overrides": mean([r["planner_overrides"] for r in rows]),
    }


@torch.no_grad()
def imagine_with_return_tail(
    model,
    critic: ReturnCriticBundle,
    *,
    semantic_bus,
    workspace,
    hidden,
    base_spatial,
    context,
    horizon: int,
    beam_width: int,
    discount: float,
    uncertainty_weight: float,
    device: str,
):
    if abs(float(discount) - float(critic.gamma)) > 1e-9:
        raise ValueError(
            f"Planner discount {discount} must match critic gamma {critic.gamma}; "
            "otherwise rollout rewards and terminal value are not on the same Bellman scale."
        )

    action_dim = int(model.cfg.action_dim)
    beam = [{
        "score": 0.0,
        "seq": [],
        "workspace": workspace,
        "hidden": hidden,
        "spatial": base_spatial,
        "context7": context[:, :model.cfg.dynamics_context_dim],
    }]

    for depth in range(int(horizon)):
        parent_indices = []
        action_values = []
        for pi in range(len(beam)):
            for action in range(action_dim):
                parent_indices.append(pi)
                action_values.append(action)

        actions = torch.tensor(action_values, dtype=torch.long, device=device)
        ws = torch.cat([beam[i]["workspace"] for i in parent_indices], 0)
        sp = torch.cat([beam[i]["spatial"] for i in parent_indices], 0)
        ctx7 = torch.cat([beam[i]["context7"] for i in parent_indices], 0)
        hid = torch.cat([beam[i]["hidden"] for i in parent_indices], 0)

        pred = model.world(
            ws,
            one_hot_action(actions, action_dim),
            base_bus=sp,
            base_context=ctx7,
        )
        immediate = pred["reward"] - float(uncertainty_weight) * pred["uncertainty"]

        n = len(action_values)
        full_context = torch.zeros(n, model.cfg.context_dim, dtype=torch.float32, device=device)
        full_context[:, :model.cfg.dynamics_context_dim] = pred["next_context"]
        full_context[:, 30] = 1.0
        semantic = semantic_bus.expand(n, -1)
        fused = model.fuse_vision(semantic, pred["next_bus"])
        memory = torch.zeros(n, model.cfg.memory_dim, dtype=torch.float32, device=device)
        nuc = model.nucleus(fused, full_context, memory, actions + 1, hid)

        terminal_return = None
        if depth == int(horizon) - 1:
            terminal_return = critic.value(pred["next_context"])

        expanded = []
        for j, (pi, action) in enumerate(zip(parent_indices, action_values)):
            node = beam[pi]
            score = node["score"] + (float(discount) ** depth) * float(immediate[j].item())
            if terminal_return is not None:
                score += (float(discount) ** (depth + 1)) * float(terminal_return[j].item())
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

    return int(beam[0]["seq"][0]), float(beam[0]["score"]), beam[0]["seq"]


@torch.no_grad()
def run_episode(
    model,
    critic,
    seed,
    mode,
    *,
    horizon,
    beam_width,
    max_steps,
    device,
    discount=0.95,
    uncertainty_weight=1.0,
):
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
        elif mode == "return_imagination":
            action, _, _ = imagine_with_return_tail(
                model,
                critic,
                semantic_bus=semantic,
                workspace=workspace,
                hidden=hidden,
                base_spatial=model.target_spatial_bus(image),
                context=context,
                horizon=horizon,
                beam_width=beam_width,
                discount=discount,
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
        "horizon": int(horizon if mode != "reactive" else 0),
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
    p.add_argument("--seed-start", type=int, default=12001)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--uncertainty-weight", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    model, saved = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    critic = ReturnCriticBundle(a.critic, a.device)
    if abs(float(a.discount) - float(critic.gamma)) > 1e-9:
        raise ValueError(f"--discount {a.discount} must equal critic gamma {critic.gamma}")

    reactive = []
    imagined = {int(h): [] for h in a.horizons}
    for i in range(a.episodes):
        seed = a.seed_start + i
        reactive.append(run_episode(
            model, critic, seed, "reactive",
            horizon=1, beam_width=a.beam_width, max_steps=a.steps, device=a.device,
            discount=a.discount, uncertainty_weight=a.uncertainty_weight,
        ))
        for h in a.horizons:
            imagined[int(h)].append(run_episode(
                model, critic, seed, "return_imagination",
                horizon=int(h), beam_width=a.beam_width, max_steps=a.steps, device=a.device,
                discount=a.discount, uncertainty_weight=a.uncertainty_weight,
            ))
        if (i + 1) % 10 == 0:
            print(f"paired_p4b_return_episodes={i+1}/{a.episodes}")

    r = summary(reactive, a.steps)
    results = {}
    any_pass = False
    for h, rows in imagined.items():
        m = summary(rows, a.steps)
        completion_adv = m["completion_rate"] - r["completion_rate"]
        collision_regression = m["mean_collisions"] - r["mean_collisions"]
        passed = bool(
            m["completion_rate"] >= 0.20
            and completion_adv > 0.05
            and collision_regression <= 2.0
        )
        any_pass = any_pass or passed
        results[str(h)] = {
            "imagination": m,
            "advantages": {
                "completion_percentage_points": 100.0 * completion_adv,
                "reward_absolute": m["mean_reward"] - r["mean_reward"],
                "collisions_reduced": r["mean_collisions"] - m["mean_collisions"],
                "steps_saved_penalized": r["mean_steps_penalized"] - m["mean_steps_penalized"],
            },
            "p4b_passed": passed,
        }

    result = {
        "format": "continuous_brain.p4b_return_planning_gate.v096.v3",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "critic": str(Path(a.critic).resolve()),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "experiment": {
            "paired_episodes": a.episodes,
            "steps_per_episode": a.steps,
            "seed_start": a.seed_start,
            "seed_end": a.seed_start + a.episodes - 1,
            "horizons": a.horizons,
            "beam_width": a.beam_width,
            "discount": a.discount,
            "critic_gamma": critic.gamma,
            "discount_matches_critic": abs(float(a.discount) - float(critic.gamma)) <= 1e-9,
            "uncertainty_weight": a.uncertainty_weight,
            "parents_used": False,
            "training_performed_during_evaluation": False,
            "memory_written": False,
            "critic_input": "predicted_structured_context7_only",
            "planner_score": "discounted predicted reward - uncertainty + gamma^H * learned full return-to-go",
            "success_bonus": 0.0,
            "return_weight": 1.0,
        },
        "reactive": r,
        "horizons": results,
        "thresholds": {
            "absolute_completion_rate": 0.20,
            "completion_advantage": 0.05,
            "max_collision_regression_per_episode": 2.0,
        },
        "p4b_passed": bool(any_pass),
        "episodes": {"reactive": reactive, **{f"h{h}": rows for h, rows in imagined.items()}},
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "reactive": r,
        "horizons": results,
        "p4b_passed": any_pass,
        "full_result": str(out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
