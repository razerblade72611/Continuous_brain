from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from continuous_brain.environment import ForagingWorld
from continuous_brain.infant import one_hot_action
from continuous_brain.parented_train import load_checkpoint


def image_tensor(env, size, device):
    arr = env.render_rgb(size).astype(np.float32) / 255.0
    return torch.tensor(
        np.transpose(arr, (2, 0, 1))[None],
        dtype=torch.float32,
        device=device,
    )


def context_tensor(obs, env, device):
    context = np.zeros(48, dtype=np.float32)
    context[:6] = obs[:6]
    context[6] = env.energy
    context[30] = 1.0
    return torch.tensor(context[None], dtype=torch.float32, device=device)


@torch.no_grad()
def choose_imagined_action(model, workspace, image, context):
    n = model.cfg.action_dim

    actions = torch.arange(n, dtype=torch.long, device=workspace.device)
    action_oh = one_hot_action(actions, n)

    workspace_batch = workspace.expand(n, -1)
    base_spatial = model.target_spatial_bus(image).expand(n, -1)
    base_context = context.expand(n, -1)

    pred = model.world(
        workspace_batch,
        action_oh,
        base_bus=base_spatial,
        base_context=base_context,
    )

    predicted_reward = pred["reward"]
    predicted_success = torch.sigmoid(pred["success_logit"])
    uncertainty = pred["uncertainty"]

    # P4 scoring rule is fixed BEFORE seeing the experiment:
    #
    #   expected immediate reward
    #   + modest bonus for predicted target success
    #   - epistemic uncertainty
    #
    # No simulator target/distance is inspected here.
    score = (
        predicted_reward
        + 0.25 * predicted_success
        - uncertainty
    )

    action = int(torch.argmax(score).item())

    return action, {
        "scores": score.detach().cpu().tolist(),
        "predicted_reward": predicted_reward.detach().cpu().tolist(),
        "predicted_success": predicted_success.detach().cpu().tolist(),
        "uncertainty": uncertainty.detach().cpu().tolist(),
    }


@torch.no_grad()
def run_episode(model, seed, mode, max_steps, device):
    env = ForagingWorld(seed=seed, max_episode_steps=max_steps)

    # Use an explicit reset in both conditions. Same seed => same initial world.
    obs = env.reset()

    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(
        1, model.cfg.memory_dim, dtype=torch.float32, device=device
    )
    previous_action = torch.zeros(1, dtype=torch.long, device=device)

    total_reward = 0.0
    successes = 0
    first_success_step = None
    planner_overrides = 0
    action_counts = [0 for _ in range(model.cfg.action_dim)]

    for step in range(1, max_steps + 1):
        image = image_tensor(env, model.cfg.image_size, device)
        context = context_tensor(obs, env, device)

        vision = model.vision(image)
        spatial = model.spatial(image)
        fused = model.fuse_vision(vision, spatial)

        nucleus = model.nucleus(
            fused,
            context,
            memory,
            previous_action,
            hidden,
        )

        hidden = nucleus["hidden"]
        workspace = nucleus["workspace"]

        reactive_action = int(
            torch.argmax(model.motor(workspace), dim=-1)[0].item()
        )

        if mode == "reactive":
            action = reactive_action
        elif mode == "imagination":
            action, _ = choose_imagined_action(
                model,
                workspace,
                image,
                context,
            )
            planner_overrides += int(action != reactive_action)
        else:
            raise ValueError(mode)

        action_counts[action] += 1

        next_obs, reward, done, info = env.step(action)

        total_reward += float(reward)

        if bool(info.get("reached")):
            successes += 1
            if first_success_step is None:
                first_success_step = step

        previous_action = torch.tensor(
            [action + 1],
            dtype=torch.long,
            device=device,
        )

        obs = next_obs

        if done:
            break

    return {
        "seed": int(seed),
        "mode": mode,
        "reward": float(total_reward),
        "successes": int(successes),
        "completed": bool(successes > 0),
        "first_success_step": first_success_step,
        "final_energy": float(env.energy),
        "planner_overrides": int(planner_overrides),
        "action_counts": action_counts,
    }


def mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def summarize(rows, max_steps):
    completed = [r["completed"] for r in rows]

    first_steps = [
        r["first_success_step"]
        if r["first_success_step"] is not None
        else max_steps + 1
        for r in rows
    ]

    return {
        "episodes": len(rows),
        "completion_rate": mean([float(x) for x in completed]),
        "mean_reward": mean([r["reward"] for r in rows]),
        "mean_targets_reached": mean([r["successes"] for r in rows]),
        "mean_first_success_step_penalized": mean(first_steps),
        "mean_final_energy": mean([r["final_energy"] for r in rows]),
        "mean_planner_overrides": mean(
            [r["planner_overrides"] for r in rows]
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=5001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model, saved = load_checkpoint(args.checkpoint, args.device)
    model.eval()

    reactive = []
    imagination = []

    for i in range(args.episodes):
        seed = args.seed_start + i

        reactive.append(
            run_episode(
                model,
                seed,
                "reactive",
                args.steps,
                args.device,
            )
        )

        imagination.append(
            run_episode(
                model,
                seed,
                "imagination",
                args.steps,
                args.device,
            )
        )

        if (i + 1) % 10 == 0:
            print(f"paired_episodes={i + 1}/{args.episodes}")

    r = summarize(reactive, args.steps)
    m = summarize(imagination, args.steps)

    completion_advantage = (
        m["completion_rate"] - r["completion_rate"]
    )

    reward_difference = (
        m["mean_reward"] - r["mean_reward"]
    )

    reward_relative_advantage = reward_difference / max(
        abs(r["mean_reward"]), 1e-8
    )

    # Predeclared P4 rule:
    #
    # PASS if imagination yields:
    #   > 5 percentage-point completion advantage
    #       OR
    #   > 10% mean reward advantage
    #
    # while completion does not regress by more than 2 percentage points.
    performance_gate = (
        completion_advantage > 0.05
        or reward_relative_advantage > 0.10
    )

    no_material_completion_regression = (
        completion_advantage >= -0.02
    )

    result = {
        "format": "continuous_brain.p4_planning_gate.v095",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_updates": int(
            saved.get(
                "updates",
                (saved.get("metadata") or {}).get("updates", 0),
            )
        ),
        "experiment": {
            "paired_episodes": args.episodes,
            "steps_per_episode": args.steps,
            "seed_start": args.seed_start,
            "seed_end": args.seed_start + args.episodes - 1,
            "parents_used": False,
            "training_performed": False,
            "memory_written": False,
            "planning_horizon": 1,
            "candidate_actions": int(model.cfg.action_dim),
            "planner_score":
                "predicted_reward + 0.25*predicted_success - uncertainty",
        },
        "reactive": r,
        "imagination": m,
        "advantages": {
            "completion_percentage_points":
                100.0 * completion_advantage,
            "reward_absolute": reward_difference,
            "reward_relative":
                reward_relative_advantage,
            "targets_reached":
                m["mean_targets_reached"]
                - r["mean_targets_reached"],
            "first_success_steps_saved":
                r["mean_first_success_step_penalized"]
                - m["mean_first_success_step_penalized"],
        },
        "thresholds": {
            "completion_advantage": 0.05,
            "reward_relative_advantage": 0.10,
            "max_completion_regression": 0.02,
        },
        "performance_gate": bool(performance_gate),
        "no_material_completion_regression":
            bool(no_material_completion_regression),
        "p4_passed": bool(
            performance_gate
            and no_material_completion_regression
        ),
        "episodes": {
            "reactive": reactive,
            "imagination": imagination,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps({
        "reactive": r,
        "imagination": m,
        "advantages": result["advantages"],
        "performance_gate": result["performance_gate"],
        "no_material_completion_regression":
            result["no_material_completion_regression"],
        "p4_passed": result["p4_passed"],
        "full_result": str(out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
