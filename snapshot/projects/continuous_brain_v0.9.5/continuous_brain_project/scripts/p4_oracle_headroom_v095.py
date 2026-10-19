from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from p4_compare_v095 import run_episode, summarize
from continuous_brain.environment import ForagingWorld
from continuous_brain.parented_train import load_checkpoint


def run_oracle_episode(seed, max_steps):
    """
    DIAGNOSTIC ONLY.

    Uses the real simulator to test every possible immediate action,
    then chooses the one with the highest TRUE immediate reward.

    This is privileged information and is never used by the child.
    It measures how much one-step planning headroom exists in this task.
    """
    env = ForagingWorld(seed=seed, max_episode_steps=max_steps)
    env.reset()

    total_reward = 0.0
    successes = 0
    first_success_step = None
    action_counts = [0] * env.action_dim

    for step in range(1, max_steps + 1):
        candidates = []

        for action in range(env.action_dim):
            probe = copy.deepcopy(env)
            _, reward, _, info = probe.step(action)

            candidates.append((
                float(reward),
                int(bool(info.get("reached"))),
                action,
            ))

        # Highest true immediate reward.
        # Reached is only a tie breaker.
        _, _, action = max(
            candidates,
            key=lambda x: (x[0], x[1], -x[2])
        )

        action_counts[action] += 1

        _, reward, done, info = env.step(action)

        total_reward += float(reward)

        if bool(info.get("reached")):
            successes += 1
            if first_success_step is None:
                first_success_step = step

        if done:
            break

    return {
        "seed": int(seed),
        "mode": "one_step_true_reward_oracle_DIAGNOSTIC_ONLY",
        "reward": float(total_reward),
        "successes": int(successes),
        "completed": bool(successes > 0),
        "first_success_step": first_success_step,
        "final_energy": float(env.energy),
        "planner_overrides": 0,
        "action_counts": action_counts,
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

    model, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()

    reactive = []
    imagination = []
    oracle = []

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

        oracle.append(
            run_oracle_episode(seed, args.steps)
        )

        if (i + 1) % 10 == 0:
            print(
                f"diagnostic_triplets={i + 1}/{args.episodes}"
            )

    r = summarize(reactive, args.steps)
    m = summarize(imagination, args.steps)
    o = summarize(oracle, args.steps)

    imagination_gain = (
        m["mean_reward"] - r["mean_reward"]
    )

    oracle_gain = (
        o["mean_reward"] - r["mean_reward"]
    )

    imagination_relative = (
        imagination_gain
        / max(abs(r["mean_reward"]), 1e-8)
    )

    oracle_relative = (
        oracle_gain
        / max(abs(r["mean_reward"]), 1e-8)
    )

    capture = (
        imagination_gain / oracle_gain
        if oracle_gain > 1e-8
        else None
    )

    result = {
        "format":
            "continuous_brain.p4_one_step_headroom_diagnostic.v095",

        "diagnostic_only": True,

        "checkpoint":
            str(Path(args.checkpoint).resolve()),

        "episodes": args.episodes,
        "steps": args.steps,

        "seed_start": args.seed_start,
        "seed_end":
            args.seed_start + args.episodes - 1,

        "reactive": r,
        "imagination": m,

        "one_step_true_reward_oracle": o,

        "reward_analysis": {
            "imagination_absolute_gain":
                imagination_gain,

            "imagination_relative_gain":
                imagination_relative,

            "oracle_absolute_headroom":
                oracle_gain,

            "oracle_relative_headroom":
                oracle_relative,

            "fraction_of_oracle_headroom_captured":
                capture,
        },

        "interpretation": {
            "oracle_headroom_below_10_percent":
                "The current proof world cannot fairly demonstrate "
                "the predeclared +10% one-step reward gate. "
                "Move to a planning-sensitive environment.",

            "oracle_headroom_at_least_10_percent_but_imagination_low":
                "The task contains enough one-step decision headroom. "
                "The planner/world-value ranking needs improvement.",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True
        )
    )

    print(json.dumps({
        "reactive": r,
        "imagination": m,
        "one_step_true_reward_oracle": o,
        "reward_analysis":
            result["reward_analysis"],
        "full_result":
            str(out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
