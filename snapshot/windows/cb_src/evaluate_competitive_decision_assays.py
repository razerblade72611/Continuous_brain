#!/usr/bin/env python3
import argparse
import json
import warnings
from pathlib import Path

import torch
import torch_directml

from continuous_brain.cued_detour_world import CuedDetourWorld
from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _teacher_action,
)
from evaluate_competitive_heldout import (
    build_model,
    competition_stats,
    eval_closed_loop,
)


@torch.no_grad()
def eval_junction_retention(
    model,
    cfg,
    condition,
    device,
    seed,
    corridor_length,
):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=float(corridor_length),
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    memory = torch.zeros(1, cfg.memory_dim, device=device)

    cue = str(env.cue)
    teacher_steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_competition=(condition != "base"),
            compute_aux=False,
        )
        hidden = out["hidden"]

        teacher_action = int(_teacher_action(env))

        # First lateral teacher action = cue-dependent junction decision.
        if teacher_action in (3, 4):
            if env.steps < env.cue_steps:
                raise AssertionError(
                    "junction decision occurred while cue was still visible"
                )
            if float(env.agent[1]) < float(env._junction_y()) + 0.08 - 1e-5:
                raise AssertionError(
                    "junction decision detected before intended geometry"
                )

            logits = out["logits"][0].detach().float().cpu()
            full_pred = int(torch.argmax(logits).item())

            left_logit = float(logits[3].item())
            right_logit = float(logits[4].item())
            side_pred = 3 if left_logit >= right_logit else 4

            correct_logit = (
                left_logit if teacher_action == 3 else right_logit
            )
            wrong_logit = (
                right_logit if teacher_action == 3 else left_logit
            )

            entropies, weights = competition_stats(out["competition"])
            entropy = (
                sum(entropies) / len(entropies)
                if entropies else None
            )

            return {
                "seed": int(seed),
                "cue": cue,
                "corridor_length": float(corridor_length),
                "junction_env_steps": int(env.steps),
                "teacher_steps_before_junction": int(teacher_steps),
                "junction_y": float(env._junction_y()),
                "agent_x": float(env.agent[0]),
                "agent_y": float(env.agent[1]),
                "cue_hidden": True,
                "teacher_action": int(teacher_action),
                "full_pred": int(full_pred),
                "full_action_correct": bool(
                    full_pred == teacher_action
                ),
                "side_pred": int(side_pred),
                "side_choice_correct": bool(
                    side_pred == teacher_action
                ),
                "left_logit": left_logit,
                "right_logit": right_logit,
                "correct_minus_wrong_side_logit": float(
                    correct_logit - wrong_logit
                ),
                "selector_entropy": entropy,
                "selector_weights": weights,
            }

        if teacher_action != 1:
            raise AssertionError(
                f"unexpected pre-junction teacher action "
                f"{teacher_action} at step {env.steps}"
            )

        obs, _reward, _done, _info = env.step(teacher_action)
        prev_action = torch.tensor(
            [teacher_action + 1],
            dtype=torch.long,
            device=device,
        )
        teacher_steps += 1

    raise RuntimeError("episode ended before junction decision")


def avg(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--condition",
        required=True,
        choices=("base", "shared", "independent", "uniform"),
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=91000)
    ap.add_argument(
        "--corridor-lengths",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 0.9],
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    warnings.simplefilter("always")

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")

    device = torch_directml.device()
    model, cfg, sha = build_model(
        args.condition,
        args.checkpoint,
        device,
    )

    print("device:", device)
    print("condition:", args.condition)
    print("checkpoint_sha256:", sha)
    print("episodes_per_length:", args.episodes)
    print("corridor_lengths:", args.corridor_lengths)
    print("paired_assays: closed_loop + junction_retention")

    result = {
        "format": "competitive_decision_assays_v1",
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "base_checkpoint_sha256": sha,
        "episodes_per_length": int(args.episodes),
        "seed_base": int(args.seed_base),
        "corridor_lengths": [
            float(x) for x in args.corridor_lengths
        ],
        "lengths": {},
    }

    for li, corridor_length in enumerate(args.corridor_lengths):
        rows = []

        for i in range(args.episodes):
            seed = args.seed_base + li * 10000 + i

            closed = eval_closed_loop(
                model,
                cfg,
                args.condition,
                device,
                seed,
                corridor_length,
            )

            junction = eval_junction_retention(
                model,
                cfg,
                args.condition,
                device,
                seed,
                corridor_length,
            )

            # eval_closed_loop() is a legacy helper and does not include the
            # seed in its return payload.  The authoritative seed is the one
            # passed here and recorded below.  The junction helper does return
            # its seed, so keep that consistency assertion.
            if int(junction["seed"]) != int(seed):
                raise AssertionError("junction seed mismatch")

            rows.append({
                "seed": int(seed),
                "closed_loop_success": bool(closed["success"]),
                "closed_loop_return": float(closed["return"]),
                "closed_loop_steps": int(closed["steps"]),
                "junction": junction,
            })

        closed_success = avg([
            float(r["closed_loop_success"]) for r in rows
        ])
        junction_full = avg([
            float(r["junction"]["full_action_correct"]) for r in rows
        ])
        junction_side = avg([
            float(r["junction"]["side_choice_correct"]) for r in rows
        ])
        junction_margin = avg([
            r["junction"]["correct_minus_wrong_side_logit"]
            for r in rows
        ])

        left_rows = [
            r for r in rows if r["junction"]["cue"] == "left"
        ]
        right_rows = [
            r for r in rows if r["junction"]["cue"] == "right"
        ]

        key = f"{float(corridor_length):.3f}"
        result["lengths"][key] = {
            "closed_loop_success_rate": closed_success,
            "closed_loop_mean_return": avg([
                r["closed_loop_return"] for r in rows
            ]),
            "closed_loop_mean_steps": avg([
                float(r["closed_loop_steps"]) for r in rows
            ]),
            "junction_full_action_accuracy": junction_full,
            "junction_side_choice_accuracy": junction_side,
            "junction_mean_correct_minus_wrong_side_logit":
                junction_margin,
            "junction_left_side_accuracy": avg([
                float(r["junction"]["side_choice_correct"])
                for r in left_rows
            ]),
            "junction_right_side_accuracy": avg([
                float(r["junction"]["side_choice_correct"])
                for r in right_rows
            ]),
            "per_seed": rows,
        }

        print(
            f"length={float(corridor_length):.3f} "
            f"closed_success={closed_success:.4f} "
            f"junction_full={junction_full:.4f} "
            f"junction_side={junction_side:.4f} "
            f"junction_margin={junction_margin:.6f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("wrote:", output)
    print("PAIRED_DECISION_ASSAYS_COMPLETE")


if __name__ == "__main__":
    main()
