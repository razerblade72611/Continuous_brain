#!/usr/bin/env python3
import argparse
import json
import math
import warnings
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.competitive_infant_brain import CompetitiveInfantBrain
from continuous_brain.cued_detour_world import CuedDetourWorld
from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
    _teacher_action,
)
from continuous_brain.infant import one_hot_action

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"


def build_model(condition, checkpoint, device):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"unexpected base checkpoint SHA256: {sha}")

    use_shared = condition in ("base", "shared", "uniform")
    model = CompetitiveInfantBrain.grow_from(
        base,
        use_shared=use_shared,
    )

    if condition != "base":
        if checkpoint is None:
            raise ValueError(f"--checkpoint is required for {condition}")
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if payload["base_checkpoint_sha256"] != sha:
            raise RuntimeError("checkpoint base SHA mismatch")
        if payload["condition"] != condition:
            raise RuntimeError(
                f"checkpoint condition {payload['condition']!r} "
                f"does not match requested {condition!r}"
            )
        model.competition.load_state_dict(payload["competition"])
        model.recurrent_residual.load_state_dict(payload["recurrent_residual"])
        model.motor_residual.load_state_dict(payload["motor_residual"])
        model.world_residual.load_state_dict(payload["world_residual"])

    if condition == "uniform":
        # Uniform control is defined by a permanently zero selector query.
        with torch.no_grad():
            model.competition.query.weight.zero_()
            model.competition.query.bias.zero_()

    model = model.to(device)
    model.eval()
    model.base.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, sha


def competition_stats(comp):
    if comp is None:
        return [], []

    if "weights" in comp:
        weights = [comp["weights"]]
    else:
        weights = [
            comp["recurrent_weights"],
            comp["motor_weights"],
            comp["world_weights"],
        ]

    entropies = []
    mean_weights = []
    for w in weights:
        wc = w.detach().float().cpu()
        entropy = -(wc * wc.clamp_min(1e-12).log()).sum(dim=-1)
        entropies.extend(float(x) for x in entropy)
        mean_weights.append(
            [float(x) for x in wc.mean(dim=0)]
        )
    return entropies, mean_weights


@torch.no_grad()
def eval_closed_loop(model, cfg, condition, device, seed, corridor_length):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=float(corridor_length),
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    memory = torch.zeros(1, cfg.memory_dim, device=device)

    total_return = 0.0
    entropies = []
    selector_weight_rows = []
    info = {"reached_correct": False}

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
        action = int(torch.argmax(out["logits"], dim=-1).item())

        e, w = competition_stats(out["competition"])
        entropies.extend(e)
        selector_weight_rows.extend(w)

        obs, reward, _done, info = env.step(action)
        total_return += float(reward)
        prev_action = torch.tensor(
            [action + 1],
            dtype=torch.long,
            device=device,
        )

    return {
        "success": bool(info.get("reached_correct", False)),
        "return": total_return,
        "steps": int(env.steps),
        "selector_entropy": (
            sum(entropies) / len(entropies)
            if entropies else None
        ),
        "selector_weights": selector_weight_rows,
    }


@torch.no_grad()
def eval_teacher_forced(model, cfg, condition, device, seed, corridor_length):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=float(corridor_length),
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    memory = torch.zeros(1, cfg.memory_dim, device=device)

    motor_losses = []
    world_losses = []
    correct_actions = 0
    actions = 0
    entropies = []
    selector_weight_rows = []

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(obs, device, cfg)

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

        teacher_action = _teacher_action(env)
        target = torch.tensor(
            [teacher_action],
            dtype=torch.long,
            device=device,
        )

        motor_losses.append(
            float(F.cross_entropy(out["logits"], target).cpu())
        )
        pred = int(torch.argmax(out["logits"], dim=-1).item())
        correct_actions += int(pred == teacher_action)
        actions += 1

        action_oh = one_hot_action(target, cfg.action_dim)
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        e, w = competition_stats(out["competition"])
        entropies.extend(e)
        selector_weight_rows.extend(w)

        obs2, _reward, _done, _info = env.step(teacher_action)
        _, next_context_dyn = _context_tensor(obs2, device, cfg)
        world_losses.append(
            float(
                F.smooth_l1_loss(
                    world_out["next_context"],
                    next_context_dyn,
                ).cpu()
            )
        )

        prev_action = torch.tensor(
            [teacher_action + 1],
            dtype=torch.long,
            device=device,
        )
        obs = obs2

    return {
        "motor_loss": sum(motor_losses) / len(motor_losses),
        "world_loss": sum(world_losses) / len(world_losses),
        "action_accuracy": correct_actions / actions,
        "selector_entropy": (
            sum(entropies) / len(entropies)
            if entropies else None
        ),
        "selector_weights": selector_weight_rows,
    }


def avg(xs):
    return sum(xs) / len(xs) if xs else None


def aggregate_weight_rows(rows):
    if not rows:
        return None
    width = len(rows[0])
    return [
        sum(row[i] for row in rows) / len(rows)
        for i in range(width)
    ]


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

    result = {
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "base_checkpoint_sha256": sha,
        "episodes_per_length": args.episodes,
        "seed_base": args.seed_base,
        "corridor_lengths": args.corridor_lengths,
        "lengths": {},
    }

    for li, corridor_length in enumerate(args.corridor_lengths):
        closed = []
        teacher = []

        for i in range(args.episodes):
            seed = args.seed_base + li * 10000 + i
            closed.append(
                eval_closed_loop(
                    model, cfg, args.condition, device,
                    seed, corridor_length,
                )
            )
            teacher.append(
                eval_teacher_forced(
                    model, cfg, args.condition, device,
                    seed, corridor_length,
                )
            )

        all_closed_weights = [
            row
            for ep in closed
            for row in ep["selector_weights"]
        ]
        all_teacher_weights = [
            row
            for ep in teacher
            for row in ep["selector_weights"]
        ]

        summary = {
            "closed_loop_success_rate": avg(
                [1.0 if x["success"] else 0.0 for x in closed]
            ),
            "closed_loop_mean_return": avg(
                [x["return"] for x in closed]
            ),
            "closed_loop_mean_steps": avg(
                [x["steps"] for x in closed]
            ),
            "closed_loop_selector_entropy": avg(
                [
                    x["selector_entropy"]
                    for x in closed
                    if x["selector_entropy"] is not None
                ]
            ),
            "closed_loop_selector_mean_weights": aggregate_weight_rows(
                all_closed_weights
            ),
            "teacher_motor_loss": avg(
                [x["motor_loss"] for x in teacher]
            ),
            "teacher_world_loss": avg(
                [x["world_loss"] for x in teacher]
            ),
            "teacher_action_accuracy": avg(
                [x["action_accuracy"] for x in teacher]
            ),
            "teacher_selector_entropy": avg(
                [
                    x["selector_entropy"]
                    for x in teacher
                    if x["selector_entropy"] is not None
                ]
            ),
            "teacher_selector_mean_weights": aggregate_weight_rows(
                all_teacher_weights
            ),
        }

        key = f"{corridor_length:.3f}"
        result["lengths"][key] = summary

        print(
            f"length={corridor_length:.3f} "
            f"success={summary['closed_loop_success_rate']:.4f} "
            f"return={summary['closed_loop_mean_return']:.4f} "
            f"steps={summary['closed_loop_mean_steps']:.2f} "
            f"motor_loss={summary['teacher_motor_loss']:.6f} "
            f"world_loss={summary['teacher_world_loss']:.6f} "
            f"action_acc={summary['teacher_action_accuracy']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("wrote:", output)
    print("HELDOUT_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
