#!/usr/bin/env python3
"""
Continuous Brain v0.20.4 — Continuation-Consistent Option Imagination v1

READ-ONLY planning repair. No learned tensor changes.

Execution contract:
- computation_chunk_steps is only a compute chunk;
- reaching a chunk boundary does NOT terminate an option;
- imagination may branch to a new option only after learned predicted
  termination;
- if termination is not predicted before the real remaining episode horizon,
  the same option owns the entire imagined remainder.

This repairs the specific mismatch established by
option_continuation_prediction_v1 while preserving all learned weights.
"""
from __future__ import annotations

import torch

from .infant import InfantBrain, one_hot_action
from .planning_v103 import (
    _option_policy_action,
    _policy_tail_value,
    _selector_candidates,
)


@torch.no_grad()
def rollout_option_continuation_consistent(
    model: InfantBrain,
    latent: torch.Tensor,
    option: int,
    *,
    remaining_steps: int,
    computation_chunk_steps: int = 12,
    discount: float = .95,
    min_commit: int = 3,
    terminate_threshold: float = .50,
) -> dict:
    z = latent
    option = int(option)
    remaining_steps = max(1, int(remaining_steps))
    chunk = max(1, int(computation_chunk_steps))

    rewards = []
    actions = []
    term_ps = []
    chunk_boundaries = 0
    terminated = False
    termination_step = None

    option_t = torch.tensor(
        [option],
        dtype=torch.long,
        device=z.device,
    )

    for local_step in range(remaining_steps):
        action = _option_policy_action(model, z, option)
        action_oh = one_hot_action(
            torch.tensor([action], device=z.device),
            model.cfg.action_dim,
        )
        pred = model.planning.step(z, action_oh)

        rewards.append(float(pred["reward"][0].item()))
        actions.append(int(action))
        z = pred["latent"]

        age = local_step + 1
        p_term = float(
            torch.sigmoid(
                model.options.termination_logit(z, option_t)
            )[0].item()
        )
        term_ps.append(p_term)

        if age % chunk == 0 and age < remaining_steps:
            chunk_boundaries += 1

        if (
            age >= int(min_commit)
            and p_term >= float(terminate_threshold)
        ):
            terminated = True
            termination_step = age
            break

    discounted_reward = sum(
        (float(discount) ** i) * r
        for i, r in enumerate(rewards)
    )

    return {
        "latent": z,
        "option": option,
        "actions": actions,
        "primitive_steps": len(actions),
        "discounted_reward": float(discounted_reward),
        "terminated": bool(terminated),
        "termination_step": termination_step,
        "ended_by_episode_horizon": bool(
            not terminated and len(actions) >= remaining_steps
        ),
        "chunk_boundaries_crossed": int(chunk_boundaries),
        "termination_probabilities": term_ps,
    }


@torch.no_grad()
def continuation_consistent_option_choice(
    model: InfantBrain,
    root: torch.Tensor,
    *,
    remaining_steps: int,
    option_depth: int = 2,
    selector_topk: int = 2,
    beam_width: int = 8,
    computation_chunk_steps: int = 12,
    discount: float = .95,
    min_commit: int = 3,
    terminate_threshold: float = .50,
) -> dict:
    remaining_steps = max(1, int(remaining_steps))
    depth_limit = max(1, int(option_depth))
    width = max(1, int(beam_width))

    root_candidates = _selector_candidates(
        model, root, int(selector_topk)
    )
    if not root_candidates:
        raise RuntimeError("no selector-supported root options")

    selector_choice = int(root_candidates[0][0])

    beam = [{
        "latent": root,
        "score": 0.0,
        "primitive_steps": 0,
        "options": [],
        "actions": [],
        "first_selector_logit": float("-inf"),
        "can_branch": True,
        "macro_diagnostics": [],
        "rank": 0.0,
        "tail_value": 0.0,
    }]
    completed = []

    for _depth in range(depth_limit):
        expanded = []

        for node in beam:
            if not node["can_branch"]:
                completed.append(node)
                continue

            node_remaining = (
                remaining_steps - int(node["primitive_steps"])
            )
            if node_remaining <= 0:
                terminal = dict(node)
                terminal["can_branch"] = False
                terminal["rank"] = float(terminal["score"])
                terminal["tail_value"] = 0.0
                completed.append(terminal)
                continue

            candidates = _selector_candidates(
                model,
                node["latent"],
                int(selector_topk),
            )

            for option, selector_logit in candidates:
                macro = rollout_option_continuation_consistent(
                    model,
                    node["latent"],
                    int(option),
                    remaining_steps=node_remaining,
                    computation_chunk_steps=computation_chunk_steps,
                    discount=discount,
                    min_commit=min_commit,
                    terminate_threshold=terminate_threshold,
                )

                offset = int(node["primitive_steps"])
                weight = float(discount) ** offset
                score = (
                    float(node["score"])
                    + weight * float(macro["discounted_reward"])
                )

                can_branch = bool(
                    macro["terminated"]
                    and (
                        offset
                        + int(macro["primitive_steps"])
                        < remaining_steps
                    )
                )

                child = {
                    "latent": macro["latent"],
                    "score": float(score),
                    "primitive_steps":
                        offset + int(macro["primitive_steps"]),
                    "options":
                        list(node["options"]) + [int(option)],
                    "actions":
                        list(node["actions"]) + list(macro["actions"]),
                    "first_selector_logit": (
                        float(selector_logit)
                        if not node["options"]
                        else float(node["first_selector_logit"])
                    ),
                    "can_branch": can_branch,
                    "macro_diagnostics":
                        list(node["macro_diagnostics"]) + [{
                            "option": int(option),
                            "primitive_steps":
                                int(macro["primitive_steps"]),
                            "terminated":
                                bool(macro["terminated"]),
                            "termination_step":
                                macro["termination_step"],
                            "ended_by_episode_horizon":
                                bool(macro["ended_by_episode_horizon"]),
                            "chunk_boundaries_crossed":
                                int(macro["chunk_boundaries_crossed"]),
                        }],
                }

                if can_branch:
                    tail = _policy_tail_value(
                        model,
                        child["latent"],
                        int(selector_topk),
                    )
                    child["tail_value"] = float(tail)
                    child["rank"] = (
                        float(child["score"])
                        + (
                            float(discount)
                            ** int(child["primitive_steps"])
                        ) * float(tail)
                    )
                else:
                    # No handoff exists under the execution contract, so no
                    # next-option tail value is added.
                    child["tail_value"] = 0.0
                    child["rank"] = float(child["score"])

                expanded.append(child)

        if not expanded:
            break

        expanded.sort(
            key=lambda x: (
                x["rank"],
                x["first_selector_logit"],
            ),
            reverse=True,
        )
        beam = expanded[:width]

    candidates = completed + beam
    if not candidates:
        raise RuntimeError(
            "continuation-consistent planner produced no candidates"
        )

    candidates.sort(
        key=lambda x: (
            x["rank"],
            x["first_selector_logit"],
        ),
        reverse=True,
    )
    best = candidates[0]

    if not best["options"]:
        raise RuntimeError("chosen plan has no option")

    illegal = 0
    for prev in best["macro_diagnostics"][:-1]:
        if not prev["terminated"]:
            illegal += 1

    return {
        "chosen_option": int(best["options"][0]),
        "selector_choice": selector_choice,
        "selector_overridden": bool(
            int(best["options"][0]) != selector_choice
        ),
        "rank": float(best["rank"]),
        "score": float(best["score"]),
        "tail_value": float(best["tail_value"]),
        "imagined_options": list(best["options"]),
        "imagined_actions": list(best["actions"]),
        "imagined_primitive_steps":
            int(best["primitive_steps"]),
        "macro_diagnostics":
            list(best["macro_diagnostics"]),
        "illegal_cap_handoffs": int(illegal),
        "execution_contract_respected": bool(illegal == 0),
    }
