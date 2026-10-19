#!/usr/bin/env python3
"""
Continuous Brain v0.20.5 — Short-Horizon Same-Option Bootstrap Planner v1

No training. CPU reference backend.

Contract
--------
Real execution:
- an active option remains active until its learned termination fires.

Imagination:
- roll a candidate option for at most `chunk_steps`;
- if learned termination fires inside the chunk, higher-level search may branch;
- if termination has NOT fired at the chunk boundary, DO NOT branch and DO NOT
  open-loop roll farther;
- instead bootstrap with the learned action-value of continuing the SAME option
  with that option's own policy action.

This uses the existing OptionCortex q_head. No new value head is introduced.
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
def same_option_policy_value(
    model: InfantBrain,
    latent: torch.Tensor,
    option: int,
) -> float:
    """Learned Q for continuing `option` with its own policy-selected action."""
    option = int(option)
    action = _option_policy_action(model, latent, option)
    option_t = torch.tensor(
        [option],
        dtype=torch.long,
        device=latent.device,
    )
    action_t = one_hot_action(
        torch.tensor([action], device=latent.device),
        model.cfg.action_dim,
    )
    return float(
        model.options.action_value(
            latent,
            option_t,
            action_t,
        )[0].item()
    )


@torch.no_grad()
def rollout_option_chunk(
    model: InfantBrain,
    latent: torch.Tensor,
    option: int,
    *,
    chunk_steps: int = 12,
    discount: float = .95,
    min_commit: int = 3,
    terminate_threshold: float = .50,
) -> dict:
    """Short open-loop imagination chunk for one learned option."""
    z = latent
    option = int(option)
    rewards = []
    actions = []
    terminated = False
    termination_step = None
    termination_probability = 0.0

    option_t = torch.tensor(
        [option],
        dtype=torch.long,
        device=z.device,
    )

    for local_step in range(max(1, int(chunk_steps))):
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
        termination_probability = float(
            torch.sigmoid(
                model.options.termination_logit(
                    z,
                    option_t,
                )
            )[0].item()
        )

        if (
            age >= int(min_commit)
            and termination_probability
                >= float(terminate_threshold)
        ):
            terminated = True
            termination_step = age
            break

    discounted_reward = sum(
        (float(discount) ** i) * reward
        for i, reward in enumerate(rewards)
    )

    return {
        "latent": z,
        "option": option,
        "actions": actions,
        "primitive_steps": len(actions),
        "discounted_reward": float(discounted_reward),
        "terminated": bool(terminated),
        "termination_step": termination_step,
        "termination_probability":
            float(termination_probability),
        "truncated_while_active": bool(
            not terminated
            and len(actions) >= max(1, int(chunk_steps))
        ),
    }


@torch.no_grad()
def same_option_bootstrap_choice(
    model: InfantBrain,
    root: torch.Tensor,
    *,
    option_depth: int = 2,
    selector_topk: int = 2,
    beam_width: int = 8,
    chunk_steps: int = 12,
    discount: float = .95,
    min_commit: int = 3,
    terminate_threshold: float = .50,
) -> dict:
    """
    High-level option search.

    Search branches only from true predicted option boundaries.
    An unterminated chunk is a terminal search leaf whose value is bootstrapped
    from the SAME option's learned action-value.
    """
    depth_limit = max(1, int(option_depth))
    width = max(1, int(beam_width))

    root_candidates = _selector_candidates(
        model,
        root,
        int(selector_topk),
    )
    if not root_candidates:
        raise RuntimeError(
            "no selector-supported root options"
        )
    selector_choice = int(root_candidates[0][0])

    # Nodes in `frontier` are legal option boundaries where a new option may be
    # chosen. Unterminated chunks go directly to `leaves`.
    frontier = [{
        "latent": root,
        "score": 0.0,
        "primitive_steps": 0,
        "options": [],
        "actions": [],
        "first_selector_logit": float("-inf"),
        "macro_diagnostics": [],
    }]
    leaves = []

    for depth in range(depth_limit):
        expanded_boundaries = []

        for node in frontier:
            candidates = _selector_candidates(
                model,
                node["latent"],
                int(selector_topk),
            )

            for option, selector_logit in candidates:
                macro = rollout_option_chunk(
                    model,
                    node["latent"],
                    int(option),
                    chunk_steps=chunk_steps,
                    discount=discount,
                    min_commit=min_commit,
                    terminate_threshold=terminate_threshold,
                )

                offset = int(node["primitive_steps"])
                next_steps = (
                    offset
                    + int(macro["primitive_steps"])
                )
                score = (
                    float(node["score"])
                    + (float(discount) ** offset)
                    * float(macro["discounted_reward"])
                )

                diagnostic = {
                    "option": int(option),
                    "primitive_steps":
                        int(macro["primitive_steps"]),
                    "terminated":
                        bool(macro["terminated"]),
                    "termination_step":
                        macro["termination_step"],
                    "termination_probability":
                        float(macro[
                            "termination_probability"
                        ]),
                    "truncated_while_active":
                        bool(macro[
                            "truncated_while_active"
                        ]),
                }

                child = {
                    "latent": macro["latent"],
                    "score": float(score),
                    "primitive_steps": next_steps,
                    "options":
                        list(node["options"])
                        + [int(option)],
                    "actions":
                        list(node["actions"])
                        + list(macro["actions"]),
                    "first_selector_logit": (
                        float(selector_logit)
                        if not node["options"]
                        else float(
                            node["first_selector_logit"]
                        )
                    ),
                    "macro_diagnostics":
                        list(node["macro_diagnostics"])
                        + [diagnostic],
                }

                if macro["terminated"]:
                    # Learned termination authorizes higher-level handoff.
                    next_tail = _policy_tail_value(
                        model,
                        child["latent"],
                        int(selector_topk),
                    )
                    child["bootstrap_kind"] = (
                        "next_option"
                    )
                    child["bootstrap_option"] = None
                    child["bootstrap_value"] = float(
                        next_tail
                    )
                    child["rank"] = (
                        float(child["score"])
                        + (
                            float(discount)
                            ** int(
                                child["primitive_steps"]
                            )
                        )
                        * float(next_tail)
                    )

                    if depth + 1 < depth_limit:
                        expanded_boundaries.append(
                            child
                        )
                    else:
                        leaves.append(child)
                else:
                    # CRITICAL REPAIR:
                    # no handoff. Estimate continuation of SAME option.
                    same_tail = same_option_policy_value(
                        model,
                        child["latent"],
                        int(option),
                    )
                    child["bootstrap_kind"] = (
                        "same_option"
                    )
                    child["bootstrap_option"] = int(
                        option
                    )
                    child["bootstrap_value"] = float(
                        same_tail
                    )
                    child["rank"] = (
                        float(child["score"])
                        + (
                            float(discount)
                            ** int(
                                child["primitive_steps"]
                            )
                        )
                        * float(same_tail)
                    )
                    leaves.append(child)

        if not expanded_boundaries:
            frontier = []
            break

        expanded_boundaries.sort(
            key=lambda x: (
                x["rank"],
                x["first_selector_logit"],
            ),
            reverse=True,
        )
        frontier = expanded_boundaries[:width]

    # If depth loop ended with legal boundary nodes that were not emitted,
    # retain their already-computed heuristic rank.
    leaves.extend(frontier)

    if not leaves:
        raise RuntimeError(
            "same-option bootstrap planner "
            "produced no candidates"
        )

    leaves.sort(
        key=lambda x: (
            x["rank"],
            x["first_selector_logit"],
        ),
        reverse=True,
    )
    best = leaves[0]

    if not best["options"]:
        raise RuntimeError("chosen plan has no option")

    illegal_handoffs = 0
    for previous in best["macro_diagnostics"][:-1]:
        if not previous["terminated"]:
            illegal_handoffs += 1

    return {
        "chosen_option":
            int(best["options"][0]),
        "selector_choice":
            int(selector_choice),
        "selector_overridden":
            bool(
                int(best["options"][0])
                != int(selector_choice)
            ),
        "rank": float(best["rank"]),
        "score": float(best["score"]),
        "bootstrap_kind":
            best["bootstrap_kind"],
        "bootstrap_option":
            best["bootstrap_option"],
        "bootstrap_value":
            float(best["bootstrap_value"]),
        "imagined_options":
            list(best["options"]),
        "imagined_actions":
            list(best["actions"]),
        "imagined_primitive_steps":
            int(best["primitive_steps"]),
        "macro_diagnostics":
            list(best["macro_diagnostics"]),
        "illegal_cap_handoffs":
            int(illegal_handoffs),
        "execution_contract_respected":
            bool(illegal_handoffs == 0),
    }
