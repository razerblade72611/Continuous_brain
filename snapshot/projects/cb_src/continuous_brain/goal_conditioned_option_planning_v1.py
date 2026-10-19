#!/usr/bin/env python3
"""
Continuous Brain v0.20.0 — Goal-Conditioned Hierarchical Option Planning v1

NO TRAINING. CPU reference backend only.

This module preserves all learned tensors and changes only the planner's
high-level *scoring rule*. Primitive control remains owned by the already
trained temporal option policy.

Historical v0.10.3 ranking:
    imagined discounted reward + terminal option-policy action value

Goal-conditioned ranking:
    historical reward
  + success_weight * discounted predicted success probability
  - collision_weight * discounted predicted collision probability
  + goal_value_weight * discounted learned goal_value
  + q_tail_weight * terminal option-policy action value

The intervention is explicit and causal: at every high-level decision boundary
the same root latent is evaluated under both goal profiles. Only the goal
weights differ. The actual environment is advanced by the chosen option's
primitive actions, so later observations depend on model-selected actions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import torch

from .environment import DetourWorld
from .infant import InfantBrain, one_hot_action
from .planning_v100 import _env_context, _env_image_tensor
from .planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    _policy_tail_value,
    _selector_candidates,
)


@dataclass(frozen=True)
class GoalProfile:
    name: str
    success_weight: float
    collision_weight: float
    goal_value_weight: float
    q_tail_weight: float = 1.0

    def as_dict(self):
        return asdict(self)


COMPLETION_GOAL = GoalProfile(
    name="completion",
    success_weight=0.75,
    collision_weight=0.15,
    goal_value_weight=0.25,
    q_tail_weight=1.0,
)

SAFETY_GOAL = GoalProfile(
    name="safety",
    success_weight=0.25,
    collision_weight=1.25,
    goal_value_weight=0.10,
    q_tail_weight=1.0,
)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


@torch.no_grad()
def rollout_option_goal(
    model: InfantBrain,
    latent: torch.Tensor,
    option: int,
    goal: GoalProfile,
    *,
    max_option_steps: int,
    discount: float,
    min_commit: int,
    terminate_threshold: float,
) -> dict:
    planner = model.planning
    z = latent

    actions = []
    rewards = []
    successes = []
    collisions = []
    goal_values = []
    terminated = False

    for local_step in range(max(1, int(max_option_steps))):
        action = _option_policy_action(model, z, int(option))
        action_oh = one_hot_action(
            torch.tensor([action], device=z.device),
            model.cfg.action_dim,
        )
        pred = planner.step(z, action_oh)

        reward = float(pred["reward"][0].item())
        success_p = float(torch.sigmoid(pred["success_logit"])[0].item())
        collision_p = float(torch.sigmoid(pred["collision_logit"])[0].item())
        goal_value = float(pred["goal_value"][0].item())

        actions.append(int(action))
        rewards.append(reward)
        successes.append(success_p)
        collisions.append(collision_p)
        goal_values.append(goal_value)

        z = pred["latent"]
        age = local_step + 1
        if age >= int(min_commit):
            option_t = torch.tensor(
                [int(option)],
                dtype=torch.long,
                device=z.device,
            )
            p_term = float(
                torch.sigmoid(
                    model.options.termination_logit(z, option_t)
                )[0].item()
            )
            if p_term >= float(terminate_threshold):
                terminated = True
                break

    def disc(vals):
        return sum(
            (float(discount) ** i) * float(v)
            for i, v in enumerate(vals)
        )

    reward_score = disc(rewards)
    success_score = disc(successes)
    collision_score = disc(collisions)
    learned_goal_score = disc(goal_values)

    goal_utility = (
        reward_score
        + float(goal.success_weight) * success_score
        - float(goal.collision_weight) * collision_score
        + float(goal.goal_value_weight) * learned_goal_score
    )

    return {
        "latent": z,
        "option": int(option),
        "actions": actions,
        "primitive_steps": len(actions),
        "terminated": bool(terminated),
        "reward_score": float(reward_score),
        "success_score": float(success_score),
        "collision_score": float(collision_score),
        "learned_goal_score": float(learned_goal_score),
        "goal_utility": float(goal_utility),
    }


@torch.no_grad()
def goal_conditioned_option_choice(
    model: InfantBrain,
    root: torch.Tensor,
    goal: GoalProfile,
    *,
    option_depth: int,
    selector_topk: int,
    beam_width: int,
    max_option_steps: int,
    discount: float,
    min_commit: int,
    terminate_threshold: float,
) -> dict:
    depth_limit = max(1, int(option_depth))
    width = max(1, int(beam_width))

    root_candidates = _selector_candidates(
        model,
        root,
        selector_topk,
    )
    if not root_candidates:
        raise RuntimeError("no selector-supported options")

    selector_choice = int(root_candidates[0][0])
    beam = [{
        "latent": root,
        "score": 0.0,
        "primitive_steps": 0,
        "options": [],
        "actions": [],
        "first_selector_logit": float("-inf"),
        "components": {
            "reward": 0.0,
            "success": 0.0,
            "collision": 0.0,
            "goal_value": 0.0,
        },
    }]

    for _depth in range(depth_limit):
        expanded = []
        for node in beam:
            candidates = _selector_candidates(
                model,
                node["latent"],
                selector_topk,
            )
            for option, selector_logit in candidates:
                macro = rollout_option_goal(
                    model,
                    node["latent"],
                    option,
                    goal,
                    max_option_steps=max_option_steps,
                    discount=discount,
                    min_commit=min_commit,
                    terminate_threshold=terminate_threshold,
                )
                offset = int(node["primitive_steps"])
                outer = float(discount) ** offset

                components = dict(node["components"])
                components["reward"] += outer * macro["reward_score"]
                components["success"] += outer * macro["success_score"]
                components["collision"] += outer * macro["collision_score"]
                components["goal_value"] += outer * macro["learned_goal_score"]

                score = (
                    components["reward"]
                    + float(goal.success_weight) * components["success"]
                    - float(goal.collision_weight) * components["collision"]
                    + float(goal.goal_value_weight) * components["goal_value"]
                )

                expanded.append({
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
                    "components": components,
                })

        if not expanded:
            break

        for node in expanded:
            tail = _policy_tail_value(
                model,
                node["latent"],
                selector_topk,
            )
            node["q_tail"] = float(tail)
            node["rank"] = (
                float(node["score"])
                + (float(discount) ** int(node["primitive_steps"]))
                * float(goal.q_tail_weight)
                * float(tail)
            )

        expanded.sort(
            key=lambda x: (
                x["rank"],
                x["first_selector_logit"],
            ),
            reverse=True,
        )
        beam = expanded[:width]

    if not beam or not beam[0]["options"]:
        raise RuntimeError(
            "goal-conditioned planner produced no option"
        )

    best = beam[0]
    return {
        "goal": goal.as_dict(),
        "chosen_option": int(best["options"][0]),
        "selector_choice": int(selector_choice),
        "selector_overridden":
            bool(int(best["options"][0]) != selector_choice),
        "rank": float(best["rank"]),
        "imagined_options": list(best["options"]),
        "imagined_actions": list(best["actions"]),
        "imagined_primitive_steps":
            int(best["primitive_steps"]),
        "score_components": {
            **{
                k: float(v)
                for k, v in best["components"].items()
            },
            "q_tail": float(best.get("q_tail", 0.0)),
        },
    }


@torch.no_grad()
def run_goal_episode(
    model: InfantBrain,
    seed: int,
    goal: GoalProfile,
    *,
    counterfactual_goal: GoalProfile,
    option_depth: int,
    selector_topk: int,
    beam_width: int,
    max_option_steps: int,
    max_steps: int,
    discount: float,
    min_commit: int,
    terminate_threshold: float,
    device: str,
) -> dict:
    env = DetourWorld(
        seed=int(seed),
        max_episode_steps=int(max_steps),
    )
    obs = env.reset()

    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(
        1,
        model.cfg.memory_dim,
        dtype=torch.float32,
        device=device,
    )
    previous_action = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )

    current_option = None
    option_age = 0
    total_reward = 0.0
    completed = False
    steps_used = int(max_steps)

    high_level_decisions = 0
    goal_swap_opportunity = 0
    goal_swap_changed_option = 0
    planner_overrides = 0
    selector_overrides = 0
    option_switches = 0

    action_counts = {
        str(i): 0
        for i in range(model.cfg.action_dim)
    }
    option_counts = {
        str(i): 0
        for i in range(model.cfg.option_count)
    }
    trace = []

    for step in range(1, int(max_steps) + 1):
        image = _env_image_tensor(
            env,
            model.cfg.image_size,
            device,
        )
        context, context7 = _env_context(obs, device)

        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(
            semantic,
            spatial_online,
        )
        nuc = model.nucleus(
            fused,
            context,
            memory,
            previous_action,
            hidden,
        )
        hidden = nuc["hidden"]
        reactive = int(
            torch.argmax(
                model.motor(nuc["workspace"]),
                dim=-1,
            )[0].item()
        )
        stable_spatial = model.target_spatial_bus(image)
        root = model.planning.encode_state(
            stable_spatial,
            context7,
        )

        boundary = _active_option_should_terminate(
            model,
            root,
            current_option,
            option_age,
            min_commit,
            terminate_threshold,
        )

        if boundary:
            chosen = goal_conditioned_option_choice(
                model,
                root,
                goal,
                option_depth=option_depth,
                selector_topk=selector_topk,
                beam_width=beam_width,
                max_option_steps=max_option_steps,
                discount=discount,
                min_commit=min_commit,
                terminate_threshold=terminate_threshold,
            )
            cf = goal_conditioned_option_choice(
                model,
                root,
                counterfactual_goal,
                option_depth=option_depth,
                selector_topk=selector_topk,
                beam_width=beam_width,
                max_option_steps=max_option_steps,
                discount=discount,
                min_commit=min_commit,
                terminate_threshold=terminate_threshold,
            )

            previous_option = current_option
            current_option = int(chosen["chosen_option"])
            option_age = 0
            high_level_decisions += 1
            goal_swap_opportunity += 1
            changed = (
                int(chosen["chosen_option"])
                != int(cf["chosen_option"])
            )
            goal_swap_changed_option += int(changed)
            selector_overrides += int(
                chosen["selector_overridden"]
            )
            option_switches += int(
                previous_option is not None
                and int(previous_option)
                != int(current_option)
            )

            trace.append({
                "step": int(step),
                "active_goal": goal.name,
                "active_choice":
                    int(chosen["chosen_option"]),
                "counterfactual_goal":
                    counterfactual_goal.name,
                "counterfactual_choice":
                    int(cf["chosen_option"]),
                "goal_swap_changed_option":
                    bool(changed),
                "active_plan": chosen,
                "counterfactual_plan": cf,
            })

        action = _option_policy_action(
            model,
            root,
            int(current_option),
        )
        planner_overrides += int(action != reactive)
        option_counts[str(int(current_option))] += 1
        action_counts[str(int(action))] += 1
        option_age += 1

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor(
            [action + 1],
            dtype=torch.long,
            device=device,
        )

        if bool(info.get("reached")):
            completed = True
            steps_used = int(step)
        if done:
            break

    return {
        "seed": int(seed),
        "goal": goal.as_dict(),
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "reward": float(total_reward),
        "final_energy": float(env.energy),
        "high_level_decisions":
            int(high_level_decisions),
        "goal_swap_opportunity":
            int(goal_swap_opportunity),
        "goal_swap_changed_option":
            int(goal_swap_changed_option),
        "goal_swap_change_rate": (
            float(goal_swap_changed_option)
            / float(goal_swap_opportunity)
            if goal_swap_opportunity
            else 0.0
        ),
        "planner_overrides":
            int(planner_overrides),
        "selector_overrides":
            int(selector_overrides),
        "option_switches":
            int(option_switches),
        "action_counts": action_counts,
        "option_counts": option_counts,
        "high_level_trace": trace,
    }


def summarize(rows: list[dict], max_steps: int) -> dict:
    return {
        "episodes": len(rows),
        "completion_rate":
            _mean(float(r["completed"]) for r in rows),
        "mean_reward":
            _mean(r["reward"] for r in rows),
        "mean_steps_penalized":
            _mean(
                r["steps"]
                if r["completed"]
                else int(max_steps) + 1
                for r in rows
            ),
        "mean_collisions":
            _mean(r["collisions"] for r in rows),
        "mean_high_level_decisions":
            _mean(r["high_level_decisions"] for r in rows),
        "mean_goal_swap_change_rate":
            _mean(r["goal_swap_change_rate"] for r in rows),
        "total_goal_swap_opportunities":
            int(sum(r["goal_swap_opportunity"] for r in rows)),
        "total_goal_swap_changed_options":
            int(sum(r["goal_swap_changed_option"] for r in rows)),
        "pooled_goal_swap_change_rate": (
            float(sum(
                r["goal_swap_changed_option"]
                for r in rows
            ))
            / float(sum(
                r["goal_swap_opportunity"]
                for r in rows
            ))
            if sum(
                r["goal_swap_opportunity"]
                for r in rows
            )
            else 0.0
        ),
    }
