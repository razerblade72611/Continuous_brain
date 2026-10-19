"""v0.10.3 hierarchical option-level imagination.

v0.10.2 demonstrated that the child-owned temporal option policy can complete the
DetourWorld task, while primitive-action beam search destroys that behavior as
soon as it is allowed to override the option policy.  v0.10.3 therefore changes
the *planning contract* without changing any learned tensor:

* an active option owns primitive control through ``options.action_policy``;
* recurrent imagination is consulted only at learned option boundaries;
* candidate options are rolled forward as macro-actions by repeatedly applying
  that option's own primitive policy inside the child-owned planning model;
* the high-level planner compares option sequences, never unrestricted primitive
  action sequences.

This module performs no training and adds no parameters.  A v0.10.2 checkpoint
is consumed bit-for-bit unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterable

import torch

from .environment import DetourWorld
from .infant import InfantBrain, one_hot_action
from .parented_train import load_checkpoint
from .planning_v100 import _env_context, _env_image_tensor
from .planning_v102 import _run_option_episode


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_option_should_terminate(model: InfantBrain, latent: torch.Tensor,
                                    current_option: int | None, age: int,
                                    min_commit: int, terminate_threshold: float) -> bool:
    if current_option is None:
        return True
    if int(age) < int(min_commit):
        return False
    option = torch.tensor([int(current_option)], dtype=torch.long, device=latent.device)
    probability = torch.sigmoid(model.options.termination_logit(latent, option))[0].item()
    return bool(probability >= float(terminate_threshold))


def _option_policy_action(model: InfantBrain, latent: torch.Tensor, option: int) -> int:
    option_t = torch.tensor([int(option)], dtype=torch.long, device=latent.device)
    return int(model.options.action_logits(latent, option_t).argmax(dim=-1)[0].item())


def _selector_candidates(model: InfantBrain, latent: torch.Tensor, topk: int) -> list[tuple[int, float]]:
    logits = model.options.selector_logits(latent)[0]
    k = max(1, min(int(topk), int(model.cfg.option_count)))
    values, indices = torch.topk(logits, k=k, dim=-1)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]


def _policy_tail_value(model: InfantBrain, latent: torch.Tensor, selector_topk: int) -> float:
    """Value the best selector-supported next option using its own policy action.

    This is not an unrestricted max over primitive actions.  Each candidate option
    contributes exactly one primitive action: the action selected by that option's
    learned policy at the terminal imagined state.
    """
    values = []
    for option, _ in _selector_candidates(model, latent, selector_topk):
        action = _option_policy_action(model, latent, option)
        option_t = torch.tensor([option], dtype=torch.long, device=latent.device)
        action_t = one_hot_action(torch.tensor([action], device=latent.device), model.cfg.action_dim)
        values.append(float(model.options.action_value(latent, option_t, action_t)[0].item()))
    return max(values) if values else 0.0


@torch.no_grad()
def _rollout_option(model: InfantBrain, latent: torch.Tensor, option: int, *,
                    max_option_steps: int, discount: float, min_commit: int,
                    terminate_threshold: float) -> dict:
    """Imagine one option as a macro-action using only its learned primitive policy."""
    planner = model.planning
    z = latent
    rewards: list[float] = []
    actions: list[int] = []
    terminated = False
    for local_step in range(max(1, int(max_option_steps))):
        action = _option_policy_action(model, z, int(option))
        action_oh = one_hot_action(torch.tensor([action], device=z.device), model.cfg.action_dim)
        pred = planner.step(z, action_oh)
        reward = float(pred["reward"][0].item())
        rewards.append(reward)
        actions.append(int(action))
        z = pred["latent"]
        age = local_step + 1
        if age >= int(min_commit):
            option_t = torch.tensor([int(option)], dtype=torch.long, device=z.device)
            p_term = float(torch.sigmoid(model.options.termination_logit(z, option_t))[0].item())
            if p_term >= float(terminate_threshold):
                terminated = True
                break
    discounted_return = sum((float(discount) ** i) * r for i, r in enumerate(rewards))
    return {
        "latent": z,
        "option": int(option),
        "actions": actions,
        "primitive_steps": len(actions),
        "discounted_reward": float(discounted_return),
        "terminated": bool(terminated),
    }


@torch.no_grad()
def _hierarchical_option_choice(model: InfantBrain, root: torch.Tensor, *,
                                option_depth: int, selector_topk: int,
                                beam_width: int, max_option_steps: int,
                                discount: float, min_commit: int,
                                terminate_threshold: float) -> dict:
    """Beam search over *options*, not primitive actions.

    Each edge in the beam is a deterministic macro rollout of one learned option.
    Primitive actions are always supplied by that option's own action policy.
    """
    depth_limit = max(1, int(option_depth))
    width = max(1, int(beam_width))
    root_candidates = _selector_candidates(model, root, selector_topk)
    selector_choice = int(root_candidates[0][0])
    beam = [{
        "latent": root,
        "score": 0.0,
        "primitive_steps": 0,
        "options": [],
        "actions": [],
        "first_selector_logit": float("-inf"),
    }]

    for depth in range(depth_limit):
        expanded = []
        for node in beam:
            candidates = _selector_candidates(model, node["latent"], selector_topk)
            for option, selector_logit in candidates:
                macro = _rollout_option(
                    model, node["latent"], option,
                    max_option_steps=max_option_steps,
                    discount=discount,
                    min_commit=min_commit,
                    terminate_threshold=terminate_threshold,
                )
                offset = int(node["primitive_steps"])
                score = float(node["score"]) + (float(discount) ** offset) * float(macro["discounted_reward"])
                expanded.append({
                    "latent": macro["latent"],
                    "score": score,
                    "primitive_steps": offset + int(macro["primitive_steps"]),
                    "options": list(node["options"]) + [int(option)],
                    "actions": list(node["actions"]) + list(macro["actions"]),
                    "first_selector_logit": (
                        float(selector_logit) if not node["options"] else float(node["first_selector_logit"])
                    ),
                })
        if not expanded:
            break
        for node in expanded:
            tail = _policy_tail_value(model, node["latent"], selector_topk)
            node["rank"] = float(node["score"]) + (
                float(discount) ** int(node["primitive_steps"])
            ) * float(tail)
        # Selector logit is only a deterministic tie-breaker; it is not blended into the score.
        expanded.sort(key=lambda x: (x["rank"], x["first_selector_logit"]), reverse=True)
        beam = expanded[:width]

    if not beam or not beam[0]["options"]:
        raise RuntimeError("Hierarchical option planner produced no candidate option")
    best = beam[0]
    return {
        "chosen_option": int(best["options"][0]),
        "selector_choice": int(selector_choice),
        "selector_overridden": bool(int(best["options"][0]) != selector_choice),
        "rank": float(best.get("rank", best["score"])),
        "imagined_options": list(best["options"]),
        "imagined_actions": list(best["actions"]),
        "imagined_primitive_steps": int(best["primitive_steps"]),
    }


@torch.no_grad()
def _hierarchical_action(model: InfantBrain, spatial: torch.Tensor, context7: torch.Tensor,
                         current_option: int | None, option_age: int, *,
                         option_depth: int, selector_topk: int, beam_width: int,
                         max_option_steps: int, discount: float, min_commit: int,
                         terminate_threshold: float) -> dict:
    root = model.planning.encode_state(spatial, context7)
    boundary = _active_option_should_terminate(
        model, root, current_option, option_age, min_commit, terminate_threshold
    )
    if not boundary:
        option = int(current_option)
        return {
            "action": _option_policy_action(model, root, option),
            "option": option,
            "option_age": int(option_age),
            "switched": False,
            "high_level_decision": False,
            "selector_overridden": False,
            "selector_choice": option,
            "imagined_options": [option],
            "imagined_actions": [],
            "rank": None,
        }

    plan = _hierarchical_option_choice(
        model, root,
        option_depth=option_depth,
        selector_topk=selector_topk,
        beam_width=beam_width,
        max_option_steps=max_option_steps,
        discount=discount,
        min_commit=min_commit,
        terminate_threshold=terminate_threshold,
    )
    option = int(plan["chosen_option"])
    switched = current_option is None or option != int(current_option)
    return {
        "action": _option_policy_action(model, root, option),
        "option": option,
        "option_age": 0,
        "switched": bool(switched),
        "high_level_decision": True,
        "selector_overridden": bool(plan["selector_overridden"]),
        "selector_choice": int(plan["selector_choice"]),
        "imagined_options": list(plan["imagined_options"]),
        "imagined_actions": list(plan["imagined_actions"]),
        "rank": float(plan["rank"]),
    }


@torch.no_grad()
def _run_hierarchical_episode(model: InfantBrain, seed: int, *, option_depth: int,
                              selector_topk: int, beam_width: int, max_option_steps: int,
                              max_steps: int, discount: float, min_commit: int,
                              terminate_threshold: float, device: str) -> dict:
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    current_option = None
    option_age = 0
    option_switches = 0
    high_level_decisions = 0
    selector_overrides = 0
    planner_overrides = 0
    option_counts = {str(i): 0 for i in range(model.cfg.option_count)}
    action_counts = {str(i): 0 for i in range(model.cfg.action_dim)}
    total_reward = 0.0
    completed = False
    steps_used = int(max_steps)
    high_level_trace = []

    for step in range(1, int(max_steps) + 1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        context, context7 = _env_context(obs, device)
        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]
        reactive = int(torch.argmax(model.motor(nuc["workspace"]), dim=-1)[0].item())
        stable_spatial = model.target_spatial_bus(image)

        decision = _hierarchical_action(
            model, stable_spatial, context7, current_option, option_age,
            option_depth=option_depth,
            selector_topk=selector_topk,
            beam_width=beam_width,
            max_option_steps=max_option_steps,
            discount=discount,
            min_commit=min_commit,
            terminate_threshold=terminate_threshold,
        )
        action = int(decision["action"])
        current_option = int(decision["option"])
        option_age = int(decision["option_age"])
        option_switches += int(decision["switched"])
        high_level_decisions += int(decision["high_level_decision"])
        selector_overrides += int(decision["selector_overridden"])
        if decision["high_level_decision"]:
            high_level_trace.append({
                "step": int(step),
                "chosen_option": int(current_option),
                "selector_choice": int(decision["selector_choice"]),
                "selector_overridden": bool(decision["selector_overridden"]),
                "imagined_options": list(decision["imagined_options"]),
                "imagined_actions": list(decision["imagined_actions"]),
                "rank": decision["rank"],
            })
        option_counts[str(current_option)] += 1
        action_counts[str(action)] += 1
        planner_overrides += int(action != reactive)
        option_age += 1

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
        "mode": "hierarchical_option_planning",
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "reward": float(total_reward),
        "final_energy": float(env.energy),
        "option_switches": int(option_switches),
        "high_level_decisions": int(high_level_decisions),
        "selector_overrides": int(selector_overrides),
        "planner_overrides": int(planner_overrides),
        "option_counts": option_counts,
        "action_counts": action_counts,
        "high_level_trace": high_level_trace,
    }


def _summary(rows: list[dict], max_steps: int) -> dict:
    return {
        "episodes": len(rows),
        "completion_rate": _mean(float(r["completed"]) for r in rows),
        "mean_reward": _mean(r["reward"] for r in rows),
        "mean_steps_penalized": _mean(r["steps"] if r["completed"] else int(max_steps) + 1 for r in rows),
        "mean_collisions": _mean(r["collisions"] for r in rows),
        "mean_option_switches": _mean(r.get("option_switches", 0) for r in rows),
        "mean_planner_overrides": _mean(r.get("planner_overrides", 0) for r in rows),
        "mean_high_level_decisions": _mean(r.get("high_level_decisions", 0) for r in rows),
        "mean_selector_overrides": _mean(r.get("selector_overrides", 0) for r in rows),
    }


def evaluate_hierarchical_options(checkpoint: str | Path, *, episodes: int = 10,
                                  steps: int = 120, seed_start: int = 9001,
                                  option_depth: int = 2, selector_topk: int = 2,
                                  beam_width: int = 8, max_option_steps: int = 12,
                                  discount: float = 0.95, min_commit: int = 3,
                                  terminate_threshold: float = 0.50,
                                  device: str = "cpu", output: str | Path | None = None) -> dict:
    checkpoint = Path(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})
    if metadata.get("option_cortex_version") != "latent-temporal-options-v1":
        raise RuntimeError("v0.10.3 hierarchical planning requires a trained v0.10.2 option checkpoint")

    reactive = []
    direct = []
    hierarchical = []
    for i in range(int(episodes)):
        seed = int(seed_start) + i
        reactive.append(_run_option_episode(
            model, seed, "reactive", horizon=0, beam_width=beam_width, max_steps=steps,
            discount=discount, min_commit=min_commit,
            terminate_threshold=terminate_threshold, device=device,
        ))
        direct.append(_run_option_episode(
            model, seed, "option_direct", horizon=0, beam_width=beam_width, max_steps=steps,
            discount=discount, min_commit=min_commit,
            terminate_threshold=terminate_threshold, device=device,
        ))
        hierarchical.append(_run_hierarchical_episode(
            model, seed,
            option_depth=option_depth,
            selector_topk=selector_topk,
            beam_width=beam_width,
            max_option_steps=max_option_steps,
            max_steps=steps,
            discount=discount,
            min_commit=min_commit,
            terminate_threshold=terminate_threshold,
            device=device,
        ))
        print(f"paired_v103_hierarchical_episodes={i+1}/{episodes}")

    r = _summary(reactive, steps)
    d = _summary(direct, steps)
    h = _summary(hierarchical, steps)
    completion_advantage = h["completion_rate"] - r["completion_rate"]
    collision_regression = h["mean_collisions"] - r["mean_collisions"]
    passed = bool(
        h["completion_rate"] >= 0.20
        and completion_advantage > 0.05
        and collision_regression <= 2.0
    )
    result = {
        "format": "continuous_brain.p4b_hierarchical_option_planning_gate.v103",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "hierarchical_planning_version": "option-macro-imagination-v1",
        "option_cortex_version": metadata.get("option_cortex_version"),
        "planning_cortex_version": metadata.get("planning_cortex_version"),
        "source_training_updates": int(saved.get("updates", metadata.get("updates", 0))),
        "planning_updates": int(metadata.get("planning_updates", 0)),
        "planning_bellman_q_updates": int(metadata.get("planning_bellman_q_updates", 0)),
        "option_updates": int(metadata.get("option_updates", 0)),
        "experiment": {
            "paired_episodes": int(episodes),
            "steps_per_episode": int(steps),
            "seed_start": int(seed_start),
            "seed_end": int(seed_start) + int(episodes) - 1,
            "option_depth": int(option_depth),
            "selector_topk": int(selector_topk),
            "beam_width": int(beam_width),
            "max_option_steps": int(max_option_steps),
            "discount": float(discount),
            "min_commit": int(min_commit),
            "terminate_threshold": float(terminate_threshold),
            "primitive_control": "active option action policy; unrestricted primitive beam search disabled",
            "high_level_search": "beam search over selector-supported learned options as macro-actions",
            "macro_rollout": "deterministic option-policy primitive actions in recurrent child-owned planning latent until learned termination or max_option_steps",
            "score": "imagined discounted reward + selector-supported option-policy terminal action value",
            "selector_logit_usage": "candidate filtering and exact-score tie-break only; no weighted score blend",
            "parents_used": False,
            "training_performed_during_evaluation": False,
            "memory_written": False,
            "detour_geometry_labels_used": False,
        },
        "reactive": r,
        "option_direct_diagnostic": d,
        "hierarchical_option_planning": h,
        "advantages_vs_reactive": {
            "completion_percentage_points": 100.0 * completion_advantage,
            "collisions_reduced": r["mean_collisions"] - h["mean_collisions"],
            "collision_regression": collision_regression,
            "reward_absolute": h["mean_reward"] - r["mean_reward"],
        },
        "thresholds": {
            "absolute_completion": 0.20,
            "completion_advantage": 0.05,
            "max_collision_regression_per_episode": 2.0,
        },
        "p4b_passed": passed,
        "episodes": {
            "reactive": reactive,
            "option_direct": direct,
            "hierarchical_option_planning": hierarchical,
        },
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
