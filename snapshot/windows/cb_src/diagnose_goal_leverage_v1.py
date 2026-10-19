#!/usr/bin/env python3
"""
Continuous Brain v0.20.1 — Goal Leverage / Consequence Calibration Diagnostic.

READ ONLY. NO TRAINING. CPU ONLY.

Purpose
-------
v0.20.0 established that changing only the explicit goal changes the high-level
option at ~10% of decision boundaries, but the two goal profiles do not yet
produce cleanly separated realized behavior.

This diagnostic asks WHY, without changing any learned tensor.

At each option boundary on autonomous DIRECT-OPTION trajectories, it:
1. evaluates EVERY learned option, not just selector top-k;
2. records the historical reward + terminal-Q score;
3. records predicted success, collision and learned-goal components;
4. computes completion-goal and safety-goal rankings;
5. compares all-option ranking against the selector-top2 ranking;
6. measures how large the goal-induced score swing is relative to the
   historical top-option margin;
7. executes the selector's direct option in the real DetourWorld and records
   realized option-segment collisions and whether the segment completes.

This distinguishes:
- selector bottleneck;
- goal-score leverage/scale bottleneck;
- weak consequence calibration;
- or a mixture.

No held-out/generalization claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    _policy_tail_value,
    _selector_candidates,
)
from continuous_brain.infant import one_hot_action
from continuous_brain.environment import DetourWorld
from continuous_brain.goal_conditioned_option_planning_v1 import (
    COMPLETION_GOAL,
    SAFETY_GOAL,
    rollout_option_goal,
)

CURRENT = Path("state/child_v113_dml_smoke/infant_training.pt")
MEMORY = Path(
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
)
INTENT = Path("state/option_intent_executive_v1/latest.portable.pt")
CLOSED_V019 = Path("state/recurrent_option_progress_v1/latest.portable.pt")

EXPECTED = {
    "current":
        "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
    "memory":
        "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "intent":
        "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "v019":
        "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
}

SEED_START = 13001
EPISODES = 20
STEPS = 120
SELECTOR_TOPK = 2
MAX_OPTION_STEPS = 12
DISCOUNT = .95
MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def protected_hashes():
    return {
        "current": sha(CURRENT),
        "memory": sha(MEMORY),
        "intent": sha(INTENT),
        "v019": sha(CLOSED_V019),
    }


def require_provenance():
    paths = {
        "current": CURRENT,
        "memory": MEMORY,
        "intent": INTENT,
        "v019": CLOSED_V019,
    }
    for name, expected in EXPECTED.items():
        p = paths[name]
        if not p.is_file():
            raise FileNotFoundError(p)
        got = sha(p)
        if got != expected:
            raise RuntimeError(
                f"{name} SHA mismatch expected={expected} actual={got}"
            )


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pearson(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / math.sqrt(vx * vy)


def score_components(macro, goal):
    tail = float(macro["q_tail"])
    outer_tail = (DISCOUNT ** int(macro["primitive_steps"])) * tail
    historical = float(macro["reward_score"]) + outer_tail
    goal_adjustment = (
        float(goal.success_weight) * float(macro["success_score"])
        - float(goal.collision_weight) * float(macro["collision_score"])
        + float(goal.goal_value_weight) * float(macro["learned_goal_score"])
        + (float(goal.q_tail_weight) - 1.0) * outer_tail
    )
    return {
        "historical": historical,
        "goal_adjustment": goal_adjustment,
        "total": historical + goal_adjustment,
        "reward": float(macro["reward_score"]),
        "success": float(macro["success_score"]),
        "collision": float(macro["collision_score"]),
        "goal_value": float(macro["learned_goal_score"]),
        "q_tail_discounted": outer_tail,
    }


@torch.no_grad()
def evaluate_options(model, root):
    selector = _selector_candidates(model, root, SELECTOR_TOPK)
    selector_ids = [int(o) for o, _ in selector]
    selector_logits = {int(o): float(logit) for o, logit in selector}

    rows = []
    for option in range(int(model.cfg.option_count)):
        macro = rollout_option_goal(
            model,
            root,
            int(option),
            COMPLETION_GOAL,
            max_option_steps=MAX_OPTION_STEPS,
            discount=DISCOUNT,
            min_commit=MIN_COMMIT,
            terminate_threshold=TERMINATE_THRESHOLD,
        )
        tail = _policy_tail_value(
            model,
            macro["latent"],
            SELECTOR_TOPK,
        )
        macro["q_tail"] = float(tail)

        c = score_components(macro, COMPLETION_GOAL)
        s = score_components(macro, SAFETY_GOAL)

        rows.append({
            "option": int(option),
            "selector_supported": bool(option in selector_ids),
            "selector_logit": selector_logits.get(option),
            "primitive_steps": int(macro["primitive_steps"]),
            "completion": c,
            "safety": s,
        })

    def argmax(goal_name, subset=None):
        candidates = rows if subset is None else [
            r for r in rows if r["option"] in subset
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: r[goal_name]["total"],
        )["option"]

    completion_all = argmax("completion")
    safety_all = argmax("safety")
    completion_top2 = argmax("completion", selector_ids)
    safety_top2 = argmax("safety", selector_ids)

    hist_sorted = sorted(
        rows,
        key=lambda r: r["completion"]["historical"],
        reverse=True,
    )
    historical_margin = (
        hist_sorted[0]["completion"]["historical"]
        - hist_sorted[1]["completion"]["historical"]
        if len(hist_sorted) >= 2 else 0.0
    )

    goal_swings = []
    for r in rows:
        goal_swings.append(
            abs(
                r["completion"]["goal_adjustment"]
                - r["safety"]["goal_adjustment"]
            )
        )
    max_goal_swing = max(goal_swings) if goal_swings else 0.0

    return {
        "selector_ids": selector_ids,
        "selector_logits": selector_logits,
        "options": rows,
        "completion_all": completion_all,
        "safety_all": safety_all,
        "completion_top2": completion_top2,
        "safety_top2": safety_top2,
        "all_options_goal_swap": completion_all != safety_all,
        "top2_goal_swap": completion_top2 != safety_top2,
        "selector_suppressed_swap": (
            completion_all != safety_all
            and completion_top2 == safety_top2
        ),
        "historical_margin": float(historical_margin),
        "max_goal_swing": float(max_goal_swing),
        "goal_swing_to_historical_margin": (
            float(max_goal_swing) / max(float(historical_margin), 1e-9)
        ),
    }


@torch.no_grad()
def run_episode(model, seed):
    env = DetourWorld(seed=int(seed), max_episode_steps=STEPS)
    obs = env.reset()

    hidden = model.nucleus.initial_state(1, "cpu")
    memory = torch.zeros(
        1,
        model.cfg.memory_dim,
        dtype=torch.float32,
    )
    previous_action = torch.zeros(1, dtype=torch.long)

    current_option = None
    option_age = 0
    option_segment = None
    boundaries = []

    completed = False
    steps_used = STEPS
    total_reward = 0.0

    for step in range(1, STEPS + 1):
        image = _env_image_tensor(env, model.cfg.image_size, "cpu")
        context, context7 = _env_context(obs, "cpu")

        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(
            fused,
            context,
            memory,
            previous_action,
            hidden,
        )
        hidden = nuc["hidden"]

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
            MIN_COMMIT,
            TERMINATE_THRESHOLD,
        )

        if boundary:
            if option_segment is not None:
                option_segment["realized_collision_delta"] = (
                    int(env.collisions) - option_segment["start_collisions"]
                )
                option_segment["realized_steps"] = (
                    int(step) - option_segment["start_step"]
                )
                option_segment["realized_completed"] = bool(completed)

            diag = evaluate_options(model, root)
            selector = _selector_candidates(
                model,
                root,
                SELECTOR_TOPK,
            )
            if not selector:
                raise RuntimeError("selector produced no options")
            current_option = int(selector[0][0])
            option_age = 0

            chosen_row = next(
                r for r in diag["options"]
                if r["option"] == current_option
            )
            option_segment = {
                "seed": int(seed),
                "boundary_step": int(step),
                "chosen_option": int(current_option),
                "predicted_collision_score":
                    chosen_row["completion"]["collision"],
                "predicted_success_score":
                    chosen_row["completion"]["success"],
                "predicted_reward_score":
                    chosen_row["completion"]["reward"],
                "start_collisions": int(env.collisions),
                "start_step": int(step),
                "realized_collision_delta": None,
                "realized_steps": None,
                "realized_completed": False,
            }
            diag["realized_segment"] = option_segment
            boundaries.append(diag)

        action = _option_policy_action(
            model,
            root,
            int(current_option),
        )
        option_age += 1
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor(
            [action + 1],
            dtype=torch.long,
        )

        if bool(info.get("reached")):
            completed = True
            steps_used = int(step)

        if done:
            if option_segment is not None:
                option_segment["realized_collision_delta"] = (
                    int(env.collisions) - option_segment["start_collisions"]
                )
                option_segment["realized_steps"] = (
                    int(step) - option_segment["start_step"] + 1
                )
                option_segment["realized_completed"] = bool(completed)
            break

    if option_segment is not None and option_segment["realized_collision_delta"] is None:
        option_segment["realized_collision_delta"] = (
            int(env.collisions) - option_segment["start_collisions"]
        )
        option_segment["realized_steps"] = (
            STEPS - option_segment["start_step"] + 1
        )
        option_segment["realized_completed"] = bool(completed)

    return {
        "seed": int(seed),
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "reward": float(total_reward),
        "boundaries": boundaries,
    }


def diagnose(boundaries):
    n = len(boundaries)
    all_swap = sum(int(x["all_options_goal_swap"]) for x in boundaries)
    top2_swap = sum(int(x["top2_goal_swap"]) for x in boundaries)
    suppressed = sum(int(x["selector_suppressed_swap"]) for x in boundaries)

    ratios = [
        x["goal_swing_to_historical_margin"]
        for x in boundaries
    ]

    segments = [
        x["realized_segment"]
        for x in boundaries
        if x.get("realized_segment")
        and x["realized_segment"]["realized_collision_delta"] is not None
    ]
    pred_collision = [
        s["predicted_collision_score"] for s in segments
    ]
    real_collision = [
        s["realized_collision_delta"] for s in segments
    ]
    pred_success = [
        s["predicted_success_score"] for s in segments
    ]
    real_success = [
        float(s["realized_completed"]) for s in segments
    ]

    all_rate = all_swap / n if n else 0.0
    top2_rate = top2_swap / n if n else 0.0
    suppressed_rate = suppressed / n if n else 0.0
    ratio_median = median(ratios)

    collision_r = pearson(pred_collision, real_collision)
    success_r = pearson(pred_success, real_success)

    findings = []

    if all_rate >= 0.25 and top2_rate <= 0.15 and suppressed_rate >= 0.10:
        findings.append("SELECTOR_CANDIDATE_BOTTLENECK")

    if all_rate < 0.20 or ratio_median < 1.0:
        findings.append("GOAL_SCORE_LEVERAGE_WEAK_RELATIVE_TO_HISTORICAL_MARGIN")

    if collision_r < 0.20:
        findings.append("PREDICTED_COLLISION_CALIBRATION_WEAK")

    if success_r < 0.20:
        findings.append("PREDICTED_SUCCESS_CALIBRATION_WEAK")

    if not findings:
        findings.append("GOAL_LEVERAGE_PRESENT_WITHOUT_SINGLE_DOMINANT_BOTTLENECK")

    return {
        "decision_boundaries": n,
        "all_options_goal_swap_rate": all_rate,
        "top2_goal_swap_rate": top2_rate,
        "selector_suppressed_swap_rate": suppressed_rate,
        "median_goal_swing_to_historical_margin": ratio_median,
        "mean_goal_swing_to_historical_margin": mean(ratios),
        "predicted_vs_realized_collision_pearson": collision_r,
        "predicted_vs_realized_success_pearson": success_r,
        "realized_segments": len(segments),
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="state/goal_leverage_diagnostic_v1/result.json",
    )
    args = ap.parse_args()

    require_provenance()
    before = protected_hashes()

    model, saved = load_checkpoint(CURRENT, "cpu")
    model.eval()

    print("GOAL_LEVERAGE_DIAGNOSTIC_V1")
    print("device: cpu")
    print("training_performed: false")
    print("all_options_enumerated: true")
    print("autonomous_direct_option_trajectories: true")
    print("v019_remains_closed: true")

    episodes = []
    for i in range(EPISODES):
        episodes.append(run_episode(model, SEED_START + i))
        print(f"diagnostic_episodes={i+1}/{EPISODES}")

    boundaries = [
        b
        for ep in episodes
        for b in ep["boundaries"]
    ]
    diagnosis = diagnose(boundaries)

    after = protected_hashes()
    if before != after:
        raise RuntimeError("protected artifact changed")

    result = {
        "format": "continuous_brain.goal_leverage_diagnostic_v1",
        "software_status": "PASS",
        "training_performed": False,
        "device": "cpu",
        "seed_start": SEED_START,
        "episodes": EPISODES,
        "steps": STEPS,
        "protected_unchanged": True,
        "v019_remains_closed": True,
        "goal_profiles": {
            "completion": COMPLETION_GOAL.as_dict(),
            "safety": SAFETY_GOAL.as_dict(),
        },
        "diagnosis": diagnosis,
        "episode_summaries": [
            {
                "seed": e["seed"],
                "completed": e["completed"],
                "steps": e["steps"],
                "collisions": e["collisions"],
                "reward": e["reward"],
                "boundary_count": len(e["boundaries"]),
            }
            for e in episodes
        ],
        "boundaries": boundaries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nDIAGNOSIS")
    for k, v in diagnosis.items():
        if k != "findings":
            print(f"{k}: {v}")
    for finding in diagnosis["findings"]:
        print("finding:", finding)

    print("\nPROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print("GOAL_LEVERAGE_DIAGNOSTIC_RUN: PASS")
    print("result:", output)


if __name__ == "__main__":
    main()
