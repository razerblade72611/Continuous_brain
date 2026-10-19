#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    _selector_candidates,
)
from continuous_brain.environment import DetourWorld

CURRENT = Path("state/child_v113_dml_smoke/infant_training.pt")
MEMORY = Path("state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt")
INTENT = Path("state/option_intent_executive_v1/latest.portable.pt")
CLOSED_V019 = Path("state/recurrent_option_progress_v1/latest.portable.pt")

EXPECTED = {
    "current": "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
    "memory": "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "intent": "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "v019": "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
}

SEED_START = 14001
EPISODES = 40
MAX_EPISODE_STEPS = 120
MAX_OPTION_STEPS = 12
MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50
MAX_BOUNDARIES = 80


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
    paths = {"current": CURRENT, "memory": MEMORY, "intent": INTENT, "v019": CLOSED_V019}
    for name, expected in EXPECTED.items():
        p = paths[name]
        if not p.is_file():
            raise FileNotFoundError(p)
        got = sha(p)
        if got != expected:
            raise RuntimeError(f"{name} SHA mismatch expected={expected} actual={got}")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pearson(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x*x for x in dx)
    vy = sum(y*y for y in dy)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum(x*y for x, y in zip(dx, dy)) / math.sqrt(vx*vy)


def reconstruct_env(seed, action_history):
    env = DetourWorld(seed=int(seed), max_episode_steps=MAX_EPISODE_STEPS)
    obs = env.reset()
    for action in action_history:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("replay reached terminal state before boundary")
    return env, obs


@torch.no_grad()
def counterfactual_segment(model, seed, action_history, hidden_before, memory, previous_action, option):
    env, obs = reconstruct_env(seed, action_history)
    hidden = hidden_before.clone()
    mem = memory.clone()
    prev = previous_action.clone()
    start_collisions = int(env.collisions)
    total_reward = 0.0
    completed = False
    terminated = False
    actions = []

    for local_step in range(MAX_OPTION_STEPS):
        image = _env_image_tensor(env, model.cfg.image_size, "cpu")
        context, context7 = _env_context(obs, "cpu")

        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, mem, prev, hidden)
        hidden = nuc["hidden"]

        stable_spatial = model.target_spatial_bus(image)
        root = model.planning.encode_state(stable_spatial, context7)

        if local_step >= MIN_COMMIT and _active_option_should_terminate(
            model, root, int(option), local_step, MIN_COMMIT, TERMINATE_THRESHOLD
        ):
            terminated = True
            break

        action = _option_policy_action(model, root, int(option))
        actions.append(int(action))

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        prev = torch.tensor([action + 1], dtype=torch.long)

        if bool(info.get("reached")):
            completed = True
        if done:
            break

    return {
        "option": int(option),
        "completed": bool(completed),
        "collision_delta": int(env.collisions) - start_collisions,
        "reward": float(total_reward),
        "steps": len(actions),
        "terminated": bool(terminated),
        "actions": actions,
    }


def quality_tuple(row):
    return (
        int(row["completed"]),
        -int(row["collision_delta"]),
        float(row["reward"]),
    )


@torch.no_grad()
def run_episode(model, seed, boundary_budget):
    env = DetourWorld(seed=int(seed), max_episode_steps=MAX_EPISODE_STEPS)
    obs = env.reset()

    hidden = model.nucleus.initial_state(1, "cpu")
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32)
    previous_action = torch.zeros(1, dtype=torch.long)

    current_option = None
    option_age = 0
    action_history = []
    boundaries = []

    for step in range(1, MAX_EPISODE_STEPS + 1):
        hidden_before = hidden.clone()
        memory_before = memory.clone()
        previous_before = previous_action.clone()

        image = _env_image_tensor(env, model.cfg.image_size, "cpu")
        context, context7 = _env_context(obs, "cpu")

        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]

        stable_spatial = model.target_spatial_bus(image)
        root = model.planning.encode_state(stable_spatial, context7)

        boundary = _active_option_should_terminate(
            model, root, current_option, option_age, MIN_COMMIT, TERMINATE_THRESHOLD
        )

        if boundary:
            all_candidates = _selector_candidates(model, root, int(model.cfg.option_count))
            if len(all_candidates) != int(model.cfg.option_count):
                raise RuntimeError("selector did not return every option")

            selector_rows = [
                {
                    "option": int(option),
                    "selector_rank": rank + 1,
                    "selector_logit": float(logit),
                    "top2": bool(rank < 2),
                }
                for rank, (option, logit) in enumerate(all_candidates)
            ]

            realized = []
            for sr in selector_rows:
                rr = counterfactual_segment(
                    model,
                    seed,
                    list(action_history),
                    hidden_before,
                    memory_before,
                    previous_before,
                    sr["option"],
                )
                realized.append({**sr, **rr})

            best = max(realized, key=quality_tuple)
            chosen = int(all_candidates[0][0])

            boundaries.append({
                "seed": int(seed),
                "step": int(step),
                "actual_selector_choice": chosen,
                "best_realized_option": int(best["option"]),
                "best_realized_in_top2": bool(best["top2"]),
                "options": realized,
            })

            current_option = chosen
            option_age = 0

            if len(boundaries) >= boundary_budget:
                return boundaries

        action = _option_policy_action(model, root, int(current_option))
        option_age += 1
        action_history.append(int(action))

        obs, _, done, _ = env.step(action)
        previous_action = torch.tensor([action + 1], dtype=torch.long)
        if done:
            break

    return boundaries


def summarize(boundaries):
    options = [row for b in boundaries for row in b["options"]]
    top2 = [r for r in options if r["top2"]]
    outside = [r for r in options if not r["top2"]]

    selector_logits = [r["selector_logit"] for r in options]
    completion = [float(r["completed"]) for r in options]
    collisions = [float(r["collision_delta"]) for r in options]
    rewards = [float(r["reward"]) for r in options]
    ranks = [float(r["selector_rank"]) for r in options]

    return {
        "boundaries": len(boundaries),
        "option_counterfactuals": len(options),
        "top2": {
            "n": len(top2),
            "completion_rate": mean(float(r["completed"]) for r in top2),
            "mean_collision_delta": mean(r["collision_delta"] for r in top2),
            "mean_reward": mean(r["reward"] for r in top2),
        },
        "outside_top2": {
            "n": len(outside),
            "completion_rate": mean(float(r["completed"]) for r in outside),
            "mean_collision_delta": mean(r["collision_delta"] for r in outside),
            "mean_reward": mean(r["reward"] for r in outside),
        },
        "best_realized_option_outside_top2_rate": mean(
            float(not b["best_realized_in_top2"]) for b in boundaries
        ),
        "selector_top1_is_best_realized_rate": mean(
            float(b["actual_selector_choice"] == b["best_realized_option"])
            for b in boundaries
        ),
        "selector_logit_vs_completion_pearson": pearson(selector_logits, completion),
        "selector_logit_vs_collision_pearson": pearson(selector_logits, collisions),
        "selector_logit_vs_reward_pearson": pearson(selector_logits, rewards),
        "selector_rank_vs_collision_pearson": pearson(ranks, collisions),
        "selector_rank_vs_reward_pearson": pearson(ranks, rewards),
    }


def diagnose(s):
    findings = []

    if (
        s["top2"]["completion_rate"] >= s["outside_top2"]["completion_rate"] + .10
        or s["top2"]["mean_collision_delta"] <= s["outside_top2"]["mean_collision_delta"] - .50
    ):
        findings.append("SELECTOR_IS_REALIZED_COMPETENCE_PRIOR")

    if s["best_realized_option_outside_top2_rate"] >= .20:
        findings.append("SELECTOR_ALSO_SUPPRESSES_SOME_USEFUL_OPTIONS")

    if (
        "SELECTOR_IS_REALIZED_COMPETENCE_PRIOR" in findings
        and "SELECTOR_ALSO_SUPPRESSES_SOME_USEFUL_OPTIONS" in findings
    ):
        recommendation = (
            "Keep selector support as a soft competence prior. "
            "Permit goal-driven challengers only when selector support and "
            "predicted-success evidence jointly justify promotion."
        )
    elif "SELECTOR_IS_REALIZED_COMPETENCE_PRIOR" in findings:
        recommendation = (
            "Keep hard selector support for now and improve goal sensitivity "
            "inside selector-supported options."
        )
    else:
        recommendation = (
            "Selector support is not strongly validated by realized segments; "
            "investigate option execution/model-based ranking before another planner change."
        )

    return {
        "findings": findings or ["NO_STRONG_SELECTOR_COMPETENCE_SIGNAL"],
        "recommendation": recommendation,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="state/option_support_counterfactual_v1/result.json",
    )
    args = ap.parse_args()

    require_provenance()
    before = protected_hashes()

    model, _ = load_checkpoint(CURRENT, "cpu")
    model.eval()

    print("OPTION_SUPPORT_COUNTERFACTUAL_V1")
    print("device: cpu")
    print("training_performed: false")
    print("development_window_reused: 14001-14040")
    print("new_development_window_consumed: false")
    print("all_options_real_environment_counterfactual: true")
    print("max_boundaries:", MAX_BOUNDARIES)

    boundaries = []
    for i in range(EPISODES):
        remaining = MAX_BOUNDARIES - len(boundaries)
        if remaining <= 0:
            break
        rows = run_episode(model, SEED_START + i, remaining)
        boundaries.extend(rows)
        print(
            f"diagnostic_episodes={i+1}/{EPISODES} "
            f"boundaries={len(boundaries)}/{MAX_BOUNDARIES}"
        )
        if len(boundaries) >= MAX_BOUNDARIES:
            break

    if not boundaries:
        raise RuntimeError("no option boundaries collected")

    summary = summarize(boundaries)
    diagnosis = diagnose(summary)

    after = protected_hashes()
    if before != after:
        raise RuntimeError("protected artifact changed")

    result = {
        "format": "continuous_brain.option_support_counterfactual_v1",
        "software_status": "PASS",
        "training_performed": False,
        "device": "cpu",
        "seed_window": {
            "start": SEED_START,
            "end": SEED_START + EPISODES - 1,
            "reused_from_v0202": True,
            "new_development_window_consumed": False,
        },
        "summary": summary,
        "diagnosis": diagnosis,
        "boundaries": boundaries,
        "protected_unchanged": True,
        "v019_remains_closed": True,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nDIAGNOSIS")
    for finding in diagnosis["findings"]:
        print("finding:", finding)
    print("recommendation:", diagnosis["recommendation"])

    print("\nPROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print("OPTION_SUPPORT_COUNTERFACTUAL_RUN: PASS")
    print("result:", output)


if __name__ == "__main__":
    main()
