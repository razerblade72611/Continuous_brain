#!/usr/bin/env python3
"""
Continuous Brain v0.20.6 — Fresh Validation of Same-Option Bootstrap Planner

NO TRAINING. CPU reference backend.

This is the single frozen validation promised after v0.20.5.

Fresh relative to prior planner-development windows:
    seeds 15001..15100

No claim is made that these seeds were globally held out from every historical
training run. The claim is narrower: they were not used in v0.20.0-v0.20.5
planner diagnosis/repair selection.

Promotion gate (frozen before run)
----------------------------------
- illegal cap handoffs == 0
- same-option bootstrap used > 0
- repaired completion >= historical hierarchical completion
- repaired completion >= direct option completion - 0.05
- repaired collisions <= historical hierarchical collisions + 5
- repaired collisions <= direct option collisions + 10

If PASS:
    freeze continuation-bootstrap semantics and move to memory-guided autonomous
    planning. Do not continue Detour planner tuning.

If FAIL:
    stop planner-repair tuning anyway. Use the stronger direct learned-option
    controller as execution backbone for the next memory/goal milestone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import _env_context, _env_image_tensor
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    evaluate_hierarchical_options,
)
from continuous_brain.environment import DetourWorld
from continuous_brain.same_option_bootstrap_planning_v1 import (
    same_option_bootstrap_choice,
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

SEED_START = 15001
EPISODES = 100
STEPS = 120

OPTION_DEPTH = 2
SELECTOR_TOPK = 2
BEAM_WIDTH = 8
CHUNK_STEPS = 12
DISCOUNT = .95
MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50

DIRECT_COMPLETION_TOLERANCE = .05
HISTORICAL_COLLISION_TOLERANCE = 5.0
DIRECT_COLLISION_TOLERANCE = 10.0

BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260919


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


def percentile(xs, q):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def paired_bootstrap_difference(a_rows, b_rows, field, draws=BOOTSTRAP_DRAWS):
    """
    Difference A-B over paired seeds.
    For completion this is a paired binary mean difference.
    """
    amap = {int(r["seed"]): r for r in a_rows}
    bmap = {int(r["seed"]): r for r in b_rows}
    seeds = sorted(set(amap) & set(bmap))
    if not seeds:
        raise RuntimeError("no paired seeds")

    diffs = []
    for seed in seeds:
        if field == "completed":
            av = float(bool(amap[seed]["completed"]))
            bv = float(bool(bmap[seed]["completed"]))
        else:
            av = float(amap[seed][field])
            bv = float(bmap[seed][field])
        diffs.append(av - bv)

    observed = mean(diffs)
    rng = random.Random(BOOTSTRAP_SEED + hash(field) % 10000)
    boots = []
    n = len(diffs)
    for _ in range(int(draws)):
        boots.append(
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        )

    return {
        "n_pairs": n,
        "difference": observed,
        "ci95_low": percentile(boots, .025),
        "ci95_high": percentile(boots, .975),
    }


@torch.no_grad()
def run_episode(model, seed):
    env = DetourWorld(
        seed=int(seed),
        max_episode_steps=STEPS,
    )
    obs = env.reset()

    hidden = model.nucleus.initial_state(1, "cpu")
    memory = torch.zeros(
        1,
        model.cfg.memory_dim,
        dtype=torch.float32,
    )
    previous_action = torch.zeros(
        1,
        dtype=torch.long,
    )

    current_option = None
    option_age = 0
    total_reward = 0.0
    completed = False
    steps_used = STEPS

    high_level = 0
    selector_overrides = 0
    option_switches = 0
    illegal_cap_handoffs = 0
    same_option_bootstraps = 0
    next_option_bootstraps = 0

    for step in range(1, STEPS + 1):
        image = _env_image_tensor(
            env,
            model.cfg.image_size,
            "cpu",
        )
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
            plan = same_option_bootstrap_choice(
                model,
                root,
                option_depth=OPTION_DEPTH,
                selector_topk=SELECTOR_TOPK,
                beam_width=BEAM_WIDTH,
                chunk_steps=CHUNK_STEPS,
                discount=DISCOUNT,
                min_commit=MIN_COMMIT,
                terminate_threshold=TERMINATE_THRESHOLD,
            )

            previous = current_option
            current_option = int(plan["chosen_option"])
            option_age = 0
            high_level += 1
            selector_overrides += int(plan["selector_overridden"])
            option_switches += int(
                previous is not None
                and int(previous) != int(current_option)
            )
            illegal_cap_handoffs += int(plan["illegal_cap_handoffs"])
            same_option_bootstraps += int(
                plan["bootstrap_kind"] == "same_option"
            )
            next_option_bootstraps += int(
                plan["bootstrap_kind"] == "next_option"
            )

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
            steps_used = step

        if done:
            break

    return {
        "seed": int(seed),
        "completed": bool(completed),
        "steps": int(steps_used),
        "collisions": int(env.collisions),
        "reward": float(total_reward),
        "high_level_decisions": int(high_level),
        "selector_overrides": int(selector_overrides),
        "option_switches": int(option_switches),
        "illegal_cap_handoffs": int(illegal_cap_handoffs),
        "same_option_bootstraps": int(same_option_bootstraps),
        "next_option_bootstraps": int(next_option_bootstraps),
    }


def summarize(rows):
    return {
        "episodes": len(rows),
        "completion_rate": mean(float(r["completed"]) for r in rows),
        "mean_collisions": mean(r["collisions"] for r in rows),
        "mean_reward": mean(r["reward"] for r in rows),
        "mean_steps_penalized": mean(
            r["steps"] if r["completed"] else STEPS + 1
            for r in rows
        ),
        "mean_high_level_decisions":
            mean(r["high_level_decisions"] for r in rows),
        "mean_selector_overrides":
            mean(r["selector_overrides"] for r in rows),
        "mean_option_switches":
            mean(r["option_switches"] for r in rows),
        "illegal_cap_handoffs":
            sum(r["illegal_cap_handoffs"] for r in rows),
        "same_option_bootstraps":
            sum(r["same_option_bootstraps"] for r in rows),
        "next_option_bootstraps":
            sum(r["next_option_bootstraps"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="state/same_option_bootstrap_fresh_validation_v1/result.json",
    )
    args = ap.parse_args()

    require_provenance()
    before = protected_hashes()

    print("SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION_V1")
    print("device: cpu")
    print("training_performed: false")
    print("seed_start:", SEED_START)
    print("episodes:", EPISODES)
    print("fresh_relative_to_planner_repair_windows: true")
    print("global_training_holdout_claimed: false")
    print("detour_planner_tuning_after_this_run: forbidden")

    baseline = evaluate_hierarchical_options(
        CURRENT,
        episodes=EPISODES,
        steps=STEPS,
        seed_start=SEED_START,
        device="cpu",
        output=None,
        option_depth=OPTION_DEPTH,
        selector_topk=SELECTOR_TOPK,
        beam_width=BEAM_WIDTH,
        max_option_steps=CHUNK_STEPS,
        discount=DISCOUNT,
        min_commit=MIN_COMMIT,
        terminate_threshold=TERMINATE_THRESHOLD,
    )

    if baseline.get("checkpoint_sha256") != EXPECTED["current"]:
        raise RuntimeError("baseline checkpoint provenance mismatch")
    if baseline.get("experiment", {}).get(
        "training_performed_during_evaluation"
    ) is not False:
        raise RuntimeError("baseline unexpectedly trained")

    model, _ = load_checkpoint(CURRENT, "cpu")
    model.eval()

    repaired_rows = []
    for i in range(EPISODES):
        repaired_rows.append(
            run_episode(model, SEED_START + i)
        )
        print(f"fresh_validation_episodes={i+1}/{EPISODES}")

    repaired = summarize(repaired_rows)
    historical = baseline["hierarchical_option_planning"]
    direct = baseline["option_direct_diagnostic"]

    historical_rows = baseline["episodes"]["hierarchical_option_planning"]
    direct_rows = baseline["episodes"]["option_direct_diagnostic"]

    paired_stats = {
        "completion_vs_historical":
            paired_bootstrap_difference(
                repaired_rows, historical_rows, "completed"
            ),
        "completion_vs_direct":
            paired_bootstrap_difference(
                repaired_rows, direct_rows, "completed"
            ),
        "collisions_vs_historical":
            paired_bootstrap_difference(
                repaired_rows, historical_rows, "collisions"
            ),
        "collisions_vs_direct":
            paired_bootstrap_difference(
                repaired_rows, direct_rows, "collisions"
            ),
    }

    checks = {
        "execution_contract":
            repaired["illegal_cap_handoffs"] == 0,
        "same_option_bootstrap_used":
            repaired["same_option_bootstraps"] > 0,
        "completion_ge_historical":
            repaired["completion_rate"]
            >= float(historical["completion_rate"]),
        "completion_near_direct":
            repaired["completion_rate"]
            >= float(direct["completion_rate"])
            - DIRECT_COMPLETION_TOLERANCE,
        "collisions_near_historical":
            repaired["mean_collisions"]
            <= float(historical["mean_collisions"])
            + HISTORICAL_COLLISION_TOLERANCE,
        "collisions_near_direct":
            repaired["mean_collisions"]
            <= float(direct["mean_collisions"])
            + DIRECT_COLLISION_TOLERANCE,
    }
    passed = all(checks.values())

    after = protected_hashes()
    if before != after:
        raise RuntimeError("protected artifact changed")

    result = {
        "format":
            "continuous_brain.same_option_bootstrap_fresh_validation_v1",
        "validation_passed": bool(passed),
        "training_performed": False,
        "device": "cpu",
        "seed_protocol": {
            "seed_start": SEED_START,
            "seed_end": SEED_START + EPISODES - 1,
            "episodes": EPISODES,
            "fresh_relative_to_v0200_v0205":
                True,
            "global_training_holdout_claimed":
                False,
        },
        "checks": checks,
        "thresholds": {
            "direct_completion_tolerance":
                DIRECT_COMPLETION_TOLERANCE,
            "historical_collision_tolerance":
                HISTORICAL_COLLISION_TOLERANCE,
            "direct_collision_tolerance":
                DIRECT_COLLISION_TOLERANCE,
        },
        "baseline": {
            "direct": direct,
            "historical_hierarchical":
                historical,
        },
        "same_option_bootstrap": repaired,
        "paired_bootstrap_ci95": paired_stats,
        "episodes": {
            "same_option_bootstrap": repaired_rows,
            "direct": direct_rows,
            "historical_hierarchical":
                historical_rows,
        },
        "protected_unchanged": True,
        "v019_remains_closed": True,
        "memory_preserved": True,
        "v018_intent_preserved": True,
        "next_step_if_pass": (
            "Freeze continuation-bootstrap planner and move directly to "
            "memory-guided autonomous planning."
        ),
        "next_step_if_fail": (
            "Stop Detour planner repair and use direct learned options as "
            "execution backbone for memory-guided autonomous planning."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nBASELINE")
    print(
        "direct completion/collisions:",
        direct["completion_rate"],
        direct["mean_collisions"],
    )
    print(
        "historical hierarchical completion/collisions:",
        historical["completion_rate"],
        historical["mean_collisions"],
    )

    print("\nSAME_OPTION_BOOTSTRAP")
    print(
        "completion/collisions:",
        repaired["completion_rate"],
        repaired["mean_collisions"],
    )
    print(
        "illegal_cap_handoffs:",
        repaired["illegal_cap_handoffs"],
    )
    print(
        "same_option_bootstraps:",
        repaired["same_option_bootstraps"],
    )
    print(
        "next_option_bootstraps:",
        repaired["next_option_bootstraps"],
    )

    print("\nPAIRED_BOOTSTRAP_CI95")
    for key, value in paired_stats.items():
        print(
            key,
            "diff=", value["difference"],
            "ci95=[",
            value["ci95_low"],
            ",",
            value["ci95_high"],
            "]",
        )

    print()
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print(
        "DETOUR_PLANNER_TUNING_AFTER_THIS_RUN: forbidden"
    )
    print(
        "SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION:",
        "PASS" if passed else "FAIL",
    )
    print("SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION_COMPLETE")
    print("result:", output)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
