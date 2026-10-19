#!/usr/bin/env python3
"""
v0.20.5 spent-window repair diagnostic.

No training. CPU only.
Reuses 14001..14040. This is NOT a fresh scientific admission test.

Development repair criterion (frozen before run):
- no illegal cap handoffs;
- same-option bootstrap is actually selected;
- completion >= 0.475 (historical 0.525 minus 0.05);
- collisions <= historical + 5;
- completion improves by >= 0.20 absolute over v0.20.4's 0.15.

PASS => freeze and run ONE fresh-seed validation, then move to memory-guided
planning.
FAIL => stop planner-score repair loop and use direct options as the execution
backbone for the next memory/goal integration milestone.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from continuous_brain.parented_train import load_checkpoint
from continuous_brain.planning_v100 import (
    _env_context,
    _env_image_tensor,
)
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
    evaluate_hierarchical_options,
)
from continuous_brain.environment import DetourWorld
from continuous_brain.same_option_bootstrap_planning_v1 import (
    same_option_bootstrap_choice,
)

CURRENT = Path(
    "state/child_v113_dml_smoke/infant_training.pt"
)
MEMORY = Path(
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
INTENT = Path(
    "state/option_intent_executive_v1/latest.portable.pt"
)
CLOSED_V019 = Path(
    "state/recurrent_option_progress_v1/latest.portable.pt"
)

EXPECTED = {
    "current":
        "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
    "memory":
        "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "intent":
        "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "v019":
        "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
    "planning_v103_source":
        "35541798a810ae50ba287f784819f29f142f06cf22429bd8b84f1afe48c0c074",
    "infant_source":
        "592f5a29e3c7b6b374f9637f0fb4db47d3fd3f484d36b24352249b60876db3fe",
}

SEED_START = 14001
EPISODES = 40
STEPS = 120
OPTION_DEPTH = 2
SELECTOR_TOPK = 2
BEAM_WIDTH = 8
CHUNK_STEPS = 12
DISCOUNT = .95
MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50

HISTORICAL_COMPLETION_FLOOR = .475
MAX_COLLISION_DELTA_VS_HISTORICAL = 5.0
MIN_IMPROVEMENT_VS_V0204 = .20
V0204_COMPLETION = .15


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(c)
    return h.hexdigest()


def protected():
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
    for name in (
        "current",
        "memory",
        "intent",
        "v019",
    ):
        p = paths[name]
        if not p.is_file():
            raise FileNotFoundError(p)
        got = sha(p)
        if got != EXPECTED[name]:
            raise RuntimeError(
                f"{name} SHA mismatch "
                f"expected={EXPECTED[name]} "
                f"actual={got}"
            )

    import continuous_brain.infant as infant
    import continuous_brain.planning_v103 as p103

    if sha(Path(infant.__file__)) != EXPECTED[
        "infant_source"
    ]:
        raise RuntimeError(
            "infant.py differs from audited source"
        )
    if sha(Path(p103.__file__)) != EXPECTED[
        "planning_v103_source"
    ]:
        raise RuntimeError(
            "planning_v103.py differs from audited source"
        )


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


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
    traces = []

    for step in range(1, STEPS + 1):
        image = _env_image_tensor(
            env,
            model.cfg.image_size,
            "cpu",
        )
        context, context7 = _env_context(
            obs,
            "cpu",
        )

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

        stable_spatial = model.target_spatial_bus(
            image
        )
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
                terminate_threshold=(
                    TERMINATE_THRESHOLD
                ),
            )

            previous = current_option
            current_option = int(
                plan["chosen_option"]
            )
            option_age = 0

            high_level += 1
            selector_overrides += int(
                plan["selector_overridden"]
            )
            option_switches += int(
                previous is not None
                and int(previous)
                    != int(current_option)
            )
            illegal_cap_handoffs += int(
                plan["illegal_cap_handoffs"]
            )
            same_option_bootstraps += int(
                plan["bootstrap_kind"]
                == "same_option"
            )
            next_option_bootstraps += int(
                plan["bootstrap_kind"]
                == "next_option"
            )

            traces.append({
                "step": int(step),
                "plan": plan,
            })

        action = _option_policy_action(
            model,
            root,
            int(current_option),
        )
        option_age += 1

        obs, reward, done, info = env.step(
            action
        )
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
        "selector_overrides":
            int(selector_overrides),
        "option_switches": int(option_switches),
        "illegal_cap_handoffs":
            int(illegal_cap_handoffs),
        "same_option_bootstraps":
            int(same_option_bootstraps),
        "next_option_bootstraps":
            int(next_option_bootstraps),
        "trace": traces,
    }


def summary(rows):
    return {
        "episodes": len(rows),
        "completion_rate":
            mean(float(r["completed"]) for r in rows),
        "mean_collisions":
            mean(r["collisions"] for r in rows),
        "mean_reward":
            mean(r["reward"] for r in rows),
        "mean_steps_penalized":
            mean(
                r["steps"]
                if r["completed"]
                else STEPS + 1
                for r in rows
            ),
        "mean_high_level_decisions":
            mean(
                r["high_level_decisions"]
                for r in rows
            ),
        "mean_selector_overrides":
            mean(
                r["selector_overrides"]
                for r in rows
            ),
        "mean_option_switches":
            mean(
                r["option_switches"]
                for r in rows
            ),
        "illegal_cap_handoffs":
            sum(
                r["illegal_cap_handoffs"]
                for r in rows
            ),
        "same_option_bootstraps":
            sum(
                r["same_option_bootstraps"]
                for r in rows
            ),
        "next_option_bootstraps":
            sum(
                r["next_option_bootstraps"]
                for r in rows
            ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=(
            "state/same_option_bootstrap_planning_v1/"
            "result.json"
        ),
    )
    args = ap.parse_args()

    require_provenance()
    before = protected()

    print("SAME_OPTION_BOOTSTRAP_PLANNING_V1")
    print("device: cpu")
    print("training_performed: false")
    print(
        "spent_development_window: 14001-14040"
    )
    print(
        "new_development_window_consumed: false"
    )
    print("chunk_steps:", CHUNK_STEPS)
    print(
        "unterminated_chunk_bootstrap: "
        "same_option_policy_action_value"
    )
    print(
        "chunk_boundary_allows_option_switch: false"
    )

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
        terminate_threshold=(
            TERMINATE_THRESHOLD
        ),
    )
    if (
        baseline.get("checkpoint_sha256")
        != EXPECTED["current"]
    ):
        raise RuntimeError(
            "baseline provenance mismatch"
        )

    model, _ = load_checkpoint(
        CURRENT,
        "cpu",
    )
    model.eval()

    rows = []
    for i in range(EPISODES):
        rows.append(
            run_episode(
                model,
                SEED_START + i,
            )
        )
        print(
            f"same_option_bootstrap_episodes="
            f"{i+1}/{EPISODES}"
        )

    repaired = summary(rows)

    after = protected()
    if before != after:
        raise RuntimeError(
            "protected artifact changed"
        )

    historical = baseline[
        "hierarchical_option_planning"
    ]
    direct = baseline[
        "option_direct_diagnostic"
    ]

    comparison = {
        "completion_delta_vs_historical":
            repaired["completion_rate"]
            - float(
                historical["completion_rate"]
            ),
        "collision_delta_vs_historical":
            repaired["mean_collisions"]
            - float(
                historical["mean_collisions"]
            ),
        "completion_delta_vs_direct":
            repaired["completion_rate"]
            - float(direct["completion_rate"]),
        "collision_delta_vs_direct":
            repaired["mean_collisions"]
            - float(direct["mean_collisions"]),
        "completion_delta_vs_v0204":
            repaired["completion_rate"]
            - V0204_COMPLETION,
    }

    checks = {
        "execution_contract":
            repaired[
                "illegal_cap_handoffs"
            ] == 0,
        "same_option_bootstrap_used":
            repaired[
                "same_option_bootstraps"
            ] > 0,
        "completion_near_historical":
            repaired["completion_rate"]
            >= HISTORICAL_COMPLETION_FLOOR,
        "collisions_near_historical":
            comparison[
                "collision_delta_vs_historical"
            ]
            <= MAX_COLLISION_DELTA_VS_HISTORICAL,
        "material_recovery_vs_v0204":
            comparison[
                "completion_delta_vs_v0204"
            ]
            >= MIN_IMPROVEMENT_VS_V0204,
    }
    dev_repair_passed = all(
        checks.values()
    )

    result = {
        "format":
            "continuous_brain.same_option_bootstrap_planning_v1",
        "software_status": "PASS",
        "development_repair_passed":
            bool(dev_repair_passed),
        "scientific_admission_decided":
            False,
        "training_performed": False,
        "device": "cpu",
        "seed_window": {
            "start": SEED_START,
            "end":
                SEED_START + EPISODES - 1,
            "reused_spent_development":
                True,
            "new_development_window_consumed":
                False,
        },
        "repair_contract": {
            "chunk_steps": CHUNK_STEPS,
            "chunk_boundary_allows_option_switch":
                False,
            "unterminated_chunk_bootstrap":
                "same_option_policy_action_value",
            "terminated_chunk_tail":
                "selector_supported_next_option_policy_value",
        },
        "thresholds": {
            "historical_completion_floor":
                HISTORICAL_COMPLETION_FLOOR,
            "max_collision_delta_vs_historical":
                MAX_COLLISION_DELTA_VS_HISTORICAL,
            "min_completion_improvement_vs_v0204":
                MIN_IMPROVEMENT_VS_V0204,
            "v0204_completion":
                V0204_COMPLETION,
        },
        "checks": checks,
        "baseline": {
            "direct": direct,
            "historical_hierarchical":
                historical,
        },
        "same_option_bootstrap": repaired,
        "comparison": comparison,
        "episodes": rows,
        "protected_unchanged": True,
        "v019_remains_closed": True,
        "memory_preserved": True,
        "v018_intent_preserved": True,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nBASELINE")
    print(
        "direct completion/collisions:",
        direct["completion_rate"],
        direct["mean_collisions"],
    )
    print(
        "historical hierarchical "
        "completion/collisions:",
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

    for key, value in comparison.items():
        print(f"{key}: {value}")

    print()
    for key, value in checks.items():
        print(
            f"{key}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("V019_REMAINS_CLOSED: true")
    print(
        "SCIENTIFIC_ADMISSION_DECIDED: false"
    )
    print(
        "SAME_OPTION_BOOTSTRAP_DEV_REPAIR:",
        "PASS"
        if dev_repair_passed
        else "FAIL",
    )
    print(
        "SAME_OPTION_BOOTSTRAP_PLANNING_COMPLETE"
    )
    print("result:", output)

    raise SystemExit(
        0 if dev_repair_passed else 4
    )


if __name__ == "__main__":
    main()
