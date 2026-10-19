#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    GoalDecoder,
    assert_pair_identical,
    build_substrate,
    choose_goal_option,
    collect_decision_snapshot,
    load_goal_decoder,
    save_goal_decoder,
    sha256,
)
from continuous_brain.varied_cued_detour_world import (
    VariedCuedDetourWorld,
    LEFT,
    RIGHT,
)
from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
)
from continuous_brain.planning_v103 import (
    _active_option_should_terminate,
    _option_policy_action,
)

TRAIN_IDS = range(0, 512)
DEV_IDS = range(1600, 1728)

MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50
MAX_STEPS = 160

TRAIN_EPOCHS = 600
LR = .02
WEIGHT_DECAY = 1e-4
SEED = 21001

# Frozen development gate.
MIN_GOAL_ACC = .95
MIN_SIDE_ACC = .95
MIN_OPTION_COVERAGE = .90
MIN_OWN_SELECTED_ACTION = .90
MIN_DONOR_GOAL = .90
MIN_DONOR_SELECTED_ACTION = .85
MIN_GOAL_FLIP_SELECTION = .80
MIN_AUTONOMOUS_COMMITMENT = .80
MAX_BASELINE_COMMITMENT = .70


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def goal_side_from_logits(logits, *, zero_tol=1e-10):
    if float(logits.abs().max().item()) <= float(zero_tol):
        return None
    return int(torch.argmax(logits, dim=-1)[0].item())


def selector_top1(base, root):
    root = root.to(next(base.parameters()).device)
    logits = base.options.selector_logits(root)[0]
    return int(torch.argmax(logits).item())


@torch.no_grad()
def run_postdecision(
    base,
    cfg,
    condition_id,
    cue,
    prefix_actions,
    initial_option,
    device="cpu",
):
    env = VariedCuedDetourWorld(
        int(condition_id),
        cue=str(cue),
        max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()

    for action in prefix_actions:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("prefix replay ended episode")

    # This is the delayed decision boundary.
    if int(env.teacher_action()) not in (LEFT, RIGHT):
        raise AssertionError("prefix replay did not reach delayed decision")
    if env.steps < env.cue_steps:
        raise AssertionError("cue still visible at decision")

    current_option = int(initial_option)
    option_age = 0
    first_lateral_action = None
    first_option_steps = 0
    initial_option_active = True
    option_switches = 0
    total_reward = 0.0
    final_info = {"reached_correct": False}

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        _, context_dyn = _context_tensor(obs, device, cfg)
        stable_spatial = base.target_spatial_bus(image)
        root = base.planning.encode_state(stable_spatial, context_dyn)

        if _active_option_should_terminate(
            base,
            root,
            current_option,
            option_age,
            MIN_COMMIT,
            TERMINATE_THRESHOLD,
        ):
            if option_age > 0:
                initial_option_active = False
            new_option = selector_top1(base, root)
            option_switches += int(new_option != current_option)
            current_option = new_option
            option_age = 0

        action = _option_policy_action(base, root, current_option)
        if initial_option_active:
            first_option_steps += 1

        if first_lateral_action is None and action in (LEFT, RIGHT):
            first_lateral_action = int(action)

        obs, reward, _, final_info = env.step(action)
        total_reward += float(reward)
        option_age += 1

    target_action = LEFT if cue == "left" else RIGHT
    return {
        "condition_id": int(condition_id),
        "cue": str(cue),
        "initial_option": int(initial_option),
        "first_lateral_action": first_lateral_action,
        "commitment_correct": bool(first_lateral_action == target_action),
        "completed_correct": bool(final_info.get("reached_correct", False)),
        "collisions": int(getattr(env, "collisions", 0)),
        "reward": float(total_reward),
        "postdecision_steps": int(env.steps - len(prefix_actions)),
        "initial_option_steps": int(first_option_steps),
        "option_switches": int(option_switches),
    }


def collect_features(base, cfg, admitted, ids, device="cpu", verbose_every=64):
    rows = []
    paired = []
    for j, cid in enumerate(ids, start=1):
        left = collect_decision_snapshot(
            base, cfg, admitted, cid, "left", device=device
        )
        right = collect_decision_snapshot(
            base, cfg, admitted, cid, "right", device=device
        )
        assert_pair_identical(left, right)
        rows.extend((left, right))
        paired.append((left, right))
        if verbose_every and (j == 1 or j % verbose_every == 0):
            print(f"feature_conditions={j}/{len(ids)}")
    return rows, paired


def tensor_dataset(rows):
    x = torch.cat([r["recall"] for r in rows], dim=0).float()
    y = torch.tensor(
        [int(r["target_side"]) for r in rows],
        dtype=torch.long,
    )
    return x, y


def accuracy(decoder, rows):
    x, y = tensor_dataset(rows)
    with torch.no_grad():
        pred = decoder(x).argmax(dim=-1)
    overall = float((pred == y).float().mean().item())
    left_mask = y == 0
    right_mask = y == 1
    left = float((pred[left_mask] == y[left_mask]).float().mean().item())
    right = float((pred[right_mask] == y[right_mask]).float().mean().item())
    pair_ok = []
    for i in range(0, len(rows), 2):
        pair_ok.append(
            int(pred[i].item() == y[i].item())
            and int(pred[i + 1].item() == y[i + 1].item())
        )
    return {
        "overall": overall,
        "left": left,
        "right": right,
        "pair": mean(float(v) for v in pair_ok),
    }


def train_goal_decoder(train_rows):
    torch.manual_seed(SEED)
    decoder = GoalDecoder()
    opt = torch.optim.Adam(
        decoder.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    x, y = tensor_dataset(train_rows)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        logits = decoder(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 50 == 0:
            acc = float(
                (logits.argmax(dim=-1) == y).float().mean().item()
            )
            print(
                f"goal_decoder_epoch={epoch:04d}/{TRAIN_EPOCHS} "
                f"loss={float(loss.item()):.6f} acc={acc:.4f}"
            )
        if epoch >= 50:
            with torch.no_grad():
                acc = float(
                    (decoder(x).argmax(dim=-1) == y)
                    .float().mean().item()
                )
            if acc >= .999:
                print(f"goal_decoder_early_stop_epoch={epoch}")
                break
    decoder.eval()
    return decoder


@torch.no_grad()
def evaluate_integration(base, cfg, decoder, dev_pairs, device="cpu"):
    rows = []
    own_runs = []
    baseline_runs = []

    own_goal_correct = []
    own_selected_action_correct = []
    option_coverage = []
    donor_goal_correct = []
    donor_selected_action_correct = []
    goal_flip_selection_changed = []
    zero_exact_baseline = []
    goal_intervened = []

    for idx, (left, right) in enumerate(dev_pairs, start=1):
        pair = {"left": left, "right": right}

        for cue in ("left", "right"):
            own = pair[cue]
            donor = pair["right" if cue == "left" else "left"]
            root = own["root"]

            own_goal_logits = decoder(own["recall"])
            own_goal = goal_side_from_logits(own_goal_logits)
            expected = 0 if cue == "left" else 1

            own_choice = choose_goal_option(base, root, own_goal)
            own_goal_correct.append(float(own_goal == expected))
            own_selected_action_correct.append(
                float(own_choice["selected"]["action"] == own["target_action"])
            )
            option_coverage.append(
                float(own_choice["goal_candidate_count"] > 0)
            )
            goal_intervened.append(float(own_choice["goal_intervened"]))

            donor_goal = goal_side_from_logits(decoder(donor["recall"]))
            donor_expected = 1 - expected
            donor_choice = choose_goal_option(base, root, donor_goal)
            donor_goal_correct.append(float(donor_goal == donor_expected))
            donor_action = RIGHT if donor_expected == 1 else LEFT
            donor_selected_action_correct.append(
                float(donor_choice["selected"]["action"] == donor_action)
            )

            flipped_goal = None if own_goal is None else 1 - own_goal
            flipped_choice = choose_goal_option(base, root, flipped_goal)
            goal_flip_selection_changed.append(
                float(
                    flipped_choice["selected"]["option"]
                    != own_choice["selected"]["option"]
                )
            )

            zero = torch.zeros_like(own["recall"])
            zero_goal = goal_side_from_logits(decoder(zero))
            zero_choice = choose_goal_option(base, root, zero_goal)
            zero_exact_baseline.append(
                float(
                    zero_goal is None
                    and zero_choice["selected"]["option"]
                    == zero_choice["baseline"]["option"]
                )
            )

            own_run = run_postdecision(
                base,
                cfg,
                own["condition_id"],
                cue,
                own["prefix_actions"],
                own_choice["selected"]["option"],
                device=device,
            )
            baseline_run = run_postdecision(
                base,
                cfg,
                own["condition_id"],
                cue,
                own["prefix_actions"],
                own_choice["baseline"]["option"],
                device=device,
            )
            own_runs.append(own_run)
            baseline_runs.append(baseline_run)

            rows.append(
                {
                    "condition_id": own["condition_id"],
                    "cue": cue,
                    "own_goal_logits": [
                        float(x) for x in own_goal_logits[0]
                    ],
                    "own_goal": own_goal,
                    "own_choice": own_choice,
                    "donor_goal": donor_goal,
                    "donor_choice": donor_choice,
                    "flipped_choice": flipped_choice,
                    "zero_goal": zero_goal,
                    "zero_choice": zero_choice,
                    "own_run": own_run,
                    "baseline_run": baseline_run,
                }
            )

        if idx == 1 or idx % 16 == 0:
            print(f"integration_dev_conditions={idx}/{len(dev_pairs)}")

    own_commit = mean(float(r["commitment_correct"]) for r in own_runs)
    base_commit = mean(float(r["commitment_correct"]) for r in baseline_runs)
    own_complete = mean(float(r["completed_correct"]) for r in own_runs)
    base_complete = mean(float(r["completed_correct"]) for r in baseline_runs)

    summary = {
        "own_goal_accuracy": mean(own_goal_correct),
        "own_selected_action_correct": mean(own_selected_action_correct),
        "option_goal_coverage": mean(option_coverage),
        "donor_goal_accuracy": mean(donor_goal_correct),
        "donor_selected_action_correct": mean(donor_selected_action_correct),
        "goal_flip_selection_changed": mean(goal_flip_selection_changed),
        "zero_memory_exact_baseline": mean(zero_exact_baseline),
        "goal_intervention_rate": mean(goal_intervened),
        "autonomous_commitment_correct": own_commit,
        "baseline_commitment_correct": base_commit,
        "autonomous_completion_correct": own_complete,
        "baseline_completion_correct": base_complete,
        "commitment_advantage": own_commit - base_commit,
        "completion_advantage": own_complete - base_complete,
        "mean_own_collisions": mean(r["collisions"] for r in own_runs),
        "mean_baseline_collisions": mean(
            r["collisions"] for r in baseline_runs
        ),
        "mean_initial_option_steps": mean(
            r["initial_option_steps"] for r in own_runs
        ),
    }

    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/memory_guided_goal_option_v1",
    )
    args = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"
    decoder_path = state_dir / "goal_decoder.pt"

    base_before = sha256(BASE_CHECKPOINT)
    memory_before = sha256(MEMORY_CHECKPOINT)

    print("MEMORY_GUIDED_GOAL_OPTION_V1")
    print("device: cpu")
    print("detour_planner_repair_closed: true")
    print("execution_backbone: learned_direct_options")
    print("trainable_component: goal_decoder_128_to_2_only")
    print("primitive_action_head_trained: false")
    print("option_policy_trained: false")
    print("memory_trained: false")
    print("base_trained: false")
    print(f"train_conditions: {TRAIN_IDS.start}-{TRAIN_IDS.stop-1}")
    print(f"dev_conditions: {DEV_IDS.start}-{DEV_IDS.stop-1}")

    base, cfg, admitted = build_substrate("cpu")

    train_rows, _ = collect_features(
        base, cfg, admitted, TRAIN_IDS, device="cpu"
    )
    decoder = train_goal_decoder(train_rows)
    train_acc = accuracy(decoder, train_rows)

    save_goal_decoder(
        decoder_path,
        decoder,
        {
            "train_condition_start": TRAIN_IDS.start,
            "train_condition_end": TRAIN_IDS.stop - 1,
            "seed": SEED,
        },
    )
    decoder, _ = load_goal_decoder(decoder_path, "cpu")

    dev_rows, dev_pairs = collect_features(
        base, cfg, admitted, DEV_IDS, device="cpu", verbose_every=32
    )
    dev_acc = accuracy(decoder, dev_rows)
    integration, detail = evaluate_integration(
        base, cfg, decoder, dev_pairs, device="cpu"
    )

    checks = {
        "dev_goal_pair_accuracy":
            dev_acc["pair"] >= MIN_GOAL_ACC,
        "dev_left_accuracy":
            dev_acc["left"] >= MIN_SIDE_ACC,
        "dev_right_accuracy":
            dev_acc["right"] >= MIN_SIDE_ACC,
        "option_goal_coverage":
            integration["option_goal_coverage"] >= MIN_OPTION_COVERAGE,
        "own_selected_action":
            integration["own_selected_action_correct"]
            >= MIN_OWN_SELECTED_ACTION,
        "donor_goal":
            integration["donor_goal_accuracy"] >= MIN_DONOR_GOAL,
        "donor_selected_action":
            integration["donor_selected_action_correct"]
            >= MIN_DONOR_SELECTED_ACTION,
        "goal_flip_changes_selection":
            integration["goal_flip_selection_changed"]
            >= MIN_GOAL_FLIP_SELECTION,
        "zero_memory_baseline":
            integration["zero_memory_exact_baseline"] == 1.0,
        "autonomous_commitment":
            integration["autonomous_commitment_correct"]
            >= MIN_AUTONOMOUS_COMMITMENT,
        "baseline_not_solving_memory_task":
            integration["baseline_commitment_correct"]
            <= MAX_BASELINE_COMMITMENT,
    }
    passed = all(checks.values())

    if sha256(BASE_CHECKPOINT) != base_before:
        raise RuntimeError("base checkpoint changed")
    if sha256(MEMORY_CHECKPOINT) != memory_before:
        raise RuntimeError("memory checkpoint changed")

    result = {
        "format": "continuous_brain.memory_guided_goal_option_v1",
        "development_gate_passed": bool(passed),
        "training": {
            "goal_decoder_only": True,
            "base": False,
            "memory": False,
            "options": False,
            "planner": False,
        },
        "split": {
            "train": [TRAIN_IDS.start, TRAIN_IDS.stop - 1],
            "dev": [DEV_IDS.start, DEV_IDS.stop - 1],
            "test_touched": False,
        },
        "train_goal_accuracy": train_acc,
        "dev_goal_accuracy": dev_acc,
        "integration": integration,
        "checks": checks,
        "details": detail,
        "protected_unchanged": True,
        "detour_planner_repair_closed": True,
        "v019_remains_closed": True,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nGOAL_DECODER")
    print("train:", train_acc)
    print("dev:", dev_acc)

    print("\nMEMORY_TO_GOAL_TO_OPTION")
    for key, value in integration.items():
        print(f"{key}: {value}")

    print("\nGATE")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("TEST_TOUCHED: false")
    print("V019_REMAINS_CLOSED: true")
    print("DETOUR_PLANNER_REPAIR_CLOSED: true")
    print(
        "MEMORY_GUIDED_GOAL_OPTION_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print("MEMORY_GUIDED_GOAL_OPTION_RUN_COMPLETE")
    print("result:", result_path)
    print("goal_decoder:", decoder_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
