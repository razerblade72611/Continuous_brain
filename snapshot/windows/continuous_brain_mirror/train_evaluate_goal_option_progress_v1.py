#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.additive_goal_options_v1 import (
    GOAL_DECODER_PATH,
    load_goal_options,
    sha256,
)
from continuous_brain.goal_option_progress_v1 import (
    GOAL_OPTIONS_PATH,
    TemporalProgressHead,
    load_progress_head,
    save_progress_head,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    assert_pair_identical,
    build_substrate,
    collect_decision_snapshot,
    load_goal_decoder,
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

TRAIN_IDS = range(0, 1024)
FRESH_DEV_IDS = range(1856, 1980)

MAX_STEPS = 160
DIRECT_MIN_COMMIT = 3
DIRECT_TERMINATE_THRESHOLD = .50
PROGRESS_MIN_COMMIT = 1
PROGRESS_STOP_THRESHOLD_STEPS = .50

EPOCHS = 120
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 22101

# Frozen fresh-DEV gate.
MIN_GOAL_ACC = .95
MIN_OWN_COMMIT = .95
MIN_PAIR_COMMIT = .90
MIN_DONOR_COMMIT = .95
MIN_FLIP_COMMIT = .95
MIN_TERM_RATE = .95
MIN_TERM_WITHIN2 = .85
MAX_DURATION_MAE = 1.50
MIN_ADVANTAGE = .25
MAX_BASELINE = .70


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def root_from_env(base, cfg, env, obs):
    image = _image_tensor(env, cfg.image_size, "cpu")
    _, context_dyn = _context_tensor(obs, "cpu", cfg)
    spatial = base.target_spatial_bus(image)
    return base.planning.encode_state(spatial, context_dyn)


@torch.no_grad()
def collect_progress_dataset(base, cfg, bank):
    features = []
    remaining = []
    episodes = 0
    max_remaining = 0

    for n, cid in enumerate(TRAIN_IDS, start=1):
        for cue in ("left", "right"):
            env = VariedCuedDetourWorld(
                cid,
                cue=cue,
                max_episode_steps=MAX_STEPS,
                image_size_hint=int(cfg.image_size),
            )
            obs = env.reset()
            goal = 0 if cue == "left" else 1

            lateral_roots = []
            stop_root = None

            while not env.done:
                root = root_from_env(base, cfg, env, obs)
                teacher = int(env.teacher_action())

                if teacher in (LEFT, RIGHT):
                    lateral_roots.append(root.detach().cpu())
                    obs, _, _, _ = env.step(teacher)
                elif lateral_roots:
                    stop_root = root.detach().cpu()
                    break
                else:
                    obs, _, done, _ = env.step(teacher)
                    if done:
                        raise RuntimeError(
                            "TRAIN episode ended before lateral skill"
                        )

            if not lateral_roots:
                raise AssertionError("no lateral roots collected")
            if stop_root is None:
                # If the lateral traverse reaches terminal directly there is
                # no nonterminal stop state; keep the dense positive-duration
                # examples but no zero target for this episode.
                pass

            g = torch.tensor([goal], dtype=torch.long)
            total = len(lateral_roots)
            max_remaining = max(max_remaining, total)

            for i, root in enumerate(lateral_roots):
                feat = bank.features(root, g).detach().cpu()
                features.append(feat)
                remaining.append(float(total - i))

            if stop_root is not None:
                feat = bank.features(stop_root, g).detach().cpu()
                features.append(feat)
                remaining.append(0.0)

            episodes += 1

        if n == 1 or n % 128 == 0:
            print(
                f"progress_dataset_conditions="
                f"{n}/{len(TRAIN_IDS)}"
            )

    x = torch.cat(features, dim=0).float()
    y_steps = torch.tensor(remaining, dtype=torch.float32)
    y_log = torch.log1p(y_steps)

    return {
        "features": x,
        "remaining_steps": y_steps,
        "target_log1p": y_log,
        "episodes": episodes,
        "samples": int(x.shape[0]),
        "zero_targets": int((y_steps == 0).sum().item()),
        "max_remaining": int(max_remaining),
    }


def train_progress(dataset):
    torch.manual_seed(SEED)
    random.seed(SEED)

    x = dataset["features"]
    y = dataset["target_log1p"]
    y_steps = dataset["remaining_steps"]

    head = TemporalProgressHead(int(x.shape[1]))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(SEED)
    n = int(x.shape[0])
    metrics = {}

    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        total_loss = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx]
            yb = y[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = head(xb)
            loss = F.smooth_l1_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)

        head.eval()
        with torch.no_grad():
            pred_log = head(x).clamp_min(0.0)
            pred_steps = torch.expm1(pred_log)
            mae = float(
                (pred_steps - y_steps).abs().mean().item()
            )
            within2 = float(
                (
                    (pred_steps - y_steps).abs() <= 2.0
                ).float().mean().item()
            )
            zero_mask = y_steps == 0
            zero_stop = float(
                (
                    pred_steps[zero_mask]
                    <= PROGRESS_STOP_THRESHOLD_STEPS
                ).float().mean().item()
            ) if zero_mask.any() else 0.0

        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(1, seen),
            "train_duration_mae": mae,
            "train_within_2": within2,
            "train_zero_stop_rate": zero_stop,
        }

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"progress_epoch={epoch:03d}/{EPOCHS} "
                f"loss={metrics['loss']:.6f} "
                f"mae={mae:.4f} "
                f"within2={within2:.4f} "
                f"zero_stop={zero_stop:.4f}"
            )

    return head, metrics


def decode_goal(decoder, recall):
    return int(torch.argmax(decoder(recall), dim=-1)[0].item())


@torch.no_grad()
def teacher_lateral_steps(cfg, cid, cue, prefix):
    env = VariedCuedDetourWorld(
        cid,
        cue=cue,
        max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()
    for action in prefix:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("prefix ended episode")

    expected = LEFT if cue == "left" else RIGHT
    count = 0
    while (
        not env.done
        and int(env.teacher_action()) in (LEFT, RIGHT)
    ):
        action = int(env.teacher_action())
        if action != expected:
            raise AssertionError("teacher lateral direction mismatch")
        obs, _, _, _ = env.step(action)
        count += 1
    return count


@torch.no_grad()
def run_goal_skill(
    base,
    cfg,
    bank,
    progress,
    *,
    cid,
    actual_cue,
    prefix,
    goal_side,
):
    env = VariedCuedDetourWorld(
        cid,
        cue=actual_cue,
        max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()
    for action in prefix:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("prefix ended episode")

    actual_target = LEFT if actual_cue == "left" else RIGHT
    goal_target = LEFT if goal_side == 0 else RIGHT

    skill_active = True
    skill_age = 0
    skill_steps = 0
    skill_terminated = False
    predicted_remaining_at_start = None
    predicted_remaining_at_stop = None

    current_option = None
    option_age = 0
    first_lateral = None
    final_info = {"reached_correct": False}

    while not env.done:
        root = root_from_env(base, cfg, env, obs)

        if skill_active:
            goal_t = torch.tensor(
                [int(goal_side)], dtype=torch.long
            )
            feat = bank.features(root, goal_t)
            pred_log = progress(feat).clamp_min(0.0)
            pred_remaining = float(
                torch.expm1(pred_log)[0].item()
            )
            if predicted_remaining_at_start is None:
                predicted_remaining_at_start = pred_remaining

            if (
                skill_age >= PROGRESS_MIN_COMMIT
                and pred_remaining
                    <= PROGRESS_STOP_THRESHOLD_STEPS
            ):
                skill_active = False
                skill_terminated = True
                predicted_remaining_at_stop = pred_remaining
                current_option = int(
                    torch.argmax(
                        base.options.selector_logits(root)[0]
                    ).item()
                )
                option_age = 0
                continue

            action = int(
                torch.argmax(
                    bank.action_logits(root, goal_t),
                    dim=-1,
                )[0].item()
            )
            skill_age += 1
            skill_steps += 1
        else:
            if _active_option_should_terminate(
                base,
                root,
                current_option,
                option_age,
                DIRECT_MIN_COMMIT,
                DIRECT_TERMINATE_THRESHOLD,
            ):
                current_option = int(
                    torch.argmax(
                        base.options.selector_logits(root)[0]
                    ).item()
                )
                option_age = 0

            action = int(
                _option_policy_action(
                    base, root, current_option
                )
            )
            option_age += 1

        if (
            first_lateral is None
            and action in (LEFT, RIGHT)
        ):
            first_lateral = action

        obs, _, _, final_info = env.step(action)

    return {
        "first_lateral": first_lateral,
        "commitment_follows_goal":
            bool(first_lateral == goal_target),
        "commitment_correct":
            bool(first_lateral == actual_target),
        "completed_correct":
            bool(final_info.get("reached_correct", False)),
        "skill_steps": int(skill_steps),
        "skill_terminated": bool(skill_terminated),
        "predicted_remaining_at_start":
            predicted_remaining_at_start,
        "predicted_remaining_at_stop":
            predicted_remaining_at_stop,
    }


@torch.no_grad()
def run_baseline(base, cfg, cid, cue, prefix):
    env = VariedCuedDetourWorld(
        cid,
        cue=cue,
        max_episode_steps=MAX_STEPS,
        image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()
    for action in prefix:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("prefix ended episode")

    current = None
    age = 0
    first_lateral = None
    final = {"reached_correct": False}

    while not env.done:
        root = root_from_env(base, cfg, env, obs)
        if _active_option_should_terminate(
            base,
            root,
            current,
            age,
            DIRECT_MIN_COMMIT,
            DIRECT_TERMINATE_THRESHOLD,
        ):
            current = int(
                torch.argmax(
                    base.options.selector_logits(root)[0]
                ).item()
            )
            age = 0

        action = int(
            _option_policy_action(base, root, current)
        )
        age += 1
        if (
            first_lateral is None
            and action in (LEFT, RIGHT)
        ):
            first_lateral = action
        obs, _, _, final = env.step(action)

    target = LEFT if cue == "left" else RIGHT
    return {
        "commitment_correct":
            bool(first_lateral == target),
        "completed_correct":
            bool(final.get("reached_correct", False)),
    }


@torch.no_grad()
def evaluate_fresh_dev(
    base,
    cfg,
    admitted,
    decoder,
    bank,
    progress,
):
    own = []
    donor = []
    flip = []
    baseline = []
    pair = []
    goals = []
    terminated = []
    duration_error = []
    own_complete = []
    base_complete = []
    details = []

    for n, cid in enumerate(FRESH_DEV_IDS, start=1):
        left = collect_decision_snapshot(
            base, cfg, admitted, cid, "left"
        )
        right = collect_decision_snapshot(
            base, cfg, admitted, cid, "right"
        )
        assert_pair_identical(left, right)
        members = {"left": left, "right": right}
        pair_ok = []

        for cue in ("left", "right"):
            snap = members[cue]
            donor_snap = members[
                "right" if cue == "left" else "left"
            ]

            own_goal = decode_goal(
                decoder, snap["recall"]
            )
            donor_goal = decode_goal(
                decoder, donor_snap["recall"]
            )
            expected_goal = 0 if cue == "left" else 1
            goals.append(float(own_goal == expected_goal))

            teacher_steps = teacher_lateral_steps(
                cfg,
                cid,
                cue,
                snap["prefix_actions"],
            )

            r_own = run_goal_skill(
                base,
                cfg,
                bank,
                progress,
                cid=cid,
                actual_cue=cue,
                prefix=snap["prefix_actions"],
                goal_side=own_goal,
            )
            r_donor = run_goal_skill(
                base,
                cfg,
                bank,
                progress,
                cid=cid,
                actual_cue=cue,
                prefix=snap["prefix_actions"],
                goal_side=donor_goal,
            )
            r_flip = run_goal_skill(
                base,
                cfg,
                bank,
                progress,
                cid=cid,
                actual_cue=cue,
                prefix=snap["prefix_actions"],
                goal_side=1 - own_goal,
            )
            r_base = run_baseline(
                base,
                cfg,
                cid,
                cue,
                snap["prefix_actions"],
            )

            own.append(float(r_own["commitment_correct"]))
            donor.append(
                float(r_donor["commitment_follows_goal"])
            )
            flip.append(
                float(r_flip["commitment_follows_goal"])
            )
            baseline.append(
                float(r_base["commitment_correct"])
            )
            pair_ok.append(
                bool(r_own["commitment_correct"])
            )
            terminated.append(
                float(r_own["skill_terminated"])
            )
            duration_error.append(
                abs(
                    float(r_own["skill_steps"])
                    - float(teacher_steps)
                )
            )
            own_complete.append(
                float(r_own["completed_correct"])
            )
            base_complete.append(
                float(r_base["completed_correct"])
            )

            details.append(
                {
                    "condition_id": cid,
                    "cue": cue,
                    "own_goal": own_goal,
                    "donor_goal": donor_goal,
                    "teacher_lateral_steps":
                        teacher_steps,
                    "own": r_own,
                    "donor": r_donor,
                    "flip": r_flip,
                    "baseline": r_base,
                }
            )

        pair.append(float(all(pair_ok)))

        if n == 1 or n % 16 == 0:
            print(
                f"progress_fresh_dev_conditions="
                f"{n}/{len(FRESH_DEV_IDS)}"
            )

    own_rate = mean(own)
    base_rate = mean(baseline)

    summary = {
        "goal_accuracy": mean(goals),
        "own_commitment_correct": own_rate,
        "paired_both_sides_commitment": mean(pair),
        "donor_commitment_follows_donor_goal":
            mean(donor),
        "explicit_goal_flip_follows_flipped_goal":
            mean(flip),
        "baseline_commitment_correct": base_rate,
        "commitment_advantage":
            own_rate - base_rate,
        "skill_termination_rate":
            mean(terminated),
        "duration_mae_steps":
            mean(duration_error),
        "termination_within_2_teacher_steps":
            mean(float(e <= 2.0) for e in duration_error),
        "own_completion_correct":
            mean(own_complete),
        "baseline_completion_correct":
            mean(base_complete),
        "completion_advantage":
            mean(own_complete) - mean(base_complete),
    }
    return summary, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/goal_option_progress_v1",
    )
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / "result.json"
    progress_path = state / "progress_head.pt"

    if not GOAL_OPTIONS_PATH.is_file():
        raise FileNotFoundError(GOAL_OPTIONS_PATH)
    if not GOAL_DECODER_PATH.is_file():
        raise FileNotFoundError(GOAL_DECODER_PATH)

    base_sha_before = sha256(BASE_CHECKPOINT)
    memory_sha_before = sha256(MEMORY_CHECKPOINT)
    goal_options_sha = sha256(GOAL_OPTIONS_PATH)
    goal_decoder_sha = sha256(GOAL_DECODER_PATH)

    print("GOAL_OPTION_TEMPORAL_PROGRESS_V1")
    print("device: cpu")
    print("goal_option_policy_training_performed: false")
    print("historical_option_training_performed: false")
    print("memory_training_performed: false")
    print("planner_training_performed: false")
    print("trainable_component: temporal_progress_head_only")
    print("train_conditions: 0-1023")
    print("fresh_dev_conditions: 1856-1979")
    print("test_touched: false")

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(
        GOAL_DECODER_PATH, "cpu"
    )
    bank, _ = load_goal_options(
        GOAL_OPTIONS_PATH,
        goal_decoder_sha,
        "cpu",
    )

    frozen_goal_options = {
        k: v.detach().cpu().clone()
        for k, v in bank.state_dict().items()
    }

    dataset = collect_progress_dataset(
        base, cfg, bank
    )
    dataset_meta = {
        "episodes": dataset["episodes"],
        "samples": dataset["samples"],
        "zero_targets": dataset["zero_targets"],
        "max_remaining": dataset["max_remaining"],
    }
    print("progress_dataset:", dataset_meta)

    progress, train_metrics = train_progress(
        dataset
    )
    save_progress_head(
        progress_path,
        progress,
        goal_options_sha256=goal_options_sha,
        goal_decoder_sha256=goal_decoder_sha,
        metadata={
            "train_start": 0,
            "train_end": 1023,
            "seed": SEED,
            "epochs": EPOCHS,
            "target": "log1p_remaining_lateral_steps",
        },
    )
    progress, _ = load_progress_head(
        progress_path,
        expected_goal_options_sha256=
            goal_options_sha,
        expected_goal_decoder_sha256=
            goal_decoder_sha,
        device="cpu",
    )

    dev, details = evaluate_fresh_dev(
        base,
        cfg,
        admitted,
        decoder,
        bank,
        progress,
    )

    for key, value in frozen_goal_options.items():
        got = bank.state_dict()[key].detach().cpu()
        if not torch.equal(value, got):
            raise RuntimeError(
                f"goal option tensor changed: {key}"
            )

    if sha256(BASE_CHECKPOINT) != base_sha_before:
        raise RuntimeError("base checkpoint changed")
    if sha256(MEMORY_CHECKPOINT) != memory_sha_before:
        raise RuntimeError("memory checkpoint changed")
    if sha256(GOAL_OPTIONS_PATH) != goal_options_sha:
        raise RuntimeError("goal option checkpoint changed")

    checks = {
        "fresh_dev_goal_accuracy":
            dev["goal_accuracy"] >= MIN_GOAL_ACC,
        "own_commitment":
            dev["own_commitment_correct"] >= MIN_OWN_COMMIT,
        "paired_both_sides_commitment":
            dev["paired_both_sides_commitment"]
            >= MIN_PAIR_COMMIT,
        "donor_commitment":
            dev["donor_commitment_follows_donor_goal"]
            >= MIN_DONOR_COMMIT,
        "goal_flip_commitment":
            dev[
                "explicit_goal_flip_follows_flipped_goal"
            ] >= MIN_FLIP_COMMIT,
        "skill_termination":
            dev["skill_termination_rate"] >= MIN_TERM_RATE,
        "termination_timing":
            dev["termination_within_2_teacher_steps"]
            >= MIN_TERM_WITHIN2,
        "duration_mae":
            dev["duration_mae_steps"] <= MAX_DURATION_MAE,
        "commitment_advantage":
            dev["commitment_advantage"]
            >= MIN_ADVANTAGE,
        "baseline_not_solving_memory_task":
            dev["baseline_commitment_correct"]
            <= MAX_BASELINE,
    }
    passed = all(checks.values())

    result = {
        "format":
            "continuous_brain.goal_option_temporal_progress_v1",
        "development_gate_passed": bool(passed),
        "training": {
            "temporal_progress_head": True,
            "goal_option_policy": False,
            "goal_option_bank": False,
            "historical_options": False,
            "memory": False,
            "base": False,
            "planner": False,
        },
        "split": {
            "train": [0, 1023],
            "fresh_dev": [1856, 1979],
            "test_touched": False,
        },
        "progress_dataset": dataset_meta,
        "train_metrics": train_metrics,
        "fresh_dev": dev,
        "checks": checks,
        "details": details,
        "protected_unchanged": True,
        "goal_option_bank_bit_exact": True,
        "historical_six_options_bit_exact": True,
        "v019_remains_closed": True,
        "detour_planner_repair_closed": True,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nFRESH_DEV")
    for k, v in dev.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("GOAL_OPTION_BANK_BIT_EXACT: true")
    print("HISTORICAL_SIX_OPTIONS_BIT_EXACT: true")
    print("TEST_TOUCHED: false")
    print("V019_REMAINS_CLOSED: true")
    print("DETOUR_PLANNER_REPAIR_CLOSED: true")
    print(
        "GOAL_OPTION_TEMPORAL_PROGRESS_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print("GOAL_OPTION_TEMPORAL_PROGRESS_RUN_COMPLETE")
    print("result:", result_path)
    print("progress_head:", progress_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
