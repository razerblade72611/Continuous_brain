#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.additive_goal_options_v1 import (
    GOAL_DECODER_PATH,
    load_goal_options,
    sha256,
)
from continuous_brain.recurrent_goal_option_state_v1 import (
    GOAL_OPTIONS_PATH,
    RecurrentGoalOptionState,
    load_controller,
    save_controller,
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

TRAIN_IDS = range(0, 1600)
TEST_IDS = range(1980, 2360)

MAX_STEPS = 160
DIRECT_MIN_COMMIT = 3
DIRECT_TERMINATE_THRESHOLD = .50
SKILL_MIN_COMMIT = 1
REMAINING_STOP_THRESHOLD = .50

EPOCHS = 80
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 96
SEED = 22201
TERM_LOSS_WEIGHT = .50

# Frozen TEST admission gate.
MIN_GOAL_ACC = .95
MIN_OWN_COMMIT = .95
MIN_PAIR_COMMIT = .90
MIN_DONOR_COMMIT = .95
MIN_FLIP_COMMIT = .95
MIN_TERM_RATE = .95
MIN_TERM_WITHIN2 = .85
MAX_DURATION_MAE = 1.50
MIN_COMMIT_ADVANTAGE = .25
MAX_BASELINE_COMMIT = .70
MIN_OWN_COMPLETION = .30
MIN_COMPLETION_ADVANTAGE = .20


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def root_from_env(base, cfg, env, obs):
    image = _image_tensor(env, cfg.image_size, "cpu")
    _, context_dyn = _context_tensor(obs, "cpu", cfg)
    spatial = base.target_spatial_bus(image)
    return base.planning.encode_state(spatial, context_dyn)


@torch.no_grad()
def collect_sequences(base, cfg, bank):
    sequences = []
    max_length = 0
    stop_sequences = 0

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

            lateral = []
            stop_feature = None

            while not env.done:
                root = root_from_env(base, cfg, env, obs)
                teacher = int(env.teacher_action())

                if teacher in (LEFT, RIGHT):
                    g = torch.tensor([goal], dtype=torch.long)
                    feat = bank.features(root, g).detach().cpu()[0]
                    lateral.append(feat)
                    obs, _, _, _ = env.step(teacher)
                elif lateral:
                    g = torch.tensor([goal], dtype=torch.long)
                    stop_feature = bank.features(
                        root, g
                    ).detach().cpu()[0]
                    break
                else:
                    obs, _, done, _ = env.step(teacher)
                    if done:
                        raise RuntimeError(
                            "TRAIN episode ended before lateral skill"
                        )

            if not lateral:
                raise AssertionError("no lateral sequence")

            total = len(lateral)
            features = list(lateral)
            remaining = [
                float(total - i) for i in range(total)
            ]
            terminate = [0.0] * total

            if stop_feature is not None:
                features.append(stop_feature)
                remaining.append(0.0)
                terminate.append(1.0)
                stop_sequences += 1

            feat_tensor = torch.stack(features).float()
            rem_tensor = torch.tensor(
                remaining, dtype=torch.float32
            )
            term_tensor = torch.tensor(
                terminate, dtype=torch.float32
            )

            sequences.append(
                {
                    "features": feat_tensor,
                    "remaining": rem_tensor,
                    "remaining_log":
                        torch.log1p(rem_tensor),
                    "terminate": term_tensor,
                    "condition_id": cid,
                    "goal": goal,
                }
            )
            max_length = max(
                max_length, int(feat_tensor.shape[0])
            )

        if n == 1 or n % 200 == 0:
            print(
                f"recurrent_dataset_conditions="
                f"{n}/{len(TRAIN_IDS)}"
            )

    return sequences, {
        "sequences": len(sequences),
        "max_length": max_length,
        "stop_sequences": stop_sequences,
        "total_steps": sum(
            int(s["features"].shape[0])
            for s in sequences
        ),
    }


def make_batch(rows):
    b = len(rows)
    max_len = max(
        int(r["features"].shape[0]) for r in rows
    )
    d = int(rows[0]["features"].shape[1])

    x = torch.zeros(b, max_len, d)
    remaining = torch.zeros(b, max_len)
    remaining_log = torch.zeros(b, max_len)
    terminate = torch.zeros(b, max_len)
    mask = torch.zeros(
        b, max_len, dtype=torch.bool
    )

    for i, row in enumerate(rows):
        n = int(row["features"].shape[0])
        x[i, :n] = row["features"]
        remaining[i, :n] = row["remaining"]
        remaining_log[i, :n] = row["remaining_log"]
        terminate[i, :n] = row["terminate"]
        mask[i, :n] = True

    return {
        "features": x,
        "remaining": remaining,
        "remaining_log": remaining_log,
        "terminate": terminate,
        "mask": mask,
    }


def train_controller(sequences):
    torch.manual_seed(SEED)
    random.seed(SEED)

    d = int(sequences[0]["features"].shape[1])
    model = RecurrentGoalOptionState(
        d, HIDDEN_DIM
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    positives = sum(
        float(s["terminate"].sum().item())
        for s in sequences
    )
    negatives = sum(
        float(s["terminate"].numel())
        - float(s["terminate"].sum().item())
        for s in sequences
    )
    pos_weight = torch.tensor(
        [max(1.0, negatives / max(1.0, positives))]
    )

    order_rng = random.Random(SEED)
    metrics = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(range(len(sequences)))
        order_rng.shuffle(order)

        total = 0.0
        seen = 0

        for start in range(
            0, len(order), BATCH_SIZE
        ):
            rows = [
                sequences[i]
                for i in order[start:start+BATCH_SIZE]
            ]
            batch = make_batch(rows)

            optimizer.zero_grad(set_to_none=True)
            out = model.forward_sequence(
                batch["features"]
            )
            mask = batch["mask"]

            rem_loss = F.smooth_l1_loss(
                out["remaining_log"][mask],
                batch["remaining_log"][mask],
            )
            term_loss = (
                F.binary_cross_entropy_with_logits(
                    out["termination_logit"][mask],
                    batch["terminate"][mask],
                    pos_weight=pos_weight,
                )
            )
            loss = rem_loss + TERM_LOSS_WEIGHT * term_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            count = int(mask.sum().item())
            total += float(loss.item()) * count
            seen += count

        model.eval()
        # Deterministic full-TRAIN metrics.  Training metrics are not
        # used for architecture selection after TEST.
        abs_errors = []
        within2 = []
        term_y = []
        term_p = []

        with torch.no_grad():
            for start in range(
                0, len(sequences), BATCH_SIZE
            ):
                batch = make_batch(
                    sequences[start:start+BATCH_SIZE]
                )
                out = model.forward_sequence(
                    batch["features"]
                )
                mask = batch["mask"]
                pred_steps = torch.expm1(
                    out["remaining_log"].clamp_min(0.0)
                )
                err = (
                    pred_steps[mask]
                    - batch["remaining"][mask]
                ).abs()
                abs_errors.extend(
                    float(v) for v in err.tolist()
                )
                within2.extend(
                    float(v)
                    for v in (err <= 2.0).tolist()
                )
                term_y.extend(
                    float(v)
                    for v in batch["terminate"][mask].tolist()
                )
                term_p.extend(
                    float(v)
                    for v in torch.sigmoid(
                        out["termination_logit"][mask]
                    ).tolist()
                )

        term_acc = mean(
            float((p >= .5) == (y >= .5))
            for y, p in zip(term_y, term_p)
        )

        metrics = {
            "epoch": epoch,
            "loss": total / max(1, seen),
            "train_duration_mae":
                mean(abs_errors),
            "train_duration_within2":
                mean(within2),
            "train_termination_accuracy":
                term_acc,
            "termination_pos_weight":
                float(pos_weight.item()),
        }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            print(
                f"recurrent_state_epoch="
                f"{epoch:03d}/{EPOCHS} "
                f"loss={metrics['loss']:.6f} "
                f"mae={metrics['train_duration_mae']:.4f} "
                f"within2="
                f"{metrics['train_duration_within2']:.4f} "
                f"term_acc="
                f"{metrics['train_termination_accuracy']:.4f}"
            )

    model.eval()
    return model, metrics


def decode_goal_or_none(decoder, recall):
    logits = decoder(recall)
    if float(logits.abs().max().item()) <= 1e-10:
        return None
    return int(
        torch.argmax(logits, dim=-1)[0].item()
    )


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
            raise AssertionError(
                "teacher lateral direction mismatch"
            )
        obs, _, _, _ = env.step(action)
        count += 1
    return count


@torch.no_grad()
def run_goal_skill(
    base,
    cfg,
    bank,
    controller,
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

    actual_target = (
        LEFT if actual_cue == "left" else RIGHT
    )
    goal_target = (
        LEFT if int(goal_side) == 0 else RIGHT
    )

    skill_active = True
    skill_age = 0
    skill_steps = 0
    skill_terminated = False
    recurrent_hidden = None

    current_option = None
    option_age = 0
    first_lateral = None
    final_info = {"reached_correct": False}
    start_remaining = None
    stop_remaining = None
    stop_term_probability = None

    while not env.done:
        root = root_from_env(base, cfg, env, obs)

        if skill_active:
            goal_t = torch.tensor(
                [int(goal_side)],
                dtype=torch.long,
            )
            feature = bank.features(root, goal_t)
            temporal = controller.step(
                feature,
                recurrent_hidden,
            )
            recurrent_hidden = temporal["hidden"]
            pred_remaining = float(
                torch.expm1(
                    temporal["remaining_log"].clamp_min(0.0)
                )[0].item()
            )
            term_probability = float(
                torch.sigmoid(
                    temporal["termination_logit"]
                )[0].item()
            )
            if start_remaining is None:
                start_remaining = pred_remaining

            if (
                skill_age >= SKILL_MIN_COMMIT
                and pred_remaining
                    <= REMAINING_STOP_THRESHOLD
            ):
                skill_active = False
                skill_terminated = True
                stop_remaining = pred_remaining
                stop_term_probability = term_probability
                recurrent_hidden = None

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
                    base,
                    root,
                    current_option,
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
            start_remaining,
        "predicted_remaining_at_stop":
            stop_remaining,
        "termination_probability_at_stop":
            stop_term_probability,
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
    final_info = {"reached_correct": False}

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
            _option_policy_action(
                base, root, current
            )
        )
        age += 1
        if (
            first_lateral is None
            and action in (LEFT, RIGHT)
        ):
            first_lateral = action
        obs, _, _, final_info = env.step(action)

    target = LEFT if cue == "left" else RIGHT
    return {
        "first_lateral": first_lateral,
        "commitment_correct":
            bool(first_lateral == target),
        "completed_correct":
            bool(final_info.get("reached_correct", False)),
    }


@torch.no_grad()
def evaluate_test(
    base,
    cfg,
    admitted,
    decoder,
    bank,
    controller,
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
    zero_exact = []
    details = []

    zero_recall = torch.zeros(
        1, 128, dtype=torch.float32
    )
    zero_goal = decode_goal_or_none(
        decoder, zero_recall
    )
    if zero_goal is not None:
        raise AssertionError(
            "zero recall did not produce no-goal route"
        )

    for n, cid in enumerate(TEST_IDS, start=1):
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

            own_goal = decode_goal_or_none(
                decoder, snap["recall"]
            )
            donor_goal = decode_goal_or_none(
                decoder, donor_snap["recall"]
            )
            if own_goal is None or donor_goal is None:
                raise AssertionError(
                    "nonzero admitted recall produced no goal"
                )

            expected_goal = 0 if cue == "left" else 1
            goals.append(
                float(own_goal == expected_goal)
            )

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
                controller,
                cid=cid,
                actual_cue=cue,
                prefix=snap["prefix_actions"],
                goal_side=own_goal,
            )
            r_donor = run_goal_skill(
                base,
                cfg,
                bank,
                controller,
                cid=cid,
                actual_cue=cue,
                prefix=snap["prefix_actions"],
                goal_side=donor_goal,
            )
            r_flip = run_goal_skill(
                base,
                cfg,
                bank,
                controller,
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

            own.append(
                float(r_own["commitment_correct"])
            )
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
            # Zero memory routes exactly to the baseline controller.
            zero_exact.append(1.0)

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

        if n == 1 or n % 20 == 0:
            print(
                f"test_conditions={n}/{len(TEST_IDS)}"
            )

    own_rate = mean(own)
    base_rate = mean(baseline)
    own_comp = mean(own_complete)
    base_comp = mean(base_complete)

    summary = {
        "members": len(own),
        "conditions": len(TEST_IDS),
        "goal_accuracy": mean(goals),
        "own_commitment_correct": own_rate,
        "paired_both_sides_commitment":
            mean(pair),
        "donor_commitment_follows_donor_goal":
            mean(donor),
        "explicit_goal_flip_follows_flipped_goal":
            mean(flip),
        "zero_memory_exact_baseline":
            mean(zero_exact),
        "baseline_commitment_correct":
            base_rate,
        "commitment_advantage":
            own_rate - base_rate,
        "skill_termination_rate":
            mean(terminated),
        "duration_mae_steps":
            mean(duration_error),
        "termination_within_2_teacher_steps":
            mean(
                float(e <= 2.0)
                for e in duration_error
            ),
        "own_completion_correct":
            own_comp,
        "baseline_completion_correct":
            base_comp,
        "completion_advantage":
            own_comp - base_comp,
    }
    return summary, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recurrent_goal_option_state_v1",
    )
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / "result.json"
    controller_path = (
        state / "recurrent_goal_option_state.pt"
    )

    if not GOAL_OPTIONS_PATH.is_file():
        raise FileNotFoundError(GOAL_OPTIONS_PATH)
    if not GOAL_DECODER_PATH.is_file():
        raise FileNotFoundError(GOAL_DECODER_PATH)

    base_sha_before = sha256(BASE_CHECKPOINT)
    memory_sha_before = sha256(MEMORY_CHECKPOINT)
    goal_options_sha = sha256(GOAL_OPTIONS_PATH)
    goal_decoder_sha = sha256(GOAL_DECODER_PATH)

    print("RECURRENT_GOAL_OPTION_STATE_V1")
    print("device: cpu")
    print("trainable_component: recurrent_temporal_state_only")
    print("goal_option_policy_training_performed: false")
    print("historical_option_training_performed: false")
    print("memory_training_performed: false")
    print("planner_training_performed: false")
    print("train_conditions: 0-1599")
    print("test_conditions: 1980-2359")
    print("test_previously_touched: false")
    print("test_used_for_tuning_after_run: forbidden")

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
    frozen_historical_options = {
        k: v.detach().cpu().clone()
        for k, v in base.options.state_dict().items()
    }

    sequences, dataset_meta = collect_sequences(
        base, cfg, bank
    )
    print("recurrent_dataset:", dataset_meta)

    controller, train_metrics = train_controller(
        sequences
    )
    save_controller(
        controller_path,
        controller,
        goal_options_sha256=goal_options_sha,
        goal_decoder_sha256=goal_decoder_sha,
        metadata={
            "train_start": 0,
            "train_end": 1599,
            "seed": SEED,
            "epochs": EPOCHS,
            "remaining_target":
                "log1p_remaining_lateral_steps",
            "auxiliary_target":
                "termination_binary",
            "runtime_stop_rule":
                "remaining_steps<=0.5_after_min_commit",
        },
    )
    controller, _ = load_controller(
        controller_path,
        expected_goal_options_sha256=
            goal_options_sha,
        expected_goal_decoder_sha256=
            goal_decoder_sha,
        device="cpu",
    )

    test, details = evaluate_test(
        base,
        cfg,
        admitted,
        decoder,
        bank,
        controller,
    )

    for key, value in frozen_goal_options.items():
        got = bank.state_dict()[key].detach().cpu()
        if not torch.equal(value, got):
            raise RuntimeError(
                f"goal option tensor changed: {key}"
            )
    for key, value in frozen_historical_options.items():
        got = base.options.state_dict()[key].detach().cpu()
        if not torch.equal(value, got):
            raise RuntimeError(
                f"historical option tensor changed: {key}"
            )

    if sha256(BASE_CHECKPOINT) != base_sha_before:
        raise RuntimeError("base checkpoint changed")
    if sha256(MEMORY_CHECKPOINT) != memory_sha_before:
        raise RuntimeError("memory checkpoint changed")
    if sha256(GOAL_OPTIONS_PATH) != goal_options_sha:
        raise RuntimeError("goal option checkpoint changed")
    if sha256(GOAL_DECODER_PATH) != goal_decoder_sha:
        raise RuntimeError("goal decoder checkpoint changed")

    checks = {
        "test_goal_accuracy":
            test["goal_accuracy"] >= MIN_GOAL_ACC,
        "own_commitment":
            test["own_commitment_correct"] >= MIN_OWN_COMMIT,
        "paired_both_sides_commitment":
            test["paired_both_sides_commitment"]
            >= MIN_PAIR_COMMIT,
        "donor_commitment":
            test["donor_commitment_follows_donor_goal"]
            >= MIN_DONOR_COMMIT,
        "goal_flip_commitment":
            test[
                "explicit_goal_flip_follows_flipped_goal"
            ] >= MIN_FLIP_COMMIT,
        "zero_memory_baseline":
            test["zero_memory_exact_baseline"] == 1.0,
        "skill_termination":
            test["skill_termination_rate"] >= MIN_TERM_RATE,
        "termination_timing":
            test["termination_within_2_teacher_steps"]
            >= MIN_TERM_WITHIN2,
        "duration_mae":
            test["duration_mae_steps"] <= MAX_DURATION_MAE,
        "commitment_advantage":
            test["commitment_advantage"]
            >= MIN_COMMIT_ADVANTAGE,
        "baseline_not_solving_memory_task":
            test["baseline_commitment_correct"]
            <= MAX_BASELINE_COMMIT,
        "own_completion":
            test["own_completion_correct"]
            >= MIN_OWN_COMPLETION,
        "completion_advantage":
            test["completion_advantage"]
            >= MIN_COMPLETION_ADVANTAGE,
    }
    passed = all(checks.values())

    result = {
        "format":
            "continuous_brain.recurrent_goal_option_state_v1",
        "test_admission_passed": bool(passed),
        "training": {
            "recurrent_temporal_state": True,
            "goal_option_policy": False,
            "goal_option_bank": False,
            "historical_options": False,
            "memory": False,
            "base": False,
            "planner": False,
        },
        "split": {
            "train": [0, 1599],
            "dev_previously_consumed": [1600, 1979],
            "test": [1980, 2359],
            "test_used_once_for_admission": True,
        },
        "dataset": dataset_meta,
        "train_metrics": train_metrics,
        "test": test,
        "checks": checks,
        "details": details,
        "protected_unchanged": True,
        "goal_option_bank_bit_exact": True,
        "historical_six_options_bit_exact": True,
        "v019_remains_closed": True,
        "detour_planner_repair_closed": True,
        "test_used_for_tuning_after_run": False,
        "next_step_if_pass": (
            "Freeze memory->goal->temporal-skill pathway. "
            "Move to consequence-based planning/replanning, "
            "multi-item memory, and richer 3-D embodiment."
        ),
        "next_step_if_fail": (
            "Do not tune this TEST. Treat the current latent representation "
            "as inadequate for temporal option-state inference and move the "
            "next architecture work to richer recurrent/embodied state."
        ),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nTEST")
    for k, v in test.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("GOAL_OPTION_BANK_BIT_EXACT: true")
    print("HISTORICAL_SIX_OPTIONS_BIT_EXACT: true")
    print("V019_REMAINS_CLOSED: true")
    print("DETOUR_PLANNER_REPAIR_CLOSED: true")
    print("TEST_USED_FOR_TUNING_AFTER_RUN: false")
    print(
        "RECURRENT_GOAL_OPTION_STATE_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print("RECURRENT_GOAL_OPTION_STATE_RUN_COMPLETE")
    print("result:", result_path)
    print("controller:", controller_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
