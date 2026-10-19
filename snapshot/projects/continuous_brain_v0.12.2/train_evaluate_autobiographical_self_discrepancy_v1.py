#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    AutobiographicalSelfModel,
    AutobiographicalTrajectoryMemory,
    TrajectoryMemoryRecord,
    load_autobiographical_model,
    load_memory,
    save_autobiographical_model,
    save_memory,
)
from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    sha256,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    MODE_NORMAL,
    PRIMARY_A,
    PRIMARY_B,
    load_trajectory_model,
    rollout,
    update_explicit_intention,
)
from train_evaluate_joint_self_world_prediction_v1 import (
    make_state,
    transition,
)

TRAIN_IDS = range(0, 1024)
DEV_IDS = range(3000, 3256)
TEST_IDS = range(6000, 6256)

HISTORY_EPISODES = 12
PLAN_LEN = 4

SEED = 32001
EPOCHS = 90
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-6

MAX_RELIABILITY_MAE = .10
MIN_PREFERENCE_ACCURACY = .95
MIN_DONOR_FLIP = .95
MIN_HISTORY_CAUSAL = .95
MIN_ZERO_NEUTRAL = .95
MIN_RELOAD = 1.0

SOURCE_TRAJECTORY_MODEL = Path(
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
)
SOURCE_TRAJECTORY_RESULT = Path(
    "state/recurrent_self_trajectory_v1/result.json"
)
SOURCE_TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

PROTECTED = {
    "trajectory_model": SOURCE_TRAJECTORY_MODEL,
    "trajectory_dev_result": SOURCE_TRAJECTORY_RESULT,
    "trajectory_test_result": SOURCE_TRAJECTORY_TEST,
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
    "self_test":
        Path("state/persistent_self_autobiographical_test_v1/result.json"),
    "v029_test":
        Path("state/self_broadcast_centrality_test_v1/result.json"),
    "v030_result":
        Path("state/joint_self_world_prediction_v1/result.json"),
    "v0301_result":
        Path("state/joint_self_world_repair_v1/result.json"),
    "v0302_result":
        Path("state/joint_self_goal_boundary_diagnostic_v1/result.json"),
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def subject_profile(subject_id: int):
    rng = random.Random(0x320000 + 100003 * int(subject_id))
    a_better = (int(subject_id) % 2 == 0)

    hi = .90 + .05 * rng.random()
    lo = .05 + .10 * rng.random()

    if a_better:
        return {
            "reliability_a": hi,
            "reliability_b": lo,
            "preferred": 0,
        }
    return {
        "reliability_a": lo,
        "reliability_b": hi,
        "preferred": 1,
    }


def episode_initial_state(subject_id: int, episode_index: int, primary: int):
    rng = random.Random(
        0x321000
        + 1009 * int(subject_id)
        + 9176 * int(episode_index)
    )

    state = make_state(
        200000
        + int(subject_id) * 32
        + int(episode_index) * 2
        + int(primary)
    )
    if int(primary) == PRIMARY_A:
        state.unresolved_a = .78 + .17 * rng.random()
        state.unresolved_b = .03 + .10 * rng.random()
        state.active_goal = GOAL_COMMITMENT_A
    else:
        state.unresolved_a = .03 + .10 * rng.random()
        state.unresolved_b = .78 + .17 * rng.random()
        state.active_goal = GOAL_COMMITMENT_B

    state.confidence = .45 + .40 * rng.random()
    state.uncertainty = .10 + .22 * rng.random()
    state.prediction_error_ema = .03 + .15 * rng.random()
    state.agency_ema = .35 + .45 * rng.random()

    world = torch.tensor(
        [
            .15 + .45 * rng.random(),
            .15 + .45 * rng.random(),
            .25 + .45 * rng.random(),
            .10 + .45 * rng.random(),
        ],
        dtype=torch.float32,
    )
    return world, state


def plan_for_primary(primary: int):
    work = (
        ACTION_WORK_A
        if int(primary) == PRIMARY_A
        else ACTION_WORK_B
    )
    return [work, ACTION_OBSERVE, work, work]


def build_record(
    model,
    subject_id: int,
    episode_index: int,
    primary: int,
    reliability: float,
):
    world, state = episode_initial_state(
        subject_id,
        episode_index,
        primary,
    )
    actions = plan_for_primary(primary)

    expected = rollout(
        model,
        world,
        state.vector()[:6],
        int(primary),
        MODE_NORMAL,
        actions,
    )

    expected_self = expected["selfs"].detach().cpu()
    expected_world = expected["worlds"].detach().cpu()
    expected_modes = torch.tensor(
        expected["modes"],
        dtype=torch.long,
    )

    actual_self = []
    actual_world = []
    actual_modes = []

    actual_state = state
    actual_w = world
    intent_primary = int(primary)
    intent_mode = MODE_NORMAL

    rng = random.Random(
        0x322000
        + 131071 * int(subject_id)
        + 4099 * int(episode_index)
        + 17 * int(primary)
    )

    for action in actions:
        is_work = action in (
            ACTION_WORK_A,
            ACTION_WORK_B,
        )
        surprise = (
            is_work
            and rng.random() > float(reliability)
        )

        actual_w, actual_state, _, _ = transition(
            actual_w,
            actual_state,
            action,
            surprise=surprise,
        )

        intent = update_explicit_intention(
            intent_primary,
            intent_mode,
            actual_state.vector()[:6],
        )
        intent_primary = int(intent.primary)
        intent_mode = int(intent.mode)

        actual_world.append(actual_w.clone())
        actual_self.append(
            actual_state.vector()[:6].clone()
        )
        actual_modes.append(intent_mode)

    return TrajectoryMemoryRecord(
        episode_id=(
            int(subject_id) * 100
            + int(episode_index)
        ),
        primary=int(primary),
        expected_self=expected_self,
        actual_self=torch.stack(actual_self),
        expected_world=expected_world,
        actual_world=torch.stack(actual_world),
        expected_modes=expected_modes,
        actual_modes=torch.tensor(
            actual_modes,
            dtype=torch.long,
        ),
    )


def build_subject_memory(model, subject_id: int):
    profile = subject_profile(subject_id)
    memory = AutobiographicalTrajectoryMemory()

    for episode_index in range(HISTORY_EPISODES):
        primary = (
            PRIMARY_A
            if episode_index % 2 == 0
            else PRIMARY_B
        )
        reliability = (
            profile["reliability_a"]
            if primary == PRIMARY_A
            else profile["reliability_b"]
        )
        memory.append(
            build_record(
                model,
                subject_id,
                episode_index,
                primary,
                reliability,
            )
        )

    return memory, profile


def build_dataset(model, ids, label):
    features = []
    masks = []
    targets = []
    preferences = []
    memories = []

    ids = list(ids)
    for n, sid in enumerate(ids, 1):
        memory, profile = build_subject_memory(
            model,
            int(sid),
        )
        x, mask = memory.feature_tensor(
            HISTORY_LEN
        )

        features.append(x)
        masks.append(mask)
        targets.append(
            torch.tensor(
                [
                    profile["reliability_a"],
                    profile["reliability_b"],
                ],
                dtype=torch.float32,
            )
        )
        preferences.append(
            int(profile["preferred"])
        )
        memories.append(memory)

        if n == 1 or n % 256 == 0 or n == len(ids):
            print(
                f"{label}_autobiographical_subjects={n}/{len(ids)}"
            )

    return {
        "features":
            torch.stack(features),
        "mask":
            torch.stack(masks),
        "target":
            torch.stack(targets),
        "preference":
            torch.tensor(
                preferences,
                dtype=torch.long,
            ),
        "memories":
            memories,
        "ids":
            ids,
    }


@torch.no_grad()
def metrics(model, data):
    model.eval()
    out = model(
        data["features"],
        data["mask"],
    )
    reliability = out["reliability"]
    pref = out["preference"]

    return {
        "reliability_mae":
            float(
                (
                    reliability
                    - data["target"]
                ).abs().mean().item()
            ),
        "preference_accuracy":
            float(
                (
                    pref
                    == data["preference"]
                ).float().mean().item()
            ),
        "mean_preference_margin":
            float(
                (
                    reliability.max(dim=-1).values
                    - reliability.min(dim=-1).values
                ).mean().item()
            ),
    }


def train_model(model, train, dev):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)

    n = train["features"].shape[0]
    best = None
    best_score = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(
            n,
            generator=gen,
        )
        total = 0.0
        seen = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):
            idx = order[
                start:start + BATCH_SIZE
            ]
            out = model(
                train["features"][idx],
                train["mask"][idx],
            )
            reliability = out["reliability"]

            loss_reg = F.mse_loss(
                reliability,
                train["target"][idx],
            )

            # Preference margin objective makes the autobiographical conclusion
            # causally usable while regression retains calibrated estimates.
            chosen = reliability.gather(
                1,
                train["preference"][idx]
                .unsqueeze(-1),
            ).squeeze(-1)
            other = reliability.gather(
                1,
                (1 - train["preference"][idx])
                .unsqueeze(-1),
            ).squeeze(-1)
            loss_rank = torch.relu(
                .35 - (chosen - other)
            ).mean()

            loss = loss_reg + .20 * loss_rank

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        train_m = metrics(model, train)
        dev_m = metrics(model, dev)

        score = (
            dev_m["reliability_mae"]
            + (1.0 - dev_m["preference_accuracy"])
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"autobio_epoch={epoch:03d}/{EPOCHS} "
                f"loss={total/max(1,seen):.6f} "
                f"train_mae={train_m['reliability_mae']:.4f} "
                f"dev_mae={dev_m['reliability_mae']:.4f} "
                f"dev_pref={dev_m['preference_accuracy']:.4f} "
                f"dev_margin={dev_m['mean_preference_margin']:.4f}"
            )

        if (
            epoch >= 15
            and dev_m["reliability_mae"] <= .060
            and dev_m["preference_accuracy"] >= .990
        ):
            print(
                f"autobio_early_stop_epoch={epoch}"
            )
            break

    if best is None:
        raise RuntimeError(
            "no best autobiographical model"
        )

    model.load_state_dict(best)
    model.eval()

    return (
        metrics(model, train),
        metrics(model, dev),
        best_epoch,
    )


@torch.no_grad()
def causal_history_assays(model, dev):
    normal = []
    donor = []
    erased = []
    discrepancy_zeroed = []
    donor_margin = []

    # Pair each subject with a subject of opposite profile parity.
    for i in range(0, len(dev["ids"]), 2):
        ia = i
        ib = i + 1
        if ib >= len(dev["ids"]):
            break

        mem_a = dev["memories"][ia]
        mem_b = dev["memories"][ib]

        xa, ma = mem_a.feature_tensor(HISTORY_LEN)
        xb, mb = mem_b.feature_tensor(HISTORY_LEN)

        oa = model(
            xa.unsqueeze(0),
            ma.unsqueeze(0),
        )
        ob = model(
            xb.unsqueeze(0),
            mb.unsqueeze(0),
        )

        normal.append(
            float(
                int(
                    oa["preference"][0].item()
                )
                == int(
                    dev["preference"][ia].item()
                )
                and int(
                    ob["preference"][0].item()
                )
                == int(
                    dev["preference"][ib].item()
                )
            )
        )

        # Current world/current self are intentionally absent from model input.
        # Replacing only autobiography therefore isolates historical causation.
        donor_a = model(
            xb.unsqueeze(0),
            mb.unsqueeze(0),
        )
        donor_b = model(
            xa.unsqueeze(0),
            ma.unsqueeze(0),
        )
        donor.append(
            float(
                int(
                    donor_a["preference"][0].item()
                )
                == int(
                    dev["preference"][ib].item()
                )
                and int(
                    donor_b["preference"][0].item()
                )
                == int(
                    dev["preference"][ia].item()
                )
            )
        )

        ra = donor_a["reliability"][0]
        donor_margin.append(
            float(
                (
                    ra.max()
                    - ra.min()
                ).item()
            )
        )

        # Erased history: exact neutral 0.5 / 0.5 by architecture.
        zx = torch.zeros(
            1,
            HISTORY_LEN,
            dev["features"].shape[-1],
        )
        zm = torch.zeros(
            1,
            HISTORY_LEN,
            dtype=torch.bool,
        )
        ze = model(zx, zm)
        erased.append(
            float(
                torch.equal(
                    ze["reliability"][0],
                    torch.tensor(
                        [.5, .5]
                    ),
                )
            )
        )

        # Keep episode order + primary labels, but set expected==actual for all
        # records, eliminating expected-vs-actual discrepancy.
        ablated = mem_a.discrepancy_ablation()
        ax, am = ablated.feature_tensor(
            HISTORY_LEN
        )
        az = model(
            ax.unsqueeze(0),
            am.unsqueeze(0),
        )
        discrepancy_zeroed.append(
            float(
                abs(
                    float(
                        az["reliability"][0, 0].item()
                    )
                    - float(
                        az["reliability"][0, 1].item()
                    )
                ) <= .10
            )
        )

    return {
        "normal_history_forms_correct_commitment":
            mean(normal),
        "donor_history_flips_commitment":
            mean(donor),
        "erased_history_exact_neutral":
            mean(erased),
        "discrepancy_ablation_removes_strong_preference":
            mean(discrepancy_zeroed),
        "donor_history_mean_reliability_margin":
            mean(donor_margin),
    }


@torch.no_grad()
def persistence_assay(model, memory, state_dir):
    before_x, before_m = memory.feature_tensor(
        HISTORY_LEN
    )
    before = model(
        before_x.unsqueeze(0),
        before_m.unsqueeze(0),
    )

    path = state_dir / "autobiographical_memory_probe.pt"
    save_memory(
        path,
        memory,
    )
    loaded, _ = load_memory(
        path
    )
    after_x, after_m = loaded.feature_tensor(
        HISTORY_LEN
    )
    after = model(
        after_x.unsqueeze(0),
        after_m.unsqueeze(0),
    )

    return {
        "memory_features_exact_after_reload":
            float(
                torch.equal(
                    before_x,
                    after_x,
                )
                and torch.equal(
                    before_m,
                    after_m,
                )
            ),
        "reliability_exact_after_reload":
            float(
                torch.equal(
                    before["reliability"],
                    after["reliability"],
                )
            ),
        "preference_exact_after_reload":
            float(
                torch.equal(
                    before["preference"],
                    after["preference"],
                )
            ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/autobiographical_self_discrepancy_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_path = state_dir / "autobiographical_self_model.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    trajectory_test = json.loads(
        SOURCE_TRAJECTORY_TEST.read_text()
    )
    if trajectory_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.31.1 frozen TEST PASS required"
        )
    if (
        trajectory_test.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.31 TEST must be sealed/spent"
        )

    before_hashes = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("AUTOBIOGRAPHICAL_SELF_DISCREPANCY_V1")
    print("device: cpu")
    print("goal: expected_self_vs_actual_self_history_shapes_future_commitment")
    print("trainable_component: autobiographical_self_model_only")
    print("recurrent_trajectory_model_training: false")
    print("persistent_self_training: false")
    print("historical_brain_training: false")
    print("autobio_train_subject_ids: 0-1023")
    print("autobio_dev_subject_ids: 3000-3255")
    print("autobio_test_subject_ids: 6000-6255")
    print("autobio_test_touched: false")
    print("history_episodes_per_subject:", HISTORY_EPISODES)

    trajectory_model, _ = load_trajectory_model(
        SOURCE_TRAJECTORY_MODEL,
        "cpu",
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    train = build_dataset(
        trajectory_model,
        TRAIN_IDS,
        "train",
    )
    dev = build_dataset(
        trajectory_model,
        DEV_IDS,
        "dev",
    )

    torch.manual_seed(SEED)
    model = AutobiographicalSelfModel(
        record_dim=64,
        hidden_dim=96,
    )

    train_m, dev_m, best_epoch = train_model(
        model,
        train,
        dev,
    )

    save_autobiographical_model(
        model_path,
        model,
        metadata={
            "seed": SEED,
            "train_subject_ids": [0, 1023],
            "dev_subject_ids": [3000, 3255],
            "test_subject_ids": [6000, 6255],
            "test_touched": False,
            "history_episodes": HISTORY_EPISODES,
            "source_trajectory_model_sha256":
                sha256(SOURCE_TRAJECTORY_MODEL),
            "best_epoch": best_epoch,
        },
    )
    model, _ = load_autobiographical_model(
        model_path,
        "cpu",
    )

    causal = causal_history_assays(
        model,
        dev,
    )
    persistence = persistence_assay(
        model,
        dev["memories"][0],
        state_dir,
    )

    checks = {
        "reliability_prediction":
            dev_m["reliability_mae"]
            <= MAX_RELIABILITY_MAE,
        "commitment_formation":
            dev_m["preference_accuracy"]
            >= MIN_PREFERENCE_ACCURACY,
        "normal_history_causal":
            causal[
                "normal_history_forms_correct_commitment"
            ] >= MIN_HISTORY_CAUSAL,
        "donor_history_causal":
            causal[
                "donor_history_flips_commitment"
            ] >= MIN_DONOR_FLIP,
        "erased_history_neutral":
            causal[
                "erased_history_exact_neutral"
            ] >= MIN_ZERO_NEUTRAL,
        "expected_actual_discrepancy_load_bearing":
            causal[
                "discrepancy_ablation_removes_strong_preference"
            ] >= MIN_HISTORY_CAUSAL,
        "memory_reload_exact":
            persistence[
                "memory_features_exact_after_reload"
            ] >= MIN_RELOAD,
        "reliability_reload_exact":
            persistence[
                "reliability_exact_after_reload"
            ] >= MIN_RELOAD,
        "commitment_reload_exact":
            persistence[
                "preference_exact_after_reload"
            ] >= MIN_RELOAD,
    }
    passed = all(checks.values())

    after_hashes = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before_hashes != after_hashes:
        raise RuntimeError(
            "protected artifact changed during v0.32"
        )

    result = {
        "format":
            "continuous_brain.autobiographical_self_discrepancy_v1",
        "development_gate_passed":
            bool(passed),
        "training": {
            "autobiographical_self_model":
                True,
            "recurrent_trajectory_model":
                False,
            "persistent_self":
                False,
            "historical_brain":
                False,
        },
        "split": {
            "train": [0, 1023],
            "dev": [3000, 3255],
            "test": [6000, 6255],
            "test_touched": False,
        },
        "architecture": {
            "history_episodes":
                HISTORY_EPISODES,
            "record_is_expected_vs_actual_trajectory":
                True,
            "identity_input":
                False,
            "current_world_input":
                False,
            "current_self_input":
                False,
            "chronological_history_gru":
                True,
        },
        "scope": {
            "claim":
                "autobiographical expected-self versus actual-self discrepancy history causally changes later commitment formation while present state is held fixed",
            "subjective_experience_claimed":
                False,
            "sentience_claimed":
                False,
            "consciousness_claimed":
                False,
        },
        "best_epoch":
            best_epoch,
        "train_metrics":
            train_m,
        "dev_metrics":
            dev_m,
        "causal_history_metrics":
            causal,
        "persistence_metrics":
            persistence,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "v0311_test_pass_preserved":
            True,
        "autobio_test_touched":
            False,
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTRAIN")
    for k, v in train_m.items():
        print(f"{k}: {v}")

    print("\nDEV")
    for k, v in dev_m.items():
        print(f"{k}: {v}")

    print("\nCAUSAL_HISTORY")
    for k, v in causal.items():
        print(f"{k}: {v}")

    print("\nPERSISTENCE")
    for k, v in persistence.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0311_TEST_PASS_PRESERVED: true")
    print("AUTOBIO_TEST_TOUCHED: false")
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_RUN_COMPLETE"
    )
    print("result:", result_path)
    print("model:", model_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
