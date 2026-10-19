#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WAIT,
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    EndogenousGoalArbiter,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    load_self_model,
    save_self_bundle,
    sha256,
)
from continuous_brain.joint_self_world_prediction_v1 import (
    ACTION_DIM,
    SELF_VECTOR_DIM,
    WORLD_DIM,
    JointSelfWorldModel,
    build_self_state_from_vector,
    choose_action,
    load_joint_model,
    predicted_self_vector,
    save_joint_model,
    self_prediction_error,
)

TRAIN_IDS = range(0, 20000)
DEV_IDS = range(40000, 45000)
TEST_IDS = range(60000, 65000)

SEED = 30001
EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-6

MAX_WORLD_MAE = .025
MAX_SELF_MAE = .030
MIN_GOAL_ACC = .98
MIN_AGENCY_ACC = .98
MAX_SELF_RECON_MAE = .035

MIN_COUNTERFACTUAL_SELF = .95
MAX_SELF_ABLATION = .35
MIN_DONOR_FLIP = .95

MIN_SURPRISE_ERROR = .10
MIN_SURPRISE_INVESTIGATE = .95
MAX_NORMAL_INVESTIGATE = .10
MIN_RELOAD_ACTION = 1.0

SELF_MODEL = Path(
    "state/persistent_self_autobiographical_v1/agency_self_model.pt"
)
SELF_SEED = Path(
    "state/persistent_self_autobiographical_v1/persistent_self_seed.pt"
)
SELF_TEST = Path(
    "state/persistent_self_autobiographical_test_v1/result.json"
)
V029_TEST = Path(
    "state/self_broadcast_centrality_test_v1/result.json"
)

PROTECTED = {
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "goal_decoder":
        Path("state/memory_guided_goal_option_v1/goal_decoder.pt"),
    "consequence_model":
        Path("state/consequence_replanning_v1/consequence_model.pt"),
    "self_model":
        SELF_MODEL,
    "self_seed":
        SELF_SEED,
    "self_test":
        SELF_TEST,
    "v029_test":
        V029_TEST,
}


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def one_hot_goal(goal):
    x = torch.zeros(4, dtype=torch.float32)
    x[int(goal)] = 1.0
    return x


def make_state(example_id):
    rng = random.Random(
        0x300000 + 99991 * int(example_id)
    )
    s = PersistentSelfState.create(
        identity_seed=3000000 + int(example_id)
    )

    goal_class = int(example_id) % 4

    s.confidence = .20 + .70 * rng.random()
    s.agency_ema = .10 + .80 * rng.random()

    if goal_class == GOAL_NONE:
        s.unresolved_a = 0.0
        s.unresolved_b = 0.0
        s.uncertainty = .05 + .35 * rng.random()
        s.prediction_error_ema = .02 + .25 * rng.random()
    elif goal_class == GOAL_COMMITMENT_A:
        s.unresolved_a = .45 + .50 * rng.random()
        s.unresolved_b = .00 + .20 * rng.random()
        s.uncertainty = .05 + .35 * rng.random()
        s.prediction_error_ema = .02 + .25 * rng.random()
    elif goal_class == GOAL_COMMITMENT_B:
        s.unresolved_a = .00 + .20 * rng.random()
        s.unresolved_b = .45 + .50 * rng.random()
        s.uncertainty = .05 + .35 * rng.random()
        s.prediction_error_ema = .02 + .25 * rng.random()
    else:
        s.unresolved_a = .10 + .70 * rng.random()
        s.unresolved_b = .10 + .70 * rng.random()
        s.uncertainty = .83 + .15 * rng.random()
        s.prediction_error_ema = .60 + .35 * rng.random()

    s.active_goal = int(
        EndogenousGoalArbiter().choose(s)
    )
    if s.active_goal != goal_class:
        raise AssertionError(
            (example_id, s.active_goal, goal_class)
        )
    return s


def make_world(example_id):
    rng = random.Random(
        0x301000 + 104729 * int(example_id)
    )

    # Include a substantial fraction of saturated progress states so the
    # counterfactual "same world future, different self future" regime is in
    # distribution rather than an extrapolation.
    saturated = (int(example_id) % 7 == 0)

    if saturated:
        pa = 1.0
        pb = 1.0
    else:
        pa = .05 + .90 * rng.random()
        pb = .05 + .90 * rng.random()

    info = .05 + .90 * rng.random()
    volatility = .05 + .90 * rng.random()

    return torch.tensor(
        [pa, pb, info, volatility],
        dtype=torch.float32,
    )


def transition(
    world,
    state,
    action,
    *,
    surprise=False,
):
    w = world.detach().cpu().clone().float()
    s = state.clone()

    action = int(action)
    volatility = float(w[3].item())

    agency = 0.0
    raw_error = 0.0

    if action == ACTION_OBSERVE:
        old = float(w[2].item())
        w[2] = clip01(
            old + .30 * (1.0 - old)
        )
        s.uncertainty = clip01(
            s.uncertainty * .65
        )
        s.prediction_error_ema = clip01(
            s.prediction_error_ema * .60
        )
        s.confidence = clip01(
            s.confidence
            + .06 * (1.0 - s.confidence)
        )
        agency = 1.0

    elif action == ACTION_WORK_A:
        effort = .18 * (
            1.0 - .35 * volatility
        )
        if surprise:
            raw_error = .90
            s.confidence = clip01(
                s.confidence * .78
            )
            s.uncertainty = clip01(
                s.uncertainty
                + .42 * (1.0 - s.uncertainty)
            )
            s.prediction_error_ema = clip01(
                .35 * s.prediction_error_ema
                + .65
            )
            agency = 0.0
        else:
            before = float(w[0].item())
            w[0] = clip01(before + effort)
            realized = float(w[0].item()) - before
            # Internal commitment progress follows the executed effort, not
            # only the externally visible progress increment.  This creates
            # legitimate cases where two actions can have the same saturated
            # world outcome but different consequences for the continuing
            # self.
            s.unresolved_a = clip01(
                s.unresolved_a
                - 1.20 * effort
            )
            s.confidence = clip01(
                s.confidence
                + .10 * (1.0 - s.confidence)
            )
            s.uncertainty = clip01(
                s.uncertainty * .88
            )
            s.prediction_error_ema = clip01(
                s.prediction_error_ema * .70
            )
            agency = 1.0

    elif action == ACTION_WORK_B:
        effort = .18 * (
            1.0 - .35 * volatility
        )
        if surprise:
            raw_error = .90
            s.confidence = clip01(
                s.confidence * .78
            )
            s.uncertainty = clip01(
                s.uncertainty
                + .42 * (1.0 - s.uncertainty)
            )
            s.prediction_error_ema = clip01(
                .35 * s.prediction_error_ema
                + .65
            )
            agency = 0.0
        else:
            before = float(w[1].item())
            w[1] = clip01(before + effort)
            realized = float(w[1].item()) - before
            s.unresolved_b = clip01(
                s.unresolved_b
                - 1.20 * effort
            )
            s.confidence = clip01(
                s.confidence
                + .10 * (1.0 - s.confidence)
            )
            s.uncertainty = clip01(
                s.uncertainty * .88
            )
            s.prediction_error_ema = clip01(
                s.prediction_error_ema * .70
            )
            agency = 1.0

    elif action == ACTION_WAIT:
        s.prediction_error_ema = clip01(
            s.prediction_error_ema * .92
        )
        s.uncertainty = clip01(
            s.uncertainty
            + .02 * volatility
        )
        agency = 0.0

    else:
        raise ValueError(action)

    # Surprise is world-attributed and should recruit information-seeking.
    if surprise:
        s.agency_ema = clip01(
            .80 * s.agency_ema
            + .20 * 0.0
        )
    else:
        s.agency_ema = clip01(
            .80 * s.agency_ema
            + .20 * agency
        )

    s.active_goal = int(
        EndogenousGoalArbiter().choose(s)
    )

    return w, s, float(agency), float(raw_error)


def make_example(example_id):
    state = make_state(example_id)
    world = make_world(example_id)

    rng = random.Random(
        0x302000 + 130363 * int(example_id)
    )
    action = rng.randrange(0, ACTION_DIM)

    next_world, next_state, agency, _ = transition(
        world,
        state,
        action,
        surprise=False,
    )

    return {
        "world": world,
        "self": state.vector(),
        "action": int(action),
        "next_world": next_world,
        "next_self_cont":
            next_state.vector()[:6],
        "next_goal":
            int(next_state.active_goal),
        "agency": float(agency),
        "current_self":
            state.vector(),
    }


def make_dataset(ids, label):
    ids = list(ids)
    rows = []

    for n, eid in enumerate(ids, 1):
        rows.append(make_example(eid))
        if n == 1 or n % 5000 == 0 or n == len(ids):
            print(
                f"{label}_joint_examples={n}/{len(ids)}"
            )

    return {
        "world":
            torch.stack(
                [r["world"] for r in rows]
            ).float(),
        "self":
            torch.stack(
                [r["self"] for r in rows]
            ).float(),
        "action":
            torch.tensor(
                [r["action"] for r in rows],
                dtype=torch.long,
            ),
        "next_world":
            torch.stack(
                [r["next_world"] for r in rows]
            ).float(),
        "next_self_cont":
            torch.stack(
                [r["next_self_cont"] for r in rows]
            ).float(),
        "next_goal":
            torch.tensor(
                [r["next_goal"] for r in rows],
                dtype=torch.long,
            ),
        "agency":
            torch.tensor(
                [r["agency"] for r in rows],
                dtype=torch.float32,
            ),
        "current_self":
            torch.stack(
                [r["current_self"] for r in rows]
            ).float(),
    }


@torch.no_grad()
def metrics(model, data):
    model.eval()
    out = model(
        data["world"],
        data["self"],
        data["action"],
    )

    agency = (
        torch.sigmoid(
            out["agency_logit"]
        ) >= .5
    )
    agency_true = data["agency"] >= .5

    return {
        "world_mae":
            float(
                (
                    out["next_world"]
                    - data["next_world"]
                ).abs().mean().item()
            ),
        "self_continuous_mae":
            float(
                (
                    out["next_self_continuous"]
                    - data["next_self_cont"]
                ).abs().mean().item()
            ),
        "next_goal_accuracy":
            float(
                (
                    out["next_goal_logits"]
                    .argmax(dim=-1)
                    == data["next_goal"]
                ).float().mean().item()
            ),
        "agency_accuracy":
            float(
                (
                    agency == agency_true
                ).float().mean().item()
            ),
        "current_self_reconstruction_mae":
            float(
                (
                    out["current_self_reconstruction"]
                    - data["current_self"]
                ).abs().mean().item()
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

    n = train["world"].shape[0]
    best = None
    best_score = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(
            n, generator=gen
        )
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[
                start:start + BATCH_SIZE
            ]
            out = model(
                train["world"][idx],
                train["self"][idx],
                train["action"][idx],
            )

            world_loss = F.mse_loss(
                out["next_world"],
                train["next_world"][idx],
            )
            self_loss = F.mse_loss(
                out["next_self_continuous"],
                train["next_self_cont"][idx],
            )
            goal_loss = F.cross_entropy(
                out["next_goal_logits"],
                train["next_goal"][idx],
            )
            agency_loss = (
                F.binary_cross_entropy_with_logits(
                    out["agency_logit"],
                    train["agency"][idx],
                )
            )
            recon_loss = F.mse_loss(
                out["current_self_reconstruction"],
                train["current_self"][idx],
            )

            loss = (
                1.2 * world_loss
                + 1.5 * self_loss
                + .8 * goal_loss
                + .5 * agency_loss
                + .5 * recon_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        train_m = metrics(
            model, train
        )
        dev_m = metrics(
            model, dev
        )
        score = (
            dev_m["world_mae"]
            + dev_m["self_continuous_mae"]
            + .5 * dev_m[
                "current_self_reconstruction_mae"
            ]
            + (1.0 - dev_m["next_goal_accuracy"])
            + .5 * (
                1.0 - dev_m["agency_accuracy"]
            )
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"joint_epoch={epoch:03d}/{EPOCHS} "
                f"loss={total/max(1,seen):.6f} "
                f"dev_world={dev_m['world_mae']:.4f} "
                f"dev_self={dev_m['self_continuous_mae']:.4f} "
                f"dev_goal={dev_m['next_goal_accuracy']:.4f} "
                f"dev_agency={dev_m['agency_accuracy']:.4f} "
                f"dev_recon={dev_m['current_self_reconstruction_mae']:.4f}"
            )

        if (
            epoch >= 15
            and dev_m["world_mae"] <= .015
            and dev_m["self_continuous_mae"] <= .018
            and dev_m["next_goal_accuracy"] >= .995
            and dev_m["agency_accuracy"] >= .995
            and dev_m["current_self_reconstruction_mae"] <= .020
        ):
            print(
                f"joint_early_stop_epoch={epoch}"
            )
            break

    if best is None:
        raise RuntimeError("no best joint model")
    model.load_state_dict(best)
    model.eval()

    return (
        metrics(model, train),
        metrics(model, dev),
        best_epoch,
    )


@torch.no_grad()
def counterfactual_self_assay(model):
    full = []
    ablated = []
    donor = []

    for i in range(256):
        # Externally saturated world: WORK_A/WORK_B no longer change world
        # progress, so the reason to choose one comes from future SELF.
        world = torch.tensor(
            [1.0, 1.0, .55, .35],
            dtype=torch.float32,
        )

        for goal in (
            GOAL_COMMITMENT_A,
            GOAL_COMMITMENT_B,
        ):
            eid = (
                80000
                + 8 * i
                + goal
            )
            state = make_state(eid)

            # Force a strong still-unresolved commitment while preserving the
            # goal class. This is a new causal assay, not training data.
            if goal == GOAL_COMMITMENT_A:
                state.unresolved_a = .95
                state.unresolved_b = .05
            else:
                state.unresolved_a = .05
                state.unresolved_b = .95
            state.uncertainty = .20
            state.prediction_error_ema = .10
            state.active_goal = goal

            expected_action = (
                ACTION_WORK_A
                if goal == GOAL_COMMITMENT_A
                else ACTION_WORK_B
            )

            r = choose_action(
                model,
                world,
                state.vector(),
                self_weight=1.5,
            )
            full.append(
                float(
                    r["action"]
                    == expected_action
                )
            )

            # Self-consequence ablation: world future only.
            z = choose_action(
                model,
                world,
                state.vector(),
                self_weight=0.0,
            )
            ablated.append(
                float(
                    z["action"]
                    == expected_action
                )
            )

        # Donor self with same external world.
        a = make_state(
            85000 + 8 * i + GOAL_COMMITMENT_A
        )
        b = make_state(
            85000 + 8 * i + GOAL_COMMITMENT_B
        )
        a.unresolved_a = .95
        a.unresolved_b = .05
        a.uncertainty = .20
        a.prediction_error_ema = .10
        a.active_goal = GOAL_COMMITMENT_A

        b.unresolved_a = .05
        b.unresolved_b = .95
        b.uncertainty = .20
        b.prediction_error_ema = .10
        b.active_goal = GOAL_COMMITMENT_B

        ra = choose_action(
            model, world, a.vector(),
            self_weight=1.5,
        )
        rb = choose_action(
            model, world, b.vector(),
            self_weight=1.5,
        )
        donor.append(
            float(
                ra["action"] == ACTION_WORK_A
                and rb["action"] == ACTION_WORK_B
            )
        )

    return {
        "self_consequence_guided_action_accuracy":
            sum(full) / len(full),
        "self_consequence_ablated_accuracy":
            sum(ablated) / len(ablated),
        "donor_self_flips_action_with_world_fixed":
            sum(donor) / len(donor),
    }


@torch.no_grad()
def surprise_self_error_assay(model):
    errors = []
    investigate = []
    normal_investigate = []
    zeroed_investigate = []

    for i in range(128):
        eid = 90000 + 4 * i + GOAL_COMMITMENT_A
        state = make_state(eid)
        state.unresolved_a = .85
        state.unresolved_b = .05
        state.uncertainty = .20
        state.prediction_error_ema = .10
        state.active_goal = GOAL_COMMITMENT_A

        world = torch.tensor(
            [.35, .20, .45, .35],
            dtype=torch.float32,
        )

        action = torch.tensor(
            [ACTION_WORK_A],
            dtype=torch.long,
        )
        out = model(
            world.unsqueeze(0),
            state.vector().unsqueeze(0),
            action,
        )
        predicted = predicted_self_vector(
            out
        )[0]

        normal_world, normal_state, _, _ = transition(
            world,
            state,
            ACTION_WORK_A,
            surprise=False,
        )
        surprise_world, surprise_state, _, _ = transition(
            world,
            state,
            ACTION_WORK_A,
            surprise=True,
        )

        err = self_prediction_error(
            predicted,
            surprise_state.vector(),
        )
        errors.append(err)

        investigate.append(
            float(
                surprise_state.active_goal
                == GOAL_INVESTIGATE
            )
        )
        normal_investigate.append(
            float(
                normal_state.active_goal
                == GOAL_INVESTIGATE
            )
        )

        # Causal control: replace only the two self-error variables with the
        # model-predicted values. The same underlying commitment/world remains.
        z = surprise_state.clone()
        z.uncertainty = float(
            predicted[3].item()
        )
        z.prediction_error_ema = float(
            predicted[4].item()
        )
        z.active_goal = int(
            EndogenousGoalArbiter().choose(z)
        )
        zeroed_investigate.append(
            float(
                z.active_goal
                == GOAL_INVESTIGATE
            )
        )

    return {
        "mean_surprise_self_prediction_error":
            sum(errors) / len(errors),
        "surprise_recruits_investigate":
            sum(investigate) / len(investigate),
        "normal_transition_investigate_rate":
            sum(normal_investigate)
            / len(normal_investigate),
        "self_error_state_zeroed_investigate_rate":
            sum(zeroed_investigate)
            / len(zeroed_investigate),
    }


@torch.no_grad()
def reload_action_assay(model, state_dir):
    agency_model, _ = load_self_model(
        SELF_MODEL,
        "cpu",
    )
    exact = []

    for i in range(32):
        state = make_state(
            100000
            + 4 * i
            + GOAL_COMMITMENT_A
        )
        state.unresolved_a = .90
        state.unresolved_b = .10
        state.active_goal = GOAL_COMMITMENT_A

        core = PersistentSelfCore(
            agency_model,
            state=state,
        )

        world = torch.tensor(
            [1.0, 1.0, .55, .35],
            dtype=torch.float32,
        )

        before = choose_action(
            model,
            world,
            core.state.vector(),
            self_weight=1.5,
        )["action"]

        path = state_dir / "joint_self_reload_probe.pt"
        save_self_bundle(
            path,
            core,
            metadata={
                "v030_reload_probe": i
            },
        )
        loaded, _ = load_self_bundle(
            path,
            agency_model,
        )

        after = choose_action(
            model,
            world,
            loaded.state.vector(),
            self_weight=1.5,
        )["action"]

        exact.append(
            float(
                before == after
                and before == ACTION_WORK_A
            )
        )

    return {
        "action_choice_exact_after_self_reload":
            sum(exact) / len(exact)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/joint_self_world_prediction_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    model_path = state_dir / "joint_self_world_model.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    self_test = json.loads(
        SELF_TEST.read_text()
    )
    if self_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.28.1 self TEST PASS required"
        )

    v029 = json.loads(
        V029_TEST.read_text()
    )
    if v029.get(
        "test_admission_passed"
    ) is not False:
        raise RuntimeError(
            "expected preserved v0.29.3 TEST FAIL"
        )
    if (
        v029.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.29 TEST status not sealed"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("JOINT_SELF_WORLD_PREDICTION_V1")
    print("device: cpu")
    print("goal: predict_future_world_and_future_self_jointly")
    print("trainable_component: joint_self_world_model_only")
    print("historical_brain_training: false")
    print("persistent_self_training: false")
    print("v029_test_status_preserved: FAIL")
    print("joint_train_ids: 0-19999")
    print("joint_dev_ids: 40000-44999")
    print("joint_test_ids: 60000-64999")
    print("joint_test_touched: false")

    train = make_dataset(
        TRAIN_IDS,
        "train",
    )
    dev = make_dataset(
        DEV_IDS,
        "dev",
    )

    torch.manual_seed(SEED)
    model = JointSelfWorldModel(
        hidden_dim=160,
        latent_dim=96,
    )

    train_metrics, dev_metrics, best_epoch = train_model(
        model,
        train,
        dev,
    )

    save_joint_model(
        model_path,
        model,
        metadata={
            "seed": SEED,
            "train_ids": [0, 19999],
            "dev_ids": [40000, 44999],
            "test_ids": [60000, 64999],
            "test_touched": False,
            "v029_test_preserved": "FAIL",
            "best_epoch": best_epoch,
        },
    )
    model, _ = load_joint_model(
        model_path,
        "cpu",
    )

    counterfactual = counterfactual_self_assay(
        model
    )
    surprise = surprise_self_error_assay(
        model
    )
    reload = reload_action_assay(
        model,
        state_dir,
    )

    checks = {
        "world_future_prediction":
            dev_metrics["world_mae"]
            <= MAX_WORLD_MAE,
        "self_future_prediction":
            dev_metrics["self_continuous_mae"]
            <= MAX_SELF_MAE,
        "future_goal_prediction":
            dev_metrics["next_goal_accuracy"]
            >= MIN_GOAL_ACC,
        "agency_prediction":
            dev_metrics["agency_accuracy"]
            >= MIN_AGENCY_ACC,
        "current_self_preserved_in_latent":
            dev_metrics[
                "current_self_reconstruction_mae"
            ] <= MAX_SELF_RECON_MAE,

        "counterfactual_self_guides_action":
            counterfactual[
                "self_consequence_guided_action_accuracy"
            ] >= MIN_COUNTERFACTUAL_SELF,
        "self_consequence_ablation_collapses":
            counterfactual[
                "self_consequence_ablated_accuracy"
            ] <= MAX_SELF_ABLATION,
        "donor_self_flips_action":
            counterfactual[
                "donor_self_flips_action_with_world_fixed"
            ] >= MIN_DONOR_FLIP,

        "surprise_creates_self_prediction_error":
            surprise[
                "mean_surprise_self_prediction_error"
            ] >= MIN_SURPRISE_ERROR,
        "surprise_recruits_investigate":
            surprise[
                "surprise_recruits_investigate"
            ] >= MIN_SURPRISE_INVESTIGATE,
        "normal_does_not_recruit_investigate":
            surprise[
                "normal_transition_investigate_rate"
            ] <= MAX_NORMAL_INVESTIGATE,
        "zero_self_error_removes_investigate":
            surprise[
                "self_error_state_zeroed_investigate_rate"
            ] <= MAX_NORMAL_INVESTIGATE,

        "action_choice_survives_self_reload":
            reload[
                "action_choice_exact_after_self_reload"
            ] >= MIN_RELOAD_ACTION,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during v0.30"
        )

    result = {
        "format":
            "continuous_brain.joint_self_world_prediction_v1",
        "development_gate_passed": bool(passed),
        "training": {
            "joint_self_world_model": True,
            "historical_brain": False,
            "persistent_self": False,
            "working_memory": False,
            "episodic_reader": False,
        },
        "split": {
            "train": [0, 19999],
            "dev": [40000, 44999],
            "test": [60000, 64999],
            "test_touched": False,
        },
        "historical_status": {
            "v0281_self_test": "PASS",
            "v0293_self_broadcast_test": "FAIL",
            "v0293_test_spent": True,
        },
        "scope": {
            "claim":
                "joint prediction of external future and future persistent-self state with self-consequence-sensitive counterfactual action choice",
            "subjective_experience_claimed": False,
            "sentience_claimed": False,
            "consciousness_claimed": False,
        },
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "counterfactual_self_metrics":
            counterfactual,
        "surprise_self_error_metrics":
            surprise,
        "persistent_reload_metrics":
            reload,
        "checks": checks,
        "protected_unchanged": True,
        "v0293_test_status_preserved": "FAIL",
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nDEV")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nCOUNTERFACTUAL_SELF")
    for k, v in counterfactual.items():
        print(f"{k}: {v}")

    print("\nSURPRISE_SELF_ERROR")
    for k, v in surprise.items():
        print(f"{k}: {v}")

    print("\nPERSISTENT_RELOAD")
    for k, v in reload.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0293_TEST_STATUS_PRESERVED: FAIL")
    print("JOINT_TEST_TOUCHED: false")
    print(
        "JOINT_SELF_WORLD_PREDICTION_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "JOINT_SELF_WORLD_PREDICTION_RUN_COMPLETE"
    )
    print("result:", result_path)
    print("model:", model_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
