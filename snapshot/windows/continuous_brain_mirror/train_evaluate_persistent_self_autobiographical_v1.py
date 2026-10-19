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
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    GOAL_NONE,
    AgencySelfModel,
    AutobiographicalMemory,
    EndogenousGoalArbiter,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    load_self_model,
    save_self_bundle,
    save_self_model,
    sha256,
    target_internal_update,
)

TRAIN_IDS = range(0, 16000)
DEV_IDS = range(20000, 24000)
TEST_IDS = range(30000, 34000)

SEED = 28001
EPOCHS = 60
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5

MIN_AGENCY_ACC = .97
MAX_CONF_MAE = .035
MAX_UNC_MAE = .035
MIN_CAUSAL_SELF_PROB = .85
MAX_EXOGENOUS_SELF_PROB = .15

MIN_CONTINUITY = 1.0
MIN_AUTOBIO_CAUSAL = 1.0
MIN_ENDOGENOUS = 1.0
MIN_AGENCY_INTERVENTION = 1.0
MIN_TEMPORARY_INVESTIGATE_RESUME = 1.0

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
    "consequence_test":
        Path("state/consequence_replanning_test_v1/result.json"),
    "thor_admission":
        Path("state/thor_discovery_embodiment_admission_v1/result.json"),
    "thor_bridge_result":
        Path("state/thor_episodic_door_replanning_v1/result.json"),
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def make_state_from_rng(rng):
    state = PersistentSelfState.create(
        identity_seed=rng.randrange(1, 1_000_000_000)
    )
    state.unresolved_a = rng.random()
    state.unresolved_b = rng.random()
    state.active_goal = rng.randrange(0, 4)
    state.confidence = 0.15 + 0.75 * rng.random()
    state.uncertainty = 0.05 + 0.85 * rng.random()
    state.prediction_error_ema = rng.random()
    state.agency_ema = rng.random()
    state.cycle = rng.randrange(0, 500)
    return state


def sample_example(example_id):
    rng = random.Random(0x280000 + 104729 * int(example_id))
    state = make_state_from_rng(rng)

    action = rng.randrange(0, 4)
    if action == ACTION_WORK_A:
        intended = [1.0, 0.0]
    elif action == ACTION_WORK_B:
        intended = [0.0, 1.0]
    else:
        intended = [0.0, 0.0]

    work_action = action in (ACTION_WORK_A, ACTION_WORK_B)
    causal_success = bool(work_action and rng.random() < .72)

    # Ordinary observe/wait actions can execute successfully without causing
    # the world change.  For work actions this signal indicates whether the
    # intended intervention itself succeeded.
    action_success = (
        1.0
        if (not work_action or causal_success)
        else 0.0
    )

    observed = [0.0, 0.0]
    if causal_success:
        observed = list(intended)

    # Exogenous changes can occur with or without a simultaneous action.
    external = bool(rng.random() < .36)
    if external:
        observed[rng.randrange(0, 2)] = 1.0

    agency_target = 1.0 if causal_success else 0.0

    mismatch = 0.5 * (
        abs(observed[0] - intended[0])
        + abs(observed[1] - intended[1])
    )
    failure_penalty = (
        .35 if work_action and not causal_success else 0.0
    )
    prediction_error = min(
        1.0, float(mismatch + failure_penalty)
    )

    next_conf, next_unc = target_internal_update(
        state.confidence,
        state.uncertainty,
        agency_target=agency_target,
        prediction_error=prediction_error,
    )

    return {
        "self_vector": state.vector(),
        "action": int(action),
        "intended": intended,
        "observed": observed,
        "action_success": float(action_success),
        "prediction_error": float(prediction_error),
        "agency_target": float(agency_target),
        "next_confidence": float(next_conf),
        "next_uncertainty": float(next_unc),
    }


def make_dataset(ids, label):
    rows = []
    ids = list(ids)
    for i, eid in enumerate(ids, 1):
        rows.append(sample_example(eid))
        if i == 1 or i % 2000 == 0 or i == len(ids):
            print(f"{label}_self_examples={i}/{len(ids)}")

    return {
        "self_vector":
            torch.stack([r["self_vector"] for r in rows]).float(),
        "action":
            torch.tensor([r["action"] for r in rows], dtype=torch.long),
        "intended_change":
            torch.tensor([r["intended"] for r in rows], dtype=torch.float32),
        "observed_change":
            torch.tensor([r["observed"] for r in rows], dtype=torch.float32),
        "action_success":
            torch.tensor([r["action_success"] for r in rows], dtype=torch.float32),
        "prediction_error":
            torch.tensor([r["prediction_error"] for r in rows], dtype=torch.float32),
        "agency_target":
            torch.tensor([r["agency_target"] for r in rows], dtype=torch.float32),
        "next_confidence":
            torch.tensor([r["next_confidence"] for r in rows], dtype=torch.float32),
        "next_uncertainty":
            torch.tensor([r["next_uncertainty"] for r in rows], dtype=torch.float32),
    }


@torch.no_grad()
def model_metrics(model, data):
    model.eval()
    out = model(
        data["self_vector"],
        data["action"],
        data["intended_change"],
        data["observed_change"],
        data["action_success"],
        data["prediction_error"],
    )
    agency_p = torch.sigmoid(out["agency_logit"])
    agency_pred = agency_p >= .5
    agency_true = data["agency_target"] >= .5

    return {
        "agency_accuracy":
            float((agency_pred == agency_true).float().mean().item()),
        "confidence_mae":
            float(
                (out["next_confidence"] - data["next_confidence"])
                .abs().mean().item()
            ),
        "uncertainty_mae":
            float(
                (out["next_uncertainty"] - data["next_uncertainty"])
                .abs().mean().item()
            ),
    }


def train_model(train):
    torch.manual_seed(SEED)
    model = AgencySelfModel(hidden_dim=96)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)
    n = train["action"].shape[0]
    last = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=gen)
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            out = model(
                train["self_vector"][idx],
                train["action"][idx],
                train["intended_change"][idx],
                train["observed_change"][idx],
                train["action_success"][idx],
                train["prediction_error"][idx],
            )
            agency_loss = F.binary_cross_entropy_with_logits(
                out["agency_logit"],
                train["agency_target"][idx],
            )
            conf_loss = F.mse_loss(
                out["next_confidence"],
                train["next_confidence"][idx],
            )
            unc_loss = F.mse_loss(
                out["next_uncertainty"],
                train["next_uncertainty"][idx],
            )
            loss = agency_loss + 2.0 * conf_loss + 2.0 * unc_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        last = model_metrics(model, train)
        last["epoch"] = int(epoch)
        last["loss"] = total / max(1, seen)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"self_model_epoch={epoch:03d}/{EPOCHS} "
                f"loss={last['loss']:.6f} "
                f"agency={last['agency_accuracy']:.4f} "
                f"conf_mae={last['confidence_mae']:.4f} "
                f"unc_mae={last['uncertainty_mae']:.4f}"
            )

        if (
            epoch >= 10
            and last["agency_accuracy"] >= .998
            and last["confidence_mae"] <= .012
            and last["uncertainty_mae"] <= .012
        ):
            print(f"self_model_early_stop_epoch={epoch}")
            break

    model.eval()
    return model, last


def state_exact(a, b):
    scalar_names = (
        "identity_token",
        "unresolved_a",
        "unresolved_b",
        "active_goal",
        "confidence",
        "uncertainty",
        "prediction_error_ema",
        "agency_ema",
        "cycle",
        "autobiographical_cursor",
    )
    return (
        all(getattr(a, k) == getattr(b, k) for k in scalar_names)
        and torch.equal(a.identity_embedding, b.identity_embedding)
    )


def canonical_agency_probe(model):
    state = PersistentSelfState.create(777)
    state.confidence = .55
    state.uncertainty = .30
    state.prediction_error_ema = .10

    def prob(action, intended, observed, success, pe):
        sv = state.vector().unsqueeze(0)
        out = model(
            sv,
            torch.tensor([action]),
            torch.tensor([intended], dtype=torch.float32),
            torch.tensor([observed], dtype=torch.float32),
            torch.tensor([success], dtype=torch.float32),
            torch.tensor([pe], dtype=torch.float32),
        )
        return float(torch.sigmoid(out["agency_logit"])[0].item())

    self_caused = prob(
        ACTION_WORK_A,
        [1.0, 0.0],
        [1.0, 0.0],
        1.0,
        0.0,
    )
    exogenous = prob(
        ACTION_WAIT,
        [0.0, 0.0],
        [1.0, 0.0],
        1.0,
        .5,
    )
    return self_caused, exogenous


def run_continuity_assays(model, state_dir):
    identity_exact = []
    state_roundtrip = []
    memory_roundtrip = []
    resume_goal = []
    reset_control_resume = []

    for i in range(32):
        target = (
            GOAL_COMMITMENT_A
            if i % 2 == 0
            else GOAL_COMMITMENT_B
        )
        core = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(50000 + i),
        )
        core.add_commitment(target, 1.0)
        if target == GOAL_COMMITMENT_A:
            core.state.unresolved_b = .20
        else:
            core.state.unresolved_a = .20
        core.choose_goal()

        # Record a benign ongoing event so continuity includes autobiography,
        # not just a naked state tensor.
        core.observe_event(
            action=ACTION_OBSERVE,
            intended_change=(0.0, 0.0),
            observed_change=(0.0, 0.0),
            action_success=1.0,
            prediction_error=0.0,
        )
        core.state.active_goal = target

        path = state_dir / "continuity_probe.pt"
        save_self_bundle(
            path,
            core,
            metadata={"probe": i, "interrupted_mid_intention": True},
        )
        loaded, _ = load_self_bundle(path, model)

        identity_exact.append(
            float(
                core.state.identity_token
                == loaded.state.identity_token
                and torch.equal(
                    core.state.identity_embedding,
                    loaded.state.identity_embedding,
                )
            )
        )
        state_roundtrip.append(
            float(state_exact(core.state, loaded.state))
        )
        memory_roundtrip.append(
            float(
                core.memory.to_list()
                == loaded.memory.to_list()
            )
        )
        resume_goal.append(
            float(loaded.choose_goal() == target)
        )

        reset = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(50000 + i),
        )
        reset_control_resume.append(
            float(reset.choose_goal() == target)
        )

    return {
        "identity_exact_after_reload": mean(identity_exact),
        "self_state_exact_after_reload": mean(state_roundtrip),
        "autobiography_exact_after_reload": mean(memory_roundtrip),
        "unresolved_intention_resume_rate": mean(resume_goal),
        "reset_control_resume_rate": mean(reset_control_resume),
    }


def run_autobiography_causal_assays(model):
    erase_changes_goal = []
    donor_changes_goal = []
    identity_survives_donor = []
    normal_goal_correct = []

    for i in range(128):
        subject = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(60000 + i),
        )
        subject.state.unresolved_a = 1.0
        subject.state.unresolved_b = 1.0
        subject.choose_goal()

        completion = subject.record_completion(
            GOAL_COMMITMENT_A
        )
        normal_goal = subject.reconcile_commitments_from_autobiography(
            seed_a=1.0,
            seed_b=1.0,
        )
        normal_goal_correct.append(
            float(normal_goal == GOAL_COMMITMENT_B)
        )

        erased = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        erased.memory.erase_episode(completion.episode_id)
        erased_goal = erased.reconcile_commitments_from_autobiography(
            seed_a=1.0,
            seed_b=1.0,
        )
        erase_changes_goal.append(
            float(erased_goal != normal_goal)
        )

        donor = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(70000 + i),
        )
        donor.state.unresolved_a = 1.0
        donor.state.unresolved_b = 1.0
        donor.choose_goal()
        donor_completion = donor.record_completion(
            GOAL_COMMITMENT_B
        )

        swapped = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        token_before = swapped.state.identity_token
        embedding_before = swapped.state.identity_embedding.clone()
        swapped.memory.replace_episode(
            completion.episode_id,
            donor_completion,
        )
        swapped_goal = swapped.reconcile_commitments_from_autobiography(
            seed_a=1.0,
            seed_b=1.0,
        )
        donor_changes_goal.append(
            float(
                swapped_goal == GOAL_COMMITMENT_A
                and swapped_goal != normal_goal
            )
        )
        identity_survives_donor.append(
            float(
                swapped.state.identity_token == token_before
                and torch.equal(
                    swapped.state.identity_embedding,
                    embedding_before,
                )
            )
        )

    return {
        "normal_history_reconstructs_expected_goal":
            mean(normal_goal_correct),
        "erase_completion_changes_reconstructed_goal":
            mean(erase_changes_goal),
        "donor_history_changes_reconstructed_goal":
            mean(donor_changes_goal),
        "identity_anchor_survives_history_donor_swap":
            mean(identity_survives_donor),
    }


def run_endogenous_goal_assays(model):
    generated = []
    zero_ablation = []
    temporary_investigate = []
    resumes_commitment = []

    for i in range(128):
        s = PersistentSelfState.create(80000 + i)
        s.unresolved_a = 0.0
        s.unresolved_b = 0.0
        s.uncertainty = .40
        s.prediction_error_ema = .78
        core = PersistentSelfCore(model, state=s)
        generated.append(
            float(core.choose_goal() == GOAL_INVESTIGATE)
        )

        z = s.clone()
        z.prediction_error_ema = 0.0
        zero = PersistentSelfCore(model, state=z)
        zero_ablation.append(
            float(zero.choose_goal() == GOAL_NONE)
        )

        committed = PersistentSelfState.create(90000 + i)
        committed.unresolved_a = 1.0
        committed.unresolved_b = .25
        committed.uncertainty = .35
        committed.prediction_error_ema = .80
        cc = PersistentSelfCore(model, state=committed)
        temporary_investigate.append(
            float(cc.choose_goal() == GOAL_INVESTIGATE)
        )
        cc.state.prediction_error_ema = .05
        cc.state.uncertainty = .25
        resumes_commitment.append(
            float(cc.choose_goal() == GOAL_COMMITMENT_A)
        )

    return {
        "prediction_error_generates_investigate_goal":
            mean(generated),
        "prediction_error_zero_ablation_removes_investigate":
            mean(zero_ablation),
        "surprise_temporarily_interrupts_commitment":
            mean(temporary_investigate),
        "commitment_resumes_after_surprise_resolves":
            mean(resumes_commitment),
    }


def run_agency_intervention_assays(model):
    goal_changes = []
    uncertainty_orders = []
    world_error_orders = []

    for i in range(128):
        base = PersistentSelfState.create(100000 + i)
        base.unresolved_a = 0.0
        base.unresolved_b = 0.0
        base.confidence = .55
        base.uncertainty = .30
        base.prediction_error_ema = .20

        self_attr = PersistentSelfCore(model, state=base)
        world_attr = PersistentSelfCore(model, state=base)

        a = self_attr.observe_event(
            action=ACTION_WORK_A,
            intended_change=(1.0, 0.0),
            observed_change=(0.0, 1.0),
            action_success=0.0,
            prediction_error=1.0,
            forced_agency_probability=1.0,
        )
        w = world_attr.observe_event(
            action=ACTION_WORK_A,
            intended_change=(1.0, 0.0),
            observed_change=(0.0, 1.0),
            action_success=0.0,
            prediction_error=1.0,
            forced_agency_probability=0.0,
        )

        goal_changes.append(
            float(
                a["active_goal"] == GOAL_NONE
                and w["active_goal"] == GOAL_INVESTIGATE
            )
        )
        uncertainty_orders.append(
            float(
                w["next_uncertainty"]
                > a["next_uncertainty"]
            )
        )
        world_error_orders.append(
            float(
                world_attr.state.prediction_error_ema
                > self_attr.state.prediction_error_ema
            )
        )

    return {
        "agency_intervention_changes_goal":
            mean(goal_changes),
        "world_attribution_increases_uncertainty_more":
            mean(uncertainty_orders),
        "world_attribution_increases_world_error_more":
            mean(world_error_orders),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/persistent_self_autobiographical_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    model_path = state_dir / "agency_self_model.pt"
    self_bundle_path = state_dir / "persistent_self_seed.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    v025 = json.loads(PROTECTED["consequence_test"].read_text())
    if v025.get("test_admission_passed") is not True:
        raise RuntimeError("v0.25.2 TEST PASS required")
    v027 = json.loads(PROTECTED["thor_bridge_result"].read_text())
    if v027.get("development_gate_passed") is not False:
        raise RuntimeError(
            "expected preserved v0.27 formal DEV FAIL"
        )

    protected_before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("PERSISTENT_SELF_AUTOBIOGRAPHICAL_V1")
    print("device: cpu")
    print("goal: functional_self_continuity_not_robot_control")
    print("trainable_component: agency_self_model_only")
    print("historical_brain_training: false")
    print("working_memory_training: false")
    print("episodic_reader_training: false")
    print("planner_training: false")
    print("thor_training: false")
    print("self_train_ids: 0-15999")
    print("self_dev_ids: 20000-23999")
    print("self_test_ids: 30000-33999")
    print("self_test_touched: false")
    print("v027_formal_status_preserved: FAIL")

    train = make_dataset(TRAIN_IDS, "train")
    dev = make_dataset(DEV_IDS, "dev")

    model, train_metrics = train_model(train)
    save_self_model(
        model_path,
        model,
        metadata={
            "seed": SEED,
            "train_ids": [0, 15999],
            "dev_ids": [20000, 23999],
            "test_ids": [30000, 33999],
            "test_touched": False,
        },
    )
    model, _ = load_self_model(model_path, "cpu")
    dev_metrics = model_metrics(model, dev)

    causal_self_p, exogenous_self_p = canonical_agency_probe(model)
    dev_metrics[
        "canonical_self_caused_agency_probability"
    ] = causal_self_p
    dev_metrics[
        "canonical_exogenous_agency_probability"
    ] = exogenous_self_p

    continuity = run_continuity_assays(
        model, state_dir
    )
    autobiography = run_autobiography_causal_assays(model)
    endogenous = run_endogenous_goal_assays(model)
    attribution = run_agency_intervention_assays(model)

    # Save a canonical persistent self seed for the next milestone.
    seed_core = PersistentSelfCore(
        model,
        state=PersistentSelfState.create(SEED),
    )
    seed_core.add_commitment(
        GOAL_COMMITMENT_A, .65
    )
    save_self_bundle(
        self_bundle_path,
        seed_core,
        metadata={
            "purpose":
                "v0.28 persistent self seed; additive sidecar",
            "agency_self_model_sha256":
                sha256(model_path),
        },
    )

    checks = {
        "agency_accuracy":
            dev_metrics["agency_accuracy"] >= MIN_AGENCY_ACC,
        "confidence_transition":
            dev_metrics["confidence_mae"] <= MAX_CONF_MAE,
        "uncertainty_transition":
            dev_metrics["uncertainty_mae"] <= MAX_UNC_MAE,
        "canonical_self_attribution":
            causal_self_p >= MIN_CAUSAL_SELF_PROB,
        "canonical_world_attribution":
            exogenous_self_p <= MAX_EXOGENOUS_SELF_PROB,

        "identity_reload":
            continuity["identity_exact_after_reload"]
            >= MIN_CONTINUITY,
        "state_reload":
            continuity["self_state_exact_after_reload"]
            >= MIN_CONTINUITY,
        "autobiography_reload":
            continuity["autobiography_exact_after_reload"]
            >= MIN_CONTINUITY,
        "intention_resume":
            continuity["unresolved_intention_resume_rate"]
            >= MIN_CONTINUITY,
        "reset_control_does_not_resume":
            continuity["reset_control_resume_rate"] <= .05,

        "normal_autobiographical_reconstruction":
            autobiography[
                "normal_history_reconstructs_expected_goal"
            ] >= MIN_AUTOBIO_CAUSAL,
        "erase_history_changes_goal":
            autobiography[
                "erase_completion_changes_reconstructed_goal"
            ] >= MIN_AUTOBIO_CAUSAL,
        "donor_history_changes_goal":
            autobiography[
                "donor_history_changes_reconstructed_goal"
            ] >= MIN_AUTOBIO_CAUSAL,
        "identity_survives_history_intervention":
            autobiography[
                "identity_anchor_survives_history_donor_swap"
            ] >= MIN_AUTOBIO_CAUSAL,

        "endogenous_investigate":
            endogenous[
                "prediction_error_generates_investigate_goal"
            ] >= MIN_ENDOGENOUS,
        "prediction_error_ablation":
            endogenous[
                "prediction_error_zero_ablation_removes_investigate"
            ] >= MIN_ENDOGENOUS,
        "temporary_investigate":
            endogenous[
                "surprise_temporarily_interrupts_commitment"
            ] >= MIN_TEMPORARY_INVESTIGATE_RESUME,
        "commitment_resumption":
            endogenous[
                "commitment_resumes_after_surprise_resolves"
            ] >= MIN_TEMPORARY_INVESTIGATE_RESUME,

        "agency_intervention_goal":
            attribution["agency_intervention_changes_goal"]
            >= MIN_AGENCY_INTERVENTION,
        "agency_intervention_uncertainty":
            attribution[
                "world_attribution_increases_uncertainty_more"
            ] >= MIN_AGENCY_INTERVENTION,
        "agency_intervention_world_error":
            attribution[
                "world_attribution_increases_world_error_more"
            ] >= MIN_AGENCY_INTERVENTION,
    }

    passed = all(checks.values())

    protected_after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if protected_before != protected_after:
        raise RuntimeError(
            "protected cognitive artifact changed during v0.28"
        )

    result = {
        "format":
            "continuous_brain.persistent_self_autobiographical_v1",
        "development_gate_passed": bool(passed),
        "training": {
            "agency_self_model": True,
            "historical_brain": False,
            "working_memory": False,
            "episodic_reader": False,
            "planner": False,
            "thor": False,
        },
        "split": {
            "train": [0, 15999],
            "dev": [20000, 23999],
            "test": [30000, 33999],
            "test_touched": False,
        },
        "scope": {
            "claim":
                "functional persistent self-state, autobiographical continuity, agency attribution, and endogenous goal generation",
            "subjective_experience_claimed": False,
            "sentience_claimed": False,
            "consciousness_claimed": False,
            "robot_controller_milestone": False,
        },
        "train_metrics": train_metrics,
        "dev_model_metrics": dev_metrics,
        "continuity_metrics": continuity,
        "autobiography_causal_metrics": autobiography,
        "endogenous_goal_metrics": endogenous,
        "agency_intervention_metrics": attribution,
        "checks": checks,
        "protected_unchanged": True,
        "v027_formal_status_preserved": "FAIL",
        "v027_motor_polish_deferred": True,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTRAIN_MODEL")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nDEV_MODEL")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nCONTINUITY")
    for k, v in continuity.items():
        print(f"{k}: {v}")

    print("\nAUTOBIOGRAPHY_CAUSAL")
    for k, v in autobiography.items():
        print(f"{k}: {v}")

    print("\nENDOGENOUS_GOALS")
    for k, v in endogenous.items():
        print(f"{k}: {v}")

    print("\nAGENCY_INTERVENTION")
    for k, v in attribution.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("V027_FORMAL_STATUS_PRESERVED: FAIL")
    print("V027_MOTOR_POLISH_DEFERRED: true")
    print("SELF_TEST_TOUCHED: false")
    print(
        "PERSISTENT_SELF_AUTOBIOGRAPHICAL_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "PERSISTENT_SELF_AUTOBIOGRAPHICAL_RUN_COMPLETE"
    )
    print("result:", result_path)
    print("self_model:", model_path)
    print("persistent_self_seed:", self_bundle_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
