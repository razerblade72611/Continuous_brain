#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WORK_A,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    GOAL_NONE,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    load_self_model,
    save_self_bundle,
    sha256,
)
from train_evaluate_persistent_self_autobiographical_v1 import (
    make_dataset,
    model_metrics,
    canonical_agency_probe,
)

DEV_RESULT = Path(
    "state/persistent_self_autobiographical_v1/result.json"
)
SELF_MODEL = Path(
    "state/persistent_self_autobiographical_v1/agency_self_model.pt"
)
SELF_SEED = Path(
    "state/persistent_self_autobiographical_v1/persistent_self_seed.pt"
)

TEST_IDS = range(30000, 34000)

MIN_AGENCY_ACC = .97
MAX_CONF_MAE = .035
MAX_UNC_MAE = .035
MIN_CAUSAL_SELF_PROB = .85
MAX_EXOGENOUS_SELF_PROB = .15

MIN_CONTINUITY = 1.0
MIN_AUTOBIO_CAUSAL = 1.0
MIN_ENDOGENOUS = 1.0
MIN_AGENCY_INTERVENTION = 1.0

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
    "dev_result":
        DEV_RESULT,
    "self_model":
        SELF_MODEL,
    "self_seed":
        SELF_SEED,
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


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


def run_fresh_continuity(model, state_dir):
    identity_exact = []
    state_roundtrip = []
    memory_roundtrip = []
    resume_goal = []
    reset_control_resume = []

    for i in range(32):
        identity_seed = 130000 + i
        target = (
            GOAL_COMMITMENT_A
            if i % 2 == 0
            else GOAL_COMMITMENT_B
        )

        core = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(identity_seed),
        )
        core.add_commitment(target, 1.0)

        if target == GOAL_COMMITMENT_A:
            core.state.unresolved_b = .20
        else:
            core.state.unresolved_a = .20
        core.choose_goal()

        core.observe_event(
            action=ACTION_OBSERVE,
            intended_change=(0.0, 0.0),
            observed_change=(0.0, 0.0),
            action_success=1.0,
            prediction_error=0.0,
        )
        core.state.active_goal = target

        path = state_dir / "fresh_test_continuity_probe.pt"
        save_self_bundle(
            path,
            core,
            metadata={
                "test_identity_seed": identity_seed,
                "interrupted_mid_intention": True,
            },
        )

        loaded, _ = load_self_bundle(path, model)

        identity_exact.append(
            float(
                loaded.state.identity_token
                == core.state.identity_token
                and torch.equal(
                    loaded.state.identity_embedding,
                    core.state.identity_embedding,
                )
            )
        )
        state_roundtrip.append(
            float(state_exact(core.state, loaded.state))
        )
        memory_roundtrip.append(
            float(
                loaded.memory.to_list()
                == core.memory.to_list()
            )
        )
        resume_goal.append(
            float(loaded.choose_goal() == target)
        )

        reset = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(identity_seed),
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


def run_fresh_autobiography(model):
    normal = []
    erase = []
    donor = []
    identity_preserved = []

    for i in range(128):
        subject = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(140000 + i),
        )
        subject.state.unresolved_a = 1.0
        subject.state.unresolved_b = 1.0
        subject.choose_goal()

        completion = subject.record_completion(
            GOAL_COMMITMENT_A
        )
        normal_goal = (
            subject.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )
        normal.append(
            float(normal_goal == GOAL_COMMITMENT_B)
        )

        erased = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        erased.memory.erase_episode(
            completion.episode_id
        )
        erased_goal = (
            erased.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )
        erase.append(
            float(erased_goal != normal_goal)
        )

        donor_core = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(150000 + i),
        )
        donor_core.state.unresolved_a = 1.0
        donor_core.state.unresolved_b = 1.0
        donor_core.choose_goal()
        donor_completion = donor_core.record_completion(
            GOAL_COMMITMENT_B
        )

        swapped = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        token_before = swapped.state.identity_token
        emb_before = swapped.state.identity_embedding.clone()

        swapped.memory.replace_episode(
            completion.episode_id,
            donor_completion,
        )
        swapped_goal = (
            swapped.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )

        donor.append(
            float(
                swapped_goal == GOAL_COMMITMENT_A
                and swapped_goal != normal_goal
            )
        )
        identity_preserved.append(
            float(
                swapped.state.identity_token == token_before
                and torch.equal(
                    swapped.state.identity_embedding,
                    emb_before,
                )
            )
        )

    return {
        "normal_history_reconstructs_expected_goal":
            mean(normal),
        "erase_completion_changes_reconstructed_goal":
            mean(erase),
        "donor_history_changes_reconstructed_goal":
            mean(donor),
        "identity_anchor_survives_history_donor_swap":
            mean(identity_preserved),
    }


def run_fresh_endogenous(model):
    generated = []
    zero_ablation = []
    interrupt = []
    resume = []

    for i in range(128):
        s = PersistentSelfState.create(160000 + i)
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

        committed = PersistentSelfState.create(161000 + i)
        committed.unresolved_a = 1.0
        committed.unresolved_b = .25
        committed.uncertainty = .35
        committed.prediction_error_ema = .80

        cc = PersistentSelfCore(model, state=committed)
        interrupt.append(
            float(cc.choose_goal() == GOAL_INVESTIGATE)
        )

        cc.state.prediction_error_ema = .05
        cc.state.uncertainty = .25
        resume.append(
            float(cc.choose_goal() == GOAL_COMMITMENT_A)
        )

    return {
        "prediction_error_generates_investigate_goal":
            mean(generated),
        "prediction_error_zero_ablation_removes_investigate":
            mean(zero_ablation),
        "surprise_temporarily_interrupts_commitment":
            mean(interrupt),
        "commitment_resumes_after_surprise_resolves":
            mean(resume),
    }


def run_fresh_agency_intervention(model):
    goal_changes = []
    uncertainty_order = []
    world_error_order = []

    for i in range(128):
        s = PersistentSelfState.create(170000 + i)
        s.unresolved_a = 0.0
        s.unresolved_b = 0.0
        s.confidence = .55
        s.uncertainty = .30
        s.prediction_error_ema = .20

        self_attr = PersistentSelfCore(model, state=s)
        world_attr = PersistentSelfCore(model, state=s)

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
        uncertainty_order.append(
            float(
                w["next_uncertainty"]
                > a["next_uncertainty"]
            )
        )
        world_error_order.append(
            float(
                world_attr.state.prediction_error_ema
                > self_attr.state.prediction_error_ema
            )
        )

    return {
        "agency_intervention_changes_goal":
            mean(goal_changes),
        "world_attribution_increases_uncertainty_more":
            mean(uncertainty_order),
        "world_attribution_increases_world_error_more":
            mean(world_error_order),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/persistent_self_autobiographical_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    dev = json.loads(DEV_RESULT.read_text())
    if dev.get("development_gate_passed") is not True:
        raise RuntimeError(
            "v0.28.0 DEV PASS required before TEST"
        )
    if dev.get("split", {}).get("test_touched") is not False:
        raise RuntimeError(
            "v0.28 self TEST is not sealed"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("self_test_ids: 30000-33999")
    print("self_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("fresh_continuity_identity_seeds: 130000-130031")
    print("fresh_autobiography_subject_seeds: 140000-140127")
    print("fresh_autobiography_donor_seeds: 150000-150127")
    print("fresh_endogenous_goal_seeds: 160000+")
    print("fresh_agency_intervention_seeds: 170000-170127")
    print("v027_formal_status_preserved: FAIL")

    model, _ = load_self_model(SELF_MODEL, "cpu")

    test_data = make_dataset(
        TEST_IDS,
        "test",
    )
    test_model = model_metrics(
        model,
        test_data,
    )

    self_p, world_p = canonical_agency_probe(model)
    test_model[
        "canonical_self_caused_agency_probability"
    ] = self_p
    test_model[
        "canonical_exogenous_agency_probability"
    ] = world_p

    continuity = run_fresh_continuity(
        model,
        state_dir,
    )
    autobiography = run_fresh_autobiography(model)
    endogenous = run_fresh_endogenous(model)
    attribution = run_fresh_agency_intervention(model)

    checks = {
        "agency_accuracy":
            test_model["agency_accuracy"] >= MIN_AGENCY_ACC,
        "confidence_transition":
            test_model["confidence_mae"] <= MAX_CONF_MAE,
        "uncertainty_transition":
            test_model["uncertainty_mae"] <= MAX_UNC_MAE,
        "canonical_self_attribution":
            self_p >= MIN_CAUSAL_SELF_PROB,
        "canonical_world_attribution":
            world_p <= MAX_EXOGENOUS_SELF_PROB,

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
            ] >= MIN_ENDOGENOUS,
        "commitment_resumption":
            endogenous[
                "commitment_resumes_after_surprise_resolves"
            ] >= MIN_ENDOGENOUS,

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

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during frozen self TEST"
        )

    result = {
        "format":
            "continuous_brain.persistent_self_autobiographical_test_v1",
        "test_admission_passed": bool(passed),
        "training_performed": False,
        "split": {
            "self_test": [30000, 33999],
            "self_test_used": True,
            "self_test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "fresh_causal_seed_ranges": {
            "continuity": [130000, 130031],
            "autobiography_subject": [140000, 140127],
            "autobiography_donor": [150000, 150127],
            "endogenous_base": [160000, 161127],
            "agency_intervention": [170000, 170127],
        },
        "test_model_metrics": test_model,
        "continuity_metrics": continuity,
        "autobiography_causal_metrics": autobiography,
        "endogenous_goal_metrics": endogenous,
        "agency_intervention_metrics": attribution,
        "checks": checks,
        "protected_unchanged": True,
        "v027_formal_status_preserved": "FAIL",
        "scope": {
            "functional_self_continuity_claimed": True,
            "subjective_experience_claimed": False,
            "sentience_claimed": False,
            "consciousness_claimed": False,
        },
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTEST_MODEL")
    for k, v in test_model.items():
        print(f"{k}: {v}")

    print("\nFRESH_CONTINUITY")
    for k, v in continuity.items():
        print(f"{k}: {v}")

    print("\nFRESH_AUTOBIOGRAPHY_CAUSAL")
    for k, v in autobiography.items():
        print(f"{k}: {v}")

    print("\nFRESH_ENDOGENOUS_GOALS")
    for k, v in endogenous.items():
        print(f"{k}: {v}")

    print("\nFRESH_AGENCY_INTERVENTION")
    for k, v in attribution.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("V027_FORMAL_STATUS_PRESERVED: FAIL")
    print("SELF_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
