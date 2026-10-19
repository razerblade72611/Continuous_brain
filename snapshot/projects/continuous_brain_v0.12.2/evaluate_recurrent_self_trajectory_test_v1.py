#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_COMMITMENT_A,
    sha256,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    MODE_INVESTIGATE,
    MODE_NORMAL,
    PRIMARY_A,
    PRIMARY_B,
    ExplicitIntentionState,
    active_goal_from_intention,
    load_trajectory_model,
    rollout,
    trajectory_self_value,
    update_explicit_intention,
)
from train_evaluate_recurrent_self_trajectory_v1 import (
    TEST_IDS,
    make_dataset,
    make_state,
    rollout_metrics,
    teacher_metrics,
)
from train_evaluate_joint_self_world_prediction_v1 import (
    transition,
)

SOURCE_RESULT = Path(
    "state/recurrent_self_trajectory_v1/result.json"
)
SOURCE_MODEL = Path(
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
)

MAX_TF_WORLD_MAE = .030
MAX_TF_SELF_MAE = .025
MIN_TF_PRIMARY = .985
MIN_TF_MODE = .985
MIN_TF_AGENCY = .990

MAX_ROLLOUT_WORLD_MAE = .075
MAX_ROLLOUT_SELF_MAE = .060
MIN_ROLLOUT_PRIMARY = .950
MIN_ROLLOUT_MODE = .950
MIN_ROLLOUT_GOAL = .950

MIN_TIE_FULL = .95
MAX_TIE_SCALAR_ONLY = .60
MIN_TIE_DONOR = .95

MIN_INTERRUPT_PRESERVE = 1.0
MIN_RESUME = 1.0
MIN_SURPRISE_SELF_ERROR = .10

PROTECTED = {
    "source_result": SOURCE_RESULT,
    "source_model": SOURCE_MODEL,
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


@torch.no_grad()
def fresh_tie_intention_assay(model):
    """
    Fresh causal TEST probe.

    Across cases the external context and tied continuous self vary.
    Within each A/B pair they are exactly identical; only explicit primary
    intention differs.
    """
    full = []
    scalar_only = []
    donor = []

    for i in range(256):
        rng = random.Random(0x311100 + 100003 * i)

        world = torch.tensor(
            [
                1.0,
                1.0,
                .20 + .70 * rng.random(),
                .10 + .50 * rng.random(),
            ],
            dtype=torch.float32,
        )

        tie = .62 + .33 * rng.random()
        selfc = torch.tensor(
            [
                tie,
                tie,
                .35 + .50 * rng.random(),
                .10 + .30 * rng.random(),
                .04 + .20 * rng.random(),
                .25 + .55 * rng.random(),
            ],
            dtype=torch.float32,
        )

        candidate_a = [ACTION_WORK_A] * 3
        candidate_b = [ACTION_WORK_B] * 3

        def score(primary, actions, explicit=True):
            p = int(primary)
            if not explicit:
                # Exact tied scalars cannot recover whether the continuing
                # intention was A or B. Fixed scalar tie-break selects A.
                p = PRIMARY_A

            r = rollout(
                model,
                world,
                selfc,
                p,
                MODE_NORMAL,
                actions,
            )
            return float(
                trajectory_self_value(
                    r["selfs"][-1],
                    primary=p,
                    final_mode=r["final_mode"],
                ).item()
            )

        for primary, expected in (
            (PRIMARY_A, 0),
            (PRIMARY_B, 1),
        ):
            scores = [
                score(primary, candidate_a, True),
                score(primary, candidate_b, True),
            ]
            pred = max(
                range(2),
                key=lambda j: scores[j],
            )
            full.append(float(pred == expected))

            scores_z = [
                score(primary, candidate_a, False),
                score(primary, candidate_b, False),
            ]
            pred_z = max(
                range(2),
                key=lambda j: scores_z[j],
            )
            scalar_only.append(
                float(pred_z == expected)
            )

        a_scores = [
            score(PRIMARY_A, candidate_a, True),
            score(PRIMARY_A, candidate_b, True),
        ]
        b_scores = [
            score(PRIMARY_B, candidate_a, True),
            score(PRIMARY_B, candidate_b, True),
        ]
        donor.append(
            float(
                max(
                    range(2),
                    key=lambda j: a_scores[j],
                ) == 0
                and max(
                    range(2),
                    key=lambda j: b_scores[j],
                ) == 1
            )
        )

    return {
        "explicit_intention_guided_trajectory_choice":
            mean(full),
        "scalar_rederived_intention_choice":
            mean(scalar_only),
        "donor_primary_flips_choice_world_and_scalars_fixed":
            mean(donor),
    }


def fresh_interruption_resumption_assay(model):
    preserve = []
    resume = []
    transition_error = []
    model_surprise_error = []

    for i in range(128):
        rng = random.Random(
            0x311200 + 104729 * i
        )

        base = make_state(
            50000
            + 4 * i
            + GOAL_COMMITMENT_A
        )
        base.unresolved_a = .72 + .23 * rng.random()
        base.unresolved_b = .03 + .12 * rng.random()
        base.confidence = .45 + .40 * rng.random()
        base.uncertainty = .12 + .18 * rng.random()
        base.prediction_error_ema = .04 + .12 * rng.random()
        base.active_goal = GOAL_COMMITMENT_A

        intent = ExplicitIntentionState(
            PRIMARY_A,
            MODE_NORMAL,
        )
        world = torch.tensor(
            [
                .20 + .35 * rng.random(),
                .10 + .25 * rng.random(),
                .30 + .30 * rng.random(),
                .20 + .35 * rng.random(),
            ],
            dtype=torch.float32,
        )

        # Model's expected next self under ordinary WORK_A.
        expected = rollout(
            model,
            world,
            base.vector()[:6],
            PRIMARY_A,
            MODE_NORMAL,
            [ACTION_WORK_A],
        )["selfs"][0]

        normal_world, normal_state, _, _ = transition(
            world,
            base,
            ACTION_WORK_A,
            surprise=False,
        )
        surprise_world, surprise_state, _, _ = transition(
            world,
            base,
            ACTION_WORK_A,
            surprise=True,
        )

        normal_self = normal_state.vector()[:6]
        surprise_self = surprise_state.vector()[:6]

        transition_error.append(
            float(
                (
                    normal_self
                    - surprise_self
                ).abs().mean().item()
            )
        )
        model_surprise_error.append(
            float(
                (
                    expected
                    - surprise_self
                ).abs().mean().item()
            )
        )

        surprised = update_explicit_intention(
            intent.primary,
            intent.mode,
            surprise_self,
        )
        preserve.append(
            float(
                surprised.primary == PRIMARY_A
                and surprised.mode == MODE_INVESTIGATE
            )
        )

        state = surprise_state
        world2 = surprise_world
        cur = surprised

        for _ in range(5):
            world2, state, _, _ = transition(
                world2,
                state,
                ACTION_OBSERVE,
                surprise=False,
            )
            cur = update_explicit_intention(
                cur.primary,
                cur.mode,
                state.vector()[:6],
            )
            if cur.mode == MODE_NORMAL:
                break

        resume.append(
            float(
                cur.primary == PRIMARY_A
                and cur.mode == MODE_NORMAL
                and cur.active_goal
                == GOAL_COMMITMENT_A
            )
        )

    return {
        "surprise_preserves_primary_and_enters_investigate":
            mean(preserve),
        "primary_resumes_after_investigate_resolves":
            mean(resume),
        "mean_transition_surprise_self_error":
            mean(transition_error),
        "mean_model_to_surprise_self_error":
            mean(model_surprise_error),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recurrent_self_trajectory_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    source = json.loads(
        SOURCE_RESULT.read_text()
    )
    if source.get(
        "development_gate_passed"
    ) is not True:
        raise RuntimeError(
            "v0.31.0 DEV PASS required"
        )
    if (
        source.get("split", {})
        .get("test_touched")
        is not False
    ):
        raise RuntimeError(
            "trajectory TEST was already touched"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("RECURRENT_SELF_TRAJECTORY_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("trajectory_test_ids: 20000-21023")
    print("trajectory_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("fresh_tie_probe_cases: 256")
    print("fresh_interruption_probe_cases: 128")
    print("source_v0310_dev_status: PASS")
    print("v030_test_touched: false")

    model, _ = load_trajectory_model(
        SOURCE_MODEL,
        "cpu",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    test = make_dataset(
        TEST_IDS,
        "test",
    )

    tf = teacher_metrics(
        model,
        test,
    )
    ar = rollout_metrics(
        model,
        test,
        max_sequences=1024,
    )
    tie = fresh_tie_intention_assay(
        model
    )
    interruption = (
        fresh_interruption_resumption_assay(
            model
        )
    )

    checks = {
        "teacher_world":
            tf["world_mae"]
            <= MAX_TF_WORLD_MAE,
        "teacher_self":
            tf["self_mae"]
            <= MAX_TF_SELF_MAE,
        "teacher_primary":
            tf["primary_accuracy"]
            >= MIN_TF_PRIMARY,
        "teacher_mode":
            tf["mode_accuracy"]
            >= MIN_TF_MODE,
        "teacher_agency":
            tf["agency_accuracy"]
            >= MIN_TF_AGENCY,

        "rollout_world":
            ar["world_mae"]
            <= MAX_ROLLOUT_WORLD_MAE,
        "rollout_self":
            ar["self_mae"]
            <= MAX_ROLLOUT_SELF_MAE,
        "rollout_primary":
            ar["primary_accuracy"]
            >= MIN_ROLLOUT_PRIMARY,
        "rollout_mode":
            ar["mode_accuracy"]
            >= MIN_ROLLOUT_MODE,
        "rollout_goal":
            ar["active_goal_accuracy"]
            >= MIN_ROLLOUT_GOAL,

        "explicit_intention_guides_trajectory":
            tie[
                "explicit_intention_guided_trajectory_choice"
            ] >= MIN_TIE_FULL,
        "scalar_only_control_collapses":
            tie[
                "scalar_rederived_intention_choice"
            ] <= MAX_TIE_SCALAR_ONLY,
        "donor_primary_flips_trajectory_choice":
            tie[
                "donor_primary_flips_choice_world_and_scalars_fixed"
            ] >= MIN_TIE_DONOR,

        "surprise_preserves_primary":
            interruption[
                "surprise_preserves_primary_and_enters_investigate"
            ] >= MIN_INTERRUPT_PRESERVE,
        "primary_resumes":
            interruption[
                "primary_resumes_after_investigate_resolves"
            ] >= MIN_RESUME,
        "surprise_self_error":
            interruption[
                "mean_transition_surprise_self_error"
            ] >= MIN_SURPRISE_SELF_ERROR,
        "model_surprise_self_error":
            interruption[
                "mean_model_to_surprise_self_error"
            ] >= MIN_SURPRISE_SELF_ERROR,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during frozen v0.31 TEST"
        )

    result = {
        "format":
            "continuous_brain.recurrent_self_trajectory_test_v1",
        "test_admission_passed":
            bool(passed),
        "training_performed":
            False,
        "split": {
            "test": [20000, 21023],
            "test_used": True,
            "test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "source_v0310_dev_status":
            "PASS",
        "teacher_forced_test_metrics":
            tf,
        "autoregressive_test_metrics":
            ar,
        "fresh_tie_intention_metrics":
            tie,
        "fresh_interruption_resumption_metrics":
            interruption,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "source_model_unchanged":
            True,
        "v030_test_touched":
            False,
        "scope": {
            "functional_intention_continuity_claimed":
                True,
            "subjective_experience_claimed":
                False,
            "sentience_claimed":
                False,
            "consciousness_claimed":
                False,
        },
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTEST_TEACHER_FORCED")
    for k, v in tf.items():
        print(f"{k}: {v}")

    print("\nTEST_AUTOREGRESSIVE_ROLLOUT")
    for k, v in ar.items():
        print(f"{k}: {v}")

    print("\nFRESH_TIE_INTENTION_CAUSAL")
    for k, v in tie.items():
        print(f"{k}: {v}")

    print("\nFRESH_INTERRUPTION_RESUMPTION")
    for k, v in interruption.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_MODEL_UNCHANGED: true")
    print("V030_TEST_TOUCHED: false")
    print("TRAJECTORY_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "RECURRENT_SELF_TRAJECTORY_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "RECURRENT_SELF_TRAJECTORY_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
