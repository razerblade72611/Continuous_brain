#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    RECORD_FEATURE_DIM,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from train_evaluate_autobiographical_self_discrepancy_v1 import (
    DEV_IDS,
    TRAIN_IDS,
    build_dataset,
)
from train_evaluate_autobiographical_self_discrepancy_neutral_v1 import (
    causal_assays,
    metrics,
    reload_assay,
)

SOURCE_V0320_RESULT = Path(
    "state/autobiographical_self_discrepancy_v1/result.json"
)
NEUTRAL_MODEL = Path(
    "state/autobiographical_self_discrepancy_neutral_v1/"
    "neutral_autobiographical_self_model.pt"
)
TRAJECTORY_MODEL = Path(
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

MAX_RELIABILITY_MAE = .10
MIN_PREFERENCE_ACCURACY = .95
MIN_NORMAL = .95
MIN_DONOR = .95
MIN_STRUCTURAL_NEUTRAL = 1.0
MIN_DISCREPANCY_ABLATION_NEUTRAL = .95
MIN_RELOAD = 1.0

PROTECTED = {
    "v0320_result": SOURCE_V0320_RESULT,
    "neutral_model": NEUTRAL_MODEL,
    "trajectory_model": TRAJECTORY_MODEL,
    "trajectory_test": TRAJECTORY_TEST,
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
}


def zero_gradient_sanity_fixed(model):
    """
    Corrected version of the v0.32.1 evaluator check.

    IMPORTANT: intentionally NOT decorated with torch.no_grad().
    """
    model.eval()

    x = torch.zeros(
        1,
        HISTORY_LEN,
        RECORD_FEATURE_DIM,
        requires_grad=True,
    )
    mask = torch.ones(
        1,
        HISTORY_LEN,
        dtype=torch.bool,
    )

    with torch.no_grad():
        for t in range(HISTORY_LEN):
            x[0, t, 0 if t % 2 == 0 else 1] = 1.0

    out = model(x, mask)

    neutral = torch.equal(
        out["reliability"][0],
        torch.tensor([.5, .5]),
    )

    diff = (
        out["reliability"][0, 0]
        - out["reliability"][0, 1]
    )
    diff.backward()

    grad = x.grad[..., 2:]
    alive = bool(
        grad is not None
        and torch.isfinite(grad).all()
        and float(
            grad.abs().max().item()
        ) > 0.0
    )

    return {
        "zero_discrepancy_exact_neutral":
            bool(neutral),
        "zero_discrepancy_gradient_alive":
            bool(alive),
        "max_abs_discrepancy_input_gradient":
            float(
                grad.abs().max().item()
            )
            if grad is not None
            else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/autobiographical_self_discrepancy_neutral_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    source = json.loads(
        SOURCE_V0320_RESULT.read_text()
    )
    if source.get("development_gate_passed") is not False:
        raise RuntimeError(
            "v0.32.0 historical DEV FAIL must be preserved"
        )

    failed = sorted(
        k
        for k, v in source["checks"].items()
        if not bool(v)
    )
    if failed != [
        "expected_actual_discrepancy_load_bearing"
    ]:
        raise RuntimeError(
            f"unexpected v0.32.0 failures: {failed}"
        )

    trajectory_test = json.loads(
        TRAJECTORY_TEST.read_text()
    )
    if trajectory_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.31.1 TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_POSTFIX_V1")
    print("device: cpu")
    print("training_performed: false")
    print("reason: evaluator_autograd_context_bug")
    print("source_v0320_formal_status: FAIL")
    print("v0321_training_completed_before_crash: true")
    print("reuses_saved_v0321_model: true")
    print("development_window_reused: subjects 3000-3255")
    print("new_development_window_consumed: false")
    print("autobio_test_subjects: 6000-6255")
    print("autobio_test_touched: false")

    trajectory_model, _ = load_trajectory_model(
        TRAJECTORY_MODEL,
        "cpu",
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    model, model_payload = load_neutral_model(
        NEUTRAL_MODEL,
        "cpu",
    )
    model.eval()

    # Reconstruct exactly the original TRAIN/DEV datasets. No fitting occurs.
    train = build_dataset(
        trajectory_model,
        TRAIN_IDS,
        "postfix_train",
    )
    dev = build_dataset(
        trajectory_model,
        DEV_IDS,
        "postfix_dev",
    )

    train_m = metrics(model, train)
    dev_m = metrics(model, dev)
    causal = causal_assays(model, dev)
    neutral = zero_gradient_sanity_fixed(model)
    reload = reload_assay(
        model,
        NEUTRAL_MODEL,
        dev,
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
            ] >= MIN_NORMAL,
        "donor_history_causal":
            causal[
                "donor_history_flips_commitment"
            ] >= MIN_DONOR,
        "empty_history_exact_neutral":
            causal[
                "erased_history_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "discrepancy_ablation_exact_neutral":
            causal[
                "discrepancy_ablation_exact_neutral"
            ] >= MIN_DISCREPANCY_ABLATION_NEUTRAL,
        "discrepancy_ablation_bit_exact_neutral":
            causal[
                "discrepancy_ablation_bit_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "primary_labels_alone_exact_neutral":
            causal[
                "primary_labels_alone_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "zero_discrepancy_structural_neutral":
            neutral[
                "zero_discrepancy_exact_neutral"
            ],
        "zero_discrepancy_gradient_alive":
            neutral[
                "zero_discrepancy_gradient_alive"
            ],
        "model_reload_reliability_exact":
            reload[
                "reliability_exact_after_model_reload"
            ] >= MIN_RELOAD,
        "model_reload_preference_exact":
            reload[
                "preference_exact_after_model_reload"
            ] >= MIN_RELOAD,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during postfix evaluation"
        )

    result = {
        "format":
            "continuous_brain.autobiographical_self_discrepancy_neutral_postfix_v1",
        "repair_dev_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "infrastructure_repair": {
            "bug":
                "zero_gradient_sanity was incorrectly wrapped in torch.no_grad",
            "scientific_gate_reached_in_failed_wrapper":
                False,
            "saved_v0321_model_reused":
                True,
            "model_retrained":
                False,
        },
        "source_v0320_formal_status":
            "FAIL",
        "v0321_model_sha256":
            sha256(NEUTRAL_MODEL),
        "v0321_model_metadata":
            model_payload.get("metadata", {}),
        "split": {
            "train": [0, 1023],
            "reused_dev": [3000, 3255],
            "test": [6000, 6255],
            "test_touched": False,
        },
        "train_metrics":
            train_m,
        "repaired_dev_metrics":
            dev_m,
        "causal_metrics":
            causal,
        "neutrality_metrics":
            neutral,
        "reload_metrics":
            reload,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "neutral_model_unchanged":
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

    print("\nTRAIN_REEVALUATED")
    for k, v in train_m.items():
        print(f"{k}: {v}")

    print("\nDEV_REEVALUATED")
    for k, v in dev_m.items():
        print(f"{k}: {v}")

    print("\nCAUSAL")
    for k, v in causal.items():
        print(f"{k}: {v}")

    print("\nSTRUCTURAL_NEUTRALITY")
    for k, v in neutral.items():
        print(f"{k}: {v}")

    print("\nRELOAD")
    for k, v in reload.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("TRAINING_PERFORMED: false")
    print("NEUTRAL_MODEL_UNCHANGED: true")
    print("PROTECTED_UNCHANGED: true")
    print("AUTOBIO_TEST_TOUCHED: false")
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_POSTFIX_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
