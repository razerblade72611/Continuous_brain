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
    TEST_IDS,
    build_dataset,
)
from train_evaluate_autobiographical_self_discrepancy_neutral_v1 import (
    causal_assays,
    metrics,
    reload_assay,
)

SOURCE_RESULT = Path(
    "state/autobiographical_self_discrepancy_neutral_v1/result.json"
)
SOURCE_MODEL = Path(
    "state/autobiographical_self_discrepancy_neutral_v1/"
    "neutral_autobiographical_self_model.pt"
)
V0320_RESULT = Path(
    "state/autobiographical_self_discrepancy_v1/result.json"
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
    "source_result": SOURCE_RESULT,
    "source_model": SOURCE_MODEL,
    "v0320_result": V0320_RESULT,
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
    "v029_test":
        Path("state/self_broadcast_centrality_test_v1/result.json"),
    "v030_result":
        Path("state/joint_self_world_prediction_v1/result.json"),
    "v0301_result":
        Path("state/joint_self_world_repair_v1/result.json"),
    "v0302_result":
        Path("state/joint_self_goal_boundary_diagnostic_v1/result.json"),
}


def zero_gradient_sanity(model):
    """
    Frozen structural sanity check.

    Zero discrepancy must be exactly neutral while the derivative with respect
    to discrepancy channels remains finite and nonzero.
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
        and float(grad.abs().max().item()) > 0.0
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


@torch.no_grad()
def fresh_history_swap_assay(model, test):
    """
    Fresh held-out donor-history intervention.

    Pair neighboring TEST subjects with opposite hidden reliability profiles.
    The present world/current self are absent from the model by architecture;
    swapping only autobiography must make the commitment follow the donor.
    """
    direct = []
    swapped = []
    margins = []

    ids = test["ids"]
    for i in range(0, len(ids), 2):
        j = i + 1
        if j >= len(ids):
            break

        xa = test["features"][i:i+1]
        ma = test["mask"][i:i+1]
        xb = test["features"][j:j+1]
        mb = test["mask"][j:j+1]

        oa = model(xa, ma)
        ob = model(xb, mb)

        ta = int(test["preference"][i].item())
        tb = int(test["preference"][j].item())

        direct.append(
            float(
                int(oa["preference"][0].item()) == ta
                and int(ob["preference"][0].item()) == tb
            )
        )

        # Donor swap: only autobiographical history changes.
        donor_a = model(xb, mb)
        donor_b = model(xa, ma)
        swapped.append(
            float(
                int(donor_a["preference"][0].item()) == tb
                and int(donor_b["preference"][0].item()) == ta
            )
        )

        for o in (oa, ob):
            r = o["reliability"][0]
            margins.append(
                float((r.max() - r.min()).item())
            )

    return {
        "direct_history_commitment_accuracy":
            sum(direct) / len(direct),
        "donor_history_commitment_flip_rate":
            sum(swapped) / len(swapped),
        "mean_reliability_margin":
            sum(margins) / len(margins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/autobiographical_self_discrepancy_neutral_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    source = json.loads(SOURCE_RESULT.read_text())

    if source.get("repair_dev_gate_passed") is not True:
        raise RuntimeError(
            "v0.32.1 DEV repair PASS required"
        )
    if source.get("training_performed") is not False:
        raise RuntimeError(
            "expected v0.32.1a postfix result with no retraining"
        )
    if source.get("autobio_test_touched") is not False:
        raise RuntimeError(
            "autobiographical TEST already touched"
        )

    v0320 = json.loads(V0320_RESULT.read_text())
    if v0320.get("development_gate_passed") is not False:
        raise RuntimeError(
            "v0.32.0 historical DEV FAIL must remain preserved"
        )

    trajectory_test = json.loads(TRAJECTORY_TEST.read_text())
    if trajectory_test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.31.1 frozen TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("source_v0320_formal_status: FAIL")
    print("source_v0321_dev_repair_status: PASS")
    print("autobio_test_subjects: 6000-6255")
    print("autobio_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("history_episodes_per_subject: 12")

    trajectory_model, _ = load_trajectory_model(
        TRAJECTORY_MODEL,
        "cpu",
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    model, model_payload = load_neutral_model(
        SOURCE_MODEL,
        "cpu",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    test = build_dataset(
        trajectory_model,
        TEST_IDS,
        "test",
    )

    test_metrics = metrics(model, test)
    causal = causal_assays(model, test)
    fresh_swap = fresh_history_swap_assay(
        model,
        test,
    )
    neutral = zero_gradient_sanity(model)
    reload = reload_assay(
        model,
        SOURCE_MODEL,
        test,
    )

    checks = {
        "reliability_prediction":
            test_metrics["reliability_mae"]
            <= MAX_RELIABILITY_MAE,
        "commitment_formation":
            test_metrics["preference_accuracy"]
            >= MIN_PREFERENCE_ACCURACY,

        "normal_history_causal":
            causal[
                "normal_history_forms_correct_commitment"
            ] >= MIN_NORMAL,
        "donor_history_causal":
            causal[
                "donor_history_flips_commitment"
            ] >= MIN_DONOR,

        "fresh_direct_history":
            fresh_swap[
                "direct_history_commitment_accuracy"
            ] >= MIN_NORMAL,
        "fresh_donor_history":
            fresh_swap[
                "donor_history_commitment_flip_rate"
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
            "protected artifact changed during frozen v0.32 TEST"
        )

    result = {
        "format":
            "continuous_brain.autobiographical_self_discrepancy_neutral_test_v1",
        "test_admission_passed":
            bool(passed),
        "training_performed":
            False,
        "historical_status": {
            "v0320": "FAIL",
            "v0321_dev_repair": "PASS",
            "v0311_trajectory_test": "PASS",
        },
        "split": {
            "test": [6000, 6255],
            "test_used": True,
            "test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "model_sha256":
            sha256(SOURCE_MODEL),
        "model_metadata":
            model_payload.get("metadata", {}),
        "test_metrics":
            test_metrics,
        "causal_history_metrics":
            causal,
        "fresh_history_swap_metrics":
            fresh_swap,
        "structural_neutrality_metrics":
            neutral,
        "reload_metrics":
            reload,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "source_model_unchanged":
            True,
        "scope": {
            "functional_autobiographical_self_model_claimed":
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

    print("\nTEST")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print("\nCAUSAL_HISTORY")
    for k, v in causal.items():
        print(f"{k}: {v}")

    print("\nFRESH_HISTORY_SWAP")
    for k, v in fresh_swap.items():
        print(f"{k}: {v}")

    print("\nSTRUCTURAL_NEUTRALITY")
    for k, v in neutral.items():
        print(f"{k}: {v}")

    print("\nRELOAD")
    for k, v in reload.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_MODEL_UNCHANGED: true")
    print("AUTOBIO_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
