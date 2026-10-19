#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    EndogenousGoalArbiter,
    PersistentSelfState,
    sha256,
)
from continuous_brain.joint_self_world_prediction_v1 import (
    load_joint_model,
)
from continuous_brain.joint_self_world_repair_v1 import (
    load_goal_head,
)
from train_evaluate_joint_self_world_prediction_v1 import (
    DEV_IDS,
    make_dataset,
)

SOURCE_MODEL = Path(
    "state/joint_self_world_prediction_v1/joint_self_world_model.pt"
)
SOURCE_RESULT = Path(
    "state/joint_self_world_prediction_v1/result.json"
)
REPAIR_HEAD = Path(
    "state/joint_self_world_repair_v1/future_goal_repair_head.pt"
)
REPAIR_RESULT = Path(
    "state/joint_self_world_repair_v1/result.json"
)
V029_TEST = Path(
    "state/self_broadcast_centrality_test_v1/result.json"
)

PROTECTED = {
    "source_model": SOURCE_MODEL,
    "source_result": SOURCE_RESULT,
    "repair_head": REPAIR_HEAD,
    "repair_result": REPAIR_RESULT,
    "v029_test": V029_TEST,
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
}


def mean(x):
    return float(x.float().mean().item()) if x.numel() else 0.0


def frozen_arbiter_from_true_self(cont, goal_target):
    """
    Sanity check: rebuild PersistentSelfState from TRUE next continuous self,
    then run the exact frozen v0.28 EndogenousGoalArbiter.
    """
    preds = []
    arbiter = EndogenousGoalArbiter()

    for i in range(cont.shape[0]):
        s = PersistentSelfState.create(3100000 + i)
        s.unresolved_a = float(cont[i, 0].item())
        s.unresolved_b = float(cont[i, 1].item())
        s.confidence = float(cont[i, 2].item())
        s.uncertainty = float(cont[i, 3].item())
        s.prediction_error_ema = float(cont[i, 4].item())
        s.agency_ema = float(cont[i, 5].item())
        s.active_goal = int(arbiter.choose(s))
        preds.append(s.active_goal)

    pred = torch.tensor(preds, dtype=torch.long)
    return pred, mean(pred == goal_target)


def predicted_self_rule(
    cont,
    *,
    none_threshold: float,
):
    """
    Apply the v0.28 goal rule to PREDICTED continuous future-self fields.

    The only diagnostic variation is a NONE tolerance because sigmoid model
    outputs cannot be mathematically equal to zero.  Thresholds are reported
    as diagnostics only; no threshold is admitted or selected here.
    """
    a = cont[:, 0]
    b = cont[:, 1]
    uncertainty = cont[:, 3]
    pe = cont[:, 4]

    pred = torch.full(
        (cont.shape[0],),
        GOAL_NONE,
        dtype=torch.long,
    )

    investigate = (
        (pe >= 0.55)
        | (uncertainty >= 0.82)
    )
    pred[investigate] = GOAL_INVESTIGATE

    remaining = ~investigate
    none = (
        remaining
        & (torch.maximum(a, b) <= float(none_threshold))
    )
    pred[none] = GOAL_NONE

    active = remaining & (~none)
    choose_a = active & (a >= b)
    choose_b = active & (~choose_a)
    pred[choose_a] = GOAL_COMMITMENT_A
    pred[choose_b] = GOAL_COMMITMENT_B

    return pred


def class_metrics(pred, target):
    out = {}
    names = {
        GOAL_NONE: "none",
        GOAL_COMMITMENT_A: "commitment_a",
        GOAL_COMMITMENT_B: "commitment_b",
        GOAL_INVESTIGATE: "investigate",
    }
    for cls, name in names.items():
        mask = target == cls
        out[f"{name}_count"] = int(mask.sum().item())
        out[f"{name}_accuracy"] = (
            mean(pred[mask] == target[mask])
            if bool(mask.any())
            else 0.0
        )
    return out


def boundary_analysis(pred_cont, target):
    a = pred_cont[:, 0]
    b = pred_cont[:, 1]
    uncertainty = pred_cont[:, 3]
    pe = pred_cont[:, 4]

    # Use the source learned goal head only to define source errors later;
    # these values characterize how close predicted continuous self is to the
    # actual v0.28 arbitration surfaces.
    investigation_margin = torch.minimum(
        (pe - 0.55).abs(),
        (uncertainty - 0.82).abs(),
    )
    commitment_margin = (a - b).abs()
    none_magnitude = torch.maximum(a, b)

    return {
        "investigation_margin_mean":
            float(investigation_margin.mean().item()),
        "fraction_within_0p01_investigate_boundary":
            mean(investigation_margin <= .01),
        "fraction_within_0p02_investigate_boundary":
            mean(investigation_margin <= .02),
        "fraction_within_0p05_investigate_boundary":
            mean(investigation_margin <= .05),
        "commitment_ab_margin_mean":
            float(commitment_margin.mean().item()),
        "fraction_within_0p01_a_b_boundary":
            mean(commitment_margin <= .01),
        "fraction_within_0p02_a_b_boundary":
            mean(commitment_margin <= .02),
        "fraction_within_0p05_a_b_boundary":
            mean(commitment_margin <= .05),
        "true_none_predicted_commitment_magnitude_mean":
            float(
                none_magnitude[target == GOAL_NONE]
                .mean().item()
            )
            if bool((target == GOAL_NONE).any())
            else 0.0,
        "true_none_predicted_commitment_magnitude_max":
            float(
                none_magnitude[target == GOAL_NONE]
                .max().item()
            )
            if bool((target == GOAL_NONE).any())
            else 0.0,
    }


def error_boundary_analysis(
    pred,
    pred_cont,
    target,
):
    err = pred != target
    if not bool(err.any()):
        return {
            "errors": 0,
            "error_rate": 0.0,
        }

    a = pred_cont[:, 0]
    b = pred_cont[:, 1]
    uncertainty = pred_cont[:, 3]
    pe = pred_cont[:, 4]

    investigation_margin = torch.minimum(
        (pe - 0.55).abs(),
        (uncertainty - 0.82).abs(),
    )
    commitment_margin = (a - b).abs()
    none_magnitude = torch.maximum(a, b)

    return {
        "errors": int(err.sum().item()),
        "error_rate": mean(err),
        "error_investigation_margin_mean":
            float(investigation_margin[err].mean().item()),
        "error_fraction_within_0p01_investigate_boundary":
            mean(investigation_margin[err] <= .01),
        "error_fraction_within_0p02_investigate_boundary":
            mean(investigation_margin[err] <= .02),
        "error_fraction_within_0p05_investigate_boundary":
            mean(investigation_margin[err] <= .05),
        "error_commitment_ab_margin_mean":
            float(commitment_margin[err].mean().item()),
        "error_fraction_within_0p01_a_b_boundary":
            mean(commitment_margin[err] <= .01),
        "error_fraction_within_0p02_a_b_boundary":
            mean(commitment_margin[err] <= .02),
        "error_fraction_within_0p05_a_b_boundary":
            mean(commitment_margin[err] <= .05),
        "error_none_magnitude_mean":
            float(none_magnitude[err].mean().item()),
    }


def confusion(pred, target):
    m = torch.zeros(4, 4, dtype=torch.long)
    for t, p in zip(target.tolist(), pred.tolist()):
        m[int(t), int(p)] += 1
    return m.tolist()


def main():
    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    source = json.loads(SOURCE_RESULT.read_text())
    repair = json.loads(REPAIR_RESULT.read_text())
    v029 = json.loads(V029_TEST.read_text())

    if source.get("development_gate_passed") is not False:
        raise RuntimeError("expected v0.30.0 formal DEV FAIL")
    if repair.get("repair_dev_gate_passed") is not False:
        raise RuntimeError("expected v0.30.1 formal DEV repair FAIL")
    if source.get("split", {}).get("test_touched") is not False:
        raise RuntimeError("v0.30 TEST was already touched")
    if v029.get("split", {}).get("test_spent") is not True:
        raise RuntimeError("v0.29 spent TEST status not preserved")

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("JOINT_SELF_GOAL_BOUNDARY_DIAGNOSTIC_V1")
    print("device: cpu")
    print("training_performed: false")
    print("development_window_reused: 40000-44999")
    print("new_development_window_consumed: false")
    print("joint_test_ids: 60000-64999")
    print("joint_test_touched: false")
    print("source_v0300_status: FAIL")
    print("v0301_repair_status: FAIL")

    model, _ = load_joint_model(SOURCE_MODEL, "cpu")
    goal_head, _ = load_goal_head(REPAIR_HEAD, "cpu")
    model.eval()
    goal_head.eval()

    dev = make_dataset(DEV_IDS, "diagnostic_dev")

    with torch.no_grad():
        out = model(
            dev["world"],
            dev["self"],
            dev["action"],
        )
        source_pred = (
            out["next_goal_logits"]
            .argmax(dim=-1)
        )
        repair_pred = (
            goal_head(out)
            .argmax(dim=-1)
        )
        pred_cont = out[
            "next_self_continuous"
        ]

    target = dev["next_goal"]

    source_acc = mean(source_pred == target)
    repair_acc = mean(repair_pred == target)

    oracle_rule_pred, oracle_rule_acc = (
        frozen_arbiter_from_true_self(
            dev["next_self_cont"],
            target,
        )
    )

    none_thresholds = [
        1e-6,
        .005,
        .01,
        .02,
        .05,
        .10,
    ]
    rule_results = {}
    for threshold in none_thresholds:
        pred = predicted_self_rule(
            pred_cont,
            none_threshold=threshold,
        )
        key = (
            "1e-6"
            if threshold == 1e-6
            else f"{threshold:.3f}"
        )
        rule_results[key] = {
            "accuracy":
                mean(pred == target),
            "class_metrics":
                class_metrics(pred, target),
            "confusion":
                confusion(pred, target),
        }

    source_class = class_metrics(
        source_pred,
        target,
    )
    repair_class = class_metrics(
        repair_pred,
        target,
    )

    source_errors = error_boundary_analysis(
        source_pred,
        pred_cont,
        target,
    )
    repair_errors = error_boundary_analysis(
        repair_pred,
        pred_cont,
        target,
    )
    boundaries = boundary_analysis(
        pred_cont,
        target,
    )

    # Diagnose where the two learned heads disagree.
    disagreement = source_pred != repair_pred
    disagreement_count = int(
        disagreement.sum().item()
    )
    if disagreement_count:
        source_wins = int(
            (
                disagreement
                & (source_pred == target)
                & (repair_pred != target)
            ).sum().item()
        )
        repair_wins = int(
            (
                disagreement
                & (repair_pred == target)
                & (source_pred != target)
            ).sum().item()
        )
    else:
        source_wins = 0
        repair_wins = 0

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during diagnostic"
        )

    # Diagnostic classification, not an admission gate.
    # The model's true-self arbiter must be exact by construction.  We then
    # report whether errors cluster near arbitration boundaries.
    source_near = (
        source_errors.get(
            "error_fraction_within_0p05_investigate_boundary",
            0.0,
        )
        + source_errors.get(
            "error_fraction_within_0p05_a_b_boundary",
            0.0,
        )
    )

    if oracle_rule_acc < .999:
        finding = "TARGET_ARBITER_INCONSISTENCY"
    elif source_near >= .50:
        finding = "GOAL_ERRORS_CONCENTRATE_NEAR_SELF_ARBITRATION_BOUNDARIES"
    else:
        finding = "GOAL_ERRORS_NOT_EXPLAINED_BY_SIMPLE_BOUNDARY_PROXIMITY"

    result = {
        "format":
            "continuous_brain.joint_self_goal_boundary_diagnostic_v1",
        "training_performed":
            False,
        "development_window_reused":
            [40000, 44999],
        "test":
            [60000, 64999],
        "test_touched":
            False,
        "source_goal_accuracy":
            source_acc,
        "repair_goal_accuracy":
            repair_acc,
        "true_self_exact_arbiter_accuracy":
            oracle_rule_acc,
        "source_class_metrics":
            source_class,
        "repair_class_metrics":
            repair_class,
        "predicted_self_rule_threshold_diagnostics":
            rule_results,
        "predicted_self_boundary_metrics":
            boundaries,
        "source_error_boundary_metrics":
            source_errors,
        "repair_error_boundary_metrics":
            repair_errors,
        "learned_head_disagreement": {
            "count": disagreement_count,
            "source_wins": source_wins,
            "repair_wins": repair_wins,
        },
        "finding":
            finding,
        "protected_unchanged":
            True,
        "joint_test_touched":
            False,
    }

    out_path = Path(
        "state/joint_self_goal_boundary_diagnostic_v1/result.json"
    )
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nLEARNED_HEADS")
    print("source_goal_accuracy:", source_acc)
    print("repair_goal_accuracy:", repair_acc)
    print(
        "learned_head_disagreement_count:",
        disagreement_count,
    )
    print("source_wins_disagreements:", source_wins)
    print("repair_wins_disagreements:", repair_wins)

    print("\nTRUE_SELF_ARBITER_SANITY")
    print(
        "true_self_exact_arbiter_accuracy:",
        oracle_rule_acc,
    )

    print("\nSOURCE_CLASS_ACCURACY")
    for k, v in source_class.items():
        print(f"{k}: {v}")

    print("\nREPAIR_CLASS_ACCURACY")
    for k, v in repair_class.items():
        print(f"{k}: {v}")

    print("\nPREDICTED_SELF_RULE_DIAGNOSTICS")
    for threshold, row in rule_results.items():
        print(
            f"none_threshold={threshold} "
            f"accuracy={row['accuracy']}"
        )

    print("\nBOUNDARIES")
    for k, v in boundaries.items():
        print(f"{k}: {v}")

    print("\nSOURCE_ERROR_BOUNDARIES")
    for k, v in source_errors.items():
        print(f"{k}: {v}")

    print("\nREPAIR_ERROR_BOUNDARIES")
    for k, v in repair_errors.items():
        print(f"{k}: {v}")

    print("\nFINDING")
    print("finding:", finding)

    print("PROTECTED_UNCHANGED: true")
    print("JOINT_TEST_TOUCHED: false")
    print(
        "JOINT_SELF_GOAL_BOUNDARY_DIAGNOSTIC_COMPLETE"
    )
    print("result:", out_path)


if __name__ == "__main__":
    main()
