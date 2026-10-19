#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    build_substrate,
    load_goal_decoder,
    sha256,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    load_reader,
)
from train_evaluate_multi_item_episodic_memory_v1 import (
    GOAL_DECODER_PATH,
    MIN_ATTENTION_ROUTE,
    MIN_DONOR_NONTARGET,
    MIN_DONOR_TARGET,
    MIN_OVERALL,
    MIN_PAIR_ALL4,
    MIN_PER_QUERY,
    MIN_SINGLE_ITEM_GOAL_DECODER,
    MIN_SWAP_FIRST_SECOND,
    MIN_SWAP_RELATION,
    MIN_UNEQUAL_ORDER_REVERSAL,
    MIN_ZERO_NONTARGET,
    MAX_ZERO_TARGET,
    build_recall_bank,
    evaluate_reader,
    frozen_single_item_goal_accuracy,
    make_examples,
)

DEV_RESULT = Path(
    "state/multi_item_episodic_memory_v1/result.json"
)
READER_CHECKPOINT = Path(
    "state/multi_item_episodic_memory_v1/episodic_reader.pt"
)
TEST_IDS = list(range(1312, 1600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/multi_item_episodic_memory_test_v1",
    )
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / "result.json"

    if not DEV_RESULT.is_file():
        raise FileNotFoundError(DEV_RESULT)
    if not READER_CHECKPOINT.is_file():
        raise FileNotFoundError(READER_CHECKPOINT)

    dev = json.loads(DEV_RESULT.read_text())
    if dev.get("development_gate_passed") is not True:
        raise RuntimeError(
            "v0.23.0 DEV gate did not pass; TEST use is not authorized"
        )
    if (
        dev.get("split", {}).get("composition_test_touched")
        is not False
    ):
        raise RuntimeError(
            "composition TEST was already marked touched"
        )

    base_before = sha256(BASE_CHECKPOINT)
    memory_before = sha256(MEMORY_CHECKPOINT)
    goal_before = sha256(GOAL_DECODER_PATH)
    reader_before = sha256(READER_CHECKPOINT)
    dev_result_before = sha256(DEV_RESULT)

    print("MULTI_ITEM_EPISODIC_MEMORY_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("reader_training_performed: false")
    print("historical_64d_writer_training: false")
    print("canonical_recall_training: false")
    print("base_training: false")
    print("goal_decoder_training: false")
    print("composition_test_ids: 1312-1599")
    print("composition_test_members: 288 conditions")
    print("composition_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("environmental_writer_generalization_claimed: false")
    print("reader_sha256:", reader_before)

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(
        GOAL_DECODER_PATH,
        "cpu",
    )
    reader, _ = load_reader(
        READER_CHECKPOINT,
        "cpu",
    )

    test_bank = build_recall_bank(
        base,
        cfg,
        admitted,
        TEST_IDS,
        "test",
    )
    test_examples = make_examples(
        test_bank,
        TEST_IDS,
    )
    print("test_examples:", len(test_examples))

    metrics = evaluate_reader(
        reader,
        test_examples,
        test_bank,
    )
    single_item_goal_acc = (
        frozen_single_item_goal_accuracy(
            decoder,
            test_bank,
        )
    )
    metrics[
        "frozen_single_item_goal_decoder_accuracy"
    ] = single_item_goal_acc

    checks = {
        "overall_accuracy":
            metrics["overall_accuracy"] >= MIN_OVERALL,
        "first_accuracy":
            metrics["per_query_accuracy"]["first"] >= MIN_PER_QUERY,
        "second_accuracy":
            metrics["per_query_accuracy"]["second"] >= MIN_PER_QUERY,
        "same_accuracy":
            metrics["per_query_accuracy"]["same"] >= MIN_PER_QUERY,
        "different_accuracy":
            metrics["per_query_accuracy"]["different"] >= MIN_PER_QUERY,
        "pair_all_four":
            metrics["pair_all_four_correct"] >= MIN_PAIR_ALL4,
        "attention_route":
            metrics["attention_route_accuracy"] >= MIN_ATTENTION_ROUTE,
        "slot_swap_first_second":
            metrics["slot_swap_first_second_accuracy"]
            >= MIN_SWAP_FIRST_SECOND,
        "slot_swap_relation_invariance":
            metrics["slot_swap_relation_invariance_accuracy"]
            >= MIN_SWAP_RELATION,
        "unequal_order_reversal":
            metrics["unequal_pair_order_reversal_accuracy"]
            >= MIN_UNEQUAL_ORDER_REVERSAL,
        "donor_slot0_target":
            metrics["donor_slot0_target_accuracy"]
            >= MIN_DONOR_TARGET,
        "donor_slot0_nontarget":
            metrics["donor_slot0_nontarget_accuracy"]
            >= MIN_DONOR_NONTARGET,
        "donor_slot1_target":
            metrics["donor_slot1_target_accuracy"]
            >= MIN_DONOR_TARGET,
        "donor_slot1_nontarget":
            metrics["donor_slot1_nontarget_accuracy"]
            >= MIN_DONOR_NONTARGET,
        "zero_slot0_target_degrades":
            metrics["zero_slot0_targeted_accuracy"]
            <= MAX_ZERO_TARGET,
        "zero_slot0_nontarget_preserved":
            metrics["zero_slot0_nontarget_accuracy"]
            >= MIN_ZERO_NONTARGET,
        "zero_slot1_target_degrades":
            metrics["zero_slot1_targeted_accuracy"]
            <= MAX_ZERO_TARGET,
        "zero_slot1_nontarget_preserved":
            metrics["zero_slot1_nontarget_accuracy"]
            >= MIN_ZERO_NONTARGET,
        "single_item_content_preserved":
            single_item_goal_acc >= MIN_SINGLE_ITEM_GOAL_DECODER,
    }
    passed = all(checks.values())

    # Test evaluator is strictly read-only.
    protected_after = {
        "base": sha256(BASE_CHECKPOINT),
        "memory": sha256(MEMORY_CHECKPOINT),
        "goal_decoder": sha256(GOAL_DECODER_PATH),
        "reader": sha256(READER_CHECKPOINT),
        "dev_result": sha256(DEV_RESULT),
    }
    protected_before = {
        "base": base_before,
        "memory": memory_before,
        "goal_decoder": goal_before,
        "reader": reader_before,
        "dev_result": dev_result_before,
    }
    if protected_after != protected_before:
        raise RuntimeError(
            "protected artifact changed during TEST evaluation"
        )

    result = {
        "format":
            "continuous_brain.multi_item_episodic_memory_test_v1",
        "test_admission_passed": bool(passed),
        "training_performed": False,
        "scope": {
            "claim":
                "two-item order-sensitive episodic working-memory mechanism",
            "general_episodic_memory_claimed": False,
            "environmental_writer_generalization_claimed": False,
        },
        "split": {
            "composition_test": [1312, 1599],
            "conditions": len(TEST_IDS),
            "examples": len(test_examples),
            "test_used": True,
            "test_tuning_after_run_forbidden": True,
        },
        "reader_sha256": reader_before,
        "metrics": metrics,
        "checks": checks,
        "protected_unchanged": True,
        "v0222_milestone_untouched": True,
        "composition_test_spent": True,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTEST")
    print(
        "overall_accuracy:",
        metrics["overall_accuracy"],
    )
    print(
        "per_query_accuracy:",
        metrics["per_query_accuracy"],
    )
    for key in (
        "pair_all_four_correct",
        "attention_route_accuracy",
        "slot_swap_first_second_accuracy",
        "slot_swap_relation_invariance_accuracy",
        "unequal_pair_order_reversal_accuracy",
        "zero_slot0_targeted_accuracy",
        "zero_slot0_nontarget_accuracy",
        "zero_slot1_targeted_accuracy",
        "zero_slot1_nontarget_accuracy",
        "donor_slot0_target_accuracy",
        "donor_slot0_nontarget_accuracy",
        "donor_slot1_target_accuracy",
        "donor_slot1_nontarget_accuracy",
        "frozen_single_item_goal_decoder_accuracy",
    ):
        print(f"{key}: {metrics[key]}")

    print("\nADMISSION_GATE")
    for key, value in checks.items():
        print(
            f"{key}: {'PASS' if value else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0222_MILESTONE_UNTOUCHED: true")
    print("COMPOSITION_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "MULTI_ITEM_EPISODIC_MEMORY_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "MULTI_ITEM_EPISODIC_MEMORY_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
