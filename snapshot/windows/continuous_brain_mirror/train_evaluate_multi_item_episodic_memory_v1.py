#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    build_substrate,
    collect_decision_snapshot,
    sha256,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    load_goal_decoder,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_DIFFERENT,
    QUERY_FIRST,
    QUERY_NAMES,
    QUERY_SAME,
    QUERY_SECOND,
    EpisodicLateQueryReader,
    TwoSlotMemory,
    load_reader,
    save_reader,
)

GOAL_DECODER_PATH = Path(
    "state/memory_guided_goal_option_v1/goal_decoder.pt"
)

TRAIN_IDS = list(range(0, 1024))
DEV_IDS = list(range(1024, 1312))
TEST_IDS = list(range(1312, 1600))

SEED = 23001
EPOCHS = 120
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
ATTENTION_LOSS_WEIGHT = .35

# Frozen DEV gate.
MIN_OVERALL = .95
MIN_PER_QUERY = .94
MIN_PAIR_ALL4 = .90
MIN_ATTENTION_ROUTE = .95
MIN_SWAP_FIRST_SECOND = .95
MIN_SWAP_RELATION = .95
MIN_DONOR_TARGET = .95
MIN_DONOR_NONTARGET = .95
MIN_ZERO_NONTARGET = .95
MAX_ZERO_TARGET = .70
MIN_UNEQUAL_ORDER_REVERSAL = .95
MIN_SINGLE_ITEM_GOAL_DECODER = .95


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def cue_value(cue: str) -> int:
    return 0 if cue == "left" else 1


@torch.no_grad()
def build_recall_bank(base, cfg, admitted, ids, label):
    bank = {}
    for n, cid in enumerate(ids, start=1):
        for cue in ("left", "right"):
            snap = collect_decision_snapshot(
                base,
                cfg,
                admitted,
                int(cid),
                cue,
                device="cpu",
            )
            bank[(int(cid), cue)] = snap["recall"].clone().float()
        if n == 1 or n % 64 == 0 or n == len(ids):
            print(
                f"{label}_recall_conditions={n}/{len(ids)}"
            )
    return bank


def partner_ids(ids):
    n = len(ids)
    out = {}
    for i, cid in enumerate(ids):
        j = (i * 37 + 17) % n
        if j == i:
            j = (j + 1) % n
        out[cid] = ids[j]
    return out


def make_examples(bank, ids):
    partners = partner_ids(ids)
    examples = []

    for cid in ids:
        cid2 = partners[cid]
        for cue1 in ("left", "right"):
            for cue2 in ("left", "right"):
                slot1 = bank[(cid, cue1)]
                slot2 = bank[(cid2, cue2)]
                v1 = cue_value(cue1)
                v2 = cue_value(cue2)

                targets = {
                    QUERY_FIRST: v1,
                    QUERY_SECOND: v2,
                    QUERY_SAME: int(v1 == v2),
                    QUERY_DIFFERENT: int(v1 != v2),
                }

                for query in (
                    QUERY_FIRST,
                    QUERY_SECOND,
                    QUERY_SAME,
                    QUERY_DIFFERENT,
                ):
                    examples.append(
                        {
                            "condition1": int(cid),
                            "condition2": int(cid2),
                            "cue1": cue1,
                            "cue2": cue2,
                            "slot1": slot1,
                            "slot2": slot2,
                            "query": int(query),
                            "target": int(targets[query]),
                        }
                    )
    return examples


def stack_examples(examples):
    slots = torch.cat(
        [
            torch.cat(
                (e["slot1"].unsqueeze(1), e["slot2"].unsqueeze(1)),
                dim=1,
            )
            for e in examples
        ],
        dim=0,
    )
    valid = torch.ones(
        slots.shape[0], 2, dtype=torch.bool
    )
    query = torch.tensor(
        [e["query"] for e in examples],
        dtype=torch.long,
    )
    target = torch.tensor(
        [e["target"] for e in examples],
        dtype=torch.long,
    )
    return slots, valid, query, target


def train_reader(examples):
    torch.manual_seed(SEED)
    random.seed(SEED)

    slots, valid, query, target = stack_examples(examples)

    reader = EpisodicLateQueryReader(
        content_dim=int(slots.shape[-1]),
        hidden_dim=96,
        query_dim=32,
    )
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(SEED)
    n = int(slots.shape[0])

    metrics = {}

    for epoch in range(1, EPOCHS + 1):
        reader.train()
        order = torch.randperm(n, generator=generator)
        total_loss = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            sb = slots[idx]
            vb = valid[idx]
            qb = query[idx]
            tb = target[idx]

            optimizer.zero_grad(set_to_none=True)
            out = reader(sb, vb, qb)

            answer_loss = F.cross_entropy(
                out["logits"], tb
            )

            route_mask = (
                (qb == QUERY_FIRST)
                | (qb == QUERY_SECOND)
            )
            if route_mask.any():
                route_target = (
                    qb[route_mask] == QUERY_SECOND
                ).long()
                route_scores = torch.log(
                    out["attention"][route_mask]
                    .clamp_min(1e-8)
                )
                route_loss = F.nll_loss(
                    route_scores,
                    route_target,
                )
            else:
                route_loss = (
                    out["logits"].sum() * 0.0
                )

            loss = (
                answer_loss
                + ATTENTION_LOSS_WEIGHT * route_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)

        reader.eval()
        with torch.no_grad():
            out = reader(slots, valid, query)
            pred = out["logits"].argmax(dim=-1)
            acc = float(
                (pred == target).float().mean().item()
            )

            route_mask = (
                (query == QUERY_FIRST)
                | (query == QUERY_SECOND)
            )
            route_target = (
                query[route_mask] == QUERY_SECOND
            ).long()
            route_pred = out["attention"][
                route_mask
            ].argmax(dim=-1)
            route_acc = float(
                (route_pred == route_target)
                .float().mean().item()
            )

        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(1, seen),
            "train_accuracy": acc,
            "train_attention_route_accuracy":
                route_acc,
        }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            print(
                f"episodic_reader_epoch="
                f"{epoch:03d}/{EPOCHS} "
                f"loss={metrics['loss']:.6f} "
                f"acc={acc:.4f} "
                f"route={route_acc:.4f}"
            )

        if (
            epoch >= 30
            and acc >= .9995
            and route_acc >= .9995
        ):
            print(
                f"episodic_reader_early_stop_epoch="
                f"{epoch}"
            )
            break

    reader.eval()
    return reader, metrics


@torch.no_grad()
def frozen_single_item_goal_accuracy(
    decoder,
    bank,
):
    correct = []
    for (cid, cue), recall in bank.items():
        pred = int(
            torch.argmax(
                decoder(recall),
                dim=-1,
            )[0].item()
        )
        correct.append(
            float(pred == cue_value(cue))
        )
    return mean(correct)


def group_examples(examples):
    groups = {}
    for i, e in enumerate(examples):
        key = (
            e["condition1"],
            e["condition2"],
            e["cue1"],
            e["cue2"],
        )
        groups.setdefault(key, []).append(i)
    return groups


@torch.no_grad()
def evaluate_reader(
    reader,
    examples,
    bank,
):
    slots, valid, query, target = stack_examples(
        examples
    )
    out = reader(slots, valid, query)
    pred = out["logits"].argmax(dim=-1)
    correct = pred == target

    per_query = {}
    for q, name in enumerate(QUERY_NAMES):
        mask = query == q
        per_query[name] = float(
            correct[mask].float().mean().item()
        )

    route_mask = (
        (query == QUERY_FIRST)
        | (query == QUERY_SECOND)
    )
    route_target = (
        query[route_mask] == QUERY_SECOND
    ).long()
    route_pred = out["attention"][
        route_mask
    ].argmax(dim=-1)
    attention_route = float(
        (route_pred == route_target)
        .float().mean().item()
    )

    groups = group_examples(examples)
    pair_all4 = []
    for _, indices in groups.items():
        pair_all4.append(
            float(
                all(bool(correct[i].item()) for i in indices)
            )
        )

    # Slot swap intervention.
    swapped = slots[:, [1, 0]].clone()
    out_swap = reader(swapped, valid, query)
    pred_swap = out_swap["logits"].argmax(dim=-1)

    swap_target = target.clone()
    for i, e in enumerate(examples):
        if e["query"] == QUERY_FIRST:
            swap_target[i] = cue_value(e["cue2"])
        elif e["query"] == QUERY_SECOND:
            swap_target[i] = cue_value(e["cue1"])
        # SAME and DIFFERENT targets are invariant.

    swap_correct = pred_swap == swap_target
    fs_mask = (
        (query == QUERY_FIRST)
        | (query == QUERY_SECOND)
    )
    rel_mask = (
        (query == QUERY_SAME)
        | (query == QUERY_DIFFERENT)
    )
    swap_first_second = float(
        swap_correct[fs_mask].float().mean().item()
    )
    swap_relation = float(
        swap_correct[rel_mask].float().mean().item()
    )

    unequal = torch.tensor(
        [
            int(cue_value(e["cue1"]) != cue_value(e["cue2"]))
            for e in examples
        ],
        dtype=torch.bool,
    )
    unequal_fs = unequal & fs_mask
    original_pred = pred
    reversed_expected = swap_target
    order_reversal = float(
        (
            pred_swap[unequal_fs]
            == reversed_expected[unequal_fs]
        ).float().mean().item()
    )

    # Targeted zero interventions.
    zero0 = slots.clone()
    zero0[:, 0] = 0
    valid0 = valid.clone()
    valid0[:, 0] = False
    p0 = reader(zero0, valid0, query)[
        "logits"
    ].argmax(dim=-1)

    zero1 = slots.clone()
    zero1[:, 1] = 0
    valid1 = valid.clone()
    valid1[:, 1] = False
    p1 = reader(zero1, valid1, query)[
        "logits"
    ].argmax(dim=-1)

    first_mask = query == QUERY_FIRST
    second_mask = query == QUERY_SECOND

    zero_slot0_targeted = float(
        (p0[first_mask] == target[first_mask])
        .float().mean().item()
    )
    zero_slot0_nontarget = float(
        (p0[second_mask] == target[second_mask])
        .float().mean().item()
    )
    zero_slot1_targeted = float(
        (p1[second_mask] == target[second_mask])
        .float().mean().item()
    )
    zero_slot1_nontarget = float(
        (p1[first_mask] == target[first_mask])
        .float().mean().item()
    )

    # Donor replacement: replace one slot with an opposite-cue
    # memory from another condition.  The targeted query should
    # follow donor content; the other query should remain unchanged.
    donor_slots0 = slots.clone()
    donor_slots1 = slots.clone()
    donor_target0 = target.clone()
    donor_target1 = target.clone()

    ids = sorted({e["condition1"] for e in examples})
    donor_map = partner_ids(ids)

    for i, e in enumerate(examples):
        donor_cid = donor_map[e["condition1"]]
        donor_cue0 = (
            "right" if e["cue1"] == "left" else "left"
        )
        donor_cue1 = (
            "right" if e["cue2"] == "left" else "left"
        )
        donor_slots0[i, 0] = bank[
            (donor_cid, donor_cue0)
        ][0]
        donor_slots1[i, 1] = bank[
            (donor_cid, donor_cue1)
        ][0]

        if e["query"] == QUERY_FIRST:
            donor_target0[i] = cue_value(donor_cue0)
        elif e["query"] == QUERY_SECOND:
            donor_target1[i] = cue_value(donor_cue1)

    dp0 = reader(
        donor_slots0,
        valid,
        query,
    )["logits"].argmax(dim=-1)
    dp1 = reader(
        donor_slots1,
        valid,
        query,
    )["logits"].argmax(dim=-1)

    donor_slot0_target = float(
        (dp0[first_mask] == donor_target0[first_mask])
        .float().mean().item()
    )
    donor_slot0_nontarget = float(
        (dp0[second_mask] == target[second_mask])
        .float().mean().item()
    )
    donor_slot1_target = float(
        (dp1[second_mask] == donor_target1[second_mask])
        .float().mean().item()
    )
    donor_slot1_nontarget = float(
        (dp1[first_mask] == target[first_mask])
        .float().mean().item()
    )

    return {
        "overall_accuracy": float(
            correct.float().mean().item()
        ),
        "per_query_accuracy": per_query,
        "pair_all_four_correct": mean(pair_all4),
        "attention_route_accuracy":
            attention_route,
        "slot_swap_first_second_accuracy":
            swap_first_second,
        "slot_swap_relation_invariance_accuracy":
            swap_relation,
        "unequal_pair_order_reversal_accuracy":
            order_reversal,
        "zero_slot0_targeted_accuracy":
            zero_slot0_targeted,
        "zero_slot0_nontarget_accuracy":
            zero_slot0_nontarget,
        "zero_slot1_targeted_accuracy":
            zero_slot1_targeted,
        "zero_slot1_nontarget_accuracy":
            zero_slot1_nontarget,
        "donor_slot0_target_accuracy":
            donor_slot0_target,
        "donor_slot0_nontarget_accuracy":
            donor_slot0_nontarget,
        "donor_slot1_target_accuracy":
            donor_slot1_target,
        "donor_slot1_nontarget_accuracy":
            donor_slot1_nontarget,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/multi_item_episodic_memory_v1",
    )
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / "result.json"
    reader_path = state / "episodic_reader.pt"

    base_hash_before = sha256(BASE_CHECKPOINT)
    memory_hash_before = sha256(MEMORY_CHECKPOINT)
    goal_hash_before = sha256(GOAL_DECODER_PATH)

    print("MULTI_ITEM_EPISODIC_MEMORY_V1")
    print("device: cpu")
    print("historical_64d_writer_training: false")
    print("canonical_recall_training: false")
    print("base_training: false")
    print("goal_decoder_training: false")
    print("trainable_component: late_query_reader_only")
    print("slot_write_semantics: append_ordered_canonical_recall")
    print("query_available_during_write: false")
    print("composition_train_ids: 0-1023")
    print("composition_dev_ids: 1024-1311")
    print("composition_test_ids: 1312-1599")
    print("composition_test_touched: false")
    print(
        "environmental_writer_generalization_claimed: false"
    )

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(
        GOAL_DECODER_PATH,
        "cpu",
    )

    train_bank = build_recall_bank(
        base,
        cfg,
        admitted,
        TRAIN_IDS,
        "train",
    )
    dev_bank = build_recall_bank(
        base,
        cfg,
        admitted,
        DEV_IDS,
        "dev",
    )

    train_examples = make_examples(
        train_bank,
        TRAIN_IDS,
    )
    dev_examples = make_examples(
        dev_bank,
        DEV_IDS,
    )

    print(
        "train_examples:",
        len(train_examples),
    )
    print(
        "dev_examples:",
        len(dev_examples),
    )

    reader, train_metrics = train_reader(
        train_examples
    )
    save_reader(
        reader_path,
        reader,
        {
            "seed": SEED,
            "train_ids": [0, 1023],
            "dev_ids": [1024, 1311],
            "test_ids": [1312, 1599],
            "query_available_during_write":
                False,
            "source_content":
                "frozen_admitted_canonical_recall128",
        },
    )
    reader, _ = load_reader(
        reader_path,
        "cpu",
    )

    dev_metrics = evaluate_reader(
        reader,
        dev_examples,
        dev_bank,
    )
    single_item_goal_acc = (
        frozen_single_item_goal_accuracy(
            decoder,
            dev_bank,
        )
    )
    dev_metrics[
        "frozen_single_item_goal_decoder_accuracy"
    ] = single_item_goal_acc

    checks = {
        "overall_accuracy":
            dev_metrics["overall_accuracy"]
            >= MIN_OVERALL,
        "first_accuracy":
            dev_metrics["per_query_accuracy"]["first"]
            >= MIN_PER_QUERY,
        "second_accuracy":
            dev_metrics["per_query_accuracy"]["second"]
            >= MIN_PER_QUERY,
        "same_accuracy":
            dev_metrics["per_query_accuracy"]["same"]
            >= MIN_PER_QUERY,
        "different_accuracy":
            dev_metrics["per_query_accuracy"]["different"]
            >= MIN_PER_QUERY,
        "pair_all_four":
            dev_metrics["pair_all_four_correct"]
            >= MIN_PAIR_ALL4,
        "attention_route":
            dev_metrics["attention_route_accuracy"]
            >= MIN_ATTENTION_ROUTE,
        "slot_swap_first_second":
            dev_metrics[
                "slot_swap_first_second_accuracy"
            ] >= MIN_SWAP_FIRST_SECOND,
        "slot_swap_relation_invariance":
            dev_metrics[
                "slot_swap_relation_invariance_accuracy"
            ] >= MIN_SWAP_RELATION,
        "unequal_order_reversal":
            dev_metrics[
                "unequal_pair_order_reversal_accuracy"
            ] >= MIN_UNEQUAL_ORDER_REVERSAL,
        "donor_slot0_target":
            dev_metrics[
                "donor_slot0_target_accuracy"
            ] >= MIN_DONOR_TARGET,
        "donor_slot0_nontarget":
            dev_metrics[
                "donor_slot0_nontarget_accuracy"
            ] >= MIN_DONOR_NONTARGET,
        "donor_slot1_target":
            dev_metrics[
                "donor_slot1_target_accuracy"
            ] >= MIN_DONOR_TARGET,
        "donor_slot1_nontarget":
            dev_metrics[
                "donor_slot1_nontarget_accuracy"
            ] >= MIN_DONOR_NONTARGET,
        "zero_slot0_target_degrades":
            dev_metrics[
                "zero_slot0_targeted_accuracy"
            ] <= MAX_ZERO_TARGET,
        "zero_slot0_nontarget_preserved":
            dev_metrics[
                "zero_slot0_nontarget_accuracy"
            ] >= MIN_ZERO_NONTARGET,
        "zero_slot1_target_degrades":
            dev_metrics[
                "zero_slot1_targeted_accuracy"
            ] <= MAX_ZERO_TARGET,
        "zero_slot1_nontarget_preserved":
            dev_metrics[
                "zero_slot1_nontarget_accuracy"
            ] >= MIN_ZERO_NONTARGET,
        "single_item_content_preserved":
            single_item_goal_acc
            >= MIN_SINGLE_ITEM_GOAL_DECODER,
    }

    passed = all(checks.values())

    if sha256(BASE_CHECKPOINT) != base_hash_before:
        raise RuntimeError("base checkpoint changed")
    if sha256(MEMORY_CHECKPOINT) != memory_hash_before:
        raise RuntimeError("memory checkpoint changed")
    if sha256(GOAL_DECODER_PATH) != goal_hash_before:
        raise RuntimeError("goal decoder changed")

    result = {
        "format":
            "continuous_brain.multi_item_episodic_memory_v1",
        "development_gate_passed":
            bool(passed),
        "scope": {
            "claim":
                "two-item order-sensitive episodic working-memory mechanism",
            "general_episodic_memory_claimed":
                False,
            "environmental_writer_generalization_claimed":
                False,
        },
        "training": {
            "late_query_reader": True,
            "historical_64d_writer": False,
            "canonical_recall": False,
            "base": False,
            "goal_decoder": False,
        },
        "split": {
            "composition_train": [0, 1023],
            "composition_dev": [1024, 1311],
            "composition_test": [1312, 1599],
            "composition_test_touched": False,
            "source_condition_family":
                "historical TRAIN only",
        },
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "checks": checks,
        "protected_unchanged": True,
        "v0222_milestone_untouched": True,
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
    print(
        "overall_accuracy:",
        dev_metrics["overall_accuracy"],
    )
    print(
        "per_query_accuracy:",
        dev_metrics["per_query_accuracy"],
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
        print(f"{key}: {dev_metrics[key]}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0222_MILESTONE_UNTOUCHED: true")
    print("COMPOSITION_TEST_TOUCHED: false")
    print(
        "MULTI_ITEM_EPISODIC_MEMORY_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print("MULTI_ITEM_EPISODIC_MEMORY_RUN_COMPLETE")
    print("result:", result_path)
    print("reader:", reader_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
