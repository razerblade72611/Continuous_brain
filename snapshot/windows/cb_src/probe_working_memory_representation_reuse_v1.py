#!/usr/bin/env python3
"""
Frozen-writer representation-reuse assay for the winning capacity-ungated
Continuous Brain working-memory architecture.

Question
--------
Can the same 64-D memory state support a brand-new reader that did not exist
during writer/recall training, on conditions withheld from that reader?

Important scope
---------------
The writer itself has seen the TRAIN family during its original one-epoch
training.  Therefore this is NOT writer generalization and is NOT DEV evidence.

It tests a narrower but important claim:
    Is cue identity recoverable from frozen memory by an independently trained
    downstream reader, without using recurrent hidden, context, images, original
    recall weights, workspace, or motor logits?

The new readers see ONLY the 64-D memory vector.

Readers
-------
1. linear probe: 64 -> 2
2. small MLP probe: 64 -> 64 -> 2

Both are newly initialized after the original system is completely frozen.

Evaluation
----------
A. Deterministic 5-fold condition-level cross-validation. LEFT/RIGHT members of
   a condition always stay in the same fold.
B. Leave-one-level-out tests for corridor_length and cue_steps.
C. Shuffled-label negative control for the linear reader.

TRAIN only. DEV and TEST are never touched.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch_directml
from torch import nn

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
    generate_task_spec,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_FORMAT = "working_memory_recall_capacity_control_training_v1"
EXPECTED_UPDATE = 1600

FOLD_SEED = 314159
PROBE_SEEDS = (101, 202, 303)
EPOCHS_LINEAR = 400
EPOCHS_MLP = 500
LR_LINEAR = 2e-2
LR_MLP = 1e-2
WEIGHT_DECAY = 1e-4


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def state_digest(state):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build(checkpoint, device):
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != EXPECTED_FORMAT:
        raise RuntimeError(
            f"checkpoint format mismatch: {payload.get('format')}"
        )
    if payload.get("arm") != "capacity_ungated":
        raise RuntimeError("checkpoint arm mismatch")
    if int(payload.get("update", -1)) != EXPECTED_UPDATE:
        raise RuntimeError("expected update-1600 checkpoint")
    if payload.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("checkpoint base SHA mismatch")
    if payload.get("base_digest") != EXPECTED_BASE_DIGEST:
        raise RuntimeError("checkpoint base digest mismatch")

    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("loaded base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("loaded base digest mismatch")

    model = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base, cfg
    )
    model.memory.load_state_dict(payload["memory"], strict=True)
    model.recall.load_state_dict(payload["recall"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, payload


@torch.no_grad()
def decision_memory(model, cfg, device, cid, cue):
    env = VariedCuedDetourWorld(
        int(cid),
        cue=cue,
        max_episode_steps=160,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        teacher = int(env.teacher_action())
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if teacher in (LEFT, RIGHT):
            return {
                "memory": memory.detach().float().cpu()[0],
                "decision_step": int(env.steps),
                "realized_hidden_delay": int(env.steps - env.cue_steps),
            }

        out = model.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        prev = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden = out["hidden"]
        memory = out["memory_next"]

    raise RuntimeError("decision not found")


def collect_dataset(model, cfg, device):
    rows = []
    total = len(TRAIN_CONDITION_RANGE)
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        left = decision_memory(model, cfg, device, cid, "left")
        right = decision_memory(model, cfg, device, cid, "right")

        if left["decision_step"] != right["decision_step"]:
            raise AssertionError("paired decision timing differs")

        spec = generate_task_spec(int(cid))
        common = {
            "condition_id": int(cid),
            "decision_step": int(left["decision_step"]),
            "realized_hidden_delay": int(left["realized_hidden_delay"]),
            "corridor_length": float(spec.corridor_length),
            "cue_steps": int(spec.cue_steps),
        }

        rows.append({
            **common,
            "cue": "left",
            "label": 0,
            "memory": left["memory"],
        })
        rows.append({
            **common,
            "cue": "right",
            "label": 1,
            "memory": right["memory"],
        })

        if i == 1 or i % 200 == 0 or i == total:
            print(f"collected={i}/{total} condition_pairs")

    return rows


def standardize(train_x, test_x):
    mu = train_x.mean(dim=0, keepdim=True)
    sigma = train_x.std(dim=0, unbiased=False, keepdim=True)
    sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)
    return (train_x - mu) / sigma, (test_x - mu) / sigma


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(64, 2)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def fit_probe(
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    kind,
    seed,
    shuffled_labels=False,
):
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    train_x, test_x = standardize(train_x, test_x)

    if shuffled_labels:
        g = torch.Generator().manual_seed(int(seed) + 900_001)
        perm = torch.randperm(len(train_y), generator=g)
        used_y = train_y[perm]
    else:
        used_y = train_y

    if kind == "linear":
        model = LinearProbe()
        epochs = EPOCHS_LINEAR
        lr = LR_LINEAR
    elif kind == "mlp":
        model = MLPProbe()
        epochs = EPOCHS_MLP
        lr = LR_MLP
    else:
        raise ValueError(kind)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits, used_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(train_x).argmax(dim=1)
        test_pred = model(test_x).argmax(dim=1)

    return {
        "train_accuracy": float((train_pred == train_y).float().mean()),
        "test_accuracy": float((test_pred == test_y).float().mean()),
    }


def tensors_for_conditions(rows, train_conditions, test_conditions):
    train_conditions = set(train_conditions)
    test_conditions = set(test_conditions)

    train_rows = [r for r in rows if r["condition_id"] in train_conditions]
    test_rows = [r for r in rows if r["condition_id"] in test_conditions]

    train_x = torch.stack([r["memory"] for r in train_rows])
    train_y = torch.tensor(
        [r["label"] for r in train_rows],
        dtype=torch.long,
    )
    test_x = torch.stack([r["memory"] for r in test_rows])
    test_y = torch.tensor(
        [r["label"] for r in test_rows],
        dtype=torch.long,
    )

    return train_x, train_y, test_x, test_y


def evaluate_split(rows, train_conditions, test_conditions):
    tx, ty, vx, vy = tensors_for_conditions(
        rows,
        train_conditions,
        test_conditions,
    )

    result = {
        "train_conditions": len(set(train_conditions)),
        "test_conditions": len(set(test_conditions)),
        "train_examples": len(ty),
        "test_examples": len(vy),
        "linear": [],
        "mlp": [],
        "linear_shuffled_control": [],
    }

    for seed in PROBE_SEEDS:
        result["linear"].append(
            fit_probe(tx, ty, vx, vy, kind="linear", seed=seed)
        )
        result["mlp"].append(
            fit_probe(tx, ty, vx, vy, kind="mlp", seed=seed)
        )
        result["linear_shuffled_control"].append(
            fit_probe(
                tx,
                ty,
                vx,
                vy,
                kind="linear",
                seed=seed,
                shuffled_labels=True,
            )
        )

    for key in ("linear", "mlp", "linear_shuffled_control"):
        vals = [x["test_accuracy"] for x in result[key]]
        result[key + "_mean_test_accuracy"] = sum(vals) / len(vals)
        result[key + "_min_test_accuracy"] = min(vals)
        result[key + "_max_test_accuracy"] = max(vals)

    return result


def make_cv_folds():
    ids = list(TRAIN_CONDITION_RANGE)
    rng = random.Random(FOLD_SEED)
    rng.shuffle(ids)
    folds = [ids[i::5] for i in range(5)]
    return folds


def factor_holdouts(rows, factor):
    values = sorted({r[factor] for r in rows})
    all_ids = set(TRAIN_CONDITION_RANGE)
    groups = {}
    for value in values:
        test_ids = sorted({
            r["condition_id"]
            for r in rows
            if r[factor] == value
        })
        train_ids = sorted(all_ids - set(test_ids))
        groups[str(value)] = (train_ids, test_ids)
    return groups


def strip_memories(rows):
    return [
        {k: v for k, v in r.items() if k != "memory"}
        for r in rows
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "memory_reuse_probe_v1.json"
        ),
    )
    args = ap.parse_args()

    brain_device = torch_directml.device()
    model, cfg, payload = build(Path(args.checkpoint), brain_device)

    print("WORKING_MEMORY_REPRESENTATION_REUSE_PROBE_V1")
    print("brain_device:", brain_device)
    print("new_reader_device: cpu")
    print("checkpoint:", args.checkpoint)
    print("checkpoint_update:", payload["update"])
    print("writer_frozen: true")
    print("original_recall_frozen_and_unused_by_new_reader: true")
    print("hidden_not_available_to_new_reader: true")
    print("context_not_available_to_new_reader: true")
    print("image_not_available_to_new_reader: true")
    print("motor_logits_not_available_to_new_reader: true")
    print("new_reader_input: memory_64d_only")
    print("dev_touched: false")
    print("test_touched: false")

    rows = collect_dataset(model, cfg, brain_device)

    folds = make_cv_folds()
    cv = []
    all_ids = set(TRAIN_CONDITION_RANGE)
    for i, test_ids in enumerate(folds):
        train_ids = sorted(all_ids - set(test_ids))
        print(
            f"five_fold={i+1}/5 "
            f"train_conditions={len(train_ids)} "
            f"test_conditions={len(test_ids)}"
        )
        r = evaluate_split(rows, train_ids, test_ids)
        r["fold"] = i
        cv.append(r)

    def aggregate_cv(key):
        vals = [f[key + "_mean_test_accuracy"] for f in cv]
        return {
            "mean": sum(vals) / len(vals),
            "min_fold": min(vals),
            "max_fold": max(vals),
        }

    factor_results = {}
    for factor in ("corridor_length", "cue_steps"):
        factor_results[factor] = {}
        for value, (train_ids, test_ids) in factor_holdouts(rows, factor).items():
            print(
                f"factor_holdout {factor}={value} "
                f"train={len(train_ids)} test={len(test_ids)}"
            )
            factor_results[factor][value] = evaluate_split(
                rows,
                train_ids,
                test_ids,
            )

    result = {
        "format": "working_memory_representation_reuse_probe_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": int(payload["update"]),
        "split": "train_only",
        "writer_frozen": True,
        "new_reader_input": "64D_memory_only",
        "probe_seeds": list(PROBE_SEEDS),
        "five_fold": cv,
        "five_fold_summary": {
            "linear": aggregate_cv("linear"),
            "mlp": aggregate_cv("mlp"),
            "linear_shuffled_control":
                aggregate_cv("linear_shuffled_control"),
        },
        "factor_holdout": factor_results,
        "dataset_metadata": strip_memories(rows),
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    s = result["five_fold_summary"]
    print()
    print("FIVE_FOLD_READER_GENERALIZATION")
    print(
        "linear:",
        f"mean={s['linear']['mean']:.4f}",
        f"min_fold={s['linear']['min_fold']:.4f}",
        f"max_fold={s['linear']['max_fold']:.4f}",
    )
    print(
        "mlp:",
        f"mean={s['mlp']['mean']:.4f}",
        f"min_fold={s['mlp']['min_fold']:.4f}",
        f"max_fold={s['mlp']['max_fold']:.4f}",
    )
    print(
        "linear_shuffled_control:",
        f"mean={s['linear_shuffled_control']['mean']:.4f}",
        f"min_fold={s['linear_shuffled_control']['min_fold']:.4f}",
        f"max_fold={s['linear_shuffled_control']['max_fold']:.4f}",
    )

    print()
    print("FACTOR_HELD_OUT_READER_GENERALIZATION")
    for factor, groups in factor_results.items():
        print(factor)
        for value, r in groups.items():
            print(
                f"  value={value}: "
                f"linear={r['linear_mean_test_accuracy']:.4f} "
                f"mlp={r['mlp_mean_test_accuracy']:.4f} "
                f"shuffle={r['linear_shuffled_control_mean_test_accuracy']:.4f}"
            )

    # Sanity expectation only; do not turn this into an admission threshold.
    if s["linear_shuffled_control"]["mean"] > 0.65:
        raise AssertionError(
            "shuffled-label control unexpectedly high; inspect leakage"
        )

    print("historical_base_bit_exact: true")
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_REPRESENTATION_REUSE_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
