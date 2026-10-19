#!/usr/bin/env python3
"""
Frozen-writer post-write instruction composition assay.

Goal
----
Test whether a stored 64-D memory can be re-used for a downstream computation
whose meaning is determined only AFTER the memory has already been written.

The original brain/writer/recall system is frozen.  The post-write instruction
does not exist during brain rollout and is generated only after the decision
memory has been collected.

Task
----
Frozen memory contains a past cue:
    LEFT or RIGHT

After writing, a new independent instruction is supplied to a brand-new reader:
    NORMAL or INVERT

Correct downstream response:
    NORMAL: respond with the remembered cue side
    INVERT: respond with the opposite side

The new reader must therefore combine:
    stored past content + new post-write information

Readers
-------
1. Joint nonlinear reader:
       [memory64, instruction1] -> 64 tanh -> 2
   This must learn the late-bound composition.

2. Linear concatenation control:
       [memory64, instruction1] -> 2
   XOR-like cue x instruction mapping is not linearly separable in the ideal
   binary abstraction, so this is a useful interaction control.

3. Factorized cue decoder:
       memory64 -> cue2
   The cue decoder is trained independently, then the post-write instruction
   flips or preserves its prediction at inference.  This demonstrates a very
   direct form of content reuse with a late policy transform.

Evaluation
----------
- deterministic 5-fold CONDITION-held-out CV;
- leave-one-corridor-length-level-out;
- leave-one-cue-steps-level-out;
- three reader initialization seeds;
- intervention controls on the joint nonlinear reader:
    * memory zeroed;
    * instruction forced NORMAL;
    * instruction flipped;
    * LEFT/RIGHT memory swapped within each structural condition.

The writer itself has seen TRAIN before, so this remains a TRAIN-only mechanism
assay, not writer environmental generalization. DEV and TEST stay sealed.
"""

from __future__ import annotations

import argparse
import json
import random
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

FOLD_SEED = 271828
READER_SEEDS = (101, 202, 303)
EPOCHS_LINEAR = 500
EPOCHS_MLP = 700
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


def build_brain(checkpoint, device):
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
                "delay": int(env.steps - env.cue_steps),
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


def collect_memory_rows(model, cfg, device):
    """
    IMPORTANT: no post-write instruction exists anywhere in this function.
    Instructions are created only later in expand_postwrite_task().
    """
    rows = []
    total = len(TRAIN_CONDITION_RANGE)

    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        spec = generate_task_spec(int(cid))
        left = decision_memory(model, cfg, device, cid, "left")
        right = decision_memory(model, cfg, device, cid, "right")

        if left["decision_step"] != right["decision_step"]:
            raise AssertionError("paired decision timing mismatch")

        common = {
            "condition_id": int(cid),
            "corridor_length": float(spec.corridor_length),
            "cue_steps": int(spec.cue_steps),
            "decision_step": int(left["decision_step"]),
            "realized_hidden_delay": int(left["delay"]),
        }

        rows.append({
            **common,
            "cue_label": 0,
            "cue": "left",
            "memory": left["memory"],
        })
        rows.append({
            **common,
            "cue_label": 1,
            "cue": "right",
            "memory": right["memory"],
        })

        if i == 1 or i % 200 == 0 or i == total:
            print(f"collected={i}/{total} condition_pairs")

    return rows


def expand_postwrite_task(memory_rows):
    """
    Duplicate every already-written memory under BOTH later instructions.

    instruction=0 NORMAL => response == cue
    instruction=1 INVERT => response == 1-cue
    """
    examples = []
    for r in memory_rows:
        cue = int(r["cue_label"])
        for instruction in (0, 1):
            response = cue if instruction == 0 else 1 - cue
            examples.append({
                **r,
                "instruction": instruction,
                "instruction_name": "normal" if instruction == 0 else "invert",
                "response_label": int(response),
            })
    return examples


def make_folds():
    ids = list(TRAIN_CONDITION_RANGE)
    rng = random.Random(FOLD_SEED)
    rng.shuffle(ids)
    return [ids[i::5] for i in range(5)]


def factor_split(rows, factor, value):
    test_ids = sorted({
        int(r["condition_id"])
        for r in rows
        if r[factor] == value
    })
    all_ids = set(TRAIN_CONDITION_RANGE)
    train_ids = sorted(all_ids - set(test_ids))
    return train_ids, test_ids


def get_examples(examples, ids):
    ids = set(ids)
    return [e for e in examples if e["condition_id"] in ids]


def memory_stats(train_examples):
    # One copy per memory is enough; duplicates under instructions are harmless
    # but we dedupe for conceptual clarity.
    seen = set()
    mems = []
    for e in train_examples:
        key = (e["condition_id"], e["cue_label"])
        if key in seen:
            continue
        seen.add(key)
        mems.append(e["memory"])
    x = torch.stack(mems)
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return mu, sd


def joint_tensors(examples, mu, sd):
    mem = torch.stack([e["memory"] for e in examples])
    mem = (mem - mu) / sd
    instr = torch.tensor(
        [float(e["instruction"]) for e in examples],
        dtype=torch.float32,
    ).unsqueeze(1)
    x = torch.cat([mem, instr], dim=1)
    y = torch.tensor(
        [int(e["response_label"]) for e in examples],
        dtype=torch.long,
    )
    return x, y


def cue_tensors(memory_rows, ids, mu, sd):
    ids = set(ids)
    rs = [r for r in memory_rows if r["condition_id"] in ids]
    x = torch.stack([r["memory"] for r in rs])
    x = (x - mu) / sd
    y = torch.tensor([int(r["cue_label"]) for r in rs], dtype=torch.long)
    return x, y, rs


class JointLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(65, 2)

    def forward(self, x):
        return self.net(x)


class JointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class CueLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(64, 2)

    def forward(self, x):
        return self.net(x)


def fit_model(model, x, y, *, epochs, lr):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def acc(pred, y):
    return float((pred == y).float().mean())


def evaluate_joint(model, examples, mu, sd):
    x, y = joint_tensors(examples, mu, sd)

    with torch.no_grad():
        pred = model(x).argmax(dim=1)

    normal_mask = torch.tensor(
        [e["instruction"] == 0 for e in examples],
        dtype=torch.bool,
    )
    invert_mask = ~normal_mask

    # Memory-zero intervention: keep instruction, erase remembered content.
    x_mem_zero = x.clone()
    x_mem_zero[:, :64] = 0.0
    with torch.no_grad():
        pred_mem_zero = model(x_mem_zero).argmax(dim=1)

    # Force every instruction to NORMAL while targets remain original.
    x_instr_normal = x.clone()
    x_instr_normal[:, 64] = 0.0
    with torch.no_grad():
        pred_instr_normal = model(x_instr_normal).argmax(dim=1)

    # Flip every post-write instruction. Expected response must also flip.
    x_instr_flip = x.clone()
    x_instr_flip[:, 64] = 1.0 - x_instr_flip[:, 64]
    y_flip = 1 - y
    with torch.no_grad():
        pred_instr_flip = model(x_instr_flip).argmax(dim=1)

    # Per-memory dual-instruction consistency. Because examples are constructed
    # adjacent NORMAL, INVERT for each memory, predictions should be opposites.
    if len(examples) % 2 != 0:
        raise AssertionError("instruction-expanded examples not paired")
    pair_both_correct = []
    pair_flip_consistent = []
    for i in range(0, len(examples), 2):
        if (
            examples[i]["condition_id"] != examples[i + 1]["condition_id"]
            or examples[i]["cue_label"] != examples[i + 1]["cue_label"]
            or examples[i]["instruction"] != 0
            or examples[i + 1]["instruction"] != 1
        ):
            raise AssertionError("unexpected expanded-example ordering")
        pair_both_correct.append(
            bool(pred[i] == y[i] and pred[i + 1] == y[i + 1])
        )
        pair_flip_consistent.append(bool(pred[i] != pred[i + 1]))

    return {
        "accuracy": acc(pred, y),
        "normal_accuracy": acc(pred[normal_mask], y[normal_mask]),
        "invert_accuracy": acc(pred[invert_mask], y[invert_mask]),
        "memory_zero_accuracy": acc(pred_mem_zero, y),
        "instruction_forced_normal_accuracy": acc(pred_instr_normal, y),
        "instruction_flip_counterfactual_accuracy": acc(
            pred_instr_flip, y_flip
        ),
        "dual_instruction_both_correct": (
            sum(pair_both_correct) / len(pair_both_correct)
        ),
        "instruction_flip_consistency": (
            sum(pair_flip_consistent) / len(pair_flip_consistent)
        ),
    }


def evaluate_memory_swap(model, examples, mu, sd):
    """
    Swap LEFT/RIGHT memory within each structural condition but leave the
    recipient's post-write instruction unchanged.

    The correct donor-consistent response becomes:
        donor cue under recipient instruction.
    """
    # Build memory lookup by (condition, cue).
    mem = {}
    for e in examples:
        mem[(e["condition_id"], e["cue_label"])] = e["memory"]

    swapped = []
    donor_targets = []
    recipient_targets = []
    for e in examples:
        donor_cue = 1 - int(e["cue_label"])
        donor_mem = mem[(e["condition_id"], donor_cue)]
        instr = int(e["instruction"])
        donor_response = donor_cue if instr == 0 else 1 - donor_cue
        swapped.append({
            **e,
            "memory": donor_mem,
            "response_label": donor_response,
        })
        donor_targets.append(donor_response)
        recipient_targets.append(int(e["response_label"]))

    x, _ = joint_tensors(swapped, mu, sd)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)

    donor_y = torch.tensor(donor_targets, dtype=torch.long)
    recipient_y = torch.tensor(recipient_targets, dtype=torch.long)

    return {
        "swapped_memory_donor_consistent_accuracy": acc(pred, donor_y),
        "swapped_memory_original_recipient_accuracy": acc(pred, recipient_y),
    }


def factorized_reader(train_memory_rows, test_examples, train_ids, test_ids, mu, sd, seed):
    tx, ty, _ = cue_tensors(train_memory_rows, train_ids, mu, sd)
    vx_rows = [
        r for r in train_memory_rows if r["condition_id"] in set(test_ids)
    ]
    vx = torch.stack([r["memory"] for r in vx_rows])
    vx = (vx - mu) / sd
    vy = torch.tensor(
        [int(r["cue_label"]) for r in vx_rows],
        dtype=torch.long,
    )

    torch.manual_seed(int(seed))
    model = fit_model(
        CueLinear(),
        tx,
        ty,
        epochs=EPOCHS_LINEAR,
        lr=LR_LINEAR,
    )

    with torch.no_grad():
        cue_pred = model(vx).argmax(dim=1)

    cue_by_key = {
        (r["condition_id"], r["cue_label"]): int(p)
        for r, p in zip(vx_rows, cue_pred)
    }

    preds = []
    targets = []
    for e in test_examples:
        predicted_cue = cue_by_key[
            (e["condition_id"], e["cue_label"])
        ]
        instr = int(e["instruction"])
        response = (
            predicted_cue if instr == 0 else 1 - predicted_cue
        )
        preds.append(response)
        targets.append(int(e["response_label"]))

    p = torch.tensor(preds, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return {
        "accuracy": acc(p, y),
        "cue_decode_accuracy": acc(cue_pred, vy),
    }


def run_split(name, memory_rows, examples, train_ids, test_ids):
    train_ex = get_examples(examples, train_ids)
    test_ex = get_examples(examples, test_ids)
    mu, sd = memory_stats(train_ex)

    train_x, train_y = joint_tensors(train_ex, mu, sd)

    result = {
        "name": name,
        "train_conditions": len(set(train_ids)),
        "test_conditions": len(set(test_ids)),
        "joint_mlp": [],
        "joint_linear_control": [],
        "factorized_linear_cue_reader": [],
    }

    for seed in READER_SEEDS:
        torch.manual_seed(int(seed))
        mlp = fit_model(
            JointMLP(),
            train_x,
            train_y,
            epochs=EPOCHS_MLP,
            lr=LR_MLP,
        )
        mlp_metrics = evaluate_joint(mlp, test_ex, mu, sd)
        mlp_metrics.update(
            evaluate_memory_swap(mlp, test_ex, mu, sd)
        )
        mlp_metrics["seed"] = int(seed)
        result["joint_mlp"].append(mlp_metrics)

        torch.manual_seed(int(seed))
        lin = fit_model(
            JointLinear(),
            train_x,
            train_y,
            epochs=EPOCHS_LINEAR,
            lr=LR_LINEAR,
        )
        lin_metrics = evaluate_joint(lin, test_ex, mu, sd)
        lin_metrics.update(
            evaluate_memory_swap(lin, test_ex, mu, sd)
        )
        lin_metrics["seed"] = int(seed)
        result["joint_linear_control"].append(lin_metrics)

        fr = factorized_reader(
            memory_rows,
            test_ex,
            train_ids,
            test_ids,
            mu,
            sd,
            seed,
        )
        fr["seed"] = int(seed)
        result["factorized_linear_cue_reader"].append(fr)

    def avg(section, key):
        vals = [r[key] for r in result[section]]
        return sum(vals) / len(vals)

    result["summary"] = {
        "mlp_accuracy": avg("joint_mlp", "accuracy"),
        "mlp_normal_accuracy": avg("joint_mlp", "normal_accuracy"),
        "mlp_invert_accuracy": avg("joint_mlp", "invert_accuracy"),
        "mlp_dual_both_correct": avg(
            "joint_mlp", "dual_instruction_both_correct"
        ),
        "mlp_instruction_flip_consistency": avg(
            "joint_mlp", "instruction_flip_consistency"
        ),
        "mlp_memory_zero_accuracy": avg(
            "joint_mlp", "memory_zero_accuracy"
        ),
        "mlp_instruction_forced_normal_accuracy": avg(
            "joint_mlp", "instruction_forced_normal_accuracy"
        ),
        "mlp_instruction_flip_counterfactual_accuracy": avg(
            "joint_mlp",
            "instruction_flip_counterfactual_accuracy",
        ),
        "mlp_swapped_memory_donor_consistent_accuracy": avg(
            "joint_mlp",
            "swapped_memory_donor_consistent_accuracy",
        ),
        "linear_control_accuracy": avg(
            "joint_linear_control", "accuracy"
        ),
        "factorized_accuracy": avg(
            "factorized_linear_cue_reader", "accuracy"
        ),
        "factorized_cue_decode_accuracy": avg(
            "factorized_linear_cue_reader", "cue_decode_accuracy"
        ),
    }

    return result


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
            "post_write_instruction_probe_v1.json"
        ),
    )
    args = ap.parse_args()

    brain_device = torch_directml.device()
    brain, cfg, payload = build_brain(
        Path(args.checkpoint),
        brain_device,
    )

    print("WORKING_MEMORY_POST_WRITE_INSTRUCTION_PROBE_V1")
    print("brain_device:", brain_device)
    print("new_reader_device: cpu")
    print("checkpoint:", args.checkpoint)
    print("checkpoint_update:", payload["update"])
    print("writer_frozen: true")
    print("instruction_exists_during_brain_rollout: false")
    print("instruction_generated_only_after_memory_collection: true")
    print("new_reader_inputs: frozen_memory64_plus_postwrite_instruction1")
    print("original_recall_unused_by_new_reader: true")
    print("dev_touched: false")
    print("test_touched: false")

    memory_rows = collect_memory_rows(
        brain,
        cfg,
        brain_device,
    )

    # The instruction is created only AFTER all frozen brain memories exist.
    examples = expand_postwrite_task(memory_rows)

    all_ids = set(TRAIN_CONDITION_RANGE)
    results = []

    folds = make_folds()
    for i, test_ids in enumerate(folds, 1):
        train_ids = sorted(all_ids - set(test_ids))
        print(
            f"five_fold={i}/5 "
            f"train_conditions={len(train_ids)} "
            f"test_conditions={len(test_ids)}"
        )
        results.append(
            run_split(
                f"five_fold_{i}",
                memory_rows,
                examples,
                train_ids,
                test_ids,
            )
        )

    factor_results = {}
    for factor in ("corridor_length", "cue_steps"):
        values = sorted({r[factor] for r in memory_rows})
        factor_results[factor] = {}
        for value in values:
            train_ids, test_ids = factor_split(
                memory_rows,
                factor,
                value,
            )
            print(
                f"factor_holdout {factor}={value} "
                f"train={len(train_ids)} test={len(test_ids)}"
            )
            factor_results[factor][str(value)] = run_split(
                f"{factor}={value}",
                memory_rows,
                examples,
                train_ids,
                test_ids,
            )

    def cv_mean(key):
        vals = [r["summary"][key] for r in results]
        return {
            "mean": sum(vals) / len(vals),
            "min_fold": min(vals),
            "max_fold": max(vals),
        }

    summary_keys = [
        "mlp_accuracy",
        "mlp_normal_accuracy",
        "mlp_invert_accuracy",
        "mlp_dual_both_correct",
        "mlp_instruction_flip_consistency",
        "mlp_memory_zero_accuracy",
        "mlp_instruction_forced_normal_accuracy",
        "mlp_instruction_flip_counterfactual_accuracy",
        "mlp_swapped_memory_donor_consistent_accuracy",
        "linear_control_accuracy",
        "factorized_accuracy",
        "factorized_cue_decode_accuracy",
    ]

    payload_out = {
        "format": "working_memory_post_write_instruction_probe_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": int(payload["update"]),
        "split": "train_only",
        "writer_frozen": True,
        "instruction_post_write_only": True,
        "reader_seeds": list(READER_SEEDS),
        "five_fold": results,
        "five_fold_summary": {
            key: cv_mean(key) for key in summary_keys
        },
        "factor_holdout": factor_results,
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_out, indent=2, sort_keys=True))

    s = payload_out["five_fold_summary"]

    print()
    print("FIVE_FOLD_POST_WRITE_COMPOSITION")
    for key in summary_keys:
        x = s[key]
        print(
            f"{key}: mean={x['mean']:.4f} "
            f"min={x['min_fold']:.4f} "
            f"max={x['max_fold']:.4f}"
        )

    print()
    print("FACTOR_HELD_OUT_POST_WRITE_COMPOSITION")
    for factor, groups in factor_results.items():
        print(factor)
        for value, r in groups.items():
            x = r["summary"]
            print(
                f"  value={value}: "
                f"mlp={x['mlp_accuracy']:.4f} "
                f"dual={x['mlp_dual_both_correct']:.4f} "
                f"swapDonor="
                f"{x['mlp_swapped_memory_donor_consistent_accuracy']:.4f} "
                f"linear={x['linear_control_accuracy']:.4f} "
                f"factorized={x['factorized_accuracy']:.4f}"
            )

    # Pre-registered TRAIN-side mechanism expectations.
    if s["mlp_accuracy"]["mean"] < 0.95:
        raise AssertionError("late-bound MLP composition below 95%")
    if s["mlp_dual_both_correct"]["mean"] < 0.90:
        raise AssertionError("dual-instruction consistency below 90%")
    if s["mlp_instruction_flip_counterfactual_accuracy"]["mean"] < 0.95:
        raise AssertionError("instruction counterfactual failed")
    if s["mlp_swapped_memory_donor_consistent_accuracy"]["mean"] < 0.95:
        raise AssertionError("memory donor-consistency failed")
    if s["mlp_memory_zero_accuracy"]["mean"] > 0.60:
        raise AssertionError("reader succeeds without memory")
    if s["mlp_instruction_forced_normal_accuracy"]["mean"] > 0.60:
        raise AssertionError("reader succeeds without usable instruction")
    if s["factorized_accuracy"]["mean"] < 0.95:
        raise AssertionError("factorized late-bound reuse below 95%")

    print("historical_base_bit_exact: true")
    print("POST_WRITE_COMPOSITION_GATE: PASS")
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_POST_WRITE_INSTRUCTION_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
