#!/usr/bin/env python3
"""
Mechanism diagnostic for above-chance shuffled-label controls in the frozen
64-D Continuous Brain memory representation assay.

This is NOT a larger permutation-significance study. It asks WHY a reader
trained on randomized labels can still score above chance against true cue
labels.

Frozen brain/writer. TRAIN family only. DEV/TEST untouched.

Randomization schemes
---------------------
1. example_permutation
   Exact reproduction of the original probe's control:
       shuffled_y = true_y[randperm(N)]
   This randomizes individual examples independently (subject to preserved
   overall label counts).

2. pair_orientation
   One random coin flip per structural condition. Pair members remain
   complementary, but the mapping of LEFT/RIGHT memories to pseudo-labels is
   randomly normal or inverted for each condition.

3. pair_orientation_exact_balanced
   Same as #2, but exactly half of training conditions use normal orientation
   and half inverted (or as close as possible for odd counts). This forces
   global pseudo-label agreement with true cue identity to exactly 50% (up to
   one condition when odd).

For each fitted shuffled reader, report:
- pseudo-label agreement with true training labels;
- training accuracy against pseudo-labels;
- training accuracy against true labels;
- test accuracy against true labels;
- cosine with the real-label reader decision vector;
- cosine with the empirical cue axis:
      mean(memory_RIGHT - memory_LEFT)
- decision-vector norm.

If above-chance test accuracy is caused by accidental projection onto the true
cue axis, shuffled test accuracy should track these cosine alignments and/or
pseudo-label agreement.

We evaluate:
- deterministic 5-fold condition splits;
- the factor-held-out slices that had the largest original shuffled-control
  deviations: corridor_length 0.28, 0.36, 0.84 and cue_steps 1, 3, 4.
"""

from __future__ import annotations

import argparse
import json
import math
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

FOLD_SEED = 314159
# More than the original 3 seeds, but still a mechanism sweep rather than a
# permutation p-value study.
SHUFFLE_SEEDS = tuple(1000 + 37 * i for i in range(24))
REAL_SEEDS = (101, 202, 303)
EPOCHS = 400
LR = 2e-2
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("checkpoint format mismatch")
    if payload.get("arm") != "capacity_ungated":
        raise RuntimeError("checkpoint arm mismatch")
    if int(payload.get("update", -1)) != 1600:
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

    model = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    model.memory.load_state_dict(payload["memory"], strict=True)
    model.recall.load_state_dict(payload["recall"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def decision_memory(model, cfg, device, cid, cue):
    env = VariedCuedDetourWorld(cid, cue=cue, max_episode_steps=160)
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        teacher = int(env.teacher_action())
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if teacher in (LEFT, RIGHT):
            return memory.detach().float().cpu()[0]

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
        prev = torch.tensor([teacher + 1], dtype=torch.long, device=device)
        hidden = out["hidden"]
        memory = out["memory_next"]

    raise RuntimeError("decision not found")


def collect_dataset(model, cfg, device):
    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        spec = generate_task_spec(int(cid))
        lm = decision_memory(model, cfg, device, int(cid), "left")
        rm = decision_memory(model, cfg, device, int(cid), "right")
        rows.append({
            "condition_id": int(cid),
            "label": 0,
            "cue": "left",
            "corridor_length": float(spec.corridor_length),
            "cue_steps": int(spec.cue_steps),
            "memory": lm,
        })
        rows.append({
            "condition_id": int(cid),
            "label": 1,
            "cue": "right",
            "corridor_length": float(spec.corridor_length),
            "cue_steps": int(spec.cue_steps),
            "memory": rm,
        })
        if i == 1 or i % 200 == 0 or i == len(TRAIN_CONDITION_RANGE):
            print(f"collected={i}/{len(TRAIN_CONDITION_RANGE)} condition_pairs")
    return rows


def make_folds():
    ids = list(TRAIN_CONDITION_RANGE)
    rng = random.Random(FOLD_SEED)
    rng.shuffle(ids)
    return [ids[i::5] for i in range(5)]


def standardize(train_x, test_x):
    mu = train_x.mean(dim=0, keepdim=True)
    sd = train_x.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return (train_x - mu) / sd, (test_x - mu) / sd, mu, sd


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        return self.linear(x)


def decision_vector(model):
    # Binary softmax decision direction. Positive means class 1 over class 0.
    return (
        model.linear.weight[1].detach().cpu()
        - model.linear.weight[0].detach().cpu()
    )


def cosine(a, b):
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def fit_linear(train_x, train_y_used, train_y_true, test_x, test_y_true, seed):
    torch.manual_seed(int(seed))
    model = LinearProbe()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y_used)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(train_x).argmax(dim=1)
        test_pred = model(test_x).argmax(dim=1)

    return model, {
        "train_accuracy_pseudo": float(
            (train_pred == train_y_used).float().mean()
        ),
        "train_accuracy_true": float(
            (train_pred == train_y_true).float().mean()
        ),
        "test_accuracy_true": float(
            (test_pred == test_y_true).float().mean()
        ),
    }


def rows_to_tensors(rows, train_ids, test_ids):
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    tr = [r for r in rows if r["condition_id"] in train_ids]
    te = [r for r in rows if r["condition_id"] in test_ids]

    tx = torch.stack([r["memory"] for r in tr])
    ty = torch.tensor([r["label"] for r in tr], dtype=torch.long)
    vx = torch.stack([r["memory"] for r in te])
    vy = torch.tensor([r["label"] for r in te], dtype=torch.long)

    return tr, te, tx, ty, vx, vy


def empirical_cue_axis(train_rows):
    left = {}
    right = {}
    for r in train_rows:
        if r["label"] == 0:
            left[r["condition_id"]] = r["memory"]
        else:
            right[r["condition_id"]] = r["memory"]
    ids = sorted(set(left) & set(right))
    diffs = [right[cid] - left[cid] for cid in ids]
    return torch.stack(diffs).mean(dim=0)


def random_targets(train_rows, true_y, scheme, seed):
    n = len(train_rows)
    if n != len(true_y):
        raise AssertionError("row/label length mismatch")

    if scheme == "example_permutation":
        g = torch.Generator().manual_seed(int(seed) + 900_001)
        perm = torch.randperm(n, generator=g)
        pseudo = true_y[perm].clone()

    elif scheme in ("pair_orientation", "pair_orientation_exact_balanced"):
        ids = []
        seen = set()
        for r in train_rows:
            cid = int(r["condition_id"])
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)

        rng = random.Random(int(seed) + 700_003)

        if scheme == "pair_orientation":
            normal = {cid: bool(rng.getrandbits(1)) for cid in ids}
        else:
            shuffled_ids = ids[:]
            rng.shuffle(shuffled_ids)
            half = len(shuffled_ids) // 2
            normal_ids = set(shuffled_ids[:half])
            normal = {cid: cid in normal_ids for cid in ids}

        labels = []
        for r in train_rows:
            true = int(r["label"])
            keep = normal[int(r["condition_id"])]
            labels.append(true if keep else 1 - true)
        pseudo = torch.tensor(labels, dtype=torch.long)

    else:
        raise ValueError(scheme)

    agreement = float((pseudo == true_y).float().mean())
    return pseudo, agreement


def correlation(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (dx * dy)


def run_split(name, rows, train_ids, test_ids):
    train_rows, test_rows, tx0, ty, vx0, vy = rows_to_tensors(
        rows, train_ids, test_ids
    )
    tx, vx, mu, sd = standardize(tx0, vx0)

    # Cue axis transformed into the standardized feature coordinates used by
    # the probes. The original-space mean pair difference gets divided by sd.
    cue_axis_raw = empirical_cue_axis(train_rows)
    cue_axis_std = cue_axis_raw / sd[0]

    real_vectors = []
    real_results = []
    for seed in REAL_SEEDS:
        m, metrics = fit_linear(tx, ty, ty, vx, vy, seed)
        w = decision_vector(m)
        real_vectors.append(w)
        real_results.append({
            **metrics,
            "seed": int(seed),
            "cosine_cue_axis": cosine(w, cue_axis_std),
            "weight_norm": float(torch.linalg.vector_norm(w)),
        })

    real_axis = torch.stack(real_vectors).mean(dim=0)

    schemes = {}
    for scheme in (
        "example_permutation",
        "pair_orientation",
        "pair_orientation_exact_balanced",
    ):
        runs = []
        for seed in SHUFFLE_SEEDS:
            pseudo, agreement = random_targets(
                train_rows, ty, scheme, seed
            )
            m, metrics = fit_linear(tx, pseudo, ty, vx, vy, seed)
            w = decision_vector(m)
            runs.append({
                **metrics,
                "seed": int(seed),
                "pseudo_true_label_agreement": agreement,
                "cosine_real_reader": cosine(w, real_axis),
                "cosine_cue_axis": cosine(w, cue_axis_std),
                "weight_norm": float(torch.linalg.vector_norm(w)),
            })

        test_acc = [r["test_accuracy_true"] for r in runs]
        agree = [r["pseudo_true_label_agreement"] for r in runs]
        cos_real = [r["cosine_real_reader"] for r in runs]
        cos_cue = [r["cosine_cue_axis"] for r in runs]
        fit_pseudo = [r["train_accuracy_pseudo"] for r in runs]

        schemes[scheme] = {
            "runs": runs,
            "mean_test_accuracy_true": sum(test_acc) / len(test_acc),
            "min_test_accuracy_true": min(test_acc),
            "max_test_accuracy_true": max(test_acc),
            "mean_pseudo_true_label_agreement": sum(agree) / len(agree),
            "mean_train_accuracy_pseudo": sum(fit_pseudo) / len(fit_pseudo),
            "corr_agreement_vs_test_accuracy":
                correlation(agree, test_acc),
            "corr_cosine_real_vs_test_accuracy":
                correlation(cos_real, test_acc),
            "corr_cosine_cue_vs_test_accuracy":
                correlation(cos_cue, test_acc),
        }

    return {
        "name": name,
        "train_conditions": len(set(train_ids)),
        "test_conditions": len(set(test_ids)),
        "real_reader": real_results,
        "real_reader_mean_test_accuracy":
            sum(r["test_accuracy_true"] for r in real_results)
            / len(real_results),
        "real_reader_mean_cosine_cue_axis":
            sum(r["cosine_cue_axis"] for r in real_results)
            / len(real_results),
        "schemes": schemes,
    }


def factor_split(rows, factor, value):
    test_ids = sorted({
        int(r["condition_id"])
        for r in rows
        if r[factor] == value
    })
    all_ids = set(TRAIN_CONDITION_RANGE)
    train_ids = sorted(all_ids - set(test_ids))
    return train_ids, test_ids


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
            "memory_shuffle_mechanism_v1.json"
        ),
    )
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg = build(Path(args.checkpoint), device)

    print("WORKING_MEMORY_SHUFFLE_MECHANISM_DIAGNOSTIC_V1")
    print("brain_device:", device)
    print("probe_device: cpu")
    print("checkpoint:", args.checkpoint)
    print("writer_frozen: true")
    print("probe_input: memory_64d_only")
    print("original_control_shuffle_is_per_example: true")
    print("shuffle_seed_count:", len(SHUFFLE_SEEDS))
    print("purpose: mechanism_not_null_pvalue")
    print("dev_touched: false")
    print("test_touched: false")

    rows = collect_dataset(model, cfg, device)

    results = []

    folds = make_folds()
    all_ids = set(TRAIN_CONDITION_RANGE)
    for i, test_ids in enumerate(folds, 1):
        train_ids = sorted(all_ids - set(test_ids))
        print(f"running five_fold_{i}")
        results.append(
            run_split(f"five_fold_{i}", rows, train_ids, test_ids)
        )

    targeted = [
        ("corridor_length", 0.28),
        ("corridor_length", 0.36),
        ("corridor_length", 0.84),
        ("cue_steps", 1),
        ("cue_steps", 3),
        ("cue_steps", 4),
    ]
    for factor, value in targeted:
        train_ids, test_ids = factor_split(rows, factor, value)
        print(f"running {factor}={value}")
        results.append(
            run_split(
                f"{factor}={value}",
                rows,
                train_ids,
                test_ids,
            )
        )

    payload = {
        "format": "working_memory_shuffle_mechanism_v1",
        "checkpoint": str(args.checkpoint),
        "split": "train_only",
        "shuffle_seed_count": len(SHUFFLE_SEEDS),
        "shuffle_schemes": [
            "example_permutation",
            "pair_orientation",
            "pair_orientation_exact_balanced",
        ],
        "results": results,
        "dev_touched": False,
        "test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print()
    print("SHUFFLE_MECHANISM_SUMMARY")
    for result in results:
        print(result["name"])
        print(
            f"  real_test={result['real_reader_mean_test_accuracy']:.4f} "
            f"real_cos_cue={result['real_reader_mean_cosine_cue_axis']:+.4f}"
        )
        for scheme, s in result["schemes"].items():
            print(
                f"  {scheme}: "
                f"test={s['mean_test_accuracy_true']:.4f} "
                f"range=[{s['min_test_accuracy_true']:.4f},"
                f"{s['max_test_accuracy_true']:.4f}] "
                f"pseudo_true_agree="
                f"{s['mean_pseudo_true_label_agreement']:.4f} "
                f"pseudo_fit={s['mean_train_accuracy_pseudo']:.4f} "
                f"corr(agree,test)="
                f"{s['corr_agreement_vs_test_accuracy']:+.4f} "
                f"corr(cosReal,test)="
                f"{s['corr_cosine_real_vs_test_accuracy']:+.4f} "
                f"corr(cosCue,test)="
                f"{s['corr_cosine_cue_vs_test_accuracy']:+.4f}"
            )

    print("historical_base_bit_exact: true")
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_SHUFFLE_MECHANISM_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
