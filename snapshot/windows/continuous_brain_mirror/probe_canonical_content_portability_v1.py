#!/usr/bin/env python3
"""
Representation portability diagnostic for v0.14.0 canonical recall failure.

No brain training. Fresh TEST remains sealed.

Collect, at the delayed response:
1. raw working memory64;
2. canonical admitted recall residual128.

Fit tiny linear cue decoders ONLY on v3 TRAIN representations and evaluate them
on already-opened v3 DEV.

Interpretation:
- raw64 high / recall128 low:
  admitted recall transform discarded or entangled portable cue information;
- both high / executive DEV low:
  content is present, but the late-rule executive failed to generalize/use it;
- both low:
  the writer itself did not generalize robustly to v3 geometry.

Readers run intentionally on CPU and never modify the brain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_directml
from torch import nn

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.canonical_recall_executive_v1 import (
    CanonicalRecallExecutiveInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    TRAIN_CONDITION_RANGE,
    DEV_CONDITION_RANGE,
    CanonicalExecutiveRuleWorld,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_ADMITTED_SHA256 = (
    "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
)
EXPECTED_EXEC_FORMAT = "canonical_recall_executive_training_v1"
SEEDS = (101, 202, 303)
EPOCHS = 500
LR = 2e-2
WD = 1e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def clone(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_model(checkpoint, device):
    ep = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ep.get("format") != EXPECTED_EXEC_FORMAT:
        raise RuntimeError("executive checkpoint format mismatch")
    if int(ep.get("update", -1)) != 1600:
        raise RuntimeError("expected update-1600 checkpoint")
    if ep.get("source_admitted_checkpoint_sha256") != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted provenance mismatch")

    if sha(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, bsha = _load_base_checkpoint(BASE_CHECKPOINT)
    if bsha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    if digest(clone(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = CanonicalRecallExecutiveInfant.grow_from_admitted(admitted)
    model.executive.load_state_dict(ep["executive"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg


@torch.no_grad()
def representation(model, cfg, device, cid, cue):
    env = CanonicalExecutiveRuleWorld(
        int(cid),
        cue=cue,
        rule="invert_lr",
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if env.at_response_step():
            hist = model.admitted.historical_step(
                image,
                context,
                memory,
                prev,
                hidden,
            )
            recall, _ = model.admitted.recall(
                hist["hidden"],
                memory,
            )
            return {
                "raw_memory": memory.detach().float().cpu()[0],
                "canonical_recall": recall.detach().float().cpu()[0],
                "hidden": hidden.detach().float().cpu()[0],
            }

        teacher = int(env.teacher_action())
        out = model.admitted.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"]
        memory = out["memory_next"]
        prev = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )

    raise RuntimeError("no response point")


def collect(model, cfg, device, ids, split):
    rows = []
    total = len(ids)
    for i, cid in enumerate(ids, 1):
        for cue, label in (("left", 0), ("right", 1)):
            rep = representation(
                model,
                cfg,
                device,
                int(cid),
                cue,
            )
            rows.append({
                "condition_id": int(cid),
                "split": split,
                "label": int(label),
                **rep,
            })
        if i == 1 or i % 200 == 0 or i == total:
            print(f"collected_{split}={i}/{total}")
    return rows


def standardize(train_x, eval_x):
    mu = train_x.mean(0, keepdim=True)
    sd = train_x.std(0, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return (train_x - mu) / sd, (eval_x - mu) / sd


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(int(dim), 2)

    def forward(self, x):
        return self.net(x)


def fit_eval(train_x, train_y, dev_x, dev_y, seed):
    torch.manual_seed(int(seed))
    model = LinearProbe(train_x.shape[1])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(train_x).argmax(1)
        dev_pred = model(dev_x).argmax(1)

    return {
        "train_accuracy": float((train_pred == train_y).float().mean()),
        "dev_accuracy": float((dev_pred == dev_y).float().mean()),
    }


def probe_representation(train_rows, dev_rows, field):
    tx = torch.stack([r[field] for r in train_rows])
    ty = torch.tensor([r["label"] for r in train_rows], dtype=torch.long)
    vx = torch.stack([r[field] for r in dev_rows])
    vy = torch.tensor([r["label"] for r in dev_rows], dtype=torch.long)
    tx, vx = standardize(tx, vx)

    runs = [
        {"seed": seed, **fit_eval(tx, ty, vx, vy, seed)}
        for seed in SEEDS
    ]
    return {
        "runs": runs,
        "mean_train_accuracy": sum(r["train_accuracy"] for r in runs) / len(runs),
        "mean_dev_accuracy": sum(r["dev_accuracy"] for r in runs) / len(runs),
        "min_dev_accuracy": min(r["dev_accuracy"] for r in runs),
        "max_dev_accuracy": max(r["dev_accuracy"] for r in runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg = load_model(Path(args.checkpoint), device)
    admitted_ref = clone(model.admitted)

    print("CANONICAL_CONTENT_PORTABILITY_PROBE_V1")
    print("brain_device:", device)
    print("reader_device: cpu")
    print("reader_fit_split: v3_train_only")
    print("reader_eval_split: already_opened_v3_dev")
    print("fresh_test_touched: false")

    train_rows = collect(
        model,
        cfg,
        device,
        TRAIN_CONDITION_RANGE,
        "train",
    )
    dev_rows = collect(
        model,
        cfg,
        device,
        DEV_CONDITION_RANGE,
        "dev",
    )

    results = {}
    for field in (
        "raw_memory",
        "canonical_recall",
        "hidden",
    ):
        results[field] = probe_representation(
            train_rows,
            dev_rows,
            field,
        )

    for k, v in admitted_ref.items():
        if not torch.equal(
            v,
            model.admitted.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"admitted brain changed: {k}")

    payload = {
        "format": "canonical_content_portability_probe_v1",
        "results": results,
        "fresh_test_touched": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )

    print()
    print("REPRESENTATION_PORTABILITY")
    for field, r in results.items():
        print(
            f"{field}: "
            f"train={r['mean_train_accuracy']:.4f} "
            f"dev={r['mean_dev_accuracy']:.4f} "
            f"range=[{r['min_dev_accuracy']:.4f},"
            f"{r['max_dev_accuracy']:.4f}]"
        )

    print("admitted_brain_bit_exact: true")
    print("FRESH_TEST_TOUCHED: false")
    print("CANONICAL_CONTENT_PORTABILITY_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
