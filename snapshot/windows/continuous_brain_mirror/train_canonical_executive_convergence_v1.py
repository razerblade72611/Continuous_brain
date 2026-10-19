#!/usr/bin/env python3
"""
Canonical recalled-content executive convergence training v1.

Purpose
-------
The update-1600 v0.14.0 checkpoint was diagnosed as underlearned:
fixed-final TRAIN performance was only ~56% mean rule pair-full, while
raw memory64, canonical recall128, and hidden state all retained 100%
TRAIN->DEV cue decodability.

Therefore this script changes NO architecture. It continues optimization of the
same 16,896-parameter executive from the exact update-1600 checkpoint and its
optimizer state.

Each additional epoch:
- uses the same 1600 TRAIN structural conditions exactly once;
- deterministically reshuffles condition order;
- still uses all six cue×rule counterfactuals per update;
- never accesses v3 DEV or TEST.

The runner evaluates fixed-final TRAIN after each additional epoch and stops
when the frozen convergence gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

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
    CanonicalExecutiveRuleWorld,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
ORIGINAL_EXEC_CHECKPOINT = (
    "state/canonical_recall_executive_v1/latest.portable.pt"
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
ORIGINAL_FORMAT = "canonical_recall_executive_training_v1"
CONVERGENCE_FORMAT = "canonical_recall_executive_convergence_v1"

RULES = ("invert_lr", "vertical", "vertical_reverse")
CONDITIONS_PER_EPOCH = 1600
BASE_SCHEDULE_SEED = 814239
BASE_INIT_SEED = 24683


class DMLSafeAdam:
    def __init__(
        self,
        params,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = [
            {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }
            for p in self.params
        ]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count
        step_size = self.lr / bc1
        sqrt_bc2 = math.sqrt(bc2)

        for p, state in zip(self.params, self.state):
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all().item():
                raise RuntimeError("non-finite gradient")
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            m.mul_(self.beta1)
            m.add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2)
            v.add_(g * g, alpha=1.0 - self.beta2)
            denom = v.sqrt()
            denom.div_(sqrt_bc2)
            denom.add_(self.eps)
            p.add_(m / denom, alpha=-step_size)

    def portable_state(self):
        return {
            "step_count": int(self.step_count),
            "lr": float(self.lr),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "state": [
                {
                    "exp_avg": x["exp_avg"].detach().cpu(),
                    "exp_avg_sq": x["exp_avg_sq"].detach().cpu(),
                }
                for x in self.state
            ],
        }

    def load_portable_state(self, payload):
        self.step_count = int(payload["step_count"])
        self.beta1 = float(payload.get("beta1", 0.9))
        self.beta2 = float(payload.get("beta2", 0.999))
        self.eps = float(payload.get("eps", 1e-8))
        if len(payload["state"]) != len(self.state):
            raise RuntimeError("optimizer parameter count mismatch")

        for dst, src in zip(self.state, payload["state"]):
            dst["exp_avg"].copy_(
                src["exp_avg"].to(dst["exp_avg"].device)
            )
            dst["exp_avg_sq"].copy_(
                src["exp_avg_sq"].to(dst["exp_avg_sq"].device)
            )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_state_exact(reference, module, label):
    current = module.state_dict()
    for k, expected in reference.items():
        got = current[k].detach().cpu()
        if not torch.equal(expected, got):
            raise AssertionError(f"{label}: changed {k}")


def epoch_schedule(epoch_index):
    """
    epoch_index is zero-based:
      0 = original v0.14.0 epoch (already completed)
      1 = first convergence epoch
      ...
    """
    ids = list(TRAIN_CONDITION_RANGE)
    rng = random.Random(
        BASE_SCHEDULE_SEED + 1_000_003 * int(epoch_index)
    )
    rng.shuffle(ids)
    return ids


def load_protected_brain(device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    admitted_payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(
        admitted_payload["memory"],
        strict=True,
    )
    admitted.recall.load_state_dict(
        admitted_payload["recall"],
        strict=True,
    )
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = CanonicalRecallExecutiveInfant.grow_from_admitted(
        admitted
    ).to(device)
    model.train(True)
    model.admitted.eval()
    model.base.eval()

    return (
        model,
        cfg,
        clone_state(admitted),
        clone_state(base),
        model.trainable_parameters(),
    )


def run_member(model, cfg, device, cid, cue, rule):
    env = CanonicalExecutiveRuleWorld(
        int(cid),
        cue=cue,
        rule=rule,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    post_losses = []
    response_loss = None
    pred = None
    teacher_at_response = None
    delta_max = 0.0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        rule_id = torch.tensor(
            [env.instruction_id()],
            dtype=torch.long,
            device=device,
        )

        out = model.step(
            image,
            context,
            memory,
            prev,
            hidden,
            rule_id,
        )

        if not env.instruction_visible():
            if torch.count_nonzero(
                out["executive_delta"].detach().cpu()
            ).item() != 0:
                raise AssertionError(
                    "pre-rule executive delta is nonzero"
                )

        teacher = int(env.teacher_action())
        target = torch.tensor(
            [teacher],
            dtype=torch.long,
            device=device,
        )

        if env.instruction_visible():
            post_losses.append(
                F.cross_entropy(out["logits"], target)
            )

        if env.at_response_step():
            response_loss = F.cross_entropy(
                out["logits"],
                target,
            )
            pred = int(
                torch.argmax(out["logits"], dim=-1)[0].item()
            )
            teacher_at_response = teacher

        delta_max = max(
            delta_max,
            float(
                out["executive_delta"]
                .detach()
                .abs()
                .max()
                .cpu()
            ),
        )

        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"].detach()
        memory = out["memory_next"].detach()
        prev = (target + 1).long()

    if response_loss is None:
        raise AssertionError("no response loss")

    return (
        0.5 * torch.stack(post_losses).mean()
        + 0.5 * response_loss,
        bool(pred == teacher_at_response),
        delta_max,
    )


def sixway(model, cfg, device, cid):
    losses = []
    correct = {}
    delta_max = 0.0

    for cue in ("left", "right"):
        for rule in RULES:
            loss, ok, d = run_member(
                model,
                cfg,
                device,
                cid,
                cue,
                rule,
            )
            losses.append(loss)
            correct[(cue, rule)] = ok
            delta_max = max(delta_max, d)

    return torch.stack(losses).mean(), correct, delta_max


def save_checkpoint(
    path,
    model,
    optimizer,
    global_update,
    *,
    lr,
    parent_sha,
):
    torch.save(
        {
            "format": CONVERGENCE_FORMAT,
            "global_update": int(global_update),
            # Compatibility alias for evaluators / provenance.
            "update": int(global_update),
            "source_admitted_checkpoint_sha256":
                EXPECTED_ADMITTED_SHA256,
            "historical_base_sha256": EXPECTED_BASE_SHA256,
            "historical_base_digest": EXPECTED_BASE_DIGEST,
            "base_schedule_seed": BASE_SCHEDULE_SEED,
            "base_init_seed": BASE_INIT_SEED,
            "current_lr": float(lr),
            "parent_checkpoint_sha256": parent_sha,
            "executive": {
                k: v.detach().cpu()
                for k, v in model.executive.state_dict().items()
            },
            "optimizer": optimizer.portable_state(),
        },
        path,
    )


def load_parent(model, optimizer, checkpoint_path):
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    fmt = payload.get("format")

    if fmt == ORIGINAL_FORMAT:
        completed = int(payload.get("update", -1))
        if completed != 1600:
            raise RuntimeError(
                "original parent must be update 1600"
            )
    elif fmt == CONVERGENCE_FORMAT:
        completed = int(payload.get("global_update", -1))
    else:
        raise RuntimeError(
            f"unsupported parent checkpoint format: {fmt}"
        )

    if (
        payload.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("parent admitted SHA mismatch")

    model.executive.load_state_dict(
        payload["executive"],
        strict=True,
    )
    optimizer.load_portable_state(payload["optimizer"])

    return payload, completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/canonical_recall_executive_convergence_v1",
    )
    ap.add_argument(
        "--original-checkpoint",
        default=ORIGINAL_EXEC_CHECKPOINT,
    )
    ap.add_argument(
        "--target-update",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--lr",
        type=float,
        required=True,
    )
    args = ap.parse_args()

    if args.target_update % CONDITIONS_PER_EPOCH != 0:
        raise ValueError(
            "target-update must end on a full 1600-condition epoch"
        )
    if args.target_update <= 1600:
        raise ValueError("target-update must exceed 1600")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(BASE_INIT_SEED)
    device = torch_directml.device()

    (
        model,
        cfg,
        admitted_ref,
        base_ref,
        params,
    ) = load_protected_brain(device)

    optimizer = DMLSafeAdam(params, lr=args.lr)

    latest = state_dir / "latest.portable.pt"
    parent = (
        latest
        if latest.is_file()
        else Path(args.original_checkpoint)
    )
    parent_sha = sha256_file(parent)

    parent_payload, completed = load_parent(
        model,
        optimizer,
        parent,
    )

    if completed > args.target_update:
        raise RuntimeError(
            f"latest checkpoint update {completed} exceeds "
            f"requested target {args.target_update}"
        )
    if completed == args.target_update:
        print(
            f"already_at_target_update: {args.target_update}"
        )
        return

    # Explicitly set the planned stage LR after restoring Adam moments.
    optimizer.lr = float(args.lr)

    log_path = state_dir / "training.jsonl"
    log_mode = "a" if log_path.exists() else "w"

    print("CANONICAL_EXECUTIVE_CONVERGENCE_TRAINING_V1")
    print("device:", device)
    print("parent_checkpoint:", parent)
    print("parent_checkpoint_sha256:", parent_sha)
    print("start_global_update:", completed + 1)
    print("target_global_update:", args.target_update)
    print("stage_lr:", args.lr)
    print("architecture_changed: false")
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    started = time.time()

    with log_path.open(log_mode) as log:
        for global_update in range(
            completed + 1,
            args.target_update + 1,
        ):
            zero_based = global_update - 1
            epoch_index = zero_based // CONDITIONS_PER_EPOCH
            within_epoch = zero_based % CONDITIONS_PER_EPOCH
            ids = epoch_schedule(epoch_index)
            cid = ids[within_epoch]

            optimizer.zero_grad(set_to_none=True)
            loss, correct, delta_max = sixway(
                model,
                cfg,
                device,
                cid,
            )
            loss.backward()
            optimizer.step()

            record = {
                "global_update": int(global_update),
                "epoch_index": int(epoch_index),
                "within_epoch": int(within_epoch),
                "condition_id": int(cid),
                "lr": float(optimizer.lr),
                "loss": float(loss.detach().cpu()),
                "six_way_full": bool(all(correct.values())),
            }
            for rule in RULES:
                record[f"{rule}_pair_full"] = bool(
                    correct[("left", rule)]
                    and correct[("right", rule)]
                )

            log.write(json.dumps(record) + "\n")
            log.flush()

            if (
                global_update == completed + 1
                or global_update % 25 == 0
                or global_update == args.target_update
            ):
                pairs = {
                    rule: int(
                        correct[("left", rule)]
                        and correct[("right", rule)]
                    )
                    for rule in RULES
                }
                print(
                    f"update={global_update:05d}/"
                    f"{args.target_update:05d} "
                    f"epoch={epoch_index + 1} "
                    f"condition={cid:04d} "
                    f"loss={record['loss']:.6f} "
                    f"invert={pairs['invert_lr']} "
                    f"vertical={pairs['vertical']} "
                    f"vreverse={pairs['vertical_reverse']} "
                    f"six={int(all(correct.values()))} "
                    f"delta={delta_max:.5f}"
                )

            if (
                global_update % 100 == 0
                or global_update == args.target_update
            ):
                assert_state_exact(
                    admitted_ref,
                    model.admitted,
                    "admitted brain",
                )
                assert_state_exact(
                    base_ref,
                    model.base,
                    "historical base",
                )

                ckpt = state_dir / (
                    f"canonical_convergence_"
                    f"u{global_update:06d}.portable.pt"
                )
                save_checkpoint(
                    ckpt,
                    model,
                    optimizer,
                    global_update,
                    lr=optimizer.lr,
                    parent_sha=parent_sha,
                )
                latest.write_bytes(ckpt.read_bytes())
                print("saved:", ckpt)

    assert_state_exact(
        admitted_ref,
        model.admitted,
        "admitted brain final",
    )
    assert_state_exact(
        base_ref,
        model.base,
        "historical base final",
    )

    print(
        "elapsed_seconds:",
        round(time.time() - started, 3),
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("CANONICAL_EXECUTIVE_CONVERGENCE_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
