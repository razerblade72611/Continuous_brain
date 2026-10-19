#!/usr/bin/env python3
"""
TRAIN-only staged optimization for RichCanonicalExecutiveInfant.

No v3 DEV or TEST access. The v3 TRAIN family is reused only as an architecture
fit/convergence qualification because v3 DEV has already been opened by the
retired v0.14.0 architecture.

Each epoch uses all 1600 TRAIN conditions once in deterministic shuffled order.
Each optimizer step contains all 6 cue×rule members.
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
from continuous_brain.rich_canonical_executive_v1 import (
    RichCanonicalExecutiveInfant,
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
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_ADMITTED_SHA256 = (
    "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
)

FORMAT = "rich_canonical_executive_training_v1"
RULES = ("invert_lr", "vertical", "vertical_reverse")
CONDITIONS_PER_EPOCH = 1600
SCHEDULE_SEED = 947221
INIT_SEED = 515152


class DMLSafeAdam:
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.b1 = float(betas[0])
        self.b2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = [
            {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
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
        bc1 = 1.0 - self.b1 ** self.step_count
        bc2 = 1.0 - self.b2 ** self.step_count
        for p, s in zip(self.params, self.state):
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all().item():
                raise RuntimeError("non-finite gradient")
            s["m"].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            s["v"].mul_(self.b2).add_(g * g, alpha=1.0 - self.b2)
            denom = s["v"].sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            p.add_(s["m"] / denom, alpha=-self.lr / bc1)

    def portable_state(self):
        return {
            "step_count": int(self.step_count),
            "lr": float(self.lr),
            "beta1": self.b1,
            "beta2": self.b2,
            "eps": self.eps,
            "state": [
                {
                    "exp_avg": x["m"].detach().cpu(),
                    "exp_avg_sq": x["v"].detach().cpu(),
                }
                for x in self.state
            ],
        }

    def load_portable_state(self, payload):
        self.step_count = int(payload["step_count"])
        self.b1 = float(payload.get("beta1", 0.9))
        self.b2 = float(payload.get("beta2", 0.999))
        self.eps = float(payload.get("eps", 1e-8))
        if len(payload["state"]) != len(self.state):
            raise RuntimeError("optimizer state length mismatch")
        for dst, src in zip(self.state, payload["state"]):
            dst["m"].copy_(src["exp_avg"].to(dst["m"].device))
            dst["v"].copy_(src["exp_avg_sq"].to(dst["v"].device))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_exact(reference, module, label):
    for k, v in reference.items():
        got = module.state_dict()[k].detach().cpu()
        if not torch.equal(v, got):
            raise AssertionError(f"{label}: changed {k}")


def epoch_schedule(epoch_index):
    ids = list(TRAIN_CONDITION_RANGE)
    rng = random.Random(
        SCHEDULE_SEED + 1_000_003 * int(epoch_index)
    )
    rng.shuffle(ids)
    return ids


def load_model(device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    base_ref = clone_state(base)
    if state_digest(base_ref) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    admitted_ref = clone_state(admitted)
    model = RichCanonicalExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True)
    model.admitted.eval()
    model.base.eval()

    return model, cfg, admitted_ref, base_ref, model.trainable_parameters()


def run_member(model, cfg, device, cid, cue, rule):
    env = CanonicalExecutiveRuleWorld(cid, cue=cue, rule=rule)
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    post = []
    response = None
    pred = None
    teacher_at_response = None
    delta_max = 0.0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        rid = torch.tensor(
            [env.instruction_id()],
            dtype=torch.long,
            device=device,
        )

        out = model.step(image, context, memory, prev, hidden, rid)

        if not env.instruction_visible():
            if torch.count_nonzero(out["executive_delta"].detach().cpu()).item():
                raise AssertionError("pre-rule delta not zero")

        teacher = int(env.teacher_action())
        target = torch.tensor([teacher], dtype=torch.long, device=device)

        if env.instruction_visible():
            post.append(F.cross_entropy(out["logits"], target))

        if env.at_response_step():
            response = F.cross_entropy(out["logits"], target)
            pred = int(torch.argmax(out["logits"], dim=-1)[0].item())
            teacher_at_response = teacher

        delta_max = max(
            delta_max,
            float(out["executive_delta"].detach().abs().max().cpu()),
        )

        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"].detach()
        memory = out["memory_next"].detach()
        prev = (target + 1).long()

    if response is None:
        raise AssertionError("no response loss")

    loss = 0.5 * torch.stack(post).mean() + 0.5 * response
    return loss, bool(pred == teacher_at_response), delta_max


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


def save_checkpoint(path, model, opt, update, lr):
    torch.save(
        {
            "format": FORMAT,
            "global_update": int(update),
            "update": int(update),
            "source_admitted_checkpoint_sha256":
                EXPECTED_ADMITTED_SHA256,
            "historical_base_sha256": EXPECTED_BASE_SHA256,
            "historical_base_digest": EXPECTED_BASE_DIGEST,
            "schedule_seed": SCHEDULE_SEED,
            "init_seed": INIT_SEED,
            "current_lr": float(lr),
            "executive": {
                k: v.detach().cpu()
                for k, v in model.executive.state_dict().items()
            },
            "optimizer": opt.portable_state(),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/rich_canonical_executive_v1",
    )
    ap.add_argument("--target-update", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    args = ap.parse_args()

    if args.target_update % CONDITIONS_PER_EPOCH != 0:
        raise ValueError("target update must end on a full epoch")
    if args.target_update < CONDITIONS_PER_EPOCH:
        raise ValueError("target update too small")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    latest = state_dir / "latest.portable.pt"
    log_path = state_dir / "training.jsonl"

    torch.manual_seed(INIT_SEED)
    device = torch_directml.device()
    model, cfg, admitted_ref, base_ref, params = load_model(device)
    opt = DMLSafeAdam(params, lr=args.lr)

    completed = 0
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT:
            raise RuntimeError("latest checkpoint format mismatch")
        if (
            payload.get("source_admitted_checkpoint_sha256")
            != EXPECTED_ADMITTED_SHA256
        ):
            raise RuntimeError("latest admitted provenance mismatch")
        model.executive.load_state_dict(payload["executive"], strict=True)
        opt.load_portable_state(payload["optimizer"])
        completed = int(payload["global_update"])

    if completed > args.target_update:
        raise RuntimeError("latest checkpoint exceeds target")
    if completed == args.target_update:
        print("already_at_target_update:", completed)
        return

    opt.lr = float(args.lr)

    print("RICH_CANONICAL_EXECUTIVE_TRAINING_V1")
    print("device:", device)
    print("executive_parameter_count:", sum(p.numel() for p in params))
    print("start_global_update:", completed + 1)
    print("target_global_update:", args.target_update)
    print("stage_lr:", args.lr)
    print("architecture: rich_canonical_content_rule_compositor")
    print("v3_train_only: true")
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    started = time.time()
    log_mode = "a" if log_path.exists() else "w"

    with log_path.open(log_mode) as log:
        for update in range(completed + 1, args.target_update + 1):
            zero_based = update - 1
            epoch_index = zero_based // CONDITIONS_PER_EPOCH
            within_epoch = zero_based % CONDITIONS_PER_EPOCH
            cid = epoch_schedule(epoch_index)[within_epoch]

            opt.zero_grad(set_to_none=True)
            loss, correct, delta_max = sixway(
                model,
                cfg,
                device,
                cid,
            )
            loss.backward()
            opt.step()

            row = {
                "global_update": update,
                "epoch_index": epoch_index,
                "within_epoch": within_epoch,
                "condition_id": int(cid),
                "lr": float(opt.lr),
                "loss": float(loss.detach().cpu()),
                "six_way_full": bool(all(correct.values())),
            }
            for rule in RULES:
                row[f"{rule}_pair_full"] = bool(
                    correct[("left", rule)]
                    and correct[("right", rule)]
                )
            log.write(json.dumps(row) + "\n")
            log.flush()

            if update == completed + 1 or update % 25 == 0 or update == args.target_update:
                pairs = {
                    r: int(
                        correct[("left", r)]
                        and correct[("right", r)]
                    )
                    for r in RULES
                }
                print(
                    f"update={update:05d}/{args.target_update:05d} "
                    f"epoch={epoch_index + 1} "
                    f"condition={cid:04d} "
                    f"loss={row['loss']:.6f} "
                    f"invert={pairs['invert_lr']} "
                    f"vertical={pairs['vertical']} "
                    f"vreverse={pairs['vertical_reverse']} "
                    f"six={int(all(correct.values()))} "
                    f"delta={delta_max:.5f}"
                )

            if update % 100 == 0 or update == args.target_update:
                assert_exact(admitted_ref, model.admitted, "admitted brain")
                assert_exact(base_ref, model.base, "historical base")
                ckpt = state_dir / f"rich_canonical_u{update:06d}.portable.pt"
                save_checkpoint(ckpt, model, opt, update, opt.lr)
                latest.write_bytes(ckpt.read_bytes())
                print("saved:", ckpt)

    assert_exact(admitted_ref, model.admitted, "admitted brain final")
    assert_exact(base_ref, model.base, "historical base final")

    print("elapsed_seconds:", round(time.time() - started, 3))
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("RICH_CANONICAL_EXECUTIVE_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
