#!/usr/bin/env python3
"""
TRAIN-only executive-composition learning smoke.

The DEV-admitted memory brain is frozen.  Only the late-bound executive is
optimized.

Every update contains a perfectly balanced 2x2 counterfactual:
    cue LEFT/RIGHT × instruction NORMAL/INVERT.

No old DEV or TEST condition is accessed.

This is an engineering/learning smoke, not executive generalization evidence.
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
from continuous_brain.executive_working_memory_v1 import (
    ExecutiveCompositionInfant,
)
from continuous_brain.late_instruction_detour_world_v1 import (
    LateInstructionDetourWorld,
)
from continuous_brain.varied_cued_detour_world import (
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
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
EXPECTED_ADMITTED_FORMAT = (
    "working_memory_recall_capacity_control_training_v1"
)
CHECKPOINT_FORMAT = "executive_composition_training_v1"


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
                raise RuntimeError("non-finite optimizer gradient")
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
            "lr": self.lr,
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
            diff = None
            if torch.is_floating_point(expected):
                diff = float((expected - got).abs().max())
            raise AssertionError(
                f"{label}: tensor changed {k}; max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return 0.0
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.detach().abs().max().cpu())


def condition_schedule(updates, seed):
    ids = list(TRAIN_CONDITION_RANGE)
    order = []
    epoch = 0
    while len(order) < int(updates):
        rng = random.Random(int(seed) + 1_000_003 * epoch)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        order.extend(shuffled)
        epoch += 1
    return order[: int(updates)]


def load_model(device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != EXPECTED_ADMITTED_FORMAT:
        raise RuntimeError("admitted checkpoint format mismatch")
    if payload.get("arm") != "capacity_ungated":
        raise RuntimeError("admitted checkpoint arm mismatch")
    if int(payload.get("update", -1)) != 1600:
        raise RuntimeError("admitted checkpoint update mismatch")

    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(payload["memory"], strict=True)
    admitted.recall.load_state_dict(payload["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    admitted_ref = clone_state(admitted)
    base_ref = clone_state(admitted.base)

    model = ExecutiveCompositionInfant.grow_from_admitted(
        admitted
    ).to(device)
    model.train(True)
    model.admitted.eval()
    model.base.eval()

    additions = model.trainable_parameters()
    admitted_ids = {id(p) for p in admitted.parameters()}
    executive_ids = {id(p) for p in additions}
    if not admitted_ids.isdisjoint(executive_ids):
        raise AssertionError("admitted brain leaked into optimizer")

    return model, cfg, admitted_ref, base_ref, additions


def decision_metrics(logits, teacher):
    logits = logits[0]
    full_pred = int(torch.argmax(logits).item())
    d = float(
        logits[RIGHT].detach().cpu()
        - logits[LEFT].detach().cpu()
    )
    side_pred = RIGHT if d > 0.0 else LEFT
    return {
        "full_correct": bool(full_pred == teacher),
        "side_correct": bool(side_pred == teacher),
        "margin": float(d if teacher == RIGHT else -d),
    }


def run_member_teacher(
    model,
    cfg,
    device,
    *,
    cid,
    cue,
    instruction,
):
    env = LateInstructionDetourWorld(
        cid,
        cue=cue,
        instruction=instruction,
        instruction_delay=1,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    post_losses = []
    decision_loss = None
    decision = None
    residual_max = 0.0
    decision_step = None
    instruction_step = None
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        instruction_id = torch.tensor(
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
            instruction_id,
        )

        if env.instruction_visible() and instruction_step is None:
            instruction_step = int(env.steps)

        if not env.instruction_visible():
            if torch.count_nonzero(
                out["executive_residual"].detach().cpu()
            ).item() != 0:
                raise AssertionError(
                    "executive altered pre-instruction behavior"
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

        if decision_loss is None and teacher in (LEFT, RIGHT):
            decision_loss = F.cross_entropy(
                out["logits"],
                target,
            )
            decision = decision_metrics(
                out["logits"],
                teacher,
            )
            decision_step = int(env.steps)

        residual_max = max(
            residual_max,
            float(
                out["executive_residual"]
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
        steps += 1

    if decision is None or decision_loss is None:
        raise AssertionError("no late-bound lateral decision")
    if instruction_step is None:
        raise AssertionError("instruction never arrived")
    if decision_step <= instruction_step:
        raise AssertionError(
            "decision did not occur strictly after instruction arrival"
        )

    post_mean = torch.stack(post_losses).mean()
    loss = 0.5 * post_mean + 0.5 * decision_loss

    return {
        "loss": loss,
        "decision": decision,
        "residual_max": residual_max,
        "decision_step": decision_step,
        "instruction_step": instruction_step,
        "steps": steps,
    }


def run_four_way_teacher(model, cfg, device, cid):
    members = {}
    losses = []
    for cue in ("left", "right"):
        for instruction in ("normal", "invert"):
            key = f"{cue}_{instruction}"
            r = run_member_teacher(
                model,
                cfg,
                device,
                cid=cid,
                cue=cue,
                instruction=instruction,
            )
            members[key] = r
            losses.append(r["loss"])
    return torch.stack(losses).mean(), members


@torch.no_grad()
def autonomous_episode(
    model,
    cfg,
    device,
    *,
    cid,
    cue,
    instruction,
):
    env = LateInstructionDetourWorld(
        cid,
        cue=cue,
        instruction=instruction,
        instruction_delay=1,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    first_decision = None
    steps = 0
    final_info = None

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        instruction_id = torch.tensor(
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
            instruction_id,
        )
        action = int(torch.argmax(out["logits"], dim=-1)[0].item())

        if (
            first_decision is None
            and env.instruction_visible()
            and env.steps >= int(env.spec.behavioral_signature()[0])
        ):
            first_decision = action

        obs, _, _, final_info = env.step(action)
        hidden = out["hidden"]
        memory = out["memory_next"]
        prev = torch.tensor(
            [action + 1],
            dtype=torch.long,
            device=device,
        )
        steps += 1
        if steps >= 160:
            break

    return {
        "success": bool(
            final_info is not None
            and final_info.get("reached_correct", False)
        ),
        "steps": steps,
        "first_decision": first_decision,
    }


def autonomous_probe(model, cfg, device, condition_ids):
    rows = []
    for cid in condition_ids:
        for cue in ("left", "right"):
            for instruction in ("normal", "invert"):
                r = autonomous_episode(
                    model,
                    cfg,
                    device,
                    cid=cid,
                    cue=cue,
                    instruction=instruction,
                )
                rows.append(
                    {
                        "condition_id": int(cid),
                        "cue": cue,
                        "instruction": instruction,
                        **r,
                    }
                )
    return {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": (
            sum(r["success"] for r in rows) / len(rows)
            if rows
            else 0.0
        ),
        "rows": rows,
    }


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    update,
    schedule_seed,
    init_seed,
    lr,
):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_admitted_checkpoint": ADMITTED_CHECKPOINT,
        "source_admitted_checkpoint_sha256": EXPECTED_ADMITTED_SHA256,
        "historical_base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "historical_base_digest": EXPECTED_BASE_DIGEST,
        "update": int(update),
        "schedule_seed": int(schedule_seed),
        "init_seed": int(init_seed),
        "lr": float(lr),
        "instruction_delay": 1,
        "executive": {
            k: v.detach().cpu()
            for k, v in model.executive.state_dict().items()
        },
        "optimizer": optimizer.portable_state(),
    }
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument(
        "--state-dir",
        default="state/executive_composition_phase1_v1",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--schedule-seed", type=int, default=92731)
    ap.add_argument("--init-seed", type=int, default=24680)
    args = ap.parse_args()

    if args.updates < 3:
        raise ValueError("updates must be >=3")

    torch.manual_seed(args.init_seed)
    device = torch_directml.device()
    model, cfg, admitted_ref, base_ref, additions = load_model(device)

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "training.jsonl"
    if log_path.exists() and log_path.stat().st_size:
        raise RuntimeError(
            f"refusing to overwrite non-empty smoke log: {log_path}"
        )

    optimizer = DMLSafeAdam(additions, lr=args.lr)
    schedule = condition_schedule(args.updates, args.schedule_seed)

    print("EXECUTIVE_COMPOSITION_LEARNING_SMOKE_V1")
    print("device:", device)
    print("updates:", args.updates)
    print("lr:", args.lr)
    print("schedule_seed:", args.schedule_seed)
    print("init_seed:", args.init_seed)
    print("members_per_update: 4")
    print("admitted_memory_checkpoint_sha256:", EXPECTED_ADMITTED_SHA256)
    print("executive_parameter_count:", sum(p.numel() for p in additions))
    print("world_model_receives_pre_executive_workspace: true")
    print("direct_motor_shortcut: false")
    print("old_dev_touched: false")
    print("old_test_touched: false")

    started = time.time()

    with log_path.open("w") as log:
        for update, cid in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, members = run_four_way_teacher(
                model,
                cfg,
                device,
                cid,
            )
            loss.backward()

            out_grad = grad_max(model.executive.output.weight.grad)
            mem_grad = grad_max(
                model.executive.memory_features.weight.grad
            )
            instr_grad = grad_max(
                model.executive.instruction.weight.grad
            )

            if out_grad <= 0.0:
                raise RuntimeError("missing executive output gradient")
            if update > 1:
                if mem_grad <= 0.0:
                    raise RuntimeError(
                        "missing executive memory-feature gradient"
                    )
                if instr_grad <= 0.0:
                    raise RuntimeError(
                        "missing executive instruction gradient"
                    )

            optimizer.step()

            if update == 1 or update % 10 == 0 or update == args.updates:
                assert_state_exact(
                    admitted_ref,
                    model.admitted,
                    f"admitted brain changed update {update}",
                )
                assert_state_exact(
                    base_ref,
                    model.base,
                    f"historical base changed update {update}",
                )
                base_verify = True
            else:
                base_verify = False

            normal = [
                members["left_normal"]["decision"],
                members["right_normal"]["decision"],
            ]
            invert = [
                members["left_invert"]["decision"],
                members["right_invert"]["decision"],
            ]

            record = {
                "update": update,
                "condition_id": int(cid),
                "loss": float(loss.detach().cpu()),
                "normal_pair_full": bool(
                    normal[0]["full_correct"]
                    and normal[1]["full_correct"]
                ),
                "invert_pair_full": bool(
                    invert[0]["full_correct"]
                    and invert[1]["full_correct"]
                ),
                "normal_pair_side": bool(
                    normal[0]["side_correct"]
                    and normal[1]["side_correct"]
                ),
                "invert_pair_side": bool(
                    invert[0]["side_correct"]
                    and invert[1]["side_correct"]
                ),
                "four_way_side": bool(
                    all(
                        x["decision"]["side_correct"]
                        for x in members.values()
                    )
                ),
                "residual_max": max(
                    x["residual_max"]
                    for x in members.values()
                ),
                "output_grad_max": out_grad,
                "memory_feature_grad_max": mem_grad,
                "instruction_grad_max": instr_grad,
                "admitted_brain_verified": base_verify,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()

            if (
                update == 1
                or update % 5 == 0
                or update == args.updates
            ):
                print(
                    f"update={update:04d}/{args.updates} "
                    f"condition={cid:04d} "
                    f"loss={record['loss']:.6f} "
                    f"Nfull={int(record['normal_pair_full'])} "
                    f"Ifull={int(record['invert_pair_full'])} "
                    f"Nside={int(record['normal_pair_side'])} "
                    f"Iside={int(record['invert_pair_side'])} "
                    f"fourSide={int(record['four_way_side'])} "
                    f"resid={record['residual_max']:.5f} "
                    f"outg={out_grad:.4g} "
                    f"memg={mem_grad:.4g} "
                    f"instg={instr_grad:.4g} "
                    f"verify={'yes' if base_verify else 'deferred'}"
                )

    ckpt = state_dir / (
        f"executive_composition_u{args.updates:06d}.portable.pt"
    )
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        update=args.updates,
        schedule_seed=args.schedule_seed,
        init_seed=args.init_seed,
        lr=args.lr,
    )
    latest = state_dir / "latest.portable.pt"
    latest.write_bytes(ckpt.read_bytes())

    # Descriptive closed-loop probe on TRAIN only. No pass threshold.
    probe_ids = schedule[: min(20, len(schedule))]
    autonomous = autonomous_probe(
        model,
        cfg,
        device,
        probe_ids,
    )
    (state_dir / "autonomous_probe.json").write_text(
        json.dumps(autonomous, indent=2, sort_keys=True)
    )

    assert_state_exact(
        admitted_ref,
        model.admitted,
        "admitted brain changed by end of smoke",
    )
    assert_state_exact(
        base_ref,
        model.base,
        "historical base changed by end of smoke",
    )
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise AssertionError("admitted checkpoint file changed")

    print("saved:", ckpt)
    print(
        "autonomous_train_probe:",
        f"{autonomous['successes']}/{autonomous['episodes']}",
        f"rate={autonomous['success_rate']:.4f}",
    )
    print("elapsed_seconds:", round(time.time() - started, 3))
    print("portable_cpu_checkpoint: true")
    print("admitted_checkpoint_unchanged: true")
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("OLD_DEV_TOUCHED: false")
    print("OLD_TEST_TOUCHED: false")
    print("EXECUTIVE_COMPOSITION_LEARNING_SMOKE_COMPLETE")


if __name__ == "__main__":
    main()
