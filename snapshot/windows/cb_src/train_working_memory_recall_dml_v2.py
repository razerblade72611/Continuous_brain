#!/usr/bin/env python3
"""
Matched production trainer for Continuous Brain Working Memory recall arms.

Arms
----
linear : M1.5 bare bias-free zero-init memory->workspace residual
gated  : M2 normalized contextual recall:
         residual = 2*sigmoid(G[h,m]) * P_0(m)

Both arms:
- start from the untouched canonical 14k InfantBrain;
- use the same 64-D writer architecture;
- use the same train-condition schedule;
- use the same counterfactual LEFT/RIGHT pair per optimizer update;
- use the same objective and DirectML-safe optimizer;
- preserve the historical base bit-exactly.

This script is both the 25-update smoke path and the later full-epoch production
path.  No DEV/TEST conditions are accessible through its schedule.
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
from continuous_brain.infant import one_hot_action
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
)
from continuous_brain.working_memory_recall_arms_v2 import (
    RecallBridgeWorkingMemoryInfant,
)


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
CHECKPOINT_FORMAT = "working_memory_recall_training_v2"


class DMLSafeAdam:
    """Qualified DirectML-safe Adam implementation."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
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
        sqrt_bc2 = math.sqrt(bc2)
        step_size = self.lr / bc1

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

    def state_dict(self):
        return {
            "step_count": int(self.step_count),
            "lr": float(self.lr),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "state": [
                {
                    "exp_avg": s["exp_avg"].detach().cpu(),
                    "exp_avg_sq": s["exp_avg_sq"].detach().cpu(),
                }
                for s in self.state
            ],
        }

    def load_state_dict(self, payload):
        if len(payload["state"]) != len(self.state):
            raise RuntimeError("optimizer state length mismatch")
        self.step_count = int(payload["step_count"])
        self.lr = float(payload["lr"])
        self.beta1 = float(payload["beta1"])
        self.beta2 = float(payload["beta2"])
        self.eps = float(payload["eps"])
        for dst, src, p in zip(self.state, payload["state"], self.params):
            dst["exp_avg"].copy_(src["exp_avg"].to(p.device))
            dst["exp_avg_sq"].copy_(src["exp_avg_sq"].to(p.device))


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_state_exact(reference, module, label):
    current = module.state_dict()
    if reference.keys() != current.keys():
        raise AssertionError(f"{label}: state keys changed")
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(expected, actual):
            diff = None
            if torch.is_floating_point(expected):
                diff = float((expected - actual).abs().max().item())
            raise AssertionError(
                f"{label}: {name} changed; max_abs_diff={diff}"
            )


def clone_workspace(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.base.nucleus.workspace.state_dict().items()
    }


def assert_workspace_exact(reference, model, update):
    current = model.base.nucleus.workspace.state_dict()
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(expected, actual):
            diff = float((expected - actual).abs().max().item())
            raise AssertionError(
                f"workspace changed at update {update}: {name}; "
                f"max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.abs().max().cpu())


def build_model(arm, device):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"unexpected base checkpoint SHA256: {sha}")

    base_reference = clone_state(base)
    digest = state_digest(base_reference)
    if digest != EXPECTED_BASE_DIGEST:
        raise RuntimeError(f"unexpected historical base digest: {digest}")

    model = RecallBridgeWorkingMemoryInfant.grow_from(
        base,
        cfg,
        arm=arm,
    ).to(device)
    model.train(True)
    model.base.eval()

    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML autograd compatibility only; never optimizer-owned.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = model.trainable_parameters()

    base_ids = {id(p) for p in model.base.parameters()}
    addition_ids = {id(p) for p in additions}
    workspace_ids = {
        id(p) for p in model.base.nucleus.workspace.parameters()
    }

    if not base_ids.isdisjoint(addition_ids):
        raise AssertionError("historical base leaked into optimizer")
    if not workspace_ids.isdisjoint(addition_ids):
        raise AssertionError("workspace leaked into optimizer")

    return model, cfg, sha, base_reference, additions


def clear_boundary(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)
    for p in model.base.nucleus.workspace.parameters():
        if p.grad is not None:
            raise AssertionError(
                "workspace gradient tensor remained at update boundary"
            )


def condition_for_update(update: int, schedule_seed: int) -> int:
    """Deterministic no-replacement schedule within each 1600-condition epoch."""
    if update < 1:
        raise ValueError("update must be >= 1")

    train_ids = list(TRAIN_CONDITION_RANGE)
    epoch = (update - 1) // len(train_ids)
    pos = (update - 1) % len(train_ids)

    rng = random.Random(int(schedule_seed) + 1_000_003 * epoch)
    rng.shuffle(train_ids)
    return int(train_ids[pos])


def decision_metrics(logits, teacher):
    logits1 = logits[0]
    full_pred = int(torch.argmax(logits1).item())
    left_logit = float(logits1[LEFT].detach().cpu())
    right_logit = float(logits1[RIGHT].detach().cpu())
    d = right_logit - left_logit
    side_pred = RIGHT if d > 0.0 else LEFT
    return {
        "full_pred": full_pred,
        "full_pred_name": ACTION_NAMES[full_pred],
        "full_correct": bool(full_pred == teacher),
        "lateral_top_choice": bool(full_pred in (LEFT, RIGHT)),
        "side_pred": int(side_pred),
        "side_pred_name": ACTION_NAMES[side_pred],
        "side_correct": bool(side_pred == teacher),
        "right_minus_left": float(d),
        "correct_side_margin": float(d if teacher == RIGHT else -d),
    }


def run_member(
    model,
    cfg,
    device,
    *,
    condition_id,
    cue,
    decision_fraction,
    world_loss_weight,
):
    env = VariedCuedDetourWorld(
        condition_id,
        cue=cue,
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)

    all_motor_losses = []
    world_losses = []
    decision_loss = None
    decision_info = None
    decision_step = None
    memory_max = 0.0
    recall_max = 0.0
    multiplier_sum = 0.0
    multiplier_count = 0
    graph_entry_step = None
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(obs, device, cfg)

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=True,
            use_recall=True,
        )

        if out["memory_next"].grad_fn is None:
            raise AssertionError(f"memory writer detached at step {steps + 1}")

        if steps == 0:
            if out["hidden"].grad_fn is not None:
                raise AssertionError(
                    "unexpected recurrent writer path on first step"
                )
        else:
            if out["hidden"].grad_fn is None:
                raise AssertionError(
                    f"recurrent graph failed to enter through memory "
                    f"at step {steps + 1}"
                )
            if graph_entry_step is None:
                graph_entry_step = steps + 1

        if tuple(out["memory_next"].shape) != (1, int(cfg.memory_dim)):
            raise AssertionError("working-memory shape changed")

        teacher = int(env.teacher_action())
        target = torch.tensor([teacher], dtype=torch.long, device=device)

        ce = F.cross_entropy(out["logits"], target)
        all_motor_losses.append(ce)

        if decision_loss is None and teacher in (LEFT, RIGHT):
            if env.steps < env.cue_steps:
                raise AssertionError("decision occurred while cue visible")
            decision_loss = ce
            decision_step = int(env.steps)
            decision_info = decision_metrics(out["logits"], teacher)

        action_oh = one_hot_action(target, cfg.action_dim)
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, _reward, _done, _info = env.step(teacher)
        _, next_context_dyn = _context_tensor(obs2, device, cfg)
        world_losses.append(
            F.smooth_l1_loss(
                world_out["next_context"],
                next_context_dyn,
            )
        )

        memory_max = max(
            memory_max,
            float(
                out["memory_next"].detach().abs().max().cpu()
            ),
        )
        recall_max = max(
            recall_max,
            float(
                out["recall_residual"].detach().abs().max().cpu()
            ),
        )
        if out["recall_multiplier"] is not None:
            multiplier_sum += float(
                out["recall_multiplier"].detach().mean().cpu()
            )
            multiplier_count += 1

        hidden = out["hidden"]
        memory = out["memory_next"]
        prev_action = (target + 1).long()
        obs = obs2
        steps += 1

        if steps > 160:
            raise RuntimeError("episode exceeded safety bound")

    if graph_entry_step != 2:
        raise AssertionError(
            f"expected delayed recurrent graph entry at 2, "
            f"got {graph_entry_step}"
        )
    if decision_loss is None or decision_info is None:
        raise AssertionError("no delayed cue-dependent decision found")

    all_motor_mean = torch.stack(all_motor_losses).mean()
    df = float(decision_fraction)
    motor_objective = (
        (1.0 - df) * all_motor_mean
        + df * decision_loss
    )
    world_mean = torch.stack(world_losses).mean()
    total = motor_objective + float(world_loss_weight) * world_mean

    return {
        "total": total,
        "motor_objective": motor_objective,
        "all_motor_mean": all_motor_mean,
        "decision_loss": decision_loss,
        "world_mean": world_mean,
        "steps": steps,
        "decision_step": decision_step,
        "decision": decision_info,
        "memory_max": memory_max,
        "recall_max": recall_max,
        "recall_multiplier_mean": (
            multiplier_sum / multiplier_count
            if multiplier_count
            else None
        ),
    }


def run_pair(
    model,
    cfg,
    device,
    *,
    condition_id,
    decision_fraction,
    world_loss_weight,
):
    left = run_member(
        model,
        cfg,
        device,
        condition_id=condition_id,
        cue="left",
        decision_fraction=decision_fraction,
        world_loss_weight=world_loss_weight,
    )
    right = run_member(
        model,
        cfg,
        device,
        condition_id=condition_id,
        cue="right",
        decision_fraction=decision_fraction,
        world_loss_weight=world_loss_weight,
    )

    if left["steps"] != right["steps"]:
        raise AssertionError("counterfactual pair teacher path lengths differ")
    if left["decision_step"] != right["decision_step"]:
        raise AssertionError("counterfactual pair decision timing differs")

    keys = (
        "total",
        "motor_objective",
        "all_motor_mean",
        "decision_loss",
        "world_mean",
    )
    means = {
        k: 0.5 * (left[k] + right[k])
        for k in keys
    }
    return {
        **means,
        "left": left,
        "right": right,
    }


def cpu_state_dict(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    update,
    base_sha,
    base_digest,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "arm": args.arm,
        "base_checkpoint": BASE_CHECKPOINT,
        "base_checkpoint_sha256": base_sha,
        "base_digest": base_digest,
        "update": int(update),
        "lr": float(args.lr),
        "schedule_seed": int(args.schedule_seed),
        "init_seed": int(args.init_seed),
        "decision_fraction": float(args.decision_fraction),
        "world_loss_weight": float(args.world_loss_weight),
        "train_condition_range": [
            TRAIN_CONDITION_RANGE.start,
            TRAIN_CONDITION_RANGE.stop,
        ],
        "memory": cpu_state_dict(model.memory),
        "recall": cpu_state_dict(model.recall),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path,
    *,
    model,
    optimizer,
    expected_base_sha,
    args,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("resume checkpoint format mismatch")
    if payload.get("arm") != args.arm:
        raise RuntimeError("resume checkpoint arm mismatch")
    if payload.get("base_checkpoint_sha256") != expected_base_sha:
        raise RuntimeError("resume checkpoint base SHA mismatch")

    checks = {
        "schedule_seed": int(args.schedule_seed),
        "init_seed": int(args.init_seed),
        "decision_fraction": float(args.decision_fraction),
        "world_loss_weight": float(args.world_loss_weight),
    }
    for key, expected in checks.items():
        if payload[key] != expected:
            raise RuntimeError(
                f"resume mismatch {key}: {payload[key]!r} != {expected!r}"
            )

    model.memory.load_state_dict(payload["memory"])
    model.recall.load_state_dict(payload["recall"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["update"])


@torch.no_grad()
def assert_disabled_exact_base(model, cfg, device, condition_id=3):
    env = VariedCuedDetourWorld(
        condition_id,
        cue="left",
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden_arm = model.initial_state(1, device)
    hidden_plain = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        disabled = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden_arm,
            use_memory=False,
            use_recall=False,
        )
        plain = model.historical_step(
            image,
            context,
            torch.zeros_like(memory),
            prev_action,
            hidden_plain,
        )

        comparisons = (
            ("hidden", disabled["hidden"], plain["hidden"]),
            (
                "workspace",
                disabled["workspace_for_world"],
                plain["historical_workspace"],
            ),
            ("logits", disabled["logits"], plain["historical_logits"]),
        )
        for name, got, expected in comparisons:
            if not torch.equal(got, expected):
                diff = float((got - expected).abs().max().cpu())
                raise AssertionError(
                    f"disabled exact-base parity failed {name}; diff={diff}"
                )

        teacher = int(env.teacher_action())
        obs, _, _, _ = env.step(teacher)
        prev_action = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden_arm = disabled["hidden"]
        hidden_plain = plain["hidden"]
        memory = disabled["memory_next"]



def prepare_training_log_for_resume(
    log_path,
    *,
    start_update,
    arm,
    schedule_seed,
):
    """
    Make JSONL exactly consistent with the portable checkpoint.

    If an interrupted attempt wrote updates after the last saved checkpoint,
    preserve the entire pre-repair JSONL as a sibling backup, then remove only
    that uncheckpointed tail.  Never silently discard evidence.
    """
    log_path = Path(log_path)

    if start_update == 0:
        if log_path.exists() and log_path.stat().st_size:
            raise RuntimeError(
                f"refusing fresh run over non-empty training log: {log_path}"
            )
        return {
            "retained": 0,
            "trimmed": 0,
            "backup": None,
        }

    if not log_path.is_file():
        raise RuntimeError(
            f"resume checkpoint exists but training log is missing: {log_path}"
        )

    rows = []
    raw_lines = log_path.read_text().splitlines()
    for lineno, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise RuntimeError(
                f"invalid JSONL at line {lineno}: {exc}"
            ) from exc
        rows.append(row)

    updates = [int(r["update"]) for r in rows]
    if len(updates) != len(set(updates)):
        raise RuntimeError(
            "training log already contains duplicate update numbers; "
            "manual reconciliation required"
        )

    retained = [r for r in rows if int(r["update"]) <= start_update]
    tail = [r for r in rows if int(r["update"]) > start_update]

    expected_updates = list(range(1, start_update + 1))
    got_updates = [int(r["update"]) for r in retained]
    if got_updates != expected_updates:
        raise RuntimeError(
            f"resume log prefix is not exactly 1..{start_update}"
        )

    for r in retained:
        update = int(r["update"])
        if r.get("arm") != arm:
            raise RuntimeError(
                f"resume log arm mismatch at update {update}: {r.get('arm')}"
            )
        expected_condition = condition_for_update(update, schedule_seed)
        if int(r["condition_id"]) != expected_condition:
            raise RuntimeError(
                f"resume schedule mismatch update {update}: "
                f"{r['condition_id']} != {expected_condition}"
            )

    backup = None
    if tail:
        stamp = int(time.time())
        backup = log_path.with_name(
            f"{log_path.stem}.pre_resume_trim_u{start_update:06d}_{stamp}"
            f"{log_path.suffix}"
        )
        backup.write_text(
            "\n".join(raw_lines) + ("\n" if raw_lines else "")
        )
        with log_path.open("w") as f:
            for r in retained:
                f.write(json.dumps(r) + "\n")

    return {
        "retained": len(retained),
        "trimmed": len(tail),
        "backup": None if backup is None else str(backup),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=("linear", "gated"),
    )
    ap.add_argument("--updates", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--schedule-seed", type=int, default=73191)
    ap.add_argument("--init-seed", type=int, default=12345)
    ap.add_argument("--decision-fraction", type=float, default=0.50)
    ap.add_argument("--world-loss-weight", type=float, default=0.25)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--verify-every", type=int, default=5)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if not (0.0 < args.decision_fraction < 1.0):
        raise ValueError("--decision-fraction must be in (0,1)")
    if args.world_loss_weight < 0.0:
        raise ValueError("--world-loss-weight must be >= 0")

    torch.manual_seed(args.init_seed)
    device = torch_directml.device()

    (
        model,
        cfg,
        base_sha,
        base_reference,
        additions,
    ) = build_model(args.arm, device)

    optimizer = DMLSafeAdam(additions, lr=args.lr)
    base_digest_initial = state_digest(base_reference)

    start_update = 0
    if args.resume:
        start_update = load_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            expected_base_sha=base_sha,
            args=args,
        )
        assert_state_exact(
            base_reference,
            model.base,
            "base changed while loading resume",
        )

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "training.jsonl"

    if start_update > args.updates:
        raise RuntimeError(
            f"resume update {start_update} exceeds target {args.updates}"
        )

    log_repair = prepare_training_log_for_resume(
        log_path,
        start_update=start_update,
        arm=args.arm,
        schedule_seed=args.schedule_seed,
    )

    print("WORKING_MEMORY_RECALL_TRAINING_V2")
    print("arm:", args.arm)
    print("device:", device)
    print("optimizer: DMLSafeAdam")
    print("base_checkpoint_sha256:", base_sha)
    print("base_digest_initial:", base_digest_initial)
    print(
        "addition_parameter_count:",
        sum(p.numel() for p in optimizer.params),
    )
    print(
        "writer_parameter_count:",
        sum(p.numel() for p in model.memory.parameters()),
    )
    print(
        "recall_parameter_count:",
        sum(p.numel() for p in model.recall.parameters()),
    )
    print("workspace_in_optimizer: false")
    print("counterfactual_pair_per_update: true")
    print(
        "train_condition_range:",
        f"{TRAIN_CONDITION_RANGE.start}:{TRAIN_CONDITION_RANGE.stop}",
    )
    print("without_replacement_within_epoch: true")
    print("decision_fraction:", args.decision_fraction)
    print("world_loss_weight:", args.world_loss_weight)
    print("schedule_seed:", args.schedule_seed)
    print("init_seed:", args.init_seed)
    print("starting_update:", start_update)
    print("target_updates:", args.updates)
    print("state_dir:", state_dir)
    print("resume_log_retained_records:", log_repair["retained"])
    print("resume_log_trimmed_uncheckpointed_records:", log_repair["trimmed"])
    print("resume_log_backup:", log_repair["backup"])

    mode = "a" if start_update else "w"
    with log_path.open(mode, buffering=1) as log:
        for update in range(start_update + 1, args.updates + 1):
            t0 = time.time()
            clear_boundary(model, optimizer)

            condition_id = condition_for_update(
                update,
                args.schedule_seed,
            )
            if condition_id not in TRAIN_CONDITION_RANGE:
                raise AssertionError("trainer escaped frozen train split")

            workspace_before = clone_workspace(model)

            pair = run_pair(
                model,
                cfg,
                device,
                condition_id=condition_id,
                decision_fraction=args.decision_fraction,
                world_loss_weight=args.world_loss_weight,
            )

            loss = pair["total"]
            loss.backward()

            candidate_grad = grad_max(model.memory.candidate.weight.grad)
            write_gate_grad = grad_max(model.memory.write_gate.weight.grad)
            recall_projection_grad = grad_max(
                model.recall.projection.weight.grad
            )

            recall_gate_weight_grad = None
            recall_gate_bias_grad = None
            if args.arm == "gated":
                recall_gate_weight_grad = grad_max(model.recall.gate.weight.grad)
                recall_gate_bias_grad = grad_max(model.recall.gate.bias.grad)

            if candidate_grad is None or candidate_grad <= 0.0:
                raise RuntimeError("missing candidate gradient")
            if (
                update > 1
                and (write_gate_grad is None or write_gate_grad <= 0.0)
            ):
                raise RuntimeError("missing write-gate gradient")
            if update == 1:
                if recall_projection_grad not in (None, 0.0):
                    raise RuntimeError(
                        "recall projection learned before memory moved"
                    )
            if (
                update > 1
                and (
                    recall_projection_grad is None
                    or recall_projection_grad <= 0.0
                )
            ):
                raise RuntimeError("missing recall projection gradient")
            if args.arm == "gated" and update >= 3:
                if (
                    recall_gate_weight_grad is None
                    or recall_gate_weight_grad <= 0.0
                ):
                    raise RuntimeError("missing recall gate weight gradient")
                if (
                    recall_gate_bias_grad is None
                    or recall_gate_bias_grad <= 0.0
                ):
                    raise RuntimeError("missing recall gate bias gradient")

            ws_grads = [
                p.grad
                for p in model.base.nucleus.workspace.parameters()
                if p.grad is not None
            ]
            if not ws_grads:
                raise RuntimeError(
                    "missing workspace compatibility gradients"
                )
            workspace_grad_max = max(grad_max(g) for g in ws_grads)

            optimizer.step()

            assert_workspace_exact(
                workspace_before,
                model,
                update,
            )

            full_verified = False
            if (
                update % args.verify_every == 0
                or update == args.updates
            ):
                assert_state_exact(
                    base_reference,
                    model.base,
                    f"historical base changed at update {update}",
                )
                full_verified = True

            left = pair["left"]
            right = pair["right"]

            recall_multiplier_mean = None
            if args.arm == "gated":
                vals = [
                    x
                    for x in (
                        left["recall_multiplier_mean"],
                        right["recall_multiplier_mean"],
                    )
                    if x is not None
                ]
                recall_multiplier_mean = sum(vals) / len(vals)

            record = {
                "update": int(update),
                "arm": args.arm,
                "condition_id": int(condition_id),
                "epoch": int(
                    (update - 1) // len(TRAIN_CONDITION_RANGE)
                ),
                "pair_steps_each": int(left["steps"]),
                "decision_step": int(left["decision_step"]),
                "loss": float(loss.detach().cpu()),
                "motor_objective": float(
                    pair["motor_objective"].detach().cpu()
                ),
                "all_motor_mean": float(
                    pair["all_motor_mean"].detach().cpu()
                ),
                "decision_loss": float(
                    pair["decision_loss"].detach().cpu()
                ),
                "world_loss": float(
                    pair["world_mean"].detach().cpu()
                ),
                "left_decision": left["decision"],
                "right_decision": right["decision"],
                "left_memory_max": float(left["memory_max"]),
                "right_memory_max": float(right["memory_max"]),
                "left_recall_max": float(left["recall_max"]),
                "right_recall_max": float(right["recall_max"]),
                "recall_multiplier_mean": recall_multiplier_mean,
                "candidate_grad_max": float(candidate_grad),
                "write_gate_grad_max": float(
                    0.0 if write_gate_grad is None else write_gate_grad
                ),
                "recall_projection_grad_max": float(
                    0.0
                    if recall_projection_grad is None
                    else recall_projection_grad
                ),
                "recall_gate_weight_grad_max": (
                    None
                    if recall_gate_weight_grad is None
                    else float(recall_gate_weight_grad)
                ),
                "recall_gate_bias_grad_max": (
                    None
                    if recall_gate_bias_grad is None
                    else float(recall_gate_bias_grad)
                ),
                "workspace_grad_max": float(workspace_grad_max),
                "workspace_exact": True,
                "full_base_verified": bool(full_verified),
                "elapsed_sec": float(time.time() - t0),
            }
            log.write(json.dumps(record) + "\n")

            if (
                update == 1
                or update % 5 == 0
                or update == args.updates
            ):
                ld = left["decision"]
                rd = right["decision"]
                mult = (
                    "-"
                    if recall_multiplier_mean is None
                    else f"{recall_multiplier_mean:.4f}"
                )
                print(
                    f"update={update:04d}/{args.updates} "
                    f"condition={condition_id:04d} "
                    f"loss={record['loss']:.6f} "
                    f"Lfull={int(ld['full_correct'])} "
                    f"Rfull={int(rd['full_correct'])} "
                    f"Llat={int(ld['lateral_top_choice'])} "
                    f"Rlat={int(rd['lateral_top_choice'])} "
                    f"Lside={int(ld['side_correct'])} "
                    f"Rside={int(rd['side_correct'])} "
                    f"Lmargin={ld['correct_side_margin']:+.4f} "
                    f"Rmargin={rd['correct_side_margin']:+.4f} "
                    f"mem={max(left['memory_max'], right['memory_max']):.5f} "
                    f"recall={max(left['recall_max'], right['recall_max']):.6f} "
                    f"mult={mult} "
                    f"proj_grad={record['recall_projection_grad_max']:.4g} "
                    f"base_verify="
                    f"{'yes' if full_verified else 'deferred'}"
                )

            if (
                update % args.save_every == 0
                or update == args.updates
            ):
                ckpt = (
                    state_dir
                    / f"{args.arm}_recall_u{update:06d}.portable.pt"
                )
                save_checkpoint(
                    ckpt,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    base_sha=base_sha,
                    base_digest=base_digest_initial,
                    args=args,
                )
                latest = state_dir / "latest.portable.pt"
                save_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    base_sha=base_sha,
                    base_digest=base_digest_initial,
                    args=args,
                )
                print("saved:", ckpt)

    assert_state_exact(
        base_reference,
        model.base,
        "historical base changed by end of training",
    )
    assert_disabled_exact_base(
        model,
        cfg,
        device,
        condition_id=3,
    )

    final_base_digest = state_digest(clone_state(model.base))
    if final_base_digest != base_digest_initial:
        raise AssertionError("final base digest mismatch")

    print("base_digest_final:", final_base_digest)
    print("memory_and_recall_disabled_exact_base_parity: true")
    print("portable_cpu_checkpoint: true")
    print("WORKING_MEMORY_RECALL_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
