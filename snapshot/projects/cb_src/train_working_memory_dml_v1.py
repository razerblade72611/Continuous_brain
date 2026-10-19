#!/usr/bin/env python3
"""
Production counterfactual-balanced working-memory trainer for Continuous Brain.

This trains ONLY the additive 64-D working-memory writer around the protected
14k InfantBrain.  Historical base parameters/buffers are never optimizer-owned.

Every optimizer update uses one structurally unique train condition twice:
LEFT + RIGHT counterfactual members, averaged before a single optimizer step.

The motor objective explicitly reserves half its weight for the first delayed
cue-dependent lateral decision.  This prevents the many cue-independent
locomotion steps from swamping the memory-relevant learning signal.

No dev/test structural condition is used by this trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import warnings
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
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
)
from continuous_brain.working_memory_infant import WorkingMemoryInfant


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
CHECKPOINT_FORMAT = "working_memory_dml_training_v1"


class DMLSafeAdam:
    """Adam expressed only with already-qualified DirectML-safe tensor ops."""

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

            g2 = g * g
            v.mul_(self.beta2)
            v.add_(g2, alpha=1.0 - self.beta2)

            denom = v.sqrt()
            denom.div_(sqrt_bc2)
            denom.add_(self.eps)

            update = m / denom
            p.add_(update, alpha=-step_size)

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
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
            raise RuntimeError(
                "optimizer parameter count/state count mismatch"
            )
        self.step_count = int(payload["step_count"])
        self.lr = float(payload["lr"])
        self.beta1 = float(payload["beta1"])
        self.beta2 = float(payload["beta2"])
        self.eps = float(payload["eps"])
        for dst, src, p in zip(
            self.state, payload["state"], self.params
        ):
            dst["exp_avg"].copy_(
                src["exp_avg"].to(p.device)
            )
            dst["exp_avg_sq"].copy_(
                src["exp_avg_sq"].to(p.device)
            )


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
                diff = float(
                    (expected - actual).abs().max().item()
                )
            raise AssertionError(
                f"{label}: {name} changed; "
                f"max_abs_diff={diff}"
            )


def clone_workspace(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in
        model.base.nucleus.workspace.state_dict().items()
    }


def assert_workspace_exact(reference, model, update):
    current = model.base.nucleus.workspace.state_dict()
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(expected, actual):
            diff = float(
                (expected - actual).abs().max().item()
            )
            raise AssertionError(
                f"workspace changed at update {update}: "
                f"{name}; max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.abs().max().cpu())


def build_model(device):
    base, cfg, _, sha = _load_base_checkpoint(
        BASE_CHECKPOINT
    )
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"unexpected base checkpoint SHA256: {sha}"
        )

    model = WorkingMemoryInfant.grow_from(
        base, cfg
    ).to(device)
    model.train(True)
    model.base.eval()

    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML compatibility only; never optimizer-owned.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = model.trainable_parameters()

    base_ids = {id(p) for p in model.base.parameters()}
    optimizer_ids = {id(p) for p in additions}
    workspace_ids = {
        id(p)
        for p in model.base.nucleus.workspace.parameters()
    }

    if not base_ids.isdisjoint(optimizer_ids):
        raise AssertionError(
            "historical base leaked into optimizer"
        )
    if not workspace_ids.isdisjoint(optimizer_ids):
        raise AssertionError(
            "workspace leaked into optimizer"
        )

    return model, cfg, sha, additions


def clear_boundary(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(
        set_to_none=True
    )
    for p in model.base.nucleus.workspace.parameters():
        if p.grad is not None:
            raise AssertionError(
                "workspace gradient tensor remained "
                "at update boundary"
            )


def condition_for_update(update: int, schedule_seed: int) -> int:
    """
    Deterministic no-replacement schedule within each 1600-condition epoch.
    Resume requires no hidden RNG state.
    """
    if update < 1:
        raise ValueError("update must be >= 1")

    train_ids = list(TRAIN_CONDITION_RANGE)
    epoch = (update - 1) // len(train_ids)
    pos = (update - 1) % len(train_ids)

    rng = random.Random(
        int(schedule_seed) + 1_000_003 * epoch
    )
    rng.shuffle(train_ids)
    return int(train_ids[pos])


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
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )

    if hidden.grad_fn is not None or not hidden.is_leaf:
        raise AssertionError(
            "initial hidden must be a fresh leaf"
        )

    all_motor_losses = []
    world_losses = []
    decision_loss = None
    decision_margin = None
    decision_step = None
    memory_max = 0.0
    gate_sum = 0.0
    gate_count = 0
    graph_entry_step = None
    steps = 0

    while not env.done:
        image = _image_tensor(
            env, cfg.image_size, device
        )
        context, context_dyn = _context_tensor(
            obs, device, cfg
        )

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=True,
        )

        hidden = out["hidden"]

        # Qualified delayed graph topology:
        # step 1 hidden has no writer path; memory_next does.
        # step 2+ recurrent state must inherit writer influence.
        if out["memory_next"].grad_fn is None:
            raise AssertionError(
                f"memory writer detached at step {steps + 1}"
            )

        if steps == 0:
            if hidden.grad_fn is not None:
                raise AssertionError(
                    "unexpected recurrent trainable path "
                    "on step 1"
                )
        else:
            if hidden.grad_fn is None:
                raise AssertionError(
                    f"recurrent memory path detached "
                    f"at step {steps + 1}"
                )
            if graph_entry_step is None:
                graph_entry_step = steps + 1

        if tuple(out["memory_next"].shape) != (
            1, int(cfg.memory_dim)
        ):
            raise AssertionError(
                "working-memory shape changed"
            )

        teacher = int(env.teacher_action())
        target = torch.tensor(
            [teacher],
            dtype=torch.long,
            device=device,
        )

        ce = F.cross_entropy(
            out["logits"], target
        )
        all_motor_losses.append(ce)

        # First lateral teacher action is the delayed cue-dependent choice.
        if (
            decision_loss is None
            and teacher in (LEFT, RIGHT)
        ):
            if env.steps < env.cue_steps:
                raise AssertionError(
                    "decision occurred while cue visible"
                )
            decision_loss = ce
            decision_step = int(env.steps)

            logits = out["logits"][0]
            correct = logits[teacher]
            wrong = logits[
                RIGHT if teacher == LEFT else LEFT
            ]
            decision_margin = correct - wrong

        action_oh = one_hot_action(
            target, cfg.action_dim
        )
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, _reward, _done, _info = env.step(
            teacher
        )
        _, next_context_dyn = _context_tensor(
            obs2, device, cfg
        )
        world_losses.append(
            F.smooth_l1_loss(
                world_out["next_context"],
                next_context_dyn,
            )
        )

        memory_max = max(
            memory_max,
            float(
                out["memory_next"]
                .detach()
                .abs()
                .max()
                .cpu()
            ),
        )
        gate_sum += float(
            out["memory_write_gate"]
            .detach()
            .mean()
            .cpu()
        )
        gate_count += 1

        prev_action = (target + 1).long()
        memory = out["memory_next"]
        obs = obs2
        steps += 1

        if steps > 160:
            raise RuntimeError(
                "episode exceeded safety bound"
            )

    if graph_entry_step != 2:
        raise AssertionError(
            f"expected recurrent graph entry at 2, "
            f"got {graph_entry_step}"
        )
    if decision_loss is None:
        raise AssertionError(
            "no delayed cue-dependent decision found"
        )

    all_motor_mean = torch.stack(
        all_motor_losses
    ).mean()

    df = float(decision_fraction)
    motor_objective = (
        (1.0 - df) * all_motor_mean
        + df * decision_loss
    )

    world_mean = torch.stack(
        world_losses
    ).mean()

    total = (
        motor_objective
        + float(world_loss_weight) * world_mean
    )

    return {
        "total": total,
        "motor_objective": motor_objective,
        "all_motor_mean": all_motor_mean,
        "decision_loss": decision_loss,
        "decision_margin": decision_margin,
        "world_mean": world_mean,
        "steps": steps,
        "decision_step": decision_step,
        "memory_max": memory_max,
        "gate_mean": (
            gate_sum / gate_count
            if gate_count else 0.0
        ),
        "graph_entry_step": graph_entry_step,
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
        raise AssertionError(
            "counterfactual pair teacher path lengths differ"
        )
    if left["decision_step"] != right["decision_step"]:
        raise AssertionError(
            "counterfactual pair decision timing differs"
        )

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
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    payload = {
        "format": CHECKPOINT_FORMAT,
        "base_checkpoint": BASE_CHECKPOINT,
        "base_checkpoint_sha256": base_sha,
        "base_digest": base_digest,
        "update": int(update),
        "lr": float(args.lr),
        "schedule_seed": int(args.schedule_seed),
        "init_seed": int(args.init_seed),
        "decision_fraction": float(
            args.decision_fraction
        ),
        "world_loss_weight": float(
            args.world_loss_weight
        ),
        "train_condition_range": [
            TRAIN_CONDITION_RANGE.start,
            TRAIN_CONDITION_RANGE.stop,
        ],
        "memory": cpu_state_dict(model.memory),
        "optimizer": optimizer.state_dict(),
    }

    # Every tensor in payload is on CPU by construction, avoiding the
    # DirectML checkpoint portability issue encountered in the competitive
    # experiment.
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
        raise RuntimeError(
            "resume checkpoint format mismatch"
        )
    if (
        payload["base_checkpoint_sha256"]
        != expected_base_sha
    ):
        raise RuntimeError(
            "resume checkpoint base SHA mismatch"
        )

    checks = {
        "schedule_seed": int(args.schedule_seed),
        "init_seed": int(args.init_seed),
        "decision_fraction": float(
            args.decision_fraction
        ),
        "world_loss_weight": float(
            args.world_loss_weight
        ),
    }
    for key, expected in checks.items():
        got = payload[key]
        if got != expected:
            raise RuntimeError(
                f"resume mismatch {key}: "
                f"{got!r} != {expected!r}"
            )

    model.memory.load_state_dict(
        payload["memory"]
    )
    optimizer.load_state_dict(
        payload["optimizer"]
    )
    return int(payload["update"])


@torch.no_grad()
def assert_memory_disabled_exact_base(
    model, cfg, device, condition_id
):
    """
    After training, disabling the new memory mechanism must route exactly
    through the historical zero-memory base behavior.
    """
    env = VariedCuedDetourWorld(
        condition_id,
        cue="left",
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden_mem = model.initial_state(1, device)
    hidden_plain = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )

    steps = 0
    while not env.done:
        image = _image_tensor(
            env, cfg.image_size, device
        )
        context, _ = _context_tensor(
            obs, device, cfg
        )

        disabled = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden_mem,
            use_memory=False,
        )
        plain = model.base_step(
            image,
            context,
            torch.zeros_like(memory),
            prev_action,
            hidden_plain,
        )

        for name in (
            "hidden",
            "workspace_for_world",
            "logits",
        ):
            got = disabled[name]
            expected = plain[name]
            if not torch.equal(got, expected):
                diff = float(
                    (got - expected)
                    .abs()
                    .max()
                    .cpu()
                )
                raise AssertionError(
                    f"memory-disabled base parity "
                    f"failed for {name}; diff={diff}"
                )

        teacher = int(env.teacher_action())
        obs, _, _, _ = env.step(teacher)

        prev_action = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden_mem = disabled["hidden"]
        hidden_plain = plain["hidden"]
        memory = disabled["memory_next"]
        steps += 1

    if steps < 3:
        raise AssertionError(
            "disabled parity trajectory too short"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--updates", type=int, default=1600
    )
    ap.add_argument(
        "--lr", type=float, default=1e-4
    )
    ap.add_argument(
        "--schedule-seed",
        type=int,
        default=73191,
    )
    ap.add_argument(
        "--init-seed",
        type=int,
        default=12345,
    )
    ap.add_argument(
        "--decision-fraction",
        type=float,
        default=0.50,
        help=(
            "fraction of motor objective reserved "
            "for first delayed cue-dependent decision"
        ),
    )
    ap.add_argument(
        "--world-loss-weight",
        type=float,
        default=0.25,
    )
    ap.add_argument(
        "--save-every", type=int, default=100
    )
    ap.add_argument(
        "--verify-every", type=int, default=50
    )
    ap.add_argument(
        "--state-dir",
        default="state/working_memory_m1/train_v1",
    )
    ap.add_argument(
        "--resume", default=None
    )
    args = ap.parse_args()

    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if not (0.0 < args.decision_fraction < 1.0):
        raise ValueError(
            "--decision-fraction must be in (0,1)"
        )
    if args.world_loss_weight < 0.0:
        raise ValueError(
            "--world-loss-weight must be >= 0"
        )
    if args.save_every < 1:
        raise ValueError(
            "--save-every must be >= 1"
        )
    if args.verify_every < 1:
        raise ValueError(
            "--verify-every must be >= 1"
        )

    warnings.simplefilter("always")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True, exist_ok=True
    )
    log_path = state_dir / "training.jsonl"

    torch.manual_seed(args.init_seed)

    device = torch_directml.device()
    model, cfg, base_sha, additions = build_model(
        device
    )
    optimizer = DMLSafeAdam(
        additions, lr=args.lr
    )

    base_reference = clone_state(model.base)
    base_digest_initial = state_digest(
        base_reference
    )

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

    print("WORKING_MEMORY_TRAINING_V1")
    print("device:", device)
    print("optimizer: DMLSafeAdam")
    print(
        "base_checkpoint_sha256:", base_sha
    )
    print(
        "base_digest_initial:",
        base_digest_initial,
    )
    print(
        "memory_trainable_parameters:",
        sum(p.numel() for p in optimizer.params),
    )
    print(
        "workspace_parameter_count:",
        sum(
            p.numel()
            for p in
            model.base.nucleus.workspace.parameters()
        ),
    )
    print("workspace_in_optimizer: false")
    print(
        "counterfactual_pair_per_update: true"
    )
    print(
        "train_condition_range:",
        f"{TRAIN_CONDITION_RANGE.start}:"
        f"{TRAIN_CONDITION_RANGE.stop}",
    )
    print(
        "without_replacement_within_epoch: true"
    )
    print(
        "decision_fraction:",
        args.decision_fraction,
    )
    print(
        "world_loss_weight:",
        args.world_loss_weight,
    )
    print(
        "schedule_seed:",
        args.schedule_seed,
    )
    print("init_seed:", args.init_seed)
    print("starting_update:", start_update)
    print("target_updates:", args.updates)
    print("state_dir:", state_dir)

    mode = "a" if start_update else "w"
    with log_path.open(
        mode, buffering=1
    ) as log:
        for update in range(
            start_update + 1,
            args.updates + 1,
        ):
            t0 = time.time()
            clear_boundary(model, optimizer)

            condition_id = condition_for_update(
                update, args.schedule_seed
            )
            if condition_id not in TRAIN_CONDITION_RANGE:
                raise AssertionError(
                    "trainer escaped frozen train split"
                )

            workspace_before = clone_workspace(
                model
            )

            pair = run_pair(
                model,
                cfg,
                device,
                condition_id=condition_id,
                decision_fraction=
                    args.decision_fraction,
                world_loss_weight=
                    args.world_loss_weight,
            )

            loss = pair["total"]
            loss.backward()

            candidate_grad = grad_max(
                model.memory.candidate.weight.grad
            )
            gate_grad = grad_max(
                model.memory.write_gate.weight.grad
            )

            if (
                candidate_grad is None
                or candidate_grad <= 0.0
            ):
                raise RuntimeError(
                    "missing candidate gradient"
                )

            # Gate is mathematically allowed to be zero only on the first
            # update because candidate and memory start exactly at zero.
            if (
                update > 1
                and (
                    gate_grad is None
                    or gate_grad <= 0.0
                )
            ):
                raise RuntimeError(
                    "missing write-gate gradient"
                )

            ws_grads = [
                p.grad
                for p in
                model.base.nucleus.workspace.parameters()
                if p.grad is not None
            ]
            if not ws_grads:
                raise RuntimeError(
                    "missing workspace compatibility gradients"
                )
            ws_grad = max(
                grad_max(g) for g in ws_grads
            )

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
                    f"historical base changed "
                    f"at update {update}",
                )
                full_verified = True

            left = pair["left"]
            right = pair["right"]

            record = {
                "update": int(update),
                "condition_id": int(condition_id),
                "epoch": int(
                    (update - 1)
                    // len(TRAIN_CONDITION_RANGE)
                ),
                "pair_steps_each": int(
                    left["steps"]
                ),
                "decision_step": int(
                    left["decision_step"]
                ),
                "loss": float(
                    loss.detach().cpu()
                ),
                "motor_objective": float(
                    pair["motor_objective"]
                    .detach()
                    .cpu()
                ),
                "all_motor_mean": float(
                    pair["all_motor_mean"]
                    .detach()
                    .cpu()
                ),
                "decision_loss": float(
                    pair["decision_loss"]
                    .detach()
                    .cpu()
                ),
                "world_loss": float(
                    pair["world_mean"]
                    .detach()
                    .cpu()
                ),
                "left_decision_loss": float(
                    left["decision_loss"]
                    .detach()
                    .cpu()
                ),
                "right_decision_loss": float(
                    right["decision_loss"]
                    .detach()
                    .cpu()
                ),
                "left_decision_margin": float(
                    left["decision_margin"]
                    .detach()
                    .cpu()
                ),
                "right_decision_margin": float(
                    right["decision_margin"]
                    .detach()
                    .cpu()
                ),
                "left_memory_max": float(
                    left["memory_max"]
                ),
                "right_memory_max": float(
                    right["memory_max"]
                ),
                "left_gate_mean": float(
                    left["gate_mean"]
                ),
                "right_gate_mean": float(
                    right["gate_mean"]
                ),
                "candidate_grad_max": float(
                    candidate_grad
                ),
                "write_gate_grad_max": float(
                    0.0
                    if gate_grad is None
                    else gate_grad
                ),
                "workspace_grad_max": float(
                    ws_grad
                ),
                "workspace_exact": True,
                "full_base_verified":
                    bool(full_verified),
                "elapsed_sec": float(
                    time.time() - t0
                ),
            }

            log.write(
                json.dumps(record) + "\n"
            )

            if (
                update == 1
                or update % 10 == 0
                or update == args.updates
            ):
                print(
                    f"update={update:04d}/"
                    f"{args.updates} "
                    f"condition={condition_id:04d} "
                    f"steps={record['pair_steps_each']} "
                    f"loss={record['loss']:.6f} "
                    f"decision="
                    f"{record['decision_loss']:.6f} "
                    f"Lmargin="
                    f"{record['left_decision_margin']:+.4f} "
                    f"Rmargin="
                    f"{record['right_decision_margin']:+.4f} "
                    f"mem="
                    f"{max(record['left_memory_max'], record['right_memory_max']):.5f} "
                    f"cand_grad="
                    f"{record['candidate_grad_max']:.4g} "
                    f"gate_grad="
                    f"{record['write_gate_grad_max']:.4g} "
                    f"base_verify="
                    f"{'yes' if full_verified else 'deferred'}"
                )

            if (
                update % args.save_every == 0
                or update == args.updates
            ):
                ckpt = (
                    state_dir
                    / f"working_memory_u{update:06d}.portable.pt"
                )
                save_checkpoint(
                    ckpt,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    base_sha=base_sha,
                    base_digest=
                        base_digest_initial,
                    args=args,
                )
                latest = (
                    state_dir
                    / "latest.portable.pt"
                )
                save_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    base_sha=base_sha,
                    base_digest=
                        base_digest_initial,
                    args=args,
                )
                print("saved:", ckpt)

    assert_state_exact(
        base_reference,
        model.base,
        "historical base changed "
        "by end of training",
    )
    assert_memory_disabled_exact_base(
        model,
        cfg,
        device,
        condition_id=3,
    )

    final_base_digest = state_digest(
        clone_state(model.base)
    )
    if (
        final_base_digest
        != base_digest_initial
    ):
        raise AssertionError(
            "final base digest mismatch"
        )

    print(
        "base_digest_final:",
        final_base_digest,
    )
    print(
        "memory_disabled_exact_base_parity: true"
    )
    print(
        "portable_cpu_checkpoint: true"
    )
    print("WORKING_MEMORY_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
