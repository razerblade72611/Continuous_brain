#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.competitive_infant_brain import CompetitiveInfantBrain
from continuous_brain.cued_detour_world import CuedDetourWorld
from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
    _teacher_action,
)
from continuous_brain.infant import one_hot_action

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"


class DMLSafeAdam:
    """Adam using primitive tensor ops that stay on the qualified DirectML path."""

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
            raise RuntimeError("optimizer parameter count/state count mismatch")
        self.step_count = int(payload["step_count"])
        self.lr = float(payload["lr"])
        self.beta1 = float(payload["beta1"])
        self.beta2 = float(payload["beta2"])
        self.eps = float(payload["eps"])
        for dst, src, p in zip(self.state, payload["state"], self.params):
            dst["exp_avg"].copy_(src["exp_avg"].to(p.device))
            dst["exp_avg_sq"].copy_(src["exp_avg_sq"].to(p.device))


def clone_state_dict(module):
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
    assert reference.keys() == current.keys()
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(expected, actual):
            diff = None
            if torch.is_floating_point(expected):
                diff = float((expected - actual).abs().max().item())
            raise AssertionError(
                f"{label}: state changed: {name}; max_abs_diff={diff}"
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
                f"workspace changed at update {update}: {name}; max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.abs().max().cpu())


def build_model(device, shared=True):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"unexpected base checkpoint SHA256: {sha}")

    model = CompetitiveInfantBrain.grow_from(
        base,
        use_shared=shared,
    ).to(device)

    model.train(True)
    # Historical base must not acquire training-mode running-state drift.
    model.base.eval()

    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML compatibility only: calculate workspace gradients, but do not
    # give optimizer ownership of these parameters.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = (
        list(model.competition.parameters())
        + list(model.recurrent_residual.parameters())
        + list(model.motor_residual.parameters())
        + list(model.world_residual.parameters())
    )

    base_ids = {id(p) for p in model.base.parameters()}
    optimizer_ids = {id(p) for p in additions}
    workspace_ids = {
        id(p) for p in model.base.nucleus.workspace.parameters()
    }
    assert base_ids.isdisjoint(optimizer_ids)
    assert workspace_ids.isdisjoint(optimizer_ids)

    return model, cfg, sha, additions


def clear_update_boundary(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)
    for p in model.base.nucleus.workspace.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0


def train_episode(model, cfg, device, seed):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    assert hidden.grad_fn is None
    assert hidden.is_leaf

    prev_action = torch.zeros(1, dtype=torch.long, device=device)

    motor_losses = []
    world_losses = []
    teacher_return = 0.0
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(obs, device, cfg)
        memory = torch.zeros(1, cfg.memory_dim, device=device)

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_competition=True,
            compute_aux=False,
        )

        hidden = out["hidden"]
        assert hidden.grad_fn is not None, (
            f"recurrent graph detached at episode step {steps + 1}"
        )

        teacher_action = _teacher_action(env)
        target = torch.tensor(
            [teacher_action],
            dtype=torch.long,
            device=device,
        )

        motor_losses.append(
            F.cross_entropy(out["logits"], target)
        )

        action_oh = one_hot_action(target, cfg.action_dim)
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, reward, _done, _info = env.step(teacher_action)
        _, next_context_dyn = _context_tensor(obs2, device, cfg)

        world_losses.append(
            F.smooth_l1_loss(
                world_out["next_context"],
                next_context_dyn,
            )
        )

        teacher_return += float(reward)
        prev_action = (target + 1).long()
        obs = obs2
        steps += 1

    motor_loss = torch.stack(motor_losses).mean()
    world_loss = torch.stack(world_losses).mean()
    total_loss = motor_loss + world_loss

    return total_loss, motor_loss, world_loss, steps, teacher_return


@torch.no_grad()
def assert_disabled_exact_parity(model, cfg, device):
    env = CuedDetourWorld(
        seed=99001,
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    memory = torch.zeros(1, cfg.memory_dim, device=device)

    image = _image_tensor(env, cfg.image_size, device)
    context, _ = _context_tensor(obs, device, cfg)

    disabled = model.step(
        image,
        context,
        memory,
        prev_action,
        hidden,
        use_competition=False,
        compute_aux=False,
    )

    semantic = model.base.vision(image)
    spatial = model.base.spatial(image)
    fused = model.base.fuse_vision(semantic, spatial)
    streams = model.base.nucleus.project_streams(
        fused, context, memory, prev_action
    )
    core_input = torch.cat(tuple(streams), dim=-1)
    plain_hidden = model.base.nucleus.core(core_input, hidden)
    plain_workspace = model._workspace_from_hidden(plain_hidden)
    plain_logits = model.base.motor(plain_workspace)

    for name, got, expected in (
        ("hidden", disabled["hidden"], plain_hidden),
        ("workspace_for_world", disabled["workspace_for_world"], plain_workspace),
        ("logits", disabled["logits"], plain_logits),
    ):
        if not torch.equal(got, expected):
            diff = float((got - expected).abs().max().cpu())
            raise AssertionError(
                f"disabled exact parity failed for {name}; max_abs_diff={diff}"
            )


def save_checkpoint(path, model, optimizer, update, base_sha, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "competitive_broadcast_dml_training_v1",
        "base_checkpoint": BASE_CHECKPOINT,
        "base_checkpoint_sha256": base_sha,
        "update": int(update),
        "condition": args.condition,
        "lr": float(args.lr),
        "seed_base": int(args.seed_base),
        "init_seed": int(args.init_seed),
        "competition": model.competition.state_dict(),
        "recurrent_residual": model.recurrent_residual.state_dict(),
        "motor_residual": model.motor_residual.state_dict(),
        "world_residual": model.world_residual.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, expected_base_sha, expected_condition):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base_checkpoint_sha256"] != expected_base_sha:
        raise RuntimeError("resume checkpoint base SHA mismatch")
    if payload["condition"] != expected_condition:
        raise RuntimeError("resume checkpoint condition mismatch")

    model.competition.load_state_dict(payload["competition"])
    model.recurrent_residual.load_state_dict(payload["recurrent_residual"])
    model.motor_residual.load_state_dict(payload["motor_residual"])
    model.world_residual.load_state_dict(payload["world_residual"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["update"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed-base", type=int, default=82000)
    ap.add_argument("--init-seed", type=int, default=12345)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--verify-every", type=int, default=25)
    ap.add_argument(
        "--condition",
        choices=("shared", "independent"),
        default="shared",
    )
    ap.add_argument(
        "--state-dir",
        default="state/competitive_dml_shared",
    )
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.verify_every < 1:
        raise ValueError("--verify-every must be >= 1")

    warnings.simplefilter("always")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "training.jsonl"

    torch.manual_seed(args.init_seed)

    device = torch_directml.device()
    model, cfg, base_sha, additions = build_model(
        device,
        shared=(args.condition == "shared"),
    )

    optimizer = DMLSafeAdam(additions, lr=args.lr)

    base_reference = clone_state_dict(model.base)
    base_digest_initial = state_digest(base_reference)

    start_update = 0
    if args.resume:
        start_update = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            base_sha,
            args.condition,
        )
        assert_state_exact(
            base_reference,
            model.base,
            "base changed while loading resume checkpoint",
        )

    print("device:", device)
    print("condition:", args.condition)
    print("optimizer: DMLSafeAdam")
    print("base_checkpoint_sha256:", base_sha)
    print("base_digest_initial:", base_digest_initial)
    print("optimizer_parameter_count:",
          sum(p.numel() for p in optimizer.params))
    print("workspace_parameter_count:",
          sum(p.numel() for p in model.base.nucleus.workspace.parameters()))
    print("workspace_in_optimizer: false")
    print("init_seed:", args.init_seed)
    print("seed_base:", args.seed_base)
    print("starting_update:", start_update)
    print("target_updates:", args.updates)
    print("state_dir:", state_dir)

    with log_path.open("a", buffering=1) as log:
        for update in range(start_update + 1, args.updates + 1):
            t0 = time.time()
            clear_update_boundary(model, optimizer)

            workspace_before = clone_workspace(model)

            loss, motor_loss, world_loss, steps, teacher_return = train_episode(
                model,
                cfg,
                device,
                seed=args.seed_base + update,
            )

            loss.backward()

            ws_grads = [
                p.grad
                for p in model.base.nucleus.workspace.parameters()
                if p.grad is not None
            ]
            if not ws_grads:
                raise RuntimeError("missing workspace compatibility gradients")
            ws_grad = max(grad_max(g) for g in ws_grads)

            rr_grad = grad_max(model.recurrent_residual.linear.weight.grad)
            mr_grad = grad_max(model.motor_residual.linear.weight.grad)
            wr_grad = grad_max(model.world_residual.linear.weight.grad)
            competition_grads = [
                p.grad
                for p in model.competition.parameters()
                if p.grad is not None
            ]
            q_grad = (
                max(grad_max(g) for g in competition_grads)
                if competition_grads else None
            )

            optimizer.step()

            # Cheap, every-update invariant for the only historical module whose
            # parameters have requires_grad=True.
            assert_workspace_exact(workspace_before, model, update)

            full_verified = False
            if update % args.verify_every == 0 or update == args.updates:
                assert_state_exact(
                    base_reference,
                    model.base,
                    f"historical base changed at update {update}",
                )
                full_verified = True

            record = {
                "update": update,
                "steps": steps,
                "teacher_return": teacher_return,
                "loss": float(loss.detach().cpu()),
                "motor_loss": float(motor_loss.detach().cpu()),
                "world_loss": float(world_loss.detach().cpu()),
                "workspace_grad_max": ws_grad,
                "recurrent_residual_grad_max": rr_grad,
                "motor_residual_grad_max": mr_grad,
                "world_residual_grad_max": wr_grad,
                "selector_grad_max": 0.0 if q_grad is None else q_grad,
                "workspace_exact": True,
                "full_base_verified": full_verified,
                "elapsed_sec": time.time() - t0,
            }
            log.write(json.dumps(record) + "\n")

            if update == 1 or update % 10 == 0 or update == args.updates:
                print(
                    f"update={update:04d}/{args.updates} "
                    f"steps={steps} "
                    f"return={teacher_return:.6f} "
                    f"loss={record['loss']:.6f} "
                    f"motor={record['motor_loss']:.6f} "
                    f"world={record['world_loss']:.6f} "
                    f"q_grad={record['selector_grad_max']:.6g} "
                    f"workspace_exact=true "
                    f"base_verify={'yes' if full_verified else 'deferred'}"
                )

            if update % args.save_every == 0 or update == args.updates:
                ckpt = state_dir / f"competitive_{args.condition}_u{update:06d}.pt"
                save_checkpoint(
                    ckpt, model, optimizer, update, base_sha, args
                )
                latest = state_dir / "latest.pt"
                save_checkpoint(
                    latest, model, optimizer, update, base_sha, args
                )
                print("saved:", ckpt)

    assert_state_exact(
        base_reference,
        model.base,
        "historical base changed by end of training",
    )
    assert_disabled_exact_parity(model, cfg, device)

    final_base_digest = state_digest(clone_state_dict(model.base))
    if final_base_digest != base_digest_initial:
        raise AssertionError("final base digest mismatch")

    print("base_digest_final:", final_base_digest)
    print("baseline_disabled_exact_parity: true")
    print("TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
