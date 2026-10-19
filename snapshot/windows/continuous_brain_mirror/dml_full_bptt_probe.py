#!/usr/bin/env python3
import argparse
import hashlib
import math

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

CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"


class DMLSafeAdam:
    """Standard Adam equations using DirectML-safe primitive tensor ops."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = {}

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

        for p in self.params:
            g = p.grad
            if g is None:
                continue

            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }

            m = self.state[key]["exp_avg"]
            v = self.state[key]["exp_avg_sq"]

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


def clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def params_digest(params):
    h = hashlib.sha256()
    for i, p in enumerate(params):
        t = p.detach().cpu().contiguous()
        h.update(str(i).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_exact_equal(before, after, label):
    assert before.keys() == after.keys(), f"{label}: state keys changed"
    for k in before:
        if not torch.equal(before[k], after[k]):
            diff = (before[k] - after[k]).abs().max().item()
            raise AssertionError(
                f"{label}: {k} changed; max_abs_diff={diff}"
            )


def build_model(device):
    base, cfg, _, checkpoint_sha = _load_base_checkpoint(CHECKPOINT)
    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).to(device)
    model.train(True)

    # Historical base protected from optimizer updates.
    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML compatibility only.
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
    workspace_ids = {id(p) for p in model.base.nucleus.workspace.parameters()}

    assert base_ids.isdisjoint(optimizer_ids), (
        "historical base leaked into optimizer ownership"
    )
    assert workspace_ids.isdisjoint(optimizer_ids), (
        "workspace leaked into optimizer ownership"
    )

    return model, cfg, checkpoint_sha, additions


def clear_update_boundary_grads(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)

    for p in model.base.nucleus.workspace.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0, (
            "workspace gradient was not clear at update boundary"
        )


def run_teacher_episode(model, cfg, device, *, include_world_loss):
    env = CuedDetourWorld(
        seed=81501,
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)

    motor_losses = []
    world_losses = []
    hidden_history = []
    total_reward = 0.0
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
            f"hidden detached at step {steps + 1}"
        )
        hidden.retain_grad()
        hidden_history.append(hidden)

        teacher_action = _teacher_action(env)
        target = torch.tensor(
            [teacher_action],
            dtype=torch.long,
            device=device,
        )
        motor_losses.append(F.cross_entropy(out["logits"], target))

        action_oh = one_hot_action(target, cfg.action_dim)

        if include_world_loss:
            world_out = model.predict_world(
                out["workspace_for_world"],
                action_oh,
                base_context=context_dyn,
            )

        obs2, reward, _done, _info = env.step(teacher_action)

        if include_world_loss:
            _, next_context_dyn = _context_tensor(obs2, device, cfg)
            world_losses.append(
                F.smooth_l1_loss(
                    world_out["next_context"],
                    next_context_dyn,
                )
            )

        total_reward += float(reward)
        prev_action = (target + 1).long()
        obs = obs2
        steps += 1

    assert steps >= 3, "episode too short to test temporal credit"

    return {
        "steps": steps,
        "total_reward": total_reward,
        "motor_losses": motor_losses,
        "world_losses": world_losses,
        "hidden_history": hidden_history,
    }


def temporal_credit_probe(device):
    model, cfg, sha, additions = build_model(device)

    # No optimizer step in this mode.  We want the final timestep's motor loss
    # alone to send gradient into a much earlier hidden state.
    optimizer = DMLSafeAdam(additions, lr=1e-4)
    clear_update_boundary_grads(model, optimizer)

    episode = run_teacher_episode(
        model, cfg, device, include_world_loss=False
    )

    first_hidden = episode["hidden_history"][0]
    final_loss = episode["motor_losses"][-1]

    print("mode: temporal_credit")
    print("checkpoint_sha256:", sha)
    print("episode_steps:", episode["steps"])
    print("final_loss:", float(final_loss.detach().cpu()))
    print("starting final-timestep-only backward...")

    final_loss.backward()

    assert first_hidden.grad is not None, (
        "final timestep loss did not reach first hidden state"
    )
    assert torch.isfinite(first_hidden.grad).all().item(), (
        "non-finite temporal gradient"
    )
    temporal_grad_max = float(first_hidden.grad.abs().max().cpu())
    assert temporal_grad_max > 0.0, (
        "final timestep loss produced zero gradient at first hidden state"
    )

    print("first_hidden_grad_from_final_loss_max:", temporal_grad_max)
    print("FULL_BPTT_TEMPORAL_CREDIT_OK")


def training_objective_probe(device):
    model, cfg, sha, additions = build_model(device)
    optimizer = DMLSafeAdam(additions, lr=1e-4)

    base_before = clone_state(model.base)
    base_digest_before = state_digest(base_before)
    additions_digest_before = params_digest(additions)

    clear_update_boundary_grads(model, optimizer)

    episode = run_teacher_episode(
        model, cfg, device, include_world_loss=True
    )

    motor_loss = torch.stack(episode["motor_losses"]).mean()
    world_loss = torch.stack(episode["world_losses"]).mean()
    loss = motor_loss + world_loss

    print("mode: training_objective")
    print("checkpoint_sha256:", sha)
    print("episode_steps:", episode["steps"])
    print("teacher_return:", episode["total_reward"])
    print("motor_loss:", float(motor_loss.detach().cpu()))
    print("world_loss:", float(world_loss.detach().cpu()))
    print("total_loss:", float(loss.detach().cpu()))
    print("workspace_in_optimizer: false")
    print("starting full-episode backward...")

    loss.backward()

    ws_grads = [
        p.grad for p in model.base.nucleus.workspace.parameters()
        if p.grad is not None
    ]
    assert ws_grads, "workspace compatibility gradients missing"
    ws_grad_max = max(float(g.abs().max().cpu()) for g in ws_grads)
    assert ws_grad_max > 0.0

    rr = model.recurrent_residual.linear.weight.grad
    mr = model.motor_residual.linear.weight.grad
    wr = model.world_residual.linear.weight.grad

    assert rr is not None and torch.isfinite(rr).all().item()
    assert mr is not None and torch.isfinite(mr).all().item()
    assert wr is not None and torch.isfinite(wr).all().item()

    print("workspace_grad_max:", ws_grad_max)
    print("recurrent_residual_grad_max:",
          float(rr.abs().max().cpu()))
    print("motor_residual_grad_max:",
          float(mr.abs().max().cpu()))
    print("world_residual_grad_max:",
          float(wr.abs().max().cpu()))

    optimizer.step()

    base_after = clone_state(model.base)
    base_digest_after = state_digest(base_after)
    additions_digest_after = params_digest(additions)

    assert_exact_equal(base_before, base_after, "historical base")
    assert additions_digest_before != additions_digest_after, (
        "competitive/additive parameters did not change"
    )

    print("base_digest_before:", base_digest_before)
    print("base_digest_after: ", base_digest_after)
    print("base_exact_preservation: true")
    print("additions_changed: true")
    print("FULL_BPTT_TRAINING_OBJECTIVE_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=("temporal_credit", "training_objective"),
    )
    args = ap.parse_args()

    dml = torch_directml.device()
    print("device:", dml)
    print("optimizer: DMLSafeAdam")

    if args.mode == "temporal_credit":
        temporal_credit_probe(dml)
    else:
        training_objective_probe(dml)


if __name__ == "__main__":
    main()
