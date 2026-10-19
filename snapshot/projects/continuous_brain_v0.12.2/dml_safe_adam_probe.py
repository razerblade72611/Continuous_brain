#!/usr/bin/env python3
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

CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"


class DMLSafeAdam:
    """
    Minimal Adam implementation for the DirectML qualification path.
    Avoids torch.optim.Adam's lerp_ update, which falls back to CPU
    on the currently tested torch-directml build.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
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
            grad = p.grad
            if grad is None:
                continue

            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }

            exp_avg = self.state[key]["exp_avg"]
            exp_avg_sq = self.state[key]["exp_avg_sq"]

            exp_avg.mul_(self.beta1)
            exp_avg.add_(grad, alpha=1.0 - self.beta1)

            grad_sq = grad * grad
            exp_avg_sq.mul_(self.beta2)
            exp_avg_sq.add_(grad_sq, alpha=1.0 - self.beta2)

            denom = exp_avg_sq.sqrt()
            denom.div_(sqrt_bc2)
            denom.add_(self.eps)

            update = exp_avg / denom
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


def assert_exact_equal(before, after, label):
    assert before.keys() == after.keys(), f"{label}: state keys changed"
    for k in before:
        if not torch.equal(before[k], after[k]):
            diff = (before[k] - after[k]).abs().max().item()
            raise AssertionError(
                f"{label}: {k} changed; max_abs_diff={diff}"
            )


def params_digest(params):
    h = hashlib.sha256()
    for i, p in enumerate(params):
        t = p.detach().cpu().contiguous()
        h.update(str(i).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    dml = torch_directml.device()

    base, cfg, _, checkpoint_sha = _load_base_checkpoint(CHECKPOINT)
    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).to(dml)
    model.train(True)

    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML compatibility: autograd yes, optimizer ownership no.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = (
        list(model.competition.parameters())
        + list(model.recurrent_residual.parameters())
        + list(model.motor_residual.parameters())
        + list(model.world_residual.parameters())
    )

    workspace_ids = {id(p) for p in model.base.nucleus.workspace.parameters()}
    optimizer_ids = {id(p) for p in additions}
    assert workspace_ids.isdisjoint(optimizer_ids), (
        "workspace parameter leaked into optimizer ownership"
    )

    optimizer = DMLSafeAdam(additions, lr=1e-3)

    base_before = clone_state(model.base)
    base_digest_before = state_digest(base_before)
    additions_digest_before = params_digest(additions)

    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)

    for p in model.base.nucleus.workspace.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0

    env = CuedDetourWorld(
        seed=81501,
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, dml)
    prev_action = torch.zeros(1, dtype=torch.long, device=dml)
    memory = torch.zeros(1, cfg.memory_dim, device=dml)

    image = _image_tensor(env, cfg.image_size, dml)
    context, _ = _context_tensor(obs, dml, cfg)

    out = model.step(
        image,
        context,
        memory,
        prev_action,
        hidden,
        use_competition=True,
        compute_aux=False,
    )

    target = torch.tensor(
        [_teacher_action(env)],
        dtype=torch.long,
        device=dml,
    )

    loss = F.cross_entropy(out["logits"], target)

    print("device:", dml)
    print("optimizer: DMLSafeAdam")
    print("checkpoint_sha256:", checkpoint_sha)
    print("optimizer_parameter_count:",
          sum(p.numel() for p in additions))
    print("workspace_parameter_count:",
          sum(p.numel() for p in model.base.nucleus.workspace.parameters()))
    print("workspace_in_optimizer: false")
    print("loss:", float(loss.detach().cpu()))
    print("starting backward...")

    loss.backward()

    ws_grads = [
        p.grad for p in model.base.nucleus.workspace.parameters()
        if p.grad is not None
    ]
    print(
        "workspace_grad_max:",
        max(float(g.abs().max().cpu()) for g in ws_grads)
        if ws_grads else None,
    )

    rr = model.recurrent_residual.linear.weight.grad
    mr = model.motor_residual.linear.weight.grad
    print(
        "recurrent_residual_grad_max:",
        None if rr is None else float(rr.abs().max().cpu()),
    )
    print(
        "motor_residual_grad_max:",
        None if mr is None else float(mr.abs().max().cpu()),
    )

    optimizer.step()

    base_after = clone_state(model.base)
    base_digest_after = state_digest(base_after)
    additions_digest_after = params_digest(additions)

    assert_exact_equal(base_before, base_after, "historical base")
    assert additions_digest_before != additions_digest_after, (
        "optimizer step did not change any competitive/additive parameter"
    )

    print("base_digest_before:", base_digest_before)
    print("base_digest_after: ", base_digest_after)
    print("base_exact_preservation: true")
    print("additions_changed: true")
    print("DML_SAFE_ADAM_ONE_STEP_OK")


if __name__ == "__main__":
    main()
