#!/usr/bin/env python3
import hashlib
import math
import warnings

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
UPDATES = 25
LR = 1e-4


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


def clone_state_dict(module):
    # state_dict(), not parameters(): includes buffers such as BatchNorm running
    # statistics and counters as well as parameters.
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


def parameter_digest(params):
    h = hashlib.sha256()
    for i, p in enumerate(params):
        t = p.detach().cpu().contiguous()
        h.update(str(i).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_base_exact(reference, module, update):
    current = module.state_dict()
    assert reference.keys() == current.keys()
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(expected, actual):
            if torch.is_floating_point(expected):
                diff = float((expected - actual).abs().max().item())
            else:
                diff = None
            raise AssertionError(
                f"historical base changed at update {update}: "
                f"{name}; max_abs_diff={diff}"
            )


def grad_max(tensor):
    if tensor is None:
        return None
    assert torch.isfinite(tensor).all().item(), "non-finite gradient"
    return float(tensor.abs().max().cpu())


def build_model(device):
    base, cfg, _, checkpoint_sha = _load_base_checkpoint(CHECKPOINT)
    model = CompetitiveInfantBrain.grow_from(
        base, use_shared=True
    ).to(device)
    model.train(True)

    for p in model.base.parameters():
        p.requires_grad_(False)

    # DirectML-only compatibility requirement discovered experimentally.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = (
        list(model.competition.parameters())
        + list(model.recurrent_residual.parameters())
        + list(model.motor_residual.parameters())
        + list(model.world_residual.parameters())
    )

    optimizer_ids = {id(p) for p in additions}
    base_ids = {id(p) for p in model.base.parameters()}
    workspace_ids = {
        id(p) for p in model.base.nucleus.workspace.parameters()
    }

    assert base_ids.isdisjoint(optimizer_ids), (
        "historical base leaked into optimizer"
    )
    assert workspace_ids.isdisjoint(optimizer_ids), (
        "workspace leaked into optimizer"
    )

    return model, cfg, checkpoint_sha, additions


def clear_update_boundary(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)

    for p in model.base.nucleus.workspace.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad).item() == 0, (
            "workspace gradient leaked across update boundary"
        )


def train_one_episode(model, cfg, device, seed):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)

    # Episode-boundary invariant requested by independent review.
    assert hidden.grad_fn is None, (
        "new episode hidden unexpectedly belongs to previous graph"
    )
    assert hidden.is_leaf, (
        "new episode hidden is not a fresh leaf"
    )

    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )

    motor_losses = []
    world_losses = []
    steps = 0
    teacher_return = 0.0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(
            obs, device, cfg
        )
        memory = torch.zeros(
            1, cfg.memory_dim, device=device
        )

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
            f"recurrent graph detached inside episode at step {steps + 1}"
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

        action_oh = one_hot_action(
            target, cfg.action_dim
        )

        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, reward, _done, _info = env.step(
            teacher_action
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

        teacher_return += float(reward)
        prev_action = (target + 1).long()
        obs = obs2
        steps += 1

    assert steps >= 3

    motor_loss = torch.stack(motor_losses).mean()
    world_loss = torch.stack(world_losses).mean()
    total_loss = motor_loss + world_loss

    return (
        total_loss,
        motor_loss,
        world_loss,
        steps,
        teacher_return,
    )


@torch.no_grad()
def assert_disabled_exact_parity(model, cfg, device):
    # Test disabled wrapper path against the exact frozen base path after all
    # competitive training.  The competitive additions must be irrelevant.
    env = CuedDetourWorld(
        seed=99001,
        corridor_length=0.3,
        max_episode_steps=120,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )
    memory = torch.zeros(
        1, cfg.memory_dim, device=device
    )

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
        fused,
        context,
        memory,
        prev_action,
    )
    core_input = torch.cat(tuple(streams), dim=-1)
    plain_hidden = model.base.nucleus.core(
        core_input, hidden
    )
    plain_workspace = model._workspace_from_hidden(
        plain_hidden
    )
    plain_logits = model.base.motor(plain_workspace)

    pairs = (
        ("hidden", disabled["hidden"], plain_hidden),
        (
            "workspace_for_world",
            disabled["workspace_for_world"],
            plain_workspace,
        ),
        ("logits", disabled["logits"], plain_logits),
    )

    for name, got, expected in pairs:
        if not torch.equal(got, expected):
            diff = float(
                (got - expected).abs().max().cpu()
            )
            raise AssertionError(
                f"disabled parity failed for {name}; "
                f"max_abs_diff={diff}"
            )


def main():
    # Make fallback warnings repeat instead of being suppressed after first use.
    warnings.simplefilter("always")

    device = torch_directml.device()

    model, cfg, checkpoint_sha, additions = build_model(
        device
    )
    optimizer = DMLSafeAdam(additions, lr=LR)

    base_reference = clone_state_dict(model.base)
    base_digest_before = state_digest(base_reference)

    workspace_ids = {
        id(p) for p in model.base.nucleus.workspace.parameters()
    }
    optimizer_ids = {id(p) for p in optimizer.params}
    assert workspace_ids.isdisjoint(optimizer_ids)

    successful_updates = 0
    base_preserved_updates = 0
    selector_nonzero_grad_updates = 0

    print("device:", device)
    print("optimizer: DMLSafeAdam")
    print("warnings_filter: always")
    print("checkpoint_sha256:", checkpoint_sha)
    print("updates_requested:", UPDATES)
    print(
        "optimizer_parameter_count:",
        sum(p.numel() for p in optimizer.params),
    )
    print(
        "workspace_parameter_count:",
        sum(
            p.numel()
            for p in model.base.nucleus.workspace.parameters()
        ),
    )
    print("workspace_in_optimizer: false")
    print("base_snapshot_source: state_dict")

    for update in range(1, UPDATES + 1):
        clear_update_boundary(model, optimizer)

        additions_before = parameter_digest(additions)

        (
            loss,
            motor_loss,
            world_loss,
            steps,
            teacher_return,
        ) = train_one_episode(
            model,
            cfg,
            device,
            seed=81500 + update,
        )

        loss.backward()

        ws_grads = [
            p.grad
            for p in model.base.nucleus.workspace.parameters()
            if p.grad is not None
        ]
        assert ws_grads, (
            f"missing workspace compatibility grad at update {update}"
        )
        workspace_grad = max(
            grad_max(g) for g in ws_grads
        )
        assert workspace_grad > 0.0

        rr_grad = grad_max(
            model.recurrent_residual.linear.weight.grad
        )
        mr_grad = grad_max(
            model.motor_residual.linear.weight.grad
        )
        wr_grad = grad_max(
            model.world_residual.linear.weight.grad
        )

        assert rr_grad is not None
        assert mr_grad is not None
        assert wr_grad is not None

        qgrad = model.competition.query.weight.grad
        qgrad_max = grad_max(qgrad)
        if qgrad_max is not None and qgrad_max > 0.0:
            selector_nonzero_grad_updates += 1

        optimizer.step()

        additions_after = parameter_digest(additions)
        assert additions_before != additions_after, (
            f"competitive additions unchanged at update {update}"
        )

        assert_base_exact(
            base_reference,
            model.base,
            update,
        )
        base_preserved_updates += 1
        successful_updates += 1

        print(
            f"update={update:02d}/{UPDATES} "
            f"steps={steps} "
            f"return={teacher_return:.6f} "
            f"loss={float(loss.detach().cpu()):.6f} "
            f"motor={float(motor_loss.detach().cpu()):.6f} "
            f"world={float(world_loss.detach().cpu()):.6f} "
            f"ws_grad={workspace_grad:.6g} "
            f"rr_grad={rr_grad:.6g} "
            f"mr_grad={mr_grad:.6g} "
            f"wr_grad={wr_grad:.6g} "
            f"q_grad={0.0 if qgrad_max is None else qgrad_max:.6g} "
            "base_exact=true"
        )

    assert successful_updates == UPDATES
    assert base_preserved_updates == UPDATES
    assert selector_nonzero_grad_updates > 0, (
        "selector never received a nonzero gradient after residual learning"
    )

    assert_disabled_exact_parity(
        model, cfg, device
    )

    final_base_state = clone_state_dict(model.base)
    final_base_digest = state_digest(final_base_state)
    assert final_base_digest == base_digest_before

    print("successful_updates:", successful_updates)
    print("failed_updates: 0")
    print(
        "base_preserved_updates:",
        f"{base_preserved_updates}/{UPDATES}",
    )
    print(
        "selector_nonzero_grad_updates:",
        selector_nonzero_grad_updates,
    )
    print("base_digest_before:", base_digest_before)
    print("base_digest_after: ", final_base_digest)
    print("baseline_disabled_exact_parity: true")
    print("DML_25_UPDATE_SMOKE_OK")


if __name__ == "__main__":
    main()
