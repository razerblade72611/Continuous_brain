#!/usr/bin/env python3
import argparse
import copy
import hashlib
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


def tensor_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def assert_exact_equal(before, after, label):
    assert before.keys() == after.keys(), f"{label}: state keys changed"
    for k in before:
        if not torch.equal(before[k], after[k]):
            diff = (before[k] - after[k]).abs().max().item()
            raise AssertionError(f"{label}: {k} changed; max_abs_diff={diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer-step", action="store_true")
    args = ap.parse_args()

    dml = torch_directml.device()

    base, cfg, _, sha = _load_base_checkpoint(CHECKPOINT)
    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).to(dml)
    model.train(True)

    # DirectML workaround:
    # keep the historical base frozen for optimization, but allow autograd
    # to compute gradients for the workspace parameters so DirectML takes
    # its stable backward path.
    for p in model.base.parameters():
        p.requires_grad_(False)

    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    # The only parameters the optimizer is allowed to update.
    trainable_additions = list(model.competition.parameters())
    trainable_additions += list(model.recurrent_residual.parameters())
    trainable_additions += list(model.motor_residual.parameters())
    trainable_additions += list(model.world_residual.parameters())

    optimizer = torch.optim.Adam(trainable_additions, lr=1e-3, foreach=False)

    print("optimizer_foreach:", optimizer.defaults.get("foreach"))

    base_before = clone_state(model.base)
    base_digest_before = tensor_digest(base_before)

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

    print("checkpoint_sha256:", sha)
    print("loss:", float(loss.detach().cpu()))
    print("workspace_requires_grad:",
          all(p.requires_grad for p in model.base.nucleus.workspace.parameters()))
    print("motor_requires_grad:",
          any(p.requires_grad for p in model.base.motor.parameters()))
    print("optimizer_parameter_count:",
          sum(p.numel() for g in optimizer.param_groups for p in g["params"]))

    optimizer.zero_grad(set_to_none=True)

    print("starting backward...")
    loss.backward()

    ws_grads = [
        p.grad for p in model.base.nucleus.workspace.parameters()
        if p.grad is not None
    ]
    print(
        "workspace_grad_max:",
        max(float(g.abs().max().cpu()) for g in ws_grads) if ws_grads else None,
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

    if args.optimizer_step:
        optimizer.step()

    base_after = clone_state(model.base)
    base_digest_after = tensor_digest(base_after)

    assert_exact_equal(base_before, base_after, "historical base")

    print("base_digest_before:", base_digest_before)
    print("base_digest_after: ", base_digest_after)
    print("base_exact_preservation: true")
    print("optimizer_step:", bool(args.optimizer_step))
    print("DML_WORKSPACE_AUTOGRAD_ONLY_OK")


if __name__ == "__main__":
    main()
