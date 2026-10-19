#!/usr/bin/env python3
"""
DirectML qualification gate for additive writable working memory.

This is NOT a long training run and does not admit working memory into canonical
lineage.  It qualifies the mechanism and DirectML path before training.

Every optimizer update is counterfactual-balanced:
    same condition_id, LEFT member + RIGHT member, mean loss, one optimizer step.

Required gates:
  1. exact enabled-path no-op at initialization on both pair members;
  2. historical InfantBrain memory input causally affects hidden/logits;
  3. memory state shape remains exactly [1, memory_dim];
  4. first paired task loss reaches candidate projection through full BPTT;
  5. after candidate moves, paired task loss reaches write gate;
  6. learned memory becomes nonzero;
  7. historical base remains bit-exact across optimizer steps;
  8. no reported DirectML CPU fallback (checked from the captured log by shell).

The future model-level admission criterion is stronger and is intentionally NOT
claimed here: on held-out varied conditions, swapping learned LEFT/RIGHT memory
contents must predictably alter the later cue-dependent decision.
"""
from __future__ import annotations

import hashlib
import warnings

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
    VariedCuedDetourWorld,
)
from continuous_brain.working_memory_infant import WorkingMemoryInfant
from train_competitive_dml_v2 import DMLSafeAdam


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)

# Short, structurally different train-family conditions selected before running
# the qualification.  They are not dev/test conditions.
NOOP_CONDITION = 0
UPDATE1_CONDITION = 0
UPDATE2_CONDITION = 17
CAUSAL_PROBE_CONDITION = 2


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def digest_state(state):
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
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


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.abs().max().cpu())


def params_digest(params):
    h = hashlib.sha256()
    for i, p in enumerate(params):
        t = p.detach().cpu().contiguous()
        h.update(str(i).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build(device):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"unexpected base checkpoint SHA256: {sha}"
        )

    model = WorkingMemoryInfant.grow_from(base, cfg).to(device)
    model.train(True)
    model.base.eval()

    # Historical brain is optimization-frozen.
    for p in model.base.parameters():
        p.requires_grad_(False)

    # Qualified DirectML compatibility path only.  These parameters may receive
    # autograd gradients because DirectML requires that stable route, but they
    # are NEVER optimizer-owned and must remain bit-exact.
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = model.trainable_parameters()

    base_ids = {id(p) for p in model.base.parameters()}
    optimizer_ids = {id(p) for p in additions}
    workspace_ids = {
        id(p) for p in model.base.nucleus.workspace.parameters()
    }

    if not base_ids.isdisjoint(optimizer_ids):
        raise AssertionError("historical base leaked into optimizer")
    if not workspace_ids.isdisjoint(optimizer_ids):
        raise AssertionError("workspace leaked into optimizer")

    return model, cfg, sha, additions


def clear_boundary(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.base.nucleus.workspace.zero_grad(set_to_none=True)

    # With set_to_none=True the strongest and cheapest boundary assertion is
    # simply that no workspace gradient tensor remains.  Avoid count_nonzero()
    # on DirectML: that diagnostic operator itself falls back to CPU.
    for p in model.base.nucleus.workspace.parameters():
        if p.grad is not None:
            raise AssertionError(
                "workspace gradient tensor remained at update boundary"
            )


@torch.no_grad()
def exact_initial_noop_member(model, cfg, device, condition_id, cue):
    env = VariedCuedDetourWorld(
        condition_id,
        cue=cue,
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
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        got = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden_mem,
            use_memory=True,
        )
        plain = model.base_step(
            image,
            context,
            torch.zeros_like(memory),
            prev_action,
            hidden_plain,
        )

        for name in ("hidden", "workspace_for_world", "logits"):
            if not torch.equal(got[name], plain[name]):
                diff = float(
                    (got[name] - plain[name])
                    .abs()
                    .max()
                    .cpu()
                )
                raise AssertionError(
                    f"initial no-op failed condition={condition_id} "
                    f"cue={cue} field={name}; diff={diff}"
                )

        if tuple(got["memory_next"].shape) != (
            1, int(cfg.memory_dim)
        ):
            raise AssertionError(
                f"bad memory shape: "
                f"{tuple(got['memory_next'].shape)}"
            )

        # Exact-zero qualification is intentionally checked after an explicit
        # device->CPU copy.  torch.count_nonzero() on a DirectML tensor is not
        # implemented and would create a diagnostic CPU-fallback warning.
        memory_cpu = got["memory_next"].detach().cpu()
        if torch.count_nonzero(memory_cpu).item() != 0:
            raise AssertionError(
                f"initial memory became nonzero condition={condition_id} "
                f"cue={cue} step={steps}"
            )

        teacher = int(env.teacher_action())
        obs, _reward, _done, _info = env.step(teacher)

        prev_action = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden_mem = got["hidden"]
        hidden_plain = plain["hidden"]
        memory = got["memory_next"]
        steps += 1

        if steps > 160:
            raise RuntimeError("no-op episode exceeded safety bound")

    if steps < 3:
        raise AssertionError("no-op episode too short")

    return steps


@torch.no_grad()
def exact_initial_noop_pair(model, cfg, device, condition_id):
    left_steps = exact_initial_noop_member(
        model, cfg, device, condition_id, "left"
    )
    right_steps = exact_initial_noop_member(
        model, cfg, device, condition_id, "right"
    )
    if left_steps != right_steps:
        raise AssertionError(
            "counterfactual pair has unequal teacher path lengths"
        )
    return left_steps


@torch.no_grad()
def memory_input_is_causal(model, cfg, device, condition_id):
    """
    Advance to a cue-hidden state using the frozen base with zero memory, then
    intervene on ONLY the memory input at that same sensory/recurrent state.
    """
    env = VariedCuedDetourWorld(
        condition_id,
        cue="left",
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )
    zero = torch.zeros(
        1, cfg.memory_dim, device=device
    )

    # Move until the cue has been absent for at least one complete transition.
    while env.steps <= env.cue_steps:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        out = model.base_step(
            image,
            context,
            zero,
            prev_action,
            hidden,
        )
        teacher = int(env.teacher_action())
        obs, _reward, _done, _info = env.step(teacher)
        prev_action = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden = out["hidden"]

    image = _image_tensor(env, cfg.image_size, device)
    context, _ = _context_tensor(obs, device, cfg)

    probe = torch.linspace(
        -0.75,
        0.75,
        int(cfg.memory_dim),
        device=device,
    ).reshape(1, -1)

    z = model.base_step(
        image, context, zero, prev_action, hidden
    )
    p = model.base_step(
        image, context, probe, prev_action, hidden
    )
    n = model.base_step(
        image, context, -probe, prev_action, hidden
    )

    hidden_delta = float(
        (p["hidden"] - z["hidden"]).abs().max().cpu()
    )
    logits_delta = float(
        (p["logits"] - z["logits"]).abs().max().cpu()
    )
    sign_logits_delta = float(
        (p["logits"] - n["logits"]).abs().max().cpu()
    )

    if hidden_delta <= 0.0:
        raise AssertionError(
            "memory intervention did not affect recurrent hidden"
        )
    if logits_delta <= 0.0:
        raise AssertionError(
            "memory intervention did not affect policy logits"
        )
    if sign_logits_delta <= 0.0:
        raise AssertionError(
            "opposite memory interventions produced identical logits"
        )

    return {
        "condition_id": int(condition_id),
        "cue_hidden_steps": int(env.steps),
        "hidden_delta": hidden_delta,
        "logits_delta": logits_delta,
        "opposite_probe_logits_delta": sign_logits_delta,
    }


def run_member_loss(
    model,
    cfg,
    device,
    *,
    condition_id,
    cue,
):
    env = VariedCuedDetourWorld(
        condition_id,
        cue=cue,
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    if hidden.grad_fn is not None or not hidden.is_leaf:
        raise AssertionError(
            "initial hidden must be a fresh leaf"
        )

    memory = model.initial_memory(1, device)
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )

    motor_losses = []
    world_losses = []
    memory_norms = []
    memory_shapes = set()
    first_cue_hidden_memory = None
    recurrent_graph_entry_step = None
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
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

        # Delayed-memory graph semantics:
        #   step 1 base recurrent output is expected to have no grad_fn because
        #   the historical core is frozen and memory_in is the initial zero
        #   leaf.  The writer is computed *after* that recurrent step.
        #   memory_next from step 1 is differentiable, and on step 2 it becomes
        #   memory_in, so the recurrent graph must begin there.
        if out["memory_next"].grad_fn is None:
            raise AssertionError(
                f"memory writer detached at step {steps + 1}"
            )

        if steps == 0:
            if hidden.grad_fn is not None:
                raise AssertionError(
                    "unexpected recurrent gradient path on first step; "
                    "qualification expects delayed memory influence"
                )
        else:
            if hidden.grad_fn is None:
                raise AssertionError(
                    f"recurrent graph failed to enter through memory "
                    f"at step {steps + 1}"
                )
            if recurrent_graph_entry_step is None:
                recurrent_graph_entry_step = steps + 1

        memory_shape = tuple(out["memory_next"].shape)
        memory_shapes.add(memory_shape)
        if memory_shape != (1, int(cfg.memory_dim)):
            raise AssertionError(
                f"bad memory shape: {memory_shape}"
            )

        if (
            first_cue_hidden_memory is None
            and env.steps >= env.cue_steps
        ):
            first_cue_hidden_memory = out[
                "memory_next"
            ].detach().clone()

        teacher = int(env.teacher_action())
        target = torch.tensor(
            [teacher],
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

        obs2, _reward, _done, _info = env.step(teacher)
        _, next_context_dyn = _context_tensor(
            obs2, device, cfg
        )

        world_losses.append(
            F.smooth_l1_loss(
                world_out["next_context"],
                next_context_dyn,
            )
        )

        memory_norms.append(
            float(
                out["memory_next"]
                .detach()
                .abs()
                .max()
                .cpu()
            )
        )

        prev_action = (target + 1).long()
        memory = out["memory_next"]
        obs = obs2
        steps += 1

        if steps > 160:
            raise RuntimeError(
                "training episode exceeded safety bound"
            )

    if first_cue_hidden_memory is None:
        raise AssertionError(
            "episode never reached a cue-hidden state"
        )

    motor_loss = torch.stack(motor_losses).mean()
    world_loss = torch.stack(world_losses).mean()
    total_loss = motor_loss + world_loss

    if recurrent_graph_entry_step != 2:
        raise AssertionError(
            f"expected delayed recurrent graph entry at step 2, "
            f"got {recurrent_graph_entry_step}"
        )

    return {
        "total_loss": total_loss,
        "motor_loss": motor_loss,
        "world_loss": world_loss,
        "steps": steps,
        "memory_norms": memory_norms,
        "memory_shapes": sorted(memory_shapes),
        "first_hidden_memory": first_cue_hidden_memory,
        "recurrent_graph_entry_step": recurrent_graph_entry_step,
    }


def run_balanced_pair_loss(
    model,
    cfg,
    device,
    condition_id,
):
    left = run_member_loss(
        model,
        cfg,
        device,
        condition_id=condition_id,
        cue="left",
    )
    right = run_member_loss(
        model,
        cfg,
        device,
        condition_id=condition_id,
        cue="right",
    )

    if left["steps"] != right["steps"]:
        raise AssertionError(
            f"pair path-length mismatch at condition {condition_id}: "
            f"{left['steps']} vs {right['steps']}"
        )

    total = 0.5 * (
        left["total_loss"] + right["total_loss"]
    )
    motor = 0.5 * (
        left["motor_loss"] + right["motor_loss"]
    )
    world = 0.5 * (
        left["world_loss"] + right["world_loss"]
    )

    return {
        "total_loss": total,
        "motor_loss": motor,
        "world_loss": world,
        "left": left,
        "right": right,
    }


def main():
    warnings.simplefilter("always")

    device = torch_directml.device()
    model, cfg, sha, additions = build(device)

    base_reference = clone_state(model.base)
    base_digest_before = digest_state(base_reference)
    additions_before = params_digest(additions)

    print("WORKING_MEMORY_DML_QUALIFICATION_V3")
    print("device:", device)
    print("checkpoint_sha256:", sha)
    print("base_digest_before:", base_digest_before)
    print("memory_dim:", cfg.memory_dim)
    print("nucleus_dim:", cfg.nucleus_dim)
    print(
        "memory_trainable_parameters:",
        sum(p.numel() for p in additions),
    )
    print(
        "workspace_parameter_count:",
        sum(
            p.numel()
            for p in model.base.nucleus.workspace.parameters()
        ),
    )
    print("workspace_in_optimizer: false")
    print("counterfactual_pair_per_optimizer_step: true")

    noop_steps = exact_initial_noop_pair(
        model, cfg, device, NOOP_CONDITION
    )
    print("exact_initial_noop_condition:", NOOP_CONDITION)
    print("exact_initial_noop_steps_each_member:", noop_steps)
    print("INITIAL_MEMORY_NOOP_PAIR_OK")

    causal = memory_input_is_causal(
        model, cfg, device, CAUSAL_PROBE_CONDITION
    )
    print(
        "memory_probe_condition:",
        causal["condition_id"],
    )
    print(
        "memory_probe_cue_hidden_steps:",
        causal["cue_hidden_steps"],
    )
    print(
        "memory_probe_hidden_delta:",
        causal["hidden_delta"],
    )
    print(
        "memory_probe_logits_delta:",
        causal["logits_delta"],
    )
    print(
        "memory_probe_opposite_logits_delta:",
        causal["opposite_probe_logits_delta"],
    )
    print("BASE_MEMORY_PATH_CAUSAL_OK")

    optimizer = DMLSafeAdam(additions, lr=1e-4)

    # UPDATE 1: exactly balanced counterfactual pair.
    clear_boundary(model, optimizer)
    pair1 = run_balanced_pair_loss(
        model, cfg, device, UPDATE1_CONDITION
    )
    loss1 = pair1["total_loss"]

    print("update1_condition:", UPDATE1_CONDITION)
    print(
        "update1_left_steps:",
        pair1["left"]["steps"],
    )
    print(
        "update1_right_steps:",
        pair1["right"]["steps"],
    )
    print(
        "update1_total_loss:",
        float(loss1.detach().cpu()),
    )
    print(
        "update1_motor_loss:",
        float(pair1["motor_loss"].detach().cpu()),
    )
    print(
        "update1_world_loss:",
        float(pair1["world_loss"].detach().cpu()),
    )
    print(
        "update1_recurrent_graph_entry_left:",
        pair1["left"]["recurrent_graph_entry_step"],
    )
    print(
        "update1_recurrent_graph_entry_right:",
        pair1["right"]["recurrent_graph_entry_step"],
    )
    print("starting update1 backward...")
    loss1.backward()

    candidate_grad1 = grad_max(
        model.memory.candidate.weight.grad
    )
    gate_grad1 = grad_max(
        model.memory.write_gate.weight.grad
    )

    print(
        "update1_candidate_grad_max:",
        candidate_grad1,
    )
    print(
        "update1_gate_grad_max:",
        0.0 if gate_grad1 is None else gate_grad1,
    )

    if candidate_grad1 is None or candidate_grad1 <= 0.0:
        raise AssertionError(
            "paired task loss did not reach memory candidate"
        )

    # At exact zero init the gate is allowed/expected to have zero gradient.
    optimizer.step()

    assert_state_exact(
        base_reference,
        model.base,
        "historical base changed after update1",
    )

    additions_after1 = params_digest(additions)
    if additions_after1 == additions_before:
        raise AssertionError(
            "memory additions did not change after update1"
        )

    # UPDATE 2: with candidate now nonzero, gate must become load-bearing.
    clear_boundary(model, optimizer)
    pair2 = run_balanced_pair_loss(
        model, cfg, device, UPDATE2_CONDITION
    )
    loss2 = pair2["total_loss"]

    max_memory2 = max(
        pair2["left"]["memory_norms"]
        + pair2["right"]["memory_norms"]
    )

    print("update2_condition:", UPDATE2_CONDITION)
    print(
        "update2_left_steps:",
        pair2["left"]["steps"],
    )
    print(
        "update2_right_steps:",
        pair2["right"]["steps"],
    )
    print(
        "update2_total_loss:",
        float(loss2.detach().cpu()),
    )
    print(
        "update2_motor_loss:",
        float(pair2["motor_loss"].detach().cpu()),
    )
    print(
        "update2_world_loss:",
        float(pair2["world_loss"].detach().cpu()),
    )
    print(
        "update2_recurrent_graph_entry_left:",
        pair2["left"]["recurrent_graph_entry_step"],
    )
    print(
        "update2_recurrent_graph_entry_right:",
        pair2["right"]["recurrent_graph_entry_step"],
    )
    print(
        "update2_max_abs_memory:",
        max_memory2,
    )

    if max_memory2 <= 0.0:
        raise AssertionError(
            "trained memory writer still produced only zero"
        )

    print("starting update2 backward...")
    loss2.backward()

    candidate_grad2 = grad_max(
        model.memory.candidate.weight.grad
    )
    gate_grad2 = grad_max(
        model.memory.write_gate.weight.grad
    )

    print(
        "update2_candidate_grad_max:",
        candidate_grad2,
    )
    print(
        "update2_gate_grad_max:",
        0.0 if gate_grad2 is None else gate_grad2,
    )

    if candidate_grad2 is None or candidate_grad2 <= 0.0:
        raise AssertionError(
            "candidate lost gradient on update2"
        )
    if gate_grad2 is None or gate_grad2 <= 0.0:
        raise AssertionError(
            "write gate did not become gradient-bearing"
        )

    optimizer.step()

    assert_state_exact(
        base_reference,
        model.base,
        "historical base changed after update2",
    )

    base_digest_after = digest_state(
        clone_state(model.base)
    )
    additions_after2 = params_digest(additions)

    if base_digest_after != base_digest_before:
        raise AssertionError(
            "historical base digest changed"
        )
    if additions_after2 == additions_after1:
        raise AssertionError(
            "memory additions did not change after update2"
        )

    print("base_digest_after:", base_digest_after)
    print("base_exact_after_two_updates: true")
    print(
        "memory_shape_fixed:",
        f"[1,{cfg.memory_dim}]",
    )
    print("learned_memory_nonzero: true")
    print("candidate_full_bptt_gradient: true")
    print("write_gate_gradient_after_candidate_moves: true")
    print("historical_base_bit_exact: true")
    print("WORKING_MEMORY_DML_QUALIFICATION_OK")


if __name__ == "__main__":
    main()
