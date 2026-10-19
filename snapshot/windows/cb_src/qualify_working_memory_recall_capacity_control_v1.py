#!/usr/bin/env python3
"""
DirectML qualification for the exact-parameter-matched ungated recall control.

Staged prediction:
Update 1:
- writer candidate grad > 0
- control output W2 grad == 0 (memory is zero)
- control W1 grad == 0
- static scale grad == 0

Update 2:
- memory nonzero
- W2 grad > 0
- W1 grad == 0 because W2 was zero during this forward
- static scale grad == 0 because decoded output was zero during this forward

Update 3:
- recall residual nonzero
- W2 grad > 0
- W1 grad > 0
- static scale grad > 0

Historical base remains bit-exact.
"""
from __future__ import annotations

import hashlib
import math
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
    LEFT,
    RIGHT,
    VariedCuedDetourWorld,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)


BASE_CHECKPOINT = (
    "state/child_v113_dml_smoke/infant_training.pt"
)
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd"
    "4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3"
    "e57b620c882a9216fe11a4b48bb5822e"
)

UPDATE_CONDITIONS = (0, 17, 2)
DECISION_FRACTION = 0.50
WORLD_LOSS_WEIGHT = 0.25


class DMLSafeAdam:
    def __init__(
        self,
        params,
        lr=1e-4,
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
                f"{label}: base tensor changed {k}; diff={diff}"
            )


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.abs().max().cpu())


def exact_zero_grad(t):
    if t is None:
        return True
    return bool(
        torch.count_nonzero(t.detach().cpu()).item() == 0
    )


def build(device):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("base SHA mismatch")

    ref = clone_state(base)
    digest = state_digest(ref)
    if digest != EXPECTED_BASE_DIGEST:
        raise RuntimeError("base digest mismatch")

    model = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base, cfg
    ).to(device)
    model.train(True)
    model.base.eval()

    for p in model.base.parameters():
        p.requires_grad_(False)
    for p in model.base.nucleus.workspace.parameters():
        p.requires_grad_(True)

    additions = model.trainable_parameters()

    if sum(p.numel() for p in model.recall.parameters()) != 49280:
        raise AssertionError("recall parameter count is not exactly 49,280")
    if sum(p.numel() for p in additions) != 86272:
        raise AssertionError("total addition count is not exactly 86,272")

    base_ids = {id(p) for p in model.base.parameters()}
    add_ids = {id(p) for p in additions}
    if not base_ids.isdisjoint(add_ids):
        raise AssertionError("base leaked into optimizer")

    return model, cfg, sha, ref, additions


@torch.no_grad()
def exact_initial_noop_pair(model, cfg, device):
    for cue in ("left", "right"):
        env = VariedCuedDetourWorld(
            0,
            cue=cue,
            max_episode_steps=160,
        )
        obs = env.reset()
        h1 = model.initial_state(1, device)
        h0 = model.initial_state(1, device)
        mem = model.initial_memory(1, device)
        prev = torch.zeros(1, dtype=torch.long, device=device)

        while not env.done:
            image = _image_tensor(env, cfg.image_size, device)
            context, _ = _context_tensor(obs, device, cfg)

            got = model.step(
                image,
                context,
                mem,
                prev,
                h1,
                use_memory=True,
                use_recall=True,
            )
            plain = model.historical_step(
                image,
                context,
                torch.zeros_like(mem),
                prev,
                h0,
            )

            for name, a, b in (
                ("hidden", got["hidden"], plain["hidden"]),
                (
                    "workspace",
                    got["workspace_for_world"],
                    plain["historical_workspace"],
                ),
                ("logits", got["logits"], plain["historical_logits"]),
            ):
                if not torch.equal(a, b):
                    raise AssertionError(f"initial no-op failed: {name}")

            if torch.count_nonzero(
                got["recall_residual"].detach().cpu()
            ).item() != 0:
                raise AssertionError("initial recall residual is nonzero")

            teacher = int(env.teacher_action())
            obs, _, _, _ = env.step(teacher)
            prev = torch.tensor(
                [teacher + 1],
                dtype=torch.long,
                device=device,
            )
            h1 = got["hidden"]
            h0 = plain["hidden"]
            mem = got["memory_next"]


def run_member(model, cfg, device, condition_id, cue):
    env = VariedCuedDetourWorld(
        condition_id,
        cue=cue,
        max_episode_steps=160,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    motor_losses = []
    world_losses = []
    decision_loss = None
    memory_max = 0.0
    recall_max = 0.0
    graph_entry = None
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(obs, device, cfg)

        out = model.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )

        if out["memory_next"].grad_fn is None:
            raise AssertionError("writer detached")

        if steps == 0:
            if out["hidden"].grad_fn is not None:
                raise AssertionError("unexpected hidden graph at step1")
        else:
            if out["hidden"].grad_fn is None:
                raise AssertionError("hidden graph detached")
            if graph_entry is None:
                graph_entry = steps + 1

        teacher = int(env.teacher_action())
        target = torch.tensor([teacher], dtype=torch.long, device=device)
        ce = F.cross_entropy(out["logits"], target)
        motor_losses.append(ce)

        if decision_loss is None and teacher in (LEFT, RIGHT):
            decision_loss = ce

        action_oh = one_hot_action(target, cfg.action_dim)
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, _, _, _ = env.step(teacher)
        _, next_dyn = _context_tensor(obs2, device, cfg)
        world_losses.append(
            F.smooth_l1_loss(world_out["next_context"], next_dyn)
        )

        memory_max = max(
            memory_max,
            float(out["memory_next"].detach().abs().max().cpu()),
        )
        recall_max = max(
            recall_max,
            float(out["recall_residual"].detach().abs().max().cpu()),
        )

        hidden = out["hidden"]
        memory = out["memory_next"]
        prev = (target + 1).long()
        obs = obs2
        steps += 1

    if graph_entry != 2:
        raise AssertionError(f"expected graph entry 2, got {graph_entry}")
    if decision_loss is None:
        raise AssertionError("no delayed decision")

    mean_motor = torch.stack(motor_losses).mean()
    motor = (
        (1.0 - DECISION_FRACTION) * mean_motor
        + DECISION_FRACTION * decision_loss
    )
    world = torch.stack(world_losses).mean()
    total = motor + WORLD_LOSS_WEIGHT * world

    return total, memory_max, recall_max


def run_pair(model, cfg, device, condition_id):
    l = run_member(model, cfg, device, condition_id, "left")
    r = run_member(model, cfg, device, condition_id, "right")
    return {
        "loss": 0.5 * (l[0] + r[0]),
        "memory_max": max(l[1], r[1]),
        "recall_max": max(l[2], r[2]),
    }


def main():
    warnings.simplefilter("always")
    torch.manual_seed(12345)
    device = torch_directml.device()

    model, cfg, sha, base_ref, additions = build(device)
    opt = DMLSafeAdam(additions, lr=1e-4)

    print("WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_V1")
    print("device:", device)
    print("base_checkpoint_sha256:", sha)
    print("base_digest_before:", state_digest(base_ref))
    print("writer_parameters:", sum(p.numel() for p in model.memory.parameters()))
    print("recall_parameters:", sum(p.numel() for p in model.recall.parameters()))
    print("total_addition_parameters:", sum(p.numel() for p in additions))
    print("contextual_hidden_input: false")
    print("recall_bias_terms: false")

    exact_initial_noop_pair(model, cfg, device)
    print("EXACT_INITIAL_NOOP_PAIR_OK")

    for update, cid in enumerate(UPDATE_CONDITIONS, 1):
        opt.zero_grad(set_to_none=True)
        model.base.nucleus.workspace.zero_grad(set_to_none=True)

        pair = run_pair(model, cfg, device, cid)
        print(f"update{update}_condition:", cid)
        print(f"update{update}_loss:", float(pair["loss"].detach().cpu()))
        print(f"update{update}_memory_max:", pair["memory_max"])
        print(f"update{update}_recall_max:", pair["recall_max"])

        pair["loss"].backward()

        cand = grad_max(model.memory.candidate.weight.grad)
        wgate = grad_max(model.memory.write_gate.weight.grad)
        w1 = grad_max(model.recall.memory_features.weight.grad)
        w2 = grad_max(model.recall.output.weight.grad)
        scale = grad_max(model.recall.output_scale.grad)

        print(f"update{update}_candidate_grad_max:", cand)
        print(
            f"update{update}_write_gate_grad_max:",
            0.0 if wgate is None else wgate,
        )
        print(
            f"update{update}_feature_grad_max:",
            0.0 if w1 is None else w1,
        )
        print(
            f"update{update}_output_grad_max:",
            0.0 if w2 is None else w2,
        )
        print(
            f"update{update}_scale_grad_max:",
            0.0 if scale is None else scale,
        )

        if cand is None or cand <= 0.0:
            raise AssertionError("writer candidate lost gradient")

        if update == 1:
            if not exact_zero_grad(model.recall.output.weight.grad):
                raise AssertionError("output learned before memory moved")
            if not exact_zero_grad(model.recall.memory_features.weight.grad):
                raise AssertionError("feature map learned prematurely")
            if not exact_zero_grad(model.recall.output_scale.grad):
                raise AssertionError("static scale learned prematurely")

        if update == 2:
            if pair["memory_max"] <= 0.0:
                raise AssertionError("memory still zero on update2")
            if w2 is None or w2 <= 0.0:
                raise AssertionError("output did not become load-bearing")
            if not exact_zero_grad(model.recall.memory_features.weight.grad):
                raise AssertionError("feature map learned before output moved")
            if not exact_zero_grad(model.recall.output_scale.grad):
                raise AssertionError("scale learned before output moved")

        if update == 3:
            if pair["recall_max"] <= 0.0:
                raise AssertionError("recall residual still zero")
            if w2 is None or w2 <= 0.0:
                raise AssertionError("output lost gradient")
            if w1 is None or w1 <= 0.0:
                raise AssertionError("feature map did not become load-bearing")
            if scale is None or scale <= 0.0:
                raise AssertionError("scale did not become load-bearing")

        opt.step()
        assert_state_exact(
            base_ref,
            model.base,
            f"base changed after update {update}",
        )

    digest_after = state_digest(clone_state(model.base))
    if digest_after != EXPECTED_BASE_DIGEST:
        raise AssertionError("base digest changed")

    print("base_digest_after:", digest_after)
    print("historical_base_bit_exact: true")
    print("staged_capacity_control_gradients: true")
    print("learned_recall_nonzero: true")
    print("WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_OK")


if __name__ == "__main__":
    main()
