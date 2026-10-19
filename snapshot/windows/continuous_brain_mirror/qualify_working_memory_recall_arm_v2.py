#!/usr/bin/env python3
"""
DirectML qualification for Working Memory M1.5 and M2 recall arms.

Run one arm per fresh Python process:
    --arm linear   # M1.5
    --arm gated    # M2

The qualification intentionally tests staged learning:

Update 1:
  * writer candidate receives temporal task gradient
  * recall projection gradient is exactly zero because memory is exactly zero

Update 2:
  * learned memory is nonzero
  * direct recall projection receives nonzero gradient
  * M2 gate is still allowed/expected to be zero because the projection is
    still zero during this forward pass
  * M2 neutral multiplier is exactly 1.0, matching M1.5 effective gain

Update 3:
  * direct recall residual is nonzero
  * M2 gate must receive nonzero gradient now that decoded memory is nonzero

This staged signature makes it difficult for an accidental constant shortcut
or hidden direct path to masquerade as memory recall.
"""

from __future__ import annotations

import argparse
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
from continuous_brain.working_memory_recall_arms_v2 import (
    RecallBridgeWorkingMemoryInfant,
)


BASE_CHECKPOINT = (
    "state/child_v113_dml_smoke/"
    "infant_training.pt"
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
        self.params = [
            p for p in params if p.requires_grad
        ]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = [
            {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq":
                    torch.zeros_like(p),
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

        for p, state in zip(
            self.params, self.state
        ):
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all().item():
                raise RuntimeError(
                    "non-finite optimizer gradient"
                )

            m = state["exp_avg"]
            v = state["exp_avg_sq"]

            m.mul_(self.beta1)
            m.add_(
                g,
                alpha=1.0 - self.beta1,
            )
            v.mul_(self.beta2)
            v.add_(
                g * g,
                alpha=1.0 - self.beta2,
            )

            denom = v.sqrt()
            denom.div_(sqrt_bc2)
            denom.add_(self.eps)
            p.add_(
                m / denom,
                alpha=-step_size,
            )


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = (
            state[k]
            .detach()
            .cpu()
            .contiguous()
        )
        h.update(k.encode("utf-8"))
        h.update(
            str(tuple(t.shape)).encode("utf-8")
        )
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_state_exact(
    reference,
    module,
    label,
):
    current = module.state_dict()
    if reference.keys() != current.keys():
        raise AssertionError(
            f"{label}: state keys changed"
        )
    for name, expected in reference.items():
        actual = current[name].detach().cpu()
        if not torch.equal(
            expected, actual
        ):
            diff = None
            if torch.is_floating_point(expected):
                diff = float(
                    (
                        expected - actual
                    ).abs().max().item()
                )
            raise AssertionError(
                f"{label}: {name} changed; "
                f"max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return None
    if not torch.isfinite(t).all().item():
        raise RuntimeError(
            "non-finite gradient"
        )
    return float(t.abs().max().cpu())


def exact_zero_grad(t):
    if t is None:
        return True
    return bool(
        torch.count_nonzero(
            t.detach().cpu()
        ).item() == 0
    )


def build(arm, device):
    base, cfg, _, sha = _load_base_checkpoint(
        BASE_CHECKPOINT
    )
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"unexpected base SHA: {sha}"
        )

    base_reference = clone_state(base)
    digest = state_digest(base_reference)
    if digest != EXPECTED_BASE_DIGEST:
        raise RuntimeError(
            f"unexpected base digest: {digest}"
        )

    model = (
        RecallBridgeWorkingMemoryInfant
        .grow_from(
            base,
            cfg,
            arm=arm,
        )
        .to(device)
    )
    model.train(True)
    model.base.eval()

    for p in model.base.parameters():
        p.requires_grad_(False)

    # Qualified DirectML compatibility only.
    for p in (
        model.base.nucleus.workspace
        .parameters()
    ):
        p.requires_grad_(True)

    additions = model.trainable_parameters()

    base_ids = {
        id(p)
        for p in model.base.parameters()
    }
    addition_ids = {
        id(p) for p in additions
    }
    workspace_ids = {
        id(p)
        for p in
        model.base.nucleus.workspace
        .parameters()
    }

    if not base_ids.isdisjoint(
        addition_ids
    ):
        raise AssertionError(
            "base leaked into optimizer"
        )
    if not workspace_ids.isdisjoint(
        addition_ids
    ):
        raise AssertionError(
            "workspace leaked into optimizer"
        )

    return (
        model,
        cfg,
        sha,
        base_reference,
        additions,
    )


def clear_boundary(model, optimizer):
    optimizer.zero_grad(
        set_to_none=True
    )
    model.base.nucleus.workspace.zero_grad(
        set_to_none=True
    )
    for p in (
        model.base.nucleus.workspace
        .parameters()
    ):
        if p.grad is not None:
            raise AssertionError(
                "workspace gradient tensor "
                "remained at boundary"
            )


@torch.no_grad()
def exact_initial_noop_pair(
    model,
    cfg,
    device,
    *,
    condition_id=0,
):
    for cue in ("left", "right"):
        env = VariedCuedDetourWorld(
            condition_id,
            cue=cue,
            max_episode_steps=160,
        )
        obs = env.reset()

        hidden_arm = model.initial_state(
            1, device
        )
        hidden_plain = model.initial_state(
            1, device
        )
        memory = model.initial_memory(
            1, device
        )
        prev_action = torch.zeros(
            1,
            dtype=torch.long,
            device=device,
        )

        steps = 0
        while not env.done:
            image = _image_tensor(
                env,
                cfg.image_size,
                device,
            )
            context, _ = _context_tensor(
                obs,
                device,
                cfg,
            )

            got = model.step(
                image,
                context,
                memory,
                prev_action,
                hidden_arm,
                use_memory=True,
                use_recall=True,
            )
            plain = model.historical_step(
                image,
                context,
                torch.zeros_like(memory),
                prev_action,
                hidden_plain,
            )

            comparisons = (
                (
                    "hidden",
                    got["hidden"],
                    plain["hidden"],
                ),
                (
                    "workspace",
                    got["workspace_for_world"],
                    plain[
                        "historical_workspace"
                    ],
                ),
                (
                    "logits",
                    got["logits"],
                    plain[
                        "historical_logits"
                    ],
                ),
            )
            for name, actual, expected in comparisons:
                if not torch.equal(
                    actual, expected
                ):
                    diff = float(
                        (
                            actual - expected
                        ).abs().max().cpu()
                    )
                    raise AssertionError(
                        f"initial no-op failed "
                        f"{cue} {name}: {diff}"
                    )

            if (
                torch.count_nonzero(
                    got["memory_next"]
                    .detach()
                    .cpu()
                ).item()
                != 0
            ):
                raise AssertionError(
                    "initial memory became "
                    "nonzero"
                )
            if (
                torch.count_nonzero(
                    got["recall_residual"]
                    .detach()
                    .cpu()
                ).item()
                != 0
            ):
                raise AssertionError(
                    "initial recall residual "
                    "became nonzero"
                )

            teacher = int(
                env.teacher_action()
            )
            obs, _, _, _ = env.step(
                teacher
            )
            prev_action = torch.tensor(
                [teacher + 1],
                dtype=torch.long,
                device=device,
            )
            hidden_arm = got["hidden"]
            hidden_plain = plain["hidden"]
            memory = got["memory_next"]
            steps += 1

        if steps < 3:
            raise AssertionError(
                "no-op trajectory too short"
            )


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

    hidden = model.initial_state(
        1, device
    )
    memory = model.initial_memory(
        1, device
    )
    prev_action = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )

    all_motor = []
    world_losses = []
    decision_loss = None
    memory_max = 0.0
    recall_max = 0.0
    gate_mean_sum = 0.0
    multiplier_mean_sum = 0.0
    gate_count = 0
    graph_entry_step = None
    steps = 0

    while not env.done:
        image = _image_tensor(
            env,
            cfg.image_size,
            device,
        )
        context, context_dyn = (
            _context_tensor(
                obs,
                device,
                cfg,
            )
        )

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=True,
            use_recall=True,
        )

        if (
            out["memory_next"].grad_fn
            is None
        ):
            raise AssertionError(
                f"writer detached "
                f"step {steps + 1}"
            )

        if steps == 0:
            if out["hidden"].grad_fn is not None:
                raise AssertionError(
                    "unexpected writer->hidden "
                    "path at step1"
                )
        else:
            if out["hidden"].grad_fn is None:
                raise AssertionError(
                    "recurrent memory path "
                    f"detached step {steps + 1}"
                )
            if graph_entry_step is None:
                graph_entry_step = (
                    steps + 1
                )

        teacher = int(
            env.teacher_action()
        )
        target = torch.tensor(
            [teacher],
            dtype=torch.long,
            device=device,
        )
        ce = F.cross_entropy(
            out["logits"],
            target,
        )
        all_motor.append(ce)

        if (
            decision_loss is None
            and teacher in (LEFT, RIGHT)
        ):
            if env.steps < env.cue_steps:
                raise AssertionError(
                    "decision while cue visible"
                )
            decision_loss = ce

        action_oh = one_hot_action(
            target,
            cfg.action_dim,
        )
        world_out = model.predict_world(
            out["workspace_for_world"],
            action_oh,
            base_context=context_dyn,
        )

        obs2, _, _, _ = env.step(
            teacher
        )
        _, next_context_dyn = (
            _context_tensor(
                obs2,
                device,
                cfg,
            )
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
        recall_max = max(
            recall_max,
            float(
                out["recall_residual"]
                .detach()
                .abs()
                .max()
                .cpu()
            ),
        )
        if (
            out["recall_gate"]
            is not None
        ):
            gate_mean_sum += float(
                out["recall_gate"]
                .detach()
                .mean()
                .cpu()
            )
            multiplier_mean_sum += float(
                out["recall_multiplier"]
                .detach()
                .mean()
                .cpu()
            )
            gate_count += 1

        hidden = out["hidden"]
        memory = out["memory_next"]
        prev_action = (
            target + 1
        ).long()
        obs = obs2
        steps += 1

    if graph_entry_step != 2:
        raise AssertionError(
            f"expected recurrent graph "
            f"entry 2, got "
            f"{graph_entry_step}"
        )
    if decision_loss is None:
        raise AssertionError(
            "no delayed decision"
        )

    all_mean = torch.stack(
        all_motor
    ).mean()
    motor = (
        (1.0 - DECISION_FRACTION)
        * all_mean
        + DECISION_FRACTION
        * decision_loss
    )
    world = torch.stack(
        world_losses
    ).mean()
    total = (
        motor
        + WORLD_LOSS_WEIGHT * world
    )

    return {
        "total": total,
        "memory_max": memory_max,
        "recall_max": recall_max,
        "gate_mean": (
            gate_mean_sum / gate_count
            if gate_count else None
        ),
        "multiplier_mean": (
            multiplier_mean_sum / gate_count
            if gate_count else None
        ),
        "steps": steps,
    }


def run_pair_loss(
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
            "pair path lengths differ"
        )
    total = 0.5 * (
        left["total"]
        + right["total"]
    )
    return {
        "total": total,
        "memory_max": max(
            left["memory_max"],
            right["memory_max"],
        ),
        "recall_max": max(
            left["recall_max"],
            right["recall_max"],
        ),
        "gate_mean": (
            None
            if left["gate_mean"] is None
            else 0.5 * (
                left["gate_mean"]
                + right["gate_mean"]
            )
        ),
        "multiplier_mean": (
            None
            if left["multiplier_mean"] is None
            else 0.5 * (
                left["multiplier_mean"]
                + right["multiplier_mean"]
            )
        ),
    }


def recall_projection_weight(model):
    return model.recall.projection.weight


def recall_gate_weight(model):
    if model.arm != "gated":
        return None
    return model.recall.gate.weight


def recall_gate_bias(model):
    if model.arm != "gated":
        return None
    return model.recall.gate.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=("linear", "gated"),
    )
    args = ap.parse_args()

    warnings.simplefilter("always")
    torch.manual_seed(12345)

    device = torch_directml.device()
    (
        model,
        cfg,
        sha,
        base_reference,
        additions,
    ) = build(args.arm, device)

    optimizer = DMLSafeAdam(
        additions,
        lr=1e-4,
    )

    print(
        "WORKING_MEMORY_RECALL_"
        "QUALIFICATION_V1"
    )
    print("arm:", args.arm)
    print("device:", device)
    print(
        "base_checkpoint_sha256:",
        sha,
    )
    print(
        "base_digest_before:",
        state_digest(base_reference),
    )
    print(
        "memory_dim:",
        cfg.memory_dim,
    )
    print(
        "bus_dim:",
        cfg.bus_dim,
    )
    print(
        "nucleus_dim:",
        cfg.nucleus_dim,
    )
    print(
        "total_addition_parameters:",
        sum(
            p.numel()
            for p in additions
        ),
    )
    print(
        "writer_parameters:",
        sum(
            p.numel()
            for p in
            model.memory.parameters()
        ),
    )
    print(
        "recall_parameters:",
        sum(
            p.numel()
            for p in
            model.recall.parameters()
        ),
    )

    exact_initial_noop_pair(
        model,
        cfg,
        device,
        condition_id=0,
    )
    print(
        "EXACT_INITIAL_NOOP_PAIR_OK"
    )

    for update, condition_id in enumerate(
        UPDATE_CONDITIONS,
        1,
    ):
        clear_boundary(
            model, optimizer
        )
        pair = run_pair_loss(
            model,
            cfg,
            device,
            condition_id,
        )

        print(
            f"update{update}_condition:",
            condition_id,
        )
        print(
            f"update{update}_loss:",
            float(
                pair["total"]
                .detach()
                .cpu()
            ),
        )
        print(
            f"update{update}_memory_max:",
            pair["memory_max"],
        )
        print(
            f"update{update}_recall_max:",
            pair["recall_max"],
        )
        if pair["gate_mean"] is not None:
            print(
                f"update{update}_"
                "recall_gate_mean:",
                pair["gate_mean"],
            )
            print(
                f"update{update}_"
                "recall_multiplier_mean:",
                pair["multiplier_mean"],
            )
            if update <= 3 and abs(
                pair["multiplier_mean"] - 1.0
            ) > 1e-7:
                raise AssertionError(
                    "neutral M2 multiplier drifted before gate update"
                )

        pair["total"].backward()

        cand = grad_max(
            model.memory.candidate
            .weight.grad
        )
        write_gate = grad_max(
            model.memory.write_gate
            .weight.grad
        )
        projection = grad_max(
            recall_projection_weight(
                model
            ).grad
        )
        rgw = grad_max(
            None
            if recall_gate_weight(
                model
            ) is None
            else recall_gate_weight(
                model
            ).grad
        )
        rgb = grad_max(
            None
            if recall_gate_bias(
                model
            ) is None
            else recall_gate_bias(
                model
            ).grad
        )

        print(
            f"update{update}_"
            "candidate_grad_max:",
            cand,
        )
        print(
            f"update{update}_"
            "write_gate_grad_max:",
            0.0
            if write_gate is None
            else write_gate,
        )
        print(
            f"update{update}_"
            "recall_projection_grad_max:",
            0.0
            if projection is None
            else projection,
        )
        if args.arm == "gated":
            print(
                f"update{update}_"
                "recall_gate_weight_grad_max:",
                0.0
                if rgw is None
                else rgw,
            )
            print(
                f"update{update}_"
                "recall_gate_bias_grad_max:",
                0.0
                if rgb is None
                else rgb,
            )

        # Writer candidate must always remain trainable.
        if cand is None or cand <= 0.0:
            raise AssertionError(
                f"update {update}: "
                "candidate lost gradient"
            )

        if update == 1:
            if not exact_zero_grad(
                recall_projection_weight(
                    model
                ).grad
            ):
                raise AssertionError(
                    "recall projection received "
                    "gradient before memory moved"
                )
            if args.arm == "gated":
                if not exact_zero_grad(
                    recall_gate_weight(
                        model
                    ).grad
                ):
                    raise AssertionError(
                        "recall gate received "
                        "premature weight gradient"
                    )
                if not exact_zero_grad(
                    recall_gate_bias(
                        model
                    ).grad
                ):
                    raise AssertionError(
                        "recall gate received "
                        "premature bias gradient"
                    )

        if update == 2:
            if pair["memory_max"] <= 0.0:
                raise AssertionError(
                    "memory still zero on "
                    "update2"
                )
            if (
                projection is None
                or projection <= 0.0
            ):
                raise AssertionError(
                    "recall projection did "
                    "not become load-bearing "
                    "on update2"
                )
            if args.arm == "gated":
                # Projection is still exactly zero during this
                # forward, so gate should remain behaviorally inert.
                if not exact_zero_grad(
                    recall_gate_weight(
                        model
                    ).grad
                ):
                    raise AssertionError(
                        "gate weight gradient "
                        "appeared before decoded "
                        "memory affected output"
                    )
                if not exact_zero_grad(
                    recall_gate_bias(
                        model
                    ).grad
                ):
                    raise AssertionError(
                        "gate bias gradient "
                        "appeared before decoded "
                        "memory affected output"
                    )

        if update == 3:
            if pair["recall_max"] <= 0.0:
                raise AssertionError(
                    "learned recall residual "
                    "still zero on update3"
                )
            if (
                projection is None
                or projection <= 0.0
            ):
                raise AssertionError(
                    "recall projection lost "
                    "gradient on update3"
                )
            if args.arm == "gated":
                if rgw is None or rgw <= 0.0:
                    raise AssertionError(
                        "gated M2 weight did "
                        "not become load-bearing"
                    )
                if rgb is None or rgb <= 0.0:
                    raise AssertionError(
                        "gated M2 bias did "
                        "not become load-bearing"
                    )

        optimizer.step()

        assert_state_exact(
            base_reference,
            model.base,
            f"historical base changed "
            f"after update {update}",
        )

    digest_after = state_digest(
        clone_state(model.base)
    )
    if digest_after != EXPECTED_BASE_DIGEST:
        raise AssertionError(
            "base digest changed"
        )

    print(
        "base_digest_after:",
        digest_after,
    )
    print(
        "historical_base_bit_exact:",
        "true",
    )
    print(
        "staged_recall_gradients:",
        "true",
    )
    print(
        "learned_recall_nonzero:",
        "true",
    )
    print(
        "WORKING_MEMORY_RECALL_"
        "QUALIFICATION_OK"
    )


if __name__ == "__main__":
    main()
