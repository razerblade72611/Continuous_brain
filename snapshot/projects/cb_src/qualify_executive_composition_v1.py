#!/usr/bin/env python3
"""
DirectML qualification for the first brain-integrated late-bound executive.

This is no longer a working-memory architecture search.  The DEV-admitted
memory system is frozen and protected.  Only the new executive compositor is
trainable.

Each optimizer update contains all four counterfactual combinations for the
same structural condition:

    LEFT cue  × NORMAL
    RIGHT cue × NORMAL
    LEFT cue  × INVERT
    RIGHT cue × INVERT

Therefore cue identity alone and instruction identity alone are both
insufficient shortcuts.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.executive_working_memory_v1 import (
    ExecutiveCompositionInfant,
)
from continuous_brain.late_instruction_detour_world_v1 import (
    INSTRUCTION_NONE,
    LateInstructionDetourWorld,
)
from continuous_brain.varied_cued_detour_world import LEFT, RIGHT


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_ADMITTED_SHA256 = (
    "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
)
EXPECTED_ADMITTED_FORMAT = (
    "working_memory_recall_capacity_control_training_v1"
)

UPDATE_CONDITIONS = (0, 17, 2)


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
        step_size = self.lr / bc1
        sqrt_bc2 = math.sqrt(bc2)

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


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
                f"{label}: tensor changed {k}; max_abs_diff={diff}"
            )


def grad_max(t):
    if t is None:
        return 0.0
    if not torch.isfinite(t).all().item():
        raise RuntimeError("non-finite gradient")
    return float(t.detach().abs().max().cpu())


def exact_zero_grad(t):
    if t is None:
        return True
    return bool(torch.count_nonzero(t.detach().cpu()).item() == 0)


def load_admitted(device):
    admitted_sha = sha256_file(ADMITTED_CHECKPOINT)
    if admitted_sha != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError(
            f"unexpected admitted checkpoint SHA256: {admitted_sha}"
        )

    payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != EXPECTED_ADMITTED_FORMAT:
        raise RuntimeError("admitted checkpoint format mismatch")
    if payload.get("arm") != "capacity_ungated":
        raise RuntimeError("admitted checkpoint arm mismatch")
    if int(payload.get("update", -1)) != 1600:
        raise RuntimeError("admitted checkpoint is not update 1600")
    if payload.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("admitted checkpoint base SHA mismatch")
    if payload.get("base_digest") != EXPECTED_BASE_DIGEST:
        raise RuntimeError("admitted checkpoint base digest mismatch")

    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical checkpoint SHA mismatch")
    base_reference = clone_state(base)
    if state_digest(base_reference) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(payload["memory"], strict=True)
    admitted.recall.load_state_dict(payload["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()

    for p in admitted.parameters():
        p.requires_grad_(False)

    admitted_reference = clone_state(admitted)

    model = ExecutiveCompositionInfant.grow_from_admitted(
        admitted
    ).to(device)
    model.train(True)
    model.admitted.eval()
    model.base.eval()

    additions = model.trainable_parameters()
    admitted_ids = {id(p) for p in model.admitted.parameters()}
    executive_ids = {id(p) for p in additions}
    if not admitted_ids.isdisjoint(executive_ids):
        raise AssertionError("admitted brain leaked into executive optimizer")

    return (
        model,
        cfg,
        payload,
        base_reference,
        admitted_reference,
        additions,
    )


@torch.no_grad()
def exact_initial_noop(model, cfg, device):
    for cue in ("left", "right"):
        for instruction in ("normal", "invert"):
            env = LateInstructionDetourWorld(
                0,
                cue=cue,
                instruction=instruction,
                instruction_delay=1,
            )
            obs = env.reset()

            h1 = model.initial_state(1, device)
            h0 = model.admitted.initial_state(1, device)
            m1 = model.initial_memory(1, device)
            m0 = model.admitted.initial_memory(1, device)
            p1 = torch.zeros(1, dtype=torch.long, device=device)
            p0 = torch.zeros(1, dtype=torch.long, device=device)

            while not env.done:
                image = _image_tensor(env, cfg.image_size, device)
                context, _ = _context_tensor(obs, device, cfg)
                instruction_id = torch.tensor(
                    [env.instruction_id()],
                    dtype=torch.long,
                    device=device,
                )

                got = model.step(
                    image,
                    context,
                    m1,
                    p1,
                    h1,
                    instruction_id,
                )
                plain = model.admitted.step(
                    image,
                    context,
                    m0,
                    p0,
                    h0,
                    use_memory=True,
                    use_recall=True,
                )

                if not torch.equal(
                    got["executive_residual"],
                    torch.zeros_like(got["executive_residual"]),
                ):
                    raise AssertionError(
                        "executive is not an exact initial no-op"
                    )

                for name, a, b in (
                    ("hidden", got["hidden"], plain["hidden"]),
                    ("memory_next", got["memory_next"], plain["memory_next"]),
                    (
                        "admitted_workspace",
                        got["admitted_workspace"],
                        plain["workspace_for_world"],
                    ),
                    ("logits", got["logits"], plain["logits"]),
                ):
                    if not torch.equal(a, b):
                        raise AssertionError(
                            f"initial executive no-op failed: {name}"
                        )

                teacher = int(env.teacher_action())
                obs, _, _, _ = env.step(teacher)
                nxt = torch.tensor(
                    [teacher + 1],
                    dtype=torch.long,
                    device=device,
                )
                h1 = got["hidden"]
                h0 = plain["hidden"]
                m1 = got["memory_next"]
                m0 = plain["memory_next"]
                p1 = nxt
                p0 = nxt


def decision_metrics(logits, teacher):
    logits = logits[0]
    full_pred = int(torch.argmax(logits).item())
    d = float(
        logits[RIGHT].detach().cpu()
        - logits[LEFT].detach().cpu()
    )
    side_pred = RIGHT if d > 0.0 else LEFT
    return {
        "full_correct": bool(full_pred == teacher),
        "side_correct": bool(side_pred == teacher),
        "correct_side_margin": float(
            d if teacher == RIGHT else -d
        ),
    }


def run_member(
    model,
    cfg,
    device,
    *,
    condition_id,
    cue,
    instruction,
):
    env = LateInstructionDetourWorld(
        condition_id,
        cue=cue,
        instruction=instruction,
        instruction_delay=1,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    post_instruction_losses = []
    decision_loss = None
    decision = None
    residual_max = 0.0
    instruction_first_seen = None
    steps = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        instruction_id = torch.tensor(
            [env.instruction_id()],
            dtype=torch.long,
            device=device,
        )

        out = model.step(
            image,
            context,
            memory,
            prev,
            hidden,
            instruction_id,
        )

        if env.instruction_visible() and instruction_first_seen is None:
            instruction_first_seen = int(env.steps)

        if not env.instruction_visible():
            if torch.count_nonzero(
                out["executive_residual"].detach().cpu()
            ).item() != 0:
                raise AssertionError(
                    "executive changed behavior before instruction arrival"
                )

        teacher = int(env.teacher_action())
        target = torch.tensor(
            [teacher],
            dtype=torch.long,
            device=device,
        )

        if env.instruction_visible():
            ce = F.cross_entropy(out["logits"], target)
            post_instruction_losses.append(ce)

        if (
            decision_loss is None
            and teacher in (LEFT, RIGHT)
        ):
            if not env.instruction_visible():
                raise AssertionError(
                    "late-bound decision occurred before instruction arrival"
                )
            decision_loss = F.cross_entropy(
                out["logits"],
                target,
            )
            decision = decision_metrics(
                out["logits"],
                teacher,
            )

        residual_max = max(
            residual_max,
            float(
                out["executive_residual"]
                .detach()
                .abs()
                .max()
                .cpu()
            ),
        )

        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"].detach()
        memory = out["memory_next"].detach()
        prev = (target + 1).long()
        steps += 1

        if steps > 160:
            raise RuntimeError("episode exceeded safety bound")

    if not post_instruction_losses:
        raise AssertionError("no post-instruction steps")
    if decision_loss is None or decision is None:
        raise AssertionError("no late-bound decision")

    post_mean = torch.stack(post_instruction_losses).mean()
    total = 0.5 * post_mean + 0.5 * decision_loss

    return {
        "loss": total,
        "post_mean": post_mean,
        "decision_loss": decision_loss,
        "decision": decision,
        "residual_max": residual_max,
        "instruction_first_seen": instruction_first_seen,
        "steps": steps,
    }


def run_four_way(model, cfg, device, condition_id):
    members = {}
    losses = []
    for cue in ("left", "right"):
        for instruction in ("normal", "invert"):
            key = f"{cue}_{instruction}"
            result = run_member(
                model,
                cfg,
                device,
                condition_id=condition_id,
                cue=cue,
                instruction=instruction,
            )
            members[key] = result
            losses.append(result["loss"])

    return {
        "loss": torch.stack(losses).mean(),
        "members": members,
    }


def main():
    torch.manual_seed(24680)
    device = torch_directml.device()

    (
        model,
        cfg,
        payload,
        base_ref,
        admitted_ref,
        additions,
    ) = load_admitted(device)

    print("EXECUTIVE_COMPOSITION_QUALIFICATION_V1")
    print("device:", device)
    print("historical_base_sha256:", EXPECTED_BASE_SHA256)
    print("historical_base_digest:", EXPECTED_BASE_DIGEST)
    print("admitted_memory_checkpoint:", ADMITTED_CHECKPOINT)
    print("admitted_memory_checkpoint_sha256:", EXPECTED_ADMITTED_SHA256)
    print("admitted_memory_dev_passed: true")
    print("executive_parameter_count:", sum(p.numel() for p in additions))
    print("admitted_brain_in_optimizer: false")
    print("cue_x_instruction_members_per_update: 4")
    print("old_dev_touched: false")
    print("old_test_touched: false")

    exact_initial_noop(model, cfg, device)
    print("EXACT_INITIAL_EXECUTIVE_NOOP_OK")

    opt = DMLSafeAdam(additions, lr=2e-4)

    for update, cid in enumerate(UPDATE_CONDITIONS, 1):
        opt.zero_grad(set_to_none=True)
        result = run_four_way(model, cfg, device, cid)
        result["loss"].backward()

        out_grad = grad_max(model.executive.output.weight.grad)
        mem_grad = grad_max(
            model.executive.memory_features.weight.grad
        )
        instr_grad = grad_max(
            model.executive.instruction.weight.grad
        )

        residual_max = max(
            x["residual_max"]
            for x in result["members"].values()
        )

        print(f"update{update}_condition:", cid)
        print(f"update{update}_loss:", float(result["loss"].detach().cpu()))
        print(f"update{update}_residual_max:", residual_max)
        print(f"update{update}_output_grad_max:", out_grad)
        print(f"update{update}_memory_feature_grad_max:", mem_grad)
        print(f"update{update}_instruction_grad_max:", instr_grad)

        if out_grad <= 0.0:
            raise AssertionError("executive output did not receive gradient")

        if update == 1:
            if not exact_zero_grad(
                model.executive.memory_features.weight.grad
            ):
                raise AssertionError(
                    "memory feature transform learned before zero output moved"
                )
            if not exact_zero_grad(
                model.executive.instruction.weight.grad
            ):
                raise AssertionError(
                    "instruction embedding learned before zero output moved"
                )
        else:
            if mem_grad <= 0.0:
                raise AssertionError(
                    "memory feature transform did not become load-bearing"
                )
            if instr_grad <= 0.0:
                raise AssertionError(
                    "late instruction did not become load-bearing"
                )

        opt.step()

        assert_state_exact(
            admitted_ref,
            model.admitted,
            f"admitted brain changed after update {update}",
        )
        assert_state_exact(
            base_ref,
            model.base,
            f"historical base changed after update {update}",
        )

    # NONE must remain an architectural hard no-op even after learning.
    with torch.no_grad():
        arbitrary_memory = torch.randn(
            4,
            int(cfg.memory_dim),
            device=device,
        )
        none_id = torch.full(
            (4,),
            INSTRUCTION_NONE,
            dtype=torch.long,
            device=device,
        )
        residual, _ = model.executive(
            arbitrary_memory,
            none_id,
        )
        if torch.count_nonzero(residual.detach().cpu()).item() != 0:
            raise AssertionError(
                "NONE instruction is not an exact post-training no-op"
            )

    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise AssertionError("admitted checkpoint file changed")

    print("admitted_checkpoint_unchanged: true")
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("instruction_none_exact_noop_after_training: true")
    print("staged_executive_gradients: true")
    print("EXECUTIVE_COMPOSITION_QUALIFICATION_OK")


if __name__ == "__main__":
    main()
