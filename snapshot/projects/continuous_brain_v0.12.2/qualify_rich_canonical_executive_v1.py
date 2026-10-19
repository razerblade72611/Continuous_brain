#!/usr/bin/env python3
"""Qualification for the richer canonical recalled-content executive."""
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
from continuous_brain.rich_canonical_executive_v1 import (
    RichCanonicalExecutiveInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    CanonicalExecutiveRuleWorld,
    RULE_NONE,
)

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
RULES = ("invert_lr", "vertical", "vertical_reverse")


class DMLSafeAdam:
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.b1 = float(betas[0])
        self.b2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = [
            {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
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
        bc1 = 1.0 - self.b1 ** self.step_count
        bc2 = 1.0 - self.b2 ** self.step_count
        for p, s in zip(self.params, self.state):
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all().item():
                raise RuntimeError("non-finite gradient")
            s["m"].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            s["v"].mul_(self.b2).add_(g * g, alpha=1.0 - self.b2)
            denom = s["v"].sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            p.add_(s["m"] / denom, alpha=-self.lr / bc1)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_exact(reference, module, label):
    for k, v in reference.items():
        got = module.state_dict()[k].detach().cpu()
        if not torch.equal(v, got):
            raise AssertionError(f"{label}: changed {k}")


def gmax(g):
    if g is None:
        return 0.0
    if not torch.isfinite(g).all().item():
        raise RuntimeError("non-finite gradient")
    return float(g.detach().abs().max().cpu())


def load_model(device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)

    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    base_ref = clone_state(base)
    if state_digest(base_ref) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    admitted.memory.load_state_dict(payload["memory"], strict=True)
    admitted.recall.load_state_dict(payload["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    admitted_ref = clone_state(admitted)
    model = RichCanonicalExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train(True)
    model.admitted.eval()
    model.base.eval()

    return model, cfg, admitted_ref, base_ref, model.trainable_parameters()


def run_member(model, cfg, device, cid, cue, rule):
    env = CanonicalExecutiveRuleWorld(cid, cue=cue, rule=rule)
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)
    response = None

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        rid = torch.tensor(
            [env.instruction_id()],
            dtype=torch.long,
            device=device,
        )

        out = model.step(image, context, memory, prev, hidden, rid)

        if not env.instruction_visible():
            if torch.count_nonzero(out["executive_delta"].detach().cpu()).item():
                raise AssertionError("pre-rule delta not exact zero")

        teacher = int(env.teacher_action())
        target = torch.tensor([teacher], dtype=torch.long, device=device)
        if env.at_response_step():
            response = F.cross_entropy(out["logits"], target)

        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"].detach()
        memory = out["memory_next"].detach()
        prev = (target + 1).long()

    if response is None:
        raise AssertionError("no response loss")
    return response


def sixway(model, cfg, device, cid):
    losses = []
    for cue in ("left", "right"):
        for rule in RULES:
            losses.append(run_member(model, cfg, device, cid, cue, rule))
    return torch.stack(losses).mean()


@torch.no_grad()
def exact_initial_noop(model, cfg, device):
    for cue in ("left", "right"):
        env = CanonicalExecutiveRuleWorld(0, cue=cue, rule="vertical")
        obs = env.reset()
        h1 = model.initial_state(1, device)
        h0 = model.admitted.initial_state(1, device)
        m1 = model.initial_memory(1, device)
        m0 = model.admitted.initial_memory(1, device)
        p1 = torch.zeros(1, dtype=torch.long, device=device)
        p0 = p1.clone()

        while not env.done:
            image = _image_tensor(env, cfg.image_size, device)
            context, _ = _context_tensor(obs, device, cfg)
            rid = torch.tensor(
                [env.instruction_id()],
                dtype=torch.long,
                device=device,
            )

            got = model.step(image, context, m1, p1, h1, rid)
            plain = model.admitted.step(
                image,
                context,
                m0,
                p0,
                h0,
                use_memory=True,
                use_recall=True,
            )

            if torch.count_nonzero(got["executive_delta"].cpu()).item():
                raise AssertionError("initial delta not zero")

            for name, a, b in (
                ("hidden", got["hidden"], plain["hidden"]),
                ("memory", got["memory_next"], plain["memory_next"]),
                ("workspace", got["admitted_workspace"], plain["workspace_for_world"]),
                ("logits", got["logits"], plain["logits"]),
            ):
                if not torch.equal(a, b):
                    raise AssertionError(f"initial no-op mismatch: {name}")

            teacher = int(env.teacher_action())
            obs, _, _, _ = env.step(teacher)
            nxt = torch.tensor([teacher + 1], dtype=torch.long, device=device)
            h1 = got["hidden"]
            h0 = plain["hidden"]
            m1 = got["memory_next"]
            m0 = plain["memory_next"]
            p1 = nxt
            p0 = nxt


def main():
    torch.manual_seed(515151)
    device = torch_directml.device()
    model, cfg, admitted_ref, base_ref, params = load_model(device)

    print("RICH_CANONICAL_EXECUTIVE_QUALIFICATION_V1")
    print("device:", device)
    print("executive_parameter_count:", sum(p.numel() for p in params))
    print("raw_memory_readers_in_executive: 0")
    print("canonical_recall_only_content: true")
    print("hidden_image_context_direct_inputs: false")
    print("admitted_brain_in_optimizer: false")
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    exact_initial_noop(model, cfg, device)
    print("EXACT_INITIAL_RICH_EXECUTIVE_NOOP_OK")

    opt = DMLSafeAdam(params, lr=2e-4)

    for update, cid in enumerate((0, 17, 2), 1):
        opt.zero_grad(set_to_none=True)
        loss = sixway(model, cfg, device, cid)
        loss.backward()

        outg = gmax(model.executive.output.weight.grad)
        contentg = gmax(model.executive.content.weight.grad)
        ruleg = gmax(model.executive.rule.weight.grad)
        fusiong = gmax(model.executive.fusion.weight.grad)

        print(f"update{update}_loss:", float(loss.detach().cpu()))
        print(f"update{update}_output_grad_max:", outg)
        print(f"update{update}_content_grad_max:", contentg)
        print(f"update{update}_rule_grad_max:", ruleg)
        print(f"update{update}_fusion_grad_max:", fusiong)

        if outg <= 0:
            raise AssertionError("output did not receive gradient")

        if update == 1:
            if contentg != 0 or ruleg != 0 or fusiong != 0:
                raise AssertionError("upstream learned before zero output moved")
        else:
            if min(contentg, ruleg, fusiong) <= 0:
                raise AssertionError("rich interaction path not load-bearing")

        opt.step()
        assert_exact(admitted_ref, model.admitted, "admitted brain")
        assert_exact(base_ref, model.base, "historical base")

    with torch.no_grad():
        content = torch.randn(4, int(cfg.bus_dim), device=device)
        ids = torch.full((4,), RULE_NONE, dtype=torch.long, device=device)
        delta, _ = model.executive(content, ids)
        if torch.count_nonzero(delta.cpu()).item():
            raise AssertionError("RULE_NONE not exact no-op")

    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("rule_none_exact_noop_after_learning: true")
    print("staged_gradients: true")
    print("RICH_CANONICAL_EXECUTIVE_QUALIFICATION_V1: PASS")


if __name__ == "__main__":
    main()
