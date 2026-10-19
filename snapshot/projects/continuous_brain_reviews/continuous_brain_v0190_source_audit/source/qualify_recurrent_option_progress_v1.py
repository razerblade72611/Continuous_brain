#!/usr/bin/env python3
"""Qualification for v0.19.0 recurrent option-progress controller."""
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
from continuous_brain.recurrent_option_progress_v1 import (
    RecurrentOptionProgressInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    CanonicalExecutiveRuleWorld,
    RULE_NONE,
    target_action,
    UP,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
PARENT_CHECKPOINT = (
    "state/option_intent_executive_v1/latest.portable.pt"
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
EXPECTED_PARENT_FORMAT = "option_intent_executive_training_v1"
RULES = ("invert_lr", "vertical", "vertical_reverse")


class DMLSafeAdam:
    def __init__(self, params, lr=2e-4):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.b1 = .9
        self.b2 = .999
        self.eps = 1e-8
        self.n = 0
        self.state = [
            {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            for p in self.params
        ]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.n += 1
        c1 = 1 - self.b1 ** self.n
        c2 = 1 - self.b2 ** self.n
        for p, s in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all().item():
                raise RuntimeError("non-finite gradient")
            s["m"].mul_(self.b1).add_(
                g,
                alpha=1-self.b1,
            )
            s["v"].mul_(self.b2).add_(
                g*g,
                alpha=1-self.b2,
            )
            den = (
                s["v"].sqrt()
                / math.sqrt(c2)
                + self.eps
            )
            p.add_(
                s["m"] / den,
                alpha=-self.lr/c1,
            )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(
            lambda: f.read(8*1024*1024),
            b"",
        ):
            h.update(c)
    return h.hexdigest()


def clone(m):
    return {
        k: v.detach().cpu().clone()
        for k, v in m.state_dict().items()
    }


def digest(st):
    h = hashlib.sha256()
    for k in sorted(st):
        t = st[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_exact(ref, module, label):
    for k, v in ref.items():
        got = module.state_dict()[k].detach().cpu()
        if not torch.equal(v, got):
            raise AssertionError(
                f"{label}: changed {k}"
            )


def gmax(g):
    if g is None:
        return 0.0
    if not torch.isfinite(g).all().item():
        raise RuntimeError("non-finite gradient")
    return float(g.detach().abs().max().cpu())


def load(device):
    if sha(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted SHA mismatch")

    parent_sha = sha(PARENT_CHECKPOINT)
    parent = torch.load(
        PARENT_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    if (
        parent.get("format") != EXPECTED_PARENT_FORMAT
        or int(parent.get("global_update", -1)) != 4800
    ):
        raise RuntimeError(
            "expected v0.18 update-4800 parent"
        )
    if (
        parent.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError(
            "parent admitted provenance mismatch"
        )

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, bsha = _load_base_checkpoint(
        BASE_CHECKPOINT
    )
    if (
        bsha != EXPECTED_BASE_SHA256
        or digest(clone(base))
        != EXPECTED_BASE_DIGEST
    ):
        raise RuntimeError("base mismatch")

    admitted = (
        CapacityMatchedRecallWorkingMemoryInfant
        .grow_from(base, cfg)
    )
    admitted.memory.load_state_dict(
        ap["memory"],
        strict=True,
    )
    admitted.recall.load_state_dict(
        ap["recall"],
        strict=True,
    )
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = (
        RecurrentOptionProgressInfant
        .grow_from_admitted(admitted)
        .to(device)
    )
    model.load_frozen_parent_executive(
        parent["executive"]
    )
    model.train(True)
    model.admitted.eval()
    model.base.eval()
    model.frozen_parent_executive.eval()

    return (
        model,
        cfg,
        clone(admitted),
        clone(base),
        clone(model.frozen_parent_executive),
        model.trainable_parameters(),
        parent_sha,
    )


@torch.no_grad()
def protected_sequence(
    model,
    cfg,
    device,
    cid,
    cue,
):
    env = CanonicalExecutiveRuleWorld(
        cid,
        cue=cue,
        rule="invert_lr",
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )
    rows = []

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

        if env.instruction_visible():
            (
                hist,
                recall,
                _,
                _,
                base_logits,
            ) = model.components(
                image,
                context,
                memory,
                prev,
                hidden,
            )
            rows.append({
                "hidden":
                    hist["hidden"].detach(),
                "workspace":
                    hist["historical_workspace"].detach(),
                "recall":
                    recall.detach(),
                "base_logits":
                    base_logits.detach(),
                "response":
                    bool(env.at_response_step()),
            })

        if env.at_response_step():
            break

        teacher = int(env.teacher_action())
        out = model.admitted.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"]
        memory = out["memory_next"]
        prev = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )

    if len(rows) < 2 or not rows[-1]["response"]:
        raise AssertionError("bad protected sequence")
    return rows


def qualification_loss(
    model,
    cfg,
    device,
    cid,
):
    seqs = {
        c: protected_sequence(
            model,
            cfg,
            device,
            cid,
            c,
        )
        for c in ("left", "right")
    }
    if len(seqs["left"]) != len(seqs["right"]):
        raise AssertionError(
            "paired option sequences differ in length"
        )

    gate_losses = []
    invariant = []
    own = []
    donor = []
    pre = []

    rolled = {}
    active_rule = torch.tensor(
        [1],
        dtype=torch.long,
        device=device,
    )

    for cue in ("left", "right"):
        state = model.initial_option_state(
            1,
            device,
        )
        rolled[cue] = []
        for snap in seqs[cue]:
            gate, _, state, _ = (
                model.advance_progress(
                    snap["hidden"],
                    snap["workspace"],
                    snap["base_logits"],
                    state,
                    active_rule,
                )
            )
            target = torch.tensor(
                [[
                    1.0
                    if snap["response"]
                    else 0.0
                ]],
                device=device,
            )
            gate_losses.append(
                ((gate-target)**2).mean()
            )
            rolled[cue].append(
                (snap, gate)
            )

    for i in range(len(rolled["left"])):
        invariant.append(
            (
                rolled["left"][i][1]
                - rolled["right"][i][1]
            ).pow(2).mean()
        )

    for cue in ("left", "right"):
        donor_cue = (
            "right"
            if cue == "left"
            else "left"
        )
        response_snap, response_gate = (
            rolled[cue][-1]
        )
        donor_recall = (
            rolled[donor_cue][-1][0]["recall"]
        )

        for rid, rule in enumerate(RULES, 1):
            rule_id = torch.tensor(
                [rid],
                dtype=torch.long,
                device=device,
            )

            for snap, gate in rolled[cue][:-1]:
                logits, _, _ = model.apply_intent(
                    snap["recall"],
                    rule_id,
                    snap["base_logits"],
                    gate,
                )
                pre.append(
                    F.cross_entropy(
                        logits,
                        torch.tensor(
                            [UP],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )

            own_logits, _, _ = model.apply_intent(
                response_snap["recall"],
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            own.append(
                F.cross_entropy(
                    own_logits,
                    torch.tensor(
                        [target_action(cue, rule)],
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )

            donor_logits, _, _ = model.apply_intent(
                donor_recall,
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            donor.append(
                F.cross_entropy(
                    donor_logits,
                    torch.tensor(
                        [
                            target_action(
                                donor_cue,
                                rule,
                            )
                        ],
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )

    gate_l = torch.stack(gate_losses).mean()
    inv_l = torch.stack(invariant).mean()
    pre_l = torch.stack(pre).mean()
    own_l = torch.stack(own).mean()
    donor_l = torch.stack(donor).mean()

    total = (
        .45*gate_l
        + .20*pre_l
        + .20*own_l
        + .10*donor_l
        + .05*inv_l
    )
    return total


def main():
    torch.manual_seed(919191)
    device = torch_directml.device()
    (
        model,
        cfg,
        admitted_ref,
        base_ref,
        parent_ref,
        params,
        parent_sha,
    ) = load(device)

    print(
        "RECURRENT_OPTION_PROGRESS_QUALIFICATION_V1"
    )
    print("device:", device)
    print(
        "parent_v018_checkpoint_sha256:",
        parent_sha,
    )
    print(
        "trainable_parameter_count:",
        sum(p.numel() for p in params),
    )
    print(
        "frozen_parent_what_parameters:",
        sum(
            p.numel()
            for p in model
            .frozen_parent_executive
            .parameters()
        ),
    )
    print("what_path_frozen: true")
    print("progress_is_recurrent: true")
    print("progress_receives_rule_identity: false")
    print("timing_loss_uses_bce: false")
    print("admitted_brain_in_optimizer: false")
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    # Exact no-op at growth even though the frozen WHAT head is already trained.
    seq = protected_sequence(
        model,
        cfg,
        device,
        0,
        "left",
    )
    state = model.initial_option_state(
        1,
        device,
    )
    active_rule = torch.tensor(
        [1],
        dtype=torch.long,
        device=device,
    )
    for snap in seq:
        gate, _, state, _ = (
            model.advance_progress(
                snap["hidden"],
                snap["workspace"],
                snap["base_logits"],
                state,
                active_rule,
            )
        )
        if torch.count_nonzero(
            gate.detach().cpu()
        ).item():
            raise AssertionError(
                "initial recurrent gate is not exact zero"
            )
    print("EXACT_INITIAL_PROGRESS_NOOP_OK")

    opt = DMLSafeAdam(
        params,
        lr=2e-4,
    )

    for update, cid in enumerate(
        (0, 17, 2),
        1,
    ):
        opt.zero_grad()
        loss = qualification_loss(
            model,
            cfg,
            device,
            cid,
        )
        loss.backward()

        outg = gmax(
            model.progress.gate_output.weight.grad
        )
        recg = gmax(
            model.progress.recurrent.weight.grad
        )
        hing = gmax(
            model.progress.hidden_proj.weight.grad
        )
        gain_g = gmax(model.intent_gain.grad)

        print(
            f"update{update}_loss:",
            float(loss.detach().cpu()),
        )
        print(
            f"update{update}_gate_output_grad_max:",
            outg,
        )
        print(
            f"update{update}_recurrent_grad_max:",
            recg,
        )
        print(
            f"update{update}_hidden_input_grad_max:",
            hing,
        )
        print(
            f"update{update}_intent_gain_grad_max:",
            gain_g,
        )

        if outg <= 0:
            raise AssertionError(
                "gate output did not receive gradient"
            )

        if update == 1:
            if recg != 0 or hing != 0:
                raise AssertionError(
                    "progress upstream learned before "
                    "zero-init gate moved"
                )
        else:
            if min(recg, hing) <= 0:
                raise AssertionError(
                    "recurrent progress path not load-bearing"
                )

        opt.step()

        assert_exact(
            admitted_ref,
            model.admitted,
            "admitted brain",
        )
        assert_exact(
            base_ref,
            model.base,
            "historical base",
        )
        assert_exact(
            parent_ref,
            model.frozen_parent_executive,
            "frozen v0.18 WHAT",
        )

    # Exact causal masks after learning.
    with torch.no_grad():
        snap = seq[-1]
        state = model.initial_option_state(
            1,
            device,
        )
        none = torch.tensor(
            [RULE_NONE],
            dtype=torch.long,
            device=device,
        )
        gate, _, state2, _ = (
            model.advance_progress(
                snap["hidden"],
                snap["workspace"],
                snap["base_logits"],
                state,
                none,
            )
        )
        if (
            torch.count_nonzero(gate.cpu()).item()
            or torch.count_nonzero(state2.cpu()).item()
        ):
            raise AssertionError(
                "RULE_NONE did not reset progress"
            )

        zero_recall = torch.zeros_like(
            snap["recall"]
        )
        logits, delta, _ = model.apply_intent(
            zero_recall,
            active_rule,
            snap["base_logits"],
            torch.ones(
                1,
                1,
                device=device,
            ),
        )
        if torch.count_nonzero(
            delta.cpu()
        ).item():
            raise AssertionError(
                "zero recall did not eliminate intent"
            )
        if not torch.equal(
            logits,
            snap["base_logits"],
        ):
            raise AssertionError(
                "zero recall changed base logits"
            )

    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("frozen_v018_what_bit_exact: true")
    print("rule_none_resets_progress: true")
    print("zero_recall_exact_noop: true")
    print(
        "RECURRENT_OPTION_PROGRESS_QUALIFICATION_V1: PASS"
    )


if __name__ == "__main__":
    main()
