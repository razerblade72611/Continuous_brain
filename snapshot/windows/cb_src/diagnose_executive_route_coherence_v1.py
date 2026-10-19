#!/usr/bin/env python3
"""
Fresh-DEV current-memory route-coherence diagnostic for v0.13.2.

No training. Fresh TEST remains sealed.

At the delayed response, freeze the recipient recurrent hidden state and vary
three current-step cue-bearing routes independently:

H = historical frozen memory projection into the frozen recurrent step
D = admitted direct nonlinear memory->workspace recall
E = novel-rule executive memory input

Modes
-----
OWN                 H=own,   D=own,   E=own
E_DONOR             H=own,   D=own,   E=donor
DE_DONOR            H=own,   D=donor, E=donor
HDE_DONOR           H=donor, D=donor, E=donor
D_DONOR             H=own,   D=donor, E=own
H_DONOR             H=donor, D=own,   E=own
HZ_DE_OWN           H=zero,  D=own,   E=own
HZ_DE_DONOR         H=zero,  D=donor, E=donor
HRESET_HZ_DE_OWN    hidden=zero, H=zero, D=own,   E=own
HRESET_HZ_DE_DONOR  hidden=zero, H=zero, D=donor, E=donor

For donor modes, success means matching the response required by the donor cue
under the same late rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_directml

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.executive_novel_rules_v2 import (
    NovelRuleExecutiveInfant,
)
from continuous_brain.executive_novel_rule_world_v2 import (
    ACTION_NAMES,
    DEV_CONDITION_RANGE,
    ExecutiveNovelRuleWorld,
    target_action,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
EXEC_CHECKPOINT = "state/executive_novel_rules_v2/latest.portable.pt"

EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_ADMITTED_SHA256 = (
    "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
)
EXPECTED_EXEC_SHA256 = (
    "42116f09bf89298938fb3153bbb7d9f5ce2ea671407f192ca4491af64d8abe6c"
)
EXPECTED_EXEC_FORMAT = "executive_novel_rules_training_v2"

RULES = ("invert_lr", "vertical", "vertical_reverse")
MODES = (
    "OWN",
    "E_DONOR",
    "DE_DONOR",
    "HDE_DONOR",
    "D_DONOR",
    "H_DONOR",
    "HZ_DE_OWN",
    "HZ_DE_DONOR",
    "HRESET_HZ_DE_OWN",
    "HRESET_HZ_DE_DONOR",
)

DONOR_TARGET_MODES = {
    "E_DONOR",
    "DE_DONOR",
    "HDE_DONOR",
    "D_DONOR",
    "H_DONOR",
    "HZ_DE_DONOR",
    "HRESET_HZ_DE_DONOR",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def assert_state_exact(reference, module, label):
    current = module.state_dict()
    for k, expected in reference.items():
        got = current[k].detach().cpu()
        if not torch.equal(expected, got):
            raise AssertionError(f"{label} changed: {k}")


def load_model(device):
    if sha256(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")
    if sha256(EXEC_CHECKPOINT) != EXPECTED_EXEC_SHA256:
        raise RuntimeError(
            "executive checkpoint SHA mismatch; expected frozen v2 result"
        )

    admitted_payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    exec_payload = torch.load(
        EXEC_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    if exec_payload.get("format") != EXPECTED_EXEC_FORMAT:
        raise RuntimeError("executive format mismatch")
    if int(exec_payload.get("update", -1)) != 1600:
        raise RuntimeError("executive checkpoint is not update 1600")

    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(
        admitted_payload["memory"],
        strict=True,
    )
    admitted.recall.load_state_dict(
        admitted_payload["recall"],
        strict=True,
    )
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = NovelRuleExecutiveInfant.grow_from_admitted(admitted)
    model.executive.load_state_dict(
        exec_payload["executive"],
        strict=True,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg


@torch.no_grad()
def snapshot(model, cfg, device, cid, cue):
    env = ExecutiveNovelRuleWorld(
        cid,
        cue=cue,
        rule="invert_lr",
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if env.at_response_step():
            return {
                "image": image.detach().clone(),
                "context": context.detach().clone(),
                "hidden": hidden.detach().clone(),
                "memory": memory.detach().clone(),
                "prev": prev.detach().clone(),
            }

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

    raise RuntimeError("no response snapshot")


@torch.no_grad()
def composed_logits(
    model,
    snap,
    *,
    hidden,
    historical_memory,
    direct_memory,
    executive_memory,
    rule_id,
):
    hist = model.admitted.historical_step(
        snap["image"],
        snap["context"],
        historical_memory,
        snap["prev"],
        hidden,
    )

    direct_residual, _ = model.admitted.recall(
        hist["hidden"],
        direct_memory,
    )

    executive_residual, _ = model.executive(
        executive_memory,
        torch.tensor(
            [rule_id],
            dtype=torch.long,
            device=executive_memory.device,
        ),
    )

    workspace = (
        hist["historical_workspace"]
        + direct_residual
        + executive_residual
    )
    logits = model.base.motor(workspace)[0].detach().float().cpu()

    return {
        "logits": logits,
        "direct_residual":
            direct_residual.detach().float().cpu(),
        "executive_residual":
            executive_residual.detach().float().cpu(),
    }


def mode_inputs(mode, own, donor, zero, hidden, zero_hidden):
    if mode == "OWN":
        return hidden, own, own, own
    if mode == "E_DONOR":
        return hidden, own, own, donor
    if mode == "DE_DONOR":
        return hidden, own, donor, donor
    if mode == "HDE_DONOR":
        return hidden, donor, donor, donor
    if mode == "D_DONOR":
        return hidden, own, donor, own
    if mode == "H_DONOR":
        return hidden, donor, own, own
    if mode == "HZ_DE_OWN":
        return hidden, zero, own, own
    if mode == "HZ_DE_DONOR":
        return hidden, zero, donor, donor
    if mode == "HRESET_HZ_DE_OWN":
        return zero_hidden, zero, own, own
    if mode == "HRESET_HZ_DE_DONOR":
        return zero_hidden, zero, donor, donor
    raise ValueError(mode)


def classify(logits, target):
    pred = int(torch.argmax(logits).item())
    return {
        "pred": pred,
        "pred_name": ACTION_NAMES[pred],
        "correct": bool(pred == target),
    }


def eval_condition(model, cfg, device, cid):
    left = snapshot(model, cfg, device, cid, "left")
    right = snapshot(model, cfg, device, cid, "right")

    for field in ("image", "context", "prev"):
        if not torch.equal(left[field].cpu(), right[field].cpu()):
            raise AssertionError(f"paired {field} mismatch")

    snaps = {"left": left, "right": right}
    row = {
        "condition_id": int(cid),
        "modes": {},
    }

    for mode in MODES:
        row["modes"][mode] = {}

        for cue, snap in snaps.items():
            donor_cue = "right" if cue == "left" else "left"
            donor_snap = snaps[donor_cue]

            own = snap["memory"]
            donor = donor_snap["memory"]
            zero = torch.zeros_like(own)
            zero_hidden = model.initial_state(1, device)

            (
                used_hidden,
                historical_memory,
                direct_memory,
                executive_memory,
            ) = mode_inputs(
                mode,
                own,
                donor,
                zero,
                snap["hidden"],
                zero_hidden,
            )

            target_cue = (
                donor_cue
                if mode in DONOR_TARGET_MODES
                else cue
            )

            row["modes"][mode][cue] = {}

            for rule_id, rule in enumerate(RULES, 1):
                target = target_action(target_cue, rule)

                out = composed_logits(
                    model,
                    snap,
                    hidden=used_hidden,
                    historical_memory=historical_memory,
                    direct_memory=direct_memory,
                    executive_memory=executive_memory,
                    rule_id=rule_id,
                )

                row["modes"][mode][cue][rule] = {
                    **classify(out["logits"], target),
                    "target_cue": target_cue,
                    "target_action": int(target),
                    "target_action_name": ACTION_NAMES[target],
                    "direct_residual_max": float(
                        out["direct_residual"].abs().max()
                    ),
                    "executive_residual_max": float(
                        out["executive_residual"].abs().max()
                    ),
                }

    return row


def pair_correct(row, mode, rule):
    return bool(
        row["modes"][mode]["left"][rule]["correct"]
        and row["modes"][mode]["right"][rule]["correct"]
    )


def summarize(rows):
    result = {}

    for mode in MODES:
        result[mode] = {
            rule: mean([
                float(pair_correct(r, mode, rule))
                for r in rows
            ])
            for rule in RULES
        }
        result[mode]["six_way"] = mean([
            float(
                all(
                    r["modes"][mode][cue][rule]["correct"]
                    for cue in ("left", "right")
                    for rule in RULES
                )
            )
            for r in rows
        ])

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=(
            "state/executive_novel_rules_v2/"
            "route_coherence_diagnostic_v1.json"
        ),
    )
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg = load_model(device)

    admitted_ref = clone_state(model.admitted)
    base_ref = clone_state(model.base)

    print("EXECUTIVE_ROUTE_COHERENCE_DIAGNOSTIC_V1")
    print("device:", device)
    print("checkpoint_sha256:", EXPECTED_EXEC_SHA256)
    print("split: fresh_dev")
    print("fresh_test_touched: false")

    rows = []
    total = len(DEV_CONDITION_RANGE)

    for i, cid in enumerate(DEV_CONDITION_RANGE, 1):
        rows.append(
            eval_condition(
                model,
                cfg,
                device,
                int(cid),
            )
        )

        if i == 1 or i % 50 == 0 or i == total:
            s = summarize(rows)
            print(
                f"evaluated={i}/{total} "
                f"EonlyV={s['E_DONOR']['vertical']:.4f} "
                f"DE_V={s['DE_DONOR']['vertical']:.4f} "
                f"HDE_V={s['HDE_DONOR']['vertical']:.4f} "
                f"EonlyVR="
                f"{s['E_DONOR']['vertical_reverse']:.4f} "
                f"DE_VR="
                f"{s['DE_DONOR']['vertical_reverse']:.4f} "
                f"HDE_VR="
                f"{s['HDE_DONOR']['vertical_reverse']:.4f}"
            )

    assert_state_exact(
        admitted_ref,
        model.admitted,
        "admitted brain",
    )
    assert_state_exact(
        base_ref,
        model.base,
        "historical base",
    )

    summary = summarize(rows)

    result = {
        "format": "executive_route_coherence_diagnostic_v1",
        "split": "fresh_dev",
        "checkpoint_sha256": EXPECTED_EXEC_SHA256,
        "summary": summary,
        "per_condition": rows,
        "fresh_test_touched": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print()
    print("ROUTE_COHERENCE_SUMMARY")
    for mode in MODES:
        s = summary[mode]
        print(
            f"{mode}: "
            f"invert={s['invert_lr']:.4f} "
            f"vertical={s['vertical']:.4f} "
            f"vreverse={s['vertical_reverse']:.4f} "
            f"six={s['six_way']:.4f}"
        )

    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("FRESH_TEST_TOUCHED: false")
    print("EXECUTIVE_ROUTE_COHERENCE_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
