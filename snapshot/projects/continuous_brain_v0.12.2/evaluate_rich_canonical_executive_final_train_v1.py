#!/usr/bin/env python3
"""Fixed-final TRAIN causal evaluation for rich canonical executive v1."""
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
from continuous_brain.rich_canonical_executive_v1 import (
    RichCanonicalExecutiveInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    ACTION_NAMES,
    TRAIN_CONDITION_RANGE,
    RULE_NONE,
    CanonicalExecutiveRuleWorld,
    generate_canonical_rule_spec,
    target_action,
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
EXPECTED_FORMAT = "rich_canonical_executive_training_v1"
RULES = ("invert_lr", "vertical", "vertical_reverse")


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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load_model(path, device):
    ep = torch.load(path, map_location="cpu", weights_only=True)
    if ep.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("rich executive checkpoint format mismatch")
    if int(ep.get("global_update", -1)) < 1600:
        raise RuntimeError("rich executive checkpoint too early")
    if (
        ep.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("admitted provenance mismatch")

    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = RichCanonicalExecutiveInfant.grow_from_admitted(admitted)
    model.executive.load_state_dict(ep["executive"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, ep


@torch.no_grad()
def snapshot(model, cfg, device, cid, cue):
    env = CanonicalExecutiveRuleWorld(cid, cue=cue, rule="invert_lr")
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
def compose(
    model,
    snap,
    canonical_memory,
    rule_id,
    *,
    executive_on=True,
    canonical_on=True,
):
    hist = model.admitted.historical_step(
        snap["image"],
        snap["context"],
        snap["memory"],
        snap["prev"],
        snap["hidden"],
    )

    if canonical_on:
        direct, _ = model.admitted.recall(
            hist["hidden"],
            canonical_memory,
        )
    else:
        direct = torch.zeros_like(hist["historical_workspace"])

    workspace = hist["historical_workspace"] + direct

    if executive_on and canonical_on:
        delta, _ = model.executive(
            direct,
            torch.tensor(
                [rule_id],
                dtype=torch.long,
                device=canonical_memory.device,
            ),
        )
    elif executive_on:
        delta, _ = model.executive(
            torch.zeros_like(direct),
            torch.tensor(
                [rule_id],
                dtype=torch.long,
                device=canonical_memory.device,
            ),
        )
    else:
        delta = torch.zeros_like(workspace)

    logits = model.base.motor(workspace + delta)[0].detach().float().cpu()
    return logits


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
    row = {"condition_id": int(cid), "members": {}}

    for cue, snap in snaps.items():
        donor_cue = "right" if cue == "left" else "left"
        donor = snaps[donor_cue]["memory"]
        own = snap["memory"]
        zero = torch.zeros_like(own)
        row["members"][cue] = {}

        for rid, rule in enumerate(RULES, 1):
            target = target_action(cue, rule)
            donor_target = target_action(donor_cue, rule)

            normal = compose(model, snap, own, rid, executive_on=True, canonical_on=True)
            off = compose(model, snap, own, rid, executive_on=False, canonical_on=True)
            zeroed = compose(model, snap, zero, rid, executive_on=True, canonical_on=False)
            donor_logits = compose(model, snap, donor, rid, executive_on=True, canonical_on=True)
            none = compose(model, snap, own, RULE_NONE, executive_on=True, canonical_on=True)
            admitted = compose(model, snap, own, rid, executive_on=False, canonical_on=True)

            row["members"][cue][rule] = {
                "normal": classify(normal, target),
                "executive_off": classify(off, target),
                "canonical_zero": classify(zeroed, target),
                "canonical_donor": classify(donor_logits, donor_target),
                "none": classify(none, target),
                "none_vs_admitted_max_abs_diff": float(
                    (none - admitted).abs().max()
                ),
            }

    spec = generate_canonical_rule_spec(cid)
    row["factors"] = {
        "corridor_length": spec.corridor_length,
        "goal_x_offset": spec.goal_x_offset,
        "goal_y_offset": spec.goal_y_offset,
        "cue_steps": spec.cue_steps,
        "move_step": spec.move_step,
    }
    return row


def pair(row, rule, mode):
    return bool(
        row["members"]["left"][rule][mode]["correct"]
        and row["members"]["right"][rule][mode]["correct"]
    )


def six(row, mode):
    return bool(
        all(
            row["members"][cue][rule][mode]["correct"]
            for cue in ("left", "right")
            for rule in RULES
        )
    )


def summarize(rows):
    result = {}
    for mode in (
        "normal",
        "executive_off",
        "canonical_zero",
        "canonical_donor",
        "none",
    ):
        result[mode] = {
            rule: mean([float(pair(r, rule, mode)) for r in rows])
            for rule in RULES
        }
        result[mode]["six_way"] = mean([
            float(six(r, mode))
            for r in rows
        ])

    result["none_max_diff"] = max(
        r["members"][cue][rule]["none_vs_admitted_max_abs_diff"]
        for r in rows
        for cue in ("left", "right")
        for rule in RULES
    )
    return result


def factor_summary(rows):
    result = {}
    for field in (
        "corridor_length",
        "goal_x_offset",
        "goal_y_offset",
        "cue_steps",
        "move_step",
    ):
        result[field] = {}
        for value in sorted({r["factors"][field] for r in rows}):
            group = [
                r for r in rows
                if r["factors"][field] == value
            ]
            s = summarize(group)
            record = {
                "n": len(group),
                "six_way_full": s["normal"]["six_way"],
            }
            for rule in RULES:
                record[f"{rule}_pair_full"] = s["normal"][rule]
            result[field][str(value)] = record
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, payload = load_model(Path(args.checkpoint), device)
    admitted_ref = clone_state(model.admitted)
    base_ref = clone_state(model.base)

    print("RICH_CANONICAL_EXECUTIVE_FINAL_TRAIN_V1")
    print("device:", device)
    print("checkpoint_global_update:", int(payload["global_update"]))
    print("train_range: 0:1600")
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.append(eval_condition(model, cfg, device, int(cid)))
        if i == 1 or i % 100 == 0 or i == 1600:
            s = summarize(rows)
            print(
                f"evaluated={i}/1600 "
                f"invert={s['normal']['invert_lr']:.4f} "
                f"vertical={s['normal']['vertical']:.4f} "
                f"vreverse={s['normal']['vertical_reverse']:.4f} "
                f"six={s['normal']['six_way']:.4f}"
            )

    for k, v in admitted_ref.items():
        if not torch.equal(v, model.admitted.state_dict()[k].detach().cpu()):
            raise AssertionError(f"admitted changed: {k}")
    for k, v in base_ref.items():
        if not torch.equal(v, model.base.state_dict()[k].detach().cpu()):
            raise AssertionError(f"base changed: {k}")

    summary = summarize(rows)
    factors = factor_summary(rows)

    result = {
        "format": "rich_canonical_executive_final_train_v1",
        "split": "train_only",
        "checkpoint_global_update": int(payload["global_update"]),
        "summary": summary,
        "factor_summary": factors,
        "per_condition": rows,
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={summary['normal'][rule]:.4f} "
            f"donor={summary['canonical_donor'][rule]:.4f} "
            f"off={summary['executive_off'][rule]:.4f} "
            f"zero={summary['canonical_zero'][rule]:.4f}"
        )
    print(
        f"six_way={summary['normal']['six_way']:.4f} "
        f"none_diff={summary['none_max_diff']:.9g}"
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("RICH_CANONICAL_EXECUTIVE_FINAL_TRAIN_COMPLETE")


if __name__ == "__main__":
    main()
