#!/usr/bin/env python3
"""
Frozen fresh-DEV causal evaluation for the novel-rule executive v2.

No fitting on DEV.
Fresh TEST remains sealed.

The admitted recurrent/working-memory state is computed once per cue and held
fixed while the new executive is tested under every novel rule.

Causal interventions:
- executive OFF;
- executive memory ZERO;
- executive memory swapped to opposite-cue donor;
- RULE_NONE exact no-op.

Because ALL active rules are novel, donor consistency is required for every
rule; there is no inherited NORMAL exception.
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
    RULE_NONE,
    ExecutiveNovelRuleWorld,
    generate_executive_rule_spec,
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
EXPECTED_FORMAT = "executive_novel_rules_training_v2"
RULES = ("invert_lr", "vertical", "vertical_reverse")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def clone(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def digest(state):
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


def load(path, device):
    ep = torch.load(path, map_location="cpu", weights_only=True)
    if ep.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("executive checkpoint format mismatch")
    if int(ep.get("update", -1)) != 1600:
        raise RuntimeError("expected update-1600 executive checkpoint")
    if (
        ep.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("source admitted SHA mismatch")

    if sha(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted file SHA mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, bsha = _load_base_checkpoint(BASE_CHECKPOINT)
    if bsha != EXPECTED_BASE_SHA256:
        raise RuntimeError("base SHA mismatch")
    if digest(clone(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(
        base,
        cfg,
    )
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for q in admitted.parameters():
        q.requires_grad_(False)

    model = NovelRuleExecutiveInfant.grow_from_admitted(admitted)
    model.executive.load_state_dict(ep["executive"], strict=True)
    model = model.to(device)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    return model, cfg, ep


@torch.no_grad()
def snapshot(model, cfg, device, cid, cue):
    env = ExecutiveNovelRuleWorld(
        cid,
        cue=cue,
        rule="invert_lr",
    )
    obs = env.reset()
    h = model.initial_state(1, device)
    mem = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if env.at_response_step():
            return {
                "image": image.detach().clone(),
                "context": context.detach().clone(),
                "hidden": h.detach().clone(),
                "memory": mem.detach().clone(),
                "prev": prev.detach().clone(),
            }

        teacher = int(env.teacher_action())
        out = model.admitted.step(
            image,
            context,
            mem,
            prev,
            h,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        h = out["hidden"]
        mem = out["memory_next"]
        prev = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )

    raise RuntimeError("no response snapshot")


@torch.no_grad()
def forward(
    model,
    snap,
    exec_memory,
    rule_id,
    executive_on=True,
):
    admitted = model.admitted.step(
        snap["image"],
        snap["context"],
        snap["memory"],
        snap["prev"],
        snap["hidden"],
        use_memory=True,
        use_recall=True,
    )

    if executive_on:
        rid = torch.tensor(
            [rule_id],
            dtype=torch.long,
            device=snap["memory"].device,
        )
        residual, _ = model.executive(exec_memory, rid)
    else:
        residual = torch.zeros_like(admitted["workspace_for_world"])

    logits = model.base.motor(
        admitted["workspace_for_world"] + residual
    )[0].detach().float().cpu()

    return logits, residual.detach().float().cpu()


def classify(logits, target):
    pred = int(torch.argmax(logits).item())
    return {
        "pred": pred,
        "pred_name": ACTION_NAMES[pred],
        "correct": bool(pred == target),
        "target": int(target),
        "target_name": ACTION_NAMES[target],
    }


def eval_condition(model, cfg, device, cid):
    left = snapshot(model, cfg, device, cid, "left")
    right = snapshot(model, cfg, device, cid, "right")

    for k in ("image", "context", "prev"):
        if not torch.equal(left[k].cpu(), right[k].cpu()):
            raise AssertionError(f"pair {k} mismatch")

    snaps = {"left": left, "right": right}
    result = {}

    for cue, snap in snaps.items():
        donor = right["memory"] if cue == "left" else left["memory"]
        donor_cue = "right" if cue == "left" else "left"
        zero = torch.zeros_like(snap["memory"])
        result[cue] = {}

        for rid, rule in enumerate(RULES, 1):
            target = target_action(cue, rule)
            donor_target = target_action(donor_cue, rule)

            own, residual = forward(
                model,
                snap,
                snap["memory"],
                rid,
                True,
            )
            off, _ = forward(
                model,
                snap,
                snap["memory"],
                rid,
                False,
            )
            z, _ = forward(
                model,
                snap,
                zero,
                rid,
                True,
            )
            swapped, _ = forward(
                model,
                snap,
                donor,
                rid,
                True,
            )
            none, _ = forward(
                model,
                snap,
                snap["memory"],
                RULE_NONE,
                True,
            )

            result[cue][rule] = {
                "normal": classify(own, target),
                "executive_off": classify(off, target),
                "memory_zero": classify(z, target),
                "memory_swapped_donor_target": classify(
                    swapped,
                    donor_target,
                ),
                "rule_none": classify(none, target),
                "none_vs_off_max_abs_diff": float(
                    (none - off).abs().max()
                ),
                "residual_max": float(residual.abs().max()),
            }

    spec = generate_executive_rule_spec(cid)
    return {
        "condition_id": int(cid),
        "members": result,
        "factors": {
            "corridor_length": spec.corridor_length,
            "goal_x_offset": spec.goal_x_offset,
            "goal_y_offset": spec.goal_y_offset,
            "cue_steps": spec.cue_steps,
            "move_step": spec.move_step,
        },
    }


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
    s = {}

    for mode in (
        "normal",
        "executive_off",
        "memory_zero",
        "memory_swapped_donor_target",
        "rule_none",
    ):
        s[mode] = {
            rule: mean([
                float(pair(r, rule, mode))
                for r in rows
            ])
            for rule in RULES
        }
        s[mode]["six_way"] = mean([
            float(six(r, mode))
            for r in rows
        ])

    s["none_max_diff"] = max(
        r["members"][cue][rule]["none_vs_off_max_abs_diff"]
        for r in rows
        for cue in ("left", "right")
        for rule in RULES
    )

    residuals = [
        r["members"][cue][rule]["residual_max"]
        for r in rows
        for cue in ("left", "right")
        for rule in RULES
    ]
    s["residual_mean"] = mean(residuals)

    return s


def factor_summary(rows):
    out = {}

    for field in (
        "corridor_length",
        "goal_x_offset",
        "goal_y_offset",
        "cue_steps",
        "move_step",
    ):
        out[field] = {}

        for v in sorted({
            r["factors"][field]
            for r in rows
        }):
            group = [
                r for r in rows
                if r["factors"][field] == v
            ]
            s = summarize(group)
            record = {"n": len(group)}
            for rule in RULES:
                record[f"{rule}_pair_full"] = s["normal"][rule]
            record["six_way_full"] = s["normal"]["six_way"]
            out[field][str(v)] = record

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, payload = load(
        Path(args.checkpoint),
        device,
    )

    admitted_ref = clone(model.admitted)
    base_ref = clone(model.base)

    print("EXECUTIVE_NOVEL_RULE_FRESH_DEV_EVAL_V2")
    print("device:", device)
    print("checkpoint:", args.checkpoint)
    print("checkpoint_sha256:", sha(args.checkpoint))
    print("fresh_dev_range: 1600:2000")
    print("fresh_test_touched: false")

    rows = []

    for i, cid in enumerate(DEV_CONDITION_RANGE, 1):
        rows.append(
            eval_condition(
                model,
                cfg,
                device,
                int(cid),
            )
        )
        if i == 1 or i % 50 == 0 or i == len(DEV_CONDITION_RANGE):
            s = summarize(rows)
            print(
                f"evaluated={i}/400 "
                f"invert={s['normal']['invert_lr']:.4f} "
                f"vertical={s['normal']['vertical']:.4f} "
                f"vreverse={s['normal']['vertical_reverse']:.4f} "
                f"six={s['normal']['six_way']:.4f}"
            )

    for k, v in admitted_ref.items():
        if not torch.equal(
            v,
            model.admitted.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"admitted changed {k}")

    for k, v in base_ref.items():
        if not torch.equal(
            v,
            model.base.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"base changed {k}")

    s = summarize(rows)
    factors = factor_summary(rows)

    result = {
        "format": "executive_novel_rule_fresh_dev_v2",
        "split": "fresh_dev",
        "condition_range": [1600, 2000],
        "summary": s,
        "factor_summary": factors,
        "per_condition": rows,
        "fresh_test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print()
    for rule in RULES:
        print(
            f"{rule}: "
            f"normal_pair={s['normal'][rule]:.4f} "
            f"off_pair={s['executive_off'][rule]:.4f} "
            f"zero_pair={s['memory_zero'][rule]:.4f} "
            f"donor_pair="
            f"{s['memory_swapped_donor_target'][rule]:.4f}"
        )

    print(
        f"six_way_full={s['normal']['six_way']:.4f}"
    )
    print(
        f"none_exact_noop_max_diff={s['none_max_diff']:.9g}"
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("FRESH_TEST_TOUCHED: false")
    print("EXECUTIVE_NOVEL_RULE_FRESH_DEV_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
