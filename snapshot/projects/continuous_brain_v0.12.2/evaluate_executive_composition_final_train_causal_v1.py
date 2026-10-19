#!/usr/bin/env python3
"""
Fixed-final TRAIN causal evaluation for the brain-integrated executive.

No optimizer steps.
No old DEV or TEST access.

At each delayed decision, keep the admitted recurrent/working-memory state fixed
and intervene only on the new executive inputs:
- instruction NORMAL / INVERT / NONE;
- own memory / zero memory / opposite-cue memory;
- executive disabled.

This tests whether the new brain-integrated executive learned a genuine
late-bound cue × instruction composition rather than merely preserving the
admitted NORMAL policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from continuous_brain.executive_working_memory_v1 import (
    ExecutiveCompositionInfant,
)
from continuous_brain.late_instruction_detour_world_v1 import (
    INSTRUCTION_NONE,
    INSTRUCTION_NORMAL,
    INSTRUCTION_INVERT,
    LateInstructionDetourWorld,
)
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    generate_task_spec,
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
EXPECTED_EXEC_FORMAT = "executive_composition_training_v1"
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 88173


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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(vals):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(vals)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        means.append(
            sum(vals[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    return [
        means[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
        means[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
    ]


def load_model(exec_checkpoint, device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    admitted_payload = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

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

    exec_payload = torch.load(
        exec_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if exec_payload.get("format") != EXPECTED_EXEC_FORMAT:
        raise RuntimeError("executive checkpoint format mismatch")
    if (
        exec_payload.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("executive source checkpoint SHA mismatch")
    if int(exec_payload.get("update", -1)) != 1600:
        raise RuntimeError("expected update-1600 executive checkpoint")

    model = ExecutiveCompositionInfant.grow_from_admitted(
        admitted
    )
    model.executive.load_state_dict(
        exec_payload["executive"],
        strict=True,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, exec_payload


@torch.no_grad()
def snapshot(model, cfg, device, cid, cue):
    """
    Run only the protected admitted brain under teacher forcing.

    No executive instruction is needed because the admitted recurrent/memory
    state never consumes instruction.
    """
    env = LateInstructionDetourWorld(
        cid,
        cue=cue,
        instruction="normal",
        instruction_delay=1,
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)
        teacher = int(env.teacher_action())

        if teacher in (LEFT, RIGHT):
            if not env.instruction_visible():
                raise AssertionError(
                    "decision reached before instruction visibility"
                )
            return {
                "image": image.detach().clone(),
                "context": context.detach().clone(),
                "hidden": hidden.detach().clone(),
                "memory": memory.detach().clone(),
                "prev": prev.detach().clone(),
                "decision_step": int(env.steps),
                "delay": int(env.steps - env.cue_steps),
            }

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

    raise RuntimeError("no delayed decision")


@torch.no_grad()
def decision_forward(
    model,
    snap,
    *,
    executive_memory,
    instruction_id,
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
        instruction = torch.tensor(
            [instruction_id],
            dtype=torch.long,
            device=snap["memory"].device,
        )
        residual, _ = model.executive(
            executive_memory,
            instruction,
        )
    else:
        residual = torch.zeros_like(
            admitted["workspace_for_world"]
        )

    workspace = admitted["workspace_for_world"] + residual
    logits = model.base.motor(workspace)[0].detach().float().cpu()

    return {
        "logits": logits,
        "residual": residual.detach().float().cpu(),
        "admitted_logits":
            admitted["logits"][0].detach().float().cpu(),
    }


def target_side(cue, instruction):
    if instruction == "normal":
        return LEFT if cue == "left" else RIGHT
    if instruction == "invert":
        return RIGHT if cue == "left" else LEFT
    raise ValueError(instruction)


def classify(logits, target):
    full = int(torch.argmax(logits).item())
    d = float(logits[RIGHT].item() - logits[LEFT].item())
    side = RIGHT if d > 0.0 else LEFT
    return {
        "full_pred": full,
        "full_pred_name": ACTION_NAMES[full],
        "full_correct": bool(full == target),
        "side_correct": bool(side == target),
        "right_minus_left": d,
        "correct_side_margin": float(
            d if target == RIGHT else -d
        ),
    }


def eval_condition(model, cfg, device, cid):
    left = snapshot(model, cfg, device, cid, "left")
    right = snapshot(model, cfg, device, cid, "right")

    for field in ("image", "context", "prev"):
        if not torch.equal(
            left[field].cpu(),
            right[field].cpu(),
        ):
            raise AssertionError(
                f"counterfactual {field} mismatch"
            )
    if left["decision_step"] != right["decision_step"]:
        raise AssertionError("decision timing mismatch")

    snaps = {"left": left, "right": right}
    rows = {}

    for cue, snap in snaps.items():
        own = snap["memory"]
        zero = torch.zeros_like(own)
        donor = (
            right["memory"]
            if cue == "left"
            else left["memory"]
        )

        rows[cue] = {}

        for instruction_name, instruction_id in (
            ("normal", INSTRUCTION_NORMAL),
            ("invert", INSTRUCTION_INVERT),
        ):
            target = target_side(cue, instruction_name)

            normal = decision_forward(
                model,
                snap,
                executive_memory=own,
                instruction_id=instruction_id,
                executive_on=True,
            )
            off = decision_forward(
                model,
                snap,
                executive_memory=own,
                instruction_id=instruction_id,
                executive_on=False,
            )
            zero_mem = decision_forward(
                model,
                snap,
                executive_memory=zero,
                instruction_id=instruction_id,
                executive_on=True,
            )
            swapped = decision_forward(
                model,
                snap,
                executive_memory=donor,
                instruction_id=instruction_id,
                executive_on=True,
            )
            none = decision_forward(
                model,
                snap,
                executive_memory=own,
                instruction_id=INSTRUCTION_NONE,
                executive_on=True,
            )

            donor_cue = "right" if cue == "left" else "left"
            donor_target = target_side(
                donor_cue,
                instruction_name,
            )

            rows[cue][instruction_name] = {
                "normal": classify(
                    normal["logits"],
                    target,
                ),
                "executive_off": classify(
                    off["logits"],
                    target,
                ),
                "memory_zero": classify(
                    zero_mem["logits"],
                    target,
                ),
                "memory_swapped_donor_target": classify(
                    swapped["logits"],
                    donor_target,
                ),
                "instruction_none": classify(
                    none["logits"],
                    target,
                ),
                "residual_max": float(
                    normal["residual"].abs().max()
                ),
                "none_vs_off_logits_max_abs_diff": float(
                    (
                        none["logits"]
                        - off["logits"]
                    ).abs().max()
                ),
            }

    # Same memory, opposite instruction must drive opposite target decisions.
    flip_consistency = {}
    for cue in ("left", "right"):
        n = rows[cue]["normal"]["normal"]
        i = rows[cue]["invert"]["normal"]
        flip_consistency[cue] = bool(
            n["side_correct"]
            and i["side_correct"]
            and (
                n["right_minus_left"]
                * i["right_minus_left"]
                < 0.0
            )
        )

    spec = generate_task_spec(cid)

    return {
        "condition_id": int(cid),
        "realized_hidden_delay": int(left["delay"]),
        "factors": {
            "corridor_length": float(spec.corridor_length),
            "goal_x_offset": float(spec.goal_x_offset),
            "goal_y_offset": float(spec.goal_y_offset),
            "cue_steps": int(spec.cue_steps),
            "move_step": float(spec.move_step),
        },
        "members": rows,
        "instruction_flip_consistency": flip_consistency,
    }


def pair_metric(row, instruction, mode, metric):
    vals = []
    for cue in ("left", "right"):
        x = row["members"][cue][instruction][mode]
        vals.append(bool(x[metric]))
    return bool(vals[0] and vals[1])


def four_way_metric(row, mode, metric):
    return bool(
        all(
            row["members"][cue][instruction][mode][metric]
            for cue in ("left", "right")
            for instruction in ("normal", "invert")
        )
    )


def summarize(rows):
    result = {}

    for mode in (
        "normal",
        "executive_off",
        "memory_zero",
        "memory_swapped_donor_target",
        "instruction_none",
    ):
        mode_result = {}
        for instruction in ("normal", "invert"):
            mode_result[instruction] = {
                "pair_full": mean([
                    float(
                        pair_metric(
                            r,
                            instruction,
                            mode,
                            "full_correct",
                        )
                    )
                    for r in rows
                ]),
                "pair_side": mean([
                    float(
                        pair_metric(
                            r,
                            instruction,
                            mode,
                            "side_correct",
                        )
                    )
                    for r in rows
                ]),
            }
        mode_result["four_way_full"] = mean([
            float(
                four_way_metric(
                    r,
                    mode,
                    "full_correct",
                )
            )
            for r in rows
        ])
        mode_result["four_way_side"] = mean([
            float(
                four_way_metric(
                    r,
                    mode,
                    "side_correct",
                )
            )
            for r in rows
        ])
        result[mode] = mode_result

    result["instruction_flip_consistency"] = mean([
        float(
            r["instruction_flip_consistency"]["left"]
            and r["instruction_flip_consistency"]["right"]
        )
        for r in rows
    ])

    result["none_exact_noop_max_diff"] = max(
        r["members"][cue][instruction]["none_vs_off_logits_max_abs_diff"]
        for r in rows
        for cue in ("left", "right")
        for instruction in ("normal", "invert")
    )

    residuals = [
        r["members"][cue][instruction]["residual_max"]
        for r in rows
        for cue in ("left", "right")
        for instruction in ("normal", "invert")
    ]
    result["executive_residual_mean"] = mean(residuals)
    result["executive_residual_max"] = max(residuals)

    # Margin reversal strength for same memory under NORMAL vs INVERT.
    reversal = []
    for r in rows:
        for cue in ("left", "right"):
            n = r["members"][cue]["normal"]["normal"]
            i = r["members"][cue]["invert"]["normal"]
            reversal.append(
                abs(
                    n["right_minus_left"]
                    - i["right_minus_left"]
                )
            )
    result["instruction_logit_separation_mean"] = mean(reversal)
    result["instruction_logit_separation_ci95"] = bootstrap_ci(
        reversal
    )

    return result


def factor_summary(rows, summary):
    out = {}
    for field in (
        "corridor_length",
        "goal_x_offset",
        "goal_y_offset",
        "cue_steps",
        "move_step",
    ):
        out[field] = {}
        for value in sorted({
            r["factors"][field] for r in rows
        }):
            group = [
                r for r in rows
                if r["factors"][field] == value
            ]
            s = summarize(group)
            out[field][str(value)] = {
                "n": len(group),
                "normal_pair_full":
                    s["normal"]["normal"]["pair_full"],
                "invert_pair_full":
                    s["normal"]["invert"]["pair_full"],
                "normal_pair_side":
                    s["normal"]["normal"]["pair_side"],
                "invert_pair_side":
                    s["normal"]["invert"]["pair_side"],
                "four_way_side":
                    s["normal"]["four_way_side"],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, payload = load_model(
        Path(args.checkpoint),
        device,
    )

    admitted_ref = clone_state(model.admitted)
    base_ref = clone_state(model.base)

    print("EXECUTIVE_COMPOSITION_FINAL_TRAIN_CAUSAL_V1")
    print("device:", device)
    print("checkpoint:", args.checkpoint)
    print("checkpoint_sha256:", sha256_file(args.checkpoint))
    print("checkpoint_update:", payload["update"])
    print("admitted_checkpoint_sha256:", EXPECTED_ADMITTED_SHA256)
    print("condition_range: 0:1600")
    print("old_dev_touched: false")
    print("old_test_touched: false")

    rows = []
    total = len(TRAIN_CONDITION_RANGE)

    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.append(
            eval_condition(
                model,
                cfg,
                device,
                int(cid),
            )
        )
        if i == 1 or i % 100 == 0 or i == total:
            s = summarize(rows)
            print(
                f"evaluated={i}/{total} "
                f"Nfull={s['normal']['normal']['pair_full']:.4f} "
                f"Ifull={s['normal']['invert']['pair_full']:.4f} "
                f"Nside={s['normal']['normal']['pair_side']:.4f} "
                f"Iside={s['normal']['invert']['pair_side']:.4f} "
                f"fourSide={s['normal']['four_way_side']:.4f}"
            )

    # Protected-state checks.
    for k, expected in admitted_ref.items():
        got = model.admitted.state_dict()[k].detach().cpu()
        if not torch.equal(expected, got):
            raise AssertionError(
                f"admitted brain changed during evaluation: {k}"
            )
    for k, expected in base_ref.items():
        got = model.base.state_dict()[k].detach().cpu()
        if not torch.equal(expected, got):
            raise AssertionError(
                f"historical base changed during evaluation: {k}"
            )

    s = summarize(rows)
    factors = factor_summary(rows, s)

    result = {
        "format": "executive_composition_final_train_causal_v1",
        "split": "train_only",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": int(payload["update"]),
        "summary": s,
        "factor_summary": factors,
        "per_condition": rows,
        "old_dev_touched": False,
        "old_test_touched": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "normal:",
        f"pairF={s['normal']['normal']['pair_full']:.4f}",
        f"pairS={s['normal']['normal']['pair_side']:.4f}",
    )
    print(
        "invert:",
        f"pairF={s['normal']['invert']['pair_full']:.4f}",
        f"pairS={s['normal']['invert']['pair_side']:.4f}",
    )
    print(
        "four_way:",
        f"full={s['normal']['four_way_full']:.4f}",
        f"side={s['normal']['four_way_side']:.4f}",
    )
    print(
        "executive_off:",
        f"Nside={s['executive_off']['normal']['pair_side']:.4f}",
        f"Iside={s['executive_off']['invert']['pair_side']:.4f}",
    )
    print(
        "memory_zero:",
        f"Nside={s['memory_zero']['normal']['pair_side']:.4f}",
        f"Iside={s['memory_zero']['invert']['pair_side']:.4f}",
    )
    print(
        "memory_swap_donor:",
        f"Nside={s['memory_swapped_donor_target']['normal']['pair_side']:.4f}",
        f"Iside={s['memory_swapped_donor_target']['invert']['pair_side']:.4f}",
    )
    print(
        "instruction_flip_consistency:",
        f"{s['instruction_flip_consistency']:.4f}",
    )
    print(
        "instruction_logit_separation_mean:",
        f"{s['instruction_logit_separation_mean']:.6f}",
    )
    print(
        "none_exact_noop_max_diff:",
        f"{s['none_exact_noop_max_diff']:.9g}",
    )
    print(
        "executive_residual:",
        f"mean={s['executive_residual_mean']:.6f}",
        f"max={s['executive_residual_max']:.6f}",
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("OLD_DEV_TOUCHED: false")
    print("OLD_TEST_TOUCHED: false")
    print("EXECUTIVE_COMPOSITION_FINAL_TRAIN_CAUSAL_COMPLETE")


if __name__ == "__main__":
    main()
