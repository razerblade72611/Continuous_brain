#!/usr/bin/env python3
"""
Held-out DEV evaluator and causal intervention recorder for Working Memory M1.

This script uses ONLY frozen DEV condition IDs 1600..1979.
It does not make the admission decision itself; it writes one row per genuinely
independent structural condition for a separate threshold/decision script.

Primary decision point:
  teacher-force the cue-independent path to the first delayed LEFT/RIGHT action.

Interventions at that matched decision state:
  normal                   own hidden + own memory
  zero_at_decision         own hidden + zero memory
  swapped_memory           own hidden + counterfactual partner memory
  hidden_reset_own         zero hidden + own memory
  hidden_reset_swapped     zero hidden + counterfactual partner memory
  both_reset               zero hidden + zero memory

Additional trajectory controls:
  memory_disabled_all      no working memory from step 1
  zero_after_hidden        memory forced zero from cue disappearance onward

No TEST condition is accessible through this evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import torch
import torch_directml

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    DEV_CONDITION_RANGE,
    LEFT,
    RIGHT,
    VariedCuedDetourWorld,
    generate_task_spec,
)
from continuous_brain.working_memory_infant import WorkingMemoryInfant


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_FORMAT = "working_memory_dml_training_v1"
EXPECTED_UPDATE = 1600


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(module) -> str:
    h = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build_model(checkpoint: Path, device):
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("format") != EXPECTED_FORMAT:
        raise RuntimeError(
            f"checkpoint format mismatch: {payload.get('format')!r}"
        )
    if int(payload.get("update", -1)) != EXPECTED_UPDATE:
        raise RuntimeError(
            f"expected update {EXPECTED_UPDATE}, got {payload.get('update')}"
        )
    if payload.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("checkpoint base SHA mismatch")
    if payload.get("base_digest") != EXPECTED_BASE_DIGEST:
        raise RuntimeError("checkpoint base digest metadata mismatch")
    if payload.get("train_condition_range") != [0, 1600]:
        raise RuntimeError("checkpoint train split metadata mismatch")

    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("loaded historical base SHA mismatch")

    digest = state_digest(base)
    if digest != EXPECTED_BASE_DIGEST:
        raise RuntimeError(
            f"loaded historical base digest mismatch: {digest}"
        )

    model = WorkingMemoryInfant.grow_from(base, cfg)
    model.memory.load_state_dict(payload["memory"], strict=True)
    model = model.to(device)
    model.eval()
    model.base.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, payload, digest


def cpu_equal(a, b) -> bool:
    return torch.equal(a.detach().cpu(), b.detach().cpu())


@torch.no_grad()
def snapshot_to_decision(
    model,
    cfg,
    device,
    *,
    condition_id: int,
    cue: str,
    mode: str,
):
    if mode not in ("normal", "memory_disabled_all", "zero_after_hidden"):
        raise ValueError(f"unknown rollout mode {mode}")

    env = VariedCuedDetourWorld(
        condition_id,
        cue=cue,
        max_episode_steps=160,
    )
    obs = env.reset()

    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev_action = torch.zeros(
        1, dtype=torch.long, device=device
    )

    predecision_steps = 0
    while not env.done:
        teacher = int(env.teacher_action())

        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if teacher in (LEFT, RIGHT):
            if env.steps < env.cue_steps:
                raise AssertionError(
                    "decision occurred while cue still visible"
                )
            if tuple(memory.shape) != (1, int(cfg.memory_dim)):
                raise AssertionError("memory shape changed")
            return {
                "condition_id": int(condition_id),
                "cue": cue,
                "teacher_action": teacher,
                "decision_step": int(env.steps),
                "predecision_steps": int(predecision_steps),
                "image": image.detach().clone(),
                "context": context.detach().clone(),
                "prev_action": prev_action.detach().clone(),
                "hidden": hidden.detach().clone(),
                "memory": memory.detach().clone(),
                "spec": env.spec,
            }

        if teacher not in (1,):
            raise AssertionError(
                f"unexpected predecision teacher action {teacher}"
            )

        if mode == "zero_after_hidden" and env.steps >= env.cue_steps:
            memory = torch.zeros_like(memory)

        out = model.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=(mode != "memory_disabled_all"),
        )

        next_memory = out["memory_next"]
        if mode in ("memory_disabled_all",):
            next_memory = torch.zeros_like(next_memory)
        elif mode == "zero_after_hidden" and env.steps >= env.cue_steps:
            next_memory = torch.zeros_like(next_memory)

        obs, _reward, _done, _info = env.step(teacher)
        prev_action = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )
        hidden = out["hidden"]
        memory = next_memory
        predecision_steps += 1

        if predecision_steps > 160:
            raise RuntimeError("predecision rollout exceeded safety bound")

    raise RuntimeError("episode ended before delayed decision")


@torch.no_grad()
def decision_logits(model, snap, *, hidden, memory):
    out = model.base_step(
        snap["image"],
        snap["context"],
        memory,
        snap["prev_action"],
        hidden,
    )
    return out["logits"][0].detach().float().cpu()


def classify(logits, teacher_action):
    full_pred = int(torch.argmax(logits).item())
    left_logit = float(logits[LEFT].item())
    right_logit = float(logits[RIGHT].item())
    d = right_logit - left_logit
    side_pred = RIGHT if d > 0.0 else LEFT
    correct_side_margin = d if teacher_action == RIGHT else -d

    other = [
        float(logits[i].item())
        for i in range(len(ACTION_NAMES))
        if i != teacher_action
    ]
    correct_vs_all_margin = (
        float(logits[teacher_action].item()) - max(other)
    )

    return {
        "full_pred": full_pred,
        "full_pred_name": ACTION_NAMES[full_pred],
        "full_correct": bool(full_pred == teacher_action),
        "side_pred": int(side_pred),
        "side_pred_name": ACTION_NAMES[side_pred],
        "side_correct": bool(side_pred == teacher_action),
        "left_logit": left_logit,
        "right_logit": right_logit,
        "d_right_minus_left": float(d),
        "correct_side_margin": float(correct_side_margin),
        "correct_vs_all_margin": float(correct_vs_all_margin),
        "all_logits": [float(x) for x in logits.tolist()],
    }


def pair_inputs_identical(left, right):
    checks = {
        "image": cpu_equal(left["image"], right["image"]),
        "context": cpu_equal(left["context"], right["context"]),
        "prev_action": cpu_equal(left["prev_action"], right["prev_action"]),
        "decision_step": left["decision_step"] == right["decision_step"],
        "predecision_steps":
            left["predecision_steps"] == right["predecision_steps"],
    }
    if not all(checks.values()):
        raise AssertionError(
            f"counterfactual decision inputs not matched: {checks}"
        )
    return checks


def eval_pair(model, cfg, device, condition_id):
    normal_l = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="left",
        mode="normal",
    )
    normal_r = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="right",
        mode="normal",
    )
    input_checks = pair_inputs_identical(normal_l, normal_r)

    disabled_l = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="left",
        mode="memory_disabled_all",
    )
    disabled_r = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="right",
        mode="memory_disabled_all",
    )

    zero_hidden_l = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="left",
        mode="zero_after_hidden",
    )
    zero_hidden_r = snapshot_to_decision(
        model, cfg, device,
        condition_id=condition_id,
        cue="right",
        mode="zero_after_hidden",
    )

    zero_mem = torch.zeros_like(normal_l["memory"])
    zero_hidden = model.initial_state(1, device)

    modes = {}

    def add_mode(name, h_l, m_l, h_r, m_r, left_target=LEFT, right_target=RIGHT):
        l_logits = decision_logits(
            model, normal_l, hidden=h_l, memory=m_l
        )
        r_logits = decision_logits(
            model, normal_r, hidden=h_r, memory=m_r
        )
        modes[name] = {
            "left": classify(l_logits, left_target),
            "right": classify(r_logits, right_target),
            "pair_logits_max_abs_diff": float(
                (l_logits - r_logits).abs().max().item()
            ),
        }

    add_mode(
        "normal",
        normal_l["hidden"], normal_l["memory"],
        normal_r["hidden"], normal_r["memory"],
    )
    add_mode(
        "zero_at_decision",
        normal_l["hidden"], zero_mem,
        normal_r["hidden"], zero_mem,
    )
    add_mode(
        "swapped_memory",
        normal_l["hidden"], normal_r["memory"],
        normal_r["hidden"], normal_l["memory"],
    )
    add_mode(
        "hidden_reset_own",
        zero_hidden, normal_l["memory"],
        zero_hidden, normal_r["memory"],
    )
    add_mode(
        "hidden_reset_swapped",
        zero_hidden, normal_r["memory"],
        zero_hidden, normal_l["memory"],
    )
    add_mode(
        "both_reset",
        zero_hidden, zero_mem,
        zero_hidden, zero_mem,
    )

    # Trajectory-level baselines use their own hidden snapshots because memory
    # was disabled/zeroed during the hidden interval, not just at the decision.
    l_logits = decision_logits(
        model, disabled_l,
        hidden=disabled_l["hidden"],
        memory=torch.zeros_like(disabled_l["memory"]),
    )
    r_logits = decision_logits(
        model, disabled_r,
        hidden=disabled_r["hidden"],
        memory=torch.zeros_like(disabled_r["memory"]),
    )
    modes["memory_disabled_all"] = {
        "left": classify(l_logits, LEFT),
        "right": classify(r_logits, RIGHT),
        "pair_logits_max_abs_diff": float(
            (l_logits - r_logits).abs().max().item()
        ),
    }

    l_logits = decision_logits(
        model, zero_hidden_l,
        hidden=zero_hidden_l["hidden"],
        memory=torch.zeros_like(zero_hidden_l["memory"]),
    )
    r_logits = decision_logits(
        model, zero_hidden_r,
        hidden=zero_hidden_r["hidden"],
        memory=torch.zeros_like(zero_hidden_r["memory"]),
    )
    modes["zero_after_hidden"] = {
        "left": classify(l_logits, LEFT),
        "right": classify(r_logits, RIGHT),
        "pair_logits_max_abs_diff": float(
            (l_logits - r_logits).abs().max().item()
        ),
    }

    normal_d_l = modes["normal"]["left"]["d_right_minus_left"]
    normal_d_r = modes["normal"]["right"]["d_right_minus_left"]
    swap_d_l = modes["swapped_memory"]["left"]["d_right_minus_left"]
    swap_d_r = modes["swapped_memory"]["right"]["d_right_minus_left"]

    left_donor_shift = swap_d_l - normal_d_l
    right_donor_shift = normal_d_r - swap_d_r
    pair_swap_effect = 0.5 * (
        left_donor_shift + right_donor_shift
    )

    hidden_own_d_l = modes["hidden_reset_own"]["left"]["d_right_minus_left"]
    hidden_own_d_r = modes["hidden_reset_own"]["right"]["d_right_minus_left"]
    hidden_swap_d_l = modes["hidden_reset_swapped"]["left"]["d_right_minus_left"]
    hidden_swap_d_r = modes["hidden_reset_swapped"]["right"]["d_right_minus_left"]

    both_reset_diff = modes["both_reset"]["pair_logits_max_abs_diff"]
    if both_reset_diff > 1e-6:
        raise AssertionError(
            f"both-reset counterfactual logits diverged: {both_reset_diff}"
        )

    spec = generate_task_spec(condition_id)

    return {
        "condition_id": int(condition_id),
        "spec": spec.to_dict(),
        "input_checks": input_checks,
        "decision_step": int(normal_l["decision_step"]),
        "memory_l2_left": float(
            torch.linalg.vector_norm(
                normal_l["memory"].detach().float().cpu()
            ).item()
        ),
        "memory_l2_right": float(
            torch.linalg.vector_norm(
                normal_r["memory"].detach().float().cpu()
            ).item()
        ),
        "memory_pair_l2_distance": float(
            torch.linalg.vector_norm(
                (
                    normal_l["memory"] - normal_r["memory"]
                ).detach().float().cpu()
            ).item()
        ),
        "modes": modes,
        "swap": {
            "left_recipient_right_memory_shift": float(left_donor_shift),
            "right_recipient_left_memory_shift": float(right_donor_shift),
            "condition_effect": float(pair_swap_effect),
            "positive": bool(pair_swap_effect > 0.0),
        },
        "hidden_reset_memory_code": {
            "own_pair_side_correct": bool(
                hidden_own_d_l < 0.0 and hidden_own_d_r > 0.0
            ),
            "swapped_pair_donor_consistent": bool(
                hidden_swap_d_l > 0.0 and hidden_swap_d_r < 0.0
            ),
        },
    }


def summarize_mode(rows, mode):
    n = len(rows)
    left_full = sum(
        r["modes"][mode]["left"]["full_correct"] for r in rows
    )
    right_full = sum(
        r["modes"][mode]["right"]["full_correct"] for r in rows
    )
    pair_full = sum(
        r["modes"][mode]["left"]["full_correct"]
        and r["modes"][mode]["right"]["full_correct"]
        for r in rows
    )
    left_side = sum(
        r["modes"][mode]["left"]["side_correct"] for r in rows
    )
    right_side = sum(
        r["modes"][mode]["right"]["side_correct"] for r in rows
    )
    pair_side = sum(
        r["modes"][mode]["left"]["side_correct"]
        and r["modes"][mode]["right"]["side_correct"]
        for r in rows
    )
    return {
        "n": n,
        "left_full_accuracy": left_full / n,
        "right_full_accuracy": right_full / n,
        "pair_full_accuracy": pair_full / n,
        "left_side_accuracy": left_side / n,
        "right_side_accuracy": right_side / n,
        "pair_side_accuracy": pair_side / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="state/working_memory_m1/train_epoch1_v1/latest.portable.pt",
    )
    ap.add_argument(
        "--output",
        default="state/working_memory_m1/dev_eval_v1.json",
    )
    args = ap.parse_args()

    warnings.simplefilter("always")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch_directml.device()
    model, cfg, payload, base_digest = build_model(
        checkpoint, device
    )

    print("WORKING_MEMORY_DEV_CAUSAL_EVAL_V1")
    print("device:", device)
    print("checkpoint:", checkpoint)
    print("checkpoint_sha256:", sha256_file(checkpoint))
    print("checkpoint_update:", payload["update"])
    print("base_checkpoint_sha256:", payload["base_checkpoint_sha256"])
    print("base_digest:", base_digest)
    print(
        "dev_condition_range:",
        f"{DEV_CONDITION_RANGE.start}:{DEV_CONDITION_RANGE.stop}",
    )
    print("test_conditions_touched: false")

    rows = []
    for i, condition_id in enumerate(DEV_CONDITION_RANGE, 1):
        rows.append(
            eval_pair(model, cfg, device, int(condition_id))
        )
        if i == 1 or i % 20 == 0 or i == len(DEV_CONDITION_RANGE):
            normal = summarize_mode(rows, "normal")
            print(
                f"evaluated={i}/{len(DEV_CONDITION_RANGE)} "
                f"normal_pair_full={normal['pair_full_accuracy']:.4f} "
                f"Lfull={normal['left_full_accuracy']:.4f} "
                f"Rfull={normal['right_full_accuracy']:.4f}"
            )

    summaries = {
        mode: summarize_mode(rows, mode)
        for mode in (
            "normal",
            "memory_disabled_all",
            "zero_after_hidden",
            "zero_at_decision",
            "swapped_memory",
            "hidden_reset_own",
            "hidden_reset_swapped",
            "both_reset",
        )
    }

    result = {
        "format": "working_memory_dev_causal_eval_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_update": int(payload["update"]),
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "base_digest": base_digest,
        "split": "dev",
        "condition_range": [
            DEV_CONDITION_RANGE.start,
            DEV_CONDITION_RANGE.stop,
        ],
        "conditions": len(rows),
        "test_conditions_touched": False,
        "summaries": summaries,
        "per_condition": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    for mode, s in summaries.items():
        print(
            f"{mode:22s} "
            f"pair_full={s['pair_full_accuracy']:.4f} "
            f"Lfull={s['left_full_accuracy']:.4f} "
            f"Rfull={s['right_full_accuracy']:.4f} "
            f"pair_side={s['pair_side_accuracy']:.4f}"
        )

    print("wrote:", out)
    print("WORKING_MEMORY_DEV_CAUSAL_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
