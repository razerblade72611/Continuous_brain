#!/usr/bin/env python3
"""
TRAIN-only same-hidden 2x2 memory-route factorial for final recall checkpoints.

At each delayed decision, hold the recipient's recurrent hidden state and all
sensory/action inputs FIXED, then toggle only:

H = historical current-memory route
D = direct workspace recall route

Modes:
  Z  : H=off, D=off
  H  : H=on,  D=off
  D  : H=off, D=on
  HD : H=on,  D=on (normal)

This is the exact factorial needed to ask whether the intact final computation
looks like simple rescue/selection or whether HD can succeed where both isolated
memory-route conditions fail under the same hidden state.

TRAIN only. DEV/TEST are never touched.
"""
from __future__ import annotations

import argparse
import collections
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
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
)
from continuous_brain.working_memory_recall_arms_v2 import (
    RecallBridgeWorkingMemoryInfant,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_FORMAT = "working_memory_recall_training_v2"


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


def build(path, arm, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("checkpoint format mismatch")
    if payload.get("arm") != arm:
        raise RuntimeError("checkpoint arm mismatch")
    if int(payload.get("update", -1)) != 1600:
        raise RuntimeError("expected update-1600 checkpoint")
    if payload.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("checkpoint base SHA mismatch")
    if payload.get("base_digest") != EXPECTED_BASE_DIGEST:
        raise RuntimeError("checkpoint base digest mismatch")

    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError("loaded base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("loaded base digest mismatch")

    model = RecallBridgeWorkingMemoryInfant.grow_from(
        base, cfg, arm=arm
    )
    model.memory.load_state_dict(payload["memory"], strict=True)
    model.recall.load_state_dict(payload["recall"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def snapshot(model, cfg, device, cid, cue):
    env = VariedCuedDetourWorld(cid, cue=cue, max_episode_steps=160)
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)

    while not env.done:
        teacher = int(env.teacher_action())
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if teacher in (LEFT, RIGHT):
            return {
                "teacher": teacher,
                "decision_step": int(env.steps),
                "delay": int(env.steps - env.cue_steps),
                "image": image.detach().clone(),
                "context": context.detach().clone(),
                "prev": prev.detach().clone(),
                "hidden": hidden.detach().clone(),
                "memory": memory.detach().clone(),
            }

        out = model.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        prev = torch.tensor([teacher + 1], dtype=torch.long, device=device)
        hidden = out["hidden"]
        memory = out["memory_next"]

    raise RuntimeError("decision not found")


@torch.no_grad()
def forward(model, snap, hist_memory, direct_memory, enable_direct):
    historical = model.historical_step(
        snap["image"],
        snap["context"],
        hist_memory,
        snap["prev"],
        snap["hidden"],
    )

    if enable_direct:
        residual, _ = model.recall(
            historical["hidden"],
            direct_memory,
        )
    else:
        residual = torch.zeros_like(historical["historical_workspace"])

    workspace = historical["historical_workspace"] + residual
    logits = model.base.motor(workspace)[0].detach().float().cpu()
    return logits


def classify(logits, teacher):
    full_pred = int(torch.argmax(logits).item())
    d = float(logits[RIGHT].item() - logits[LEFT].item())
    side_pred = RIGHT if d > 0.0 else LEFT
    return {
        "full_correct": bool(full_pred == teacher),
        "lateral_top": bool(full_pred in (LEFT, RIGHT)),
        "side_correct": bool(side_pred == teacher),
        "correct_side_margin": float(d if teacher == RIGHT else -d),
    }


def evaluate_condition(model, cfg, device, cid):
    l = snapshot(model, cfg, device, cid, "left")
    r = snapshot(model, cfg, device, cid, "right")

    for field in ("image", "context", "prev"):
        if not torch.equal(l[field].cpu(), r[field].cpu()):
            raise AssertionError(f"counterfactual {field} mismatch")
    if l["decision_step"] != r["decision_step"]:
        raise AssertionError("decision timing mismatch")

    zero_l = torch.zeros_like(l["memory"])
    zero_r = torch.zeros_like(r["memory"])

    modes = {}
    for name, hist_on, direct_on in (
        ("Z", False, False),
        ("H", True, False),
        ("D", False, True),
        ("HD", True, True),
    ):
        ll = forward(
            model,
            l,
            l["memory"] if hist_on else zero_l,
            l["memory"] if direct_on else zero_l,
            direct_on,
        )
        rr = forward(
            model,
            r,
            r["memory"] if hist_on else zero_r,
            r["memory"] if direct_on else zero_r,
            direct_on,
        )
        modes[name] = {
            "left": classify(ll, LEFT),
            "right": classify(rr, RIGHT),
        }

    return {
        "condition_id": int(cid),
        "delay": int(l["delay"]),
        "modes": modes,
    }


def pair(row, mode, metric):
    l = row["modes"][mode]["left"]
    r = row["modes"][mode]["right"]
    if metric == "full":
        return bool(l["full_correct"] and r["full_correct"])
    if metric == "side":
        return bool(l["side_correct"] and r["side_correct"])
    if metric == "lateral":
        return bool(l["lateral_top"] and r["lateral_top"])
    raise ValueError(metric)


def margin(row, mode):
    return 0.5 * (
        row["modes"][mode]["left"]["correct_side_margin"]
        + row["modes"][mode]["right"]["correct_side_margin"]
    )


def summarize(rows, metric):
    cats = {
        "H_and_D_correct": [],
        "H_only_correct": [],
        "D_only_correct": [],
        "neither_H_nor_D_correct": [],
    }
    interaction = []

    for row in rows:
        h = pair(row, "H", metric)
        d = pair(row, "D", metric)
        hd = pair(row, "HD", metric)
        if h and d:
            key = "H_and_D_correct"
        elif h:
            key = "H_only_correct"
        elif d:
            key = "D_only_correct"
        else:
            key = "neither_H_nor_D_correct"
        cats[key].append(hd)

        interaction.append(
            margin(row, "HD")
            - margin(row, "H")
            - margin(row, "D")
            + margin(row, "Z")
        )

    return {
        "metric": metric,
        "categories": {
            k: {
                "n": len(v),
                "HD_success": sum(v),
                "HD_success_fraction": (
                    sum(v) / len(v) if v else None
                ),
            }
            for k, v in cats.items()
        },
        "joint_only_success": sum(
            pair(r, "HD", metric)
            and not pair(r, "H", metric)
            and not pair(r, "D", metric)
            for r in rows
        ),
        "interference": sum(
            not pair(r, "HD", metric)
            and (pair(r, "H", metric) or pair(r, "D", metric))
            for r in rows
        ),
        "mean_margin_interaction":
            sum(interaction) / len(interaction),
        "positive_margin_interaction_fraction":
            sum(x > 0.0 for x in interaction) / len(interaction),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("linear", "gated"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg = build(Path(args.checkpoint), args.arm, device)

    print("WORKING_MEMORY_RECALL_SAME_HIDDEN_FACTORIAL_V1")
    print("arm:", args.arm)
    print("device:", device)
    print("dev_touched: false")
    print("test_touched: false")

    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.append(evaluate_condition(model, cfg, device, int(cid)))
        if i == 1 or i % 200 == 0 or i == len(TRAIN_CONDITION_RANGE):
            print(f"evaluated={i}/{len(TRAIN_CONDITION_RANGE)}")

    result = {
        "format": "working_memory_recall_same_hidden_factorial_v1",
        "arm": args.arm,
        "split": "train",
        "conditions": len(rows),
        "full": summarize(rows, "full"),
        "side": summarize(rows, "side"),
        "lateral": summarize(rows, "lateral"),
        "dev_touched": False,
        "test_touched": False,
        "per_condition": rows,
    }

    by_delay = collections.defaultdict(list)
    for row in rows:
        by_delay[row["delay"]].append(row)
    result["by_delay"] = {
        str(k): {
            "full": summarize(v, "full"),
            "side": summarize(v, "side"),
        }
        for k, v in sorted(by_delay.items())
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    for metric in ("full", "side"):
        s = result[metric]
        print()
        print(metric.upper())
        for name, c in s["categories"].items():
            frac = (
                "nan"
                if c["HD_success_fraction"] is None
                else f"{c['HD_success_fraction']:.4f}"
            )
            print(
                f"  {name}: n={c['n']} "
                f"HD_success={c['HD_success']} fraction={frac}"
            )
        print("  joint_only_success:", s["joint_only_success"])
        print("  interference:", s["interference"])
        print(
            "  mean_margin_interaction:",
            f"{s['mean_margin_interaction']:+.6f}",
        )
        print(
            "  positive_margin_interaction_fraction:",
            f"{s['positive_margin_interaction_fraction']:.4f}",
        )

    print("historical_base_bit_exact: true")
    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("WORKING_MEMORY_RECALL_SAME_HIDDEN_FACTORIAL_COMPLETE")


if __name__ == "__main__":
    main()
