#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    VariedCuedDetourWorld,
    LEFT,
    RIGHT,
)
from continuous_brain.planning_v103 import _option_policy_action

BASE_CHECKPOINT = Path("state/child_v113_dml_smoke/infant_training.pt")
MEMORY_CHECKPOINT = Path(
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
)
EXPECTED_BASE_SHA = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_MEMORY_SHA = "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"

MEMORY_DIM = 64
NUCLEUS_DIM = 256
RECALL_HIDDEN = 256
RECALL_DIM = 128


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if torch.is_tensor(value):
                out[path] = value.detach().cpu()
            elif isinstance(value, dict):
                out.update(_flatten_tensors(value, path))
    return out


def _find_tensor(flat, shape, hints, suffix=None):
    rows = []
    for key, value in flat.items():
        low = key.lower()
        if tuple(value.shape) != tuple(shape):
            continue
        if suffix is not None and not key.endswith(suffix):
            continue
        if hints and not any(h in low for h in hints):
            continue
        if "optimizer" in low or ".state." in low:
            continue
        rows.append((key, value))
    if len(rows) != 1:
        raise RuntimeError(
            f"tensor lookup failed shape={shape} hints={hints} suffix={suffix}; "
            f"candidates={[k for k, _ in rows]}"
        )
    return rows[0][1].float()


@dataclass
class FrozenAdmittedMemory:
    candidate_w: torch.Tensor
    candidate_b: torch.Tensor
    gate_w: torch.Tensor
    gate_b: torch.Tensor
    recall_feature_w: torch.Tensor
    recall_output_w: torch.Tensor
    recall_scale: torch.Tensor

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        if sha256(path) != EXPECTED_MEMORY_SHA:
            raise RuntimeError("admitted memory checkpoint SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        flat = _flatten_tensors(payload)

        def get(shape, hints, suffix=None):
            return _find_tensor(flat, shape, hints, suffix=suffix)

        return cls(
            candidate_w=get((64, 256), ("candidate",), "candidate.weight"),
            candidate_b=get((64,), ("candidate",), "candidate.bias"),
            gate_w=get((64, 320), ("gate",), "write_gate.weight"),
            gate_b=get((64,), ("gate",), "write_gate.bias"),
            recall_feature_w=get((256, 64), ("recall", "feature")),
            recall_output_w=get((128, 256), ("recall", "output")),
            recall_scale=get((128,), ("scale", "recall")),
        )

    def to(self, device):
        return FrozenAdmittedMemory(
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
            }
        )

    def write(self, hidden, memory):
        candidate = torch.tanh(
            F.linear(hidden, self.candidate_w, self.candidate_b)
        )
        gate = torch.sigmoid(
            F.linear(
                torch.cat((hidden, memory), dim=-1),
                self.gate_w,
                self.gate_b,
            )
        )
        return memory + gate * (candidate - memory)

    def recall(self, memory):
        feature = torch.tanh(
            F.linear(memory, self.recall_feature_w, None)
        )
        out = F.linear(feature, self.recall_output_w, None)
        return out * self.recall_scale.reshape(1, -1)


class GoalDecoder(nn.Module):
    """Explicit remembered-goal register. Zero recall => exact zero logits."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 2, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, recall):
        return self.linear(recall)


@torch.no_grad()
def _brain_memory_step(base, admitted, image, context, memory, prev_action, hidden):
    semantic = base.vision(image)
    spatial = base.spatial(image)
    fused = base.fuse_vision(semantic, spatial)
    streams = base.nucleus.project_streams(
        fused, context, memory, prev_action
    )
    next_hidden = base.nucleus.core(
        torch.cat(tuple(streams), dim=-1), hidden
    )
    next_memory = admitted.write(next_hidden, memory)
    return next_hidden, next_memory


@torch.no_grad()
def collect_decision_snapshot(base, cfg, admitted, condition_id, cue, device="cpu"):
    env = VariedCuedDetourWorld(
        int(condition_id),
        cue=str(cue),
        max_episode_steps=160,
        image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()
    hidden = base.nucleus.initial_state(1, device)
    memory = torch.zeros(1, int(cfg.memory_dim), device=device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    prefix_actions = []

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, context_dyn = _context_tensor(obs, device, cfg)
        memory_in = memory.clone()
        next_hidden, next_memory = _brain_memory_step(
            base, admitted, image, context, memory, prev_action, hidden
        )

        action = int(env.teacher_action())
        if action in (LEFT, RIGHT):
            if env.steps < env.cue_steps:
                raise AssertionError("decision occurred while cue visible")
            stable_spatial = base.target_spatial_bus(image)
            root = base.planning.encode_state(stable_spatial, context_dyn)
            return {
                "condition_id": int(condition_id),
                "cue": str(cue),
                "target_side": 0 if cue == "left" else 1,
                "target_action": int(LEFT if cue == "left" else RIGHT),
                "memory": memory_in.detach().cpu(),
                "recall": admitted.recall(memory_in).detach().cpu(),
                "root": root.detach().cpu(),
                "obs": np.asarray(obs).copy(),
                "image": image.detach().cpu(),
                "context_dyn": context_dyn.detach().cpu(),
                "prefix_actions": list(prefix_actions),
                "env_steps": int(env.steps),
            }

        obs, _, done, _ = env.step(action)
        if done:
            raise RuntimeError("episode ended before delayed decision")
        prefix_actions.append(action)
        prev_action = torch.tensor([action + 1], dtype=torch.long, device=device)
        hidden = next_hidden
        memory = next_memory

    raise RuntimeError("no delayed decision")


def assert_pair_identical(left, right):
    if left["prefix_actions"] != right["prefix_actions"]:
        raise AssertionError("paired prefix actions differ")
    if not np.array_equal(left["obs"], right["obs"]):
        raise AssertionError("paired decision observations differ")
    if not torch.equal(left["image"], right["image"]):
        raise AssertionError("paired decision images differ")
    if not torch.equal(left["context_dyn"], right["context_dyn"]):
        raise AssertionError("paired decision contexts differ")
    if not torch.equal(left["root"], right["root"]):
        diff = float((left["root"] - right["root"]).abs().max())
        raise AssertionError(f"paired planning roots differ: {diff}")


@torch.no_grad()
def option_table(base, root):
    device = next(base.parameters()).device
    root = root.to(device)
    selector = base.options.selector_logits(root)[0]
    rows = []
    for option in range(int(base.cfg.option_count)):
        rows.append(
            {
                "option": option,
                "action": _option_policy_action(base, root, option),
                "selector_logit": float(selector[option].item()),
            }
        )
    return rows


@torch.no_grad()
def choose_goal_option(base, root, goal_side):
    rows = option_table(base, root)
    baseline = max(rows, key=lambda r: r["selector_logit"])
    if goal_side is None:
        return {
            "selected": baseline,
            "baseline": baseline,
            "desired_action": None,
            "goal_candidate_count": 0,
            "goal_intervened": False,
        }
    desired = int(LEFT if int(goal_side) == 0 else RIGHT)
    matching = [r for r in rows if r["action"] == desired]
    selected = max(matching, key=lambda r: r["selector_logit"]) if matching else baseline
    return {
        "selected": selected,
        "baseline": baseline,
        "desired_action": desired,
        "goal_candidate_count": len(matching),
        "goal_intervened": selected["option"] != baseline["option"],
    }


def build_substrate(device="cpu"):
    if sha256(BASE_CHECKPOINT) != EXPECTED_BASE_SHA:
        raise RuntimeError("canonical base checkpoint SHA mismatch")
    base, cfg, _, got_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if got_sha != EXPECTED_BASE_SHA:
        raise RuntimeError("base loader SHA mismatch")
    base = base.to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    admitted = FrozenAdmittedMemory.load(MEMORY_CHECKPOINT).to(device)
    return base, cfg, admitted


def save_goal_decoder(path, decoder, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "continuous_brain.memory_guided_goal_option_v1",
            "base_checkpoint_sha256": EXPECTED_BASE_SHA,
            "memory_checkpoint_sha256": EXPECTED_MEMORY_SHA,
            "goal_decoder": {
                k: v.detach().cpu() for k, v in decoder.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_goal_decoder(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base_checkpoint_sha256"] != EXPECTED_BASE_SHA:
        raise RuntimeError("goal decoder base provenance mismatch")
    if payload["memory_checkpoint_sha256"] != EXPECTED_MEMORY_SHA:
        raise RuntimeError("goal decoder memory provenance mismatch")
    decoder = GoalDecoder().to(device)
    decoder.load_state_dict(payload["goal_decoder"])
    decoder.eval()
    return decoder, payload
