"""Small learned, skill-conditioned development servo. No teacher at inference."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn


MOTOR_CONTRACT = {
    "kind": "development_servo_v1", "environment": "DevelopmentRoom",
    "observation_dim": 6, "features": "unit_goal_direction_distance_sensor_feedback_and_skill_v1",
    "skills": ["orient", "navigate"], "actions": ["idle", "advance", "turn_left", "turn_right"],
    "step_metres": 0.2, "turn_degrees": 15.0, "goal_sensor": "visible_beacon_relative_div5",
    "vision": "not_used_by_motor", "width": 64,
}


class DevelopmentMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64),
                                    nn.ReLU(), nn.Linear(64, 4))

    def forward(self, obs, orient):
        # Geometric sensor normalization, not an action rule or target oracle.
        distance = obs[:, :2].norm(dim=1, keepdim=True)
        direction = obs[:, :2] / distance.clamp_min(1e-6)
        mode = torch.nn.functional.one_hot(orient.long(), 2).to(obs.dtype)
        features = torch.cat((direction, distance.clamp(0, 1), obs[:, 2:], mode), dim=1)
        return self.policy(features)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def action(self, obs, orient_only=False, allowed=(0, 1, 2, 3)):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (6,) or not np.isfinite(obs).all():
            raise ValueError("Motor requires exactly six finite sensor values")
        # Preconditions/action masks are external to learned cognition. There is
        # deliberately no forced-forward heuristic or reference-controller fallback.
        permitted = set(allowed) & ({0, 2, 3} if orient_only else {0, 1, 2, 3})
        if not permitted:
            raise ValueError("No valid motor primitive")
        device = next(self.parameters()).device
        with torch.no_grad():
            logits = self(torch.as_tensor(obs, device=device)[None],
                          torch.tensor([orient_only], device=device))[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite motor output; refusing action")
        return max(sorted(permitted), key=lambda action: float(logits[action]))


def motor_digest(weights):
    digest = hashlib.sha256(json.dumps(MOTOR_CONTRACT, sort_keys=True).encode())
    for key in sorted(weights):
        tensor = weights[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validated_motor(bundle):
    if bundle.get("contract") != MOTOR_CONTRACT:
        raise ValueError("Motor sensor/action/environment contract differs")
    weights = bundle.get("model", {})
    if not weights or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all()
                          for t in weights.values()):
        raise ValueError("Invalid/nonfinite learned motor weights")
    with torch.random.fork_rng(devices=[]):
        model = DevelopmentMotor()
        model.load_state_dict(weights, strict=True)
    if motor_digest(weights) != bundle.get("digest"):
        raise ValueError("Learned motor digest mismatch")
    return model.eval()


def check_qualification(bundle):
    validated_motor(bundle)
    report = bundle.get("qualification", {})
    if (report.get("motor_digest") != bundle["digest"] or report.get("passed") is not True
            or report.get("teacher_actions") != 0 or report.get("training_seed") == report.get("evaluation_seed")
            or report.get("protocol") != "held_out_visible_clear_corridor_v1"):
        raise ValueError("Motor needs a passing, teacher-free qualification bound to these exact weights")
    for skill in ("orient", "navigate"):
        score = report.get("skills", {}).get(skill, {})
        attempts, successes = score.get("episodes", 0), score.get("successes", 0)
        if (not isinstance(attempts, int) or not isinstance(successes, int) or attempts < 100
                or not 0 <= successes <= attempts or successes / attempts < 0.95
                or score.get("faults", 1) != 0):
            raise ValueError("Qualification requires >=100 episodes/skill, >=95% success and zero faults")
    if report["skills"]["navigate"].get("successful_advances", 0) <= 0:
        raise ValueError("Navigation qualification must include actual learned forward movement")
    return bundle
