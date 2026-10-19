"""Parameterized options and generic precondition/effect composition.

Initial skill definitions are hand-authored. Selection/outcomes are learned;
this release does NOT claim autonomous skill discovery or open-ended language.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math

import numpy as np

GOALS = ("explore", "inspect", "reach", "rest")


@dataclass(frozen=True)
class Skill:
    name: str
    requires: tuple[str, ...]
    effect: str
    max_steps: int
    controller: str
    failures: tuple[str, ...]


SKILLS = (
    Skill("observe", (), "observed", 1, "idle", ("sensor_fault",)),
    Skill("orient", ("visible",), "aligned", 4, "motor", ("timeout", "lost_target")),
    Skill("navigate", ("aligned",), "near", 8, "motor", ("blocked", "timeout")),
    Skill("inspect", ("near", "interaction_available"), "inspected", 1, "interact", ("out_of_range",)),
    Skill("search", (), "visible", 4, "sensor_search", ("no_target", "blocked")),
    Skill("rest", (), "rested", 2, "idle", ()),
)


def sensed_facts(obs, interaction=True):
    obs = np.asarray(obs)
    distance = float(np.linalg.norm(obs[:2]) * 5)
    visible = distance > 1e-6
    angle = abs(math.degrees(math.atan2(float(obs[1]), float(obs[0])))) if visible else 180.0
    result = {"interaction_available"} if interaction else set()
    if visible:
        result.add("visible")
        if angle <= 15:
            result.add("aligned")
        if distance <= 0.55:
            result.add("near")
    return result


def available(facts):
    return [set(skill.requires).issubset(facts) for skill in SKILLS]


def compose(effect, facts, visiting=()):
    """Backward-chain arbitrary registered effects; fail explicitly on cycles."""
    if effect in facts:
        return []
    if effect in visiting:
        raise ValueError(f"Skill precondition cycle: {effect}")
    candidates = [(i, s) for i, s in enumerate(SKILLS) if s.effect == effect]
    if not candidates:
        raise ValueError(f"No skill produces {effect}")
    idx, skill = candidates[0]
    result = []
    accumulated = set(facts)
    for prerequisite in skill.requires:
        chain = compose(prerequisite, accumulated, (*visiting, effect))
        result.extend(chain)
        accumulated.update(SKILLS[i].effect for i in chain)
    result.append(idx)
    return result


@dataclass
class Intention:
    goal: str = "inspect"
    target: str = "sensed_target"
    started: int = 0
    attempts: int = 0
    budget: int = 100
    source: str = "bounded_goal_scheduler"
    history: list = field(default_factory=list)

    def plan(self, facts):
        effect = {"explore": "visible", "inspect": "inspected", "reach": "near", "rest": "rested"}[self.goal]
        plan = compose(effect, facts)
        return plan or [0]

    def state_dict(self):
        return asdict(self)


@dataclass
class SearchController:
    """Authored scan/relocate option. No coordinates, target oracle or map access.

    Counters span four-tick option calls and restarts. A full scan without a
    sighting leads to bounded forward travel, using clearance and action feedback.
    This is a development control policy, not a learned navigation achievement.
    """
    scan_remaining: int = 24
    travel_remaining: int = 12
    avoid_remaining: int = 0
    turn_direction: int = 3
    last_action: int = 0
    successful_moves: int = 0

    def action(self, obs, depth_healthy=True):
        obs = np.asarray(obs)
        if obs.shape != (6,) or not np.isfinite(obs).all():
            return 0
        if self.last_action == 1 and obs[4] > 0.5:
            self.travel_remaining -= 1
            self.successful_moves += 1
        self.last_action = 0
        if "visible" in sensed_facts(obs):
            self.scan_remaining, self.travel_remaining, self.avoid_remaining = 24, 12, 0
            return 0
        if self.travel_remaining <= 0:
            self.scan_remaining, self.travel_remaining = 24, 12
        if self.scan_remaining > 0:
            self.scan_remaining -= 1
            action = self.turn_direction
        elif not depth_healthy:
            action = self.turn_direction  # unknown depth is not evidence of clear space
        elif self.avoid_remaining:
            self.avoid_remaining -= 1
            action = self.turn_direction
        elif obs[2] <= 0.18 or obs[4] < 0.5:
            if abs(float(obs[3])) > 0.05:
                self.turn_direction = 2 if obs[3] > 0 else 3
            self.avoid_remaining = 5  # six acknowledged/requested turns, then re-check
            action = self.turn_direction
        else:
            action = 1
        self.last_action = action
        return action

    def state_dict(self):
        return asdict(self)


def primitive_action(skill_id, obs, motor, rng, search=None, depth_healthy=True):
    skill = SKILLS[skill_id]
    if skill.controller == "idle":
        return 0
    if skill.controller == "interact":
        return 4
    if skill.controller == "sensor_search":
        if search is None:
            raise ValueError("Search requires persistent controller state")
        return search.action(obs, depth_healthy)
    return motor(obs, orient_only=skill.name == "orient")


def reference_motor(obs, orient_only=False):
    """Development-only analytical teacher. Never mislabeled as neural skill."""
    angle = math.degrees(math.atan2(float(obs[1]), float(obs[0])))
    if angle > 15:
        return 3
    if angle < -15:
        return 2
    return 0 if orient_only else 1
