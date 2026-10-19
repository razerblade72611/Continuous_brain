#!/usr/bin/env python3
"""
Second-generation late-bound executive rule family.

Purpose
-------
Close the shortcut exposed by Executive Composition Production v1:
the admitted brain already solved NORMAL, so the new executive only needed to
be memory-dependent under INVERT.

This family contains NO inherited NORMAL rule.

Every instruction arrives after cue storage and maps the remembered LEFT/RIGHT
cue to a novel response action:

    INVERT_LR:
        LEFT cue  -> RIGHT action
        RIGHT cue -> LEFT action

    VERTICAL:
        LEFT cue  -> UP action
        RIGHT cue -> DOWN action

    VERTICAL_REVERSE:
        LEFT cue  -> DOWN action
        RIGHT cue -> UP action

Thus every rule requires:
    remembered cue content + late instruction.

The response is a one-step executive decision emitted after a varied physical
corridor.  The ordinary visual/context channels never contain the instruction.

Fresh structural family
-----------------------
This is NOT the already-opened first-generation memory family.

All continuous geometry values are newly chosen and a deterministic shuffle of
the valid structural grid is frozen into 2400 unique conditions:

    TRAIN 0..1599
    DEV   1600..1999
    TEST  2000..2399

No condition wraps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
import random

import numpy as np

from .environment import Environment


ACTION_NAMES = ("idle", "up", "down", "left", "right")
IDLE, UP, DOWN, LEFT, RIGHT = range(5)

RULE_NONE = 0
RULE_INVERT_LR = 1
RULE_VERTICAL = 2
RULE_VERTICAL_REVERSE = 3

RULE_NAMES = (
    "none",
    "invert_lr",
    "vertical",
    "vertical_reverse",
)

_RULE_TO_ID = {
    "invert_lr": RULE_INVERT_LR,
    "vertical": RULE_VERTICAL,
    "vertical_reverse": RULE_VERTICAL_REVERSE,
}

# Disjoint continuous geometry values from first-generation task family.
_CORRIDOR_LENGTHS = (
    0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54,
    0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82,
)
_GOAL_X_OFFSETS = (0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68)
_GOAL_Y_OFFSETS = (0.12, 0.16, 0.20, 0.24)
_CUE_STEPS = (1, 2, 3, 5)
_MOVE_STEPS = (0.055, 0.070, 0.085, 0.100)

_START_XS = (-0.15, 0.0, 0.15)
_START_YS = (-0.94, -0.86, -0.78)
_DISTRACTOR_PATTERNS = (0, 1, 2, 3, 4)

MIN_HIDDEN_DELAY_STEPS = 3
FAMILY_SIZE = 2400
TRAIN_CONDITION_RANGE = range(0, 1600)
DEV_CONDITION_RANGE = range(1600, 2000)
TEST_CONDITION_RANGE = range(2000, 2400)


def _steps_to_cover(distance: float, step: float) -> int:
    if distance <= 0:
        return 0
    return int(math.ceil((float(distance) - 1e-9) / float(step)))


def _predecision_steps(corridor_length: float, move_step: float) -> int:
    return _steps_to_cover(corridor_length + move_step, move_step)


def _hidden_delay(corridor_length: float, cue_steps: int, move_step: float) -> int:
    return _predecision_steps(corridor_length, move_step) - int(cue_steps)


_full_grid = [
    x
    for x in itertools.product(
        _CORRIDOR_LENGTHS,
        _GOAL_X_OFFSETS,
        _GOAL_Y_OFFSETS,
        _CUE_STEPS,
        _MOVE_STEPS,
    )
    if _hidden_delay(x[0], x[3], x[4]) >= MIN_HIDDEN_DELAY_STEPS
]

_rng = random.Random(0xE2132026)
_rng.shuffle(_full_grid)
_STRUCTURAL_GRID = tuple(_full_grid[:FAMILY_SIZE])
assert len(_STRUCTURAL_GRID) == FAMILY_SIZE
assert len(set(_STRUCTURAL_GRID)) == FAMILY_SIZE


@dataclass(frozen=True)
class ExecutiveRuleTaskSpec:
    condition_id: int
    start_x: float
    start_y: float
    corridor_length: float
    goal_x_offset: float
    goal_y_offset: float
    cue_steps: int
    move_step: float
    distractor_pattern: int

    @property
    def junction_y(self):
        return float(self.start_y + self.corridor_length)

    @property
    def decision_y(self):
        return float(self.junction_y + self.move_step)

    @property
    def left_goal(self):
        return (
            float(self.start_x - self.goal_x_offset),
            float(self.junction_y + self.goal_y_offset),
        )

    @property
    def right_goal(self):
        return (
            float(self.start_x + self.goal_x_offset),
            float(self.junction_y + self.goal_y_offset),
        )

    def structural_signature(self):
        return (
            round(self.corridor_length, 6),
            round(self.goal_x_offset, 6),
            round(self.goal_y_offset, 6),
            int(self.cue_steps),
            round(self.move_step, 6),
        )

    def behavioral_signature(self):
        pre = _predecision_steps(self.corridor_length, self.move_step)
        return (
            int(pre),
            int(pre - self.cue_steps),
        )

    def to_dict(self):
        d = asdict(self)
        d.update({
            "junction_y": self.junction_y,
            "decision_y": self.decision_y,
            "left_goal": list(self.left_goal),
            "right_goal": list(self.right_goal),
            "structural_signature": list(self.structural_signature()),
            "behavioral_signature": list(self.behavioral_signature()),
        })
        return d


def generate_executive_rule_spec(condition_id: int):
    cid = int(condition_id)
    if cid < 0 or cid >= FAMILY_SIZE:
        raise ValueError(f"condition_id must be 0..{FAMILY_SIZE-1}")

    (
        corridor_length,
        goal_x_offset,
        goal_y_offset,
        cue_steps,
        move_step,
    ) = _STRUCTURAL_GRID[cid]

    start_x = _START_XS[(cid * 7 + 1) % len(_START_XS)]
    start_y = _START_YS[(cid * 11 + 2) % len(_START_YS)]
    distractor = _DISTRACTOR_PATTERNS[(cid * 13 + 4) % len(_DISTRACTOR_PATTERNS)]

    spec = ExecutiveRuleTaskSpec(
        condition_id=cid,
        start_x=float(start_x),
        start_y=float(start_y),
        corridor_length=float(corridor_length),
        goal_x_offset=float(goal_x_offset),
        goal_y_offset=float(goal_y_offset),
        cue_steps=int(cue_steps),
        move_step=float(move_step),
        distractor_pattern=int(distractor),
    )

    if spec.behavioral_signature()[1] < MIN_HIDDEN_DELAY_STEPS:
        raise AssertionError("hidden delay invariant broken")
    if min(spec.left_goal[0], spec.right_goal[0]) < -0.98:
        raise AssertionError("goal escaped left bound")
    if max(spec.left_goal[0], spec.right_goal[0]) > 0.98:
        raise AssertionError("goal escaped right bound")
    if max(spec.left_goal[1], spec.right_goal[1]) > 0.92:
        raise AssertionError("goal escaped top bound")

    return spec


def split_name(condition_id: int):
    cid = int(condition_id)
    if cid in TRAIN_CONDITION_RANGE:
        return "train"
    if cid in DEV_CONDITION_RANGE:
        return "dev"
    if cid in TEST_CONDITION_RANGE:
        return "test"
    raise ValueError(cid)


def target_action(cue: str, rule: str) -> int:
    if rule == "invert_lr":
        return RIGHT if cue == "left" else LEFT
    if rule == "vertical":
        return UP if cue == "left" else DOWN
    if rule == "vertical_reverse":
        return DOWN if cue == "left" else UP
    raise ValueError(rule)


class ExecutiveNovelRuleWorld(Environment):
    observation_dim = 6
    action_dim = 5
    action_names = ACTION_NAMES

    def __init__(
        self,
        condition_id: int,
        *,
        cue: str,
        rule: str,
        instruction_delay: int = 1,
        max_episode_steps: int = 160,
    ):
        self.condition_id = int(condition_id)
        self.spec = generate_executive_rule_spec(self.condition_id)

        if cue not in ("left", "right"):
            raise ValueError("cue must be left/right")
        if rule not in _RULE_TO_ID:
            raise ValueError(f"unknown rule {rule}")
        if int(instruction_delay) < 1:
            raise ValueError("instruction_delay must be >=1")

        self.cue = cue
        self.rule = rule
        self.instruction_delay = int(instruction_delay)
        self.max_episode_steps = int(max_episode_steps)

        self.agent = np.zeros(2, dtype=np.float32)
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        self.done = False
        self.responded = False
        self.reset()

    @property
    def cue_steps(self):
        return int(self.spec.cue_steps)

    @property
    def move_step(self):
        return float(self.spec.move_step)

    @property
    def instruction_start_step(self):
        return int(self.cue_steps + self.instruction_delay)

    def instruction_visible(self):
        return bool(self.steps >= self.instruction_start_step)

    def instruction_id(self):
        if not self.instruction_visible():
            return RULE_NONE
        return _RULE_TO_ID[self.rule]

    def reset(self):
        self.agent[:] = [self.spec.start_x, self.spec.start_y]
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        self.done = False
        self.responded = False
        return self._obs()

    def _cue_active(self):
        return self.steps < self.cue_steps

    def _goal(self):
        xy = self.spec.left_goal if self.cue == "left" else self.spec.right_goal
        return np.asarray(xy, dtype=np.float32)

    def _obs(self):
        if self._cue_active():
            delta = self._goal() - self.agent
        else:
            delta = np.zeros(2, dtype=np.float32)

        return np.asarray([
            float(self.agent[0]),
            float(self.agent[1]),
            float(delta[0]),
            float(delta[1]),
            float(self.energy),
            float(self.last_reward),
        ], dtype=np.float32)

    def at_response_step(self):
        half = 0.5 * self.move_step
        return bool(
            float(self.agent[1]) >= self.spec.decision_y - half
        )

    def teacher_action(self):
        if not self.at_response_step():
            return UP
        if not self.instruction_visible():
            raise RuntimeError("response reached before instruction became visible")
        return target_action(self.cue, self.rule)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() after done")

        action = int(action)
        self.steps += 1

        # Once the response point has been reached, the next action is the
        # symbolic executive response and terminates the assay.
        if self.at_response_step():
            expected = target_action(self.cue, self.rule)
            correct = action == expected
            self.responded = True
            self.done = True
            self.last_reward = 1.0 if correct else -1.0
            return (
                self._obs(),
                self.last_reward,
                True,
                {
                    "response_correct": bool(correct),
                    "target_action": int(expected),
                    "target_action_name": ACTION_NAMES[expected],
                    "rule": self.rule,
                    "cue": self.cue,
                    "condition_id": self.condition_id,
                    "split": split_name(self.condition_id),
                },
            )

        dx = dy = 0.0
        if action == UP:
            dy = self.move_step
        elif action == DOWN:
            dy = -self.move_step
        elif action == LEFT:
            dx = -self.move_step
        elif action == RIGHT:
            dx = self.move_step

        proposed = self.agent + np.asarray([dx, dy], dtype=np.float32)

        if float(self.agent[1]) < self.spec.junction_y - 1e-6:
            proposed[0] = float(self.spec.start_x)

        proposed = np.clip(proposed, -1.0, 1.0)
        self.agent[:] = proposed

        vertical_progress = max(0.0, float(self.agent[1] - self.spec.start_y))
        self.energy = float(np.clip(1.0 - 0.12 * vertical_progress, 0.0, 1.0))
        self.last_reward = -0.004

        if self.steps >= self.max_episode_steps:
            self.done = True

        return (
            self._obs(),
            self.last_reward,
            self.done,
            {
                "response_correct": False,
                "rule": self.rule,
                "cue": self.cue,
                "condition_id": self.condition_id,
                "split": split_name(self.condition_id),
            },
        )

    @staticmethod
    def _draw_disk(img, cx, cy, radius, value):
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (xx-cx)**2 + (yy-cy)**2 <= radius**2
        img[mask] = value

    @staticmethod
    def _draw_rect(img, x0, y0, x1, y1, value):
        xa, xb = sorted((max(0, x0), min(img.shape[1], x1)))
        ya, yb = sorted((max(0, y0), min(img.shape[0], y1)))
        if xa < xb and ya < yb:
            img[ya:yb, xa:xb] = value

    def render_rgb(self, size: int = 96):
        size = max(32, int(size))
        img = np.full((size, size, 3), 18, dtype=np.uint8)

        def pix(pos):
            x = int(round((float(pos[0])+1.0)*0.5*(size-1)))
            y = int(round((1.0-(float(pos[1])+1.0)*0.5)*(size-1)))
            return x, y

        spacing = max(8, size//8)
        for i in range(0, size, spacing):
            img[i:i+1,:,:] = 31
            img[:,i:i+1,:] = 31

        start = np.asarray([self.spec.start_x, self.spec.start_y], dtype=np.float32)
        junction = np.asarray([self.spec.start_x, self.spec.junction_y], dtype=np.float32)
        sx, sy = pix(start)
        jx, jy = pix(junction)

        corridor_value = np.asarray([52,58,64], dtype=np.uint8)
        wall_half = max(2, size//35)
        self._draw_rect(
            img,
            sx-wall_half, min(sy,jy),
            sx+wall_half+1, max(sy,jy)+1,
            corridor_value,
        )

        goal_value = np.asarray([105,105,105], dtype=np.uint8)
        for goal in (
            np.asarray(self.spec.left_goal, dtype=np.float32),
            np.asarray(self.spec.right_goal, dtype=np.float32),
        ):
            gx, gy = pix(goal)
            self._draw_disk(img, gx, gy, max(3,size//24), goal_value)

        nuisance_sets = (
            ((-0.68,0.58),(0.64,0.68)),
            ((-0.58,0.68),(0.70,0.42)),
            ((-0.72,0.38),(0.54,0.58),(0.76,-0.12)),
            ((-0.50,0.48),(0.50,0.48),(0.0,0.74)),
            ((-0.66,0.18),(0.66,0.18),(0.0,0.62)),
        )
        nuisance_value = np.asarray([70,78,86], dtype=np.uint8)
        nr = max(2,size//32)
        for point in nuisance_sets[self.spec.distractor_pattern]:
            px, py = pix(point)
            self._draw_rect(
                img,
                px-nr,py-nr,px+nr+1,py+nr+1,
                nuisance_value,
            )

        ax, ay = pix(self.agent)
        self._draw_disk(
            img, ax, ay, max(3,size//20),
            np.asarray([65,145,245],dtype=np.uint8),
        )

        if self._cue_active():
            cue_value = (
                np.asarray([240,190,45],dtype=np.uint8)
                if self.cue == "left"
                else np.asarray([190,90,220],dtype=np.uint8)
            )
            cs=max(4,size//20)
            img[2:2+cs,2:2+cs] = cue_value

        bar=int(round((size-4)*self.energy))
        img[size-4:size-2,2:2+bar] = np.asarray([85,210,120],dtype=np.uint8)
        return img
