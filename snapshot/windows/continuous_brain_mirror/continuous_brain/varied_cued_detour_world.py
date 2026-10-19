#!/usr/bin/env python3
"""
Varied counterfactual delayed-cue detour family for Continuous Brain.

Purpose
-------
This is the replacement task family for the old CuedDetourWorld admission
assays.  The old environment had only two effective trials at each corridor
length: fixed geometry + LEFT cue, and fixed geometry + RIGHT cue.  Repeating
many RNG seeds therefore created pseudoreplication.

This family makes the *condition* explicit.  Every condition_id maps to one
structurally unique relative task specification.  Formal experiments instantiate
that same condition twice, once with cue="left" and once with cue="right".

The generator deliberately separates:
  * structural factors that change the delayed decision problem;
  * absolute translation factors that change observations/rendering without
    changing relative decision geometry;
  * cue-independent visual distractors.

condition_id does NOT wrap.  Exhausting the finite family raises ValueError,
so train/dev/test code cannot silently reuse a structural condition.

The learner still receives the ordinary 6-value context convention:
    [x, y, cue_dx, cue_dy, energy, last_reward]
Cue deltas and the visual cue token disappear after cue_steps.  Once hidden,
the left/right pair is observation- and render-identical under identical
pre-junction actions.

Formal training/evaluation should always pass an explicit cue.  cue=None is
provided only for interactive convenience and must not be used for admission
statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
import random
from typing import Iterable

import numpy as np

from .environment import Environment


ACTION_NAMES = ("idle", "up", "down", "left", "right")
IDLE, UP, DOWN, LEFT, RIGHT = range(5)


# Relative structural factors.  Only combinations with a genuine hidden delay
# of at least MIN_HIDDEN_DELAY_STEPS are admitted to the family.
_CORRIDOR_LENGTHS = (0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.76, 0.84)
_GOAL_X_OFFSETS = (0.38, 0.46, 0.54, 0.62, 0.70)
_GOAL_Y_OFFSETS = (0.10, 0.14, 0.18, 0.22)
_CUE_STEPS = (1, 2, 3, 4)
_MOVE_STEPS = (0.050, 0.065, 0.080, 0.095)

# Translation/nuisance factors are intentionally NOT part of the conservative
# translation-invariant structural signature.
_START_XS = (-0.18, 0.00, 0.18)
_START_YS = (-0.96, -0.88, -0.80)
_DISTRACTOR_PATTERNS = (0, 1, 2, 3)

MIN_HIDDEN_DELAY_STEPS = 3


def _steps_to_cover(distance: float, step: float) -> int:
    if distance <= 0.0:
        return 0
    return int(math.ceil((float(distance) - 1e-9) / float(step)))


def _prejunction_steps(corridor_length: float, move_step: float) -> int:
    # Teacher moves one movement step beyond the formal junction before
    # committing laterally, preserving the original assay's "clear the
    # junction, then choose" semantics.
    return _steps_to_cover(corridor_length + move_step, move_step)


def _hidden_delay_steps(
    corridor_length: float,
    cue_steps: int,
    move_step: float,
) -> int:
    return _prejunction_steps(corridor_length, move_step) - int(cue_steps)


_STRUCTURAL_GRID = tuple(
    combo
    for combo in itertools.product(
        _CORRIDOR_LENGTHS,
        _GOAL_X_OFFSETS,
        _GOAL_Y_OFFSETS,
        _CUE_STEPS,
        _MOVE_STEPS,
    )
    if _hidden_delay_steps(combo[0], combo[3], combo[4])
    >= MIN_HIDDEN_DELAY_STEPS
)

# Multiplication by 137 is a permutation because gcd(137, len(grid)) == 1.
# This spreads factor combinations throughout sequential condition_id ranges.
_PERMUTE_MULTIPLIER = 137
_PERMUTE_OFFSET = 53
assert math.gcd(_PERMUTE_MULTIPLIER, len(_STRUCTURAL_GRID)) == 1

FAMILY_SIZE = len(_STRUCTURAL_GRID)

# Frozen first-generation split.  Every condition_id has a unique relative
# structural tuple, so these ranges are structurally disjoint.
TRAIN_CONDITION_RANGE = range(0, 1600)
DEV_CONDITION_RANGE = range(1600, 1980)
TEST_CONDITION_RANGE = range(1980, FAMILY_SIZE)

assert FAMILY_SIZE == 2360
assert len(TRAIN_CONDITION_RANGE) == 1600
assert len(DEV_CONDITION_RANGE) == 380
assert len(TEST_CONDITION_RANGE) == 380


@dataclass(frozen=True)
class DetourTaskSpec:
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
    def junction_y(self) -> float:
        return float(self.start_y + self.corridor_length)

    @property
    def decision_y(self) -> float:
        return float(self.junction_y + self.move_step)

    @property
    def left_goal(self) -> tuple[float, float]:
        return (
            float(self.start_x - self.goal_x_offset),
            float(self.junction_y + self.goal_y_offset),
        )

    @property
    def right_goal(self) -> tuple[float, float]:
        return (
            float(self.start_x + self.goal_x_offset),
            float(self.junction_y + self.goal_y_offset),
        )

    def structural_signature(self) -> tuple:
        """Translation- and distractor-invariant task structure."""
        return (
            round(float(self.corridor_length), 6),
            round(float(self.goal_x_offset), 6),
            round(float(self.goal_y_offset), 6),
            int(self.cue_steps),
            round(float(self.move_step), 6),
        )

    def absolute_signature(self) -> tuple:
        """Full generated condition, excluding cue."""
        return (
            int(self.condition_id),
            round(float(self.start_x), 6),
            round(float(self.start_y), 6),
            *self.structural_signature(),
            int(self.distractor_pattern),
        )

    def behavioral_signature(self) -> tuple:
        """
        Conservative relative signature based on required discrete path shape.

        This intentionally ignores absolute translation and distractor pattern.
        Many superficially different coordinate worlds can therefore collapse
        to one behavioral signature, which is exactly what the family audit
        needs to detect.
        """
        pre = _prejunction_steps(self.corridor_length, self.move_step)
        hidden = pre - int(self.cue_steps)
        lateral = _steps_to_cover(self.goal_x_offset, self.move_step)
        post_vertical = _steps_to_cover(
            max(0.0, self.goal_y_offset - self.move_step),
            self.move_step,
        )
        total = pre + lateral + post_vertical
        return (
            int(pre),
            int(hidden),
            int(lateral),
            int(post_vertical),
            int(total),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            {
                "junction_y": self.junction_y,
                "decision_y": self.decision_y,
                "left_goal": list(self.left_goal),
                "right_goal": list(self.right_goal),
                "structural_signature": list(self.structural_signature()),
                "behavioral_signature": list(self.behavioral_signature()),
            }
        )
        return d

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def generate_task_spec(condition_id: int) -> DetourTaskSpec:
    cid = int(condition_id)
    if cid < 0 or cid >= FAMILY_SIZE:
        raise ValueError(
            f"condition_id must be in [0,{FAMILY_SIZE - 1}], got {cid}; "
            "formal task IDs never wrap because reuse would create "
            "pseudoreplication."
        )

    structural_index = (
        cid * _PERMUTE_MULTIPLIER + _PERMUTE_OFFSET
    ) % FAMILY_SIZE

    (
        corridor_length,
        goal_x_offset,
        goal_y_offset,
        cue_steps,
        move_step,
    ) = _STRUCTURAL_GRID[structural_index]

    # Absolute translations and visual nuisance are deterministic functions of
    # condition_id.  They add sensory diversity but do not define the
    # conservative structural/effective condition count.
    start_x = _START_XS[(cid * 5 + 1) % len(_START_XS)]
    start_y = _START_YS[(cid * 7 + 2) % len(_START_YS)]
    distractor_pattern = _DISTRACTOR_PATTERNS[
        (cid * 11 + 3) % len(_DISTRACTOR_PATTERNS)
    ]

    spec = DetourTaskSpec(
        condition_id=cid,
        start_x=float(start_x),
        start_y=float(start_y),
        corridor_length=float(corridor_length),
        goal_x_offset=float(goal_x_offset),
        goal_y_offset=float(goal_y_offset),
        cue_steps=int(cue_steps),
        move_step=float(move_step),
        distractor_pattern=int(distractor_pattern),
    )

    # Construction-time invariants.
    if spec.behavioral_signature()[1] < MIN_HIDDEN_DELAY_STEPS:
        raise AssertionError("generated condition violates hidden-delay gate")
    if min(spec.left_goal[0], spec.right_goal[0]) < -0.98:
        raise AssertionError("goal escaped left world bound")
    if max(spec.left_goal[0], spec.right_goal[0]) > 0.98:
        raise AssertionError("goal escaped right world bound")
    if max(spec.left_goal[1], spec.right_goal[1]) > 0.90:
        raise AssertionError("goal escaped top world bound")

    return spec


def split_name(condition_id: int) -> str:
    cid = int(condition_id)
    if cid in TRAIN_CONDITION_RANGE:
        return "train"
    if cid in DEV_CONDITION_RANGE:
        return "dev"
    if cid in TEST_CONDITION_RANGE:
        return "test"
    raise ValueError(f"invalid condition_id {cid}")


class VariedCuedDetourWorld(Environment):
    """
    One member of a matched counterfactual pair.

    Pairing rule:
        left = VariedCuedDetourWorld(condition_id=N, cue="left")
        right = VariedCuedDetourWorld(condition_id=N, cue="right")

    The two worlds share every non-cue variable exactly.
    """

    observation_dim = 6
    action_dim = 5
    action_names = ACTION_NAMES

    def __init__(
        self,
        condition_id: int,
        *,
        cue: str | None,
        max_episode_steps: int = 160,
        image_size_hint: int = 96,
    ):
        self.condition_id = int(condition_id)
        self.spec = generate_task_spec(self.condition_id)
        self.max_episode_steps = int(max_episode_steps)
        self.image_size_hint = int(image_size_hint)

        if cue not in (None, "left", "right"):
            raise ValueError("cue must be None, 'left', or 'right'")
        self.forced_cue = cue
        self.rng = random.Random(
            0xCB51A7 + 1009 * self.condition_id
        )

        self.agent = np.zeros(2, dtype=np.float32)
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        self.done = False
        self.cue = "left"
        self.reset()

    @property
    def cue_steps(self) -> int:
        return int(self.spec.cue_steps)

    @property
    def move_step(self) -> float:
        return float(self.spec.move_step)

    def _junction_y(self) -> float:
        return float(self.spec.junction_y)

    def _decision_y(self) -> float:
        return float(self.spec.decision_y)

    def _goal(self, cue: str | None = None) -> np.ndarray:
        c = self.cue if cue is None else cue
        xy = self.spec.left_goal if c == "left" else self.spec.right_goal
        return np.asarray(xy, dtype=np.float32)

    def reset(self):
        if self.forced_cue is None:
            # Convenience only. Formal experiments must force cue explicitly.
            self.cue = self.rng.choice(("left", "right"))
        else:
            self.cue = self.forced_cue

        self.agent[:] = [
            float(self.spec.start_x),
            float(self.spec.start_y),
        ]
        self.energy = 1.0
        self.last_reward = 0.0
        self.steps = 0
        self.done = False
        return self._obs()

    def _cue_active(self) -> bool:
        return self.steps < self.cue_steps

    def _obs(self) -> np.ndarray:
        if self._cue_active():
            goal = self._goal()
            delta = goal - self.agent
        else:
            delta = np.zeros(2, dtype=np.float32)

        return np.asarray(
            [
                float(self.agent[0]),
                float(self.agent[1]),
                float(delta[0]),
                float(delta[1]),
                float(self.energy),
                float(self.last_reward),
            ],
            dtype=np.float32,
        )

    def _goal_radius(self) -> float:
        # More than half a movement step so the discrete controller can reach
        # all generated targets, but much smaller than left/right separation.
        return max(0.055, 0.70 * self.move_step)

    def teacher_action(self) -> int:
        """
        Optimal demonstrator used only for labels/evaluation.

        Pre-junction actions are cue-independent.  The first cue-dependent
        action is the lateral LEFT/RIGHT commitment at decision_y.
        """
        half = 0.5 * self.move_step

        if float(self.agent[1]) < self._decision_y() - half:
            return UP

        target = self._goal()
        dx = float(target[0] - self.agent[0])
        dy = float(target[1] - self.agent[1])

        if abs(dx) > half:
            return LEFT if dx < 0.0 else RIGHT
        if abs(dy) > half:
            return DOWN if dy < 0.0 else UP
        return IDLE

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called after episode termination")

        action = int(action)
        self.steps += 1

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

        at_or_past_junction = (
            float(self.agent[1]) >= self._junction_y() - 1e-6
        )
        if not at_or_past_junction:
            # Neutral corridor: no lateral choice before the junction.
            proposed[0] = float(self.spec.start_x)

        proposed = np.clip(proposed, -1.0, 1.0)
        self.agent[:] = proposed

        # Timing/path leak protection: energy is a pure cue-independent
        # function of vertical position relative to the start.
        vertical_progress = max(
            0.0,
            float(self.agent[1] - self.spec.start_y),
        )
        self.energy = float(
            np.clip(1.0 - 0.12 * vertical_progress, 0.0, 1.0)
        )

        reward = -0.004
        reached_correct = False
        reached_side = None
        done = False

        left_goal = np.asarray(self.spec.left_goal, dtype=np.float32)
        right_goal = np.asarray(self.spec.right_goal, dtype=np.float32)
        radius = self._goal_radius()

        if float(np.linalg.norm(self.agent - left_goal)) <= radius:
            reached_side = "left"
            reached_correct = self.cue == "left"
            reward = 1.0 if reached_correct else -1.0
            done = True
        elif float(np.linalg.norm(self.agent - right_goal)) <= radius:
            reached_side = "right"
            reached_correct = self.cue == "right"
            reward = 1.0 if reached_correct else -1.0
            done = True
        elif self.steps >= self.max_episode_steps:
            done = True

        self.last_reward = float(reward)
        self.done = bool(done)

        return (
            self._obs(),
            float(reward),
            bool(done),
            {
                "reached_correct": bool(reached_correct),
                "reached_side": reached_side,
                "at_junction": bool(at_or_past_junction),
                "condition_id": self.condition_id,
                "split": split_name(self.condition_id),
            },
        )

    @staticmethod
    def _draw_disk(img, cx, cy, radius, value):
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = value

    @staticmethod
    def _draw_rect(img, x0, y0, x1, y1, value):
        xa, xb = sorted((max(0, x0), min(img.shape[1], x1)))
        ya, yb = sorted((max(0, y0), min(img.shape[0], y1)))
        if xa < xb and ya < yb:
            img[ya:yb, xa:xb] = value

    def render_rgb(self, size: int = 96) -> np.ndarray:
        size = max(32, int(size))
        img = np.full((size, size, 3), 18, dtype=np.uint8)

        def pix(pos):
            x = int(
                round((float(pos[0]) + 1.0) * 0.5 * (size - 1))
            )
            y = int(
                round(
                    (1.0 - (float(pos[1]) + 1.0) * 0.5)
                    * (size - 1)
                )
            )
            return x, y

        # Background coordinate grid.
        spacing = max(8, size // 8)
        for i in range(0, size, spacing):
            img[i : i + 1, :, :] = 31
            img[:, i : i + 1, :] = 31

        # Render corridor geometry. Both goals are always shown identically;
        # geometry therefore never reveals which side is correct.
        start = np.asarray(
            [self.spec.start_x, self.spec.start_y], dtype=np.float32
        )
        junction = np.asarray(
            [self.spec.start_x, self._junction_y()], dtype=np.float32
        )
        sx, sy = pix(start)
        jx, jy = pix(junction)

        wall_half_px = max(2, size // 35)
        corridor_value = np.asarray([52, 58, 64], dtype=np.uint8)
        self._draw_rect(
            img,
            sx - wall_half_px,
            min(sy, jy),
            sx + wall_half_px + 1,
            max(sy, jy) + 1,
            corridor_value,
        )

        left_goal = np.asarray(self.spec.left_goal, dtype=np.float32)
        right_goal = np.asarray(self.spec.right_goal, dtype=np.float32)
        goal_value = np.asarray([105, 105, 105], dtype=np.uint8)
        goal_r = max(3, size // 24)
        for goal in (left_goal, right_goal):
            gx, gy = pix(goal)
            self._draw_disk(img, gx, gy, goal_r, goal_value)

        # Cue-independent nuisance markers.  They vary appearance without
        # changing the decision problem and are ignored by the conservative
        # structural/effective signature.
        nuisance_value = np.asarray([70, 78, 86], dtype=np.uint8)
        nuisance = (
            ((-0.72, 0.62), (0.62, 0.72)),
            ((-0.62, 0.72), (0.72, 0.46)),
            ((-0.76, 0.42), (0.58, 0.62), (0.78, -0.15)),
            ((-0.55, 0.52), (0.55, 0.52), (0.0, 0.78)),
        )[self.spec.distractor_pattern]
        nr = max(2, size // 32)
        for point in nuisance:
            px, py = pix(point)
            self._draw_rect(
                img,
                px - nr,
                py - nr,
                px + nr + 1,
                py + nr + 1,
                nuisance_value,
            )

        # Agent.
        ax, ay = pix(self.agent)
        agent_value = np.asarray([65, 145, 245], dtype=np.uint8)
        self._draw_disk(
            img,
            ax,
            ay,
            max(3, size // 20),
            agent_value,
        )

        # Cue token at a fixed screen location. Only its color encodes cue.
        if self._cue_active():
            cue_value = (
                np.asarray([240, 190, 45], dtype=np.uint8)
                if self.cue == "left"
                else np.asarray([190, 90, 220], dtype=np.uint8)
            )
            cue_size = max(4, size // 20)
            img[2 : 2 + cue_size, 2 : 2 + cue_size] = cue_value

        # Cue-independent energy bar.
        bar = int(round((size - 4) * float(self.energy)))
        img[size - 4 : size - 2, 2 : 2 + bar] = np.asarray(
            [85, 210, 120], dtype=np.uint8
        )

        return img

    def describe(self) -> str:
        cue = self.cue if self._cue_active() else "hidden"
        return (
            f"condition={self.condition_id} split={split_name(self.condition_id)} "
            f"agent=({self.agent[0]:+.3f},{self.agent[1]:+.3f}) "
            f"cue={cue} geometry={self.spec.structural_signature()}"
        )


def task_family_manifest(condition_ids: Iterable[int]) -> list[dict]:
    return [generate_task_spec(cid).to_dict() for cid in condition_ids]
