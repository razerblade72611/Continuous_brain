#!/usr/bin/env python3
"""
Late-bound instruction variant of the admitted varied delayed-cue detour task.

This is a TRAIN-side executive-composition qualification family.

The sensory cue is identical to VariedCuedDetourWorld and disappears on the
same schedule.  The new task instruction is a separate executive channel that
is NEVER inserted into image, context, hidden-state input, or the 64-D memory
writer.

Instruction channel:
    NONE   = 0  (before instruction arrival)
    NORMAL = 1  (respond toward remembered cue)
    INVERT = 2  (respond opposite remembered cue)

The instruction becomes available only after the cue has disappeared and after
the writer has already had cue-bearing timesteps to store the cue.

This module intentionally reuses only TRAIN condition IDs for qualification.
It is not an admission/generalization family.  A fresh second-generation
structural family is required before executive DEV/TEST evaluation.
"""
from __future__ import annotations

import numpy as np

from .varied_cued_detour_world import (
    ACTION_NAMES,
    IDLE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
    split_name,
)

INSTRUCTION_NONE = 0
INSTRUCTION_NORMAL = 1
INSTRUCTION_INVERT = 2
INSTRUCTION_NAMES = ("none", "normal", "invert")


class LateInstructionDetourWorld(VariedCuedDetourWorld):
    """
    Same cue-memory problem, but the required use of the memory is specified
    later through a separate executive channel.

    The frozen sensory/recurrent/memory substrate never receives instruction_id.
    """

    def __init__(
        self,
        condition_id: int,
        *,
        cue: str,
        instruction: str,
        instruction_delay: int = 1,
        max_episode_steps: int = 160,
        image_size_hint: int = 96,
    ):
        if int(condition_id) not in TRAIN_CONDITION_RANGE:
            raise ValueError(
                "LateInstructionDetourWorld v1 is qualification-only and "
                "accepts TRAIN IDs 0..1599; do not use old DEV/TEST here."
            )
        if instruction not in ("normal", "invert"):
            raise ValueError("instruction must be 'normal' or 'invert'")
        self.forced_instruction = instruction
        self.instruction_delay = int(instruction_delay)
        if self.instruction_delay < 1:
            raise ValueError("instruction_delay must be >=1")
        super().__init__(
            condition_id,
            cue=cue,
            max_episode_steps=max_episode_steps,
            image_size_hint=image_size_hint,
        )

        # The existing family guarantees >=3 hidden pre-decision steps.
        # delay=1 therefore arrives after cue erasure and before commitment.
        if self.instruction_start_step >= self.spec.behavioral_signature()[0]:
            raise ValueError(
                "instruction arrives too late for this qualification condition"
            )

    @property
    def instruction_start_step(self) -> int:
        return int(self.cue_steps + self.instruction_delay)

    def instruction_visible(self) -> bool:
        return bool(self.steps >= self.instruction_start_step)

    def instruction_id(self) -> int:
        if not self.instruction_visible():
            return INSTRUCTION_NONE
        return (
            INSTRUCTION_NORMAL
            if self.forced_instruction == "normal"
            else INSTRUCTION_INVERT
        )

    @property
    def response_side(self) -> str:
        if self.forced_instruction == "normal":
            return self.cue
        return "right" if self.cue == "left" else "left"

    def _response_goal(self) -> np.ndarray:
        xy = (
            self.spec.left_goal
            if self.response_side == "left"
            else self.spec.right_goal
        )
        return np.asarray(xy, dtype=np.float32)

    def teacher_action(self) -> int:
        """
        Before the junction, behavior is cue- and instruction-independent.
        After the late instruction has arrived, the correct lateral commitment
        is cue XOR instruction.
        """
        half = 0.5 * self.move_step

        if float(self.agent[1]) < self._decision_y() - half:
            return UP

        if not self.instruction_visible():
            raise RuntimeError(
                "decision reached before late instruction became available"
            )

        target = self._response_goal()
        dx = float(target[0] - self.agent[0])
        dy = float(target[1] - self.agent[1])

        if abs(dx) > half:
            return LEFT if dx < 0.0 else RIGHT
        if abs(dy) > half:
            return DOWN if dy < 0.0 else UP
        return IDLE

    def step(self, action: int):
        """
        Identical physical dynamics to VariedCuedDetourWorld, but success is
        defined by the late-bound response side rather than the original cue.
        """
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
            proposed[0] = float(self.spec.start_x)

        proposed = np.clip(proposed, -1.0, 1.0)
        self.agent[:] = proposed

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
            reached_correct = self.response_side == "left"
            reward = 1.0 if reached_correct else -1.0
            done = True
        elif float(np.linalg.norm(self.agent - right_goal)) <= radius:
            reached_side = "right"
            reached_correct = self.response_side == "right"
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
                "response_side": self.response_side,
                "instruction": self.forced_instruction,
                "instruction_visible": self.instruction_visible(),
                "instruction_id": self.instruction_id(),
                "at_junction": bool(at_or_past_junction),
                "condition_id": self.condition_id,
                "split": split_name(self.condition_id),
            },
        )

    def describe(self) -> str:
        cue = self.cue if self._cue_active() else "hidden"
        inst = (
            self.forced_instruction
            if self.instruction_visible()
            else "not-yet-arrived"
        )
        return (
            f"condition={self.condition_id} split=train "
            f"agent=({self.agent[0]:+.3f},{self.agent[1]:+.3f}) "
            f"cue={cue} instruction={inst} "
            f"response_side={self.response_side} "
            f"geometry={self.spec.structural_signature()}"
        )
