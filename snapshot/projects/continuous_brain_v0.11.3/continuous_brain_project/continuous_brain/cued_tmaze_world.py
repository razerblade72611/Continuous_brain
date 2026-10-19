"""CuedTMazeWorld: a T-maze, matching ResourceWorld's exact interface
(4-channel 9x9 grid, 18-dim sensors, 5 actions) so it can be dropped into
the existing DevelopmentBrain/runtime.py pipeline with no model input-shape
changes.

Structure: agent starts at the bottom of a single-wide vertical corridor
(walled on both sides -- left/right actions collide with a wall and go
nowhere while in the corridor). At step 0 ONLY, one of two grid channels
(3=left-cue, 4 would break the 4-channel contract, see below) briefly
marks which side will be rewarded; every step after that, both cue
channels and every sensor entry are held at exactly zero for the cue,
by construction (verified below), not by convention. At the top of the
corridor the walls open into two goal cells; taking the correct side
(matching the now long-gone cue) gives +1 and ends the episode, the wrong
side gives -1 and ends it.

Channel budget: ResourceWorld uses 4 channels (walls, food, water, agent).
This environment has no separate food/water, so the SAME two content
channels are reused as (left-cue, right-cue) instead -- still 4 channels
total, still a direct drop-in for InfantBrain/DevelopmentBrain's
Conv2d(4, ...) input, not a new input width to accommodate.

corridor_length is a parameter specifically so retention can be tested at
several horizons -- see PROPOSAL.md for why.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np

SIZE = 9
ACTIONS = ("rest", "up", "down", "left", "right")
MOVES = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
SENSOR_DIM = 18


@dataclass
class CuedTMazeWorld:
    seed: int = 1
    corridor_length: int = 4
    horizon: int = 64

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self):
        self.walls = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.walls[0, :] = self.walls[-1, :] = 1
        self.walls[:, 0] = self.walls[:, -1] = 1
        cx = SIZE // 2
        self.junction_row = SIZE - 2 - self.corridor_length
        assert self.junction_row >= 2, "corridor_length too long for SIZE"
        # wall the corridor on both sides from start up to (not including) the junction row
        for y in range(self.junction_row + 1, SIZE - 1):
            for x in range(1, SIZE - 1):
                if x != cx:
                    self.walls[y, x] = 1
        self.left_goal = (cx - 1, self.junction_row)
        self.right_goal = (cx + 1, self.junction_row)
        self.agent = (cx, SIZE - 2)
        self.cue = self.rng.choice(("left", "right"))
        self.steps = 0
        self.done = False
        return self.observe()

    def observe(self):
        grid = np.zeros((4, SIZE, SIZE), dtype=np.float32)
        grid[0] = self.walls
        cue_active = self.steps == 0
        if cue_active and self.cue == "left":
            grid[1, 1, 1] = 1.0   # arbitrary fixed marker cell, channel 1 = "left cue"
        if cue_active and self.cue == "right":
            grid[2, 1, 1] = 1.0   # channel 2 = "right cue"
        grid[3, self.agent[1], self.agent[0]] = 1.0
        sensors = np.zeros(SENSOR_DIM, np.float32)
        sensors[0] = 1.0 if cue_active and self.cue == "left" else 0.0
        sensors[1] = 1.0 if cue_active and self.cue == "right" else 0.0
        sensors[2] = self.agent[0] / SIZE
        sensors[3] = self.agent[1] / SIZE
        sensors[4] = 0.0  # reserved, always zero (see note below on why the previous step-count signal is gone)
        sensors[5] = 1.0 if self.agent[1] == self.junction_row else 0.0  # "at junction", not which side
        # An earlier version put steps/horizon in sensors[4]. Removed: an
        # agent could deliberately waste cue-dependent time in the
        # corridor (e.g. bump the side wall once only when cued left) and
        # encode the cue as a timing/step-count difference rather than
        # retaining it internally -- exactly the leak this round asked to
        # be ruled out, not just spot-checked. With no step signal and no
        # last-action signal (never added here), the observation below is
        # verified (see test file) to be a function of POSITION ALONE
        # post-cue, for ANY action sequence reaching that position, not
        # just the one fixed sequence checked previously.
        return {"grid": grid, "sensors": sensors}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode ended; reset before stepping again")
        self.steps += 1
        x, y = self.agent
        dx, dy = MOVES[action]
        proposed = (x + dx, y + dy)
        collided = bool(self.walls[proposed[1], proposed[0]])
        self.agent = self.agent if collided else proposed
        reward, reached_correct = -0.01, False
        if self.agent == self.left_goal:
            reward = 1.0 if self.cue == "left" else -1.0
            reached_correct = self.cue == "left"
            self.done = True
        elif self.agent == self.right_goal:
            reward = 1.0 if self.cue == "right" else -1.0
            reached_correct = self.cue == "right"
            self.done = True
        elif self.steps >= self.horizon:
            self.done = True
        return self.observe(), reward, self.done, {"collided": collided, "reached_correct": reached_correct,
                                                     "at_junction": self.agent[1] == self.junction_row}
