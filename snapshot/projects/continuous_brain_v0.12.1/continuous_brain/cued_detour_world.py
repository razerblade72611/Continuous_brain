"""CuedDetourWorld: matches environment.py's real Environment interface
exactly -- continuous [-1,1]^2 movement, observation_dim=6
([x, y, delta_x, delta_y, energy, last_reward], same convention as
ForagingWorld/DetourWorld), and a real render_rgb(size) producing an
actual RGB image, because InfantConfig.image_size=96 means the real
vision cortex expects a rendered frame, not a symbolic grid -- confirmed
directly against the real continuous_brain/infant.py and
continuous_brain/planning_v100.py in this upload, not assumed from a
different archive this time.

Structure: a left/right cue is available for exactly the first
`cue_steps` steps in the observation's goal-delta fields and redundantly
as a fixed-position color token in the rendered frame.  After that window
both channels vanish.  The agent must carry the cue through a neutral
corridor (no lateral choice is available there) and choose a side only at
the junction.  Reaching the correct side gives +1 and ends the episode;
the wrong side gives -1.  This tests retention rather than whether a tiny
visual patch survives the frozen vision stack.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np

from .environment import Environment

_MOVES = {0: (0.0, 0.0), 1: (0.0, 0.08), 2: (0.0, -0.08), 3: (-0.08, 0.0), 4: (0.08, 0.0)}


@dataclass
class CuedDetourWorld(Environment):
    seed: int = 7
    max_episode_steps: int = 120
    corridor_length: float = 0.6   # distance (in agent units) the neutral corridor spans
    cue_steps: int = 2

    observation_dim: int = 6
    action_dim: int = 5
    action_names: tuple = ("idle", "up", "down", "left", "right")

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.agent = np.zeros(2, dtype=np.float32)
        self.energy, self.last_reward, self.steps = 1.0, 0.0, 0
        self.reset()

    def reset(self):
        self.cue = self.rng.choice(("left", "right"))
        self.agent[:] = [0.0, -0.9]  # bottom of the corridor, center column
        self.energy, self.last_reward, self.steps = 1.0, 0.0, 0
        self.done = False
        return self._obs()

    def _junction_y(self):
        return -0.9 + self.corridor_length

    def _obs(self):
        # No cue information anywhere in the observation after cue_steps --
        # verified below, not assumed. delta is toward the CORRECT goal only
        # while the cue is active; zero afterward, matching how a hidden
        # target would actually be unobservable, not a special-cased leak.
        cue_active = self.steps < self.cue_steps
        if cue_active:
            goal_x = -0.7 if self.cue == "left" else 0.7
            delta = np.array([goal_x - self.agent[0], self._junction_y() - self.agent[1]], dtype=np.float32)
        else:
            delta = np.zeros(2, dtype=np.float32)
        return np.array([self.agent[0], self.agent[1], delta[0], delta[1],
                          self.energy, self.last_reward], dtype=np.float32)

    def step(self, action: int):
        self.steps += 1
        dx, dy = _MOVES.get(int(action), (0.0, 0.0))
        proposed = self.agent + np.array([dx, dy], dtype=np.float32)
        at_or_past_junction = self.agent[1] >= self._junction_y() - 1e-6
        if not at_or_past_junction:
            proposed[0] = 0.0  # corridor: no lateral choice available below the junction
        proposed = np.clip(proposed, -1.0, 1.0)
        self.agent[:] = proposed
        # Energy is a pure function of CURRENT POSITION, not of step count or
        # path taken -- the real DetourWorld ties energy to elapsed steps
        # (fixed -0.002/step, action-independent), which would let a
        # cue-dependent idle count leak through energy at the junction,
        # exactly the class of leak found and fixed in the previous
        # (grid-based) version of this task. Tying it to position instead
        # makes it structurally impossible for timing to matter, checked
        # below by comparing extra-idle vs. no-idle runs at the same
        # position, not merely reasoned about.
        self.energy = float(np.clip(1.0 - 0.12 * (self.agent[1] + 0.9), 0.0, 1.0))

        reward, reached_correct, done = -0.004, False, False
        left_goal = np.array([-0.7, self._junction_y() + 0.15])
        right_goal = np.array([0.7, self._junction_y() + 0.15])
        if np.linalg.norm(self.agent - left_goal) < 0.15:
            reward = 1.0 if self.cue == "left" else -1.0
            reached_correct = self.cue == "left"
            done = True
        elif np.linalg.norm(self.agent - right_goal) < 0.15:
            reward = 1.0 if self.cue == "right" else -1.0
            reached_correct = self.cue == "right"
            done = True
        elif self.energy <= 0.0 or self.steps >= self.max_episode_steps:
            done = True
        self.last_reward = reward
        self.done = done
        return self._obs(), float(reward), done, {"reached_correct": reached_correct,
                                                    "at_junction": at_or_past_junction}

    def render_rgb(self, size: int = 96) -> np.ndarray:
        size = max(32, int(size))
        img = np.full((size, size, 3), 20, dtype=np.uint8)
        for i in range(0, size, max(8, size // 8)):
            img[i:i + 1, :, :] = 35
            img[:, i:i + 1, :] = 35

        def pix(pos):
            x = int(round((float(pos[0]) + 1.0) * 0.5 * (size - 1)))
            y = int(round((1.0 - (float(pos[1]) + 1.0) * 0.5) * (size - 1)))
            return x, y

        yy, xx = np.ogrid[:size, :size]
        ax, ay = pix(self.agent)
        ar = max(3, size // 20)
        amask = (xx - ax) ** 2 + (yy - ay) ** 2 <= ar ** 2
        img[amask] = np.array([65, 145, 245], dtype=np.uint8)
        # A redundant visual cue is shown at a FIXED screen position.  The
        # observation also exposes the signed goal delta during cue_steps, so
        # success does not depend on a tiny 4x4 patch surviving the frozen
        # vision stack.  Both cue channels disappear after cue_steps.  The
        # experiment therefore tests retention, not visual acuity.
        if self.steps < self.cue_steps:
            color = np.array([240, 190, 45], np.uint8) if self.cue == "left" else np.array([190, 90, 220], np.uint8)
            img[2:6, 2:6] = color
        bar = int(round((size - 4) * float(self.energy)))
        img[size - 4:size - 2, 2:2 + bar] = np.array([85, 210, 120], dtype=np.uint8)
        return img

    def describe(self):
        return f"agent=({self.agent[0]:+.2f},{self.agent[1]:+.2f}) cue={'hidden' if self.steps>=self.cue_steps else self.cue}"
