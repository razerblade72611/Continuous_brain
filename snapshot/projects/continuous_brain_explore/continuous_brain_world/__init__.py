"""W1: the first real world of the Continuous Brain exploration lane.

Crafter (pixels, survival needs, day/night, monsters, a tech tree, 22 achievements) plus an obligations layer:
a seeded caretaker who issues, amends and cancels requests in English, asks memory questions about what the
agent could see, and scores everything from Crafter's internal state.  See docs/explore/W1_WORLD.md.
"""
from .env import SNAPSHOT_FORMAT, W1Config, W1Env, action_names
from .scoring import aggregate, crafter_score, episode_summary, success_rates

__all__ = ["W1Env", "W1Config", "SNAPSHOT_FORMAT", "action_names", "aggregate", "crafter_score",
           "episode_summary", "success_rates"]
