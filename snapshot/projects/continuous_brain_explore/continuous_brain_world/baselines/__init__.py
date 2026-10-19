"""W1 baseline agents.  Both are reference points, not learners:

* ``RandomAgent``  -- uniform random actions, never speaks (lower reference);
* ``ScriptedAgent`` -- PRIVILEGED scripted heuristic reading ground truth from ``info`` (upper reference).
"""
from .random_agent import RandomAgent
from .scripted import ScriptedAgent

__all__ = ["RandomAgent", "ScriptedAgent"]
