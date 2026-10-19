"""The scripted executor organ (r3-H): the routine library's options, run on command, never choosing the goal.

Sol's organ rule (DAV-96): the organ receives an explicit command (an option) from the level above, returns
execution telemetry, has no authority to pick the high-level goal, and every primitive action it takes is logged
with provenance.  So the composite ``routine`` option (the whole routine's own decision) is *not* a command it
accepts: only the atomic options of ``SkillLibrary`` (drink, eat, rest, retreat, explore, requests, build).

``ExecutorOrgan`` keeps its own plan state (the library's to_state), fresh each life, on the actor's beliefs.
``BRANCH_TO_OPTION`` maps the routine's intent branch (``intent.BRANCHES``) to the organ command that serves it:
the label a learned selector imitates (r3-H arm B) and the command arm A replays.
"""
from __future__ import annotations

from continuous_brain_individual.skills import OPTIONS, SkillLibrary

COMMANDS = ("drink", "eat", "rest", "leave", "retreat", "explore", "requests", "build")
BRANCH_TO_OPTION = {"inline": None,            # fighting an adjacent zombie / sleeping where it stands: keep command
                    "flee": "retreat", "drink": "drink", "eat": "eat", "enter_shelter": "rest",
                    "shelter_step": "rest", "sleeping": "rest", "obligations": "requests", "acquire": "build",
                    "tech_tree": "build", "wander": "explore", "explore": "explore"}


def command_for(branch: str, leaving: bool, previous: str) -> str:
    """The organ command that serves the routine's current intent (the label a selector imitates)."""
    if branch == "shelter_step" and leaving:
        return "leave"
    return BRANCH_TO_OPTION[branch] or previous


class ExecutorOrgan:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, command: str, memory: dict, body: dict, step: int, commitments: list[dict]):
        """(action name, telemetry).  The action is "noop" when the command cannot be carried out now (telemetry
        says so); the organ never substitutes a goal of its own."""
        if command not in COMMANDS:
            raise ValueError(f"the organ does not accept {command!r}: only {COMMANDS}")
        lib = SkillLibrary.from_state(self.state, seed=self.seed)
        v = lib.belief_view(memory, body, step, commitments)
        if v.sleeping:
            name, status = "noop", "asleep"
        else:
            lib._prepare(v)
            if command == "leave":                     # out of its own shelter (the routine's _leave)
                lib.view = v
                name = lib._leave(v) if lib.shelter is not None else None
                done = lib.shelter is None
            else:
                name = lib.option_act(OPTIONS.index(command), v, memory, step)
                done = lib.option_done(OPTIONS.index(command), v, 0)
            status = "ok" if name is not None else "cannot"
            name = name or "noop"
            status = status if not done else status + "+done"
        self.state = lib.to_state()
        return name, {"command": command, "action": name, "status": status}
