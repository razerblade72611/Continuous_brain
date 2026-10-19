from types import SimpleNamespace

import agent_core
from agent_core.training import _admin_step, _teleport


class AlreadySelectedEvent:
    def __init__(self, success=True):
        self.metadata = {"lastActionSuccess": success}
        # AI2-THOR child Events may expose a singleton .events collection.
        # A second agent-index selection would fail for agent 1.
        self.events = [self]


class SelectedEventWorld:
    def __init__(self):
        self.controller = object()
        self.calls = []

    def step_for(self, agent_id, action, **params):
        self.calls.append((agent_id, action, params))
        return AlreadySelectedEvent(True)


def test_release_version_is_v0108():
    assert agent_core.__version__ == "0.10.25"


def test_teleport_agent_b_does_not_double_select_selected_event():
    world = SelectedEventWorld()
    assert _teleport(world, 1, {"x": 1.0, "y": 0.9, "z": 2.0}, 90.0) is True
    assert world.calls[-1][0] == 1
    assert world.calls[-1][1] == "TeleportFull"


def test_admin_step_agent_b_does_not_double_select_selected_event():
    world = SelectedEventWorld()
    assert _admin_step(world, 1, "OpenObject", objectId="Door|test", forceAction=True) is True
    assert world.calls[-1][0] == 1
    assert world.calls[-1][1] == "OpenObject"
