from types import SimpleNamespace

import numpy as np

from agent_core.models import Action
from agent_core.thor_adapter import SharedThorWorld, ThorConfig


def child_event(agent_id, x):
    return SimpleNamespace(
        metadata={
            "agentId": agent_id,
            "sceneName": "FloorPlan1_physics",
            "agent": {
                "position": {"x": x, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": [],
            "lastAction": None,
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": [],
        },
        frame=np.zeros((4, 4, 3), dtype=np.uint8) + agent_id,
        instance_detections2D={},
        instance_masks={},
    )


class FakeController:
    def __init__(self):
        self.calls = []
        self.last_event = SimpleNamespace(events=[child_event(0, 0.0), child_event(1, 1.0)])
        self.stopped = False

    def step(self, **kwargs):
        self.calls.append(kwargs)
        aid = kwargs.get("agentId", 0)
        event = self.last_event.events[aid]
        event.metadata["lastAction"] = kwargs.get("action")
        event.metadata["lastActionSuccess"] = True
        if kwargs.get("action") == "GetReachablePositions":
            event.metadata["actionReturn"] = [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ]
        self.last_event = SimpleNamespace(events=self.last_event.events)
        return self.last_event

    def stop(self):
        self.stopped = True


def fake_world():
    world = SharedThorWorld(ThorConfig(scene="FloorPlan1"), agent_count=2)
    world.controller = FakeController()
    return world


def test_shared_world_exposes_distinct_per_agent_events_and_frames():
    world = fake_world()
    a = world.agent(0); b = world.agent(1)
    a.start(); b.start()
    assert a.observe().pose.position["x"] == 0.0
    assert b.observe().pose.position["x"] == 1.0
    assert int(a.raw_event().frame[0, 0, 0]) == 0
    assert int(b.raw_event().frame[0, 0, 0]) == 1


def test_agent_action_is_routed_with_explicit_agent_id():
    world = fake_world()
    b = world.agent(1); b.start()
    b.execute(Action("RotateRight", {}, "test"))
    assert world.controller.calls[-1]["action"] == "RotateRight"
    assert world.controller.calls[-1]["agentId"] == 1


def test_get_reachable_positions_is_agent_specific():
    world = fake_world()
    a = world.agent(0); a.start()
    positions = a.get_reachable_positions()
    assert len(positions) == 2
    assert world.controller.calls[-1] == {"action": "GetReachablePositions", "agentId": 0}


def test_closing_agent_adapter_does_not_stop_shared_world():
    world = fake_world()
    a = world.agent(0); a.start()
    a.close()
    assert world.controller is not None
    assert world.controller.stopped is False
    world.close()
    assert world.controller is None


def test_shared_world_start_requests_visible_agent_bodies(monkeypatch):
    import sys
    import types

    captured = {}

    class InitController:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.last_event = SimpleNamespace(events=[child_event(0, 0.0), child_event(1, 1.0)])
        def stop(self):
            pass

    fake_module = types.SimpleNamespace(Controller=InitController)
    monkeypatch.setitem(sys.modules, "ai2thor.controller", fake_module)

    world = SharedThorWorld(ThorConfig(scene="FloorPlan1"), agent_count=2)
    world.start()
    try:
        assert captured["agentCount"] == 2
        assert captured["makeAgentsVisible"] is True
    finally:
        world.close()
