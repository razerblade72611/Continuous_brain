from __future__ import annotations

import random
from types import SimpleNamespace

import agent_core
import agent_core.training as training
from agent_core.interaction_pose import InteractionPose


def test_release_version_is_v01014():
    assert agent_core.__version__ == "0.10.15"


def test_blocked_object_search_rejects_first_target_inside_same_raw_setup(monkeypatch):
    first = SimpleNamespace(object_id="Apple|base")
    second = SimpleNamespace(object_id="Book|north")
    pose = InteractionPose(x=1.0, y=0.9, z=1.0, rotation=90.0, horizon=0.0)
    reachable = [{"x": 0.0, "y": 0.9, "z": 0.0}]
    calls = []

    def fake_try(_shared, _adapter, _active, _rng, obj, max_pose_trials=4):
        calls.append(obj.object_id)
        if obj.object_id == "Apple|base":
            return None
        return reachable, pose, "Door|north"

    monkeypatch.setattr(training, "_try_blocked_route_for_object", fake_try)
    class NoShuffle(random.Random):
        def shuffle(self, x):
            return None

    result = training._find_object_with_blocked_route(
        object(), object(), 0, NoShuffle(1), [first, second]
    )
    assert calls == ["Apple|base", "Book|north"]
    assert result is not None
    obj, component, selected_pose, door_id = result
    assert obj.object_id == "Book|north"
    assert component == reachable
    assert selected_pose == pose
    assert door_id == "Door|north"


def test_position_key_quantizes_grid_cells():
    assert training._position_key({"x": 1.01, "z": -0.49}, 0.25) == (4, -2)
