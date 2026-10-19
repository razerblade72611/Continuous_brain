from __future__ import annotations

from types import SimpleNamespace

import agent_core
import agent_core.training as training


def _obj(object_id, name, position, rotation, *, pickupable=True, moveable=False):
    return SimpleNamespace(
        object_id=object_id,
        name=name,
        position=dict(position),
        pickupable=pickupable,
        raw={"rotation": dict(rotation), "moveable": moveable},
    )


def test_release_version_is_v01017():
    assert agent_core.__version__ == "0.10.24"


def test_capture_discovery_prop_baseline_includes_rotation_and_name():
    book = _obj(
        "Book|-00.62|+00.86|+06.92",
        "Discovery_Book",
        {"x": -0.62, "y": 0.86, "z": 6.92},
        {"x": 0.0, "y": 20.0, "z": 0.0},
    )
    adapter = SimpleNamespace(observe=lambda: SimpleNamespace(objects={book.object_id: book}))
    baseline = training._capture_discovery_prop_positions(adapter)
    assert baseline[book.object_id]["object_name"] == "Discovery_Book"
    assert baseline[book.object_id]["position"]["y"] == 0.86
    assert baseline[book.object_id]["rotation"]["y"] == 20.0


def test_exact_prop_restore_uses_set_object_poses_and_preserves_other_movables(monkeypatch):
    home = {"x": -0.62, "y": 0.86, "z": 6.92}
    home_rot = {"x": 0.0, "y": 20.0, "z": 0.0}
    fallen = _obj(
        "Book|-00.62|+00.86|+06.92",
        "Discovery_Book",
        {"x": -0.62, "y": 0.12, "z": 6.92},
        {"x": 90.0, "y": 20.0, "z": 0.0},
    )
    apple = _obj(
        "Apple|+00.10|+00.80|+00.20",
        "Apple_1",
        {"x": 0.10, "y": 0.80, "z": 0.20},
        {"x": 0.0, "y": 0.0, "z": 0.0},
    )
    world = SimpleNamespace(objects={fallen.object_id: fallen, apple.object_id: apple}, held_object_id=None)
    adapter = SimpleNamespace(observe=lambda: world)
    calls = []

    def fake_admin(_shared, _aid, action, **params):
        calls.append((action, params))
        assert action == "SetObjectPoses"
        poses = {row["objectName"]: row for row in params["objectPoses"]}
        assert set(poses) == {"Discovery_Book", "Apple_1"}
        assert poses["Discovery_Book"]["position"] == home
        assert poses["Discovery_Book"]["rotation"] == home_rot
        assert poses["Apple_1"]["position"] == apple.position
        assert params["placeStationary"] is True
        fallen.position = dict(home)
        fallen.raw["rotation"] = dict(home_rot)
        return True

    monkeypatch.setattr(training, "_admin_step", fake_admin)
    ok, reason, restored = training._restore_discovery_props(
        object(), {0: adapter, 1: adapter}, 0,
        {fallen.object_id: {"object_name": "Discovery_Book", "position": home, "rotation": home_rot}},
    )
    assert ok is True
    assert reason is None
    assert restored == 1
    assert len(calls) == 1


def test_exact_prop_restore_detects_in_place_tip(monkeypatch):
    home = {"x": 1.0, "y": 1.0, "z": 2.0}
    obj = _obj("Book|x", "Discovery_Book", home, {"x": 89.0, "y": 0.0, "z": 0.0})
    world = SimpleNamespace(objects={obj.object_id: obj}, held_object_id=None)
    adapter = SimpleNamespace(observe=lambda: world)

    def fake_admin(_shared, _aid, action, **params):
        assert action == "SetObjectPoses"
        obj.raw["rotation"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        return True

    monkeypatch.setattr(training, "_admin_step", fake_admin)
    ok, reason, restored = training._restore_discovery_props(
        object(), {0: adapter, 1: adapter}, 0,
        {obj.object_id: {"object_name": obj.name, "position": home, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}}},
    )
    assert ok and reason is None and restored == 1


def test_restore_refuses_to_rewrite_registry_while_object_held():
    obj = _obj("Book|x", "Discovery_Book", {"x": 1.0, "y": 0.2, "z": 2.0}, {"x": 0, "y": 0, "z": 0})
    world = SimpleNamespace(objects={obj.object_id: obj}, held_object_id="Apple|held")
    adapter = SimpleNamespace(observe=lambda: world)
    ok, reason, restored = training._restore_discovery_props(
        object(), {0: adapter, 1: adapter}, 0,
        {obj.object_id: {"object_name": obj.name, "position": {"x": 1.0, "y": 1.0, "z": 2.0}, "rotation": {"x": 0, "y": 0, "z": 0}}},
    )
    assert not ok
    assert "holding" in reason
    assert restored == 0
