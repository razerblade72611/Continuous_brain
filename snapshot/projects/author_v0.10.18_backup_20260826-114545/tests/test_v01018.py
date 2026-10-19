from types import SimpleNamespace

import agent_core
from agent_core import thor_adapter
from agent_core import training


def _state(x, z):
    return SimpleNamespace(pose=SimpleNamespace(position={"x": x, "y": 0.9, "z": z}))


def test_release_version_is_v01018():
    assert agent_core.__version__ == "0.10.18"


def test_component_cross_verification_accepts_oblique_real_transition():
    # Reproduces the shape of the v0.10.17 false negative: the agent moved from
    # the east-room component to the base-room component, but an oblique vector
    # projection through the door transform can be negative.
    grid = 0.25
    before = _state(3.25, -0.50)
    after = _state(0.50, -1.50)
    source = {thor_adapter._grid_position_key(before.pose.position, grid), (12, -2), (11, -2)}
    assert thor_adapter._crossed_door(
        before,
        after,
        {"x": 2.4, "z": 0.825},
        source_component_keys=source,
        grid_size=grid,
    )


def test_cross_target_prefers_new_component_boundary():
    grid = 0.25
    source = {(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)}
    reachable = [
        {"x": 1.0, "y": 0.9, "z": 0.0},
        {"x": 0.75, "y": 0.9, "z": 0.0},
        {"x": 0.50, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 0.00, "y": 0.9, "z": 0.0},
        {"x": -0.25, "y": 0.9, "z": 0.0},
        {"x": -0.50, "y": 0.9, "z": 0.0},
    ]
    target = thor_adapter._choose_cross_threshold_target(
        {"x": 1.0, "z": 0.0},
        {"x": 0.0, "z": 0.0},
        reachable,
        source_component_keys=source,
        grid_size=grid,
    )
    assert thor_adapter._grid_position_key(target, grid) == (-1, 0)


def test_place_pair_preflight_opens_tests_restores_and_caches(monkeypatch):
    source = SimpleNamespace(
        object_id="Plate|x",
        name="Plate_1",
        position={"x": 0.0, "y": 1.0, "z": 0.0},
        raw={"rotation": {"x": 0.0, "y": 0.0, "z": 0.0}},
        parent_receptacles=["Cabinet|x"],
    )
    target = SimpleNamespace(object_id="Cabinet|x", openable=True, is_open=False)
    placed = SimpleNamespace(parent_receptacles=["Cabinet|x"])
    target_state = SimpleNamespace(object_id="Cabinet|x", openable=True, is_open=False)
    world = SimpleNamespace(objects={"Plate|x": placed, "Cabinet|x": target_state}, held_object_id=None)

    class Adapter:
        def observe(self):
            return world

    adapter = Adapter()
    calls = []

    def fake_admin(shared, active, action, **kwargs):
        calls.append((action, kwargs))
        if action == "OpenObject":
            target_state.is_open = True
        elif action == "CloseObject":
            target_state.is_open = False
        return True

    monkeypatch.setattr(training, "_admin_step", fake_admin)
    monkeypatch.setattr(training, "_ensure_empty_hand", lambda adapter: True)
    monkeypatch.setattr(training, "_restore_discovery_props", lambda *a, **k: (True, None, 1))

    ok, reason = training._place_pair_preflight(object(), adapter, 0, source, target)
    assert ok and reason is None
    assert [name for name, _ in calls] == ["OpenObject", "PickupObject", "PutObject", "CloseObject"]

    calls.clear()
    ok2, reason2 = training._place_pair_preflight(object(), adapter, 0, source, target)
    assert ok2 and reason2 is None
    assert calls == []  # cached


def test_place_curriculum_has_explicit_openable_receptacle_stage():
    src = training._train_place.__code__.co_consts
    text = " ".join(str(x) for x in src)
    assert "place_receptacle_open" in text
