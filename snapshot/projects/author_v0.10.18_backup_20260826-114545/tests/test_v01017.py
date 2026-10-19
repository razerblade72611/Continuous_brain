from __future__ import annotations

from types import SimpleNamespace
import inspect

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
    assert agent_core.__version__ == "0.10.18"


def test_canonical_baseline_includes_stock_and_discovery_pickupables():
    stock = _obj("Bread|x", "Bread_1", {"x": 0, "y": 1, "z": 0}, {"x": 0, "y": 0, "z": 0})
    discovery = _obj("Book|x", "Discovery_Book", {"x": 1, "y": 1, "z": 2}, {"x": 0, "y": 20, "z": 0})
    fixed = _obj("Counter|x", "CounterTop", {"x": 2, "y": 1, "z": 2}, {"x": 0, "y": 0, "z": 0}, pickupable=False, moveable=False)
    world = SimpleNamespace(objects={o.object_id:o for o in (stock, discovery, fixed)})
    baseline = training._capture_discovery_prop_positions(SimpleNamespace(observe=lambda: world))
    assert stock.object_id in baseline
    assert discovery.object_id in baseline
    assert fixed.object_id not in baseline


def test_missing_baseline_object_is_considered_displaced():
    home = {"position": {"x": 1, "y": 1, "z": 1}, "rotation": {"x": 0, "y": 0, "z": 0}}
    assert training._discovery_prop_displaced(None, home, 0.12, 8.0) is True


def test_pickup_no_longer_uses_place_object_at_point_cleanup():
    source = inspect.getsource(training._train_pickup)
    assert "_restore_object_point" not in source
    assert "_ensure_empty_hand(adapter)" in source


def test_replay_schema_is_v01017(tmp_path):
    src = inspect.getsource(training._append_embodied_transition)
    assert 'author.embodied.v01018' in src
