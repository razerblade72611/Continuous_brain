from pathlib import Path
from types import SimpleNamespace

import agent_core
import agent_core.training as training


def test_release_version_is_v01026():
    assert agent_core.__version__ == "0.10.26"


def test_toggle_route_search_parks_peer_before_route_qualification():
    src = Path(training.__file__).read_text(encoding="utf-8")
    block = src[src.index("def _train_toggle"):src.index("def _train_peer_meet")]
    park = "_park_peer_clear_of_level4_setup_search("
    find = "_find_object_with_blocked_route(shared, adapter, active, rng, toggle_candidates)"
    assert park in block
    assert find in block
    assert block.index(park) < block.index(find)
    assert "1-active" in block[block.index(park):block.index(find)]


def test_peer_setup_parking_prefers_maneuverable_cell_away_from_door_and_active(monkeypatch):
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.0, "y": 0.9, "z": 1.5},
        {"x": 3.0, "y": 0.9, "z": 3.0},
    ]
    world = SimpleNamespace(
        pose=SimpleNamespace(position={"x": 0.0, "y": 0.9, "z": 0.0}),
        objects={
            "door": SimpleNamespace(object_type="Door", position={"x": 0.0, "y": 0.0, "z": 1.5}),
        },
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=0.25),
        get_reachable_positions=lambda: list(reachable),
        observe=lambda: world,
    )
    target = SimpleNamespace(position={"x": 0.5, "y": 0.8, "z": 0.5})
    calls = []
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_maneuverable_starts", lambda *a, **k: list(reachable))
    monkeypatch.setattr(
        training,
        "_teleport",
        lambda shared, agent_id, point, yaw, horizon=0.0: calls.append((agent_id, dict(point), yaw, horizon)) or True,
    )

    ok = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, active=0, peer_id=1, candidates=[target]
    )
    assert ok is True
    assert calls
    agent_id, point, yaw, horizon = calls[0]
    assert agent_id == 1
    assert point == {"x": 3.0, "y": 0.9, "z": 3.0}
    assert yaw == 180.0
    assert horizon == 0.0


def test_peer_setup_parking_fails_closed_when_neutral_placement_is_impossible(monkeypatch):
    reachable = [{"x": 0.0, "y": 0.9, "z": 0.0}]
    world = SimpleNamespace(
        pose=SimpleNamespace(position={"x": 0.0, "y": 0.9, "z": 0.0}),
        objects={},
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=0.25),
        get_reachable_positions=lambda: list(reachable),
        observe=lambda: world,
    )
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_maneuverable_starts", lambda *a, **k: list(reachable))
    teleports = []
    monkeypatch.setattr(training, "_teleport", lambda *a, **k: teleports.append((a, k)) or True)

    ok = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, active=0, peer_id=1, candidates=[]
    )
    assert ok is False
    assert teleports == []


def test_broken_toggle_guard_is_retained_in_v01026():
    broken = SimpleNamespace(raw={"isBroken": True})
    healthy = SimpleNamespace(raw={"isBroken": False})
    assert training._toggle_target_is_functional(broken) is False
    assert training._toggle_target_is_functional(healthy) is True


def test_v01026_replay_schema_is_current():
    src = Path(training.__file__).read_text(encoding="utf-8")
    assert '"schema": "author.embodied.v01026"' in src


def test_v01026_sft_schema_is_current():
    src = Path(training.__file__).read_text(encoding="utf-8")
    assert '"schema": "author.embodied_sft.v01026"' in src
