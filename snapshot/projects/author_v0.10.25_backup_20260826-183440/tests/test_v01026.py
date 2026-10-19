from types import SimpleNamespace

import agent_core
import agent_core.training as training


def _world(active_pos, peer_pos, doors=None):
    active_world = SimpleNamespace(
        pose=SimpleNamespace(position=dict(active_pos)),
        objects=doors or {},
    )
    peer_world = SimpleNamespace(
        pose=SimpleNamespace(position=dict(peer_pos)),
        objects=doors or {},
    )
    return active_world, peer_world


def test_release_version_is_v01026():
    assert agent_core.__version__ == "0.10.26"


def test_peer_authoritative_reachability_is_used_for_peer_parking(monkeypatch):
    active_reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 3.0, "y": 0.9, "z": 3.0},
    ]
    peer_reachable = [
        {"x": 4.0, "y": 0.9, "z": 4.0},
        {"x": 4.25, "y": 0.9, "z": 4.0},
        {"x": 4.0, "y": 0.9, "z": 4.25},
    ]
    aw, pw = _world(
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=0.25),
        get_reachable_positions=lambda: list(active_reachable),
        observe=lambda: aw,
    )
    peer_adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=0.25),
        get_reachable_positions=lambda: list(peer_reachable),
        observe=lambda: pw,
    )
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_maneuverable_starts", lambda rows, *a, **k: list(rows))
    calls = []
    monkeypatch.setattr(
        training,
        "_teleport",
        lambda shared, aid, point, yaw, horizon=0.0: calls.append((aid, dict(point))) or True,
    )

    ok, reason = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, 0, 1, [], peer_adapter=peer_adapter
    )
    assert ok is True
    assert reason is None
    assert calls
    assert calls[0][0] == 1
    assert calls[0][1] in peer_reachable


def test_already_neutral_peer_does_not_require_teleport(monkeypatch):
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 3.0, "y": 0.9, "z": 3.0},
        {"x": 3.25, "y": 0.9, "z": 3.0},
        {"x": 3.0, "y": 0.9, "z": 3.25},
    ]
    aw, pw = _world(
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 3.0, "y": 0.9, "z": 3.0},
    )
    adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(reachable), observe=lambda: aw)
    peer_adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(reachable), observe=lambda: pw)
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(training, "_teleport", lambda *a, **k: calls.append((a, k)) or True)

    ok, reason = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, 0, 1, [], peer_adapter=peer_adapter
    )
    assert ok is True
    assert reason is None
    assert calls == []


def test_pair_normalization_fallback_moves_both_agents(monkeypatch):
    active_reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": -3.0, "y": 0.9, "z": -3.0},
    ]
    peer_reachable = [
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 3.0, "y": 0.9, "z": 3.0},
    ]
    aw, pw = _world(
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
    )
    adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(active_reachable), observe=lambda: aw)
    peer_adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(peer_reachable), observe=lambda: pw)
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_maneuverable_starts", lambda rows, *a, **k: list(rows))
    calls = []

    def fake_teleport(shared, aid, point, yaw, horizon=0.0):
        calls.append((aid, dict(point)))
        # Direct peer-only parking attempt fails; once active is normalized, peer succeeds.
        if len(calls) == 1 and aid == 1:
            return False
        return True

    monkeypatch.setattr(training, "_teleport", fake_teleport)
    ok, reason = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, 0, 1, [], peer_adapter=peer_adapter
    )
    assert ok is True
    assert reason is None
    assert any(aid == 0 for aid, _ in calls)
    assert calls[-1][0] == 1


def test_exhaustion_reason_contains_parking_diagnostics(monkeypatch):
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 3.0, "y": 0.9, "z": 3.0},
    ]
    aw, pw = _world(
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
    )
    adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(reachable), observe=lambda: aw)
    peer_adapter = SimpleNamespace(config=SimpleNamespace(grid_size=0.25), get_reachable_positions=lambda: list(reachable), observe=lambda: pw)
    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_maneuverable_starts", lambda rows, *a, **k: list(rows))
    monkeypatch.setattr(training, "_teleport", lambda *a, **k: False)
    ok, reason = training._park_peer_clear_of_level4_setup_search(
        object(), adapter, 0, 1, [], peer_adapter=peer_adapter
    )
    assert ok is False
    assert "active_reachable=" in reason
    assert "peer_reachable=" in reason
    assert "peer_teleport_failures=" in reason


def test_v01026_replay_schemas_are_current():
    src = open(training.__file__, encoding="utf-8").read()
    assert '"schema": "author.embodied.v01026"' in src
    assert '"schema": "author.embodied_sft.v01026"' in src
