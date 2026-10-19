from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_core.learning import LearningPaths, reset_learning_state
from agent_core.memory import EpisodicMemory
from agent_core.nav_skill import NavigationSkill
from agent_core.thor_adapter import ThorConfig, _learned_route_actions
from agent_core.world_map import PersistentWorldMap


def pos(x: float, z: float, y: float = 0.9):
    return {"x": x, "y": y, "z": z}


def test_persistent_memory_reloads_and_safe_context_excludes_visual_prose(tmp_path: Path):
    path = tmp_path / "state" / "memory" / "a" / "memory.jsonl"
    memory = EpisodicMemory(str(path), max_records=20)
    memory.add(
        "embodied_decision",
        {
            "observation": "SECRET OLD CAMERA DESCRIPTION",
            "intention": "inspect the counter",
            "attention": "counter",
            "semantic_actions": [{"kind": "MoveAhead", "target": None, "params": {"objectId": "hidden"}}],
            "success": True,
            "grounded_object_id": "Apple|1|2|3",
        },
    )
    memory.add(
        "training_summary",
        {
            "scene": "FloorPlan1",
            "episodes": 25,
            "primitive_steps": 300,
            "map_cells_learned": 18,
            "coverage_pct": 40.0,
            "peer_encounters": 3,
            "navigation_checkpoint_saved": True,
        },
    )

    reloaded = EpisodicMemory(str(path), max_records=20, load_existing=True)
    assert len(reloaded.records) == 2
    payload = json.dumps(reloaded.safe_context(limit=10))
    assert "inspect the counter" in payload
    assert "training_summary" in payload
    assert "SECRET OLD CAMERA DESCRIPTION" not in payload
    assert "Apple|1|2|3" not in payload
    assert '"objectId"' not in payload


def test_reset_learning_state_deletes_only_state_directory(tmp_path: Path):
    project = tmp_path / "project"
    state = project / "state"
    state.mkdir(parents=True)
    (state / "memory.txt").write_text("old", encoding="utf-8")
    keep = project / "keep.txt"
    keep.write_text("keep", encoding="utf-8")

    reset = reset_learning_state(state)
    assert reset == state.resolve()
    assert state.exists()
    assert not (state / "memory.txt").exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_reset_learning_state_refuses_home(monkeypatch, tmp_path: Path):
    # The explicit broad-path guard must remain in place even if the caller asks.
    with pytest.raises(ValueError):
        reset_learning_state(Path.home())


def test_world_map_learns_route_and_persists(tmp_path: Path):
    path = tmp_path / "map.json"
    world_map = PersistentWorldMap("FloorPlan1", 0.25, path)
    a, b, c = pos(0.0, 0.0), pos(0.25, 0.0), pos(0.5, 0.0)
    world_map.record_transition(a, b, True)
    world_map.record_transition(b, c, True)
    world_map.save()

    loaded = PersistentWorldMap("FloorPlan1", 0.25, path)
    route = loaded.route(a, c, interaction_distance=0.1)
    assert [round(item["x"], 2) for item in route] == [0.0, 0.25, 0.5]
    assert loaded.coverage(reachable_count=6)["coverage_pct"] == 50.0


def test_world_map_refuses_unknown_target_instead_of_fake_arrival(tmp_path: Path):
    world_map = PersistentWorldMap("FloorPlan1", 0.25, tmp_path / "map.json")
    a, b = pos(0.0, 0.0), pos(0.25, 0.0)
    world_map.record_transition(a, b, True)
    assert world_map.route(a, pos(8.0, 8.0), interaction_distance=1.25) == []


def test_navigation_skill_checkpoint_round_trip_and_merge(tmp_path: Path):
    path = tmp_path / "nav.json"
    skill = NavigationSkill(path)
    state = skill.state(pos(0, 0), 0.0, pos(0.25, 0), grid_size=0.25)
    next_state = skill.state(pos(0, 0), 90.0, pos(0.25, 0), grid_size=0.25)
    skill.update(state, "RotateRight", 1.0, next_state)
    skill.save()

    loaded = NavigationSkill(path)
    assert loaded.transitions == 1
    assert loaded.visits[state.key()] == 1
    assert loaded.q[state.key()]["RotateRight"] > 0

    other = NavigationSkill()
    other.update(state, "RotateRight", 2.0, next_state)
    loaded.merge(other)
    assert loaded.transitions == 2
    assert loaded.visits[state.key()] == 2


def test_thor_motor_layer_uses_learned_map_without_exposing_it_to_llm(tmp_path: Path):
    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    world_map = PersistentWorldMap("FloorPlan1", 0.25, paths.scene_map("FloorPlan1"))
    a, b, c = pos(0.0, 0.0), pos(0.25, 0.0), pos(0.5, 0.0)
    world_map.record_transition(a, b, True)
    world_map.record_transition(b, c, True)
    world_map.save()
    NavigationSkill(paths.navigation_checkpoint).save()

    cfg = ThorConfig(
        scene="FloorPlan1",
        grid_size=0.25,
        learning_state_dir=str(paths.root),
        use_learned_navigation=True,
    )
    world = SimpleNamespace(
        pose=SimpleNamespace(position=a, rotation={"y": 90.0})
    )
    route, actions = _learned_route_actions(cfg, world, c, interaction_distance=0.1)
    assert len(route) == 3
    assert any(action.kind == "MoveAhead" for action in actions)
