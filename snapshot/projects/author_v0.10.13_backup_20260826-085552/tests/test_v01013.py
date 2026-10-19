from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import agent_core
import agent_core.training as training
from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.interaction_pose import InteractionPose
from agent_core.models import Action, ActionKind


def _pose_state(*, x: float = 0.0, z: float = 0.0, yaw: float = 0.0):
    return SimpleNamespace(
        position={"x": x, "y": 0.9, "z": z},
        rotation={"y": yaw},
        horizon=0.0,
    )


def test_release_version_is_v01013():
    assert agent_core.__version__ == "0.10.13"


def test_blocked_route_uses_maneuverable_start_and_path_aligned_yaw(monkeypatch):
    door = SimpleNamespace(
        object_id="Door|test",
        object_type="Door",
        openable=True,
        position={"x": 1.0, "y": 0.0, "z": 0.0},
    )
    world = SimpleNamespace(objects={door.object_id: door})
    g = 0.25
    adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=g, rotate_step_degrees=90.0),
        observe=lambda: world,
    )

    # Connected current-side component from x=1.0 through x=2.25.  The selected
    # educational start is x=2.0 and has lateral neighbors, so it is maneuverable.
    reachable = [{"x": x, "y": 0.9, "z": 0.0} for x in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25)]
    reachable += [
        {"x": 2.0, "y": 0.9, "z": 0.25},
        {"x": 2.0, "y": 0.9, "z": -0.25},
    ]
    full_reachable = list(reachable)
    target_pose = InteractionPose(x=0.0, y=0.9, z=0.0, rotation=0.0, horizon=0.0)
    door_pose = InteractionPose(x=1.0, y=0.9, z=0.0, rotation=270.0, horizon=0.0)

    adapter.get_reachable_positions = lambda: list(reachable)
    adapter.get_interactable_poses = lambda object_id, **_kwargs: (
        [] if object_id == "Apple|target" else [door_pose]
    )

    monkeypatch.setattr(training, "_set_all_door_state", lambda *a, **k: True)
    monkeypatch.setattr(training, "_admin_step", lambda *a, **k: True)
    monkeypatch.setattr(
        training,
        "_maneuverable_starts",
        lambda candidates, all_reachable, grid_size, min_neighbors=2: [
            next(p for p in candidates if p["x"] == 2.0 and p["z"] == 0.0)
        ],
    )

    teleports = []
    def fake_teleport(_shared, _active, start, yaw, horizon=0.0):
        teleports.append((dict(start), float(yaw), float(horizon)))
        return True
    monkeypatch.setattr(training, "_teleport", fake_teleport)

    ok, reason, component, blocking = training._prepare_blocked_route_start(
        shared=object(), adapter=adapter, active=0, rng=random.Random(1),
        target_object_id="Apple|target", target_pose=target_pose,
        full_reachable=full_reachable,
    )

    assert ok is True
    assert reason is None
    assert blocking == "Door|test"
    assert component == reachable
    assert len(teleports) == 2
    assert teleports[0][0]["x"] == 2.0
    assert teleports[0][1] == 0.0  # temporary administrative placement
    assert teleports[1][0]["x"] == 2.0
    assert teleports[1][1] == 270.0  # first verified path segment points toward -x


def _run_semantic_route_stage(monkeypatch, tmp_path: Path, auxiliary):
    target = SimpleNamespace(object_type="Apple")
    before = SimpleNamespace(
        pose=_pose_state(x=2.0),
        objects={"Apple|target": target},
        last_action_success=True,
        last_error=None,
    )
    after = SimpleNamespace(
        pose=_pose_state(x=0.25),
        objects={"Apple|target": target},
        last_action_success=True,
        last_error=None,
    )
    execution = SimpleNamespace(
        state=after,
        primitive_actions=["MoveAhead", "OpenObject", "MoveAhead"],
        auxiliary_skills=list(auxiliary),
        primitive_steps=9,
    )
    adapter = SimpleNamespace(observe=lambda: before, execute=lambda action, step_budget=None: execution)
    verifier = SimpleNamespace(verify=lambda b, a, action, execution=None: (True, {"kind": "interaction_pose_goal"}))
    library = EmbodiedSkillLibrary(agent_label="A")
    library.training_level = 4
    captured = {}
    monkeypatch.setattr(training, "_append_embodied_transition", lambda *a, **kw: captured.update(kw))

    result = training._execute_stage(
        adapter=adapter,
        action=Action(ActionKind.NAVIGATE_TO.value, {"objectId": "Apple|target"}, "semantic target approach"),
        verifier=verifier,
        replay_path=tmp_path / "replay.jsonl",
        scene="TestScene",
        agent_label="A",
        episode=1,
        curriculum_skill="pickup",
        stage_skill="pickup_approach",
        library=library,
        object_type="Apple",
        max_steps=40,
    )
    return result, library, captured


def test_level4_semantic_route_records_and_requires_resume(monkeypatch, tmp_path: Path):
    auxiliary = [
        {"skill": "door_approach", "success": True, "primitive_steps": 3, "object_type": "Door"},
        {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
        {"skill": "door_cross", "success": True, "primitive_steps": 2, "object_type": "Door"},
    ]
    (ok, steps, _primitives, verification), library, captured = _run_semantic_route_stage(monkeypatch, tmp_path, auxiliary)
    assert ok is True
    assert steps == 9
    assert verification["semantic_route_required"] is True
    assert verification["semantic_target_preserved"] is True
    assert verification["route_resume_target_verified"] is True
    assert [r["skill"] for r in captured["auxiliary_skills"]] == ["door_approach", "door_open", "door_cross"]
    stats = library.level_stats("route_resume_target", 4)
    assert stats.attempts == 1
    assert stats.successes == 1


def test_level4_semantic_route_fails_without_verified_cross(monkeypatch, tmp_path: Path):
    auxiliary = [
        {"skill": "door_approach", "success": True, "primitive_steps": 3, "object_type": "Door"},
        {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
        {"skill": "door_cross", "success": False, "primitive_steps": 0, "object_type": "Door"},
    ]
    (ok, _steps, _primitives, verification), library, _captured = _run_semantic_route_stage(monkeypatch, tmp_path, auxiliary)
    assert ok is False
    assert verification["route_resume_target_verified"] is False
    stats = library.level_stats("route_resume_target", 4)
    assert stats.attempts == 1
    assert stats.failures == 1
