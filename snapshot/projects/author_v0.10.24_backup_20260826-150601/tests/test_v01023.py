from pathlib import Path
from types import SimpleNamespace
import json

import agent_core
import agent_core.training as training
from agent_core.interaction_pose import InteractionPose
from agent_core.learning import LearningPaths, write_json_atomic
from agent_core.migrations import repair_v01022_place_destination_regression


def test_release_version_is_v01023():
    assert agent_core.__version__ == "0.10.24"


def test_pinned_preflight_terminal_is_injected_only_when_target_component_has_poses():
    src = Path(training.__file__).with_name("thor_adapter.py").read_text(encoding="utf-8")
    block = src[src.index("if pinned_pose is not None:"):src.index("else:\n            pose = choose_best_interaction_pose", src.index("if pinned_pose is not None:"))]
    assert "pinned_pose_terminal_injected_from_preflight" in block
    assert "elif poses:" in block
    assert "reachable = list(reachable) + [trusted_terminal]" in block
    assert "a closed door cannot be bypassed by the pin" in block


def test_train_place_physically_preflights_destination_leg_before_scored_approach():
    src = Path(training.__file__).read_text(encoding="utf-8")
    block = src[src.index("def _train_place"):src.index("def _train_toggle")]
    assert "_carried_level4_destination_leg_preflight" in block
    assert block.index("_carried_level4_destination_leg_preflight") < block.index('stage_skill="place_approach"')
    assert "destination leg failed physical preflight" in block


def test_destination_leg_preflight_uses_unity_route_and_restores_start_and_door(monkeypatch):
    door = SimpleNamespace(
        object_id="Door|north",
        object_type="Door",
        openable=True,
        is_open=False,
        position={"x": 0.0, "y": 0.0, "z": 2.5},
        raw={"axisAlignedBoundingBox": {"center": {"x": 0.0, "y": 1.0, "z": 2.5}}},
    )
    target = SimpleNamespace(object_id="CounterTop|1", object_type="CounterTop")
    pose_state = SimpleNamespace(
        position={"x": 0.0, "y": 0.9, "z": 4.0},
        rotation={"y": 180.0},
        horizon=0.0,
        standing=True,
    )
    world = SimpleNamespace(
        held_object_id="PaperTowelRoll|1",
        pose=pose_state,
        objects={door.object_id: door, target.object_id: target},
        last_error=None,
    )

    class Adapter:
        config = SimpleNamespace(grid_size=0.25, rotate_step_degrees=90.0)
        def observe(self):
            return world
        def get_reachable_positions(self):
            if door.is_open:
                return [
                    {"x": 0.0, "y": 0.9, "z": z * 0.25}
                    for z in range(4, 21)
                ] + [{"x": -1.0, "y": 0.9, "z": 1.25}]
            return [{"x": 0.0, "y": 0.9, "z": z * 0.25} for z in range(12, 21)]

    adapter = Adapter()
    teleports = []

    def fake_admin(_shared, _active, action, **kwargs):
        if action == "OpenObject":
            door.is_open = True
        elif action == "CloseObject":
            door.is_open = False
        return True

    def fake_teleport(_shared, _active, pose):
        world.pose.position = dict(pose.position)
        world.pose.rotation = {"y": float(pose.rotation)}
        world.pose.horizon = float(pose.horizon)
        teleports.append((dict(pose.position), float(pose.rotation)))
        return True

    monkeypatch.setattr(training, "_admin_step", fake_admin)
    monkeypatch.setattr(training, "_teleport_pose", fake_teleport)
    monkeypatch.setattr(training, "_doorway_center_from_closed_door", lambda _door: {"x": 0.0, "y": 1.0, "z": 2.5})
    monkeypatch.setattr(training, "_held_object_clearance_depth", lambda _world, _grid: 0.85)
    monkeypatch.setattr(
        training,
        "_choose_cross_threshold_waypoints",
        lambda *args, **kwargs: (
            {"x": 0.0, "y": 0.9, "z": 3.25},
            {"x": 0.0, "y": 0.9, "z": 1.75},
        ),
    )
    monkeypatch.setattr(
        training,
        "_execute_pose_goal",
        lambda *_args, **_kwargs: SimpleNamespace(goal_verification={"reason": "interaction_pose_reached"}),
    )
    monkeypatch.setattr(training, "_pose_goal_verified", lambda *_args, **_kwargs: True)

    target_pose = InteractionPose(-1.0, 0.9, 1.25, 270.0, 0.0, True)
    ok, reason = training._carried_level4_destination_leg_preflight(
        object(), adapter, 0, target, target_pose, door.object_id, max_steps=140
    )
    assert ok is True
    assert reason is None
    assert door.is_open is False
    assert world.pose.position == {"x": 0.0, "y": 0.9, "z": 4.0}
    assert world.pose.rotation["y"] == 180.0
    assert len(teleports) >= 2  # far-side landing, then exact start restore


def _stats(attempts, successes, failures, steps):
    return {
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "primitive_steps": steps,
        "recovery_attempts": 0,
        "recovery_successes": 0,
        "recovery_exhausted": 0,
    }


def test_v01022_destination_repair_matches_two_known_classes_and_is_idempotent(tmp_path):
    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    common = {
        "schema": "author.embodied.v01022",
        "agent": "A",
        "curriculum_skill": "place",
        "curriculum_level": 4,
        "stage_skill": "place_approach",
        "success": False,
        "target_object_type": "CounterTop",
        "primitive_steps": 28,
        "auxiliary_skills": [
            {"skill": "door_approach", "success": True, "primitive_steps": 6, "object_type": "Door"},
            {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
            {"skill": "door_cross", "success": True, "primitive_steps": 9, "object_type": "Door", "verification_basis": "closed_component_transition"},
        ],
    }
    blocker = dict(common)
    blocker["verification"] = {
        "reason": "pinned_pose_replans_exhausted",
        "last_action_error": "Fridge_x is blocking the Agent from moving by (-0.25, 0, 0) with PaperTowelRoll_x",
    }
    omission = dict(common)
    omission["agent"] = "B"
    omission["primitive_steps"] = 18
    omission["verification"] = {"reason": "pinned_pose_not_available", "last_action_error": None}
    good = dict(common)
    good["success"] = True
    good["verification"] = {"reason": "interaction_pose_reached", "last_action_error": None}
    paths.embodied_replay_file.write_text("".join(json.dumps(x) + "\n" for x in (blocker, omission, good)), encoding="utf-8")

    for agent, steps in (("A", 28), ("B", 18)):
        checkpoint = paths.agent_skill_checkpoint(agent)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 3,
            "agent_label": agent,
            "skills": {
                "place_approach": _stats(2, 1, 1, 50 + steps),
                "route_resume_target": _stats(2, 1, 1, 50 + steps),
                "door_cross": _stats(2, 2, 0, 20),
            },
            "by_object_type": {
                "place_approach": {"CounterTop": _stats(2, 1, 1, 50 + steps)},
                "route_resume_target": {"CounterTop": _stats(2, 1, 1, 50 + steps)},
                "door_cross": {"Door": _stats(2, 2, 0, 20)},
            },
            "by_level": {
                "place_approach": {"4": _stats(2, 1, 1, 50 + steps)},
                "route_resume_target": {"4": _stats(2, 1, 1, 50 + steps)},
                "door_cross": {"4": _stats(2, 2, 0, 20)},
            },
        }
        write_json_atomic(checkpoint, payload)

    raw_before = paths.embodied_replay_file.read_text(encoding="utf-8")
    result = repair_v01022_place_destination_regression(paths.root)
    assert result["matched_place_destination_regressions"] == 2
    assert result["raw_replay_preserved"] is True
    for agent in ("A", "B"):
        repaired = json.loads(paths.agent_skill_checkpoint(agent).read_text(encoding="utf-8"))
        assert repaired["skills"]["place_approach"]["failures"] == 0
        assert repaired["skills"]["route_resume_target"]["failures"] == 0
        assert repaired["skills"]["door_cross"]["attempts"] == 2
    assert paths.embodied_replay_file.read_text(encoding="utf-8") == raw_before
    again = repair_v01022_place_destination_regression(paths.root)
    assert again["already_applied"] is True
