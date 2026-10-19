from pathlib import Path
from types import SimpleNamespace
import json

import agent_core
from agent_core import thor_adapter
from agent_core.interaction_pose import InteractionPose
from agent_core.learning import LearningPaths, write_json_atomic
from agent_core.migrations import repair_v01021_place_route_admission_regression


def _held_world(x=0.0, z=0.0, yaw=0.0, held_id="Bread|held"):
    obj = SimpleNamespace(raw={"axisAlignedBoundingBox": {"size": {"x": 0.30, "z": 0.50}}})
    return SimpleNamespace(
        pose=SimpleNamespace(
            position={"x": x, "y": 0.9, "z": z},
            rotation={"y": yaw},
            horizon=0.0,
        ),
        held_object_id=held_id,
        objects={held_id: obj},
    )


def test_release_version_is_v01022():
    assert agent_core.__version__ == "0.10.24"


def test_carried_door_standoff_rejects_pose_too_close_to_closed_panel():
    world = _held_world()
    center = {"x": 0.0, "y": 1.0, "z": 0.0}
    near = InteractionPose(0.0, 0.9, 0.50, 180.0, 0.0, True)
    safe = InteractionPose(0.0, 0.9, 0.90, 180.0, 0.0, True)
    assert not thor_adapter._carried_door_pose_has_standoff(near, center, world, 0.25)
    assert thor_adapter._carried_door_pose_has_standoff(safe, center, world, 0.25)


def test_failed_carried_translation_blacklists_actual_attempted_cell():
    key = (12, 8)
    # yaw=270 faces west. MoveLeft is south, not whatever a body-only shortest
    # path happened to choose.
    assert thor_adapter._attempted_translation_key(key, 270.0, "MoveLeft") == (12, 7)
    assert thor_adapter._attempted_translation_key(key, 270.0, "MoveAhead") == (11, 8)
    assert thor_adapter._attempted_translation_key(key, 270.0, "RotateLeft") is None


def test_carried_door_filter_uses_actual_carried_planner_not_body_arrival_yaw():
    src = Path(thor_adapter.__file__).read_text(encoding="utf-8")
    block = src[src.index("def _choose_accessible_closed_door"):src.index("def _carried_level4_route_admission")]
    assert "_carried_safe_pose_actions" in block
    assert "arrives_aligned" not in block
    assert "_carried_door_pose_has_standoff" in block


def test_train_place_admits_carried_route_before_scored_stage():
    import agent_core.training as training
    src = Path(training.__file__).read_text(encoding="utf-8")
    block = src[src.index("def _train_place"):src.index("def _train_toggle")]
    assert "_carried_level4_route_admission" in block
    assert block.index("_carried_level4_route_admission") < block.index("stage_skill=\"place_approach\"")
    assert "return _skip(" in block


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


def test_v01021_route_admission_repair_matches_all_known_regression_classes_and_is_idempotent(tmp_path):
    paths = LearningPaths.from_root(tmp_path / "state").ensure()

    base = {
        "schema": "author.embodied.v01021",
        "agent": "A",
        "curriculum_skill": "place",
        "curriculum_level": 4,
        "stage_skill": "place_approach",
        "success": False,
        "target_object_type": "CounterTop",
        "primitive_steps": 0,
        "verification": {},
        "auxiliary_skills": [],
    }
    zero = dict(base)

    door = dict(base)
    door["primitive_steps"] = 9
    door["verification"] = {
        "reason": "blocking_door_approach_failed",
        "last_action_error": "DiscoveryDoor_NorthArchive is blocking the Agent from moving",
    }
    door["auxiliary_skills"] = [
        {"skill": "door_approach", "success": False, "primitive_steps": 9, "object_type": "Door"}
    ]

    cross = dict(base)
    cross["primitive_steps"] = 12
    cross["verification"] = {"reason": "no_interaction_pose_in_component", "last_action_error": None}
    cross["auxiliary_skills"] = [
        {"skill": "door_approach", "success": True, "primitive_steps": 4, "object_type": "Door"},
        {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
        {"skill": "door_cross", "success": False, "primitive_steps": 0, "object_type": "Door", "verification_basis": "cross_primitive_exception"},
    ]

    blocker = dict(base)
    blocker["primitive_steps"] = 30
    blocker["verification"] = {
        "reason": "pinned_pose_replans_exhausted",
        "last_action_error": "Fridge_x is blocking the Agent from moving by (-0.25, 0, 0)",
    }
    blocker["auxiliary_skills"] = [
        {"skill": "door_approach", "success": True, "primitive_steps": 5, "object_type": "Door"},
        {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
        {"skill": "door_cross", "success": True, "primitive_steps": 8, "object_type": "Door"},
    ]

    good = dict(base)
    good["success"] = True
    good["primitive_steps"] = 20
    good["verification"] = {"reason": "interaction_pose_reached", "last_action_error": None}

    rows = [zero, door, cross, blocker, good]
    paths.embodied_replay_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    checkpoint = paths.agent_skill_checkpoint("A")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "agent_label": "A",
        "skills": {
            "place_approach": _stats(5, 1, 4, 71),
            "route_resume_target": _stats(4, 1, 3, 71),
            "door_approach": _stats(4, 3, 1, 18),
            "door_cross": _stats(3, 2, 1, 8),
        },
        "by_object_type": {
            "place_approach": {"CounterTop": _stats(5, 1, 4, 71)},
            "route_resume_target": {"CounterTop": _stats(4, 1, 3, 71)},
            "door_approach": {"Door": _stats(4, 3, 1, 18)},
            "door_cross": {"Door": _stats(3, 2, 1, 8)},
        },
        "by_level": {
            "place_approach": {"4": _stats(5, 1, 4, 71)},
            "route_resume_target": {"4": _stats(4, 1, 3, 71)},
            "door_approach": {"4": _stats(4, 3, 1, 18)},
            "door_cross": {"4": _stats(3, 2, 1, 8)},
        },
    }
    write_json_atomic(checkpoint, payload)
    raw_before = paths.embodied_replay_file.read_text(encoding="utf-8")

    result = repair_v01021_place_route_admission_regression(paths.root)
    assert result["matched_place_route_admission_regressions"] == 4
    repaired = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert repaired["skills"]["place_approach"]["failures"] == 0
    assert repaired["skills"]["route_resume_target"]["failures"] == 0
    assert repaired["skills"]["route_resume_target"]["attempts"] == 1
    assert repaired["skills"]["door_approach"]["failures"] == 0
    assert repaired["skills"]["door_cross"]["failures"] == 0
    assert paths.embodied_replay_file.read_text(encoding="utf-8") == raw_before

    again = repair_v01021_place_route_admission_regression(paths.root)
    assert again["already_applied"] is True
