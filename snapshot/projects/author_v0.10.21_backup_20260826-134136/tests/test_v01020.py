from types import SimpleNamespace
import json

import agent_core
from agent_core import thor_adapter
from agent_core.navigation import NavigationConfig, path_to_actions


def test_release_version_is_v01020():
    assert agent_core.__version__ == "0.10.21"


def test_path_to_actions_can_suppress_staging_zero_vector_turn():
    path = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
    ]
    cfg = NavigationConfig(grid_size=0.25, rotate_step_degrees=90.0, interaction_distance=0.1)
    actions = path_to_actions(path, 90.0, path[-1], cfg, face_target=False)
    assert [a.kind for a in actions] == ["MoveAhead"]
    # Historical generic behavior tries to face a zero-vector target and inserts
    # a spurious turn back toward yaw=0 at the exact staging cell.
    generic = path_to_actions(path, 90.0, path[-1], cfg)
    assert [a.kind for a in generic] == ["MoveAhead", "RotateLeft"]


def test_carried_cross_waypoints_use_same_center_lane_and_clearance_depth():
    grid = 0.25
    center = {"x": 2.4, "y": 0.0, "z": 0.40}
    source_rows = [
        {"x": x, "y": 0.9, "z": z}
        for x in (2.75, 3.0, 3.25, 3.5, 3.75)
        for z in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    ]
    outside_rows = [
        {"x": x, "y": 0.9, "z": z}
        for x in (2.0, 1.75, 1.5, 1.25)
        for z in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    ]
    reachable = source_rows + outside_rows
    source_keys = {thor_adapter._grid_position_key(p, grid) for p in source_rows}
    staging, target = thor_adapter._choose_cross_threshold_waypoints(
        {"x": 5.0, "y": 0.9, "z": 1.25},
        center,
        reachable,
        source_component_keys=source_keys,
        grid_size=grid,
        clearance_depth=1.0,
    )
    assert staging is not None and target is not None
    assert round(staging["z"] / grid) == round(target["z"] / grid)
    assert abs(staging["z"] - center["z"]) <= 0.15
    assert abs(target["z"] - center["z"]) <= 0.15
    # east-room source -> base-room target, so staging is safely east of x=2.4
    assert staging["x"] >= 3.25
    assert target["x"] <= 1.75


def test_held_object_clearance_depth_scales_with_footprint():
    small = SimpleNamespace(raw={"axisAlignedBoundingBox": {"size": {"x": 0.1, "z": 0.1}}})
    large = SimpleNamespace(raw={"axisAlignedBoundingBox": {"size": {"x": 1.0, "z": 1.0}}})
    world_small = SimpleNamespace(held_object_id="small", objects={"small": small})
    world_large = SimpleNamespace(held_object_id="large", objects={"large": large})
    assert thor_adapter._held_object_clearance_depth(world_small, 0.25) == 0.85
    assert thor_adapter._held_object_clearance_depth(world_large, 0.25) == 1.25


def test_cross_threshold_source_has_no_portal_zone_turns():
    text = open(thor_adapter.__file__, encoding="utf-8").read()
    block = text[text.index("def _cross_threshold_for_adapter"):text.index("class _AdapterMixin")]
    assert "face_target=False" in block
    assert "Carry straight through aligned portal lane." in block
    assert "straight_actions" in block


def test_v01019_place_transport_repair_is_idempotent_and_preserves_raw_replay(tmp_path):
    from agent_core.learning import LearningPaths, write_json_atomic
    from agent_core.migrations import repair_v01019_place_transport_regression

    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    bad = {
        "schema": "author.embodied.v01019",
        "agent": "A",
        "curriculum_skill": "place",
        "curriculum_level": 4,
        "stage_skill": "place_approach",
        "success": False,
        "primitive_steps": 13,
        "target_object_type": "CounterTop",
        "auxiliary_skills": [
            {"skill": "door_approach", "success": True, "primitive_steps": 4, "object_type": "Door"},
            {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
            {"skill": "door_cross", "success": False, "primitive_steps": 4, "object_type": "Door", "verification_basis": "closed_component_transition"},
        ],
        "verification": {"last_action_error": "EastPortalBoundary_Lower is blocking the Agent with Mug_x"},
    }
    good = dict(bad)
    good["success"] = True
    good["verification"] = {"last_action_error": None}
    good["auxiliary_skills"] = [{"skill": "door_cross", "success": True, "primitive_steps": 4, "object_type": "Door"}]
    paths.embodied_replay_file.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")

    checkpoint = paths.agent_skill_checkpoint("A")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def row(attempts, successes, failures, steps):
        return {"attempts": attempts, "successes": successes, "failures": failures, "primitive_steps": steps,
                "recovery_attempts": 0, "recovery_successes": 0, "recovery_exhausted": 0}

    payload = {
        "version": 3,
        "agent_label": "A",
        "skills": {
            "place_approach": row(5, 4, 1, 50),
            "route_resume_target": row(5, 4, 1, 50),
            "door_cross": row(5, 4, 1, 20),
        },
        "by_object_type": {
            "place_approach": {"CounterTop": row(5, 4, 1, 50)},
            "route_resume_target": {"CounterTop": row(5, 4, 1, 50)},
            "door_cross": {"Door": row(5, 4, 1, 20)},
        },
        "by_level": {
            "place_approach": {"4": row(5, 4, 1, 50)},
            "route_resume_target": {"4": row(5, 4, 1, 50)},
            "door_cross": {"4": row(5, 4, 1, 20)},
        },
    }
    write_json_atomic(checkpoint, payload)
    raw_before = paths.embodied_replay_file.read_text()

    result = repair_v01019_place_transport_regression(paths.root)
    assert result["matched_place_transport_regressions"] == 1
    repaired = json.loads(checkpoint.read_text())
    assert repaired["skills"]["place_approach"]["attempts"] == 4
    assert repaired["skills"]["route_resume_target"]["failures"] == 0
    assert repaired["skills"]["door_cross"]["failures"] == 0
    assert paths.embodied_replay_file.read_text() == raw_before
    again = repair_v01019_place_transport_regression(paths.root)
    assert again["already_applied"] is True
