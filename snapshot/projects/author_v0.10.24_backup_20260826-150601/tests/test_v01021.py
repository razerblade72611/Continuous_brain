from types import SimpleNamespace

import agent_core
from agent_core import thor_adapter
from agent_core.interaction_pose import InteractionPose


def _world(x=0.0, z=0.0, yaw=0.0, held=True):
    held_id = "Vase|held" if held else None
    obj = SimpleNamespace(raw={"axisAlignedBoundingBox": {"size": {"x": 0.25, "z": 0.25}}})
    return SimpleNamespace(
        pose=SimpleNamespace(position={"x": x, "y": 0.9, "z": z}, rotation={"y": yaw}),
        held_object_id=held_id,
        objects=({held_id: obj} if held_id else {}),
    )


def test_release_version_is_v01021():
    assert agent_core.__version__ == "0.10.24"


def test_blocker_detection_accepts_unity_the_agent_wording():
    assert thor_adapter._movement_blocked_error(
        "Fridge_abc is blocking the Agent from moving by (-0.25, 0, 0)"
    )
    assert thor_adapter._movement_blocked_error(
        "EastPortalBoundary_Lower is blocking the Agent from moving"
    )
    assert not thor_adapter._movement_blocked_error(
        "a held item: Vase_x with something if agent rotates Right 90 degrees"
    )


def test_carried_door_pose_rejects_jamb_lane_and_accepts_center_lane():
    # East door: hinge z≈0.825, architectural center z=0.40.  The failing v0.10.20
    # approaches used z=-0.75, far outside the 0.85 m opening.
    center = {"x": 2.4, "y": 1.0, "z": 0.40}
    hinge = {"x": 2.4, "y": 0.0, "z": 0.825}
    bad = InteractionPose(3.25, 0.9, -0.75, 270.0, 0.0, True)
    good = InteractionPose(3.25, 0.9, 0.25, 270.0, 0.0, True)
    assert not thor_adapter._carried_door_pose_is_centered(bad, center, hinge)
    assert thor_adapter._carried_door_pose_is_centered(good, center, hinge)


def test_carried_pose_route_moves_target_aligned_without_rotating_at_tight_target():
    grid = 0.25
    # Open 5x5 bay feeding a one-cell east spur. The target spur itself cannot
    # satisfy turn clearance, so a safe route must turn in the bay and translate
    # into the target already at yaw 90.
    reachable = [
        {"x": x * grid, "y": 0.9, "z": z * grid}
        for x in range(-2, 3)
        for z in range(-2, 3)
    ]
    reachable += [
        {"x": 3 * grid, "y": 0.9, "z": 0.0},
        {"x": 4 * grid, "y": 0.9, "z": 0.0},
    ]
    world = _world(0.0, 0.0, 0.0, held=True)
    pose = InteractionPose(1.0, 0.9, 0.0, 90.0, 0.0, True)
    actions = thor_adapter._carried_safe_pose_actions(
        world,
        reachable,
        pose,
        grid_size=grid,
        rotate_step_degrees=90.0,
    )
    assert actions is not None
    assert actions
    assert actions[-1].kind in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
    # At least one orientation change is needed, but it must happen before entry
    # into the tight target spur rather than at the final interaction cell.
    turn_indices = [i for i, a in enumerate(actions) if a.kind.startswith("Rotate")]
    assert turn_indices
    assert max(turn_indices) < len(actions) - 1


def test_pinned_preflight_pose_can_be_reused_when_interactable_query_omits_it():
    text = open(thor_adapter.__file__, encoding="utf-8").read()
    assert "pinned_pose_reused_from_preflight" in text
    assert "navigate there anyway and let authoritative pose verification" in text


def test_carried_door_selection_has_no_unsafe_generic_fallback():
    text = open(thor_adapter.__file__, encoding="utf-8").read()
    block = text[text.index("def _choose_accessible_closed_door"):text.index("def _navigate_to_interactable_pose")]
    assert "_carried_door_pose_is_centered" in block
    assert "if pose is None and world.held_object_id is None" in block


def test_v01020_place_carry_repair_is_idempotent_and_preserves_raw_replay(tmp_path):
    import json
    from agent_core.learning import LearningPaths, write_json_atomic
    from agent_core.migrations import repair_v01020_place_carry_route_regression

    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    bad = {
        "schema": "author.embodied.v01020",
        "agent": "B",
        "curriculum_skill": "place",
        "curriculum_level": 4,
        "stage_skill": "place_approach",
        "success": False,
        "primitive_steps": 30,
        "target_object_type": "CounterTop",
        "auxiliary_skills": [
            {"skill": "door_approach", "success": True, "primitive_steps": 5, "object_type": "Door"},
            {"skill": "door_open", "success": True, "primitive_steps": 1, "object_type": "Door"},
            {"skill": "door_cross", "success": True, "primitive_steps": 8, "object_type": "Door"},
        ],
        "verification": {
            "reason": "pinned_pose_replans_exhausted",
            "last_action_error": "a held item: Vase_x with something if agent rotates Left 90 degrees",
        },
    }
    good = dict(bad)
    good["success"] = True
    good["verification"] = {"reason": "interaction_pose_reached", "last_action_error": None}
    paths.embodied_replay_file.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")

    def row(attempts, successes, failures, steps):
        return {"attempts": attempts, "successes": successes, "failures": failures, "primitive_steps": steps,
                "recovery_attempts": 0, "recovery_successes": 0, "recovery_exhausted": 0}

    checkpoint = paths.agent_skill_checkpoint("B")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "agent_label": "B",
        "skills": {
            "place_approach": row(6, 5, 1, 90),
            "route_resume_target": row(6, 5, 1, 90),
            "door_cross": row(6, 6, 0, 48),
        },
        "by_object_type": {
            "place_approach": {"CounterTop": row(6, 5, 1, 90)},
            "route_resume_target": {"CounterTop": row(6, 5, 1, 90)},
            "door_cross": {"Door": row(6, 6, 0, 48)},
        },
        "by_level": {
            "place_approach": {"4": row(6, 5, 1, 90)},
            "route_resume_target": {"4": row(6, 5, 1, 90)},
            "door_cross": {"4": row(6, 6, 0, 48)},
        },
    }
    write_json_atomic(checkpoint, payload)
    raw_before = paths.embodied_replay_file.read_text()
    result = repair_v01020_place_carry_route_regression(paths.root)
    assert result["matched_place_carry_route_regressions"] == 1
    repaired = json.loads(checkpoint.read_text())
    assert repaired["skills"]["place_approach"]["failures"] == 0
    assert repaired["skills"]["route_resume_target"]["failures"] == 0
    assert repaired["skills"]["door_cross"]["attempts"] == 6
    assert paths.embodied_replay_file.read_text() == raw_before
    again = repair_v01020_place_carry_route_regression(paths.root)
    assert again["already_applied"] is True
