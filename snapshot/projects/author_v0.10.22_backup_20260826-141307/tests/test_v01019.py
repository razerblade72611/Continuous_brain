from types import SimpleNamespace

import agent_core
from agent_core import thor_adapter, training
from agent_core.interaction_pose import InteractionPose


def _state(x, z):
    return SimpleNamespace(pose=SimpleNamespace(position={"x": x, "y": 0.9, "z": z}))


def test_release_version_is_v01019():
    assert agent_core.__version__ == "0.10.22"


def test_closed_doorway_center_uses_aabb_not_hinge():
    door = SimpleNamespace(
        position={"x": 2.4, "y": 0.0, "z": 0.825},
        raw={"axisAlignedBoundingBox": {"center": {"x": 2.4, "y": 1.0, "z": 0.40}}},
    )
    center = thor_adapter._doorway_center_from_closed_door(door)
    assert center["x"] == 2.4
    assert center["z"] == 0.40


def test_cross_waypoints_prefer_portal_center_not_hinge_boundary():
    grid = 0.25
    center = {"x": 2.4, "y": 0.0, "z": 0.40}
    # east-room source side x >= 2.75; base-room side x <= 2.0
    source_rows = [
        {"x": x, "y": 0.9, "z": z}
        for x in (2.75, 3.0, 3.25)
        for z in (0.25, 0.50, 0.75, 1.00, 1.25)
    ]
    outside_rows = [
        {"x": x, "y": 0.9, "z": z}
        for x in (2.0, 1.75)
        for z in (0.25, 0.50, 0.75, 1.00, 1.25)
    ]
    reachable = source_rows + outside_rows
    source_keys = {thor_adapter._grid_position_key(p, grid) for p in source_rows}
    staging, target = thor_adapter._choose_cross_threshold_waypoints(
        {"x": 3.0, "y": 0.9, "z": 1.25},
        center,
        reachable,
        source_component_keys=source_keys,
        grid_size=grid,
    )
    assert staging is not None and target is not None
    # The old hinge-biased route ran at z≈1.25 and hit EastPortalBoundary_Upper.
    assert abs(staging["z"] - center["z"]) <= 0.15
    assert abs(target["z"] - center["z"]) <= 0.15
    assert thor_adapter._grid_position_key(target, grid) not in source_keys


def test_post_route_component_transition_promotes_provisional_cross():
    grid = 0.25
    row = {"skill": "door_cross", "success": False, "verification_basis": "closed_component_transition"}
    source_keys = {(12, 2), (11, 2), (10, 2)}
    thor_adapter._promote_pending_crosses(_state(1.0, 0.5), [(row, source_keys)], grid)
    assert row["success"] is True
    assert row["recovery"] is True
    assert row["verification_basis"] == "post_route_component_transition"


def test_place_pose_preflight_uses_scored_forceaction_false_and_caches(monkeypatch):
    pose = InteractionPose(1.0, 0.9, 1.0, 180.0, 30.0, True)
    source = SimpleNamespace(
        object_id="Plate|x",
        name="Plate_1",
        position={"x": 0.0, "y": 1.0, "z": 0.0},
        raw={"rotation": {"x": 0.0, "y": 0.0, "z": 0.0}},
        parent_receptacles=["CounterTop|x"],
    )
    target_state = SimpleNamespace(object_id="CounterTop|x", openable=False, is_open=None)
    placed = SimpleNamespace(parent_receptacles=["CounterTop|x"])
    world = SimpleNamespace(
        objects={"Plate|x": placed, "CounterTop|x": target_state},
        held_object_id=None,
        last_error=None,
    )

    class Adapter:
        config = SimpleNamespace(grid_size=0.25)
        def observe(self):
            return world

    adapter = Adapter()
    target = target_state
    calls = []

    monkeypatch.setattr(training, "_teleport_pose", lambda *a, **k: True)
    monkeypatch.setattr(training, "_ensure_empty_hand", lambda *a, **k: True)
    monkeypatch.setattr(training, "_restore_discovery_props", lambda *a, **k: (True, None, 1))

    def fake_admin(shared, active, action, **kwargs):
        calls.append((action, dict(kwargs)))
        return True

    monkeypatch.setattr(training, "_admin_step", fake_admin)
    ok, reason = training._place_pose_preflight(object(), adapter, 0, source, target, pose)
    assert ok and reason is None
    put = next(kwargs for action, kwargs in calls if action == "PutObject")
    assert put["forceAction"] is False

    calls.clear()
    ok2, reason2 = training._place_pose_preflight(object(), adapter, 0, source, target, pose)
    assert ok2 and reason2 is None
    assert calls == []


def test_level4_place_pins_preflighted_interaction_pose():
    text = open(training.__file__, encoding="utf-8").read()
    assert 'approach_params["_targetInteractionPose"] = pose.as_dict()' in text
    assert "_place_pose_preflight" in text


def test_v01018_place_regression_repair_is_idempotent_and_preserves_replay(tmp_path):
    from agent_core.learning import LearningPaths, write_json_atomic
    from agent_core.migrations import repair_v01018_place_regression

    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    replay_rows = [
        {
            "schema":"author.embodied.v01018","agent":"A","curriculum_skill":"place","curriculum_level":4,
            "stage_skill":"place_approach","success":False,"primitive_steps":20,"target_object_type":"CounterTop",
            "auxiliary_skills":[{"skill":"door_cross","success":False,"primitive_steps":6,"object_type":"Door"}],
            "verification":{},
        },
        {
            "schema":"author.embodied.v01018","agent":"A","curriculum_skill":"place","curriculum_level":4,
            "stage_skill":"place_release","success":False,"primitive_steps":9,"target_object_type":"CounterTop",
            "auxiliary_skills":[],
            "verification":{"last_action_error":"No valid positions to place object found","interaction_recovery_used":True,"interaction_recovery_exhausted":True},
        },
    ]
    paths.embodied_replay_file.write_text("\n".join(__import__('json').dumps(r) for r in replay_rows)+"\n")
    checkpoint = paths.agent_skill_checkpoint("A")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    def row(attempts, successes, failures, steps, recovery_attempts=0, recovery_exhausted=0):
        return {"attempts":attempts,"successes":successes,"failures":failures,"primitive_steps":steps,
                "recovery_attempts":recovery_attempts,"recovery_successes":0,"recovery_exhausted":recovery_exhausted}
    payload = {
        "version":3,"agent_label":"A",
        "skills":{
            "place_approach":row(10,8,2,200),
            "route_resume_target":row(10,8,2,200),
            "door_cross":row(10,8,2,100),
            "place_release":row(10,8,2,90,1,1),
        },
        "by_object_type":{
            "place_approach":{"CounterTop":row(5,4,1,100)},
            "route_resume_target":{"CounterTop":row(5,4,1,100)},
            "door_cross":{"Door":row(5,4,1,50)},
            "place_release":{"CounterTop":row(5,4,1,45,1,1)},
        },
        "by_level":{
            "place_approach":{"4":row(5,4,1,100)},
            "route_resume_target":{"4":row(5,4,1,100)},
            "door_cross":{"4":row(5,4,1,50)},
            "place_release":{"4":row(5,4,1,45,1,1)},
        },
    }
    write_json_atomic(checkpoint, payload)
    raw_before = paths.embodied_replay_file.read_text()
    result = repair_v01018_place_regression(paths.root)
    assert result["matched_place_approach_cross_regressions"] == 1
    assert result["matched_invalid_place_releases"] == 1
    repaired = __import__('json').loads(checkpoint.read_text())
    assert repaired["skills"]["place_approach"]["attempts"] == 9
    assert repaired["skills"]["door_cross"]["failures"] == 1
    assert repaired["skills"]["place_release"]["attempts"] == 9
    assert repaired["skills"]["place_release"]["recovery_exhausted"] == 0
    assert paths.embodied_replay_file.read_text() == raw_before
    again = repair_v01018_place_regression(paths.root)
    assert again["already_applied"] is True
