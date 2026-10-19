from __future__ import annotations

from types import SimpleNamespace

import agent_core
from agent_core.executor import Verifier
from agent_core.interaction_pose import InteractionPose
from agent_core.models import Action, ActionKind, AgentPose, ObjectState, WorldState
from agent_core.thor_adapter import ThorConfig, ThorExecution, _navigate_to_interactable_pose
from agent_core.training import _approach_params


def _world(*, x: float = 0.0, z: float = 0.0, yaw: float = 0.0, horizon: float = 0.0, distance: float = 2.1, visible: bool = True) -> WorldState:
    door = ObjectState(
        object_id="Door|test",
        object_type="Door",
        name="Door",
        visible=visible,
        position={"x": 2.4, "y": 0.0, "z": 0.8},
        distance=distance,
        openable=True,
        is_open=False,
    )
    return WorldState(
        scene="TestScene",
        pose=AgentPose(
            position={"x": x, "y": 0.9, "z": z},
            rotation={"x": 0.0, "y": yaw, "z": 0.0},
            horizon=horizon,
        ),
        objects={door.object_id: door},
        held_object_id=None,
        last_action="NavigateTo",
        last_action_success=True,
        last_error=None,
        frame_available=False,
    )


def test_release_version_is_v0104():
    assert agent_core.__version__ == "0.10.11"


def test_pose_authoritative_navigate_can_pass_beyond_object_center_radius():
    before = _world(distance=2.1)
    after = _world(x=1.0, z=0.0, yaw=90.0, distance=2.1)
    execution = ThorExecution(
        event=None,
        state=after,
        primitive_steps=4,
        primitive_actions=["MoveAhead"],
        goal_verification={
            "kind": "interaction_pose_goal",
            "objectId": "Door|test",
            "pose_goal_verified": True,
            "reason": "interaction_pose_reached",
        },
    )
    ok, details = Verifier(default_interaction_distance=1.5).verify(
        before,
        after,
        Action(ActionKind.NAVIGATE_TO.value, {"objectId": "Door|test", "interactionDistance": 1.25}),
        execution=execution,
    )
    assert ok
    assert details["verification_basis"] == "controller_interaction_pose"
    assert details["distance"] == 2.1
    assert "max_distance" not in details


def test_controller_pose_failure_overrides_near_object_center_distance():
    before = _world(distance=0.8)
    after = _world(distance=0.8)
    execution = ThorExecution(
        event=None,
        state=after,
        primitive_steps=2,
        primitive_actions=["RotateRight"],
        goal_verification={
            "kind": "interaction_pose_goal",
            "objectId": "Door|test",
            "pose_goal_verified": False,
            "reason": "interaction_pose_replans_exhausted",
        },
    )
    ok, details = Verifier().verify(
        before,
        after,
        Action(ActionKind.NAVIGATE_TO.value, {"objectId": "Door|test", "interactionDistance": 1.25}),
        execution=execution,
    )
    assert not ok
    assert details["verification_basis"] == "controller_interaction_pose"


def test_legacy_navigate_without_goal_evidence_keeps_compatibility_fallback():
    before = _world(distance=1.4)
    after = _world(distance=1.4)
    action = Action(ActionKind.NAVIGATE_TO.value, {"objectId": "Door|test", "interactionDistance": 1.25})
    ok, details = Verifier().verify(before, after, action)
    assert ok
    assert details["verification_basis"] == "legacy_object_center_fallback"
    assert details["max_distance"] == 1.55


def test_levels_1_to_3_pin_exact_setup_pose_only():
    adapter = SimpleNamespace(config=SimpleNamespace(interaction_distance=1.25))
    pose = InteractionPose(1.0, 0.9, 0.25, 90.0, -30.0, True)
    for level in (1, 2, 3):
        params = _approach_params(adapter, "Door|test", ActionKind.OPEN.value, level, pose)
        assert params["_targetInteractionPose"] == pose.as_dict()
    for level in (0, 4, 5):
        params = _approach_params(adapter, "Door|test", ActionKind.OPEN.value, level, pose)
        assert "_targetInteractionPose" not in params


def test_pinned_curriculum_pose_is_used_even_when_another_pose_is_cheaper(monkeypatch):
    pinned = InteractionPose(1.0, 0.9, 0.0, 90.0, 0.0, True)
    cheaper = InteractionPose(0.25, 0.9, 0.0, 0.0, 0.0, True)

    class FakeAdapter:
        def __init__(self):
            self.config = ThorConfig(
                scene="TestScene",
                use_interactable_poses=True,
                use_interaction_actionability=False,
                max_interaction_pose_replans=3,
                grid_size=0.25,
            )
            self.state = _world(x=0.0, z=0.0, distance=2.1)
            self.selected = []

        def observe(self):
            return self.state

        def raw_event(self):
            return SimpleNamespace(metadata={})

        def get_reachable_positions(self):
            return [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.75, "y": 0.9, "z": 0.0},
                {"x": 1.0, "y": 0.9, "z": 0.0},
            ]

        def get_interactable_poses(self, object_id, **kwargs):
            return [cheaper, pinned]

    adapter = FakeAdapter()

    def fake_execute_pose_goal(fake_adapter, pose, reachable, step_budget=None):
        fake_adapter.selected.append(pose)
        fake_adapter.state = _world(
            x=pose.x,
            z=pose.z,
            yaw=pose.rotation,
            horizon=pose.horizon,
            distance=2.1,
        )
        return ThorExecution(SimpleNamespace(metadata={}), fake_adapter.state, 1, ["MoveToPinnedPose"])

    monkeypatch.setattr("agent_core.thor_adapter._execute_pose_goal", fake_execute_pose_goal)

    result = _navigate_to_interactable_pose(
        adapter,
        Action(
            ActionKind.NAVIGATE_TO.value,
            {
                "objectId": "Door|test",
                "_interactionKind": ActionKind.OPEN.value,
                "_targetInteractionPose": pinned.as_dict(),
            },
            "curriculum pin",
        ),
        step_budget=20,
    )

    assert adapter.selected == [pinned]
    assert result.goal_verification["pose_goal_verified"] is True
    assert result.goal_verification["target_pose_key"] == list(pinned.key(0.25))
