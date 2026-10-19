from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_core.actionability import PoseActionabilityMemory
from agent_core.interaction_pose import InteractionPose
from agent_core.models import Action, ActionKind, AgentPose, ObjectState, WorldState
from agent_core.thor_adapter import ThorConfig, ThorExecution, _execute_interaction_with_pose_recovery


def _door_world(*, x: float, z: float, is_open: bool, success: bool | None = True, error: str | None = None):
    door = ObjectState(
        object_id="Door|test",
        object_type="Door",
        name="TestDoor",
        visible=True,
        position={"x": 1.0, "y": 0.0, "z": 0.0},
        distance=abs(1.0 - x),
        openable=True,
        is_open=is_open,
    )
    return WorldState(
        scene="TestScene",
        pose=AgentPose(
            position={"x": x, "y": 0.9, "z": z},
            rotation={"x": 0.0, "y": 90.0, "z": 0.0},
            horizon=0.0,
        ),
        objects={door.object_id: door},
        held_object_id=None,
        last_action="OpenObject",
        last_action_success=success,
        last_error=error,
        frame_available=False,
    )


def test_actionability_memory_persists_failed_and_successful_pose_outcomes(tmp_path: Path):
    path = tmp_path / "actionability.json"
    memory = PoseActionabilityMemory(path, agent_label="A", grid_size=0.25)
    bad = InteractionPose(0.0, 0.9, 0.0, 90.0, 0.0, True)
    good = InteractionPose(-0.25, 0.9, 0.0, 90.0, 0.0, True)

    memory.record("OpenObject", "Door|test", bad, success=False, error="blocked")
    memory.record("OpenObject", "Door|test", good, success=True, recovery=True)

    loaded = PoseActionabilityMemory(path, agent_label="A", grid_size=0.25)
    assert loaded.stats("OpenObject", "Door|test", bad).failures == 1
    assert loaded.stats("OpenObject", "Door|test", good).successes == 1
    assert loaded.stats("OpenObject", "Door|test", good).recoveries == 1
    assert loaded.penalty("OpenObject", "Door|test", bad, object_position={"x": 1.0, "y": 0.0, "z": 0.0}) > loaded.penalty(
        "OpenObject", "Door|test", good, object_position={"x": 1.0, "y": 0.0, "z": 0.0}
    )


def test_open_failure_repositions_and_retries_same_semantic_target(tmp_path: Path, monkeypatch):
    class FakeAdapter:
        def __init__(self):
            self.config = ThorConfig(
                scene="TestScene",
                learning_state_dir=str(tmp_path),
                use_interaction_actionability=True,
                max_interaction_action_retries=2,
                preferred_open_clearance_m=1.0,
                grid_size=0.25,
            )
            self.agent_label = "A"
            self._pose_actionability_memory = None
            self.state = _door_world(x=0.5, z=0.0, is_open=False)
            self.open_attempts = 0

        def observe(self):
            return self.state

        def raw_event(self):
            return SimpleNamespace(metadata={})

        def get_reachable_positions(self):
            return [
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": -0.25, "y": 0.9, "z": 0.0},
            ]

        def get_interactable_poses(self, object_id, **kwargs):
            assert object_id == "Door|test"
            return [
                InteractionPose(0.5, 0.9, 0.0, 90.0, 0.0, True),
                InteractionPose(-0.25, 0.9, 0.0, 90.0, 0.0, True),
            ]

        def _execute_primitive(self, action):
            assert action.kind == ActionKind.OPEN.value
            self.open_attempts += 1
            x = float(self.state.pose.position["x"])
            if x > 0.0:
                self.state = _door_world(
                    x=x,
                    z=0.0,
                    is_open=False,
                    success=False,
                    error="Object failed to open/close successfully.",
                )
            else:
                self.state = _door_world(x=x, z=0.0, is_open=True, success=True, error=None)
            return ThorExecution(SimpleNamespace(metadata={}), self.state, 1, [action.kind])

    adapter = FakeAdapter()

    def fake_pose_goal(fake_adapter, pose, reachable, step_budget=None):
        fake_adapter.state = _door_world(x=pose.x, z=pose.z, is_open=False, success=True, error=None)
        return ThorExecution(SimpleNamespace(metadata={}), fake_adapter.state, 1, ["MoveToRecoveryPose"])

    monkeypatch.setattr("agent_core.thor_adapter._execute_pose_goal", fake_pose_goal)
    monkeypatch.setattr("agent_core.thor_adapter._pose_goal_verified", lambda *args, **kwargs: True)

    action = Action(ActionKind.OPEN.value, {"objectId": "Door|test", "forceAction": False}, "open selected door")
    result = _execute_interaction_with_pose_recovery(adapter, action, step_budget=10)

    assert result.state.objects["Door|test"].is_open is True
    assert adapter.open_attempts == 2
    assert result.primitive_actions == ["OpenObject", "MoveToRecoveryPose", "OpenObject"]
    telemetry = list(result.auxiliary_skills or [])
    assert telemetry[0]["success"] is False
    assert telemetry[-1]["success"] is True
    assert telemetry[-1]["recovery"] is True

    memory = PoseActionabilityMemory(
        tmp_path / "checkpoints" / "a" / "interaction_actionability.json",
        agent_label="A",
        grid_size=0.25,
    )
    bad = InteractionPose(0.5, 0.9, 0.0, 90.0, 0.0, True)
    good = InteractionPose(-0.25, 0.9, 0.0, 90.0, 0.0, True)
    assert memory.stats("OpenObject", "Door|test", bad).failures == 1
    assert memory.stats("OpenObject", "Door|test", good).successes == 1


def test_actionability_delta_does_not_duplicate_seeded_history(tmp_path: Path):
    seed_path = tmp_path / "seed.json"
    live_path = tmp_path / "live.json"
    pose = InteractionPose(0.0, 0.9, 0.0, 90.0, 0.0, True)

    seed = PoseActionabilityMemory(seed_path, agent_label="A")
    seed.record("OpenObject", "Door|test", pose, success=False, error="blocked")

    live_path.write_text(seed_path.read_text())
    live = PoseActionabilityMemory(live_path, agent_label="A")
    live.record("OpenObject", "Door|test", pose, success=True, recovery=True)

    delta = live.delta_from(seed)
    stats = delta.stats("OpenObject", "Door|test", pose)
    assert stats.attempts == 1
    assert stats.successes == 1
    assert stats.failures == 0
    assert stats.recoveries == 1


def test_pose_penalty_can_prefer_safer_actionable_pose_over_nearest_candidate():
    from agent_core.interaction_pose import choose_best_interaction_pose
    from agent_core.navigation import NavigationConfig

    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 0.5, "y": 0.9, "z": 0.0},
    ]
    near = InteractionPose(0.25, 0.9, 0.0, 90.0, 0.0, True)
    safe = InteractionPose(0.5, 0.9, 0.0, 90.0, 0.0, True)
    chosen = choose_best_interaction_pose(
        [near, safe],
        reachable,
        reachable[0],
        90.0,
        0.0,
        NavigationConfig(grid_size=0.25),
        pose_penalty=lambda pose: 100.0 if pose == near else 0.0,
    )
    assert chosen == safe
