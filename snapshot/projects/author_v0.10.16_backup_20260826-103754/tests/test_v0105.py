from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import agent_core
from agent_core.migrations import repair_v0104_missing_cross_threshold
from agent_core.models import Action, ActionKind, AgentPose, ObjectState, WorldState
from agent_core.thor_adapter import ThorConfig, ThorExecution, _AdapterMixin, _crossed_door


def _open_door_world(*, x: float, z: float, yaw: float = 0.0) -> WorldState:
    door = ObjectState(
        object_id="Door|test",
        object_type="Door",
        name="Door",
        visible=True,
        position={"x": 0.0, "y": 0.0, "z": 1.0},
        distance=((x - 0.0) ** 2 + (z - 1.0) ** 2) ** 0.5,
        openable=True,
        is_open=True,
    )
    return WorldState(
        scene="TestScene",
        pose=AgentPose(
            position={"x": x, "y": 0.9, "z": z},
            rotation={"x": 0.0, "y": yaw, "z": 0.0},
            horizon=0.0,
        ),
        objects={door.object_id: door},
        held_object_id=None,
        last_action="Pass",
        last_action_success=True,
        last_error=None,
        frame_available=False,
    )


def test_release_version_is_v0105():
    assert agent_core.__version__ == "0.10.16"


def test_live_adapter_dispatch_executes_cross_threshold(monkeypatch):
    class FakeAdapter(_AdapterMixin):
        def __init__(self):
            self.config = ThorConfig(scene="TestScene", grid_size=0.25, rotate_step_degrees=90.0)
            self.state = _open_door_world(x=0.0, z=0.0, yaw=0.0)

        def observe(self):
            return self.state

        def raw_event(self):
            return SimpleNamespace(metadata={})

        def get_reachable_positions(self):
            return [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.0, "y": 0.9, "z": 0.50},
                {"x": 0.0, "y": 0.9, "z": 0.75},
                {"x": 0.0, "y": 0.9, "z": 1.00},
                {"x": 0.0, "y": 0.9, "z": 1.25},
                {"x": 0.0, "y": 0.9, "z": 1.50},
                {"x": 0.0, "y": 0.9, "z": 1.75},
                {"x": 0.0, "y": 0.9, "z": 2.00},
            ]

        def _execute_primitive(self, action):
            # Keep the test focused on live dispatch + crossing target execution;
            # path geometry itself is separately tested in test_v010.py.
            if action.kind == ActionKind.MOVE_AHEAD.value:
                pos = dict(self.state.pose.position)
                pos["z"] = float(pos["z"]) + 0.25
                self.state = _open_door_world(x=float(pos["x"]), z=float(pos["z"]), yaw=float(self.state.pose.rotation.get("y", 0.0)))
            elif action.kind == ActionKind.ROTATE_RIGHT.value:
                yaw = (float(self.state.pose.rotation.get("y", 0.0)) + 90.0) % 360.0
                self.state = _open_door_world(x=float(self.state.pose.position["x"]), z=float(self.state.pose.position["z"]), yaw=yaw)
            elif action.kind == ActionKind.ROTATE_LEFT.value:
                yaw = (float(self.state.pose.rotation.get("y", 0.0)) - 90.0) % 360.0
                self.state = _open_door_world(x=float(self.state.pose.position["x"]), z=float(self.state.pose.position["z"]), yaw=yaw)
            return ThorExecution(SimpleNamespace(metadata={}), self.state, 1, [action.kind])

    adapter = FakeAdapter()
    before = adapter.observe()
    result = adapter.execute(Action(ActionKind.CROSS_THRESHOLD.value, {"objectId": "Door|test"}), step_budget=20)
    after = result.state
    assert result.primitive_steps > 0
    assert _crossed_door(before, after, before.objects["Door|test"].position)


def test_live_adapter_dispatch_executes_face_target():
    class FakeAdapter(_AdapterMixin):
        def __init__(self):
            self.config = ThorConfig(scene="TestScene", rotate_step_degrees=90.0)
            world = _open_door_world(x=0.0, z=0.0, yaw=90.0)
            self.state = world

        def observe(self):
            return self.state

        def _execute_primitive(self, action):
            yaw = float(self.state.pose.rotation.get("y", 0.0))
            if action.kind == ActionKind.ROTATE_RIGHT.value:
                yaw = (yaw + 90.0) % 360.0
            elif action.kind == ActionKind.ROTATE_LEFT.value:
                yaw = (yaw - 90.0) % 360.0
            self.state = _open_door_world(x=0.0, z=0.0, yaw=yaw)
            return ThorExecution(SimpleNamespace(metadata={}), self.state, 1, [action.kind])

    adapter = FakeAdapter()
    result = adapter.execute(Action(ActionKind.FACE_TARGET.value, {"objectId": "Door|test"}), step_budget=10)
    assert result.primitive_steps > 0


def test_repair_removes_only_v0104_missing_cross_failures_and_is_idempotent(tmp_path: Path):
    state = tmp_path / "state"
    replay = state / "replay" / "embodied_replay.jsonl"
    checkpoint = state / "checkpoints" / "a" / "embodied_skills.json"
    replay.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)

    bad = {
        "schema": "author.embodied.v0104",
        "agent": "A",
        "curriculum_level": 1,
        "stage_skill": "door_cross",
        "action": "CrossThreshold",
        "target_object_type": "Door",
        "success": False,
        "primitive_steps": 0,
        "verification": {"exception": "NameError: name '_cross_threshold_for_adapter' is not defined"},
    }
    legitimate_failure = dict(bad)
    legitimate_failure["schema"] = "author.embodied.v0105"
    replay.write_text(json.dumps(bad) + "\n" + json.dumps(legitimate_failure) + "\n", encoding="utf-8")

    payload = {
        "version": 3,
        "agent_label": "A",
        "skills": {"door_cross": {"attempts": 6, "successes": 4, "failures": 2, "primitive_steps": 20}},
        "by_object_type": {"door_cross": {"Door": {"attempts": 6, "successes": 4, "failures": 2, "primitive_steps": 20}}},
        "by_level": {"door_cross": {"1": {"attempts": 6, "successes": 4, "failures": 2, "primitive_steps": 20}}},
    }
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    result = repair_v0104_missing_cross_threshold(state)
    assert result["matched_replay_rows"] == 1
    repaired = json.loads(checkpoint.read_text(encoding="utf-8"))
    for row in (
        repaired["skills"]["door_cross"],
        repaired["by_object_type"]["door_cross"]["Door"],
        repaired["by_level"]["door_cross"]["1"],
    ):
        assert row["attempts"] == 5
        assert row["successes"] == 4
        assert row["failures"] == 1
    assert checkpoint.with_name("embodied_skills.pre_v0105_cross_repair.json").exists()

    again = repair_v0104_missing_cross_threshold(state)
    assert again["already_applied"] is True
    repaired_again = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert repaired_again["skills"]["door_cross"]["attempts"] == 5
