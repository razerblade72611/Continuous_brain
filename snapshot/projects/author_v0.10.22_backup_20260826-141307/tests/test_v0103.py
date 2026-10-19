from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import agent_core
from agent_core.actionability import PoseActionabilityMemory
from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.interaction_pose import InteractionPose
from agent_core.models import Action, ActionKind, AgentPose, ObjectState, WorldState
from agent_core.thor_adapter import ThorConfig, ThorExecution, _execute_interaction_with_pose_recovery


def _door_world(*, x: float, z: float, yaw: float = 90.0, is_open: bool, success: bool | None = True, error: str | None = None):
    door = ObjectState(
        object_id="Door|test", object_type="Door", name="TestDoor", visible=True,
        position={"x": 1.0, "y": 0.0, "z": 0.0}, distance=((1.0-x)**2+z**2)**0.5,
        openable=True, is_open=is_open,
    )
    return WorldState(
        scene="TestScene",
        pose=AgentPose(position={"x": x, "y": 0.9, "z": z}, rotation={"x": 0.0, "y": yaw, "z": 0.0}, horizon=0.0),
        objects={door.object_id: door}, held_object_id=None, last_action="OpenObject",
        last_action_success=success, last_error=error, frame_available=False,
    )


def test_release_version_is_v0103():
    assert agent_core.__version__ == "0.10.22"


def test_v0102_actionability_is_migrated_to_spatial_rows(tmp_path: Path):
    path = tmp_path / "interaction_actionability.json"
    path.write_text('''{"version":1,"agent_label":"A","grid_size":0.25,"pose_stats":{"OpenObject|Door_test|2|0|90|0":{"attempts":2,"successes":0,"failures":2,"recoveries":1}}}''')
    memory = PoseActionabilityMemory(path, agent_label="A", grid_size=0.25)
    assert memory.position_stats["OpenObject|Door_test|2|0"].failures == 2
    alternate_yaw = InteractionPose(0.5, 0.9, 0.0, 180.0, 30.0, True)
    assert memory.position_stats_for("OpenObject", "Door|test", alternate_yaw).failures == 2


def test_spatial_failure_penalizes_unseen_orientation_same_cell(tmp_path: Path):
    memory = PoseActionabilityMemory(tmp_path / "a.json", agent_label="A", grid_size=0.25)
    bad = InteractionPose(0.5, 0.9, 0.0, 90.0, 0.0, True)
    same_cell_new_orientation = InteractionPose(0.5, 0.9, 0.0, 180.0, 60.0, True)
    safe = InteractionPose(-0.25, 0.9, 0.0, 90.0, 0.0, True)
    memory.record("OpenObject", "Door|test", bad, success=False, error="Object failed to open/close successfully.")
    target = {"x": 1.0, "y": 0.0, "z": 0.0}
    assert memory.penalty("OpenObject", "Door|test", same_cell_new_orientation, object_position=target) > memory.penalty("OpenObject", "Door|test", safe, object_position=target)


def test_recovery_skips_same_bad_region_and_reaches_safe_clearance(tmp_path: Path, monkeypatch):
    class FakeAdapter:
        def __init__(self):
            self.config = ThorConfig(
                scene="TestScene", learning_state_dir=str(tmp_path), use_interaction_actionability=True,
                max_interaction_action_retries=3, preferred_open_clearance_m=1.0,
                interaction_recovery_exclusion_radius_m=0.50,
                interaction_recovery_min_position_change_m=0.25, grid_size=0.25,
            )
            self.agent_label = "A"
            self._pose_actionability_memory = None
            self.state = _door_world(x=0.5, z=0.0, is_open=False)
            self.open_attempts = 0
            self.moved_to = []

        def observe(self): return self.state
        def raw_event(self): return SimpleNamespace(metadata={})
        def get_reachable_positions(self):
            return [{"x": x, "y": 0.9, "z": 0.0} for x in (0.5, 0.25, 0.0, -0.25)]
        def get_interactable_poses(self, object_id, **kwargs):
            return [
                InteractionPose(0.5, 0.9, 0.0, 180.0, 30.0, True),   # same bad X/Z
                InteractionPose(0.25, 0.9, 0.0, 90.0, 0.0, True),    # too near failed region / <1m clearance
                InteractionPose(-0.25, 0.9, 0.0, 90.0, 0.0, True),   # safe 1.25m clearance
            ]
        def _execute_primitive(self, action):
            self.open_attempts += 1
            x = float(self.state.pose.position["x"])
            ok = x < 0.0
            self.state = _door_world(x=x, z=0.0, is_open=ok, success=ok, error=None if ok else "Object failed to open/close successfully.")
            return ThorExecution(SimpleNamespace(metadata={}), self.state, 1, [action.kind])

    adapter = FakeAdapter()
    def fake_pose_goal(fake_adapter, pose, reachable, step_budget=None):
        fake_adapter.moved_to.append((pose.x, pose.z))
        fake_adapter.state = _door_world(x=pose.x, z=pose.z, yaw=pose.rotation, is_open=False, success=True)
        return ThorExecution(SimpleNamespace(metadata={}), fake_adapter.state, 1, ["MoveToRecoveryPose"])
    monkeypatch.setattr("agent_core.thor_adapter._execute_pose_goal", fake_pose_goal)
    monkeypatch.setattr("agent_core.thor_adapter._pose_goal_verified", lambda *args, **kwargs: True)

    result = _execute_interaction_with_pose_recovery(adapter, Action(ActionKind.OPEN.value, {"objectId":"Door|test","forceAction":False}, "open"), step_budget=20)
    assert result.state.objects["Door|test"].is_open is True
    assert adapter.open_attempts == 2
    assert adapter.moved_to == [(-0.25, 0.0)]
    summary = [r for r in result.auxiliary_skills or [] if r.get("kind") == "interaction_recovery_summary"][-1]
    assert summary["recovery_attempted"] is True
    assert summary["recovery_success"] is True
    assert summary["recovery_exhausted"] is False


def test_skill_checkpoint_v2_migrates_and_new_recovery_outcomes_are_explicit(tmp_path: Path):
    path = tmp_path / "skills.json"
    path.write_text('''{"version":2,"agent_label":"A","skills":{"door_open":{"attempts":4,"successes":3,"failures":1,"primitive_steps":7,"recoveries":1}},"by_object_type":{},"by_level":{},"legacy_v010":{"version":1,"skills":{},"by_object_type":{}}}''')
    lib = EmbodiedSkillLibrary(path, agent_label="A")
    assert lib.stats("door_open").recoveries == 1
    lib.record("door_open", success=True, primitive_steps=3, object_type="Door", recovery=True, recovery_success=True)
    lib.record("door_open", success=False, primitive_steps=4, object_type="Door", recovery=True, recovery_exhausted=True)
    stats = lib.stats("door_open")
    assert stats.recoveries == 3
    assert stats.recovery_successes == 1
    assert stats.recovery_exhausted == 1
    lib.save()
    reloaded = EmbodiedSkillLibrary(path, agent_label="A")
    assert reloaded.as_dict()["version"] == 3
    assert reloaded.stats("door_open").recovery_successes == 1
