from pathlib import Path

import agent_core
from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.grounding import SemanticGrounder
from agent_core.models import ActionKind, AgentPose, ObjectState, WorldState
from agent_core.roam import EmbodiedPlanner
from agent_core.task_contracts import parse_task_contract


class OneResponseClient:
    def __init__(self, response: str):
        self.response = response

    def chat(self, *args, **kwargs):
        return self.response


def _skills(tmp_path: Path) -> EmbodiedSkillLibrary:
    lib = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    for _ in range(5):
        lib.record("door_approach", success=True, primitive_steps=4, object_type="Door")
        lib.record("door_cross", success=True, primitive_steps=4, object_type="Door")
    return lib


def _world(*, open_state=False, distance=2.0, visible=True, scene="FloorPlan1Discovery_physics") -> WorldState:
    door = ObjectState(
        object_id="Door|AUTHOR|NorthArchive",
        object_type="Door",
        name="DiscoveryDoor_NorthArchive",
        visible=visible,
        position={"x": 0.0, "y": 0.0, "z": distance},
        distance=distance,
        openable=True,
        is_open=open_state,
    )
    return WorldState(
        scene=scene,
        pose=AgentPose(
            position={"x": 0.0, "y": 0.9, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        ),
        objects={door.object_id: door},
        held_object_id=None,
        last_action="Pass",
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def _planner(tmp_path: Path, response: str) -> EmbodiedPlanner:
    return EmbodiedPlanner(
        OneResponseClient(response),
        SemanticGrounder(),
        allow_interactions=True,
        skill_library=_skills(tmp_path),
        skill_min_attempts=5,
        skill_min_success_rate=0.65,
    )


def test_release_version_is_v01026():
    assert agent_core.__version__ == "0.10.26"


def test_exit_contract_tells_llm_private_motor_may_select_doorway():
    payload = parse_task_contract("Leave the kitchen.").for_llm()
    assert payload["required_outcome"] == "physically leave the starting room/region"
    assert "private motor layer may select" in payload["note"]
    assert "objectId" not in repr(payload)
    assert "coordinate" not in repr(payload).lower()


def test_exit_region_overrides_raw_moveahead_with_private_door_approach(tmp_path: Path):
    # This mirrors the live qwen2-vl behavior that caused dozens of MoveAhead loops.
    response = '{"observation":"door somewhere ahead","intention":"leave","say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"MoveAhead","target":null,"params":{}}]}'
    decision = _planner(tmp_path, response).decide(
        _world(open_state=False, distance=2.0, visible=False),
        "task",
        "Leave the kitchen.",
        [],
        None,
    )
    assert decision.motor_handoff is True
    assert decision.semantic_actions[-1].kind == "Approach"
    assert decision.semantic_actions[-1].target == "exit doorway"
    assert decision.actions[0].kind == ActionKind.NAVIGATE_TO.value
    assert decision.actions[0].params["objectId"] == "Door|AUTHOR|NorthArchive"
    assert decision.actions[0].params["_skill"] == "door_approach"


def test_exit_region_private_door_opens_when_reached_without_llm_focus(tmp_path: Path):
    response = '{"observation":"near doorway","intention":"leave","say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    decision = _planner(tmp_path, response).decide(
        _world(open_state=False, distance=0.9, visible=True),
        "task",
        "Leave the kitchen.",
        [],
        None,
    )
    assert decision.motor_handoff is True
    assert decision.semantic_actions[-1].kind == "OpenObject"
    assert decision.actions[0].kind == ActionKind.OPEN.value
    assert decision.actions[0].params["objectId"] == "Door|AUTHOR|NorthArchive"
    assert decision.actions[0].params["_skill"] == "door_open"


def test_exit_region_private_open_door_crosses_when_reached(tmp_path: Path):
    response = '{"observation":"open doorway","intention":"leave","say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    decision = _planner(tmp_path, response).decide(
        _world(open_state=True, distance=0.9, visible=True),
        "task",
        "Leave the kitchen.",
        [],
        None,
    )
    assert decision.motor_handoff is True
    assert decision.semantic_actions[-1].kind == "CrossThreshold"
    assert decision.actions[0].kind == ActionKind.CROSS_THRESHOLD.value
    assert decision.actions[0].params["objectId"] == "Door|AUTHOR|NorthArchive"
    assert decision.actions[0].params["_skill"] == "door_cross"


def test_exit_region_private_handoff_is_discovery_only(tmp_path: Path):
    response = '{"observation":"door","intention":"leave","say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"MoveAhead","target":null,"params":{}}]}'
    decision = _planner(tmp_path, response).decide(
        _world(open_state=False, distance=2.0, visible=False, scene="FloorPlan1"),
        "task",
        "Leave the kitchen.",
        [],
        None,
    )
    assert decision.motor_handoff is False
    assert decision.actions[0].kind == ActionKind.MOVE_AHEAD.value
