from __future__ import annotations

from pathlib import Path

from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.grounding import SemanticGrounder
from agent_core.learning import LearningPaths
from agent_core.models import AgentPose, ObjectState, WorldState, ActionKind
from agent_core.roam import EmbodiedPlanner
from agent_core.thor_adapter import _choose_cross_threshold_target
from agent_core.training import _curriculum_from_config, build_parser


class OneResponseClient:
    def __init__(self, response: str):
        self.response = response

    def chat(self, *args, **kwargs):
        return self.response


def _world_with_open_door() -> WorldState:
    door = ObjectState(
        object_id="Door|AUTHOR|NorthArchive",
        object_type="Door",
        name="Door",
        visible=True,
        position={"x": 0.0, "y": 0.0, "z": 1.0},
        distance=0.9,
        openable=True,
        is_open=True,
    )
    return WorldState(
        scene="FloorPlan1Discovery_physics",
        pose=AgentPose(position={"x": 0.0, "y": 0.9, "z": 0.0}, rotation={"x": 0.0, "y": 0.0, "z": 0.0}),
        objects={door.object_id: door},
        held_object_id=None,
        last_action="Pass",
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_embodied_skill_library_persists_and_readiness(tmp_path: Path):
    path = tmp_path / "skills.json"
    lib = EmbodiedSkillLibrary(path, agent_label="A")
    for _ in range(5):
        lib.record("cross_door", success=True, primitive_steps=3, object_type="Door")
    lib.save()

    loaded = EmbodiedSkillLibrary(path, agent_label="A")
    assert loaded.ready("cross_door", "Door", min_attempts=5, min_success_rate=0.8)
    assert loaded.ready("door_cross", "Door", min_attempts=5, min_success_rate=0.8)
    assert loaded.stats("cross_door", "Door").successes == 5
    assert loaded.stats("door_cross", "Door").successes == 5
    assert loaded.compact_summary()["door_cross"]["attempts"] == 5


def test_learning_paths_add_per_agent_skills_and_sft(tmp_path: Path):
    paths = LearningPaths.from_root(tmp_path / "state").ensure()
    assert paths.agent_skill_checkpoint("A").name == "embodied_skills.json"
    assert paths.agent_skill_checkpoint("A").parent.name == "a"
    assert paths.embodied_replay_file.name == "embodied_replay.jsonl"
    assert paths.embodied_sft_file.parent.name == "datasets"
    assert paths.datasets_dir.exists()


def test_cross_threshold_target_is_beyond_selected_door():
    current = {"x": 0.0, "y": 0.9, "z": 0.0}
    door = {"x": 0.0, "y": 0.0, "z": 1.0}
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.75},
        {"x": 0.0, "y": 0.9, "z": 1.25},
        {"x": 0.0, "y": 0.9, "z": 1.75},
        {"x": 1.5, "y": 0.9, "z": 1.75},
    ]
    target = _choose_cross_threshold_target(current, door, reachable)
    assert target is not None
    assert target["z"] > door["z"]
    assert abs(target["x"] - door["x"]) <= 0.9


def test_trained_escape_handoff_crosses_already_open_model_selected_door(tmp_path: Path):
    skills = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    for _ in range(5):
        skills.record("cross_door", success=True, primitive_steps=3, object_type="Door")
    response = '{"observation":"open doorway","intention":"leave through this door","say":"","focus":"door","pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    planner = EmbodiedPlanner(
        OneResponseClient(response),
        SemanticGrounder(),
        allow_interactions=True,
        skill_library=skills,
        skill_min_attempts=5,
        skill_min_success_rate=0.65,
    )
    decision = planner.decide(_world_with_open_door(), "task", "Escape.", [], None)
    assert decision.motor_handoff is True
    assert decision.actions[0].kind == ActionKind.CROSS_THRESHOLD.value
    assert decision.semantic_actions[-1].kind == "CrossThreshold"


def test_curriculum_override_and_parser():
    assert _curriculum_from_config({}, "open_cross,pickup") == {"door": 1.0, "pickup": 1.0}
    args = build_parser().parse_args(["--episodes", "10", "--curriculum", "open_cross,pickup", "--no-sft-export"])
    assert args.curriculum == "open_cross,pickup"
    assert args.no_sft_export is True
