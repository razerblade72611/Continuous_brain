from pathlib import Path
from types import SimpleNamespace
import json

import agent_core
import agent_core.training as training
from agent_core.learning import LearningPaths, write_json_atomic
from agent_core.grounding import SemanticGrounder, GroundingError
from agent_core.models import ActionKind, AgentPose, ObjectState, WorldState
from agent_core.migrations import repair_v01023_broken_toggle_target_regression


def test_release_version_is_v01024():
    assert agent_core.__version__ == "0.10.25"


def test_broken_toggle_target_is_rejected_but_healthy_laptop_remains_eligible():
    broken = SimpleNamespace(raw={"isBroken": True})
    healthy = SimpleNamespace(raw={"isBroken": False})
    missing = SimpleNamespace(raw={})
    assert training._toggle_target_is_functional(broken) is False
    assert training._toggle_target_is_functional(healthy) is True
    assert training._toggle_target_is_functional(missing) is True


def test_train_toggle_filters_broken_targets_before_route_selection():
    src = Path(training.__file__).read_text(encoding="utf-8")
    block = src[src.index("def _train_toggle"):src.index("def _train_peer_meet")]
    line = "obj.toggleable and obj.is_toggled is not None and _toggle_target_is_functional(obj)"
    assert line in block
    assert block.index(line) < block.index("_find_object_with_blocked_route")
    assert "no functional toggleable target" in block




def _world_with_toggle(obj):
    return WorldState(
        scene="FloorPlan1Discovery_physics",
        pose=AgentPose(position={"x": 0.0, "y": 0.9, "z": 0.0}, rotation={"x": 0.0, "y": 0.0, "z": 0.0}, horizon=0.0),
        objects={obj.object_id: obj},
        held_object_id=None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_runtime_grounding_does_not_advertise_or_ground_toggle_on_for_broken_target():
    obj = ObjectState(
        object_id="Laptop|1",
        object_type="Laptop",
        name="Laptop_1",
        visible=True,
        position={"x": 0.0, "y": 0.8, "z": 0.5},
        distance=0.5,
        toggleable=True,
        is_toggled=False,
        raw={"isBroken": True},
    )
    layer = SemanticGrounder(interaction_distance=1.25)
    assert layer._eligible(obj, ActionKind.TOGGLE_ON.value) is False
    assert layer.try_focus("Laptop", _world_with_toggle(obj)) is True
    context = layer.focus_snapshot(_world_with_toggle(obj))
    assert context is not None
    assert "ToggleObjectOn" not in context["currently_possible_actions"]
    try:
        layer.ground(ActionKind.TOGGLE_ON.value, "Laptop", {}, _world_with_toggle(obj))
    except GroundingError:
        pass
    else:
        raise AssertionError("broken Laptop must not ground to ToggleObjectOn")
    # Broken state stays private: compact semantic object data does not expose it.
    assert "isBroken" not in obj.compact()

def _stats(attempts, successes, failures, steps, recovery_attempts=0, recovery_successes=0, recovery_exhausted=0):
    return {
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "primitive_steps": steps,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "recovery_exhausted": recovery_exhausted,
    }


def test_v01023_broken_toggle_repair_removes_only_four_invalid_interactions_and_is_idempotent(tmp_path):
    paths = LearningPaths.from_root(tmp_path / "state").ensure()

    rows = []
    for episode, agent, steps in ((5, "A", 9), (12, "B", 9), (23, "A", 7), (57, "A", 7)):
        rows.append({
            "schema": "author.embodied.v01023",
            "agent": agent,
            "episode": episode,
            "curriculum_skill": "toggle",
            "curriculum_level": 4,
            "stage_skill": "toggle_interaction",
            "success": False,
            "target_object_type": "Laptop",
            "primitive_steps": steps,
            "verification": {
                "last_action_error": "Target is broken and cannot be Toggled On!",
                "interaction_recovery_used": True,
                "interaction_recovery_success": False,
                "interaction_recovery_exhausted": True,
            },
        })
    # Similar-looking rows that must remain untouched.
    rows.extend([
        {
            "schema": "author.embodied.v01023",
            "agent": "A",
            "curriculum_skill": "toggle",
            "curriculum_level": 4,
            "stage_skill": "toggle_interaction",
            "success": True,
            "target_object_type": "Laptop",
            "primitive_steps": 1,
            "verification": {"last_action_error": None},
        },
        {
            "schema": "author.embodied.v01023",
            "agent": "A",
            "curriculum_skill": "toggle",
            "curriculum_level": 4,
            "stage_skill": "toggle_interaction",
            "success": False,
            "target_object_type": "DeskLamp",
            "primitive_steps": 4,
            "verification": {"last_action_error": "Object not visible"},
        },
    ])
    paths.embodied_replay_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    for agent, failures, fail_steps in (("A", 3, 23), ("B", 1, 9)):
        checkpoint = paths.agent_skill_checkpoint(agent)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 3,
            "agent_label": agent,
            "skills": {
                "toggle_interaction": _stats(30, 30-failures, failures, 30 + fail_steps, failures, 0, failures),
                "toggle_approach": _stats(30, 30, 0, 100),
            },
            "by_object_type": {
                "toggle_interaction": {"Laptop": _stats(failures, 0, failures, fail_steps, failures, 0, failures)},
            },
            "by_level": {
                "toggle_interaction": {"4": _stats(30, 30-failures, failures, 30 + fail_steps, failures, 0, failures)},
                "toggle_approach": {"4": _stats(30, 30, 0, 100)},
            },
        }
        write_json_atomic(checkpoint, payload)

    raw_before = paths.embodied_replay_file.read_text(encoding="utf-8")
    result = repair_v01023_broken_toggle_target_regression(paths.root)
    assert result["matched_broken_toggle_regressions"] == 4
    assert result["raw_replay_preserved"] is True

    for agent in ("A", "B"):
        payload = json.loads(paths.agent_skill_checkpoint(agent).read_text(encoding="utf-8"))
        row = payload["skills"]["toggle_interaction"]
        assert row["failures"] == 0
        assert row["recovery_attempts"] == 0
        assert row["recovery_exhausted"] == 0
        assert payload["skills"]["toggle_approach"]["attempts"] == 30
        assert payload["skills"]["toggle_approach"]["failures"] == 0

    assert paths.embodied_replay_file.read_text(encoding="utf-8") == raw_before
    again = repair_v01023_broken_toggle_target_regression(paths.root)
    assert again["already_applied"] is True
