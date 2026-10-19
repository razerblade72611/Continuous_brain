from agent_core.grounding import SemanticGrounder
from agent_core.models import AgentPose, ObjectState, WorldState
from agent_core.roam import EmbodiedPlanner


class FakeClient:
    def __init__(self, responses):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls = []

    def _next(self, system, user, image=False):
        self.calls.append({"system": system, "user": user, "image": image})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    def chat(self, system, user, **kwargs):
        return self._next(system, user, False)

    def chat_with_image(self, system, user, image_bytes, **kwargs):
        return self._next(system, user, True)


def make_world(apple_visible=True):
    apple = ObjectState(
        object_id="Apple|1",
        object_type="Apple",
        name="Apple_1",
        visible=apple_visible,
        position={"x": 1.0, "y": 0.8, "z": 1.0},
        distance=1.0,
        pickupable=True,
    )
    counter = ObjectState(
        object_id="CounterTop|1",
        object_type="CounterTop",
        name="CounterTop_1",
        visible=True,
        position={"x": 2.0, "y": 0.9, "z": 2.0},
        distance=2.0,
        receptacle=True,
    )
    statue = ObjectState(
        object_id="Statue|1",
        object_type="Statue",
        name="Statue_1",
        visible=False,
        position={"x": 3.0, "y": 0.9, "z": 3.0},
        distance=4.0,
    )
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            horizon=0.0,
        ),
        objects={o.object_id: o for o in (apple, counter, statue)},
        held_object_id=None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_autonomous_prompt_does_not_leak_simulator_object_catalog():
    client = FakeClient(
        '{"say":"I will turn.","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"RotateRight","target":null,"params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder(), allow_interactions=True)
    planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    sent = client.calls[0]["user"]
    system = client.calls[0]["system"]
    for leaked in ["Apple|1", "CounterTop|1", "Statue|1", "Apple_1", "CounterTop_1", "Statue_1"]:
        assert leaked not in sent
        assert leaked not in system
    # The autonomous prompt must not even receive object class names from metadata.
    for leaked in ["Apple", "CounterTop", "Statue"]:
        assert leaked not in sent


def test_task_text_is_present_only_when_user_supplies_task():
    client = FakeClient(
        '{"say":"I will look.","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"LookDown","target":null,"params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    planner.decide(make_world(), "task", "Pick up the Apple.", [], image_bytes=b"fake")
    assert "Pick up the Apple." in client.calls[0]["user"]


def test_model_can_choose_free_primitive_burst_and_pace():
    client = FakeClient(
        '{"say":"Going this way.","pace":"fast","task_complete":false,"actions":['
        '{"kind":"RotateLeft","target":null,"params":{}},'
        '{"kind":"MoveAhead","target":null,"params":{}},'
        '{"kind":"MoveAhead","target":null,"params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=None)
    assert [a.kind for a in decision.actions] == ["RotateLeft", "MoveAhead", "MoveAhead"]
    assert decision.pace == "fast"
    assert not decision.fallback


def test_semantic_object_action_is_grounded_only_after_model_names_target():
    client = FakeClient(
        '{"say":"I want to handle it.","pace":"slow","task_complete":false,'
        '"actions":[{"kind":"PickupObject","target":"apple","params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder(), allow_interactions=True)
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.actions[0].kind == "PickupObject"
    assert decision.actions[0].params["objectId"] == "Apple|1"
    assert decision.semantic_actions[0].target == "apple"


def test_hidden_object_cannot_be_grounded_from_hallucinated_semantic_name():
    client = FakeClient([
        '{"say":"I see one.","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"PickupObject","target":"apple","params":{}}]}',
        '{"say":"I will just look around.","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"RotateRight","target":null,"params":{}}]}',
    ])
    planner = EmbodiedPlanner(client, SemanticGrounder(), allow_interactions=True)
    decision = planner.decide(make_world(apple_visible=False), "autonomous", None, [], image_bytes=b"fake")
    assert len(client.calls) == 2
    assert decision.actions[0].kind == "RotateRight"
    assert not decision.fallback


def test_model_failure_idles_instead_of_controller_roaming_for_it():
    class FailingClient(FakeClient):
        def chat(self, *args, **kwargs):
            raise RuntimeError("offline")
        def chat_with_image(self, *args, **kwargs):
            raise RuntimeError("offline")

    planner = EmbodiedPlanner(FailingClient(""), SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.fallback
    assert len(decision.actions) == 1
    assert decision.actions[0].kind == "Pass"



def test_empty_actions_is_valid_observe_not_failure():
    client = FakeClient(
        '{"say":"I will remain here.","focus":null,"pace":"normal","task_complete":false,"actions":[]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.fallback is False
    assert decision.semantic_actions[0].kind == "Observe"
    assert decision.actions[0].kind == "Pass"


def test_wait_alias_normalizes_to_observe():
    client = FakeClient(
        '{"say":"Waiting.","focus":null,"pace":"slow","task_complete":false,'
        '"actions":[{"kind":"Wait","target":null,"params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=None)
    assert decision.semantic_actions[0].kind == "Observe"
    assert decision.actions[0].kind == "Pass"


def test_primitive_target_becomes_attention_only_not_hidden_navigation():
    client = FakeClient(
        '{"say":"Continuing.","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"MoveAhead","target":"apple","params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.actions[0].kind == "MoveAhead"
    assert decision.actions[0].params == {}
    assert decision.semantic_actions[0].target is None
    assert decision.attention == "apple"


def test_model_can_choose_face_semantically():
    client = FakeClient(
        '{"say":"I want to face it.","focus":"apple","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"Face","target":"apple","params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.semantic_actions[0].kind == "Face"
    assert decision.actions[0].kind == "FaceTarget"


def test_already_satisfied_approach_gets_neutral_corrective_retry():
    client = FakeClient([
        '{"say":"Closer.","focus":"apple","pace":"normal","task_complete":false,'
        '"actions":[{"kind":"Approach","target":"apple","params":{}}]}',
        '{"say":"I will look elsewhere.","focus":null,"pace":"normal","task_complete":false,'
        '"actions":[{"kind":"RotateRight","target":null,"params":{}}]}',
    ])
    planner = EmbodiedPlanner(client, SemanticGrounder(interaction_distance=1.25))
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert len(client.calls) == 2
    assert decision.fallback is False
    assert decision.actions[0].kind == "RotateRight"
    assert "already within approach distance" in client.calls[1]["user"]


def test_focus_state_in_prompt_contains_only_model_semantic_reference_not_simulator_id():
    client = FakeClient(
        '{"say":"Observe.","focus":null,"pace":"normal","task_complete":false,"actions":[]}'
    )
    grounder = SemanticGrounder()
    grounder.try_focus("apple", make_world())
    planner = EmbodiedPlanner(client, grounder)
    planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    sent = client.calls[0]["user"]
    assert '"semantic_reference":"apple"' in sent
    assert "Apple|1" not in sent
    assert "Apple_1" not in sent
    assert "CounterTop|1" not in sent


def test_body_feedback_can_be_passed_without_object_metadata():
    client = FakeClient(
        '{"say":"Responding to my movement result.","focus":null,"pace":"normal","task_complete":false,"actions":[]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    feedback = {
        "actions": [{"kind": "MoveAhead", "success": False, "distance_moved": 0.0}],
        "decision_success": False,
        "distance_moved": 0.0,
    }
    planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake", body_feedback=feedback)
    sent = client.calls[0]["user"]
    assert '"distance_moved":0.0' in sent
    assert '"success":false' in sent
    assert "Apple|1" not in sent


def test_plain_text_semantic_action_is_mechanically_repaired_without_inventing_a_choice():
    client = FakeClient("Approach the apple")
    planner = EmbodiedPlanner(client, SemanticGrounder(interaction_distance=0.5))
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.fallback is False
    assert decision.semantic_actions[0].kind == "Approach"
    assert decision.semantic_actions[0].target == "apple"


def test_plain_descriptive_non_json_output_becomes_valid_observe():
    client = FakeClient("No object or behavior is being recommended.")
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.fallback is False
    assert decision.semantic_actions[0].kind == "Observe"
    assert "No object" in decision.say


def test_json_parser_accepts_valid_object_before_trailing_extra_text():
    client = FakeClient(
        '{"say":"Looking.","focus":null,"pace":"normal","task_complete":false,"actions":[]} trailing'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder())
    decision = planner.decide(make_world(), "autonomous", None, [], image_bytes=b"fake")
    assert decision.fallback is False
    assert decision.say == "Looking."


def test_recent_progress_summary_reports_repeated_stagnant_actions_without_prescribing_behavior():
    from agent_core.roam import EmbodiedDecision, EmbodiedLoop, SemanticAction

    loop = object.__new__(EmbodiedLoop)
    loop.recent = [
        {
            "semantic_actions": [{"kind": "MoveAhead", "target": None, "params": {}}],
            "body_feedback": {"distance_moved": 0.0, "yaw_change_degrees": 0.0},
        }
        for _ in range(7)
    ]
    decision = EmbodiedDecision(
        actions=[],
        semantic_actions=[SemanticAction("MoveAhead", None, {})],
        physical_semantic_actions=[],
        spoken=[],
    )
    summary = loop._recent_progress_summary(
        {"distance_moved": 0.0, "yaw_change_degrees": 0.0},
        decision,
    )
    assert summary["window_decisions"] == 8
    assert summary["repeated_action_signature"] == "MoveAhead"
    assert summary["repeated_action_count"] == 8
    assert summary["stagnant"] is True
    assert "recommend" not in repr(summary).lower()
