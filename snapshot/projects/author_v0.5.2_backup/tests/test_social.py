from agent_core.grounding import SemanticGrounder
from agent_core.models import AgentPose, WorldState
from agent_core.roam import EmbodiedPlanner
from agent_core.social import SpeechBus
from agent_core.social_session import SocialSessionController


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, "image": False})
        return self.response

    def chat_with_image(self, system, user, image_bytes, **kwargs):
        self.calls.append({"system": system, "user": user, "image": True})
        return self.response


def world_at(x, z):
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose(
            position={"x": float(x), "y": 0.9, "z": float(z)},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            horizon=0.0,
        ),
        objects={},
        held_object_id=None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_speech_bus_delivers_only_inside_physical_range():
    bus = SpeechBus({0: "A", 1: "B"}, speech_range_m=4.0)
    worlds = {0: world_at(0, 0), 1: world_at(3, 0)}
    delivered = bus.speak(0, "Hello there.", worlds)
    assert len(delivered) == 1
    assert bus.peek(1)[0].text == "Hello there."
    assert bus.peek(1)[0].speaker_label == "A"


def test_speech_bus_does_not_deliver_out_of_range():
    bus = SpeechBus({0: "A", 1: "B"}, speech_range_m=4.0)
    worlds = {0: world_at(0, 0), 1: world_at(4.01, 0)}
    assert bus.speak(0, "Can you hear me?", worlds) == []
    assert bus.peek(1) == []


def test_consume_removes_heard_speech_without_cross_agent_state():
    bus = SpeechBus({0: "A", 1: "B"}, speech_range_m=4.0)
    worlds = {0: world_at(0, 0), 1: world_at(1, 0)}
    bus.speak(0, "Only these words cross.", worlds)
    heard = bus.consume(1)
    assert heard[0].for_llm() == {"speaker": "A", "text": "Only these words cross.", "distance_m": 1.0}
    assert bus.peek(1) == []
    assert set(heard[0].for_llm()) == {"speaker", "text", "distance_m"}


def test_social_planner_requires_explicit_speak_action_to_send_speech():
    client = FakeClient(
        '{"say":"I notice another presence.","focus":null,"pace":"normal",'
        '"task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder(), social_enabled=True, agent_label="A")
    decision = planner.decide(world_at(0, 0), "autonomous", None, [], image_bytes=b"frame", social_context={"self_label":"A","heard_since_last_turn":[]})
    assert decision.say == "I notice another presence."
    assert decision.spoken == []


def test_social_planner_explicit_speak_has_no_physical_unity_action():
    client = FakeClient(
        '{"say":"","focus":null,"pace":"normal","task_complete":false,'
        '"actions":[{"kind":"Speak","target":null,"params":{"text":"Hello."}}]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder(), social_enabled=True, agent_label="A")
    decision = planner.decide(world_at(0, 0), "autonomous", None, [], image_bytes=b"frame", social_context={"self_label":"A","heard_since_last_turn":[]})
    assert decision.spoken == ["Hello."]
    assert decision.actions == []
    assert decision.physical_semantic_actions == []
    assert decision.semantic_actions[0].kind == "Speak"


def test_heard_speech_is_present_in_receiver_prompt_only_when_supplied():
    client = FakeClient(
        '{"say":"","focus":null,"pace":"normal","task_complete":false,"actions":[]}'
    )
    planner = EmbodiedPlanner(client, SemanticGrounder(), social_enabled=True, agent_label="B")
    social = {"self_label":"B","heard_since_last_turn":[{"speaker":"A","text":"Hello B.","distance_m":2.0}],"speech_range_m":4.0}
    planner.decide(world_at(0, 0), "autonomous", None, [], image_bytes=b"frame", social_context=social)
    sent = client.calls[0]["user"]
    assert "Hello B." in sent
    assert "private" not in sent.lower()


def test_social_system_prompt_does_not_require_social_behavior():
    planner = EmbodiedPlanner(FakeClient("{}"), SemanticGrounder(), social_enabled=True, agent_label="A")
    prompt = planner.system_prompt.lower()
    assert "not required to" in prompt
    assert "speak" in prompt
    assert "cooperate" in prompt
    assert "you must cooperate" not in prompt
    assert "you should speak" not in prompt


def test_social_session_controller_routes_tasks_independently():
    c = SocialSessionController({0: "A", 1: "B"})
    c.handle("/a Find a cup.")
    assert c.state.tasks[0] == "Find a cup."
    assert c.state.tasks[1] is None
    c.handle("/b Open the refrigerator.")
    assert c.state.tasks[1] == "Open the refrigerator."
    c.handle("/a roam")
    assert c.state.tasks[0] is None
    assert c.state.tasks[1] == "Open the refrigerator."


def test_social_session_controller_shared_task_and_shared_roam():
    c = SocialSessionController({0: "A", 1: "B"})
    c.handle("/both Put the apple in the sink.")
    assert c.state.tasks == {0: "Put the apple in the sink.", 1: "Put the apple in the sink."}
    c.handle("/both roam")
    assert c.state.tasks == {0: None, 1: None}


def test_plain_text_defaults_to_shared_user_task_not_autonomous_injection():
    c = SocialSessionController({0: "A", 1: "B"})
    c.handle("Tell me what you each see.")
    assert c.state.tasks[0] == "Tell me what you each see."
    assert c.state.tasks[1] == "Tell me what you each see."


def test_peer_camera_cone_sensor_reports_neutral_geometry_without_scene_object_names():
    from agent_core.social import peer_presence_for

    a = world_at(0, 0)
    b = world_at(0, 2)
    rows = peer_presence_for(
        0,
        {0: a, 1: b},
        {0: "A", 1: "B"},
        field_of_view_degrees=90.0,
        speech_range_m=4.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.in_camera_cone is True
    assert row.distance_m == 2.0
    payload = row.for_llm()
    assert payload["another_embodied_agent_in_camera_cone"] is True
    assert "peer_label" not in payload
    assert "coffee" not in repr(payload).lower()
    assert "metal" not in repr(payload).lower()


def test_peer_camera_cone_sensor_changes_with_body_heading():
    from agent_core.social import peer_presence_for

    a = world_at(0, 0)
    b = world_at(0, 2)
    a.pose.rotation["y"] = 180.0
    row = peer_presence_for(0, {0: a, 1: b}, {0: "A", 1: "B"})[0]
    assert row.in_camera_cone is False
    assert row.for_llm() == {"another_embodied_agent_in_camera_cone": False}


def test_social_prompt_explains_peer_sensor_without_requiring_interaction():
    planner = EmbodiedPlanner(FakeClient("{}"), SemanticGrounder(), social_enabled=True, agent_label="A")
    prompt = planner.system_prompt.lower()
    assert "peer_camera_cone_sensor" in prompt
    assert "colored capsule-like bodies" in prompt
    assert "does not prove" in prompt
    assert "you are not required to speak" in prompt


def test_peer_sensor_is_supplied_as_social_sensory_input_only():
    client = FakeClient('{"say":"","focus":null,"pace":"normal","task_complete":false,"actions":[]}')
    planner = EmbodiedPlanner(client, SemanticGrounder(), social_enabled=True, agent_label="A")
    social = {
        "self_label": "A",
        "heard_since_last_turn": [],
        "speech_range_m": 4.0,
        "peer_camera_cone_sensor": [{
            "another_embodied_agent_in_camera_cone": True,
            "distance_m": 1.5,
            "bearing_degrees": -12.0,
        }],
    }
    planner.decide(world_at(0, 0), "autonomous", None, [], image_bytes=b"frame", social_context=social)
    sent = client.calls[0]["user"]
    assert "another_embodied_agent_in_camera_cone" in sent
    assert "1.5" in sent
    assert "CoffeeMachine" not in sent
    assert "MetalRack" not in sent
