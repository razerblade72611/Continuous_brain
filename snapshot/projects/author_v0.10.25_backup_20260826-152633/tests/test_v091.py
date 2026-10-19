from __future__ import annotations

from dataclasses import dataclass

from agent_core.grounding import SemanticGrounder
from agent_core.models import AgentPose, ObjectState, WorldState
from agent_core.roam import EmbodiedPlanner
from agent_core.social import SpeechBus, TextBus
from agent_core.social_session import SocialEmbodiedSession
from agent_core.task_contracts import (
    capture_task_baseline,
    contract_satisfied,
    discovery_region,
    next_required_interaction,
    parse_task_contract,
)


def obj(oid: str, kind: str, *, distance: float = 1.0, **kw) -> ObjectState:
    return ObjectState(
        object_id=oid,
        object_type=kind,
        name=kind,
        visible=True,
        position={"x": 0.0, "y": 1.0, "z": max(0.2, distance)},
        distance=distance,
        **kw,
    )


def world(objects=(), *, scene="FloorPlan1", x=0.0, z=0.0, held=None) -> WorldState:
    return WorldState(
        scene=scene,
        pose=AgentPose({"x": x, "y": 0.9, "z": z}, {"x": 0.0, "y": 0.0, "z": 0.0}),
        objects={o.object_id: o for o in objects},
        held_object_id=held,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


class SequenceClient:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = []

    def _next(self, system, user, image=None, **kwargs):
        self.calls.append({"system": system, "user": user, "image": image, "kwargs": kwargs})
        if not self.responses:
            raise RuntimeError("No fake response left")
        return self.responses.pop(0)

    def chat(self, system, user, **kwargs):
        return self._next(system, user, **kwargs)

    def chat_with_image(self, system, user, image_bytes, **kwargs):
        return self._next(system, user, image=image_bytes, **kwargs)


def social_context(peer="B"):
    return {
        "self_identity": {"label": "A", "kind": "independently embodied agent"},
        "peer_identity": [{"label": peer, "relation": "another independently embodied agent like you", "private_state_shared": False}],
        "texts_since_last_turn": [],
        "heard_since_last_turn": [],
        "text_channel_available": True,
    }


def test_text_each_other_contract_resolves_only_known_peer():
    c = parse_task_contract("Remember to message each other.", self_label="A", peer_labels=["B"])
    assert c.kind == "TextPeer"
    assert c.recipient == "B"
    payload = c.for_llm()
    assert payload["required_capability"] == "TextPeer"
    assert "objectId" not in repr(payload)
    assert "coordinate" not in repr(payload).lower()


def test_text_task_forces_retry_until_model_emits_real_textpeer():
    first = '{"observation":"","intention":null,"say":"I will message B.","focus":null,"pace":"normal","task_complete":true,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    second = '{"observation":"","intention":null,"say":"","focus":null,"pace":"normal","task_complete":true,"actions":[{"kind":"TextPeer","target":"B","params":{"text":"Can you help me find the door?"}}]}'
    client = SequenceClient(first, second)
    planner = EmbodiedPlanner(client, SemanticGrounder(), social_enabled=True, agent_label="A")
    decision = planner.decide(
        world(), "task", "Text B and ask B to help find the door.", [], None, social_context=social_context()
    )
    assert len(client.calls) == 2
    assert decision.texted == [{"recipient": "B", "text": "Can you help me find the door?"}]
    assert "TextPeer" in client.calls[1]["user"]
    assert "say" in client.calls[1]["user"]


def test_pickup_task_handoff_converts_preparation_to_pickup_only_after_model_focus():
    apple = obj("Apple|1", "Apple", pickupable=True, distance=0.9)
    response = '{"observation":"apple nearby","intention":null,"say":"","focus":"apple","pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    planner = EmbodiedPlanner(SequenceClient(response), SemanticGrounder(), allow_interactions=True)
    decision = planner.decide(world([apple]), "task", "Pick up the apple.", [], None)
    assert decision.motor_handoff is True
    assert decision.semantic_actions[-1].kind == "PickupObject"
    assert decision.actions[0].kind == "PickupObject"
    assert decision.actions[0].params["objectId"] == "Apple|1"


def test_autonomous_observe_never_gets_task_motor_handoff():
    apple = obj("Apple|1", "Apple", pickupable=True, distance=0.9)
    response = '{"observation":"apple nearby","intention":"look at apple","say":"","focus":"apple","pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    planner = EmbodiedPlanner(SequenceClient(response), SemanticGrounder(), allow_interactions=True)
    decision = planner.decide(world([apple]), "autonomous", None, [], None)
    assert decision.motor_handoff is False
    assert decision.semantic_actions[0].kind == "Observe"
    assert decision.actions[0].kind == "Pass"


def test_composite_put_contract_stages_pickup_then_put():
    apple = obj("Apple|1", "Apple", pickupable=True)
    counter = obj("Counter|1", "CounterTop", receptacle=True)
    c = parse_task_contract("Pick up the apple and put it on the countertop.")
    assert c.kind == "PutObject"
    assert next_required_interaction(c, world([apple, counter], held=None)) == ("PickupObject", "apple")
    assert next_required_interaction(c, world([apple, counter], held="Apple|1")) == ("PutObject", "countertop")


def test_discovery_escape_requires_actual_region_crossing():
    start = world(scene="FloorPlan1Discovery_physics", x=0.0, z=0.0)
    baseline = capture_task_baseline(start)
    contract = parse_task_contract("Escape.")
    assert discovery_region(start) == "kitchen"
    assert not contract_satisfied(contract, world(scene=start.scene, x=2.6, z=0.0), baseline)
    outside = world(scene=start.scene, x=3.0, z=0.0)
    assert discovery_region(outside) == "east_experiment"
    assert contract_satisfied(contract, outside, baseline)
    # Stock FloorPlan1 has no connected-region verifier, so an LLM claim cannot fake escape.
    stock_baseline = capture_task_baseline(world(scene="FloorPlan1"))
    assert not contract_satisfied(contract, world(scene="FloorPlan1", x=5.0, z=5.0), stock_baseline)


def test_escape_task_opens_model_selected_reachable_door_without_choosing_route():
    door = obj("Door|AUTHOR|NorthArchive", "Door", openable=True, is_open=False, distance=0.9)
    response = '{"observation":"closed door","intention":null,"say":"","focus":"door","pace":"normal","task_complete":false,"actions":[{"kind":"Observe","target":null,"params":{}}]}'
    planner = EmbodiedPlanner(SequenceClient(response), SemanticGrounder(), allow_interactions=True)
    decision = planner.decide(
        world([door], scene="FloorPlan1Discovery_physics"), "task", "Escape.", [], None
    )
    assert decision.motor_handoff is True
    assert decision.semantic_actions[-1].kind == "OpenObject"
    assert decision.actions[0].params["objectId"] == "Door|AUTHOR|NorthArchive"


class DummyMemory:
    def __init__(self):
        self.rows = []

    def add(self, kind, payload):
        self.rows.append((kind, payload))


class DummyThor:
    def __init__(self, state):
        self.state = state

    def observe(self):
        return self.state


@dataclass
class DummyLoop:
    thor: DummyThor
    memory: DummyMemory


def test_social_unknown_task_claim_is_ignored_not_completed():
    loops = {
        0: DummyLoop(DummyThor(world()), DummyMemory()),
        1: DummyLoop(DummyThor(world()), DummyMemory()),
    }
    session = SocialEmbodiedSession(loops, {0: "A", 1: "B"}, SpeechBus({0: "A", 1: "B"}), TextBus({0: "A", 1: "B"}))
    session.controller.state.tasks[0] = "Become friends and think about the room."
    session._tracked_tasks[0] = session.controller.state.tasks[0]
    session._task_baselines[0] = capture_task_baseline(loops[0].thor.observe())
    session._complete_task_if_needed(0, claimed=True)
    assert session.controller.state.tasks[0] is not None
    assert any(kind == "user_task_completion_claim_ignored" for kind, _ in loops[0].memory.rows)


def test_social_text_task_completes_only_after_actual_recipient_delivery_event():
    loops = {
        0: DummyLoop(DummyThor(world()), DummyMemory()),
        1: DummyLoop(DummyThor(world()), DummyMemory()),
    }
    bus = TextBus({0: "A", 1: "B"})
    session = SocialEmbodiedSession(loops, {0: "A", 1: "B"}, SpeechBus({0: "A", 1: "B"}), bus)
    task = "Text B and ask B to help with the door."
    session.controller.state.tasks[0] = task
    session._tracked_tasks[0] = task
    session._task_baselines[0] = capture_task_baseline(loops[0].thor.observe())
    session._task_text_baseline[0] = 0

    session._complete_task_if_needed(0, claimed=True)
    assert session.controller.state.tasks[0] == task

    event = bus.send(0, "B", "Can you help with the door?")
    assert event is not None
    session._task_text_events[0].append(event)
    session._complete_task_if_needed(0, claimed=False)
    assert session.controller.state.tasks[0] is None


def test_v091_unity_builder_uses_fully_qualified_unity_scene_manager():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "unity_world" / "DiscoveryWorldBuilder.cs").read_text()
    assert "UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene" in text
    assert "UnityEngine.SceneManagement.UnityEngine.SceneManagement" not in text


def test_v091_release_version_metadata():
    import agent_core
    assert agent_core.__version__ == "0.10.25"
