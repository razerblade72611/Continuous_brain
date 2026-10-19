
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_core.models import AgentPose, WorldState
from agent_core.social import SpeechBus, TextBus
from agent_core.social_session import SocialEmbodiedSession


def world() -> WorldState:
    return WorldState(
        scene="FloorPlan1Discovery_physics",
        pose=AgentPose({"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 0.0}),
        objects={},
        held_object_id=None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


class Memory:
    def __init__(self):
        self.rows = []

    def add(self, kind, payload):
        self.rows.append((kind, payload))


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or ["I found a door. Can you check the other exit?"])
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        return self.responses[min(len(self.calls)-1, len(self.responses)-1)]


@dataclass
class Planner:
    client: FakeClient


@dataclass
class Thor:
    state: WorldState

    def observe(self):
        return self.state


@dataclass
class Loop:
    thor: Thor
    memory: Memory
    planner: Planner


def make_session(client=None, **kwargs):
    client = client or FakeClient()
    loops = {
        0: Loop(Thor(world()), Memory(), Planner(client)),
        1: Loop(Thor(world()), Memory(), Planner(client)),
    }
    session = SocialEmbodiedSession(
        loops,
        {0: "A", 1: "B"},
        SpeechBus({0: "A", 1: "B"}),
        TextBus({0: "A", 1: "B"}),
        **kwargs,
    )
    return session, loops, client


def test_v093_discovery_doors_use_real_hinge_swing_child():
    root = Path(__file__).resolve().parents[1]
    text = (root / "unity_world" / "DiscoveryWorldBuilder.cs").read_text()
    assert 'GameObject swing = new GameObject("DoorSwing");' in text
    assert 'swing.transform.SetParent(pivot.transform, false);' in text
    assert 'leaf.transform.SetParent(swing.transform, true);' in text
    assert 'opener.MovingParts = new[] { swing };' in text
    assert 'ClearStaticRecursive(pivot);' in text
    assert 'PreparePortalClearance(NorthDoorCenter, northPivot, true' in text
    assert 'PreparePortalClearance(EastDoorCenter, eastPivot, false' in text
    assert 'CreateInvisibleBarrier(' in text
    assert 'c.enabled = false;' in text
    assert 'ConvertDoorToOpenable(northDoor, "Door|AUTHOR|NorthArchive", -90f' in text
    assert 'ConvertDoorToOpenable(eastDoor, "Door|AUTHOR|EastExperiment", 90f' in text


def test_v093_periodic_peer_text_suppresses_echo_of_recent_message():
    client = FakeClient(["I found a door. Can you check the other exit?"])
    session, loops, _ = make_session(client=client, peer_text_every=1)
    decision = type("Decision", (), {
        "observation": "I found a door.",
        "intention": "inspect it",
        "attention": "door",
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=None, decision=decision, received_texts=[], required_by_task=False
    )
    assert text
    event = session._deliver_peer_text(0, 1, text, origin=origin, count_for_task=False)
    assert event is not None

    # B's local composer attempts to parrot the exact inbound message.
    text2, origin2 = session._compose_peer_text(
        1, 0, task=None, decision=decision, received_texts=[event], required_by_task=False
    )
    assert text2 == ""
    assert origin2 == "duplicate-suppressed"


def test_v093_no_message_token_is_silent():
    client = FakeClient(["NO_MESSAGE"])
    session, loops, _ = make_session(client=client, peer_text_every=1)
    decision = type("Decision", (), {
        "observation": "Nothing changed.",
        "intention": None,
        "attention": None,
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=None, decision=decision, received_texts=[], required_by_task=False
    )
    assert text == ""
    # NO_MESSAGE is an intentional LLM choice; the fallback must not turn it into chatter.
    assert origin in {"duplicate-suppressed", "none", "llm-no-message"}


def test_v093_required_text_task_still_allows_delivery():
    client = FakeClient(["B, I found a closed door near the refrigerator."])
    session, loops, _ = make_session(client=client, peer_text_every=1)
    task = "Text B and tell B I found a closed door."
    decision = type("Decision", (), {
        "observation": "I found a closed door near the refrigerator.",
        "intention": "coordinate with B",
        "attention": "door",
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=task, decision=decision, received_texts=[], required_by_task=True
    )
    event = session._deliver_peer_text(0, 1, text, origin=origin, count_for_task=True)
    assert event is not None
    assert "door" in event.text.lower()


def test_v093_release_version_metadata():
    import agent_core
    assert agent_core.__version__ == "0.10.10"
