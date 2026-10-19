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
    def __init__(self, response: str = "I found a door. Can you check the other side?"):
        self.response = response
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        return self.response


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


def test_v092_walkable_generated_surfaces_use_exact_ai2thor_floor_name():
    root = Path(__file__).resolve().parents[1]
    text = (root / "unity_world" / "DiscoveryWorldBuilder.cs").read_text()
    assert 'CreateWalkableFloor("NorthFloor"' in text
    assert 'CreateWalkableFloor("NorthThreshold"' in text
    assert 'CreateWalkableFloor("EastFloor"' in text
    assert 'CreateWalkableFloor("EastThreshold"' in text
    assert 'go.name = "Floor";' in text
    assert 'go.tag = "Structure";' in text
    assert 'AI2-THOR 4.3.0 GetReachablePositions' in text


def test_v092_periodic_peer_text_is_llm_authored_and_delivered():
    session, loops, client = make_session(peer_text_every=3)
    decision = type("Decision", (), {
        "observation": "I can see a closed door.",
        "intention": "inspect the doorway",
        "attention": "door",
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=None, decision=decision, received_texts=[], required_by_task=False
    )
    assert origin == "llm-peer-compose"
    assert "door" in text.lower()
    event = session._deliver_peer_text(0, 1, text, origin=origin, count_for_task=False)
    assert event is not None
    assert session.text_bus.peek(1)[0].text == text
    assert client.calls
    # The compose call contains A's own state, never B's private state.
    _, user, _ = client.calls[-1]
    assert "I can see a closed door" in user
    assert "private" not in user.lower()


def test_v092_text_task_recovery_delivery_can_count_for_completion():
    session, loops, client = make_session()
    task = "Text B and ask what B can see."
    session.controller.state.tasks[0] = task
    session._tracked_tasks[0] = task
    session._task_text_baseline[0] = 0
    decision = type("Decision", (), {
        "observation": "I am by the refrigerator.",
        "intention": "contact B",
        "attention": "B",
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=task, decision=decision, received_texts=[], required_by_task=True
    )
    event = session._deliver_peer_text(0, 1, text, origin=origin, count_for_task=True)
    assert event is not None
    session._complete_task_if_needed(0, claimed=False)
    assert session.controller.state.tasks[0] is None


def test_v092_status_relay_fallback_is_visible_and_not_claimed_as_llm_authored():
    class BrokenClient:
        def chat(self, *args, **kwargs):
            raise RuntimeError("offline")

    session, loops, _ = make_session(client=BrokenClient(), peer_text_fallback=True)
    decision = type("Decision", (), {
        "observation": "I see the east doorway.",
        "intention": None,
        "attention": "door",
    })()
    text, origin = session._compose_peer_text(
        0, 1, task=None, decision=decision, received_texts=[], required_by_task=False
    )
    assert origin == "status-relay-fallback"
    assert text.startswith("[status relay]")
    assert "east doorway" in text


def test_v092_release_version_metadata():
    import agent_core
    assert agent_core.__version__ == "0.10.24"
