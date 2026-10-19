from agent_core.models import AgentPose, ObjectState, WorldState
from agent_core.roam import RoamPlanner


class FakeClient:
    def __init__(self, response: str):
        self.response = response

    def chat(self, *args, **kwargs):
        return self.response

    def chat_with_image(self, *args, **kwargs):
        return self.response


def make_world():
    apple = ObjectState(
        object_id="Apple|1",
        object_type="Apple",
        name="Apple_1",
        visible=True,
        position={"x": 1.0, "y": 0.8, "z": 1.0},
        distance=2.0,
        pickupable=True,
    )
    counter = ObjectState(
        object_id="CounterTop|1",
        object_type="CounterTop",
        name="CounterTop_1",
        visible=False,
        position={"x": 2.0, "y": 0.9, "z": 2.0},
        distance=3.0,
        receptacle=True,
    )
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            horizon=0.0,
        ),
        objects={apple.object_id: apple, counter.object_id: counter},
        held_object_id=None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_roam_llm_can_choose_navigation_target():
    client = FakeClient(
        '{"reason":"Inspect the counter area.","action":{"kind":"NavigateTo","params":{"objectId":"CounterTop|1"}}}'
    )
    planner = RoamPlanner(client, allow_interactions=False)
    decision = planner.decide(make_world(), [])
    assert decision.action.kind == "NavigateTo"
    assert decision.action.params["objectId"] == "CounterTop|1"
    assert decision.action.params["interactionDistance"] == 1.25
    assert decision.fallback is False


def test_roam_rejects_interaction_when_disabled_and_falls_back():
    client = FakeClient(
        '{"reason":"Grab it.","action":{"kind":"PickupObject","params":{"objectId":"Apple|1"}}}'
    )
    planner = RoamPlanner(client, allow_interactions=False)
    decision = planner.decide(make_world(), [])
    assert decision.fallback is True
    assert decision.action.kind in {"NavigateTo", "RotateRight"}


def test_roam_accepts_pickup_when_interactions_enabled():
    client = FakeClient(
        '{"reason":"Inspect by holding it.","action":{"kind":"PickupObject","params":{"objectId":"Apple|1"}}}'
    )
    planner = RoamPlanner(client, allow_interactions=True)
    decision = planner.decide(make_world(), [])
    assert decision.action.kind == "PickupObject"
    assert decision.action.params["objectId"] == "Apple|1"
