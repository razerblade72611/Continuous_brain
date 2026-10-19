from agent_core.models import AgentPose, ObjectState, WorldState
from agent_core.roam import RoamPlanner


class FakeClient:
    def __init__(self, responses):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls = 0

    def _next(self):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]

    def chat(self, *args, **kwargs):
        return self._next()

    def chat_with_image(self, *args, **kwargs):
        return self._next()


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
        visible=True,
        position={"x": 2.0, "y": 0.9, "z": 2.0},
        distance=3.0,
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


def test_roam_llm_can_choose_primitive_burst_and_pace():
    client = FakeClient(
        '{"reason":"Wander left.","pace":"fast","actions":['
        '{"kind":"RotateLeft","params":{}},'
        '{"kind":"MoveAhead","params":{}},'
        '{"kind":"MoveAhead","params":{}}]}'
    )
    planner = RoamPlanner(client, allow_interactions=False, max_burst_actions=6)
    decision = planner.decide(make_world(), [])
    assert [a.kind for a in decision.actions] == ["RotateLeft", "MoveAhead", "MoveAhead"]
    assert decision.pace == "fast"
    assert decision.fallback is False


def test_roam_llm_uses_candidate_alias_for_navigation():
    client = FakeClient(
        '{"reason":"Inspect counter.","pace":"normal","actions":['
        '{"kind":"NavigateTo","target":"c01","params":{}}]}'
    )
    planner = RoamPlanner(client, allow_interactions=False)
    decision = planner.decide(make_world(), [])
    assert decision.action.kind == "NavigateTo"
    assert decision.action.params["objectId"] == "CounterTop|1"


def test_roam_reprompts_after_invalid_alias():
    client = FakeClient([
        '{"reason":"Go.","pace":"normal","actions":[{"kind":"NavigateTo","target":"bad","params":{}}]}',
        '{"reason":"Walk instead.","pace":"normal","actions":[{"kind":"MoveAhead","params":{}}]}',
    ])
    planner = RoamPlanner(client, allow_interactions=False)
    decision = planner.decide(make_world(), [])
    assert client.calls == 2
    assert decision.fallback is False
    assert decision.action.kind == "MoveAhead"


def test_roam_reprompts_when_navigation_target_is_on_cooldown():
    # The recent Apple is deliberately pushed behind non-recent visible candidates, so it is c01 here.
    client = FakeClient([
        '{"reason":"Back to apple.","pace":"normal","actions":[{"kind":"NavigateTo","target":"c01","params":{}}]}',
        '{"reason":"Survey instead.","pace":"normal","actions":[{"kind":"RotateRight","params":{}}]}',
    ])
    planner = RoamPlanner(client, allow_interactions=False, target_cooldown=6)
    recent = [{"objectIds": ["Apple|1"], "target_names": ["Apple_1"], "success": True}]
    decision = planner.decide(make_world(), recent)
    assert client.calls == 2
    assert decision.fallback is False
    assert decision.action.kind == "RotateRight"


def test_roam_accepts_pickup_alias_when_interactions_enabled():
    client = FakeClient(
        '{"reason":"Pick it up.","pace":"slow","actions":['
        '{"kind":"PickupObject","target":"c00","params":{}}]}'
    )
    planner = RoamPlanner(client, allow_interactions=True)
    decision = planner.decide(make_world(), [])
    assert decision.action.kind == "PickupObject"
    assert decision.action.params["objectId"] == "Apple|1"


def test_roam_rejects_interaction_when_disabled_and_falls_back():
    client = FakeClient(
        '{"reason":"Grab it.","pace":"normal","actions":['
        '{"kind":"PickupObject","target":"c00","params":{}}]}'
    )
    planner = RoamPlanner(client, allow_interactions=False)
    decision = planner.decide(make_world(), [])
    assert decision.fallback is True
    assert decision.action.kind in {"RotateRight", "RotateLeft", "LookUp"}
