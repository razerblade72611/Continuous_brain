from agent_core.executor import Verifier
from agent_core.models import Action, AgentPose, ObjectState, WorldState


def world(apple, counter, held):
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose({"x": 0, "y": 0.9, "z": 0}, {"x": 0, "y": 0, "z": 0}),
        objects={apple.object_id: apple, counter.object_id: counter},
        held_object_id=held,
        last_action="PutObject",
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def make_obj(oid, kind, held=False, parents=None):
    return ObjectState(
        object_id=oid,
        object_type=kind,
        name=kind,
        visible=True,
        position={"x": 0, "y": 1, "z": 1},
        distance=1.0,
        pickupable=kind == "Apple",
        receptacle=kind == "CounterTop",
        is_picked_up=held,
        parent_receptacles=list(parents or []),
    )


def test_put_verification_requires_requested_receptacle():
    counter = make_obj("CounterTop|1", "CounterTop")
    apple_before = make_obj("Apple|1", "Apple", held=True)
    apple_wrong = make_obj("Apple|1", "Apple", parents=["DiningTable|1"])
    before = world(apple_before, counter, held="Apple|1")
    after = world(apple_wrong, counter, held=None)

    ok, details = Verifier().verify(before, after, Action("PutObject", {"objectId": "CounterTop|1"}))
    assert not ok
    assert details["source_parent_receptacles"] == ["DiningTable|1"]


def test_put_verification_passes_for_requested_receptacle():
    counter = make_obj("CounterTop|1", "CounterTop")
    apple_before = make_obj("Apple|1", "Apple", held=True)
    apple_after = make_obj("Apple|1", "Apple", parents=["CounterTop|1"])
    before = world(apple_before, counter, held="Apple|1")
    after = world(apple_after, counter, held=None)

    ok, _ = Verifier().verify(before, after, Action("PutObject", {"objectId": "CounterTop|1"}))
    assert ok
