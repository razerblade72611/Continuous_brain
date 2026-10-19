from agent_core.goal_verifier import goal_satisfied
from agent_core.goals import parse_goal
from agent_core.models import AgentPose, ObjectState, WorldState


def make_world(*objects, held=None):
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose({"x": 0, "y": 0.9, "z": 0}, {"x": 0, "y": 0, "z": 0}),
        objects={obj.object_id: obj for obj in objects},
        held_object_id=held,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def obj(oid, kind, *, held=False, parents=None, visible=True, distance=1.0, receptacle=False):
    return ObjectState(
        object_id=oid,
        object_type=kind,
        name=kind,
        visible=visible,
        position={"x": 1.0, "y": 1.0, "z": 1.0},
        distance=distance,
        pickupable=kind == "Apple",
        receptacle=receptacle,
        is_picked_up=held,
        parent_receptacles=list(parents or []),
    )


def test_parse_composite_goal_resolves_pronoun():
    spec = parse_goal("Pick up the Apple and put it on the CounterTop.")
    assert spec.pickup_object == "Apple"
    assert spec.put_object == "Apple"
    assert spec.put_target == "CounterTop"
    assert spec.source_object == "Apple"


def test_composite_goal_not_satisfied_when_apple_only_held():
    apple = obj("Apple|1", "Apple", held=True)
    counter = obj("CounterTop|1", "CounterTop", receptacle=True)
    world = make_world(apple, counter, held=apple.object_id)
    assert not goal_satisfied("Pick up the Apple and put it on the CounterTop.", world)


def test_composite_goal_satisfied_only_when_parent_receptacle_matches():
    counter = obj("CounterTop|1", "CounterTop", receptacle=True)
    apple = obj("Apple|1", "Apple", parents=[counter.object_id])
    world = make_world(apple, counter)
    assert goal_satisfied("Pick up the Apple and put it on the CounterTop.", world)
