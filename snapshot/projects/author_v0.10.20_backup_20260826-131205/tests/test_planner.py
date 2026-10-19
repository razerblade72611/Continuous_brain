from agent_core.models import AgentPose, ObjectState, WorldState
from agent_core.planner import RulePlanner


def make_world(held=False, apple_distance=1.0, counter_distance=1.0):
    apple = ObjectState(
        object_id="Apple|1",
        object_type="Apple",
        name="Apple_1",
        visible=True,
        position={"x": 0.5, "y": 1.0, "z": 0.5},
        distance=apple_distance,
        pickupable=True,
        is_picked_up=held,
    )
    counter = ObjectState(
        object_id="CounterTop|1",
        object_type="CounterTop",
        name="CounterTop_1",
        visible=True,
        position={"x": 1.5, "y": 1.0, "z": 0.5},
        distance=counter_distance,
        receptacle=True,
    )
    return WorldState(
        scene="FloorPlan1_physics",
        pose=AgentPose({"x": 0, "y": 0.9, "z": 0}, {"x": 0, "y": 0, "z": 0}),
        objects={apple.object_id: apple, counter.object_id: counter},
        held_object_id=apple.object_id if held else None,
        last_action=None,
        last_action_success=True,
        last_error=None,
        frame_available=True,
    )


def test_composite_planner_picks_up_before_putting():
    plan = RulePlanner(make_world(held=False)).plan("Pick up the Apple and put it on the CounterTop.")
    assert plan.actions[0].kind == "PickupObject"
    assert plan.actions[0].params["objectId"] == "Apple|1"


def test_composite_planner_puts_after_source_is_held():
    plan = RulePlanner(make_world(held=True)).plan("Pick up the Apple and put it on the CounterTop.")
    assert plan.actions[0].kind == "PutObject"
    assert plan.actions[0].params["objectId"] == "CounterTop|1"


def test_planner_navigates_when_target_is_far():
    plan = RulePlanner(make_world(held=False, apple_distance=3.0)).plan("Pick up the Apple.")
    assert plan.actions[0].kind == "NavigateTo"
    assert plan.actions[0].params["objectId"] == "Apple|1"
