import pytest

from agent_core.grounding import GroundingError, SemanticGrounder
from agent_core.models import AgentPose, ObjectState, WorldState


def world():
    objects = [
        ObjectState("Apple|1", "Apple", "Apple_1", True, {"x": 1, "y": 1, "z": 1}, 1.0, pickupable=True),
        ObjectState("Sink|1", "Sink", "Sink_1", True, {"x": 2, "y": 1, "z": 1}, 2.0),
        ObjectState("SinkBasin|1", "SinkBasin", "SinkBasin_1", True, {"x": 2, "y": 1, "z": 1}, 2.1, receptacle=True),
        ObjectState("Fridge|1", "Fridge", "Fridge_1", True, {"x": 3, "y": 1, "z": 1}, 2.5, openable=True, is_open=False),
        ObjectState("Mug|hidden", "Mug", "Mug_hidden", False, {"x": 4, "y": 1, "z": 1}, 3.0, pickupable=True),
    ]
    return WorldState(
        "FloorPlan1",
        AgentPose({"x": 0, "y": 0.9, "z": 0}, {"x": 0, "y": 0, "z": 0}),
        {o.object_id: o for o in objects},
        None, None, True, None, True,
    )


def test_grounder_maps_semantic_pickup_to_visible_object():
    grounded = SemanticGrounder().ground("PickupObject", "apple", {}, world())
    assert grounded.action.params["objectId"] == "Apple|1"


def test_grounder_filters_put_target_to_receptacle():
    grounded = SemanticGrounder().ground("PutObject", "sink", {}, world())
    assert grounded.action.params["objectId"] == "SinkBasin|1"


def test_grounder_does_not_use_hidden_objects_as_omniscient_targets():
    with pytest.raises(GroundingError):
        SemanticGrounder().ground("PickupObject", "mug", {}, world())


def test_grounder_can_use_pronoun_only_after_prior_grounding():
    g = SemanticGrounder()
    g.ground("Approach", "apple", {}, world())
    grounded = g.ground("PickupObject", "it", {}, world())
    assert grounded.action.params["objectId"] == "Apple|1"
