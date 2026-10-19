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
    w = world()
    # Direct manipulation must be within reach in v0.4.1.
    w.objects["SinkBasin|1"].distance = 1.2
    grounded = SemanticGrounder().ground("PutObject", "sink", {}, w)
    assert grounded.action.params["objectId"] == "SinkBasin|1"


def test_grounder_does_not_use_hidden_objects_as_omniscient_targets():
    with pytest.raises(GroundingError):
        SemanticGrounder().ground("PickupObject", "mug", {}, world())


def test_grounder_can_use_pronoun_only_after_prior_grounding():
    g = SemanticGrounder()
    # Establish focus without forcing unnecessary motion when already close.
    assert g.try_focus("apple", world())
    grounded = g.ground("PickupObject", "it", {}, world())
    assert grounded.action.params["objectId"] == "Apple|1"



def test_grounder_prefers_actual_rendered_viewport_when_tracking_available():
    w = world()
    # Simulate a VLM-visible object that AI2-THOR's generic `visible` flag did not mark.
    microwave = ObjectState(
        "Microwave|1", "Microwave", "Microwave_1", False,
        {"x": 1.0, "y": 1.2, "z": 2.0}, 1.2, openable=True, is_open=False,
    )
    w.objects[microwave.object_id] = microwave
    w.viewport_tracking_available = True
    w.viewport_object_ids = {"Microwave|1"}
    grounded = SemanticGrounder().ground("OpenObject", "microwave", {}, w)
    assert grounded.action.params["objectId"] == "Microwave|1"


def test_grounder_will_not_use_metadata_visible_object_outside_tracked_viewport():
    w = world()
    w.viewport_tracking_available = True
    w.viewport_object_ids = {"Sink|1"}
    with pytest.raises(GroundingError):
        SemanticGrounder().ground("PickupObject", "apple", {}, w)


def test_focus_snapshot_exists_only_after_model_selected_grounding():
    w = world()
    g = SemanticGrounder(interaction_distance=1.25)
    assert g.focus_snapshot(w) is None
    # Apple is already within approach distance, so use non-moving attention grounding.
    assert g.try_focus("apple", w)
    snap = g.focus_snapshot(w)
    assert snap["semantic_reference"] == "apple"
    assert snap["within_approach_distance"] is True
    assert "PickupObject" in snap["currently_possible_actions"]
    assert "Apple|1" not in str(snap)
    assert "Apple_1" not in str(snap)


def test_repeated_approach_is_rejected_when_already_close():
    w = world()
    with pytest.raises(GroundingError, match="already satisfied"):
        SemanticGrounder(interaction_distance=1.25).ground("Approach", "apple", {}, w)


def test_face_is_grounded_as_internal_face_target():
    w = world()
    grounded = SemanticGrounder().ground("Face", "apple", {}, w)
    assert grounded.action.kind == "FaceTarget"
    assert grounded.action.params["objectId"] == "Apple|1"
