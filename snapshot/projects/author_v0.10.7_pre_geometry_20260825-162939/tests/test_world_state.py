from types import SimpleNamespace

from agent_core.world_state import normalize_event


def test_normalize_event():
    event = SimpleNamespace(
        metadata={
            "agent": {
                "position": {"x": 1, "y": 2, "z": 3},
                "rotation": {"x": 0, "y": 90, "z": 0},
                "cameraHorizon": 0,
            },
            "objects": [
                {
                    "objectId": "Apple|1",
                    "objectType": "Apple",
                    "name": "Apple",
                    "visible": True,
                    "distance": 1.2,
                    "pickupable": True,
                    "receptacle": False,
                    "openable": False,
                    "isPickedUp": True,
                    "parentReceptacles": ["Bowl|1"],
                }
            ],
            "lastAction": "PickupObject",
            "lastActionSuccess": True,
        },
        frame=b"123",
    )

    state = normalize_event(event, "FloorPlan1", 4)
    assert state.scene == "FloorPlan1"
    assert state.held_object_id == "Apple|1"
    assert state.objects["Apple|1"].pickupable is True
    assert state.step_index == 4



def test_normalize_event_captures_hidden_viewport_ids_without_changing_object_visibility():
    class Event:
        frame = object()
        instance_detections2D = {"Microwave|1": [10, 10, 20, 20]}
        instance_masks = None
        metadata = {
            "sceneName": "FloorPlan1",
            "agent": {"position": {"x": 0, "y": 0.9, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}},
            "objects": [{
                "objectId": "Microwave|1", "objectType": "Microwave", "name": "Microwave_1",
                "visible": False, "position": {"x": 1, "y": 1, "z": 1}, "distance": 1.0,
            }],
            "lastActionSuccess": True,
        }

    state = normalize_event(Event(), "FloorPlan1")
    assert state.viewport_tracking_available is True
    assert state.viewport_object_ids == {"Microwave|1"}
    assert state.objects["Microwave|1"].visible is False
