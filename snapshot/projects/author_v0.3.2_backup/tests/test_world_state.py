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
