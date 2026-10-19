from agent_core.models import Action, ActionKind, ActionOutcome, AgentPose, ObjectState, WorldState
from agent_core.recovery import RecoveryManager


def make_world(error=None):
    return WorldState(
        scene="FloorPlan1",
        pose=AgentPose({"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}),
        objects={},
        held_object_id=None,
        last_action="PickupObject",
        last_action_success=False,
        last_error=error,
        frame_available=True,
    )


def test_interaction_failure_recovers_range():
    action = Action(ActionKind.PICKUP.value, {"objectId": "Apple|1"})
    outcome = ActionOutcome(action=action, success=False, error="Object is too far away")
    decision = RecoveryManager().classify(action, outcome, make_world(outcome.error))
    assert len(decision.actions) == 2
    assert decision.actions[0].kind == "NavigateTo"


def test_movement_failure_changes_heading():
    action = Action(ActionKind.MOVE_AHEAD.value)
    outcome = ActionOutcome(action=action, success=False, error="movement blocked")
    decision = RecoveryManager().classify(action, outcome, make_world(outcome.error))
    assert decision.actions[0].kind == "RotateRight"
