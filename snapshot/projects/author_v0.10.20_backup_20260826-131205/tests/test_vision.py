from agent_core.models import AgentPose, WorldState
from agent_core.vision import pose_delta


def state(x=0.0, z=0.0, yaw=0.0, horizon=0.0):
    return WorldState(
        "FloorPlan1",
        AgentPose({"x": x, "y": 0.9, "z": z}, {"x": 0.0, "y": yaw, "z": 0.0}, horizon=horizon),
        {}, None, None, True, None, True,
    )


def test_pose_delta_reports_neutral_proprioception():
    delta = pose_delta(state(), state(x=0.25, yaw=90.0, horizon=30.0))
    assert delta["distance_moved"] == 0.25
    assert delta["yaw_change_degrees"] == 90.0
    assert delta["camera_horizon_change_degrees"] == 30.0
    assert delta["position_changed"] is True
    assert delta["rotation_changed"] is True
