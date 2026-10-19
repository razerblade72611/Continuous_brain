from __future__ import annotations

from types import SimpleNamespace

import agent_core
import agent_core.training as training
from agent_core.interaction_pose import InteractionPose


def test_release_version_is_v01011():
    assert agent_core.__version__ == "0.10.26"


def test_level3_start_branch_resolves_facing_yaw_and_faces_first_segment(monkeypatch):
    g = 0.25
    # Straight path from x=1.5 back toward the target at x=0.0, plus lateral
    # neighbors at the start so it is a maneuverable Level-3 candidate.
    reachable = [{"x": i * g, "y": 0.9, "z": 0.0} for i in range(7)]
    reachable += [
        {"x": 1.5, "y": 0.9, "z": 0.25},
        {"x": 1.5, "y": 0.9, "z": -0.25},
    ]
    pose = InteractionPose(x=0.0, y=0.9, z=0.0, rotation=0.0, horizon=0.0)
    adapter = SimpleNamespace(config=SimpleNamespace(grid_size=g, rotate_step_degrees=90.0))
    seen = {}

    def fake_teleport(shared, active, start, yaw, horizon=0.0):
        seen.update(start=start, yaw=yaw, horizon=horizon, active=active)
        return True

    monkeypatch.setattr(training, "_teleport", fake_teleport)
    ok, reason = training._prepare_agent_start(
        shared=object(), adapter=adapter, active=0,
        rng=__import__("random").Random(7), level=3,
        pose=pose, reachable=reachable,
    )
    assert ok is True
    assert reason is None
    assert seen["start"]["x"] == 1.5
    # First path step is toward -x, AI2-THOR yaw 270 degrees.
    assert seen["yaw"] == 270.0
