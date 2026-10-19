import math
import numpy as np

from continuous_brain.embodied import AI2ThorConfig, AI2ThorEmbodiedEnv


class FakeEvent:
    def __init__(self, controller, action_return=None):
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.frame[..., 1] = 60
        self.depth_frame = np.full((48, 64), 2.0, dtype=np.float32)
        self.metadata = {
            "agent": {
                "position": {"x": controller.x, "y": 0.0, "z": controller.z},
                "rotation": {"x": 0.0, "y": controller.yaw, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "lastActionSuccess": controller.success,
            "objects": controller.objects(),
            "actionReturn": action_return,
        }


class FakeController:
    def __init__(self, **kwargs):
        self.x = 0.0; self.z = 0.0; self.yaw = 0.0; self.success = True
        self.door_open = False; self.lamp_on = False; self.stopped = False
        self.last_event = FakeEvent(self)

    def objects(self):
        return [
            {"objectId": "Door|1", "objectType": "Door", "position": {"x": 0.0, "y": 0.0, "z": 1.0}, "visible": True, "openable": True, "isOpen": self.door_open, "toggleable": False, "pickupable": False},
            {"objectId": "Lamp|1", "objectType": "DeskLamp", "position": {"x": 1.0, "y": 0.0, "z": 0.5}, "visible": True, "openable": False, "toggleable": True, "isToggled": self.lamp_on, "pickupable": False},
        ]

    def reset(self, scene=None):
        self.x = 0.0; self.z = 0.0; self.yaw = 0.0; self.success = True
        self.last_event = FakeEvent(self); return self.last_event

    def step(self, action=None, **kwargs):
        self.success = True
        if action == "GetReachablePositions":
            pts = [{"x": x, "y": 0.0, "z": z} for x in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)]
            self.last_event = FakeEvent(self, action_return=pts); return self.last_event
        if action == "TeleportFull":
            self.x = float(kwargs.get("x", self.x)); self.z = float(kwargs.get("z", self.z)); self.yaw = float((kwargs.get("rotation") or {}).get("y", self.yaw))
        elif action == "MoveAhead":
            mag = float(kwargs.get("moveMagnitude", 0.25)); a = math.radians(self.yaw); self.x += math.sin(a) * mag; self.z += math.cos(a) * mag
        elif action == "RotateLeft": self.yaw = (self.yaw - float(kwargs.get("degrees", 30))) % 360
        elif action == "RotateRight": self.yaw = (self.yaw + float(kwargs.get("degrees", 30))) % 360
        elif action == "OpenObject": self.door_open = True
        elif action == "ToggleObjectOn": self.lamp_on = True
        self.last_event = FakeEvent(self); return self.last_event

    def stop(self): self.stopped = True


def make_env(level=1, goal_signal="teacher-nav"):
    return AI2ThorEmbodiedEnv(
        AI2ThorConfig(scene="Fake", curriculum_level=level, random_start=False, goal_signal=goal_signal, max_episode_steps=20),
        controller_factory=FakeController,
    )


def test_thor_adapter_preserves_v03_brain_interface():
    env = make_env(1)
    try:
        obs = env.reset()
        assert obs.shape == (6,)
        assert env.action_dim == 5
        assert env.action_names == ("idle", "advance", "turn_left", "turn_right", "interact")
        next_obs, reward, done, info = env.step(1)
        assert next_obs.shape == (6,)
        assert isinstance(reward, float)
        assert "primitive_action" in info
        snap = env.visual_snapshot()
        assert snap["environment_type"] == "ai2thor"
        assert snap["camera"]["width"] > 0
        assert len(snap["world_bounds"]) == 4
    finally:
        env.close()


def test_sensor_only_masks_teacher_goal_slots():
    env = make_env(1, goal_signal="sensor-only")
    try:
        obs = env.reset()
        assert obs[0] == 0.0
        assert obs[1] == 0.0
    finally:
        env.close()


def test_door_curriculum_can_interact():
    env = make_env(6)
    try:
        env.reset()
        # L6 selects the door in this fake scene. Interaction is allowed even if the
        # fake target is not yet spatially realistic; the test verifies action mapping.
        _, _, _, info = env.step(4)
        assert info["primitive_action"] in {"OpenObject", "Pass"}
        assert env.controller.door_open is True
    finally:
        env.close()
