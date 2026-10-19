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
        self.door_open = False; self.lamp_on = False; self.stopped = False; self.last_action = "Pass"
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
        self.last_action = str(action)
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


def make_env(level=1, goal_signal="teacher-nav", **kwargs):
    return AI2ThorEmbodiedEnv(
        AI2ThorConfig(scene="Fake", curriculum_level=level, random_start=False, goal_signal=goal_signal, max_episode_steps=20, **kwargs),
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


def test_curriculum_action_masks():
    expected = {
        0: (0, 2, 3),
        1: (0, 1, 2, 3),
        5: (0, 1, 2, 3),
        6: (0, 1, 2, 3, 4),
        9: (0, 1, 2, 3, 4),
    }
    for level, mask in expected.items():
        env = make_env(level)
        try:
            assert env.valid_action_indices() == mask
        finally:
            env.close()


def test_l0_hard_masks_translation_and_keeps_position_locked():
    env = make_env(0)
    try:
        env.reset()
        start = env._position_from_metadata(env._metadata()).copy()
        _, _, _, info = env.step(1)  # invalid ADVANCE from an external caller
        end = env._position_from_metadata(env._metadata()).copy()
        assert np.allclose(start, end)
        assert info["executed_action"] == 0
        assert info["primitive_action"] == "Pass"
        assert 1 not in info["allowed_actions"]
    finally:
        env.close()


def test_l0_angular_shaping_rewards_correct_turn_over_wrong_turn():
    correct = make_env(0, orient_success_degrees=5.0)
    wrong = make_env(0, orient_success_degrees=5.0)
    try:
        correct.reset(); wrong.reset()
        # Force the same target: +X at yaw 0 => positive/right bearing.
        for env in (correct, wrong):
            pos = env._position_from_metadata(env._metadata())
            env.target.position = pos + np.array([1.0, 0.0], dtype=np.float32)
            env._best_progress_metric = abs(env._goal_local()[3])
        _, reward_correct, _, info_correct = correct.step(3)
        _, reward_wrong, _, info_wrong = wrong.step(2)
        assert info_correct["delta_bearing"] > 0
        assert info_wrong["delta_bearing"] < 0
        assert reward_correct > reward_wrong
    finally:
        correct.close(); wrong.close()


def test_stagnation_terminates_and_records_statistics():
    env = make_env(0, stagnation_steps=3, orient_success_degrees=5.0)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        env.target.position = pos + np.array([1.0, 0.0], dtype=np.float32)
        env._best_progress_metric = abs(env._goal_local()[3])
        done = False; info = {}
        for _ in range(3):
            _, _, done, info = env.step(0)
        assert done is True
        assert info["task_reason"] == "stagnation"
        assert info["curriculum_stats"]["stagnations_total"] == 1
        assert info["curriculum_stats"]["recent_attempts"] == 1
    finally:
        env.close()


def test_curriculum_statistics_state_round_trip():
    env = make_env(0, stagnation_steps=1, orient_success_degrees=5.0)
    clone = make_env(0, stagnation_steps=1, orient_success_degrees=5.0)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        env.target.position = pos + np.array([1.0, 0.0], dtype=np.float32)
        env._best_progress_metric = abs(env._goal_local()[3])
        env.step(0)
        state = env.state_dict()
        clone.load_state_dict(state)
        assert clone.curriculum_attempts == env.curriculum_attempts == 1
        assert clone.curriculum_stagnations == env.curriculum_stagnations == 1
        assert clone._curriculum_stats()["recent_attempts"] == 1
    finally:
        env.close(); clone.close()
