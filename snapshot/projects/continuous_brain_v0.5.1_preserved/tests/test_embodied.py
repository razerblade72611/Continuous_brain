import math
import numpy as np

from continuous_brain.embodied import AI2ThorConfig, AI2ThorEmbodiedEnv


class FakeEvent:
    def __init__(self, controller, action_return=None):
        yy, xx = np.mgrid[0:48, 0:64]
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        # Make the synthetic camera depend on pose/yaw so the sensor-integrity
        # watchdog sees the same kind of frame motion expected from AI2-THOR.
        self.frame[..., 0] = (xx + int(round(controller.x * 31.0))) % 256
        self.frame[..., 1] = (yy * 3 + int(round(controller.z * 29.0))) % 256
        self.frame[..., 2] = (int(round(controller.yaw)) + xx // 2) % 256
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
            pts = [{"x": x, "y": 0.0, "z": z} for x in (-0.50, -0.25, 0.0, 0.25, 0.50) for z in (-0.50, -0.25, 0.0, 0.25, 0.50)]
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
            assert env.valid_action_indices_for_level(level) == mask
            if level != 1:
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


def test_v042_level_statistics_are_isolated():
    env = make_env(0, rehearsal_fraction=0.0, stagnation_steps=1, orient_success_degrees=5.0)
    clone = make_env(1, rehearsal_fraction=0.0)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        env.target.position = pos + np.array([1.0, 0.0], dtype=np.float32)
        env._best_progress_metric = abs(env._goal_local()[3])
        env.step(0)  # stagnation => one L0 outcome
        clone.load_state_dict(env.state_dict())
        assert clone._curriculum_stats(0)["recent_attempts"] == 1
        assert clone._curriculum_stats(1)["recent_attempts"] == 0
    finally:
        env.close(); clone.close()


def test_v041_legacy_statistics_do_not_seed_l1_graduation():
    env = make_env(1, rehearsal_fraction=0.0)
    try:
        env.load_state_dict({
            "curriculum_attempts": 100,
            "curriculum_successes": 100,
            "recent_outcomes": [{"success": True, "steps": 4}] * 100,
        })
        assert env._curriculum_stats(1)["recent_attempts"] == 0
        assert env._legacy_v041_stats["curriculum_attempts"] == 100
    finally:
        env.close()


def test_l1_short_teacher_path_uses_local_waypoints():
    env = make_env(1, rehearsal_fraction=0.0, nav_initial_path_steps=2, nav_max_path_steps=2)
    try:
        env.reset()
        assert env.target.level == 1
        assert env.target.waypoints
        assert len(env.target.waypoints) == 2
        assert np.allclose(env.target.position, env.target.waypoints[-1])
        active = env._active_goal_position()
        assert np.allclose(active, env.target.waypoints[0])
    finally:
        env.close()


def test_l1_angular_reward_preserves_l0_orientation_skill():
    correct = make_env(1, rehearsal_fraction=0.0, nav_angular_scale=0.003)
    wrong = make_env(1, rehearsal_fraction=0.0, nav_angular_scale=0.003)
    try:
        for env in (correct, wrong):
            env.reset()
            pos = env._position_from_metadata(env._metadata())
            goal = pos + np.array([1.0, 0.0], dtype=np.float32)  # right at yaw 0
            env.target.position = goal.copy()
            env.target.waypoints = [goal.copy()]
            env.target.waypoint_index = 0
        _, rc, _, ic = correct.step(3)
        _, rw, _, iw = wrong.step(2)
        assert ic["delta_bearing"] > 0
        assert iw["delta_bearing"] < 0
        assert rc > rw
    finally:
        correct.close(); wrong.close()


def test_l1_forward_progress_and_actual_motion_are_reported():
    env = make_env(1, rehearsal_fraction=0.0)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([0.0, 1.0], dtype=np.float32)
        env.target.position = goal.copy()
        env.target.waypoints = [goal.copy()]
        env.target.waypoint_index = 0
        _, reward, _, info = env.step(1)
        assert info["primitive_action"] == "MoveAhead"
        assert info["actual_move_distance"] > 0.20
        assert info["delta_distance"] > 0.20
        assert reward > 0.0
    finally:
        env.close()


def test_l1_useful_turn_resets_stagnation_even_without_distance_change():
    env = make_env(1, rehearsal_fraction=0.0, stagnation_steps=3)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([1.0, 0.0], dtype=np.float32)
        env.target.position = goal.copy(); env.target.waypoints = [goal.copy()]
        env._route_unavailable = False
        env._stagnation_count = 2
        _, _, done, info = env.step(3)  # correct right turn
        assert done is False
        assert info["delta_distance"] == 0.0
        assert info["delta_bearing"] > 0.0
        assert info["stagnation_steps"] == 0
    finally:
        env.close()


def test_new_level_exploration_floor_restarts():
    env = make_env(
        1, rehearsal_fraction=0.0,
        level_epsilon_start=0.40, level_epsilon_end=0.10,
        level_epsilon_decay_steps=100,
    )
    try:
        env._level_action_steps[1] = 0
        assert abs(env.exploration_floor() - 0.40) < 1e-6
        env._level_action_steps[1] = 50
        assert abs(env.exploration_floor() - 0.25) < 1e-6
        env._level_action_steps[1] = 100
        assert abs(env.exploration_floor() - 0.10) < 1e-6
    finally:
        env.close()


def test_l1_one_cell_target_cannot_succeed_without_translation():
    env = make_env(1, rehearsal_fraction=0.0, nav_success_radius=0.12)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([0.0, 0.25], dtype=np.float32)
        env.target.position = goal.copy(); env.target.waypoints = [goal.copy()]
        env.target.waypoint_index = 0; env._route_unavailable = False
        _, _, done, info = env.step(0)
        assert done is False
        assert info["reached"] is False
        assert info["successful_translations_episode"] == 0
    finally:
        env.close()


def test_l1_alignment_gate_masks_advance_until_facing_waypoint():
    env = make_env(1, rehearsal_fraction=0.0, nav_alignment_gate_degrees=15.0)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        right_goal = pos + np.array([0.25, 0.0], dtype=np.float32)
        env.target.position = right_goal.copy(); env.target.waypoints = [right_goal.copy()]
        env._route_unavailable = False
        assert 1 not in env.valid_action_indices()
        env.controller.yaw = 90.0; env.event = FakeEvent(env.controller)
        assert 1 in env.valid_action_indices()
    finally:
        env.close()


def test_l1_native_collision_training_does_not_preempt_with_depth_guard():
    env = make_env(1, rehearsal_fraction=0.0, nav_native_collision_training=True)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([0.0, 0.25], dtype=np.float32)
        env.target.position = goal.copy(); env.target.waypoints = [goal.copy()]
        env._route_unavailable = False
        env.event.depth_frame[:] = 0.05
        _, _, _, info = env.step(1)
        assert info["primitive_action"] == "MoveAhead"
        assert info["actual_move_distance"] > 0.20
    finally:
        env.close()


def test_l1_optional_depth_guard_remains_available():
    env = make_env(1, rehearsal_fraction=0.0, nav_native_collision_training=False, safety_distance=0.20)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([0.0, 0.25], dtype=np.float32)
        env.target.position = goal.copy(); env.target.waypoints = [goal.copy()]
        env._route_unavailable = False
        env.event.depth_frame[:] = 0.05
        _, _, _, info = env.step(1)
        assert info["primitive_action"] == "SafetyStop"
        assert info["actual_move_distance"] == 0.0
    finally:
        env.close()


def test_l1_success_requires_real_move_and_reports_skill_goal():
    env = make_env(1, rehearsal_fraction=0.0, nav_success_radius=0.12)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        goal = pos + np.array([0.0, 0.25], dtype=np.float32)
        env.target.position = goal.copy(); env.target.waypoints = [goal.copy()]
        env._route_unavailable = False
        _, reward, done, info = env.step(1)
        assert done is True and info["reached"] is True
        assert info["successful_translation"] is True
        assert info["curriculum_stats"]["goals_total"] == 1
        assert reward > 0.5
    finally:
        env.close()


class BlockingController(FakeController):
    def step(self, action=None, **kwargs):
        if action == "MoveAhead" and abs(self.x) < 1e-6 and abs(self.z) < 1e-6 and abs((self.yaw % 360.0) - 0.0) < 1e-6:
            self.success = False
            self.last_action = str(action)
            self.last_event = FakeEvent(self)
            return self.last_event
        return super().step(action=action, **kwargs)


def test_l1_aligned_failed_edge_triggers_alternate_replan():
    env = AI2ThorEmbodiedEnv(
        AI2ThorConfig(
            scene="Fake", curriculum_level=1, random_start=False, rehearsal_fraction=0.0,
            nav_initial_path_steps=2, nav_min_path_steps=2, nav_max_path_steps=2,
            nav_edge_failure_threshold=2, nav_native_collision_training=True,
        ),
        controller_factory=BlockingController,
    )
    try:
        env.reset()
        env.controller.x = 0.0; env.controller.z = 0.0; env.controller.yaw = 0.0
        env.event = FakeEvent(env.controller)
        env.target.position = np.array([0.0, 0.50], dtype=np.float32)
        env.target.waypoints = [np.array([0.0, 0.25], dtype=np.float32), np.array([0.0, 0.50], dtype=np.float32)]
        env.target.waypoint_index = 0; env._route_unavailable = False
        _, reward, done, info = env.step(1)
        assert info["primitive_action"] == "MoveAhead"
        assert info["blocked_advance"] is True
        assert info["route_replanned"] is True
        assert info["replans_episode"] == 1
        assert info["route_unavailable"] is False
        assert done is False
        assert reward < 0.0
        # Alternate route should no longer begin with the failed +Z edge.
        assert not np.allclose(env.target.waypoints[0], np.array([0.0, 0.25], dtype=np.float32))
    finally:
        env.close()


class FrozenRGBController(FakeController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frozen = self.last_event.frame.copy()

    def _freeze(self, event):
        event.frame = self._frozen.copy()
        return event

    def reset(self, scene=None):
        event = super().reset(scene=scene)
        if hasattr(self, "_frozen"):
            event.frame = self._frozen.copy()
        return event

    def step(self, action=None, **kwargs):
        return self._freeze(super().step(action=action, **kwargs))


def test_policy_depth_rejects_near_zero_discovery_band():
    env = make_env(1, rehearsal_fraction=0.0, sensor_depth_min_trusted=0.25)
    try:
        env.reset()
        env.event.depth_frame[:] = 0.14
        obs = env._obs()
        # Invalid near-zero depth becomes unknown/open rather than a false obstacle.
        assert obs[2] == 1.0
        snap = env.visual_snapshot()
        assert snap["sensor_health"]["depth_healthy"] is False
        assert snap["sensor_health"]["depth_valid_fraction"] == 0.0
    finally:
        env.close()


def test_rgb_watchdog_detects_frozen_frame_after_pose_changes():
    env = AI2ThorEmbodiedEnv(
        AI2ThorConfig(
            scene="Fake", curriculum_level=0, random_start=False, rehearsal_fraction=0.0,
            orient_success_degrees=1.0, sensor_rgb_stale_motion_events=2, sensor_watchdog=True,
        ),
        controller_factory=FrozenRGBController,
    )
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata())
        env.target.position = pos + np.array([1.0, 0.0], dtype=np.float32)
        env._best_progress_metric = abs(env._goal_local()[3])
        _o, _r, d1, i1 = env.step(3)
        assert i1["sensor_health"]["rgb_stale"] is False
        _o, _r, d2, i2 = env.step(3)
        assert d2 is True
        assert i2["sensor_fault"] is True
        assert i2["task_reason"] == "sensor_stale"
        assert i2["sensor_health"]["rgb_stale"] is True
    finally:
        env.close()


def test_camera_snapshot_exposes_sequence_hash_and_health():
    env = make_env(0)
    try:
        env.reset()
        first = env.visual_snapshot()
        env.step(3)
        second = env.visual_snapshot()
        assert second["camera"]["seq"] > first["camera"]["seq"]
        assert second["camera"]["hash"] != first["camera"]["hash"]
        assert "rgb_diff" in second["sensor_health"]
    finally:
        env.close()
