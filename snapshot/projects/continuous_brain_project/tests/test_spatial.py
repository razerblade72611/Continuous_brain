import math
from pathlib import Path

import numpy as np

from continuous_brain.agent import ContinuousAgent
from continuous_brain.config import BrainConfig
from continuous_brain.embodied import AI2ThorConfig, AI2ThorEmbodiedEnv
from continuous_brain.environment import ForagingWorld
from continuous_brain.spatial import SpatialMemory2D
from continuous_brain.vision import NullVisionBackbone

from test_embodied import FakeController, FakeEvent


def test_spatial_depth_rejects_known_discovery_self_occlusion_range():
    sm = SpatialMemory2D(resolution=0.25)
    sm.ensure_bounds([-2, 2, -2, 2], preserve=False)
    sm.observe_pose(0.0, 0.0, 0.0, 1.0)
    depth = np.full((48, 64), 0.14, dtype=np.float32)
    integrated = sm.integrate_depth(depth, fov_degrees=90.0, min_depth=0.25)
    assert integrated == 0
    assert sm.counts()["occupied"] == 0


def test_spatial_planner_avoids_persistently_occupied_cell():
    sm = SpatialMemory2D(resolution=0.25, unknown_cost=1.2)
    sm.ensure_bounds([-1, 1, -1, 1], preserve=False)
    # Block the direct north cell while leaving room to route around it.
    sm.mark_occupied_world(0.0, 0.25, amount=3.0)
    path = sm.plan(np.array([0.0, 0.0], np.float32), np.array([0.0, 0.75], np.float32))
    assert path
    cells = [sm.world_to_cell(float(p[0]), float(p[1])) for p in path]
    assert sm.world_to_cell(0.0, 0.25) not in cells
    assert np.linalg.norm(path[-1] - np.array([0.0, 0.75], np.float32)) < 0.2


def test_spatial_state_round_trip_and_scene_guard():
    sm = SpatialMemory2D(resolution=0.25)
    sm.ensure_bounds([-1, 1, -1, 1], preserve=False)
    sm.observe_pose(0.25, -0.25, 90.0, 0.9)
    sm.mark_occupied_world(0.5, -0.25, amount=2.0)
    sm.remember_landmarks([{
        "objectId": "Mug|1", "objectType": "Mug", "visible": True,
        "position": {"x": 0.5, "z": 0.5},
    }])
    state = sm.state_dict(scene="SceneA")

    clone = SpatialMemory2D(resolution=0.25)
    assert clone.load_state_dict(state, scene="SceneA") is True
    assert clone.counts()["occupied"] >= 1
    assert clone.counts()["landmarks"] == 1
    assert abs(clone.pose.x - 0.25) < 1e-6

    other = SpatialMemory2D(resolution=0.25)
    assert other.load_state_dict(state, scene="DifferentScene") is False


def make_l2(**kwargs):
    values = dict(
        scene="Fake", curriculum_level=2, random_start=False,
        goal_signal="internal-map", rehearsal_fraction=0.0,
        spatial_target_min_distance=0.25, spatial_target_max_distance=0.75,
        spatial_min_successful_translations=1, spatial_success_radius=0.18, max_episode_steps=60,
    )
    values.update(kwargs)
    cfg = AI2ThorConfig(**values)
    return AI2ThorEmbodiedEnv(cfg, controller_factory=FakeController)


def test_l2_uses_spatial_map_waypoint_not_reachable_route_teacher():
    env = make_l2()
    try:
        # If L2 accidentally calls the v0.4.x reachable-grid route planner, reset fails.
        def forbidden(*_a, **_kw):
            raise AssertionError("reachable-grid route teacher leaked into L2")
        env._find_grid_path = forbidden
        obs = env.reset()
        assert obs.shape == (6,)
        assert env.target.level == 2
        assert env.target.kind == "spatial_point"
        assert env._spatial_waypoint is not None
        assert env._spatial_path
        _, _, _, info = env.step(0)
        assert info["waypoint_source"] == "spatial-map"
        assert info["goal_signal"] == "internal-map"
        assert info["spatial_stats"]["plans"] >= 1
    finally:
        env.close()


def test_l2_success_cannot_occur_without_real_translation():
    env = make_l2(spatial_min_successful_translations=2)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata()).copy()
        env.target.position = pos.copy()  # numerically at goal, but no locomotion yet
        env._spatial_waypoint = pos.copy()
        success, _reason = env._task_success(False)
        assert success is False
        env._episode_successful_translations = 2
        success, reason = env._task_success(False)
        assert success is True
        assert reason == "spatial_arrived"
    finally:
        env.close()


class BlockedController(FakeController):
    def step(self, action=None, **kwargs):
        if action == "MoveAhead":
            self.success = False
            self.last_action = "MoveAhead"
            self.last_event = FakeEvent(self)
            self.last_event.metadata["lastActionSuccess"] = False
            return self.last_event
        return super().step(action=action, **kwargs)


def test_l2_aligned_collision_marks_spatial_obstacle_and_replans():
    cfg = AI2ThorConfig(
        scene="Fake", curriculum_level=2, random_start=False,
        goal_signal="internal-map", rehearsal_fraction=0.0,
        spatial_target_min_distance=0.25, spatial_target_max_distance=0.75,
        spatial_min_successful_translations=1, spatial_depth_min_trusted=0.25,
        max_episode_steps=60,
    )
    env = AI2ThorEmbodiedEnv(cfg, controller_factory=BlockedController)
    try:
        env.reset()
        pos = env._position_from_metadata(env._metadata()).copy()
        goal = pos + np.array([0.0, 0.5], dtype=np.float32)  # straight ahead at yaw 0
        env.target.position = goal.copy()
        env._spatial_waypoint = pos + np.array([0.0, 0.25], dtype=np.float32)
        env._spatial_path = [env._spatial_waypoint.copy(), goal.copy()]
        before = env.spatial.counts()["occupied"]
        _, _, _, info = env.step(1)
        after = env.spatial.counts()["occupied"]
        assert info["blocked_advance"] is True
        assert info["actual_move_distance"] == 0.0
        assert after >= before + 1
        assert info["spatial_replans_episode"] >= 1
    finally:
        env.close()


def test_environment_spatial_state_persists_across_checkpoint_state():
    env = make_l2()
    clone = make_l2()
    try:
        env.reset()
        env.spatial.mark_occupied_world(0.25, 0.25, amount=10.0)
        state = env.state_dict()
        clone.load_state_dict(state)
        assert clone.spatial.counts()["occupied"] >= 1
        assert state["format_version"] >= 4
    finally:
        env.close(); clone.close()


def test_agent_mastery_snapshot_returns_true_for_stop_on_mastery(tmp_path: Path):
    env = ForagingWorld(seed=2)
    agent = ContinuousAgent(
        env, BrainConfig(seed=2), tmp_path / "brain",
        vision_backbone=NullVisionBackbone(), stop_on_mastery=True,
    )
    try:
        ready = agent._maybe_snapshot_mastered({
            "graduation_ready": True,
            "configured_curriculum_level": 0,
            "curriculum_level": 0,
            "configured_curriculum_stats": {"recent_success_rate": 1.0, "recent_attempts": 100},
        })
        assert ready is True
        assert agent.stop_on_mastery is True
        assert (tmp_path / "brain" / "skills" / "L0_mastered.pt").exists()
    finally:
        agent.episodic.close()
        env.close()


def test_routine_spatial_replanning_does_not_hide_stagnation():
    env = make_l2(spatial_replan_every_steps=1, stagnation_steps=3)
    try:
        env.reset()
        # Hold still. Planner may refresh every step, but that must not count as
        # behavioral progress and suppress the watchdog.
        done = False; info = {}
        for _ in range(3):
            _, _, done, info = env.step(0)
        assert done is True
        assert info["task_reason"] == "stagnation"
    finally:
        env.close()


def test_status_json_remains_serializable_with_numpy_spatial_map(tmp_path: Path):
    env = make_l2()
    agent = ContinuousAgent(
        env, BrainConfig(seed=5), tmp_path / "spatial_status",
        vision_backbone=NullVisionBackbone(),
    )
    try:
        agent.save()
        import json
        status = json.loads((tmp_path / "spatial_status" / "status.json").read_text())
        assert status["version"] == "0.5.3"
        assert isinstance(status["spatial"], dict)
        assert "known" in status["spatial"]
    finally:
        agent.episodic.close(); env.close()


class ImmediateMasteryEnv(ForagingWorld):
    def step(self, action):
        obs = self._obs()
        info = {
            "reached": True,
            "graduation_ready": True,
            "configured_curriculum_level": 0,
            "curriculum_level": 0,
            "configured_curriculum_stats": {"recent_attempts": 100, "recent_success_rate": 1.0},
            "curriculum_stats": {"recent_attempts": 100, "recent_success_rate": 1.0},
            "allowed_actions": list(range(self.action_dim)),
        }
        return obs, 1.0, True, info


def test_stop_on_mastery_exits_agent_run_after_confirmed_skill(tmp_path: Path):
    env = ImmediateMasteryEnv(seed=9)
    agent = ContinuousAgent(
        env, BrainConfig(seed=9, checkpoint_every=1000), tmp_path / "stop_mastery",
        vision_backbone=NullVisionBackbone(), stop_on_mastery=True,
    )
    agent.run(max_steps=50, realtime=False, log_every=1000)
    assert agent.total_steps == 1
    assert (tmp_path / "stop_mastery" / "skills" / "L0_mastered.pt").exists()


def test_native_collision_hard_lock_survives_free_depth_rays():
    sm = SpatialMemory2D(resolution=0.25)
    sm.ensure_bounds([-1.0, 1.0, -1.0, 1.0], preserve=False)
    sm.observe_pose(0.0, 0.0, 0.0)
    sm.mark_blocked_ahead(0.25)
    blocked = sm.world_to_cell(0.0, 0.25)
    assert blocked is not None and int(sm.hard_occupied[blocked]) == 1
    depth = np.full((48, 64), 5.0, dtype=np.float32)
    for _ in range(50):
        sm.integrate_depth(depth, fov_degrees=90.0, min_depth=0.25)
    assert int(sm.hard_occupied[blocked]) == 1
    assert sm._cell_cost(blocked) == float("inf")


def test_v050_spatial_state_loads_with_empty_hard_collision_mask():
    sm = SpatialMemory2D(resolution=0.25)
    sm.ensure_bounds([-1.0, 1.0, -1.0, 1.0], preserve=False)
    state = sm.state_dict(scene="Fake")
    state["format_version"] = 1
    state.pop("hard_occupied", None)
    clone = SpatialMemory2D(resolution=0.25)
    assert clone.load_state_dict(state, scene="Fake") is True
    assert np.count_nonzero(clone.hard_occupied) == 0
