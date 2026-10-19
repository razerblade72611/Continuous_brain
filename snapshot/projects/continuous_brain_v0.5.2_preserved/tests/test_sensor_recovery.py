import math
from pathlib import Path

import numpy as np

from continuous_brain.agent import ContinuousAgent
from continuous_brain.config import BrainConfig
from continuous_brain.embodied import AI2ThorConfig, AI2ThorEmbodiedEnv
from continuous_brain.environment import Environment
from continuous_brain.vision import NullVisionBackbone


class Event:
    def __init__(self, controller, *, frozen=False):
        h, w = 48, 64
        if frozen:
            self.frame = np.full((h, w, 3), 77, dtype=np.uint8)
            self.depth_frame = np.zeros((h, w), dtype=np.float32)
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            self.frame = np.zeros((h, w, 3), dtype=np.uint8)
            self.frame[..., 0] = (xx + int(controller.yaw)) % 255
            self.frame[..., 1] = (yy * 3 + int(controller.x * 50)) % 255
            self.frame[..., 2] = (int(controller.yaw * 2) + xx) % 255
            self.depth_frame = np.full((h, w), 2.0, dtype=np.float32)
        self.metadata = {
            "agent": {"position": {"x": controller.x, "y": 0.0, "z": controller.z},
                      "rotation": {"x": 0.0, "y": controller.yaw, "z": 0.0},
                      "cameraHorizon": 0.0},
            "lastActionSuccess": True,
            "objects": [],
            "actionReturn": None,
        }


class RecoverableController:
    def __init__(self, *, frozen=False, **kwargs):
        self.frozen = frozen
        self.x = 0.0; self.z = 0.0; self.yaw = 0.0
        self.stopped = False
        self.last_event = Event(self, frozen=self.frozen)

    def reset(self, scene=None):
        self.x = 0.0; self.z = 0.0; self.yaw = 0.0
        self.last_event = Event(self, frozen=self.frozen)
        return self.last_event

    def step(self, action=None, **kwargs):
        if action == "GetReachablePositions":
            ev = Event(self, frozen=self.frozen)
            ev.metadata["actionReturn"] = [
                {"x": x, "y": 0.0, "z": z}
                for x in (-0.5, -0.25, 0.0, 0.25, 0.5)
                for z in (-0.5, -0.25, 0.0, 0.25, 0.5)
            ]
            self.last_event = ev
            return ev
        if action == "RotateRight":
            self.yaw = (self.yaw + float(kwargs.get("degrees", 30.0))) % 360.0
        elif action == "RotateLeft":
            self.yaw = (self.yaw - float(kwargs.get("degrees", 30.0))) % 360.0
        elif action == "MoveAhead":
            a = math.radians(self.yaw); mag = float(kwargs.get("moveMagnitude", 0.25))
            self.x += math.sin(a) * mag; self.z += math.cos(a) * mag
        elif action == "TeleportFull":
            self.x = float(kwargs.get("x", self.x)); self.z = float(kwargs.get("z", self.z))
            self.yaw = float((kwargs.get("rotation") or {}).get("y", self.yaw))
        self.last_event = Event(self, frozen=self.frozen)
        return self.last_event

    def stop(self):
        self.stopped = True


class RestartFactory:
    def __init__(self):
        self.instances = []
    def __call__(self, **kwargs):
        # First controller is permanently frozen. A recreated controller is healthy.
        c = RecoverableController(frozen=(len(self.instances) == 0), **kwargs)
        self.instances.append(c)
        return c


def test_controller_restart_recovers_frozen_rgb_and_depth():
    factory = RestartFactory()
    env = AI2ThorEmbodiedEnv(AI2ThorConfig(
        scene="Fake", curriculum_level=2, random_start=False, rehearsal_fraction=0.0,
        sensor_soft_reset_attempts=1, sensor_controller_restart_attempts=2,
        sensor_recovery_verify_rotations=1, sensor_recovery_min_depth_valid_fraction=0.01,
    ), controller_factory=factory)
    try:
        # Constructor uses the frozen first controller. Recovery must escalate because
        # scene reset cannot repair a controller whose event stream is permanently frozen.
        result = env.recover_sensor_fault()
        assert result["recovered"] is True
        assert result["method"] == "controller-restart"
        assert len(factory.instances) >= 2
        assert factory.instances[0].stopped is True
        health = env._sensor_health_payload()
        assert health["controller_restarts_total"] == 1
        assert health["last_recovery"] == "controller-restart"
        assert health["depth_healthy"] is True
    finally:
        env.close()


class InfraFaultEnv(Environment):
    observation_dim = 6
    action_dim = 5
    action_names = ("idle", "a", "b", "c", "d")
    def __init__(self):
        self.n = 0; self.recovered = 0; self.energy = 1.0
    def reset(self): return np.zeros(6, dtype=np.float32)
    def step(self, action):
        self.n += 1
        if self.n == 1:
            return np.ones(6, dtype=np.float32), -1.0, True, {
                "sensor_fault": True, "infrastructure_fault": True,
                "curriculum_level": 2, "configured_curriculum_level": 2,
                "allowed_actions": [0,1,2,3], "reached": False,
            }
        return np.full(6, 0.5, dtype=np.float32), 0.25, False, {
            "sensor_fault": False, "infrastructure_fault": False,
            "curriculum_level": 2, "configured_curriculum_level": 2,
            "allowed_actions": [0,1,2,3], "reached": False,
        }
    def recover_sensor_fault(self):
        self.recovered += 1
        return {"attempted": True, "recovered": True, "method": "controller-restart", "fatal": False}
    def render_rgb(self, size=96): return np.zeros((32,32,3), dtype=np.uint8)
    def describe(self): return "infra-test"


def test_agent_quarantines_infrastructure_fault_transition(tmp_path: Path):
    env = InfraFaultEnv()
    cfg = BrainConfig(seed=3, batch_size=32, learn_every=1000, consolidate_every=1000, checkpoint_every=1000)
    agent = ContinuousAgent(env, cfg, tmp_path / "state", vision_backbone=NullVisionBackbone())
    agent.run(max_steps=2, realtime=False, log_every=1000)
    assert env.recovered == 1
    # First tick was a sensor fault and must not enter replay/reward totals.
    assert len(agent.replay) == 1
    assert abs(agent.total_reward - 0.25) < 1e-6


class AlwaysFrozenFactory:
    def __init__(self): self.instances = []
    def __call__(self, **kwargs):
        c = RecoverableController(frozen=True, **kwargs); self.instances.append(c); return c


def test_unrecoverable_sensor_stream_trips_clean_fatal_result():
    factory = AlwaysFrozenFactory()
    env = AI2ThorEmbodiedEnv(AI2ThorConfig(
        scene="Fake", curriculum_level=2, random_start=False, rehearsal_fraction=0.0,
        sensor_soft_reset_attempts=1, sensor_controller_restart_attempts=1,
        sensor_recovery_verify_rotations=1, sensor_max_controller_restarts_per_window=2,
    ), controller_factory=factory)
    try:
        result = env.recover_sensor_fault()
        assert result["recovered"] is False
        assert result["fatal"] is True
        assert env._sensor_health_payload()["recovery_failures_total"] == 1
    finally:
        env.close()


class FatalInfraFaultEnv(InfraFaultEnv):
    def recover_sensor_fault(self):
        self.recovered += 1
        return {"attempted": True, "recovered": False, "method": "failed", "fatal": True}


def test_agent_stops_cleanly_when_sensor_recovery_fails(tmp_path: Path):
    env = FatalInfraFaultEnv()
    cfg = BrainConfig(seed=4, batch_size=32, learn_every=1000, consolidate_every=1000, checkpoint_every=1000)
    agent = ContinuousAgent(env, cfg, tmp_path / "fatal_state", vision_backbone=NullVisionBackbone())
    agent.run(max_steps=10, realtime=False, log_every=1000)
    assert env.recovered == 1
    assert agent.total_steps == 1
    assert len(agent.replay) == 0
    assert agent.total_reward == 0.0
    assert (tmp_path / "fatal_state" / "brain.pt").exists()


class RaisingController(RecoverableController):
    def step(self, action=None, **kwargs):
        if action in {"RotateRight", "RotateLeft", "MoveAhead", "Pass"}:
            raise ConnectionError("simulated Unity socket failure")
        return super().step(action=action, **kwargs)


class RaisingThenHealthyFactory:
    def __init__(self): self.instances=[]
    def __call__(self, **kwargs):
        if not self.instances:
            c=RaisingController(frozen=False, **kwargs)
        else:
            c=RecoverableController(frozen=False, **kwargs)
        self.instances.append(c); return c


def test_controller_step_exception_enters_recovery_path_instead_of_crashing():
    factory=RaisingThenHealthyFactory()
    env=AI2ThorEmbodiedEnv(AI2ThorConfig(
        scene="Fake", curriculum_level=0, random_start=False, rehearsal_fraction=0.0,
        orient_success_degrees=1.0, sensor_soft_reset_attempts=0,
        sensor_controller_restart_attempts=1, sensor_recovery_verify_rotations=1,
    ), controller_factory=factory)
    try:
        env.reset()
        pos=env._position_from_metadata(env._metadata())
        env.target.position=pos+np.array([1.0,0.0],dtype=np.float32)
        env._best_progress_metric=abs(env._goal_local()[3])
        _obs,_reward,done,info=env.step(3)
        assert done is True
        assert info["infrastructure_fault"] is True
        assert "ConnectionError" in (info["sensor_health"]["controller_error"] or "")
        result=env.recover_sensor_fault()
        assert result["recovered"] is True
        assert result["method"] == "controller-restart"
    finally:
        env.close()
