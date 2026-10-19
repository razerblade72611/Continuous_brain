from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import agent_core
from agent_core.thor_adapter import (
    SharedThorWorld,
    ThorConfig,
    SimulatorTransportError,
)


def test_release_version_is_v01012():
    assert agent_core.__version__ == "0.10.18"


def test_shared_world_wraps_broken_pipe_as_transport_error(tmp_path: Path):
    class BrokenController:
        server = SimpleNamespace()
        last_event = None
        def step(self, **_kwargs):
            raise BrokenPipeError(32, "Broken pipe")

    world = SharedThorWorld(
        ThorConfig(simulator_step_timeout_seconds=1.0),
        agent_count=2,
        heartbeat_path=tmp_path / "heartbeat.json",
    )
    world.controller = BrokenController()
    try:
        try:
            world.step_for(1, "TeleportFull", x=0.0, y=0.9, z=0.0)
        except SimulatorTransportError as exc:
            assert exc.operation == "TeleportFull"
            assert "Broken pipe" in str(exc)
        else:
            raise AssertionError("BrokenPipeError was not converted to SimulatorTransportError")
    finally:
        world.controller = None


def test_shared_world_close_ignores_broken_transport_and_detaches_controller(tmp_path: Path):
    class BrokenController:
        def stop(self):
            raise BrokenPipeError(32, "Broken pipe")

    world = SharedThorWorld(ThorConfig(), agent_count=2, heartbeat_path=tmp_path / "heartbeat.json")
    world.controller = BrokenController()
    world.close()
    assert world.controller is None


def test_worker_transport_failure_is_unscored_restarts_and_continues(monkeypatch, tmp_path: Path):
    import agent_core.training as training

    class FakeAdapter:
        def start(self):
            return None
        def get_reachable_positions(self):
            return [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ]

    class FakeShared:
        last = None
        def __init__(self, _cfg, agent_count=2, heartbeat_path=None):
            self.agent_count = agent_count
            self.heartbeat_path = heartbeat_path
            self.adapters = {0: FakeAdapter(), 1: FakeAdapter()}
            self.restarts = 0
            FakeShared.last = self
        def start(self):
            return None
        def close(self):
            return None
        def agent(self, i):
            return self.adapters[i]
        def set_heartbeat_context(self, **_kwargs):
            return None
        def restart(self):
            self.restarts += 1

    calls = {"n": 0}
    def fake_train_door(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SimulatorTransportError("TeleportFull", BrokenPipeError(32, "Broken pipe"))
        return training.EpisodeResult(
            "door", True, 1, 1, 0, kwargs["agent_label"],
            sequence=["MoveAhead"], object_type="Door", verification=[]
        )

    monkeypatch.setattr(training, "SharedThorWorld", FakeShared)
    monkeypatch.setattr(training, "_train_door", fake_train_door)

    spec = training.WorkerSpec(
        worker_id=0,
        episodes=1,
        scene="TestScene",
        grid_size=0.25,
        rotate_step_degrees=90.0,
        visibility_distance=1.5,
        field_of_view=90,
        width=64,
        height=64,
        headless=False,
        thor_executable=None,
        seed=1,
        max_steps_per_episode=20,
        worker_root=str(tmp_path / "worker_00"),
        replay_max_records=100,
        curriculum={"door": 1.0},
        level=3,
        max_setup_attempts_per_episode=3,
        export_sft=False,
    )
    metrics = training._worker_main(spec)
    assert metrics["episodes"] == 1
    assert metrics["successful_episodes"] == 1
    assert metrics["failed_episodes"] == 0
    assert metrics["skipped_setups"] == 1
    assert metrics["simulator_transport_failures"] == 1
    assert metrics["simulator_restarts"] == 1
    assert FakeShared.last.restarts == 1
