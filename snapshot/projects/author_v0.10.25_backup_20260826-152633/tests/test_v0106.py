from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import agent_core
from agent_core.thor_adapter import (
    SharedThorWorld,
    SimulatorStepTimeout,
    ThorConfig,
    _call_with_hard_timeout,
)
from agent_core.training import _teleport


def _child(agent_id: int):
    return SimpleNamespace(
        metadata={
            "agentId": agent_id,
            "sceneName": "TestScene",
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": [],
            "lastAction": None,
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": [],
        },
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        instance_detections2D={},
        instance_masks={},
    )


class _BlockingController:
    def __init__(self):
        self.last_event = SimpleNamespace(events=[_child(0), _child(1)])
        self.stopped = False

    def step(self, **_kwargs):
        # A blocking pipe read is closer to the real ai2thor==4.3 FIFO failure
        # than a busy Python loop. SIGALRM must interrupt this system call.
        read_fd, write_fd = os.pipe()
        try:
            os.read(read_fd, 8)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def stop(self):
        self.stopped = True


def test_release_version_is_v0106():
    assert agent_core.__version__ == "0.10.25"


def test_hard_timeout_interrupts_blocking_fifo_style_read():
    if os.name != "posix":
        pytest.skip("SIGALRM watchdog is POSIX-only")
    started = time.monotonic()
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(SimulatorStepTimeout) as exc:
            _call_with_hard_timeout(lambda: os.read(read_fd, 1), 0.05, "TeleportFull")
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert exc.value.operation == "TeleportFull"
    assert time.monotonic() - started < 0.5


def test_shared_step_timeout_writes_actionable_heartbeat(tmp_path: Path):
    if os.name != "posix":
        pytest.skip("SIGALRM watchdog is POSIX-only")
    heartbeat = tmp_path / "heartbeat.json"
    world = SharedThorWorld(
        ThorConfig(scene="TestScene", simulator_step_timeout_seconds=0.05),
        agent_count=2,
        heartbeat_path=heartbeat,
    )
    world.controller = _BlockingController()
    world.set_heartbeat_context(worker_id=0, level=3, episode=46, raw_attempt=46, skill="door", agent="B")
    with pytest.raises(SimulatorStepTimeout):
        world.step_for(1, "TeleportFull", position={"x": 1.0, "y": 0.9, "z": 1.0})
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["status"] == "timeout"
    assert payload["operation"] == "TeleportFull"
    assert payload["episode"] == 46
    assert payload["agent"] == "B"


def test_training_teleport_uses_shared_watchdog(tmp_path: Path):
    if os.name != "posix":
        pytest.skip("SIGALRM watchdog is POSIX-only")
    world = SharedThorWorld(
        ThorConfig(scene="TestScene", simulator_step_timeout_seconds=0.05),
        agent_count=2,
        heartbeat_path=tmp_path / "heartbeat.json",
    )
    world.controller = _BlockingController()
    with pytest.raises(SimulatorStepTimeout) as exc:
        _teleport(world, 0, {"x": 0.5, "y": 0.9, "z": 0.5}, 90.0)
    assert exc.value.operation == "TeleportFull"


def test_restart_replaces_poisoned_controller_and_preserves_world_object(monkeypatch, tmp_path: Path):
    created = []

    class FreshController:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.last_event = SimpleNamespace(events=[_child(0), _child(1)])
            self.stopped = False
            created.append(self)
        def stop(self):
            self.stopped = True

    monkeypatch.setitem(sys.modules, "ai2thor.controller", types.SimpleNamespace(Controller=FreshController))
    world = SharedThorWorld(
        ThorConfig(scene="TestScene", simulator_start_timeout_seconds=1.0),
        agent_count=2,
        heartbeat_path=tmp_path / "heartbeat.json",
    )
    poisoned = FreshController()
    world.controller = poisoned
    world.restart()
    assert poisoned.stopped is True
    assert world.restart_count == 1
    assert world.controller is not poisoned
    assert len(world.controller.last_event.events) == 2
    assert world.controller.kwargs["agentCount"] == 2
    world.close()


def test_stage_timeout_propagates_without_scoring_or_replay(tmp_path: Path):
    from agent_core.embodied_skills import EmbodiedSkillLibrary
    from agent_core.executor import Verifier
    from agent_core.models import Action
    from agent_core.training import _execute_stage

    class TimeoutAdapter:
        def observe(self):
            return SimpleNamespace()
        def execute(self, _action, step_budget=None):
            raise SimulatorStepTimeout("TeleportFull", 0.05)

    replay = tmp_path / "replay.jsonl"
    library = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    before_attempts = library.skills["door_approach"].attempts
    with pytest.raises(SimulatorStepTimeout):
        _execute_stage(
            adapter=TimeoutAdapter(),
            action=Action("NavigateTo", {"objectId": "Door|test"}, "test"),
            verifier=Verifier(),
            replay_path=replay,
            scene="TestScene",
            agent_label="A",
            episode=46,
            curriculum_skill="door",
            stage_skill="door_approach",
            library=library,
            object_type="Door",
            max_steps=20,
        )
    assert library.skills["door_approach"].attempts == before_attempts
    assert not replay.exists()


def test_worker_timeout_is_unscored_restarts_and_continues(monkeypatch, tmp_path: Path):
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
            raise SimulatorStepTimeout("TeleportFull", 0.05)
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
    assert metrics["simulator_timeouts"] == 1
    assert metrics["simulator_restarts"] == 1
    assert FakeShared.last.restarts == 1
