from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_core
from agent_core.thor_adapter import (
    SimulatorStepTimeout,
    _configure_controller_fifo_timeout,
    _install_ai2thor_fifo_timeout_patch,
)
from agent_core.training import _cardinal_neighbor_degree, _maneuverable_starts


def test_release_version_is_v01010():
    assert agent_core.__version__ == "0.10.15"


def test_maneuverable_starts_reject_dead_end_and_keep_open_cell():
    g = 0.25
    reachable = [
        {"x":0.0,"z":0.0}, {"x":0.25,"z":0.0}, {"x":0.5,"z":0.0},
        {"x":0.25,"z":0.25}, {"x":0.25,"z":-0.25},
    ]
    assert _cardinal_neighbor_degree(reachable[0], reachable, g) == 1
    assert _cardinal_neighbor_degree(reachable[1], reachable, g) == 4
    safe = _maneuverable_starts([reachable[0], reachable[1]], reachable, g)
    assert reachable[0] not in safe
    assert reachable[1] in safe


def test_fifo_select_patch_times_out_without_sigalrm(monkeypatch, tmp_path: Path):
    if os.name != "posix":
        pytest.skip("FIFO select test is POSIX-only")
    import struct
    fake_parent = types.ModuleType("ai2thor")
    fake_fifo = types.ModuleType("ai2thor.fifo_server")
    class FieldType:
        class EOM: value = 255
        END_OF_MESSAGE = EOM()
        METADATA = object()
        METADATA_PATCH = object()
    class FakeFifo:
        header_format = "!BI"
        header_size = struct.calcsize(header_format)
        _author_select_timeout_patch = False
        def _recv_message(self):
            raise AssertionError("unpatched recv should not run")
    fake_fifo.FifoServer = FakeFifo
    fake_fifo.FieldType = FieldType
    fake_parent.fifo_server = fake_fifo
    monkeypatch.setitem(sys.modules, "ai2thor", fake_parent)
    monkeypatch.setitem(sys.modules, "ai2thor.fifo_server", fake_fifo)
    _install_ai2thor_fifo_timeout_patch(0.05)
    server = FakeFifo()
    server.server_pipe = None
    server.server_pipe_path = str(tmp_path / "server.pipe")
    server.field_types = {}
    server.image_fields = set()
    server.form_field_map = {}
    os.mkfifo(server.server_pipe_path)
    import threading
    stop = threading.Event()
    def writer():
        fd = os.open(server.server_pipe_path, os.O_WRONLY)
        stop.wait(1.0)
        os.close(fd)
    t = threading.Thread(target=writer, daemon=True); t.start()
    server._author_receive_timeout_seconds = 0.05
    server._author_operation = "TeleportFull"
    started=time.monotonic()
    try:
        with pytest.raises(SimulatorStepTimeout) as exc:
            server._recv_message()
    finally:
        stop.set(); t.join(timeout=1.0)
        if server.server_pipe is not None:
            server.server_pipe.close()
    assert exc.value.operation == "TeleportFull"
    assert time.monotonic()-started < 0.5


def test_configure_fifo_timeout_sets_operation():
    server=SimpleNamespace()
    controller=SimpleNamespace(server=server)
    _configure_controller_fifo_timeout(controller, 3.5, "TeleportFull")
    assert server._author_receive_timeout_seconds == 3.5
    assert server._author_operation == "TeleportFull"
