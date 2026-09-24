import os
import stat
import subprocess
from pathlib import Path

import pytest
import torch

from continuous_brain_snn.snapshot import SnapshotExistsError, SnapshotIntegrityError, SnapshotStore

REPO = Path(__file__).resolve().parents[1]


def test_every_snapshot_is_new_and_verified(tmp_path):
    store = SnapshotStore(tmp_path)
    a = store.save_state({"w": {"x": torch.ones(3)}}, "cycle")
    b = store.save_state({"w": {"x": torch.zeros(3)}}, "cycle")
    assert a != b
    assert store.verify(a) and store.verify(b)
    assert torch.equal(store.load_state(a, "w")["x"], torch.ones(3))
    assert torch.equal(store.load_state(b, "w")["x"], torch.zeros(3))
    log = (tmp_path / "logs" / "version_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 2


def test_snapshot_files_are_read_only(tmp_path):
    store = SnapshotStore(tmp_path)
    snap = store.save_state({"w": {"x": torch.ones(1)}}, "ro")
    for p in snap.iterdir():
        assert not (p.stat().st_mode & stat.S_IWUSR), p


def test_tampering_is_detected(tmp_path):
    store = SnapshotStore(tmp_path)
    snap = store.save_state({"w": {"x": torch.ones(4)}}, "t")
    f = snap / "w.pt"
    f.chmod(f.stat().st_mode | stat.S_IWUSR)
    with open(f, "ab") as fh:
        fh.write(b"corruption")
    assert not store.verify(snap)
    with pytest.raises(SnapshotIntegrityError):
        store.load_state(snap, "w")


def test_incomplete_snapshot_is_ignored(tmp_path):
    store = SnapshotStore(tmp_path)
    good = store.save_state({"w": {"x": torch.ones(1)}}, "run")
    (store.checkpoints / "29990101T000000000000Z_run").mkdir()   # crashed mid-write: no COMPLETE
    assert store.latest("run") == good
    assert not store.verify(store.checkpoints / "29990101T000000000000Z_run")


def test_restore_never_overwrites(tmp_path):
    store = SnapshotStore(tmp_path)
    snap = store.save_state({"w": {"x": torch.ones(1)}}, "r")
    dest = tmp_path / "restored"
    store.restore_to(snap, dest)
    assert (dest / "w.pt").exists()
    with pytest.raises(SnapshotExistsError):
        store.restore_to(snap, dest)


def test_snapshot_files_copies_sources(tmp_path):
    src = tmp_path / "ckpt.bin"
    src.write_bytes(b"weights-v1")
    store = SnapshotStore(tmp_path / "store")
    snap = store.snapshot_files([src], "pre-port")
    src.write_bytes(b"weights-v2-MUTATED")
    assert (snap / "ckpt.bin").read_bytes() == b"weights-v1"
    assert store.verify(snap)


def test_shell_script_refuses_to_overwrite(tmp_path):
    work = tmp_path / "proj"
    (work / "scripts").mkdir(parents=True)
    (work / "logs").mkdir()
    (work / "checkpoints").mkdir()
    script = work / "scripts" / "snapshot.sh"
    script.write_text((REPO / "scripts" / "snapshot.sh").read_text())
    script.chmod(0o755)
    # Pin `date` so both runs compute the same timestamp: a guaranteed collision.
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "date").write_text("#!/usr/bin/env bash\necho 20260101T000000Z\n")
    (fakebin / "date").chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    data = work / "checkpoint.bin"
    data.write_text("v1")
    first = subprocess.run([str(script), str(data), "same"], capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    data.write_text("v2-MUTATED")
    second = subprocess.run([str(script), str(data), "same"], capture_output=True, text=True, env=env)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
    snaps = list((work / "checkpoints").iterdir())
    assert len(snaps) == 1
    assert (snaps[0] / "checkpoint.bin").read_text() == "v1"
