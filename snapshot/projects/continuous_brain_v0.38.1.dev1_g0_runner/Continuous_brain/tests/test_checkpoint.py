"""DAV-36: save/restore continuation produces matching trajectory over a deterministic test window."""

import stat

import numpy as np
import pytest

from continuous_brain_snn.checkpoint import CheckpointMismatchError, load_checkpoint, save_checkpoint
from continuous_brain_snn.config import SCHEMA_VERSION
from continuous_brain_snn.engine import SNNEngine
from continuous_brain_snn.parity import SpikeRecorder
from continuous_brain_snn.snapshot import SnapshotIntegrityError, SnapshotStore


def _continue(engine, steps=300):
    rec = SpikeRecorder()
    engine.run(steps, recorder=rec)
    return engine.state_hash(), rec.events()


def test_restore_reproduces_trajectory(tmp_path, cfg):
    store = SnapshotStore(tmp_path)
    e = SNNEngine(cfg)
    e.run(250)
    ckpt = save_checkpoint(e, store, linear_issue="DAV-36")
    h1, (s1, n1) = _continue(e)
    r = load_checkpoint(ckpt, store)
    h2, (s2, n2) = _continue(r)
    assert h1 == h2 and np.array_equal(s1, s2) and np.array_equal(n1, n2)


def test_manifest_records_everything(tmp_path, cfg):
    store = SnapshotStore(tmp_path)
    e = SNNEngine(cfg)
    e.run(10)
    m = store.load_state(save_checkpoint(e, store, linear_issue="DAV-36"), "manifest")
    assert m["schema_version"] == SCHEMA_VERSION and m["stage"] == "E0"
    assert m["anatomy_hash"] == e.anatomy.anatomy_hash() and m["config_hash"] == cfg.config_hash()
    assert m["step"] == 10 and m["linear_issue"] == "DAV-36" and "git_rev" in m["code"]
    assert m["connectivity_hash"] == e.slots.connectivity_hash()


def test_checkpoints_never_overwrite(tmp_path, cfg):
    store = SnapshotStore(tmp_path)
    e = SNNEngine(cfg)
    a = save_checkpoint(e, store)
    b = save_checkpoint(e, store)
    assert a != b and store.verify(a) and store.verify(b)


def test_tampered_checkpoint_is_refused(tmp_path, cfg):
    store = SnapshotStore(tmp_path)
    ckpt = save_checkpoint(SNNEngine(cfg), store)
    f = ckpt / "state.pt"
    f.chmod(f.stat().st_mode | stat.S_IWUSR)
    with open(f, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SnapshotIntegrityError):
        load_checkpoint(ckpt, store)


def test_anatomy_mismatch_is_refused(tmp_path, cfg, monkeypatch):
    store = SnapshotStore(tmp_path)
    ckpt = save_checkpoint(SNNEngine(cfg), store)
    import continuous_brain_snn.checkpoint as ck
    from continuous_brain_snn.anatomy import build_anatomy
    from continuous_brain_snn.config import small_config
    monkeypatch.setattr(ck, "build_anatomy", lambda c: build_anatomy(small_config(seed=99)))
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(ckpt, store)
