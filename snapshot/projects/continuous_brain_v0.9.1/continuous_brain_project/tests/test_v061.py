import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from continuous_brain.cognition import CognitiveConfig, CognitiveCore, SequenceLearner
from continuous_brain.development import DevelopmentRoom
from continuous_brain.individual import IndividualRuntime
from continuous_brain.legacy_motor import read_legacy_motor
from continuous_brain.lifetime import LifetimeStore
from continuous_brain.model import ContinuousBrainNet
from continuous_brain.skills import SearchController
from continuous_brain.upgrade import upgrade_individual
from continuous_brain.vision import PixelStatsBackbone


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def runtime(path, profile="tiny", **kw):
    return IndividualRuntime(DevelopmentRoom(seed=7), path, CognitiveConfig.profile(profile),
                             PixelStatsBackbone(), **kw)


def legacy(path):
    path.mkdir()
    # Exact schema/dimensions reported by David, synthetic weights only.
    net = ContinuousBrainNet(6, 5, 64, 128, 48)
    data = {"version": "0.4.3", "format_version": 6, "steps": 23941, "goals": 1803,
            "mastered_levels": [0, 1], "model": net.state_dict(),
            "vision": {"name": "pixel-stats", "output_dim": 48, "pretrained": False},
            "environment_state": {"format_version": 3, "edge_failures": {},
                "legacy_v041_stats": {}, "level_action_steps": {"0": 100, "1": 200},
                "level_totals": {"0": {}, "1": {}}, "recent_outcomes_by_level": {"0": [], "1": []}}}
    torch.save(data, path / "brain.pt")
    (path / "history.txt").write_text("preserve earlier history")
    return data


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_reported_v043_schema_imports_frozen_weights(tmp_path):
    data = legacy(tmp_path / "legacy")
    before = digest(tmp_path / "legacy/brain.pt")
    r = runtime(tmp_path / "new", import_v053=tmp_path / "legacy")
    try:
        equal(r.motor.state_dict(), data["model"])
        assert not any(p.requires_grad for p in r.motor.parameters())
        assert r.self_model.lineage["legacy_mastered_levels"] == [0, 1]
        assert r.self_model.lineage["legacy_steps"] == 23941
        assert r.self_model.lineage["validation"] == "v043_format6_curriculum"
        r.run(8, realtime=False, log_every=100)
        equal(r.motor.state_dict(), data["model"])
    finally:
        r.close()
    assert digest(tmp_path / "legacy/brain.pt") == before


@pytest.mark.parametrize("change", ["version", "format", "missing_schema", "nonfinite", "vision", "shape"])
def test_invalid_legacy_fails_before_destination_created(tmp_path, change):
    r = runtime(tmp_path / "original")
    r.close()
    data = legacy(tmp_path / "legacy")
    if change == "version":
        data["version"] = "unknown"
    elif change == "format":
        data["format_version"] = 999
    elif change == "missing_schema":
        del data["environment_state"]["level_totals"]
    elif change == "nonfinite":
        data["model"]["visual_gate_logit"] = torch.tensor(float("nan"))
    elif change == "vision":
        data["vision"]["name"] = "different-48d-backbone"
    elif change == "shape":
        data["model"]["q_head.2.weight"] = torch.zeros(4, 128)
    torch.save(data, tmp_path / "legacy/brain.pt")
    before = digest(tmp_path / "original/individual.sqlite3")
    with pytest.raises((ValueError, RuntimeError)):
        upgrade_individual(tmp_path / "original", tmp_path / "new", tmp_path / "legacy")
    assert not (tmp_path / "new").exists()
    assert before == digest(tmp_path / "original/individual.sqlite3")


@pytest.mark.parametrize("attach,profile", [(False, "tiny"), (True, "tiny"), (True, "standard")])
def test_upgrade_preserves_all_learned_state_and_original_bytes(tmp_path, attach, profile):
    r = runtime(tmp_path / "original", profile=profile)
    r.run(12, realtime=False, log_every=100)
    state = r.store.load()
    state["version"] = "0.6.0"
    state.pop("search_controller")
    state.pop("motor_digest")
    for row in state["learner"]["replay"]:
        row.pop("control_revision", None)
    for seq in state["learner"]["protected"]:
        for row in seq:
            row.pop("control_revision", None)
    r.store.save(state)
    original_rows = r.store.db.execute("SELECT * FROM experiences").fetchall()
    r.close()
    legacy_data = legacy(tmp_path / "legacy")
    before = digest(tmp_path / "original/individual.sqlite3")
    old = digest(tmp_path / "legacy/brain.pt")
    report = upgrade_individual(tmp_path / "original", tmp_path / "new",
                                tmp_path / "legacy" if attach else None)
    assert report["decisions"] == 12 and report["updates"] == 2
    target = LifetimeStore(tmp_path / "new")
    upgraded = target.load()
    for key in ("model", "learner", "workspace", "intention", "observation", "environment", "rng",
                "torch_rng", "cuda_rng", "total_reward", "config", "vision", "world_signature"):
        equal(upgraded[key], state[key])
    for key in state["self_model"]:
        if key != "lineage":
            equal(upgraded["self_model"][key], state["self_model"][key])
    assert target.db.execute("SELECT * FROM experiences ORDER BY id LIMIT ?", (len(original_rows),)).fetchall() == original_rows
    if attach:
        equal(upgraded["motor"], legacy_data["model"])
        assert upgraded["self_model"]["lineage"]["motor_attachment"]["legacy_goals"] == 1803
        assert (tmp_path / "new/legacy_v053/history.txt").read_text() == "preserve earlier history"
    target.close()
    assert before == digest(tmp_path / "original/individual.sqlite3")
    assert old == digest(tmp_path / "legacy/brain.pt")
    resumed = runtime(tmp_path / "new", profile=profile)
    try:
        assert resumed.self_model.individual_id == state["self_model"]["individual_id"]
        assert resumed.learner.active_replay_count() == 0
        assert resumed.learner.train() is None
        equal(resumed.net.state_dict(), state["model"])
        resumed.run(8, realtime=False, log_every=100)
        assert resumed.self_model.decisions == 20
        assert resumed.store.count("experience") == 20
        assert resumed.learner.updates == 3
        if attach:
            equal(resumed.motor.state_dict(), legacy_data["model"])
    finally:
        resumed.close()


def test_upgrade_rejects_running_existing_nested_or_duplicate_motor(tmp_path):
    r = runtime(tmp_path / "running")
    try:
        with pytest.raises(RuntimeError, match="already running"):
            upgrade_individual(tmp_path / "running", tmp_path / "new")
    finally:
        r.close()
    with pytest.raises(ValueError, match="non-nested"):
        upgrade_individual(tmp_path / "running", tmp_path / "running/new")
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        upgrade_individual(tmp_path / "running", tmp_path / "existing")


def test_refuse_motor_replacement_and_incomplete_upgrade(tmp_path):
    legacy(tmp_path / "legacy")
    r = runtime(tmp_path / "original", import_v053=tmp_path / "legacy")
    r.close()
    with pytest.raises(ValueError, match="already has"):
        upgrade_individual(tmp_path / "original", tmp_path / "new", tmp_path / "legacy")
    marker_dir = tmp_path / "incomplete"
    marker_dir.mkdir()
    (marker_dir / "UPGRADE_INCOMPLETE").write_text("interrupted")
    with pytest.raises(ValueError, match="Incomplete"):
        runtime(marker_dir)


def test_search_eventually_moves_without_visible_target_and_stays_bounded():
    s = SearchController()
    obs = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
    actions = [s.action(obs) for _ in range(38)]
    assert actions[:24] == [3]*24
    assert 1 in actions[24:]
    assert s.successful_moves == 12
    assert s.scan_remaining < 24


@pytest.mark.parametrize("depth,clearance,ack", [(False, 1, 1), (True, 0, 1), (True, 1, 0)])
def test_search_does_not_advance_on_unknown_or_blocked_depth(depth, clearance, ack):
    s = SearchController(scan_remaining=0)
    obs = np.array([0, 0, clearance, 0.2, ack, 0], dtype=np.float32)
    assert s.action(obs, depth_healthy=depth) in (2, 3)


def test_search_relocates_to_reveal_occluded_target_without_oracle():
    env = DevelopmentRoom(max_episode_steps=1000)
    env.position = np.array([-1.5, 0.0])
    env.target = np.array([-1.5, 2.0])
    env.yaw = 0.0
    start = env.position.copy()
    obs = env._obs()
    assert not np.any(obs[:2])
    controller = SearchController()
    seen = False
    for _ in range(250):
        action = controller.action(obs)
        obs, _, _, _ = env.step(action)
        if np.any(obs[:2]):
            seen = True
            break
    assert not np.array_equal(start, env.position)
    assert seen


def test_search_controller_survives_restart_exactly(tmp_path):
    r = runtime(tmp_path)
    r.search = SearchController(scan_remaining=0, travel_remaining=7, avoid_remaining=3, turn_direction=2)
    r.save()
    state = copy.deepcopy(r.search.state_dict())
    r.close()
    r = runtime(tmp_path)
    try:
        assert r.search.state_dict() == state
    finally:
        r.close()


def test_replay_does_not_bridge_controller_change_or_train_obsolete_rows():
    learner = SequenceLearner(CognitiveCore(CognitiveConfig.profile("tiny")))
    learner.control_revision = "new"
    for serial in range(1, 13):
        learner.append({"serial": serial, "control_revision": "old" if serial <= 8 else "new",
                        "packet": {"goal": 0}})
    assert learner.active_replay_count() == 4
    assert not learner._samples()
    for serial in (13, 14):
        learner.append({"serial": serial, "control_revision": "new", "packet": {"goal": 0}})
    assert all(row["control_revision"] == "new" for seq in learner._samples() for row in seq)


def test_validation_does_not_consume_torch_rng(tmp_path):
    legacy(tmp_path / "legacy")
    before = torch.get_rng_state().clone()
    read_legacy_motor(tmp_path / "legacy", {"name": "pixel-stats", "dim": 48, "model_id": None})
    assert torch.equal(torch.get_rng_state(), before)


def test_interrupted_upgrade_is_non_runnable_and_original_survives(tmp_path, monkeypatch):
    import continuous_brain.upgrade as module
    r = runtime(tmp_path / "original")
    r.close()
    legacy(tmp_path / "legacy")
    before = digest(tmp_path / "original/individual.sqlite3")
    def fail(*args):
        raise OSError("injected disk failure")
    monkeypatch.setattr(module, "archive_legacy_motor", fail)
    with pytest.raises(OSError, match="injected"):
        upgrade_individual(tmp_path / "original", tmp_path / "new", tmp_path / "legacy")
    assert (tmp_path / "new/UPGRADE_INCOMPLETE").is_file()
    with pytest.raises(ValueError, match="Incomplete"):
        runtime(tmp_path / "new")
    assert digest(tmp_path / "original/individual.sqlite3") == before


def test_split_search_run_matches_uninterrupted_movement(tmp_path):
    def create(path):
        r = runtime(path)
        r._select = lambda *args: 4
        return r
    full = create(tmp_path / "full")
    full.env.position = np.array([-1.5, 0.0])
    full.env.target = np.array([-1.5, 2.0])
    full.env.yaw = 0
    full.obs = full.env._obs()
    full.save()
    full.close()
    upgrade_individual(tmp_path / "full", tmp_path / "split")
    full = create(tmp_path / "full")
    full.run(15, realtime=False, log_every=100)
    expected = full.store.load()
    full.close()
    split = create(tmp_path / "split")
    split.run(7, realtime=False, log_every=100)
    split.close()
    split = create(tmp_path / "split")
    split.run(8, realtime=False, log_every=100)
    actual = split.store.load()
    split.close()
    for key in ("model", "learner", "workspace", "environment", "search_controller", "total_reward"):
        equal(actual[key], expected[key])


def test_search_memory_does_not_mislabel_authored_actions_as_neural(tmp_path):
    legacy(tmp_path / "legacy")
    r = runtime(tmp_path / "new", import_v053=tmp_path / "legacy")
    try:
        r._select = lambda *args: 4
        r.run(1, realtime=False, log_every=100)
        metadata = json.loads(r.store.db.execute(
            "SELECT metadata FROM experiences WHERE kind='experience'").fetchone()[0])
        assert metadata["primitive_source"] == "authored_sensor_search"
    finally:
        r.close()


def test_protected_old_replay_does_not_block_new_formative_experiences():
    learner = SequenceLearner(CognitiveCore(CognitiveConfig.profile("tiny")))
    old = [{"serial": i, "control_revision": "old", "packet": {"goal": 0}} for i in range(6)]
    learner.protected = [copy.deepcopy(old) for _ in range(32)]
    learner.control_revision = "new"
    for i in range(6):
        learner.append({"serial": i, "control_revision": "new", "packet": {"goal": 0}})
    assert len(learner.protected) == 33
    assert learner.protected[:32] == [old]*32
    assert learner._eligible(learner.protected[-1])
