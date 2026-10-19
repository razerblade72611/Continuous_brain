import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_brain.cognition import CognitiveConfig, tensor_packet
from continuous_brain.development import DevelopmentRoom
from continuous_brain.individual import IndividualRuntime
from continuous_brain.lifetime import LifetimeStore
from continuous_brain.motor import (DevelopmentMotor, MOTOR_CONTRACT, motor_digest,
                                     validated_motor, check_qualification)
from continuous_brain.motor_lab import (MotorTrainer, MotorLabLock, atomic_save,
                                        evaluate_motor, load_qualified)
from continuous_brain.motor_upgrade import attach_development_motor
from continuous_brain.vision import PixelStatsBackbone
from test_v061 import equal, legacy


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def runtime(path, profile="tiny", **kwargs):
    return IndividualRuntime(DevelopmentRoom(seed=7), path, CognitiveConfig.profile(profile),
                             PixelStatsBackbone(), **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def qualified(tmp_path_factory):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("motor_qualified")
    trainer = MotorTrainer(seed=7, cases=256)
    initial = copy.deepcopy(trainer.payload())
    for _ in range(2000):
        trainer.update()
    saved = trainer.payload()
    atomic_save(saved, path / "motor_training.pt")
    report = evaluate_motor(trainer.model, trainer.seed, 100)
    candidate = {"contract": MOTOR_CONTRACT, "model": copy.deepcopy(saved["model"]),
                 "digest": saved["digest"], "updates": trainer.updates, "qualification": report}
    check_qualification(candidate)
    atomic_save(candidate, path / "qualified_motor.pt")
    yield path, candidate, initial
    torch.set_num_threads(old_threads)


def test_motor_learning_changes_weights_and_passes_without_teacher(qualified, monkeypatch):
    _, candidate, initial = qualified
    assert candidate["digest"] != initial["digest"]
    model = validated_motor(candidate)
    data = initial["data"]
    with torch.no_grad():
        trained_loss = torch.nn.functional.cross_entropy(model(data["obs"], data["orient"]), data["actions"])
        fresh = validated_motor(initial)
        initial_loss = torch.nn.functional.cross_entropy(fresh(data["obs"], data["orient"]), data["actions"])
    assert trained_loss < initial_loss * 0.2
    # Qualification's executed path must not call the teaching function at all.
    def forbidden(*args, **kw):
        raise AssertionError("Teacher used during motor evaluation")
    monkeypatch.setattr("continuous_brain.motor_lab.reference_motor", forbidden)
    report = evaluate_motor(model, 7, 100)
    assert report["passed"] and report["teacher_actions"] == 0
    assert report["skills"]["navigate"]["successful_advances"] > 100
    assert report["evaluation_seed"] != report["training_seed"]


def test_untrained_motor_fails_qualification(qualified):
    _, _, initial = qualified
    report = evaluate_motor(validated_motor(initial), 7, 10)
    assert not report["passed"]
    assert report["skills"]["navigate"]["successes"] < 10


def test_training_resume_is_exact_and_preserves_optimizer(qualified):
    _, _, initial = qualified
    first = MotorTrainer(seed=7, cases=256, saved=copy.deepcopy(initial))
    for _ in range(6):
        first.update()
    split = copy.deepcopy(first.payload())
    for _ in range(6):
        first.update()
    second = MotorTrainer(seed=7, cases=256, saved=split)
    for _ in range(6):
        second.update()
    equal(first.payload(), second.payload())


@pytest.mark.parametrize("field", ["digest", "contract", "nonfinite", "teacher", "score", "sample", "seed"])
def test_qualification_rejects_bad_or_mismatched_evidence(qualified, field):
    _, bundle, _ = qualified
    bundle = copy.deepcopy(bundle)
    if field == "digest":
        bundle["digest"] = "wrong"
    elif field == "contract":
        bundle["contract"]["turn_degrees"] = 30
    elif field == "nonfinite":
        bundle["model"]["policy.0.weight"][0, 0] = float("nan")
    elif field == "teacher":
        bundle["qualification"]["teacher_actions"] = 1
    elif field == "score":
        bundle["qualification"]["skills"]["navigate"]["successes"] = 0
    elif field == "sample":
        bundle["qualification"]["skills"]["orient"]["episodes"] = 10
    elif field == "seed":
        bundle["qualification"]["evaluation_seed"] = bundle["qualification"]["training_seed"]
    with pytest.raises(ValueError):
        check_qualification(bundle)


def test_stale_motor_qualification_rejected(qualified, tmp_path):
    import shutil
    source, _, _ = qualified
    shutil.copytree(source, tmp_path / "lab")
    saved = torch.load(tmp_path / "lab/motor_training.pt", weights_only=False)
    trainer = MotorTrainer(7, 256, saved)
    trainer.update()
    atomic_save(trainer.payload(), tmp_path / "lab/motor_training.pt")
    with MotorLabLock(tmp_path / "lab"):
        with pytest.raises(ValueError, match="stale"):
            load_qualified(tmp_path / "lab")


def test_lab_rejects_individual_and_concurrent_writer(tmp_path):
    r = runtime(tmp_path / "individual")
    r.close()
    with pytest.raises(ValueError, match="separate"):
        MotorLabLock(tmp_path / "individual", create=True)
    with MotorLabLock(tmp_path / "lab", create=True):
        with pytest.raises(RuntimeError, match="already in use"):
            MotorLabLock(tmp_path / "lab", create=True)


@pytest.mark.parametrize("profile", ["tiny", "standard"])
def test_attachment_preserves_executive_and_archives_old_motor(qualified, tmp_path, profile):
    lab, candidate, _ = qualified
    legacy(tmp_path / "legacy")
    r = runtime(tmp_path / "source", profile, import_v053=tmp_path / "legacy")
    r.run(8, realtime=False, log_every=10000)
    old = copy.deepcopy(r.store.load())
    old["version"] = "0.6.1"
    old.pop("servo_motor", None)
    old.pop("controller_updates", None)
    r.store.save(old)
    rows = r.store.db.execute("SELECT * FROM experiences").fetchall()
    r.close()
    before = digest(tmp_path / "source/individual.sqlite3")
    report = attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    assert report["individual_id"] == old["self_model"]["individual_id"]
    assert report["decisions"] == 8 and report["updates"] == 1
    assert before == digest(tmp_path / "source/individual.sqlite3")
    target = LifetimeStore(tmp_path / "new")
    state = target.load()
    for key in ("model", "learner", "workspace", "intention", "environment", "rng", "torch_rng", "cuda_rng",
                "observation", "total_reward", "search_controller", "config", "vision"):
        equal(state[key], old[key])
    for key in old["self_model"]:
        if key != "lineage":
            equal(state["self_model"][key], old["self_model"][key])
    assert target.db.execute("SELECT * FROM experiences ORDER BY id LIMIT ?", (len(rows),)).fetchall() == rows
    target.close()
    archived = torch.load(tmp_path / "new/motor_history/replacement_000.pt", weights_only=False)
    equal(archived["motor"], old["motor"])
    equal(archived["motor_hidden"], old["motor_hidden"])
    assert digest(tmp_path / "new/legacy_v053/brain.pt") == digest(tmp_path / "source/legacy_v053/brain.pt")
    resumed = runtime(tmp_path / "new", profile)
    try:
        assert resumed.motor_source() == "learned_development_motor"
        assert resumed.servo_motor.parameter_count == 5060
        assert resumed.learner.active_replay_count() == 0
        assert resumed.controller_updates == 0
        resumed.run(8, realtime=False, log_every=10000)
        assert resumed.self_model.decisions == 16
        assert resumed.learner.updates == 2
        assert resumed.controller_updates == 1
        assert motor_digest(resumed.servo_motor.state_dict()) == candidate["digest"]
    finally:
        resumed.close()


def test_execution_is_neural_and_evaluation_is_read_only(qualified, tmp_path, monkeypatch):
    lab, candidate, _ = qualified
    r = runtime(tmp_path / "source")
    r.run(8, realtime=False, log_every=10000)
    r.close()
    attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    before = digest(tmp_path / "new/individual.sqlite3")
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference motor called in learned execution")
    monkeypatch.setattr("continuous_brain.individual.reference_motor", forbidden)
    e = runtime(tmp_path / "new", evaluate=True, planner_diagnostic=True)
    try:
        e.run(30, realtime=False, log_every=10000)
        assert e.self_model.successes > e.evaluation_start["successes"]
        assert e.last_selection == "diagnostic_planner"
        assert e.teacher_probability() == 1
        assert e.controller_updates == 0
        assert e.status()["primitive_actions_by_skill"].get("navigate/advance", 0) > 0
        assert motor_digest(e.servo_motor.state_dict()) == candidate["digest"]
    finally:
        e.close()
    assert before == digest(tmp_path / "new/individual.sqlite3")
    e = runtime(tmp_path / "new", evaluate=True)
    try:
        e.run(4, realtime=False, log_every=10000)
        assert e.teacher_probability() == 0 and e.last_selection == "learned_option_value"
    finally:
        e.close()
    assert before == digest(tmp_path / "new/individual.sqlite3")


def test_new_controller_world_reranking_requires_new_updates(qualified, tmp_path):
    lab, _, _ = qualified
    r = runtime(tmp_path / "source")
    r.learner.updates = 899
    r.save()
    r.close()
    attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    r = runtime(tmp_path / "new", evaluate=True)
    try:
        with torch.no_grad():
            _, latent, q = r.net(*tensor_packet(r.packet(r.obs), "cpu"), r.workspace)
        r._select(q, [True]*6, 0, latent)
        assert r.last_imagination is None
        r.controller_updates = 200
        r._select(q, [True]*6, 0, latent)
        assert r.last_imagination is not None
    finally:
        r.close()


def test_backup_includes_motor_history(qualified, tmp_path):
    from scripts.snapshot_individual import snapshot
    lab, _, _ = qualified
    r = runtime(tmp_path / "source")
    r.close()
    attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    snapshot(tmp_path / "new", tmp_path / "backup")
    assert digest(tmp_path / "new/motor_history/replacement_000.pt") == digest(tmp_path / "backup/motor_history/replacement_000.pt")


def test_attachment_refuses_existing_running_nested_and_unqualified(qualified, tmp_path):
    lab, _, _ = qualified
    r = runtime(tmp_path / "source")
    try:
        with pytest.raises(RuntimeError, match="already running"):
            attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    finally:
        r.close()
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        attach_development_motor(tmp_path / "source", tmp_path / "existing", lab)
    with pytest.raises(ValueError, match="non-nested"):
        attach_development_motor(tmp_path / "source", tmp_path / "source/nested", lab)
    fake = tmp_path / "unqualified"
    fake.mkdir()
    torch.save({}, fake / "motor_training.pt")
    with pytest.raises(ValueError, match="No qualified"):
        attach_development_motor(tmp_path / "source", tmp_path / "new", fake)
    assert not (tmp_path / "new").exists()


def test_new_motor_rejects_wrong_body_and_bad_observations(qualified, tmp_path):
    lab, candidate, _ = qualified
    r = runtime(tmp_path / "source")
    state = r.store.load()
    state["world_signature"]["type"] = "AI2ThorEmbodiedEnv"
    r.store.save(state)
    r.close()
    with pytest.raises(ValueError, match="only for the development"):
        attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    motor = validated_motor(candidate)
    with pytest.raises(ValueError, match="six finite"):
        motor.action([0, 0, float("nan"), 0, 1, 0])
    with pytest.raises(ValueError, match="No valid"):
        motor.action([0.2, 0, 1, 0, 1, 0], allowed=(4,))


def test_planner_diagnostic_requires_evaluation(tmp_path):
    with pytest.raises(ValueError, match="read-only"):
        runtime(tmp_path, planner_diagnostic=True)


def test_interrupted_attachment_preserves_original_and_blocks_partial_copy(qualified, tmp_path, monkeypatch):
    lab, _, _ = qualified
    r = runtime(tmp_path / "source")
    r.close()
    before = digest(tmp_path / "source/individual.sqlite3")
    def fail(*args, **kw):
        raise OSError("injected attachment failure")
    monkeypatch.setattr("continuous_brain.motor_upgrade.torch.save", fail)
    with pytest.raises(OSError, match="injected"):
        attach_development_motor(tmp_path / "source", tmp_path / "new", lab)
    assert digest(tmp_path / "source/individual.sqlite3") == before
    assert (tmp_path / "new/UPGRADE_INCOMPLETE").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        runtime(tmp_path / "new")


def test_neural_motor_split_run_matches_uninterrupted(qualified, tmp_path):
    from scripts.snapshot_individual import snapshot
    lab, _, _ = qualified
    r = runtime(tmp_path / "source")
    r.run(8, realtime=False, log_every=10000)
    r.close()
    attach_development_motor(tmp_path / "source", tmp_path / "full", lab)
    snapshot(tmp_path / "full", tmp_path / "split")
    full = runtime(tmp_path / "full")
    full.run(16, realtime=False, log_every=10000)
    expected = copy.deepcopy(full.store.load())
    full.close()
    split = runtime(tmp_path / "split")
    split.run(8, realtime=False, log_every=10000)
    split.close()
    split = runtime(tmp_path / "split")
    split.run(8, realtime=False, log_every=10000)
    actual = split.store.load()
    split.close()
    for key in ("model", "learner", "workspace", "environment", "self_model", "intention",
                "servo_motor", "controller_updates", "total_reward", "torch_rng", "rng"):
        equal(actual[key], expected[key])


def test_motor_cli_sigterm_saves_training_state(tmp_path):
    import subprocess
    import sys
    import time
    process = subprocess.Popen([sys.executable, "-m", "continuous_brain.motor_cli", "train",
        "--state-dir", str(tmp_path / "lab"), "--cases", "16", "--updates", "100000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        path = tmp_path / "lab/motor_training.pt"
        while not path.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            time.sleep(0.05)
        assert path.is_file()
        process.terminate()
        assert process.wait(timeout=20) == 0
        with MotorLabLock(tmp_path / "lab"):
            saved = torch.load(path, weights_only=False)
            assert saved["updates"] >= 250
            assert saved["optimizer"]["state"]
            trainer = MotorTrainer(7, 16, saved)
            trainer.update()
            assert trainer.updates == saved["updates"] + 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
