from __future__ import annotations

import copy
import json
import random
import sqlite3
import urllib.request

import numpy as np
import pytest
import torch

from continuous_brain.lifelong import BUILD
from continuous_brain.lifelong.world import ResourceWorld, teacher_action, feedback_integrity_audit
from continuous_brain.lifelong.model import DevelopmentBrain, BrainConfig, model_payload, fingerprint, load_model
from continuous_brain.lifelong.memory import ExperienceMemory, BehaviorHistory
from continuous_brain.lifelong.learning import demonstrations, train_block, evaluate, promotion_gate
from continuous_brain.lifelong.runtime import DevelopmentRuntime, RuntimeConfig, save_model_file
from continuous_brain.lifelong.storage import LifetimeStore, read_summary, read_events
from continuous_brain.lifelong.legacy import gate_copy

torch.set_num_threads(2)


@pytest.fixture
def starter(tmp_path):
    torch.manual_seed(991)
    model = DevelopmentBrain(BrainConfig(width=16, hidden=24))
    path = tmp_path / "starter.pt"
    save_model_file(path, {"build": BUILD, "kind": "development_starter", **model_payload(model)})
    return path


def test_world_rest_depletes_and_replenishment_satiates():
    env = ResourceWorld(9, horizon=256)
    starting = env.body.copy()
    for _ in range(256):
        _, reward, done, info = env.step(0)
        assert reward < 0
        if done:
            break
    assert info["depleted"] and env.body[0] < starting[0]
    env = ResourceWorld(9)
    env.body[0] = .99
    env.food = (env.agent[0] + 1, env.agent[1])
    _, _, _, info = env.step(4)
    assert env.body[0] == 1.
    assert info["positive"] < .05


def test_independent_body_and_forged_meter():
    assert feedback_integrity_audit()["passed"]
    env = ResourceWorld(2)
    observed = env.observe()
    observed["sensors"][:3] = 0
    observed["grid"][:] = 0
    assert min(env.body) > 0 and env.walls[0, 0] == 1


def test_world_resume_including_resource_rng():
    env = ResourceWorld(42, kind="detour")
    for _ in range(9):
        env.step(teacher_action(env))
    restored = ResourceWorld.restore(env.state_dict())
    for _ in range(40):
        action = teacher_action(env)
        a, ar, ad, ai = env.step(action)
        b, br, bd, bi = restored.step(action)
        assert np.array_equal(a["grid"], b["grid"])
        assert np.array_equal(a["sensors"], b["sensors"])
        assert (ar, ad, ai) == (br, bd, bi)


def test_retrieval_preserves_history_and_uses_sensor_match():
    mem = ExperienceMemory()
    q = np.zeros(18, np.float32)
    info = {"resource": "food", "collision": False}
    mem.add(q, 4, 1., info)
    distant = np.ones(18, np.float32) * 10
    mem.add(distant, 1, -1., {"resource": None, "collision": True})
    result = mem.retrieve(q)
    assert result[4] > .9 and result[5] > .8
    assert ExperienceMemory(mem.state_dict()).retrieve(q).tolist() == result.tolist()


def test_candidate_updates_leave_reference_untouched():
    torch.manual_seed(42)
    model = DevelopmentBrain(BrainConfig(width=16, hidden=24))
    reference = copy.deepcopy(model)
    before = fingerprint(reference)
    rows = demonstrations(episodes=6, horizon=16)
    train_block(model, rows, [], updates=2, batch_size=2, sequence=4,
                rng=random.Random(1), reference=reference)
    assert fingerprint(reference) == before
    assert fingerprint(model) != before


def _report(successes, returns=2., error=.1, collisions=0.):
    summary = {k: {"successes": successes, "episodes": 100, "mean_return": returns,
                   "mean_collisions": collisions, "body_prediction_mse": error}
               for k in ("open", "detour")}
    rows = [{"seed": i, "kind": k, "success": i < successes}
            for k in summary for i in range(100)]
    return {"summary": summary, "episodes": rows, "seeds": {"open": list(range(100)), "detour": list(range(100))}, "horizon": 64}


def test_twenty_percent_floor_does_not_hide_38_to_24_regression():
    gate = promotion_gate(_report(38), _report(24, error=.01), _report(38))
    assert not gate["accepted"]
    assert len(gate["paired"]["open"]["lost"]) == 14


def test_retention_requires_acquisition_and_preserves_founder():
    before = _report(38)
    assert not promotion_gate(before, before, before)["accepted"]
    assert promotion_gate(before, _report(38, error=.08), before)["accepted"]
    # Chaining multiple small losses must not erase the fixed founder benchmark.
    assert not promotion_gate(_report(35), _report(34, error=.01), before)["accepted"]


def test_evaluation_is_json_safe_and_read_only():
    torch.manual_seed(123)
    model = DevelopmentBrain(BrainConfig(width=16, hidden=24))
    before = fingerprint(model)
    state = torch.get_rng_state().clone()
    report = evaluate(model, horizon=16, seed_sets={"open": [700001], "detour": [710001]})
    json.dumps(report, allow_nan=False)
    assert before == fingerprint(model)
    assert torch.equal(state, torch.get_rng_state())
    assert model.training


def test_graceful_resume_matches_uninterrupted_training(starter, tmp_path):
    cfg = RuntimeConfig(learning_every=16, updates_per_block=1, batch_size=2, sequence=4,
                        episode_horizon=32, evaluation_horizon=16, save_every=8)
    quiet = lambda _: None
    full = DevelopmentRuntime(tmp_path / "full", torch.device("cpu"), cfg=cfg, starter=starter, seed=123, notify=quiet)
    full.run(35)
    expected = fingerprint(full.model)
    expected_forecaster = fingerprint(full.forecaster)
    expected_hidden = full.hidden.clone()
    expected_trace = read_events(tmp_path / "full", 1000)
    full.close()
    split = DevelopmentRuntime(tmp_path / "split", torch.device("cpu"), cfg=cfg, starter=starter, seed=123, notify=quiet)
    split.run(19)
    identity = split.identity.copy()
    split.close()
    resumed = DevelopmentRuntime(tmp_path / "split", torch.device("cpu"), starter=starter, notify=quiet)
    resumed.run(16)
    assert resumed.identity == identity
    assert fingerprint(resumed.model) == expected
    assert fingerprint(resumed.forecaster) == expected_forecaster
    assert torch.equal(resumed.hidden, expected_hidden)
    actual = read_events(tmp_path / "split", 1000)
    lived = lambda rows: [r for r in rows if r["provenance"] == "lived_simulation"]
    assert lived(actual) == lived(expected_trace)
    assert resumed.blocks == 2
    assert all(not f["used_to_authorize_promotion"] for f in resumed.forecasts)
    resumed.close()


def test_save_acknowledgement_requires_reopened_committed_state(tmp_path):
    directory = tmp_path / "finalized"
    store = LifetimeStore(directory)
    summary = {"steps": 256, "identity": {"id": "persistent-test"}, "learning_blocks": 1}
    store.commit({"steps": 256}, summary, [{"step": 256, "kind": "experience",
                 "provenance": "lived_simulation", "content": {"action": 1}}])
    store.finish_save(summary)
    assert store.connection is None
    assert read_summary(directory) == summary
    assert not (directory / "lifetime.sqlite3-wal").exists() or (directory / "lifetime.sqlite3-wal").stat().st_size == 0
    with pytest.raises(RuntimeError, match="running writer"):
        LifetimeStore(directory)
    store.close()
    with LifetimeStore(directory) as reopened:
        assert reopened.load()["steps"] == 256
    assert read_events(directory)[0]["step"] == 256


def test_save_acknowledgement_rejects_stale_checkpoint(tmp_path):
    with LifetimeStore(tmp_path / "stale") as store:
        store.commit({"steps": 224}, {"steps": 224}, [])
        with pytest.raises(RuntimeError, match="committed state differs"):
            store.finish_save({"steps": 256})
        assert store.load()["steps"] == 224


def test_process_lock_and_atomic_history(tmp_path):
    with LifetimeStore(tmp_path / "state") as first:
        with pytest.raises(RuntimeError, match="writer"):
            LifetimeStore(tmp_path / "state")
        first.commit({"revision": 1}, {"revision": 1}, [{"step": 1, "kind": "test", "provenance": "test", "content": {}}])
        with pytest.raises(ValueError):
            first.commit({"revision": 2}, {"invalid": float("nan")}, [])
        assert first.load() == {"revision": 1}
        assert read_summary(tmp_path / "state") == {"revision": 1}
        assert len(read_events(tmp_path / "state")) == 1
        first.snapshot(tmp_path / "backup")
    assert read_summary(tmp_path / "backup") == {"revision": 1}


def test_legacy_state_directory_cannot_be_reused(tmp_path):
    (tmp_path / "infant_training.pt").write_bytes(b"legacy sentinel")
    with pytest.raises(ValueError, match="NEW"):
        LifetimeStore(tmp_path)
    assert (tmp_path / "infant_training.pt").read_bytes() == b"legacy sentinel"


def test_gate_crossover_changes_only_two_buffers(tmp_path):
    source, destination = tmp_path / "source.pt", tmp_path / "copy.pt"
    payload = {"config": {"protected_growth_enabled": True}, "model": {
        "planning.growth.gate": torch.tensor(.02), "options.growth.gate": torch.tensor(.02),
        "old.weight": torch.arange(12).reshape(3, 4)}, "metadata": {"updates": 3000}}
    torch.save(payload, source)
    original = source.read_bytes()
    receipt = gate_copy(source, destination, .05)
    assert receipt["other_tensors_changed"] == 0 and source.read_bytes() == original
    changed = torch.load(destination, weights_only=False)
    assert torch.equal(changed["model"]["old.weight"], payload["model"]["old.weight"])
    assert float(changed["model"]["planning.growth.gate"]) == pytest.approx(.05)
    with pytest.raises(FileExistsError):
        gate_copy(source, source, .05)


def test_legacy_paired_trace_uses_shared_start_and_leaves_checkpoint(tmp_path, monkeypatch):
    from continuous_brain.infant import InfantBrain, InfantConfig
    from continuous_brain.environment import DetourWorld
    from continuous_brain.lifelong.legacy import paired_action_trace
    cfg = InfantConfig(image_size=32, bus_dim=16, nucleus_dim=24, language_dim=32,
                      language_heads=4, language_layers=1, language_ff=48,
                      temporal_dim=16, world_hidden=24, world_ensemble=2, planning_dim=16,
                      option_dim=8)
    model = InfantBrain(cfg)
    path = tmp_path / "child.pt"
    torch.save(model.checkpoint_payload(metadata={"option_cortex_version": "latent-temporal-options-v1"}), path)
    original = path.read_bytes()
    def small_world(*, seed, max_episode_steps):
        return DetourWorld(seed=seed, max_episode_steps=4)
    monkeypatch.setattr("continuous_brain.environment.DetourWorld", small_world)
    result = paired_action_trace(path, 12001, torch.device("cpu"))
    json.dumps(result, allow_nan=False)
    a, b = result["trace"]["direct_options"], result["trace"]["hierarchical"]
    assert a[0]["observation_before"] == b[0]["observation_before"]
    assert len(a) <= 4 and len(b) <= 4
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="reserved"):
        paired_action_trace(path, 51001, torch.device("cpu"))


def test_corrupt_model_fingerprint_is_rejected():
    payload = model_payload(DevelopmentBrain(BrainConfig(width=16, hidden=24)))
    payload["model"]["policy.2.bias"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(payload, "cpu")


def test_read_only_dashboard_serves_state():
    from continuous_brain.lifelong.dashboard import Dashboard
    dash = Dashboard({"status": "<script>untrusted name</script>"}, port=0)
    try:
        with urllib.request.urlopen(dash.url + "api/state") as r:
            assert json.load(r)["status"].startswith("<script>")
        with urllib.request.urlopen(dash.url) as r:
            html = r.read().decode()
            assert "textContent" in html and "innerHTML" not in html
        request = urllib.request.Request(dash.url + "api/state", data=b'{}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 501
    finally:
        dash.close()
