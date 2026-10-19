import copy
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_brain.cognition import CognitiveConfig, CognitiveCore, SequenceLearner, tensor_packet
from continuous_brain.development import DevelopmentRoom
from continuous_brain.individual import IndividualRuntime
from continuous_brain.lifetime import LifetimeStore
from continuous_brain.model import ContinuousBrainNet
from continuous_brain.skills import compose, available, sensed_facts, SKILLS
from continuous_brain.vision import PixelStatsBackbone
from continuous_brain.individual_cli import build_parser


@pytest.fixture(autouse=True)
def limit_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def runtime(path, **kwargs):
    env = kwargs.pop("env", DevelopmentRoom(seed=7))
    return IndividualRuntime(env, path, CognitiveConfig.profile("tiny"), PixelStatsBackbone(), **kwargs)


def weights_hash(net):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()


def test_skill_composition_is_generic_and_precondition_checked():
    assert [SKILLS[i].name for i in compose("inspected", {"interaction_available"})] == [
        "search", "orient", "navigate", "inspect"]
    assert [SKILLS[i].name for i in compose("inspected", {"near", "interaction_available"})] == ["inspect"]
    assert not available(set())[3]
    with pytest.raises(ValueError):
        compose("teleport", set())


def test_workspace_depends_on_memory_and_history():
    torch.manual_seed(2)
    cfg = CognitiveConfig.profile("tiny")
    net = CognitiveCore(cfg)
    x, s, g, m = torch.zeros(1,54), torch.zeros(1,24), torch.zeros(1,dtype=torch.long), torch.zeros(1,4,62)
    h = net.initial_state()
    h1, z1, q1 = net(x,s,g,m,h)
    h2, z2, q2 = net(x,s,g,m,h1)
    hm, zm, qm = net(x,s,g,torch.ones_like(m),h)
    assert h1.shape == (1,4,64)
    assert q1.shape == (1,6)
    assert not torch.allclose(h1,h2)
    assert not torch.allclose(z1,zm)
    (q1.square().mean()+qm.square().mean()).backward()
    assert net.memory.weight.grad.abs().sum() > 0
    assert net.recurrence.weight_ih.grad.abs().sum() > 0


def test_contract_rejects_extra_privileged_fields():
    with pytest.raises(ValueError, match="contract"):
        tensor_packet({"sensory": [], "self": [], "goal": 0, "memories": [],
                       "ground_truth_location": [1,2]}, "cpu")


def test_sequence_training_changes_parameters_and_stays_finite(tmp_path):
    r = runtime(tmp_path)
    before = weights_hash(r.net)
    try:
        r.run(decisions=12,realtime=False,log_every=100)
        assert r.learner.updates == 2
        assert weights_hash(r.net) != before
        assert np.isfinite(list(r.learner.last.values())).all()
        assert r.learner.last["gradient_norm"] > 0
        assert len(r.learner.protected) >= 1
        assert r.store.count("experience") == 12
        assert len(r.last_retrieved) > 0
    finally:
        r.close()


def test_restart_restores_identity_workspace_intention_and_learner(tmp_path):
    r = runtime(tmp_path)
    r.run(decisions=8,realtime=False,log_every=100)
    state = copy.deepcopy(r.self_model.state_dict())
    workspace = r.workspace.clone()
    intention = copy.deepcopy(r.intention.state_dict())
    h = weights_hash(r.net)
    r.close()
    resumed = runtime(tmp_path)
    try:
        assert resumed.self_model.state_dict() == state
        assert resumed.intention.state_dict() == intention
        assert torch.equal(resumed.workspace, workspace)
        assert weights_hash(resumed.net) == h
        assert resumed.learner.updates == 1
        assert len(resumed.learner.replay) == 8
        assert len(resumed.learner.optimizer.state) > 0
        resumed.run(decisions=4,realtime=False,log_every=100)
        assert resumed.self_model.decisions == 12
    finally:
        resumed.close()


def test_split_run_exactly_matches_uninterrupted_cpu_run(tmp_path):
    full = runtime(tmp_path/"full")
    full.run(decisions=16,realtime=False,log_every=100)
    expected = weights_hash(full.net)
    expected_workspace = full.workspace.clone()
    expected_position = full.env.position.copy()
    expected_reward = full.total_reward
    full.close()
    split = runtime(tmp_path/"split")
    split.run(decisions=8,realtime=False,log_every=100)
    split.close()
    resumed = runtime(tmp_path/"split")
    try:
        resumed.run(decisions=8,realtime=False,log_every=100)
        assert weights_hash(resumed.net) == expected
        assert torch.equal(resumed.workspace, expected_workspace)
        assert np.array_equal(resumed.env.position, expected_position)
        assert resumed.total_reward == expected_reward
    finally:
        resumed.close()


def test_episode_boundary_does_not_zero_workspace(tmp_path):
    r = runtime(tmp_path,env=DevelopmentRoom(seed=3,max_episode_steps=1))
    try:
        r.run(decisions=3,realtime=False,log_every=100)
        assert r.self_model.tasks == 3
        assert torch.count_nonzero(r.workspace) > 0
        assert r.self_model.decisions == 3
    finally:
        r.close()


def test_transaction_rolls_back_uncommitted_memories_with_checkpoint(tmp_path):
    r = runtime(tmp_path)
    identity = r.self_model.individual_id
    r.decision()  # intentionally crash-equivalent close, not run/save
    assert r.store.count("experience") == 1
    r.close()
    resumed = runtime(tmp_path)
    try:
        assert resumed.self_model.individual_id == identity
        assert resumed.self_model.decisions == 0
        assert resumed.store.count("experience") == 0
    finally:
        resumed.close()


def test_only_one_writer_per_individual(tmp_path):
    r = runtime(tmp_path)
    try:
        with pytest.raises(RuntimeError,match="already running"):
            LifetimeStore(tmp_path)
    finally:
        r.close()


def test_dimension_change_fails_without_replacing_identity(tmp_path):
    r = runtime(tmp_path)
    identity = r.self_model.individual_id
    r.close()
    with pytest.raises(ValueError,match="configuration differs"):
        IndividualRuntime(DevelopmentRoom(),tmp_path,CognitiveConfig.profile("compact"),PixelStatsBackbone())
    r = runtime(tmp_path)
    assert r.self_model.individual_id == identity
    r.close()


def test_evaluation_has_no_teacher_learning_or_persistent_changes(tmp_path):
    r = runtime(tmp_path)
    r.run(decisions=8,realtime=False,log_every=100)
    identity, original = r.self_model.individual_id, weights_hash(r.net)
    updates = r.learner.updates
    r.close()
    before = hashlib.sha256((tmp_path/"individual.sqlite3").read_bytes()).hexdigest()
    e = runtime(tmp_path,evaluate=True)
    try:
        e.run(decisions=4,realtime=False,log_every=100)
        assert e.teacher_probability() == 0
        assert e.last_selection == "learned_option_value"
        assert e.learner.updates == updates
        assert weights_hash(e.net) == original
    finally:
        e.close()
    after = hashlib.sha256((tmp_path/"individual.sqlite3").read_bytes()).hexdigest()
    assert before == after
    r = runtime(tmp_path)
    assert r.self_model.individual_id == identity
    assert r.self_model.decisions == 8
    r.close()


class FaultRoom(DevelopmentRoom):
    def step(self, action):
        obs,r,done,info = super().step(action)
        return obs, -1000, True, {"infrastructure_fault": True}
    def recover_sensor_fault(self):
        return {"recovered": False}


def test_fault_does_not_update_workspace_reward_memory_or_outcomes(tmp_path):
    r = runtime(tmp_path,env=FaultRoom())
    before = r.workspace.clone()
    try:
        r.run(decisions=20,realtime=False,log_every=100)
        assert r.self_model.decisions == 1
        assert r.self_model.faults == 1
        assert r.self_model.tasks == 0
        assert r.self_model.skill_stats == {}
        assert r.total_reward == 0
        assert torch.equal(before,r.workspace)
        assert len(r.learner.replay) == 0
        assert r.store.count("experience") == 0
        assert r.store.count("diagnostic") == 1
    finally:
        r.close()


def test_diagnostics_are_never_retrieved(tmp_path):
    s = LifetimeStore(tmp_path)
    try:
        s.append(0,"diagnostic",np.ones(54),np.ones(62),message="secret metadata")
        vectors,ids = s.retrieve(np.ones(54),4,62)
        assert not ids
        assert np.count_nonzero(vectors) == 0
    finally:
        s.close()


def test_import_preserves_exact_motor_weights_and_original_files(tmp_path):
    source = tmp_path/"old"; source.mkdir()
    torch.manual_seed(9)
    motor = ContinuousBrainNet(6,5,16,32,48)
    torch.save({"model":motor.state_dict(),"version":"0.5.3","steps":123,
                "goals":17,"environment_state":{"scene":"Fake"}},source/"brain.pt")
    (source/"episodic.sqlite3").write_bytes(b"preserved history")
    oldhash = hashlib.sha256((source/"brain.pt").read_bytes()).hexdigest()
    r = runtime(tmp_path/"new",import_v053=source)
    try:
        assert weights_hash(r.motor) == weights_hash(motor)
        assert all(not p.requires_grad for p in r.motor.parameters())
        assert r.self_model.lineage["legacy_steps"] == 123
        assert (tmp_path/"new/legacy_v053/episodic.sqlite3").read_bytes() == b"preserved history"
        r.run(decisions=8,realtime=False,log_every=100)
        assert weights_hash(r.motor) == weights_hash(motor)
        assert hashlib.sha256((source/"brain.pt").read_bytes()).hexdigest() == oldhash
    finally:
        r.close()
    clone = runtime(tmp_path/"new")
    assert weights_hash(clone.motor) == weights_hash(motor)
    clone.close()


def test_import_rejects_same_shape_wrong_environment(tmp_path):
    source = tmp_path/"old"; source.mkdir()
    net = ContinuousBrainNet(6,5,16,32)
    torch.save({"model":net.state_dict(),"environment_state":{}},source/"brain.pt")
    with pytest.raises(ValueError,match="foraging"):
        runtime(tmp_path/"new",import_v053=source)


def test_import_rejects_incompatible_vision(tmp_path):
    source = tmp_path/"old"; source.mkdir()
    net = ContinuousBrainNet(6,5,16,32,384)
    torch.save({"model":net.state_dict(),"environment_state":{"scene":"Fake"}},source/"brain.pt")
    with pytest.raises(ValueError,match="visual dimension"):
        runtime(tmp_path/"new",import_v053=source)


def test_protected_replay_does_not_bridge_fault_gap():
    learner = SequenceLearner(CognitiveCore(CognitiveConfig.profile("tiny")))
    for serial in (1,2,4,5,6,7):
        learner.append({"serial":serial})
    assert not learner.protected
    assert not learner._samples()


def test_development_observation_hides_occluded_beacon():
    env = DevelopmentRoom()
    env.position = np.array([-1.5,0.0]); env.target = np.array([-1.5,2.0]); env.yaw = 0
    assert np.array_equal(env._obs()[:2],np.zeros(2))
    env.target = np.array([-1.5,0.3])
    assert env._obs()[0] > 0


def test_offline_consolidation_changes_only_learning_state(tmp_path):
    r = runtime(tmp_path)
    try:
        r.run(decisions=12,realtime=False,log_every=100)
        ticks = r.self_model.ticks
        pos = r.env.position.copy()
        before = weights_hash(r.net)
        r.learner.train()
        assert weights_hash(r.net) != before
        assert r.self_model.ticks == ticks
        assert np.array_equal(pos,r.env.position)
    finally:
        r.close()


def test_corrupt_checkpoint_fails_closed(tmp_path):
    r = runtime(tmp_path); r.close()
    conn = sqlite3.connect(tmp_path/"individual.sqlite3")
    conn.execute("UPDATE checkpoint SET digest='broken'"); conn.commit(); conn.close()
    with pytest.raises(RuntimeError,match="integrity"):
        runtime(tmp_path)


def test_cli_defaults_are_v06_not_legacy():
    args = build_parser().parse_args([])
    assert args.environment == "development"
    assert args.profile == "standard"
    assert args.goal_signal == "internal-map"
    assert args.vision_backbone == "pixel-stats"


def test_shallow_imagination_engages_only_after_warmup(tmp_path):
    r = runtime(tmp_path)
    try:
        packet = r.packet(r.obs)
        with torch.no_grad():
            _,z,q = r.net(*tensor_packet(packet,"cpu"),r.workspace)
        r.evaluate = True
        r._select(q,[True]*6,0,z)
        assert r.last_imagination is None
        r.learner.updates = r.cfg.imagination_after_updates
        r._select(q,[True]*6,0,z)
        assert len(r.last_imagination["rewards"]) == 6
    finally:
        r.close()


def test_backup_and_fork_preserve_history_but_label_identity(tmp_path):
    from scripts.snapshot_individual import snapshot
    r = runtime(tmp_path/"source")
    r.run(decisions=8,realtime=False,log_every=100)
    identity, h = r.self_model.individual_id, weights_hash(r.net)
    with pytest.raises(RuntimeError,match="already running"):
        snapshot(tmp_path/"source",tmp_path/"while_running")
    r.close()
    result = snapshot(tmp_path/"source",tmp_path/"backup")
    assert result["individual_id"] == identity
    snapshot(tmp_path/"source",tmp_path/"fork",True)
    fork = runtime(tmp_path/"fork")
    try:
        assert fork.self_model.individual_id != identity
        assert fork.self_model.lineage["parent_individual_id"] == identity
        assert weights_hash(fork.net) == h
        assert fork.store.count("experience") == 8
    finally:
        fork.close()
    with pytest.raises(FileExistsError):
        snapshot(tmp_path/"source",tmp_path/"backup")


def test_thor_runtime_uses_retained_adapter_with_mock_controller(tmp_path):
    from continuous_brain.embodied import AI2ThorConfig, AI2ThorEmbodiedEnv
    from test_sensor_recovery import RecoverableController
    source = tmp_path/"old"; source.mkdir()
    motor = ContinuousBrainNet(6,5,16,32,48)
    torch.save({"model":motor.state_dict(),"version":"0.5.3",
                "environment_state":{"scene":"Fake"}},source/"brain.pt")
    env = AI2ThorEmbodiedEnv(AI2ThorConfig(scene="Fake",curriculum_level=2,
        random_start=False,rehearsal_fraction=0.0,goal_signal="internal-map"),
        controller_factory=RecoverableController)
    r = runtime(tmp_path/"new",env=env,import_v053=source)
    try:
        r.run(decisions=3,realtime=False,log_every=100)
        assert r.self_model.decisions == 3
        assert r.status()["motor_source"] == "imported_learned"
        assert weights_hash(r.motor) == weights_hash(motor)
    finally:
        r.close()


def test_no_intrinsic_bonus_for_idle_success_without_goal_completion(tmp_path):
    r = runtime(tmp_path)
    try:
        r._select = lambda *args: 0  # force observe, goal is still inspect
        r.decision()
        row = r.learner.replay[-1]
        assert not row["success"]
        assert row["reward"] <= 0
    finally:
        r.close()


def test_formative_memory_remains_in_candidate_pool(tmp_path):
    s = LifetimeStore(tmp_path)
    try:
        first = np.zeros(54,dtype=np.float32); first[0] = 1
        later = np.zeros(54,dtype=np.float32); later[1] = 1
        s.append(0,"experience",first,np.ones(62))
        for t in range(20):
            s.append(t+1,"experience",later,np.zeros(62))
        vec,ids = s.retrieve(first,1,62,scan_limit=8)
        assert ids == [1]
        assert np.all(vec == 1)
    finally:
        s.close()


def test_empirical_skill_estimates_enter_neural_self_input(tmp_path):
    r = runtime(tmp_path)
    try:
        before = r.self_model.vector()
        r.self_model.observe_outcome("orient",True,3)
        after = r.self_model.vector()
        assert after.shape == (24,)
        assert after[13] > before[13]  # orient confidence
        assert after[19] > before[19]  # orient cost
    finally:
        r.close()


def test_sigterm_saves_and_releases_lock(tmp_path):
    import subprocess
    import sys
    import time
    process = subprocess.Popen([sys.executable,"-m","continuous_brain.individual_cli",
        "--profile","tiny","--state-dir",str(tmp_path),"--fast","--checkpoint-every","1",
        "--log-every","1000","--torch-threads","1"],
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic()+20
        observed = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            path = tmp_path/"individual_status.json"
            if path.exists():
                observed = json.loads(path.read_text())
                if observed["individual"]["decisions"] >= 2:
                    break
            time.sleep(0.02)
        assert observed and observed["individual"]["decisions"] >= 2
        process.terminate()
        assert process.wait(timeout=15) == 0
        resumed = runtime(tmp_path)
        try:
            assert resumed.self_model.individual_id == observed["individual"]["individual_id"]
            assert resumed.self_model.decisions >= 2
        finally:
            resumed.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_individual_dashboard_http_and_state():
    import urllib.request
    from continuous_brain.individual_visualizer import individual_visualizer
    view = individual_visualizer(port=0)
    try:
        view.update({"individual":{"name":"test"}})
        url = view.start()
        html = urllib.request.urlopen(url).read().decode()
        assert "Persistent Individual Core" in html
        assert "not evidence of consciousness" in html
        data = json.load(urllib.request.urlopen(url+"/state"))
        assert data["individual"]["name"] == "test"
    finally:
        view.close()
