import torch

from conftest import small_config, small_core
from continuous_brain.agent import ContinuousBrain
from continuous_brain.sleep import CoreHealthProbe, Probe, RecallProbe, RegressionSuite
from continuous_brain.testing import TinyBackbone


class _Scripted(Probe):
    """Returns a scripted sequence of scores (to force pass/fail deterministically)."""

    def __init__(self, name, component, scores):
        self.name, self.component, self.scores = name, component, list(scores)
        self.guarded = True

    def evaluate(self, system, episodes):
        return self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]


def _brain(tmp_path, probes):
    return ContinuousBrain(small_core(), TinyBackbone(hidden=64), tmp_path, small_config(),
                           suite=RegressionSuite(probes))


def _experience(brain):
    for q, a, o in [("name?", " David.", 1.0), ("color?", " Blue.", -1.0), ("home?", " Texas.", 0.5)]:
        r = brain.tick(message=q)
        brain.feedback(r.episode_id, outcome=o, target=a)
        brain.run(3)


def _learned_state(brain):
    return {"w": brain.core.w.clone(),
            "prefix": {k: v.clone() for k, v in brain.system.prefix.state_dict().items()}}


def _same(a, b):
    return torch.equal(a["w"], b["w"]), all(torch.equal(a["prefix"][k], b["prefix"][k]) for k in a["prefix"])


def test_skips_with_no_episodes(tmp_path):
    rep = _brain(tmp_path, [RecallProbe()]).sleep()
    assert rep.skipped and not rep.rolled_back


def test_passing_sleep_keeps_updates_and_consolidates(tmp_path):
    brain = _brain(tmp_path, [RecallProbe(), CoreHealthProbe(low_hz=0.0, high_hz=1e6)])
    _experience(brain)
    before = _learned_state(brain)
    rep = brain.sleep()
    assert rep.passed and rep.accepted_components == ["bridge", "core"]
    core_same, prefix_same = _same(before, _learned_state(brain))
    assert not core_same and not prefix_same
    assert rep.stats["bridge"]["loss_after"] < rep.stats["bridge"]["loss_before"]
    assert all(e.consolidated_in for e in brain.store.all())
    assert brain.snapshots.verify(rep.pre_snapshot) and brain.snapshots.verify(rep.post_snapshot)


def test_core_regression_rolls_back_core_only(tmp_path):
    brain = _brain(tmp_path, [_Scripted("core_task", "core", [1.0, 0.0, 1.0]), RecallProbe()])
    _experience(brain)
    before = _learned_state(brain)
    rep = brain.sleep()
    assert rep.rolled_back_components == ["core"] and rep.accepted_components == ["bridge"]
    core_same, prefix_same = _same(before, _learned_state(brain))
    assert core_same and not prefix_same
    assert not any(e.consolidated_in for e in brain.store.all())   # stays eligible


def test_bridge_regression_rolls_back_bridge_only(tmp_path):
    brain = _brain(tmp_path, [_Scripted("lang", "bridge", [1.0, 0.0, 1.0])])
    _experience(brain)
    before = _learned_state(brain)
    rep = brain.sleep()
    assert rep.rolled_back_components == ["bridge"]
    core_same, prefix_same = _same(before, _learned_state(brain))
    assert prefix_same and not core_same


def test_persistent_failure_rolls_back_everything(tmp_path):
    brain = _brain(tmp_path, [_Scripted("core_task", "core", [1.0, 0.0, 0.0])])
    _experience(brain)
    before = _learned_state(brain)
    rep = brain.sleep()
    assert set(rep.rolled_back_components) == {"bridge", "core"}
    assert _same(before, _learned_state(brain)) == (True, True)


def test_exception_rolls_back_everything(tmp_path):
    class Boom(Probe):
        name, component, guarded = "boom", "core", True
        calls = 0

        def evaluate(self, system, episodes):
            Boom.calls += 1
            if Boom.calls > 1:
                raise RuntimeError("probe crashed")
            return 0.0

    brain = _brain(tmp_path, [Boom()])
    _experience(brain)
    before = _learned_state(brain)
    rep = brain.sleep()
    assert not rep.passed and any("exception" in f for f in rep.failures)
    assert _same(before, _learned_state(brain)) == (True, True)


def test_sleep_log_is_written(tmp_path):
    brain = _brain(tmp_path, [RecallProbe()])
    _experience(brain)
    brain.sleep()
    assert (tmp_path / "logs" / "sleep_log.jsonl").read_text().strip()
