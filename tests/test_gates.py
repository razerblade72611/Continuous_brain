"""G1-G5 harness on fixture runtimes (engineering validation; fixtures are not mechanisms)."""

import numpy as np
import pytest

from continuous_brain_protocols import ResultStatus, SeedFamily, SeedPurpose
from continuous_brain_eval.gates import (STANDARD_INTERVENTIONS, STANDARD_SCHEDULES, InteractionMatrix,
                                         RetentionThreshold, run_g1, run_g2, run_g3, run_g4_drift,
                                         run_g4_save_restore, run_g5)
from continuous_brain_eval.seeds import SeedOverlapError
from continuous_brain_unified import Coprocessor, CoprocessorResponse, UnifiedRuntime

from fixtures import fixture_runtime, observation

FRESH = SeedFamily("g-test", 905000, 905009, SeedPurpose.ENGINEERING)


class Echo(Coprocessor):
    name = "echo"

    def query(self, request):
        return CoprocessorResponse(True, self.name, {"ok": True})


class RecordMean:
    mechanism_id, slot = "fixture_autobiography", "autobiography"

    def evaluate(self, view, seed):
        return {"record_mean": float(np.mean(view.read("autobiography.record")))}


def test_g1_is_blocked_with_nothing_bound_and_refuses_spent_seeds():
    rec = run_g1(lambda s: UnifiedRuntime(), [], FRESH, [], ticks=2)
    assert rec.status == ResultStatus.BLOCKED
    with pytest.raises(SeedOverlapError):
        run_g1(lambda s: UnifiedRuntime(), [], SeedFamily("bad", 80000, 80010, SeedPurpose.TEST), [], ticks=2)


def test_g1_runs_evaluators_but_emits_no_pass_without_an_approved_prereg():
    rec = run_g1(fixture_runtime, [RecordMean()], FRESH,
                 [RetentionThreshold("fixture_autobiography", "record_mean", 1.0, "fixture")], ticks=5,
                 observe=observation, prereg_id="UI-GATE-v0", max_seeds=3)
    assert rec.status == ResultStatus.ENGINEERING_ONLY
    assert rec.metrics["threshold_checks"][0]["meets"] is True
    rec.validate()


def _matrix():
    ivs = tuple(i for i in STANDARD_INTERVENTIONS if i.name in ("erase_autobiography", "stale_plan"))
    metrics = {"self_mean": lambda v: float(np.mean(v.read("self_state.vector"))),
               "wm_mean": lambda v: float(np.mean(v.read("working_memory.contents")))}
    cells = {("erase_autobiography", "self_mean"): "-", ("erase_autobiography", "wm_mean"): "0",
             ("stale_plan", "wm_mean"): "0"}
    return InteractionMatrix(ivs, metrics, cells, equivalence_margin=1e-9, start_tick=2)


def test_g2_matrix_must_be_frozen_and_detects_predicted_signs():
    m = _matrix()
    with pytest.raises(RuntimeError, match="freeze"):
        run_g2(fixture_runtime, m, [905001, 905002, 905003], ticks=6, observe=observation)
    m.freeze()
    rec = run_g2(fixture_runtime, m, [905001, 905002, 905003, 905004], ticks=6, observe=observation, n_resamples=500)
    cells = {(c["intervention"], c["metric"]): c for c in rec.metrics["cells"]}
    assert cells[("erase_autobiography", "self_mean")]["consistent_with_prediction"]
    assert cells[("erase_autobiography", "wm_mean")]["consistent_with_prediction"]   # non-interference holds
    assert cells[("stale_plan", "wm_mean")]["status"] == "BLOCKED"                    # no planner bound
    assert rec.status == ResultStatus.ENGINEERING_ONLY
    m.equivalence_margin = 0.5
    with pytest.raises(RuntimeError, match="changed"):
        run_g2(fixture_runtime, m, [905001], ticks=3, observe=observation)


def test_g3_blocks_schedules_whose_slots_are_unbound():
    metrics = {"self_mean": lambda v: float(np.mean(v.read("self_state.vector")))}
    rec = run_g3(fixture_runtime, STANDARD_SCHEDULES, metrics, [905001, 905002], ticks=20, margin=1.0, observe=observation)
    status = {s["schedule"]: s["status"] for s in rec.metrics["schedules"]}
    assert status["rapid_goal_churn"] == "BLOCKED"
    assert status["long_run_shared_state"] == "RUN"
    assert rec.status == ResultStatus.ENGINEERING_ONLY


def test_g4_save_restore_is_exact_and_drift_is_reported(tmp_path):
    rec = run_g4_save_restore(fixture_runtime, seed=905003, save_at=5, continue_ticks=6, observe=observation,
                              store_root=str(tmp_path))
    assert rec.metrics["exact_channels_preserved"] and rec.metrics["final_state_equal"]
    assert rec.metrics["snapshot_verified"]
    drift = run_g4_drift(fixture_runtime, 905003, window_ticks=5, windows=4,
                         metric=lambda v: float(np.mean(v.read("working_memory.contents"))), observe=observation)
    assert len(drift.metrics["window_values"]) == 4 and drift.status == ResultStatus.ENGINEERING_ONLY


def test_g5_reports_borrowed_capability_and_is_never_gated():
    metrics = {"self_mean": lambda v: float(np.mean(v.read("self_state.vector")))}
    rec = run_g5(lambda s: fixture_runtime(s, coprocessor=Echo(), enabled=True), metrics, [905001, 905002], ticks=4,
                 observe=observation)
    assert rec.metrics["gated"] is False
    assert rec.metrics["borrowed"]["self_mean"] > 0
    assert rec.metrics["coprocessor_calls"]["with"] > 0
    assert rec.status == ResultStatus.ENGINEERING_ONLY


def test_gates_refuse_spent_or_unregistered_seeds():
    m = _matrix()
    m.freeze()
    with pytest.raises(SeedOverlapError):
        run_g2(fixture_runtime, m, [80001], ticks=3, observe=observation)
    with pytest.raises(SeedOverlapError):
        run_g4_save_restore(fixture_runtime, seed=7, save_at=2, continue_ticks=2, observe=observation)


def test_an_unrelated_approved_prereg_cannot_turn_a_gate_green(tmp_path, monkeypatch):
    import json
    (tmp_path / "preregistrations.json").write_text(json.dumps({"preregistrations": [
        {"id": "OTHER", "title": "t", "status": "APPROVED", "approved_by": "David", "allowed_items": ["B4"]}]}))
    monkeypatch.setenv("CONTINUOUS_BRAIN_REGISTRY", str(tmp_path))
    import shutil
    from pathlib import Path
    shutil.copy(Path(__file__).resolve().parents[1] / "registry" / "seed_ranges.json", tmp_path / "seed_ranges.json")
    rec = run_g4_save_restore(lambda s: UnifiedRuntime(), seed=905001, save_at=2, continue_ticks=2, prereg_id="OTHER")
    assert rec.status == ResultStatus.ENGINEERING_ONLY and "verdict" not in rec.metrics
