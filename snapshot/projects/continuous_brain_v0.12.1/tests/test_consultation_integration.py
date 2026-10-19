"""Evidence-contract tests. Synthetic report fixtures are not capability evidence."""
import json

import numpy as np
import pytest

from continuous_brain.evidence_v114 import P4B_PROTOCOL, sha256
from continuous_brain import generalization_gate_v110 as gate


def fixture_report(tmp_path):
    checkpoint = tmp_path / "fixture-not-a-model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity for report validation only")
    report = dict(format="continuous_brain.p4b_hierarchical_option_planning_gate.v103",
        checkpoint_sha256=sha256(checkpoint), p4b_passed=True,
        experiment=dict(P4B_PROTOCOL),
        hierarchical_option_planning=dict(completion_rate=.4, mean_collisions=1.),
        reactive=dict(completion_rate=.1, mean_collisions=1.))
    path = tmp_path / "fixture-report.json"
    return checkpoint, path, report


def test_valid_consistent_report(tmp_path):
    checkpoint, path, report = fixture_report(tmp_path)
    path.write_text(json.dumps(report))
    assert gate.verify_replication_prerequisite(path, checkpoint)["replication_p4b_passed"] is True


@pytest.mark.parametrize("change", ["hash", "false", "string", "protocol", "metrics", "format"])
def test_unqualified_reports_rejected(tmp_path, change):
    checkpoint, path, report = fixture_report(tmp_path)
    if change == "hash": report["checkpoint_sha256"] = "0" * 64
    if change == "false": report["p4b_passed"] = False
    if change == "string": report["p4b_passed"] = "false"
    if change == "protocol": report["experiment"]["paired_episodes"] = 1
    if change == "metrics": report["hierarchical_option_planning"]["completion_rate"] = 0.
    if change == "format": report["format"] = "unrelated"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError): gate.verify_replication_prerequisite(path, checkpoint)


def test_rejection_precedes_model_loading(tmp_path, monkeypatch):
    checkpoint, path, report = fixture_report(tmp_path)
    report["p4b_passed"] = False
    path.write_text(json.dumps(report))
    monkeypatch.setattr(gate, "load_checkpoint", lambda *a: pytest.fail("Model should not load"))
    with pytest.raises(RuntimeError):
        gate.evaluate_generalization_gate(checkpoint, replication_report=path,
            episodes=1, steps=2, seed_start=101)


def test_output_cannot_overwrite_input(tmp_path):
    checkpoint, path, report = fixture_report(tmp_path)
    path.write_text(json.dumps(report))
    before = checkpoint.read_bytes()
    for output in (checkpoint, path):
        with pytest.raises(ValueError):
            gate.evaluate_generalization_gate(checkpoint, replication_report=path,
                episodes=1, steps=2, seed_start=101, output=output)
    assert checkpoint.read_bytes() == before


def test_positive_point_estimate_can_be_inconclusive():
    reference = [dict(seed=i, completed=False, collisions=0, reward=0) for i in range(20)]
    candidate = [dict(row, completed=(row["seed"] < 2)) for row in reference]
    result = gate.paired_comparison(candidate, reference)["completion_rate"]
    assert result["mean_difference"] > .05
    assert result["lo"] == 0 and not result["supported_advantage"]


def test_paired_seeds_must_match():
    with pytest.raises(ValueError): gate.paired_comparison([dict(seed=1)], [dict(seed=2)])
    with pytest.raises(ValueError): gate.paired_comparison([dict(seed=1)]*2, [dict(seed=1)]*2)


def test_empty_or_nonfinite_bootstrap_rejected():
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError): gate._bootstrap_ci(np.array(values))


def test_reserved_seed_roles():
    for start in (51000, 61000, 71000):
        with pytest.raises(ValueError):
            gate.validate_evaluation_settings("randomized_detour", 100, 160, start, "diagnostic")
    gate.validate_evaluation_settings("randomized_detour", 100, 160, 41001, "diagnostic")
    gate.validate_evaluation_settings("randomized_detour", 100, 160, 61001, "confirmation")
    gate.validate_evaluation_settings("two_barrier", 100, 160, 71001, "confirmation")
    with pytest.raises(ValueError):
        gate.validate_evaluation_settings("two_barrier", 100, 160, 61001, "confirmation")


def test_reward_support_is_not_completion_support():
    report = dict(pass_thresholds_fixed_before_evaluation=dict(gate.PASS_THRESHOLDS),
        hierarchical_option_planning=dict(completion_rate=.4), reactive=dict(completion_rate=.1),
        advantages_vs_reactive=dict(completion_percentage_points=30, collision_regression=0),
        episodes={"hierarchical_option_planning": [dict(seed=1, completed=False, collisions=2)]},
        benchmark_threshold_passed=True,
        statistically_supported_advantage=dict(completion_rate=False, reward=True, collisions_reduced=True))
    diagnosis = gate.diagnose_gate_result(report)
    assert diagnosis["next_step"].startswith("Inspect")
    assert diagnosis["non_completed_seeds"] == [1]
