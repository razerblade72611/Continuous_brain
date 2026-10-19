#!/usr/bin/env python3
"""Fixture tests for the diagnostic; no neural DEV/TEST execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import diagnose_compositional_goal_completion_v1 as d


def fixture(second_after=.34):
    bootstrap = {"context_0": [.1, .9], "context_1": [.8, .2], "context_2": [.3, .8]}
    clauses = [{"context": "context_0", "primary": 1, "baseline_reliability": .1, "target_reliability": .1 + .12},
               {"context": "context_1", "primary": 2, "baseline_reliability": .2, "target_reliability": .2 + .12}]
    goal = {"operator": "THEN", "clauses": clauses}
    initial_caps = {"context_0": [.2, .9], "context_1": [.8, .2], "context_2": [.3, .8]}
    final_rel = {"context_0": [.4, .9], "context_1": [.8, second_after], "context_2": [.3, .8]}
    final_caps = copy.deepcopy(initial_caps)
    final_caps["context_0"][0] = .7
    both_met = second_after >= clauses[1]["target_reliability"]
    row = {
        "subject_id": 130000, "goal_generated": True, "goal": goal,
        "bootstrap_reliability": bootstrap, "initial_capabilities": initial_caps,
        "final_reliability": final_rel, "final_capabilities": final_caps,
        "trace": [{"clause_index": 0, "context": "context_0", "primary": 1}],
        "clause_attempts": [1, 0], "clause_capability_gains": [.7 - .2, 0.0],
        "both_final_targets_met": both_met, "both_clauses_complete": both_met,
        "both_capability_gains_at_least_0_35": False,
        "progress": {"goal": goal, "cursor": 2 if both_met else 1, "attempts": [1, 0], "exhausted": False},
    }
    for mode in d.MODES:
        trace = [{"context": "context_0", "primary": 1} for _ in range(32)]
        if mode == "misbind_second":
            trace[16:] = [{"context": "context_2", "primary": 2} for _ in range(16)]
        row[mode + "_trace"] = trace
        row[mode + "_final_reliability"] = copy.deepcopy(final_rel)
        row[mode + "_final_capabilities"] = copy.deepcopy(final_caps)
        row[mode + "_missing_target_unchanged"] = True
        row[mode + "_both_targets_met"] = both_met
    return row


def replay_evidence(row):
    audit = d.audit_row(row)
    post = copy.deepcopy(row["bootstrap_reliability"])
    post["context_1"] = row["drop_second_final_reliability"]["context_1"]
    evidence = d.attribution_evidence(audit, row, row["bootstrap_reliability"], post,
                                      row["initial_capabilities"], row["initial_capabilities"], row["goal"])
    for c in evidence["clauses"]:
        c["interruption_was_target_episode"] = False
        c["history_probe_target_met"] = {"bootstrap_12": False,
                                           "drop_oldest_only_11": c["interruption_crosses_target"],
                                           "retain_all_plus_interruption_13": False,
                                           "actual_rolling_window_12": c["target_met_after_interruption"]}
    return audit, {"subject_id": row["subject_id"], "replay": evidence}


class DiagnosticTests(unittest.TestCase):
    def test_detects_apparent_completion_without_second_practice(self):
        audit = d.audit_row(fixture())
        self.assertTrue(audit["normal_both_complete"])
        self.assertTrue(audit["normal_completion_without_both_practiced"])
        self.assertTrue(audit["normal_completion_with_subthreshold_gain"])
        self.assertFalse(audit["normal_predicate_and_gain_conjunction"])
        self.assertTrue(audit["controls"]["drop_second"]["both_targets_met"])

    def test_confirms_reproduced_interruption_crossing(self):
        audit, replay = replay_evidence(fixture())
        result = d.summary([audit], [replay])
        self.assertEqual(result["attribution_status"], "ALL_OBSERVED_CONTROL_SUCCESSES_EXPLAINED_BY_INTERRUPTION")
        self.assertEqual(result["replay_counts"]["drop_second_successful_controls_explained"], 1)
        self.assertEqual(result["replay_counts"]["clause_2_crossings_without_target_episode_or_gain"], 1)

    def test_distinguishes_no_control_successes(self):
        audit, replay = replay_evidence(fixture(second_after=.21))
        self.assertEqual(d.summary([audit], [replay])["attribution_status"], "NO_SUCCESSFUL_CONTROLS_TO_ATTRIBUTE")

    def test_reproduction_mismatch_blocks_confirmation(self):
        audit, replay = replay_evidence(fixture())
        replay["replay"]["matches"]["bootstrap_exact"] = False
        result = d.summary([audit], [replay])
        self.assertEqual(result["attribution_status"], "REPLAY_MISMATCH_UNRESOLVED")
        self.assertEqual(result["mismatched_subject_ids"], [130000])

    def test_capability_change_prevents_no_gain_explanation(self):
        row = fixture()
        audit = d.audit_row(row)
        before = copy.deepcopy(row["initial_capabilities"])
        before["context_1"][1] = .1
        post = copy.deepcopy(row["bootstrap_reliability"])
        post["context_1"][1] = .34
        e = d.attribution_evidence(audit, row, row["bootstrap_reliability"], post,
                                    before, row["initial_capabilities"], row["goal"])
        self.assertFalse(e["control_success_explained_by_interruption"]["drop_second"])

    def test_reads_original_threshold_instead_of_adjusting_it(self):
        row = fixture(second_after=.319)
        audit = d.audit_row(row)
        self.assertFalse(audit["second_target_met_in_untouched_control_context"])
        row["goal"]["clauses"][1]["target_reliability"] = .30
        with self.assertRaisesRegex(ValueError, "target rule changed"):
            d.audit_row(row)

    def test_rejects_missing_context_visit(self):
        row = fixture()
        row["drop_second_trace"][0] = {"context": "context_1", "primary": 2}
        with self.assertRaisesRegex(ValueError, "control policy changed"):
            d.audit_row(row)

    def test_rejects_wrong_control_predicate_flag(self):
        row = fixture()
        row["drop_second_both_targets_met"] = False
        with self.assertRaisesRegex(ValueError, "predicate flag disagrees"):
            d.audit_row(row)

    def test_rejects_disagreement_between_untouched_readouts(self):
        row = fixture()
        row["misbind_second_final_reliability"]["context_1"][1] = .35
        with self.assertRaisesRegex(ValueError, "readouts differ"):
            d.audit_row(row)

    def test_rejects_changed_missing_capability(self):
        row = fixture()
        row["drop_second_final_capabilities"]["context_1"][1] = .3
        with self.assertRaisesRegex(ValueError, "capability changed"):
            d.audit_row(row)

    def test_rejects_non_dev_subject(self):
        row = fixture()
        row["subject_id"] = 140000
        with self.assertRaisesRegex(ValueError, "non-DEV"):
            d.audit_row(row)

    def test_history_probes_preserve_intended_record_order(self):
        original = list(range(12))
        probes = d.probe_windows(original, 99)
        self.assertEqual(probes["bootstrap_12"], (list(range(12)), 12))
        self.assertEqual(probes["drop_oldest_only_11"], (list(range(1, 12)), 12))
        self.assertEqual(probes["retain_all_plus_interruption_13"], (list(range(12)) + [99], 13))
        self.assertEqual(probes["actual_rolling_window_12"], (list(range(1, 12)) + [99], 12))
        self.assertEqual(original, list(range(12)))

    def test_history_probe_rejects_wrong_bootstrap_length(self):
        with self.assertRaises(ValueError):
            d.probe_windows(list(range(11)), 99)

    def test_records_only_does_not_claim_confirmation(self):
        result = d.summary([d.audit_row(fixture())], None)
        self.assertFalse(result["replay_performed"])
        self.assertEqual(result["attribution_status"], "SOURCE_PATH_INFERENCE_ONLY")

    def test_no_goal_rows_remain_in_subject_count(self):
        audit = d.audit_row({"subject_id": 130000, "goal": None, "goal_generated": False})
        result = d.summary([audit], None)
        self.assertEqual(result["record_audit_counts"]["subjects"], 1)
        self.assertEqual(result["record_audit_counts"]["goals_generated"], 0)

    def test_preserved_report_must_be_fail(self):
        with self.assertRaisesRegex(ValueError, "DEV FAIL"):
            d.check_source_report({"family": "v0.37", "version": "0.37.0", "status": "COMPLETED", "development_gate_passed": True}, [], None)

    def test_no_overwrite_and_finite_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            d.write_json(path, {"historical": "FAIL"})
            with self.assertRaises(FileExistsError):
                d.write_json(path, {"historical": "PASS"})
            self.assertEqual(json.loads(path.read_text()), {"historical": "FAIL"})
        with self.assertRaises(ValueError):
            d.strict_json('{"value": NaN}')


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"COMPOSITIONAL_COMPLETION_DIAGNOSTIC_SOFTWARE_TESTS: PASS ({result.testsRun} tests)")
    print("Fixture tests only; no scientific replay performed.")
