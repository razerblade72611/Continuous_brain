#!/usr/bin/env python3
"""Software fixtures only. No checkpoints or scientific DEV/TEST subjects."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from continuous_brain.compositional_autobiographical_goals_v1 import (
    CONTEXTS, CompositeGoal, CompositeProgress, GoalClause, author_goal,
    run_composite, run_control, targets_met,
)
import evaluate_compositional_autobiographical_goals_v1 as evaluation


class FixtureBackend:
    """Deterministic test double; not the project's neural self-model."""
    def __init__(self, increment=.04):
        self.values = {"context_0": [.10, .9], "context_1": [.8, .20], "context_2": [.30, .8]}
        self.increment = increment
        self.commands = []

    def reliabilities(self):
        return copy.deepcopy(self.values)

    def execute(self, context, primary, step):
        self.commands.append([context, primary, step])
        self.values[context][primary - 1] = min(1, self.values[context][primary - 1] + self.increment)
        return {"fixture_step": step}


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureBackend()
        self.goal = author_goal(self.backend.reliabilities())

    def test_selects_two_context_qualified_deficits(self):
        self.assertEqual([(c.context, c.primary) for c in self.goal.clauses], [("context_0", 1), ("context_1", 2)])
        self.assertAlmostEqual(self.goal.clauses[0].target_reliability, .22)
        self.assertAlmostEqual(self.goal.clauses[1].target_reliability, .32)

    def test_author_has_no_privileged_inputs(self):
        self.assertEqual(list(inspect.signature(author_goal).parameters), ["reliabilities"])
        with self.assertRaises(TypeError):
            author_goal(self.backend.reliabilities(), subject_id=123)

    def test_empty_evidence_has_no_goal(self):
        self.assertIsNone(author_goal({c: [.5, .5] for c in CONTEXTS}))

    def test_one_directional_context_cannot_fake_composition(self):
        estimates = {c: [.5, .5] for c in CONTEXTS}
        estimates["context_0"] = [.1, .9]
        self.assertIsNone(author_goal(estimates))

    def test_invalid_estimates_fail_closed(self):
        for value in (float("nan"), float("inf"), -1, 1.1):
            estimates = self.backend.reliabilities()
            estimates["context_0"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                author_goal(estimates)
        with self.assertRaises(ValueError):
            author_goal({"context_0": [.1, .9]})

    def test_transplanted_histories_move_the_goal(self):
        estimates = self.backend.reliabilities()
        rotation = {c: CONTEXTS[(i + 1) % 3] for i, c in enumerate(CONTEXTS)}
        moved = author_goal({rotation[c]: pair for c, pair in estimates.items()})
        self.assertEqual([(c.context, c.primary) for c in moved.clauses], [("context_1", 1), ("context_2", 2)])

    def test_then_order_controls_execution(self):
        progress = CompositeProgress(self.goal)
        trace = run_composite(self.backend, progress)
        self.assertTrue(progress.complete)
        self.assertTrue(all(targets_met(self.goal, self.backend.reliabilities())))
        phases = [t["clause_index"] for t in trace]
        self.assertEqual(phases, sorted(phases))
        self.assertEqual(set(phases), {0, 1})
        self.assertEqual(self.backend.values["context_2"], [.30, .8])

    def test_clauses_cannot_share_one_context(self):
        with self.assertRaises(ValueError):
            CompositeGoal((self.goal.clauses[0], self.goal.clauses[0]))

    def test_rejects_invalid_target_and_operator(self):
        with self.assertRaises(ValueError):
            GoalClause("context_0", 1, .3, .2)
        raw = self.goal.payload()
        raw["operator"] = "OR"
        with self.assertRaises(ValueError):
            CompositeGoal.restore(raw)

    def test_clause_boundary_checks_completion_before_timeout(self):
        progress = CompositeProgress(self.goal, attempts=[16, 0])
        values = self.backend.reliabilities()
        values["context_0"][0] = self.goal.clauses[0].target_reliability
        clause = progress.next_clause(values)
        self.assertEqual(clause, self.goal.clauses[1])
        self.assertFalse(progress.exhausted)

    def test_no_learning_exhausts_first_clause(self):
        backend = FixtureBackend(increment=0)
        progress = CompositeProgress(self.goal)
        trace = run_composite(backend, progress)
        self.assertEqual(len(trace), 16)
        self.assertTrue(progress.exhausted)
        self.assertFalse(progress.complete)
        self.assertEqual(progress.attempts, [16, 0])

    def test_missing_clause_does_not_meet_original_goal(self):
        trace = run_control(self.backend, self.goal, "drop_second")
        self.assertEqual(len(trace), 32)
        self.assertFalse(all(targets_met(self.goal, self.backend.reliabilities())))
        self.assertEqual(self.backend.values["context_1"], [.8, .20])

    def test_misbound_context_does_not_meet_original_goal(self):
        trace = run_control(self.backend, self.goal, "misbind_second")
        self.assertEqual(len(trace), 32)
        self.assertEqual(trace[16]["context"], "context_2")
        self.assertEqual(trace[16]["primary"], 2)
        self.assertFalse(all(targets_met(self.goal, self.backend.reliabilities())))
        self.assertEqual(self.backend.values["context_1"], [.8, .20])

    def test_checkpoint_restarts_identical_continuation(self):
        progress = CompositeProgress(self.goal)
        trace = run_composite(self.backend, progress, pause_after=1)
        raw = json.loads(json.dumps({"progress": progress.payload(), "values": self.backend.values,
                                     "commands": self.backend.commands, "trace": trace}))
        trace = run_composite(self.backend, progress, trace)
        restored = FixtureBackend()
        restored.values = raw["values"]
        restored.commands = raw["commands"]
        restored_progress = CompositeProgress.restore(raw["progress"])
        restored_trace = run_composite(restored, restored_progress, raw["trace"])
        self.assertEqual(progress.payload(), restored_progress.payload())
        self.assertEqual(trace, restored_trace)
        self.assertEqual(self.backend.values, restored.values)
        self.assertEqual(self.backend.commands, restored.commands)

    def test_generator_is_not_call_count_driven(self):
        first = self.goal.payload()
        for _ in range(10):
            self.assertEqual(author_goal(self.backend.reliabilities()).payload(), first)

    def test_frozen_history_does_not_advance(self):
        frozen = self.backend.reliabilities()
        run_composite(self.backend, CompositeProgress(self.goal))
        self.assertEqual(author_goal(frozen).payload(), self.goal.payload())
        self.assertNotEqual(author_goal(self.backend.reliabilities()).payload(), self.goal.payload())

    def test_dev_seed_ranges_are_disjoint_and_test_rejected(self):
        ranges = []
        for c in CONTEXTS:
            seeds = {evaluation.context_trace_id(s, c) for s in evaluation.DEV_IDS}
            self.assertEqual(len(seeds), 128)
            ranges.append(seeds)
        self.assertEqual(len(set.union(*ranges)), 384)
        self.assertTrue(all(not (seeds & set(evaluation.SEALED_TEST_IDS)) for seeds in ranges))
        for sid in (120000, 129999, 130128, 140000, 141000, 142000):
            with self.subTest(sid=sid), self.assertRaises(ValueError):
                evaluation.context_trace_id(sid, CONTEXTS[0])


class IntegrityTests(unittest.TestCase):
    @staticmethod
    def valid_result():
        return {"family": "v0.36", "test_admission_passed": True,
                "split": {"test": [120000, 120127], "test_spent": True},
                "v036_test_spent": True, "protected_unchanged": True,
                "source_models_unchanged": True, "checks": {"fixture": True}}

    def test_source_pass_and_spent_both_required(self):
        evaluation.check_source_result(self.valid_result())
        for key in ("test_admission_passed", "v036_test_spent", "protected_unchanged", "source_models_unchanged"):
            payload = self.valid_result()
            payload[key] = False
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                evaluation.check_source_result(payload)
        payload = self.valid_result()
        payload["checks"] = {}
        with self.assertRaises(RuntimeError):
            evaluation.check_source_result(payload)

    def test_results_cannot_overwrite_previous_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            evaluation.write_json(path, {"passed": False})
            with self.assertRaises(FileExistsError):
                evaluation.write_json(path, {"passed": True})
            self.assertEqual(json.loads(path.read_text()), {"passed": False})

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dep = root / "fixture.py"
            dep.write_text("original\n")
            manifest = {"release_files": {}, "source_dependencies": {"fixture.py": evaluation.digest(dep)}}
            (root / "COMPOSITIONAL_AUTOBIOGRAPHICAL_GOALS_MANIFEST_v1.json").write_text(json.dumps(manifest))
            dep.write_text("altered\n")
            with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
                evaluation.preflight(root)

    def test_empty_or_partial_samples_cannot_pass(self):
        _, checks = evaluation.summarize([])
        self.assertFalse(all(checks.values()))
        self.assertFalse(checks["all_128_dev_subjects_accounted_for"])
        self.assertFalse(checks["eight_checkpoint_continuations_accounted_for"])

    def test_no_goal_subjects_remain_in_denominator(self):
        rows = [{"subject_id": 130000, "goal_generated": True, "both_clauses_complete": True},
                {"subject_id": 130001, "goal_generated": False}]
        metrics, _ = evaluation.summarize(rows)
        self.assertEqual(metrics["goal_generation_rate"], .5)
        self.assertEqual(metrics["both_clauses_complete_rate"], .5)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"COMPOSITIONAL_AUTOBIOGRAPHICAL_GOALS_SOFTWARE_TESTS: PASS ({result.testsRun} tests)")
    print("Scientific DEV/TEST not executed by these tests.")
