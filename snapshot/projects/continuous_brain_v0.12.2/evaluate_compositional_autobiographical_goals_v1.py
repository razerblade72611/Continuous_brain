#!/usr/bin/env python3
"""v0.37.0 DEV only. Frozen models, append-only runs, no TEST execution path."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import traceback
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

from continuous_brain.compositional_autobiographical_goals_v1 import (
    CONTEXTS, CompositeGoal, CompositeProgress, author_goal, goal_payload,
    run_composite, run_control, targets_met,
)

DEV_IDS = range(130000, 130128)
SEALED_TEST_IDS = range(140000, 140128)
PERSISTENCE_SUBJECTS = 8
SOURCE_RESULT = "state/recursive_autobiographical_curriculum_test_v1/result.json"
AUTOBIO_MODEL = "state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt"
TRAJECTORY_MODEL = "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
PROTECTED = (
    SOURCE_RESULT, AUTOBIO_MODEL, TRAJECTORY_MODEL,
    "state/endogenous_goal_genesis_test_v1/result.json",
    "state/recursive_autobiographical_curriculum_v1/result.json",
    "state/child_v113_dml_smoke/infant_training.pt",
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt",
    "state/multi_item_episodic_memory_v1/episodic_reader.pt",
    "state/persistent_self_autobiographical_v1/agency_self_model.pt",
    "state/persistent_self_autobiographical_v1/persistent_self_seed.pt",
)
THRESHOLDS = {
    "goal_generation_rate": (">=", .95),
    "selected_primitives_match_context_fragility_rate": (">=", .95),
    "context_history_transplant_equivariance_rate": (">=", .95),
    "context_history_transplant_changes_goal_rate": (">=", .95),
    "erased_history_no_goal_rate": ("==", 1.0),
    "zero_discrepancy_no_goal_rate": ("==", 1.0),
    "both_clauses_complete_rate": (">=", .70),
    "both_capability_gains_at_least_0_35_rate": (">=", .70),
    "both_final_targets_met_rate": (">=", .70),
    "plan_binding_correct_given_goal_rate": ("==", 1.0),
    "untargeted_capabilities_unchanged_given_goal_rate": ("==", 1.0),
    "drop_second_both_targets_met_rate": ("==", 0.0),
    "misbind_second_both_targets_met_rate": ("==", 0.0),
    "drop_second_missing_target_unchanged_given_goal_rate": ("==", 1.0),
    "misbind_second_missing_target_unchanged_given_goal_rate": ("==", 1.0),
    "completion_advantage_over_drop_second": (">=", .70),
    "completion_advantage_over_misbind_second": (">=", .70),
    "frozen_history_repeats_goal_given_goal_rate": ("==", 1.0),
    "checkpoint_continuation_exact_rate": ("==", 1.0),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_json(path, value):
    # A new run directory owns these files; never overwrite a previous result.
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(dumps(value))


def context_trace_id(subject_id, context):
    if subject_id not in DEV_IDS:
        raise ValueError("v0.37.0 permits DEV subjects 130000..130127 only")
    c = CONTEXTS.index(context)
    return 130000 + 1000 * c + ((subject_id - 130000) ^ (c % 2))


def check_source_result(result):
    if (result.get("family") != "v0.36"
        or result.get("test_admission_passed") is not True
        or result.get("split", {}).get("test") != [120000, 120127]
        or result.get("split", {}).get("test_spent") is not True
        or result.get("v036_test_spent") is not True
        or result.get("protected_unchanged") is not True
        or result.get("source_models_unchanged") is not True
        or not result.get("checks")
        or any(v is not True for v in result["checks"].values())):
        raise RuntimeError("an intact, closed v0.36.1 TEST PASS is required")


def preflight(root):
    manifest_path = root / "COMPOSITIONAL_AUTOBIOGRAPHICAL_GOALS_MANIFEST_v1.json"
    manifest = json.loads(manifest_path.read_text())
    checked = {}
    for section in ("release_files", "source_dependencies"):
        for rel, expected in manifest[section].items():
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(f"required {section} file missing: {rel}")
            actual = digest(path)
            if actual != expected:
                raise RuntimeError(f"source hash mismatch: {rel}; preserve it and report this error")
            checked[rel] = actual
    for rel in PROTECTED:
        if not (root / rel).is_file():
            raise FileNotFoundError(f"required protected file missing: {rel}")
        checked[rel] = digest(root / rel)
    # Include optional package initialization because Python executes it on import.
    init = root / "continuous_brain/__init__.py"
    if init.is_file():
        checked[str(init.relative_to(root))] = digest(init)
    checked[str(manifest_path.relative_to(root))] = digest(manifest_path)
    check_source_result(json.loads((root / SOURCE_RESULT).read_text()))
    return checked


class FrozenBackend:
    """Three evaluator contexts; one shared pair of frozen neural models.

    Each context keeps independent autobiography and simulated capability.
    These are repetitions of the existing A/B substrate, not new motor skills.
    """
    def __init__(self, root, trajectory, autobiographical, subject_id):
        self.subject_id = int(subject_id)
        self.trajectory = trajectory
        self.autobiographical = autobiographical
        self.legacy = importlib.import_module("evaluate_self_development_loop_v1")
        self.memory_module = importlib.import_module("continuous_brain.autobiographical_self_discrepancy_v1")
        self.trace_ids = {c: context_trace_id(subject_id, c) for c in CONTEXTS}
        self.memories = {}
        self.states = {}
        for c, sid in self.trace_ids.items():
            self.memories[c] = self.legacy.build_bootstrap_memory(trajectory, sid)
            a, b = self.legacy.initial_capabilities(sid)
            self.states[c] = self.legacy.DevelopmentalState(a, b, 0)

    def clone(self):
        other = copy.copy(self)
        other.memories = copy.deepcopy(self.memories)
        other.states = copy.deepcopy(self.states)
        return other

    def reliabilities(self):
        return {c: self.legacy.reliability_from_memory(self.autobiographical, self.memories[c]).tolist()
                for c in CONTEXTS}

    def capabilities(self):
        return {c: [self.states[c].capability_for(p) for p in (1, 2)] for c in CONTEXTS}

    def interrupt(self):
        for c in CONTEXTS:
            sid = self.trace_ids[c]
            primary = 2 if sid % 2 == 0 else 1
            record, _ = self.legacy.execute_development_episode(
                self.trajectory, sid, -1, primary, self.legacy.STRATEGY_DIRECT, self.states[c])
            self.memories[c].append(record)

    def execute(self, context, primary, global_step):
        if context not in CONTEXTS or primary not in (1, 2):
            raise ValueError("invalid execution command")
        state = self.states[context]
        # Common work-outcome uniforms by subject and global episode across arms.
        # World initialization remains context-specific; this is not a claim of
        # identical world trajectories under the deliberately misbound control.
        rng = random.Random(3700001 + self.subject_id * 100003 + global_step * 1009)
        record, metrics = self.legacy.execute_development_episode(
            self.trajectory, self.trace_ids[context], state.cycle, primary,
            self.legacy.STRATEGY_SCAFFOLD, state,
            forced_uniforms=[rng.random(), rng.random()])
        state.cycle += 1
        self.memories[context].append(record)
        return {k: float(v) for k, v in metrics.items()}

    def snapshot(self):
        memories = {}
        for c in CONTEXTS:
            rows = []
            for r in self.memories[c].records:
                rows.append({k: v.tolist() if hasattr(v, "tolist") else v for k, v in vars(r).items()})
            memories[c] = rows
        return {"format": "v037_context_backend_v1", "subject_id": self.subject_id,
                "trace_ids": self.trace_ids, "states": {c: asdict(s) for c, s in self.states.items()},
                "memories": memories}

    def restore(self, raw):
        import torch
        if (raw.get("format") != "v037_context_backend_v1" or raw["subject_id"] != self.subject_id
            or raw["trace_ids"] != self.trace_ids):
            raise ValueError("checkpoint belongs to a different subject or format")
        other = copy.copy(self)
        other.states = {c: self.legacy.DevelopmentalState(**raw["states"][c]) for c in CONTEXTS}
        other.memories = {}
        for c in CONTEXTS:
            records = []
            for row in raw["memories"][c]:
                values = dict(row)
                for key in ("expected_self", "actual_self", "expected_world", "actual_world", "expected_modes", "actual_modes"):
                    values[key] = torch.tensor(values[key], dtype=torch.long if key.endswith("modes") else torch.float32)
                records.append(self.memory_module.TrajectoryMemoryRecord(**values))
            other.memories[c] = self.memory_module.AutobiographicalTrajectoryMemory(records)
        return other


def permutation_controls(backend, goal):
    rotation = {c: CONTEXTS[(i + 1) % 3] for i, c in enumerate(CONTEXTS)}
    donor = backend.clone()
    donor.memories = {rotation[c]: copy.deepcopy(m) for c, m in backend.memories.items()}
    donor_goal = author_goal(donor.reliabilities())
    expected = None
    if goal is not None:
        expected = goal.payload()
        for clause in expected["clauses"]:
            clause["context"] = rotation[clause["context"]]
    erased = backend.clone()
    erased.memories = {c: backend.memory_module.AutobiographicalTrajectoryMemory() for c in CONTEXTS}
    zero = backend.clone()
    zero.memories = {c: m.discrepancy_ablation() for c, m in backend.memories.items()}
    return {
        "context_history_transplant_equivariant": goal is not None and goal_payload(donor_goal) == expected,
        "context_history_transplant_changes_goal": goal is not None and goal_payload(donor_goal) != goal.payload(),
        "erased_history_no_goal": author_goal(erased.reliabilities()) is None,
        "zero_discrepancy_no_goal": author_goal(zero.reliabilities()) is None,
        "donor_goal": goal_payload(donor_goal),
    }


def run_subject(backend, run_dir, persistence=False):
    initial_rel = backend.reliabilities()
    goal = author_goal(initial_rel)
    row = {"subject_id": backend.subject_id, "bootstrap_reliability": initial_rel,
           "goal": goal_payload(goal), "goal_generated": goal is not None,
           "both_clauses_complete": False, "both_final_targets_met": False,
           "both_capability_gains_at_least_0_35": False,
           "selected_primitives_match_context_fragility": False,
           "drop_second_both_targets_met": False, "misbind_second_both_targets_met": False}
    row.update(permutation_controls(backend, goal))
    if goal is None:
        if persistence:
            row["checkpoint_continuation_exact"] = False
        return row
    original_semantics = goal.payload()
    row["selected_primitives_match_context_fragility"] = all(
        c.primary == (1 if backend.trace_ids[c.context] % 2 == 0 else 2) for c in goal.clauses)
    backend.interrupt()
    # Goals are immutable and not regenerated by an unrelated episode.
    row["goal_persists_through_interruption"] = original_semantics == goal.payload()
    initial_caps = backend.capabilities()
    control_start = backend.clone()
    progress = CompositeProgress(goal)
    checkpoint = None
    if persistence:
        trace = run_composite(backend, progress, pause_after=1)
        checkpoint = {"progress": progress.payload(), "backend": backend.snapshot(), "trace": trace}
        snap_path = run_dir / f"checkpoint_{backend.subject_id}.json"
        write_json(snap_path, checkpoint)
        # Parse bytes actually written, not the in-memory payload.
        checkpoint = json.loads(snap_path.read_text())
        trace = run_composite(backend, progress, trace)
        restored_backend = backend.restore(checkpoint["backend"])
        restored_progress = CompositeProgress.restore(checkpoint["progress"])
        restored_trace = run_composite(restored_backend, restored_progress, checkpoint["trace"])
        row["checkpoint_continuation_exact"] = (
            progress.payload() == restored_progress.payload()
            and trace == restored_trace and backend.snapshot() == restored_backend.snapshot())
    else:
        trace = run_composite(backend, progress)
    final_rel = backend.reliabilities()
    final_caps = backend.capabilities()
    gains = [final_caps[c.context][c.primary - 1] - initial_caps[c.context][c.primary - 1] for c in goal.clauses]
    selected = {(c.context, c.primary) for c in goal.clauses}
    row.update({
        "both_clauses_complete": progress.complete,
        "both_final_targets_met": all(targets_met(goal, final_rel)),
        "both_capability_gains_at_least_0_35": all(g >= .35 for g in gains),
        "clause_capability_gains": gains, "clause_attempts": progress.attempts,
        "final_reliability": final_rel, "initial_capabilities": initial_caps, "final_capabilities": final_caps,
        "plan_binding_correct": bool(trace) and all(
            (t["context"], t["primary"]) == (goal.clauses[t["clause_index"]].context, goal.clauses[t["clause_index"]].primary)
            for t in trace),
        "untargeted_capabilities_unchanged": all(
            final_caps[c][p - 1] == initial_caps[c][p - 1] for c in CONTEXTS for p in (1, 2) if (c, p) not in selected),
        "frozen_history_repeats_goal": goal_payload(author_goal(initial_rel)) == goal.payload(),
        "updated_history_next_goal_descriptive_only": goal_payload(author_goal(final_rel)),
        "trace": trace, "progress": progress.payload(),
    })
    for mode in ("drop_second", "misbind_second"):
        control = control_start.clone()
        control_trace = run_control(control, goal, mode)
        caps = control.capabilities()
        second = goal.clauses[1]
        row[mode + "_both_targets_met"] = all(targets_met(goal, control.reliabilities()))
        row[mode + "_missing_target_unchanged"] = caps[second.context][second.primary - 1] == initial_caps[second.context][second.primary - 1]
        row[mode + "_final_reliability"] = control.reliabilities()
        row[mode + "_final_capabilities"] = caps
        row[mode + "_trace"] = control_trace
    return row


def mean(values):
    values = list(values)
    return sum(float(v) for v in values) / len(values) if values else 0.0


def summarize(rows):
    # No failed/no-goal subject is dropped from scientific success denominators.
    metrics = {}
    for field in ("goal_generated", "selected_primitives_match_context_fragility",
                  "context_history_transplant_equivariant", "context_history_transplant_changes_goal",
                  "erased_history_no_goal", "zero_discrepancy_no_goal", "both_clauses_complete",
                  "both_capability_gains_at_least_0_35", "both_final_targets_met",
                  "drop_second_both_targets_met", "misbind_second_both_targets_met"):
        name = {"goal_generated": "goal_generation", "context_history_transplant_equivariant": "context_history_transplant_equivariance"}.get(field, field)
        metrics[name + "_rate"] = mean(bool(r.get(field, False)) for r in rows)
    generated = [r for r in rows if r["goal_generated"]]
    for field in ("plan_binding_correct", "untargeted_capabilities_unchanged", "drop_second_missing_target_unchanged",
                  "misbind_second_missing_target_unchanged", "frozen_history_repeats_goal"):
        metrics[field + "_given_goal_rate"] = mean(bool(r.get(field, False)) for r in generated)
    persistence_rows = [r for r in rows if "checkpoint_continuation_exact" in r]
    metrics["checkpoint_continuation_exact_rate"] = mean(r["checkpoint_continuation_exact"] for r in persistence_rows)
    for mode in ("drop_second", "misbind_second"):
        metrics["completion_advantage_over_" + mode] = metrics["both_final_targets_met_rate"] - metrics[mode + "_both_targets_met_rate"]
    checks = {}
    for name, (operator, limit) in THRESHOLDS.items():
        value = metrics[name]
        checks[name] = math.isfinite(value) and (value >= limit if operator == ">=" else value == limit)
    checks["all_128_dev_subjects_accounted_for"] = [r["subject_id"] for r in rows] == list(DEV_IDS)
    checks["eight_checkpoint_continuations_accounted_for"] = len(persistence_rows) == PERSISTENCE_SUBJECTS
    checks["persistent_goals_survive_interruption"] = bool(generated) and all(r.get("goal_persists_through_interruption", False) for r in generated)
    return metrics, checks


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    # Old evaluators resolve checkpoint paths relative to the project root.
    import os
    os.chdir(root)
    try:
        before = preflight(root)
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(37001)
        torch.use_deterministic_algorithms(True)
        from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
        from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import load_neutral_model
        trajectory, _ = load_trajectory_model(root / TRAJECTORY_MODEL, "cpu")
        autobiographical, _ = load_neutral_model(root / AUTOBIO_MODEL, "cpu")
        for model in (trajectory, autobiographical):
            model.eval()
            model.requires_grad_(False)
        # Import all legacy simulator dependencies before reserving a run.
        importlib.import_module("evaluate_self_development_loop_v1")
    except Exception as exc:
        print(f"COMPOSITIONAL_GOALS_PREFLIGHT: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if args.preflight_only:
        print("COMPOSITIONAL_GOALS_PREFLIGHT: PASS (no subjects executed)")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "state/compositional_autobiographical_goals_v1/runs" / (stamp + "_" + uuid.uuid4().hex[:10])
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "integrity_before.json", before)
    write_json(run_dir / "frozen_run_config.json", {
        "family": "v0.37", "version": "0.37.0", "thresholds": THRESHOLDS,
        "dev_subjects": [130000, 130127], "sealed_test_subjects": [140000, 140127],
        "dev_context_trace_ranges": [[130000, 130127], [131000, 131127], [132000, 132127]],
        "sealed_test_context_trace_ranges": [[140000, 140127], [141000, 141127], [142000, 142127]],
        "target_delta": .12, "per_clause_horizon": 16, "control_budget": 32,
        "continuation_subjects": [130000, 130007], "training_performed": False,
    })
    result = {"family": "v0.37", "version": "0.37.0", "status": "INFRASTRUCTURE_ERROR",
              "development_gate_passed": None, "training_performed": False,
              "split": {"test_touched": False}, "error": "run did not finish"}
    code = 3
    original_stdout = sys.stdout
    with (run_dir / "run.log").open("x", encoding="utf-8") as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            print("COMPOSITIONAL_AUTOBIOGRAPHICAL_GOALS_V1 | v0.37.0 DEV")
            print("Frozen CPU models; no neural training. Three contexts; two ordered clauses.")
            print("DEV 130000-130127; TEST 140000-140127 remains sealed.")
            print("Context capability is a simulated environment variable, not a trained neural skill.")
            rows = []
            with torch.no_grad(), (run_dir / "subjects.jsonl").open("x", encoding="utf-8") as rows_file:
                for i, sid in enumerate(DEV_IDS):
                    backend = FrozenBackend(root, trajectory, autobiographical, sid)
                    row = run_subject(backend, run_dir, persistence=i < PERSISTENCE_SUBJECTS)
                    rows.append(row)
                    rows_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                    rows_file.flush()
                    if i == 0 or (i + 1) % 16 == 0:
                        print(f"compositional_goal_subjects={i + 1}/128")
            metrics, checks = summarize(rows)
            result = {"family": "v0.37", "version": "0.37.0", "status": "COMPLETED",
                      "training_performed": False, "thresholds": THRESHOLDS, "metrics": metrics, "checks": checks,
                      "counts": {"subjects": len(rows), "generated": sum(r["goal_generated"] for r in rows),
                                 "both_complete": sum(r["both_clauses_complete"] for r in rows), "continuations": PERSISTENCE_SUBJECTS},
                      "split": {"dev": [130000, 130127], "test": [140000, 140127], "test_touched": False},
                      "source_v0361_test_status": "PASS", "v036_test_rerun": False,
                      "scope": {"handwritten_composition_rule": True, "context_semantics_learned": False,
                                "new_neural_skills_trained": False, "synthetic_capability_dynamics": True,
                                "open_ended_goal_invention": False, "consciousness_evidence": False}}
        except (Exception, KeyboardInterrupt) as exc:
            traceback.print_exc(file=sys.stdout)
            result = {"family": "v0.37", "version": "0.37.0", "status": "INFRASTRUCTURE_ERROR",
                      "development_gate_passed": None, "error": f"{type(exc).__name__}: {exc}",
                      "training_performed": False, "split": {"test_touched": False}}
        finally:
            try:
                after = {p: digest(root / p) if (root / p).is_file() else None for p in before}
                unchanged = before == after
                write_json(run_dir / "integrity_after.json", after)
                result["protected_unchanged"] = unchanged
                result["protected_hashes"] = before
                if not unchanged:
                    result["status"] = "INTEGRITY_ERROR"
                    result["development_gate_passed"] = None
                    result["changed_paths"] = [p for p in before if before[p] != after[p]]
                if result["status"] == "COMPLETED":
                    passed = all(result["checks"].values())
                    result["development_gate_passed"] = passed
                    code = 0 if passed else 4
                    for key, value in result["metrics"].items():
                        print(f"{key}: {value}")
                    for key, value in result["checks"].items():
                        print(f"{key}: {'PASS' if value else 'FAIL'}")
                    print("COMPOSITIONAL_GOALS_DEV_GATE:", "PASS" if passed else "FAIL")
                else:
                    code = 3
                    print("COMPOSITIONAL_GOALS_RUN:", result["status"])
                write_json(run_dir / "result.json", result)
                print("TRAINING_PERFORMED: false")
                print("V037_TEST_TOUCHED: false")
                print("PROTECTED_UNCHANGED:", str(unchanged).lower())
                print("COMPOSITIONAL_GOALS_RUN_COMPLETE")
            finally:
                sys.stdout = original_stdout
    bundle = run_dir / "result_bundle.zip"
    with ZipFile(bundle, "x", ZIP_DEFLATED) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p != bundle:
                z.write(p, p.name)
    print(f"result: {run_dir / 'result.json'}")
    print(f"bundle: {bundle}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
