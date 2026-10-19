#!/usr/bin/env python3
"""v0.37.1: audit preserved DEV records and replay the interruption boundary.

No admission test, tuning, training, target-development rerun, or TEST access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

CONTEXTS = ("context_0", "context_1", "context_2")
MODES = ("drop_second", "misbind_second")
DEV_IDS = list(range(130000, 130128))
DEFAULT_SOURCE = "state/compositional_autobiographical_goals_v1/runs/20260920T055442Z_8b224fc43c"
MANIFEST = "COMPOSITIONAL_GOAL_COMPLETION_DIAGNOSTIC_MANIFEST_v1.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value(table, clause):
    v = float(table[clause["context"]][clause["primary"] - 1])
    require(math.isfinite(v) and 0 <= v <= 1, "invalid reliability/capability value")
    return v


def met(table, clause):
    return value(table, clause) >= clause["target_reliability"]


def audit_row(row):
    """Independent predicate/capability/trace audit; uses no model or simulator."""
    sid = row["subject_id"]
    require(sid in DEV_IDS, "source contains a non-DEV subject")
    goal = row["goal"]
    if goal is None:
        require(row["goal_generated"] is False, "inconsistent no-goal row")
        return {"subject_id": sid, "goal_generated": False}
    require(row["goal_generated"] is True, "goal missing generation flag")
    clauses = goal["clauses"]
    require(goal["operator"] == "THEN" and len(clauses) == 2, "wrong goal grammar")
    require(len({c["context"] for c in clauses}) == 2, "clauses share a context")
    for c in clauses:
        require(c["context"] in CONTEXTS and c["primary"] in (1, 2), "invalid clause binding")
        baseline = value(row["bootstrap_reliability"], c)
        require(c["baseline_reliability"] == baseline, "goal baseline differs from saved bootstrap")
        require(c["target_reliability"] == min(1.0, baseline + .12), "goal target rule changed")
    require(row["progress"]["goal"] == goal, "goal changed during execution")
    trace = row["trace"]
    counts = [sum(t["clause_index"] == i for t in trace) for i in range(2)]
    require(counts == row["clause_attempts"] == row["progress"]["attempts"], "episode counts disagree")
    require(all(0 <= n <= 16 for n in counts), "episode budget exceeded")
    stages = [t["clause_index"] for t in trace]
    require(stages == sorted(stages) and all(i in (0, 1) for i in stages), "THEN ordering violated")
    for t in trace:
        c = clauses[t["clause_index"]]
        require((t["context"], t["primary"]) == (c["context"], c["primary"]), "normal trace misbound")
    final_met = [met(row["final_reliability"], c) for c in clauses]
    gains = [value(row["final_capabilities"], c) - value(row["initial_capabilities"], c) for c in clauses]
    require(gains == row["clause_capability_gains"], "saved capability gains disagree")
    require(all(final_met) == row["both_final_targets_met"], "normal predicate flag disagrees")
    require(all(g >= .35 for g in gains) == row["both_capability_gains_at_least_0_35"], "capability gate flag disagrees")
    completed = row["progress"]["cursor"] == 2 and not row["progress"]["exhausted"]
    require(completed == row["both_clauses_complete"], "lifecycle completion flag disagrees")
    first, second = clauses
    second_pairs = []
    control = {}
    spare = next(c for c in CONTEXTS if c not in {first["context"], second["context"]})
    for mode in MODES:
        control_trace = row[mode + "_trace"]
        require(len(control_trace) == 32, f"{mode}: incomplete control trace")
        for step, t in enumerate(control_trace):
            expected = (spare, second["primary"]) if mode == "misbind_second" and step >= 16 else (first["context"], first["primary"])
            require((t["context"], t["primary"]) == expected, f"{mode}: control policy changed")
            require(t["context"] != second["context"], f"{mode}: missing context was visited")
        rel = row[mode + "_final_reliability"]
        caps = row[mode + "_final_capabilities"]
        predicates = [met(rel, c) for c in clauses]
        unchanged = value(caps, second) == value(row["initial_capabilities"], second)
        require(all(predicates) == row[mode + "_both_targets_met"], f"{mode}: control predicate flag disagrees")
        require(unchanged and row[mode + "_missing_target_unchanged"] is True, f"{mode}: missing target capability changed")
        second_pairs.append(rel[second["context"]])
        control[mode] = {"both_targets_met": all(predicates), "first_target_met": predicates[0],
                         "second_target_met": predicates[1], "second_reliability": value(rel, second),
                         "missing_context_visited": False, "missing_target_capability_gain": 0.0}
    require(second_pairs[0] == second_pairs[1], "untouched second-context readouts differ between controls")
    return {
        "subject_id": sid, "goal_generated": True, "goal": goal,
        "normal_both_complete": completed, "normal_final_targets_met": final_met,
        "normal_both_targets_met": all(final_met), "normal_attempts": counts,
        "normal_capability_gains": gains,
        "normal_both_gain_threshold": all(g >= .35 for g in gains),
        "normal_predicate_and_gain_conjunction": all(final_met) and all(g >= .35 for g in gains),
        "normal_completion_without_both_practiced": completed and any(n == 0 for n in counts),
        "normal_completion_with_subthreshold_gain": completed and any(g < .35 for g in gains),
        "normal_exhausted_clause": row["progress"]["cursor"] if row["progress"]["exhausted"] else None,
        "second_target_met_in_untouched_control_context": control[MODES[0]]["second_target_met"],
        "second_control_context_reliability_pair": second_pairs[0], "controls": control,
    }


def check_source_report(report, rows, old):
    require(report.get("family") == "v0.37" and report.get("version") == "0.37.0", "wrong source version")
    require(report.get("status") == "COMPLETED" and report.get("development_gate_passed") is False,
            "requires preserved completed v0.37.0 DEV FAIL")
    require(report.get("training_performed") is False and report.get("protected_unchanged") is True,
            "source training/integrity contract failed")
    require(report.get("split") == {"dev": [130000, 130127], "test": [140000, 140127], "test_touched": False},
            "source split or sealed TEST flag differs")
    require([r["subject_id"] for r in rows] == DEV_IDS, "source rows must contain all 128 DEV IDs in order")
    require(report["thresholds"] == {k: list(v) for k, v in old.THRESHOLDS.items()}, "source thresholds differ")
    metrics, checks = old.summarize(rows)
    require(metrics == report["metrics"] and checks == report["checks"], "source rows do not reproduce saved report")
    require(not all(checks.values()), "source FAIL cannot be rewritten")
    require(report["counts"] == {
        "subjects": 128, "generated": sum(r["goal_generated"] for r in rows),
        "both_complete": sum(r["both_clauses_complete"] for r in rows), "continuations": 8,
    }, "source report counts disagree")


def probe_windows(records, interruption_record):
    """Four history interventions. The 13-record condition is diagnostic only."""
    require(len(records) == 12, "expected exactly 12 original bootstrap records")
    return {
        "bootstrap_12": (list(records), 12),
        "drop_oldest_only_11": (list(records[1:]), 12),
        "retain_all_plus_interruption_13": (list(records) + [interruption_record], 13),
        "actual_rolling_window_12": (list(records[1:]) + [interruption_record], 12),
    }


def attribution_evidence(audit, row, bootstrap_rel, post_rel, cap_before, cap_after, goal_payload):
    """Score replay evidence without treating a counterfactual as an admission."""
    matches = {
        "bootstrap_exact": bootstrap_rel == row["bootstrap_reliability"],
        "authored_goal_exact": goal_payload == row["goal"],
        "post_interrupt_capabilities_exact": cap_after == row["initial_capabilities"],
    }
    clauses = []
    for index, c in enumerate(row["goal"]["clauses"]):
        before, after = value(bootstrap_rel, c), value(post_rel, c)
        gain = value(cap_after, c) - value(cap_before, c)
        crossing = before < c["target_reliability"] <= after
        clauses.append({"clause_index": index, "context": c["context"], "primary": c["primary"],
                        "target": c["target_reliability"], "bootstrap_reliability": before,
                        "after_interruption_reliability": after, "interruption_reliability_change": after - before,
                        "target_met_after_interruption": after >= c["target_reliability"],
                        "interruption_crosses_target": crossing, "target_capability_change_during_interruption": gain})
    second = row["goal"]["clauses"][1]
    matches["untouched_control_context_exact"] = post_rel[second["context"]] == audit["second_control_context_reliability_pair"]
    second_crosses_without_gain = clauses[1]["interruption_crosses_target"] and clauses[1]["target_capability_change_during_interruption"] == 0.0
    explained = {}
    for mode in MODES:
        saved = audit["controls"][mode]
        predicted_both = saved["first_target_met"] and met(post_rel, second)
        matches[mode + "_joint_predicate_exact"] = predicted_both == saved["both_targets_met"]
        explained[mode] = bool(saved["both_targets_met"] and second_crosses_without_gain and all(matches.values()))
    return {"matches": matches, "clauses": clauses, "control_success_explained_by_interruption": explained}


def replay_boundary(root, old, core, models, row, audit):
    """Only original bootstrap + original interruption. No target work is run."""
    trajectory, autobiographical = models
    backend = old.FrozenBackend(root, trajectory, autobiographical, row["subject_id"])
    bootstrap_rel = backend.reliabilities()
    goal = core.author_goal(bootstrap_rel)
    if row["goal"] is None:
        return {"matches": {"bootstrap_exact": bootstrap_rel == row["bootstrap_reliability"],
                             "authored_goal_exact": core.goal_payload(goal) is None}, "clauses": []}
    cap_before = backend.capabilities()
    originals = {c: list(backend.memories[c].records) for c in CONTEXTS}
    backend.interrupt()
    post_rel = backend.reliabilities()
    cap_after = backend.capabilities()
    result = attribution_evidence(audit, row, bootstrap_rel, post_rel, cap_before, cap_after, core.goal_payload(goal))
    for clause_detail, c in zip(result["clauses"], row["goal"]["clauses"]):
        context = c["context"]
        original = originals[context]
        appended = backend.memories[context].records[-1]
        windows = probe_windows(original, appended)
        estimates = {}
        counts = {}
        for name, (records, width) in windows.items():
            memory = backend.memory_module.AutobiographicalTrajectoryMemory(records)
            x, mask = memory.feature_tensor(width)
            estimate = autobiographical(x.unsqueeze(0), mask.unsqueeze(0))["reliability"][0].detach().cpu().tolist()
            estimates[name] = estimate
            counts[name] = {"valid_records": len(records), "target_records": sum(r.primary == c["primary"] for r in records)}
        clause_detail["history_probe_reliabilities"] = estimates
        clause_detail["history_probe_counts"] = counts
        clause_detail["evicted_record_primary"] = int(original[0].primary)
        clause_detail["interruption_record_primary"] = int(appended.primary)
        clause_detail["interruption_was_target_episode"] = int(appended.primary) == c["primary"]
        idx = c["primary"] - 1
        clause_detail["history_probe_target_met"] = {name: pair[idx] >= c["target_reliability"] for name, pair in estimates.items()}
        result["matches"][f"clause_{clause_detail['clause_index']}_bootstrap_probe_exact"] = estimates["bootstrap_12"] == bootstrap_rel[context]
        result["matches"][f"clause_{clause_detail['clause_index']}_rolling_probe_exact"] = estimates["actual_rolling_window_12"] == post_rel[context]
    # Explanation requires ALL reproduction checks and no target episode.
    exact = all(result["matches"].values())
    for mode in MODES:
        result["control_success_explained_by_interruption"][mode] = bool(
            result["control_success_explained_by_interruption"][mode] and exact
            and not result["clauses"][1]["interruption_was_target_episode"])
    return result


def summary(audits, replays):
    generated = [a for a in audits if a["goal_generated"]]
    counts = {"subjects": len(audits), "goals_generated": len(generated)}
    for field in ("normal_both_complete", "normal_both_targets_met", "normal_both_gain_threshold",
                  "normal_predicate_and_gain_conjunction", "normal_completion_without_both_practiced",
                  "normal_completion_with_subthreshold_gain", "second_target_met_in_untouched_control_context"):
        counts[field] = sum(a[field] for a in generated)
    counts["normal_exhausted_clause_1"] = sum(a["normal_exhausted_clause"] == 0 for a in generated)
    counts["normal_exhausted_clause_2"] = sum(a["normal_exhausted_clause"] == 1 for a in generated)
    for mode in MODES:
        counts[mode + "_both_targets_met"] = sum(a["controls"][mode]["both_targets_met"] for a in generated)
    if replays is None:
        return {"record_audit_counts": counts, "replay_performed": False,
                "attribution_status": "SOURCE_PATH_INFERENCE_ONLY"}
    mismatches = [r["subject_id"] for r in replays if not all(r["replay"]["matches"].values())]
    probes = [c for r in replays for c in r["replay"]["clauses"]]
    replay_counts = {"subjects_replayed": len(replays), "subjects_with_any_reproduction_mismatch": len(mismatches),
                     "clause_probes": len(probes)}
    for index in range(2):
        subset = [c for c in probes if c["clause_index"] == index]
        replay_counts[f"clause_{index + 1}_interruption_crossings"] = sum(c["interruption_crosses_target"] for c in subset)
        replay_counts[f"clause_{index + 1}_crossings_without_target_episode_or_gain"] = sum(
            c["interruption_crosses_target"] and not c["interruption_was_target_episode"]
            and c["target_capability_change_during_interruption"] == 0.0 for c in subset)
    probe_crossings = Counter()
    for c in probes:
        for name, crosses in c["history_probe_target_met"].items():
            probe_crossings[name] += int(crosses)
    for mode in MODES:
        replay_counts[mode + "_successful_controls_explained"] = sum(
            r["replay"].get("control_success_explained_by_interruption", {}).get(mode, False) for r in replays)
    exact = len(replays) == len(audits) and not mismatches
    all_explained = exact and all(replay_counts[mode + "_successful_controls_explained"] == counts[mode + "_both_targets_met"] for mode in MODES)
    any_control_success = any(counts[mode + "_both_targets_met"] for mode in MODES)
    if not exact:
        status = "REPLAY_MISMATCH_UNRESOLVED"
    elif all_explained and any_control_success:
        status = "ALL_OBSERVED_CONTROL_SUCCESSES_EXPLAINED_BY_INTERRUPTION"
    elif not any_control_success:
        status = "NO_SUCCESSFUL_CONTROLS_TO_ATTRIBUTE"
    else:
        status = "CONTROL_SUCCESSES_NOT_FULLY_EXPLAINED"
    return {"record_audit_counts": counts, "replay_performed": True, "replay_counts": replay_counts,
            "mismatched_subject_ids": mismatches, "history_probe_target_crossing_counts": dict(probe_crossings),
            "attribution_status": status,
            "history_probe_note": "11/13-record probes alter sequence length; descriptive interventions, not validated repairs."}


def load_inputs(root, source):
    manifest = strict_json((root / MANIFEST).read_text())
    for group in ("release_files", "pinned_v0370_files"):
        for rel, expected in manifest[group].items():
            require((root / rel).is_file() and digest(root / rel) == expected, f"source hash mismatch: {rel}")
    old = importlib.import_module("evaluate_compositional_autobiographical_goals_v1")
    core = importlib.import_module("continuous_brain.compositional_autobiographical_goals_v1")
    current = old.preflight(root)
    require(source.is_dir(), f"source run not found: {source}")
    report = strict_json((source / "result.json").read_text())
    rows = [strict_json(line) for line in (source / "subjects.jsonl").read_text().splitlines() if line.strip()]
    check_source_report(report, rows, old)
    original_before = strict_json((source / "integrity_before.json").read_text())
    original_after = strict_json((source / "integrity_after.json").read_text())
    require(original_before == original_after == report["protected_hashes"] == current,
            "original/current protected hashes differ; do not replay changed models or sources")
    protected = {str(root / rel): sha for rel, sha in current.items()}
    for group in ("release_files", "pinned_v0370_files"):
        for rel in manifest[group]:
            protected[str(root / rel)] = digest(root / rel)
    protected[str(root / MANIFEST)] = digest(root / MANIFEST)
    for path in source.rglob("*"):
        if path.is_file():
            protected[str(path)] = digest(path)
    return old, core, report, rows, protected


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE)
    parser.add_argument("--records-only", action="store_true", help="audit recorded data without neural replay; attribution stays unconfirmed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = (root / args.source_run).resolve()
    try:
        old, core, source_result, rows, protected = load_inputs(root, source)
        audits = [audit_row(row) for row in rows]
        models = None
        if not args.records_only:
            import torch
            torch.set_num_threads(1)
            torch.manual_seed(37001)
            torch.use_deterministic_algorithms(True)
            from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
            from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import load_neutral_model
            trajectory, _ = load_trajectory_model(root / old.TRAJECTORY_MODEL, "cpu")
            autobiographical, _ = load_neutral_model(root / old.AUTOBIO_MODEL, "cpu")
            for model in (trajectory, autobiographical):
                model.eval()
                model.requires_grad_(False)
            models = (trajectory, autobiographical)
    except Exception as exc:
        print(f"COMPOSITIONAL_COMPLETION_DIAGNOSTIC_PREFLIGHT: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "state/compositional_goal_completion_diagnostic_v1/runs" / (stamp + "_" + uuid.uuid4().hex[:10])
    require(source != out and source not in out.parents, "output must not be inside the source run")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "integrity_before.json", protected)
    write_json(out / "source_record_audit.json", audits)
    result = {"family": "v0.37", "version": "0.37.1", "kind": "DIAGNOSTIC",
              "status": "INFRASTRUCTURE_ERROR", "historical_v0370_status": "DEV_FAIL",
              "development_gate_passed": None, "admission_performed": False, "training_performed": False,
              "thresholds_changed": False, "test_touched": False, "source_run": str(source),
              "source_result_sha256": digest(source / "result.json"), "target_development_episodes_replayed": 0,
              "original_failed_checks": [k for k, v in source_result["checks"].items() if not v]}
    original_stdout = sys.stdout
    code = 3
    with (out / "run.log").open("x", encoding="utf-8") as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            print("=== v0.37.1 Compositional Goal Completion Attribution Diagnostic ===")
            print("SOURCE_V0370_STATUS: DEV_FAIL_PRESERVED")
            print("READ ONLY inputs. CPU inference only. No training or admission.")
            print("Spent DEV bootstrap/interruption replay only. TEST remains sealed.")
            replays = None
            if models is not None:
                replays = []
                with torch.no_grad(), (out / "boundary_replay.jsonl").open("x", encoding="utf-8") as stream:
                    for i, (row, audit) in enumerate(zip(rows, audits), 1):
                        replay = replay_boundary(root, old, core, models, row, audit)
                        record = {"subject_id": row["subject_id"], "replay": replay}
                        replays.append(record)
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        if i == 1 or i % 16 == 0:
                            print(f"interruption_boundary_subjects={i}/128")
            result["findings"] = summary(audits, replays)
            result["status"] = "DIAGNOSTIC_COMPLETE"
            code = 0
            for key, val in result["findings"].items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        print(f"{key}.{subkey}: {subval}")
                else:
                    print(f"{key}: {val}")
        except (Exception, KeyboardInterrupt) as exc:
            traceback.print_exc(file=sys.stdout)
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                after = {p: digest(p) if Path(p).is_file() else None for p in protected}
                unchanged = after == protected
                write_json(out / "integrity_after.json", after)
                result["protected_unchanged"] = unchanged
                if not unchanged:
                    result["status"] = "INTEGRITY_ERROR"
                    result["changed_paths"] = [p for p in protected if protected[p] != after[p]]
                    code = 3
                write_json(out / "result.json", result)
                print("PROTECTED_UNCHANGED:", str(unchanged).lower())
                print("V037_TEST_TOUCHED: false")
                print("ADMISSION_PERFORMED: false")
                print("SOURCE_V0370_STATUS: DEV_FAIL_PRESERVED")
                print("COMPOSITIONAL_COMPLETION_DIAGNOSTIC:", result["status"])
            finally:
                sys.stdout = original_stdout
    bundle = out / "result_bundle.zip"
    with ZipFile(bundle, "x", ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != bundle:
                archive.write(path, path.name)
    print(f"result: {out / 'result.json'}")
    print(f"bundle: {bundle}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
