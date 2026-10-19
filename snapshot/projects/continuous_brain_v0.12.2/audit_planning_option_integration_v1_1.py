#!/usr/bin/env python3
"""Read stored checkpoint metadata and source interfaces without importing the model.

Python standard library only. No torch import, model constructor, training,
inference, environment rollout, or checkpoint write is performed. Checkpoint
pickle globals are restricted to inert tensor/storage descriptors. Unsupported
formats are reported; there is no unrestricted deserialization fallback.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import sys
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

VERSION = "0.19.1-audit-replacement-1"
FORMAT = "continuous_brain.planning_option_integration_audit.v1_1"
REQUIRED_CLASSES = {"PlanningCortex", "OptionCortex", "WorldModelEnsemble",
                    "CorticalNucleus", "InfantBrain"}
EXTRA_CLASSES = {"InfantConfig", "CapacityMatchedRecallWorkingMemoryInfant",
                 "OptionIntentExecutive", "RecurrentOptionProgressInfant"}
GROUPS = {"planning", "options", "world", "nucleus", "spatial",
          "target_spatial", "vision", "target_vision", "motor", "memory",
          "recall", "executive", "progress", "intent_gain"}
KNOWN_CHECKPOINTS = {
    "historical_base": ("state/child_v113_dml_smoke/infant_training.pt",
        "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"),
    "admitted_memory": ("state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt",
        "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"),
    "v018_intent_parent": ("state/option_intent_executive_v1/latest.portable.pt",
        "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2"),
    "closed_v019_progress": ("state/recurrent_option_progress_v1/latest.portable.pt", None),
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "datasets",
             "data", "assets", "site-packages", ".cache", ".tox", "build", "dist"}
MAX_PICKLE = 16 * 1024 * 1024
MAX_SOURCE = 2 * 1024 * 1024
MAX_REPORT = 8 * 1024 * 1024
MAX_FILES = 100000
MAX_CHECKPOINTS = 4096


@dataclass(frozen=True)
class StorageKind:
    name: str


@dataclass(frozen=True)
class StorageInfo:
    kind: str
    key: str
    location: str
    elements: int


@dataclass
class TensorInfo:
    storage: StorageInfo
    offset: int
    shape: tuple
    stride: tuple
    requires_grad: bool = False
    dtype: str | None = None


def rebuild_tensor(storage, offset, shape, stride, *extra):
    if not isinstance(storage, StorageInfo):
        raise pickle.UnpicklingError("unsupported tensor storage")
    shape, stride = tuple(shape), tuple(stride)
    if (len(shape) > 32 or len(shape) != len(stride)
            or any(type(x) is not int or x < 0 for x in shape)):
        raise pickle.UnpicklingError("invalid tensor dimensions")
    grad = bool(extra[0]) if extra else False
    dtype = next((x for x in extra if isinstance(x, str) and x.startswith("torch.")), None)
    return TensorInfo(storage, int(offset), shape, stride, grad, dtype)


def rebuild_parameter(data, requires_grad, *unused):
    if not isinstance(data, TensorInfo):
        raise pickle.UnpicklingError("unsupported parameter")
    data.requires_grad = bool(requires_grad)
    return data


class MetadataUnpickler(pickle.Unpickler):
    """Whitelist only inert PyTorch metadata constructors; never import globals."""
    STORAGE_NAMES = {"FloatStorage", "DoubleStorage", "HalfStorage", "BFloat16Storage",
                     "LongStorage", "IntStorage", "ShortStorage", "CharStorage",
                     "ByteStorage", "BoolStorage", "ComplexFloatStorage",
                     "ComplexDoubleStorage", "UntypedStorage"}
    DTYPES = {"float32", "float64", "float16", "bfloat16", "int64", "int32", "int16",
              "int8", "uint8", "bool", "complex64", "complex128"}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module in {"torch", "torch.storage"} and name in self.STORAGE_NAMES:
            return StorageKind(name)
        if module == "torch" and name in self.DTYPES:
            return "torch." + name
        if module == "torch._utils":
            if name in {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"}:
                return rebuild_tensor
            if name in {"_rebuild_parameter", "_rebuild_parameter_with_state"}:
                return rebuild_parameter
        if module in {"builtins", "__builtin__"} and name in {"set", "frozenset", "slice", "complex"}:
            return {"set": set, "frozenset": frozenset, "slice": slice, "complex": complex}[name]
        raise pickle.UnpicklingError(f"unsupported metadata global: {module}.{name}")

    def persistent_load(self, pid):
        if (not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage"
                or not isinstance(pid[1], StorageKind)):
            raise pickle.UnpicklingError("unsupported persistent object")
        return StorageInfo(pid[1].name, str(pid[2]), str(pid[3]), int(pid[4]))


def decode_metadata(data):
    if len(data) > MAX_PICKLE:
        raise ValueError("checkpoint metadata exceeds 16 MiB limit")
    result = MetadataUnpickler(io.BytesIO(data)).load()
    if not isinstance(result, dict):
        raise ValueError("checkpoint root is not a dictionary")
    return result


def read_checkpoint_metadata(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        candidates = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise ValueError("expected exactly one PyTorch data.pkl member")
        entry = archive.getinfo(candidates[0])
        if entry.file_size > MAX_PICKLE:
            raise ValueError("checkpoint metadata exceeds 16 MiB limit")
        payload = decode_metadata(archive.read(entry))
        prefix = candidates[0][:-len("data.pkl")]
        item_sizes = {"FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2,
                      "BFloat16Storage": 2, "LongStorage": 8, "IntStorage": 4,
                      "ShortStorage": 2, "CharStorage": 1, "ByteStorage": 1,
                      "BoolStorage": 1, "ComplexFloatStorage": 8,
                      "ComplexDoubleStorage": 16, "UntypedStorage": 1}
        visited = set()
        def check(value):
            if isinstance(value, TensorInfo):
                storage = value.storage
                name = prefix + "data/" + storage.key
                stored = archive.getinfo(name)
                expected = storage.elements * item_sizes[storage.kind]
                if expected < 0 or stored.file_size != expected:
                    raise ValueError(f"storage size mismatch: {name}")
            elif isinstance(value, (dict, list, tuple)) and id(value) not in visited:
                visited.add(id(value))
                for child in (value.values() if isinstance(value, dict) else value):
                    check(child)
        check(payload)
        return payload


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stat_key(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "device": st.st_dev, "inode": st.st_ino}


def clean(value, depth=0):
    if depth > 7:
        return "<depth limit>"
    if value is None or type(value) in {bool, int, str}:
        return value if not isinstance(value, str) else value[:4000]
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): clean(v, depth + 1) for k, v in list(value.items())[:500]
                if not isinstance(v, TensorInfo)}
    if isinstance(value, (list, tuple)):
        return [clean(v, depth + 1) for v in value[:100]]
    return f"<{type(value).__name__}>"


def tensor_manifest(payload):
    rows = []
    def walk(value, prefix="", depth=0):
        if isinstance(value, TensorInfo):
            normalized = re.sub(r"^(?:(?:model|model_state_dict|state_dict|module|base|admitted|brain)\.)+", "", prefix)
            group = normalized.split(".")[0]
            rows.append({"key": prefix, "normalized_key": normalized,
                         "group": group if group in GROUPS else "other",
                         "shape": list(value.shape), "stride": list(value.stride),
                         "numel": math.prod(value.shape), "storage_type": value.storage.kind,
                         "dtype": value.dtype, "storage_key": value.storage.key,
                         "storage_offset": value.offset,
                         "stored_device": value.storage.location})
        elif isinstance(value, dict) and depth < 6:
            for key, item in value.items():
                if str(key).lower() in {"optimizer", "optimizer_state_dict", "rng", "rng_state"}:
                    continue
                walk(item, f"{prefix}.{key}" if prefix else str(key), depth + 1)
    walk(payload)
    rows.sort(key=lambda x: x["key"])
    return rows


def describe_checkpoint(path):
    row = {"path": str(path), "before": stat_key(path), "metadata_status": "unread"}
    try:
        payload = read_checkpoint_metadata(path)
        tensors = tensor_manifest(payload)
        groups = {}
        for item in tensors:
            group = groups.setdefault(item["group"], {"tensor_count": 0, "numel": 0})
            group["tensor_count"] += 1
            group["numel"] += item["numel"]
        row.update(metadata_status="parsed", top_level_keys=[str(x) for x in payload],
                   format=clean(payload.get("format", payload.get("format_version"))),
                   updates=clean(payload.get("updates", payload.get("global_update"))),
                   config=clean(payload.get("config", payload.get("cfg", {}))),
                   metadata=clean(payload.get("metadata", {})),
                   provenance={str(k): clean(v) for k, v in payload.items()
                               if any(s in str(k) for s in ("sha256", "digest", "seed", "source_"))},
                   groups=groups, tensor_manifest=tensors,
                   has_planning_and_options=bool(groups.get("planning") and groups.get("options")))
        row["parameter_presence_is_training_proof"] = False
    except Exception as exc:
        row.update(metadata_status="unavailable", error=f"{type(exc).__name__}: {exc}",
                   has_planning_and_options=False)
    return row


def source_record(path, data):
    source = data.decode("utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    classes, functions = [], []
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(term in name.lower() for term in ("planning", "options", "world", "load_checkpoint", "teacher_action")):
                calls.add(name)
        if isinstance(node, ast.ClassDef) and node.name in REQUIRED_CLASSES | EXTRA_CLASSES:
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({"name": item.name,
                                    "signature": f"{item.name}({ast.unparse(item.args)})",
                                    "line": item.lineno})
            classes.append({"name": node.name, "line": node.lineno,
                            "bases": [ast.unparse(x) for x in node.bases], "methods": methods})
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0:
            if any(s in node.name for s in ("option", "planning", "load_checkpoint", "protected_sequence")):
                functions.append({"name": node.name, "signature": f"{node.name}({ast.unparse(node.args)})",
                                  "line": node.lineno})
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "classes": classes, "functions": functions, "relevant_calls": sorted(calls)}


def scan_files(roots, issues):
    files, seen = [], set()
    for root in roots:
        if not root.is_dir():
            issues.append({"path": str(root), "error": "scan root missing"})
            continue
        def onerror(exc):
            issues.append({"path": str(exc.filename), "error": str(exc)})
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                             and not d.startswith("planning_option_integration_audit")
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".json"} or path.is_symlink():
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    files.append(path.resolve())
                    if len(files) >= MAX_FILES:
                        issues.append({"error": "inventory file limit reached; scan incomplete"})
                        return files
    return files


def verify_p4b_report(data):
    """Verify original protocol and arithmetic; never infer a new behavioral pass."""
    errors = []
    def require(ok, text):
        if not ok:
            errors.append(text)
    exp = data.get("experiment")
    if not isinstance(exp, dict):
        errors.append("experiment must be a dictionary")
        exp = {}
    require(data.get("format") == "continuous_brain.p4b_hierarchical_option_planning_gate.v103", "format")
    expected = {"paired_episodes": 100, "steps_per_episode": 120, "seed_start": 12001,
                "seed_end": 12100, "option_depth": 2, "selector_topk": 2,
                "beam_width": 8, "max_option_steps": 12, "discount": 0.95,
                "min_commit": 3, "terminate_threshold": 0.5}
    for key, value in expected.items():
        require(exp.get(key) == value, f"protocol:{key}")
    require(exp.get("training_performed_during_evaluation") is False, "evaluation training flag")
    require(exp.get("parents_used") is False, "parents flag")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(data.get("checkpoint_sha256", "")))), "checkpoint hash")
    recomputed = {}
    for mode, summary_key in (("reactive", "reactive"), ("option_direct", "option_direct_diagnostic"),
                              ("hierarchical_option_planning", "hierarchical_option_planning")):
        episodes = data.get("episodes")
        rows = episodes.get(mode) if isinstance(episodes, dict) else None
        valid = isinstance(rows, list) and len(rows) == 100
        require(valid, f"{mode}:100 rows")
        if not valid:
            continue
        try:
            require(sorted(r["seed"] for r in rows) == list(range(12001, 12101)), f"{mode}:seeds")
            require(all(type(r["completed"]) is bool for r in rows), f"{mode}:boolean outcomes")
            require(all(type(r["collisions"]) in {int, float} and math.isfinite(r["collisions"])
                        and r["collisions"] >= 0 for r in rows), f"{mode}:collisions")
            completion = sum(r["completed"] is True for r in rows) / 100
            collisions = sum(float(r["collisions"]) for r in rows) / 100
            summary = data.get(summary_key)
            if not isinstance(summary, dict):
                summary = {}
            require(math.isclose(float(summary.get("completion_rate", -1)), completion, abs_tol=1e-12), f"{mode}:completion arithmetic")
            require(math.isclose(float(summary.get("mean_collisions", -1)), collisions, abs_tol=1e-12), f"{mode}:collision arithmetic")
            recomputed[mode] = {"completion_rate": completion, "mean_collisions": collisions}
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{mode}:malformed rows:{exc}")
    original_pass = None
    direct_advantage = None
    if len(recomputed) == 3:
        h, r, d = (recomputed[k] for k in ("hierarchical_option_planning", "reactive", "option_direct"))
        original_pass = (h["completion_rate"] >= .20
                         and h["completion_rate"] - r["completion_rate"] > .05
                         and h["mean_collisions"] - r["mean_collisions"] <= 2.)
        require(data.get("p4b_passed") is original_pass, "pass flag arithmetic")
        direct_advantage = 100 * (h["completion_rate"] - d["completion_rate"])
    return {"complete_original_protocol_and_arithmetic": not errors, "errors": errors,
            "original_gate_pass_recomputed": original_pass,
            "hierarchical_minus_direct_percentage_points": direct_advantage,
            "recomputed": recomputed, "new_behavioral_evaluation": False}


def read_evidence(files, source_bytes, issues, log):
    reports = []
    for path in files:
        if path.suffix.lower() != ".json" or not re.search(r"p4b|v103|hierarch|planning|v114", str(path), re.I):
            continue
        try:
            if path.stat().st_size > MAX_REPORT:
                issues.append({"path": str(path), "error": "historical report exceeds 8 MiB; not read"})
                continue
            raw = path.read_bytes()
            data = json.loads(raw)
            if not isinstance(data, dict) or "hierarchical_option_planning" not in data:
                continue
            check = verify_p4b_report(data)
            row = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                   "checkpoint_sha256": data.get("checkpoint_sha256"),
                   "checkpoint": data.get("checkpoint"), "verification": check,
                   "claimed_p4b_passed": data.get("p4b_passed")}
            reports.append(row)
            source_bytes[f"historical_reports/{len(reports):03d}_{path.name}"] = raw
        except (OSError, ValueError, TypeError) as exc:
            issues.append({"path": str(path), "error": f"report: {exc}"})
    log(f"Historical planning reports found: {len(reports)}")
    return reports


def priority(row, known_paths, historical_reports):
    if row["path"] in known_paths:
        return (0, list(known_paths).index(row["path"]), row["path"])
    metadata = row.get("metadata") or {}
    referenced = any(str(r.get("checkpoint")) == row["path"] for r in historical_reports)
    trained_claim = (isinstance(metadata, dict)
                     and metadata.get("option_cortex_version") == "latent-temporal-options-v1")
    return (1 if referenced else 2 if row.get("has_planning_and_options") and trained_claim
            else 3 if row.get("has_planning_and_options") else 4, 0, row["path"])


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_audit(project, extra_roots, max_hashes, log, report, bundle_sources, discover_siblings=True):
    source_dir = project / "continuous_brain"
    if not (source_dir / "infant.py").is_file():
        raise ValueError(f"Project source missing: {source_dir / 'infant.py'}")
    roots = [project]
    if discover_siblings:
        roots += sorted(p.resolve() for p in project.parent.glob("continuous_brain*")
                        if p.is_dir() and not p.is_symlink() and p.resolve() != project
                        and ((p / "state").is_dir() or (p / "continuous_brain_project/state").is_dir()))
    roots += [p.resolve() for p in extra_roots]
    roots = list(dict.fromkeys(roots))
    report["scan_roots"] = [str(p) for p in roots]
    issues = report["issues"]
    sources = []
    # Search class definitions statically; bundle only relevant local Python sources.
    source_paths = sorted(source_dir.rglob("*.py"))
    source_paths += sorted(project.glob("*planning*option*audit*.py"))
    source_paths += sorted(project.glob("*recurrent_option_progress*.py"))
    for path in source_paths:
        if path.is_symlink() or "__pycache__" in path.parts:
            continue
        try:
            if path.stat().st_size > MAX_SOURCE:
                issues.append({"path": str(path), "error": "source exceeds 2 MiB; not read"})
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig")
            relevant = (any(f"class {name}" in text for name in REQUIRED_CLASSES | EXTRA_CLASSES)
                        or path.name in {"planning_v103.py", "planning_v102.py", "planning_v100.py",
                                         "parented_train.py", "protected_growth_v114.py",
                                         "canonical_executive_rule_world_v3.py"}
                        or "recurrent_option_progress" in path.name)
            if not relevant:
                continue
            record = source_record(path, raw)
            sources.append(record)
            bundle_sources["sources/" + path.relative_to(project).as_posix()] = raw
        except (SyntaxError, UnicodeError, OSError) as exc:
            issues.append({"path": str(path), "error": f"source: {exc}"})
    report["sources"] = sources
    found = {c["name"] for s in sources for c in s["classes"]}
    report["required_classes_found"] = sorted(REQUIRED_CLASSES & found)
    report["required_classes_missing"] = sorted(REQUIRED_CLASSES - found)
    log("Source interfaces found: " + ", ".join(sorted(REQUIRED_CLASSES & found)))
    scan_issue_start = len(issues)
    files = scan_files(roots, issues)
    report["inventory_complete"] = len(issues) == scan_issue_start
    checkpoint_paths = [p for p in files if p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if len(checkpoint_paths) > MAX_CHECKPOINTS:
        raise RuntimeError("More than 4096 checkpoints found; use --current-project-only and targeted --scan-root directories")
    report["checkpoint_inventory"] = rows = []
    log(f"Checkpoint files found: {len(checkpoint_paths)}; reading metadata only")
    for i, path in enumerate(checkpoint_paths, 1):
        rows.append(describe_checkpoint(path))
        if i == 1 or i % 25 == 0 or i == len(checkpoint_paths):
            log(f"Metadata inspected: {i}/{len(checkpoint_paths)}")
    report["historical_reports"] = historical = read_evidence(files, bundle_sources, issues, log)
    known_paths = {str((project / relative).resolve()): name for name, (relative, _) in KNOWN_CHECKPOINTS.items()}
    known_existing = [row for row in rows if row["path"] in known_paths]
    ordered = sorted(rows, key=lambda r: priority(r, known_paths, historical))
    selected = ordered[:max(max_hashes, len(known_existing))]
    for i, row in enumerate(selected, 1):
        log(f"Hashing checkpoint {i}/{len(selected)}: {row['path']}")
        try:
            row["sha256_before"] = sha256(row["path"])
        except OSError as exc:
            row["hash_error"] = str(exc)
    by_path = {row["path"]: row for row in rows}
    report["known_checkpoints"] = known = {}
    for name, (relative, expected) in KNOWN_CHECKPOINTS.items():
        path = str((project / relative).resolve())
        row = by_path.get(path)
        known[name] = {"path": path, "exists": row is not None,
                       "expected_sha256": expected, "sha256": row.get("sha256_before") if row else None,
                       "metadata_status": row.get("metadata_status") if row else "missing"}
        known[name]["matches_expected_sha256"] = (known[name]["sha256"] == expected
                                                   if expected and known[name]["sha256"] else None)
    for row in rows:
        row["matching_historical_reports"] = [r["path"] for r in historical
            if row.get("sha256_before") and row["sha256_before"] == r.get("checkpoint_sha256")]
        row["matching_verified_p4b_report"] = any(
            row.get("sha256_before") and row["sha256_before"] == r.get("checkpoint_sha256")
            and r["verification"]["complete_original_protocol_and_arithmetic"]
            and r["verification"]["original_gate_pass_recomputed"]
            for r in historical)
    report["planning_option_candidates"] = [{k: row.get(k) for k in
        ("path", "sha256_before", "config", "metadata", "groups", "matching_historical_reports",
         "matching_verified_p4b_report")} for row in ordered if row.get("has_planning_and_options")]
    report["hash_coverage"] = {"limit_requested": max_hashes,
        "content_hash_paths": [row["path"] for row in selected],
        "unhashed_inventory_paths": [row["path"] for row in rows if row not in selected],
        "note": "All discovered checkpoint file stats are compared; selected contents are hashed twice. Others are not content-certified."}
    log("Verifying source and checkpoint preservation")
    changes = []
    for row in selected:
        try:
            row["sha256_after"] = sha256(row["path"])
            if not row.get("sha256_before") or row["sha256_after"] != row["sha256_before"]:
                changes.append({"path": row["path"], "kind": "checkpoint hash changed or unavailable"})
        except OSError as exc:
            changes.append({"path": row["path"], "kind": "checkpoint read error: " + str(exc)})
    for row in rows:
        try:
            row["after"] = stat_key(row["path"])
            if row["before"] != row["after"]:
                changes.append({"path": row["path"], "kind": "checkpoint stat changed"})
        except OSError as exc:
            changes.append({"path": row["path"], "kind": "checkpoint stat error: " + str(exc)})
    for source in sources:
        try:
            if sha256(source["path"]) != source["sha256"]:
                changes.append({"path": source["path"], "kind": "source changed"})
        except OSError as exc:
            changes.append({"path": source["path"], "kind": str(exc)})
    report["detected_input_changes"] = changes
    report["checkpoint_mutated"] = bool(any("checkpoint" in change["kind"] for change in changes))
    report["read_only_audit"] = not changes
    report["checkpoint_mutation_scope"] = report["hash_coverage"]
    canonical_ok = known["historical_base"]["metadata_status"] == "parsed"
    report["audit_complete"] = (not report["required_classes_missing"] and canonical_ok
                                and report["inventory_complete"] and not changes)
    report["status"] = "COMPLETE" if report["audit_complete"] else "INCOMPLETE"
    report["findings"] = [
        "Stored tensors and training metadata are inventoried; their presence does not prove a learned capability.",
        "Historical report verification concerns existing evidence, not a new evaluation or generalization result.",
        "Canonical recall and goal interfaces must be connected to the planning representation explicitly.",
        "Preserve the planner's spatial representation and option identities when comparing checkpoints.",
        "The next capability comparison must include direct options alongside hierarchical imagination.",
    ]
    if not any(r.get("matching_verified_p4b_report") for r in rows):
        report["findings"].append("No checkpoint was linked by content hash to a complete, passing original P4b report in this scan.")
    if any(row["metadata_status"] != "parsed" for row in rows):
        report["findings"].append("Some checkpoint metadata could not be read; see their individual errors. No missing weights were initialized.")
    return 0 if report["audit_complete"] else 2


def write_outputs(output, report, logs, bundle_sources):
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    log_bytes = ("\n".join(logs) + "\n").encode()
    atomic_write(output / "report.json", report_bytes)
    atomic_write(output / "report.log", log_bytes)
    content = {"report.json": report_bytes, "report.log": log_bytes, **bundle_sources}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(content.items())}
    tmp = output / "audit_bundle.zip.tmp"
    with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            archive.writestr(name, data)
        archive.writestr("MANIFEST_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(output / "audit_bundle.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--scan-root", type=Path, action="append", default=[],
                        help="Additional directory containing historical local checkpoints/reports")
    parser.add_argument("--current-project-only", action="store_true",
                        help="Skip automatic discovery of sibling project directories")
    parser.add_argument("--max-full-hashes", type=int, default=12,
                        help="Maximum full-content checkpoint comparisons; known parents are always included")
    args = parser.parse_args(argv)
    if not 1 <= args.max_full_hashes <= 4096:
        parser.error("--max-full-hashes must be between 1 and 4096")
    project = args.project_root.resolve()
    output_root = project / "state/planning_option_integration_audit_v1_1"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_root / "runs" / run_id
    logs = []
    def log(message):
        print(message, flush=True)
        logs.append(message)
    report = {"format": FORMAT, "version": VERSION, "run_id": run_id,
              "project_root": str(project), "status": "ERROR", "audit_complete": False,
              "read_only_audit": None, "training_performed": False,
              "inference_performed": False, "model_imported_or_constructed": False,
              "checkpoint_mutated": None, "capability_pass": None,
              "generalization_established": False, "v3_dev_touched": False,
              "v3_test_touched": False, "fresh_v4_touched": False,
              "runtime": {"python": sys.version, "platform": platform.platform()}, "issues": []}
    bundle_sources = {"audit_implementation.py": Path(__file__).read_bytes()}
    exit_code = 2
    log(f"PLANNING_OPTION_INTEGRATION_AUDIT_V1_1 {VERSION}")
    log("CPU file inspection only; checkpoint weights are not allocated or modified.")
    try:
        exit_code = run_audit(project, args.scan_root, args.max_full_hashes, log, report,
                             bundle_sources, discover_siblings=not args.current_project_only)
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        log("AUDIT_ERROR: " + report["fatal_error"])
    tail = ["READ_ONLY_AUDIT: " + json.dumps(report["read_only_audit"]),
            "TRAINING_PERFORMED: false",
            "CHECKPOINT_MUTATED: " + json.dumps(report["checkpoint_mutated"]),
            "CAPABILITY_EVALUATED: false",
            "PLANNING_OPTION_INTEGRATION_AUDIT_COMPLETE" if exit_code == 0 else "PLANNING_OPTION_INTEGRATION_AUDIT_INCOMPLETE",
            "PLANNING_OPTION_INTEGRATION_AUDIT_RUN: " + ("PASS" if exit_code == 0 else "INCOMPLETE"),
            f"Audit bundle: {output_root / 'audit_bundle.zip'}"]
    logs.extend(tail)
    try:
        write_outputs(output, report, logs, bundle_sources)
        # Latest outputs are replaced even for an incomplete run, preventing stale success.
        for name in ("report.json", "report.log", "audit_bundle.zip"):
            atomic_write(output_root / name, (output / name).read_bytes())
    except Exception as exc:
        print(f"AUDIT_OUTPUT_WRITE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("\n".join(tail), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
