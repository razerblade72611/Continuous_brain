"""Copy-on-upgrade: keep the original individual and legacy state untouched."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from .legacy_motor import read_legacy_motor, archive_legacy_motor
from .lifetime import LifetimeStore


def upgrade_individual(source, destination, legacy_motor=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    paths = [source, destination]
    if legacy_motor is not None:
        legacy_motor = Path(legacy_motor).resolve()
        paths.append(legacy_motor)
    for i, a in enumerate(paths):
        for b in paths[i+1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Source, destination and legacy source must be separate non-nested directories")
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new path. Nothing is overwritten")
    if not (source / "individual.sqlite3").is_file():
        raise FileNotFoundError("Source must contain an existing individual.sqlite3")
    if (source / "UPGRADE_INCOMPLETE").exists():
        raise ValueError("Source is an incomplete upgrade; use the untouched original")
    original = LifetimeStore(source)  # holds the stopped-source lock until done
    target = None
    try:
        state = original.load()
        if not state or state.get("version") not in ("0.6.0", "0.6.1", "0.6.2", "0.7.0"):
            raise ValueError("Expected a v0.6 individual")
        imported = None
        if legacy_motor is not None:
            if state.get("motor") is not None or state.get("servo_motor") is not None or (source / "legacy_v053").exists():
                raise ValueError("This individual already has a motor/archive; replacement is not automatic")
            imported = read_legacy_motor(legacy_motor, state["vision"])
        # Validation above happens before creating any destination. An I/O failure
        # below leaves a marked, non-runnable directory and untouched originals.
        destination.mkdir(parents=True, exist_ok=False)
        marker = destination / "UPGRADE_INCOMPLETE"
        marker.write_text("Do not run this directory until upgrade completes. Originals are untouched.\n")
        target = LifetimeStore(destination)
        original.db.backup(target.db)
        if imported:
            proof = {**imported["provenance"], "attached_at_decision": state["self_model"]["decisions"],
                     "kind": "existing_executive_attached_legacy_motor"}
            archive_legacy_motor(legacy_motor, destination / "legacy_v053", proof["sha256"])
            state["motor"] = imported["weights"]
            state["motor_metadata"] = imported["metadata"]
            state["motor_hidden"] = torch.zeros(1, imported["metadata"]["hidden_dim"])
            state["motor_prev"] = 0
            state["motor_digest"] = proof["sha256"]
            state["self_model"]["lineage"] = {**state["self_model"]["lineage"], "motor_attachment": proof}
            target.append(state["self_model"]["ticks"], "lineage", provenance=proof)
        elif (source / "legacy_v053").exists():
            shutil.copytree(source / "legacy_v053", destination / "legacy_v053")
        if (source / "motor_history").exists():
            shutil.copytree(source / "motor_history", destination / "motor_history")
        if (source / "semantic_cortex").exists():
            shutil.copytree(source / "semantic_cortex", destination / "semantic_cortex")
        source_version = state.get("version", "0.6.0")
        report = {"version": source_version, "individual_id": state["self_model"]["individual_id"],
                  "decisions": state["self_model"]["decisions"],
                  "updates": state["learner"]["updates"], "memories": original.count("experience"),
                  "source": str(source), "destination": str(destination),
                  "motor_source": "learned_development_motor" if state.get("servo_motor") is not None else
                    "imported_learned" if state.get("motor") is not None else "development_reference_teacher",
                  "legacy_motor_attached": bool(imported),
                  "preserved": ["identity", "executive_weights", "workspace", "optimizer", "target_network",
                                "replay", "episodic_memory", "intentions", "world_state", "rng"],
                  "note": "Continuation copy, not a new individual. Do not run both copies concurrently."}
        target.append(state["self_model"]["ticks"], "upgrade", report=report)
        # Old executables must refuse this copy rather than silently reverting
        # controller semantics. No learned state or historical rows are rewritten.
        source_version = state.get("version", "0.6.0")
        state["version"] = source_version if source_version in ("0.6.2", "0.7.0") else "0.6.1"
        target.save(state)
        target.load()  # verifies serialized checkpoint integrity
        (destination / "upgrade_report.json").write_text(json.dumps(report, indent=2))
        target.close()
        marker.unlink()
        return report
    finally:
        if target:
            target.close()
        original.close()
