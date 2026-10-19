"""Attach a qualified motor to a continuation copy, never overwrite an individual."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from .lifetime import LifetimeStore
from .motor_lab import MotorLabLock, load_qualified


def attach_development_motor(source, destination, motor_state):
    source, destination, motor_state = map(lambda p: Path(p).resolve(), (source, destination, motor_state))
    paths = (source, destination, motor_state)
    for i, a in enumerate(paths):
        for b in paths[i+1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Source, destination and motor lab must be separate, non-nested directories")
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new path. Originals are never overwritten")
    if not (source / "individual.sqlite3").is_file():
        raise FileNotFoundError("Source individual checkpoint not found")
    if (source / "UPGRADE_INCOMPLETE").exists():
        raise ValueError("Source is an incomplete upgrade; use the untouched original")
    original = LifetimeStore(source)
    target = None
    try:
        state = original.load()
        if not state or state.get("version") not in ("0.6.0", "0.6.1", "0.6.2", "0.7.0"):
            raise ValueError("Expected an existing v0.6 individual")
        world = state["world_signature"]
        if (world["type"] != "DevelopmentRoom" or world["obs_dim"] != 6 or
                world["actions"] != ["idle", "advance", "turn_left", "turn_right", "interact"]):
            raise ValueError("This motor is qualified only for the development room, not THOR/other bodies")
        with MotorLabLock(motor_state):
            candidate = load_qualified(motor_state)
        destination.mkdir(parents=True, exist_ok=False)
        marker = destination / "UPGRADE_INCOMPLETE"
        marker.write_text("Incomplete continuation copy. Do not run or remove this marker. Originals are untouched.\n")
        target = LifetimeStore(destination)
        original.db.backup(target.db)
        for name in ("legacy_v053", "motor_history"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name)
        lineage = state["self_model"]["lineage"]
        history = list(lineage.get("motor_replacements", []))
        archive = destination / "motor_history" / f"replacement_{len(history):03d}.pt"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError("Motor history path exists; refusing to overwrite archived learning")
        with archive.open("xb") as handle:
            torch.save({key: state.get(key) for key in (
                "motor", "motor_metadata", "motor_hidden", "motor_prev", "motor_digest", "servo_motor")}, handle)
        proof = {"kind": "qualified_development_motor_attachment", "digest": candidate["digest"],
                 "attached_at_decision": state["self_model"]["decisions"],
                 "motor_updates": candidate["updates"], "qualification": candidate["qualification"],
                 "previous_motor_archive": archive.relative_to(destination).as_posix()}
        history.append(proof)
        state["self_model"]["lineage"] = {**lineage, "motor_replacements": history}
        state["servo_motor"] = candidate
        state["motor"] = state["motor_metadata"] = state["motor_hidden"] = None
        state["motor_prev"] = 0
        state["motor_digest"] = candidate["digest"]
        state["controller_updates"] = 0
        state["version"] = "0.7.0" if state.get("version") == "0.7.0" else "0.6.2"
        # No executive tensors, replay rows, self estimates, intentions or RNG are reset.
        # Old outcome statistics remain lifetime statistics; run_window separates new results.
        report = {"version": state["version"], "individual_id": state["self_model"]["individual_id"],
                  "decisions": state["self_model"]["decisions"], "updates": state["learner"]["updates"],
                  "memories": original.count("experience"), "motor_source": "learned_development_motor",
                  "motor_digest": candidate["digest"], "motor_teacher_actions_at_execution": 0,
                  "preserved": ["identity", "executive_weights", "workspace", "optimizer", "target_network",
                                "replay", "memories", "intentions", "world_state", "rng"],
                  "archived_previous_motor": proof["previous_motor_archive"],
                  "destination": str(destination), "source": str(source),
                  "note": "Continuation copy. Do not run source and destination concurrently."}
        target.append(state["self_model"]["ticks"], "motor_replacement", provenance=proof)
        target.save(state)
        target.load()
        (destination / "upgrade_report.json").write_text(json.dumps(report, indent=2))
        target.close()
        marker.unlink()
        return report
    finally:
        if target:
            target.close()
        original.close()
