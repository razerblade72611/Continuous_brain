#!/usr/bin/env python3
"""Non-overwriting, stopped-individual backup or explicitly identified fork."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import uuid

from continuous_brain.lifetime import LifetimeStore


def snapshot(source, destination, fork=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not (source / "individual.sqlite3").is_file():
        raise FileNotFoundError("Source is not an individual state directory")
    if destination.exists():
        raise FileExistsError("Destination already exists; backups are never overwritten")
    if source in destination.parents or destination in source.parents:
        raise ValueError("Use separate, non-nested source and destination directories")
    original = LifetimeStore(source)  # refuses a running source
    target = None
    try:
        state = original.load()
        if state is None:
            raise ValueError("No checkpoint to snapshot")
        target = LifetimeStore(destination)
        original.db.backup(target.db)
        for name in ("legacy_v053", "motor_history"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name)
        parent = state["self_model"]["individual_id"]
        if fork:
            state["self_model"]["individual_id"] = str(uuid.uuid4())
            state["self_model"]["lineage"] = {
                **state["self_model"]["lineage"], "parent_individual_id": parent,
                "fork_decision": state["self_model"]["decisions"], "kind": "explicit_fork"}
        target.save(state)
        result = {"kind": "fork" if fork else "backup", "source_individual_id": parent,
                  "individual_id": state["self_model"]["individual_id"],
                  "decisions": state["self_model"]["decisions"], "destination": str(destination)}
        (destination / "snapshot.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        original.close()
        if target is not None:
            target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--fork", action="store_true", help="new ID, same learned state, explicit parent lineage")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.source, args.destination, args.fork), indent=2))


if __name__ == "__main__":
    main()
