from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import shutil
from typing import Any

from .learning import LearningPaths, load_json, read_jsonl, utc_now, write_json_atomic


_REPAIR_ID = "v0105_missing_cross_threshold_repair"


def _is_v0104_missing_cross_failure(row: dict[str, Any]) -> bool:
    if row.get("schema") != "author.embodied.v0104":
        return False
    if row.get("stage_skill") != "door_cross" or row.get("action") != "CrossThreshold":
        return False
    if bool(row.get("success")):
        return False
    verification = row.get("verification") or {}
    exc = str(verification.get("exception") or "")
    return "_cross_threshold_for_adapter" in exc and "not defined" in exc


def _decrement_row(row: dict[str, Any] | None, *, count: int, primitive_steps: int = 0) -> None:
    if not isinstance(row, dict) or count <= 0:
        return
    row["attempts"] = max(0, int(row.get("attempts", 0)) - int(count))
    row["failures"] = max(0, int(row.get("failures", 0)) - int(count))
    row["primitive_steps"] = max(0, int(row.get("primitive_steps", 0)) - int(primitive_steps))
    attempts = int(row.get("attempts", 0))
    successes = int(row.get("successes", 0))
    failures = int(row.get("failures", 0))
    # Keep the persisted counters internally consistent without inventing success.
    if successes + failures > attempts:
        row["failures"] = max(0, attempts - successes)


def repair_v0104_missing_cross_threshold(state_dir: str | Path) -> dict[str, Any]:
    """Remove only synthetic v0.10.4 CrossThreshold failures caused by the missing helper.

    Raw replay rows remain untouched as audit evidence.  A marker makes the repair
    idempotent and checkpoint backups are written before any mutation.
    """
    paths = LearningPaths.from_root(state_dir).ensure()
    migration_dir = paths.root / "migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    marker = migration_dir / f"{_REPAIR_ID}.json"
    if marker.exists():
        payload = load_json(marker, {})
        payload["already_applied"] = True
        return payload

    matches = [row for row in read_jsonl(paths.embodied_replay_file) if _is_v0104_missing_cross_failure(row)]
    per_agent: dict[str, Counter] = defaultdict(Counter)
    per_agent_level: dict[str, Counter] = defaultdict(Counter)
    per_agent_object: dict[str, Counter] = defaultdict(Counter)
    per_agent_steps: Counter = Counter()

    for row in matches:
        agent = str(row.get("agent") or "").strip().upper()
        if not agent:
            continue
        per_agent[agent]["count"] += 1
        per_agent_steps[agent] += max(0, int(row.get("primitive_steps", 0)))
        level = row.get("curriculum_level")
        if level is not None:
            per_agent_level[agent][str(int(level))] += 1
        obj_type = str(row.get("target_object_type") or "Door")
        per_agent_object[agent][obj_type] += 1

    repaired: dict[str, Any] = {}
    for agent, counter in sorted(per_agent.items()):
        count = int(counter["count"])
        if count <= 0:
            continue
        checkpoint = paths.agent_skill_checkpoint(agent)
        if not checkpoint.exists():
            repaired[agent] = {"matched_replay_rows": count, "checkpoint_found": False}
            continue
        payload = load_json(checkpoint, {})
        backup = checkpoint.with_name("embodied_skills.pre_v0105_cross_repair.json")
        if not backup.exists():
            shutil.copy2(checkpoint, backup)

        steps = int(per_agent_steps[agent])
        _decrement_row((payload.get("skills") or {}).get("door_cross"), count=count, primitive_steps=steps)

        by_object = (payload.get("by_object_type") or {}).get("door_cross") or {}
        for obj_type, obj_count in per_agent_object[agent].items():
            _decrement_row(by_object.get(obj_type), count=int(obj_count), primitive_steps=0)

        by_level = (payload.get("by_level") or {}).get("door_cross") or {}
        for level, level_count in per_agent_level[agent].items():
            _decrement_row(by_level.get(level), count=int(level_count), primitive_steps=0)

        write_json_atomic(checkpoint, payload)
        repaired[agent] = {
            "matched_replay_rows": count,
            "checkpoint_found": True,
            "backup": str(backup),
            "levels": dict(per_agent_level[agent]),
            "object_types": dict(per_agent_object[agent]),
        }

    result = {
        "repair_id": _REPAIR_ID,
        "applied_at": utc_now(),
        "state_dir": str(paths.root),
        "matched_replay_rows": len(matches),
        "agents": repaired,
        "raw_replay_preserved": True,
        "already_applied": False,
    }
    write_json_atomic(marker, result)
    return result
