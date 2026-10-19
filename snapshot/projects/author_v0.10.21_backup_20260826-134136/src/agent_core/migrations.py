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

_V01018_PLACE_REPAIR_ID = "v01019_v01018_place_regression_repair"


def _v01018_place_cross_regression(row: dict[str, Any]) -> bool:
    if row.get("schema") != "author.embodied.v01018":
        return False
    if str(row.get("curriculum_skill") or "") != "place" or int(row.get("curriculum_level") or -1) != 4:
        return False
    if row.get("stage_skill") != "place_approach" or bool(row.get("success")):
        return False
    aux = list(row.get("auxiliary_skills") or [])
    cross = next((a for a in aux if str(a.get("skill") or "") == "door_cross"), None)
    return isinstance(cross, dict) and not bool(cross.get("success"))


def _v01018_invalid_place_release(row: dict[str, Any]) -> bool:
    if row.get("schema") != "author.embodied.v01018":
        return False
    if str(row.get("curriculum_skill") or "") != "place" or int(row.get("curriculum_level") or -1) != 4:
        return False
    if row.get("stage_skill") != "place_release" or bool(row.get("success")):
        return False
    verification = row.get("verification") or {}
    error = str(verification.get("last_action_error") or "")
    return (
        "No valid positions to place object found" in error
        or "Target openable Receptacle is CLOSED" in error
    )


def _decrement_stats_row(
    row: dict[str, Any] | None,
    *,
    success: bool,
    primitive_steps: int = 0,
    recovery: bool = False,
    recovery_success: bool = False,
    recovery_exhausted: bool = False,
) -> None:
    if not isinstance(row, dict):
        return
    row["attempts"] = max(0, int(row.get("attempts", 0)) - 1)
    key = "successes" if success else "failures"
    row[key] = max(0, int(row.get(key, 0)) - 1)
    row["primitive_steps"] = max(0, int(row.get("primitive_steps", 0)) - max(0, int(primitive_steps)))
    if recovery:
        # Checkpoints may use either historical `recoveries` or serialized
        # `recovery_attempts`; preserve whichever representation exists.
        if "recovery_attempts" in row:
            row["recovery_attempts"] = max(0, int(row.get("recovery_attempts", 0)) - 1)
        if "recoveries" in row:
            row["recoveries"] = max(0, int(row.get("recoveries", 0)) - 1)
    if recovery_success:
        row["recovery_successes"] = max(0, int(row.get("recovery_successes", 0)) - 1)
    if recovery_exhausted:
        row["recovery_exhausted"] = max(0, int(row.get("recovery_exhausted", 0)) - 1)
    attempts = int(row.get("attempts", 0))
    successes = int(row.get("successes", 0))
    failures = int(row.get("failures", 0))
    if successes + failures > attempts:
        # Only ever reduce the bucket corresponding to the removed replay row.
        if success:
            row["successes"] = max(0, attempts - failures)
        else:
            row["failures"] = max(0, attempts - successes)


def repair_v01018_place_regression(state_dir: str | Path) -> dict[str, Any]:
    """Unscore only known v0.10.18 Level-4 place trainer/verifier regressions.

    The raw embodied replay is preserved as audit evidence.  This repair removes:
      * place_approach/route_resume/door_cross failures where the v0.10.18 cross
        helper recorded `door_cross=false`, and
      * place_release failures whose simulator error proves the trainer admitted a
        closed or non-placeable receptacle contract.

    Successful approach/open/cross/release evidence is retained.  The operation is
    idempotent and backs up each checkpoint before mutation.
    """
    paths = LearningPaths.from_root(state_dir).ensure()
    migration_dir = paths.root / "migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    marker = migration_dir / f"{_V01018_PLACE_REPAIR_ID}.json"
    if marker.exists():
        payload = load_json(marker, {})
        payload["already_applied"] = True
        return payload

    replay = read_jsonl(paths.embodied_replay_file)
    bad_approach = [row for row in replay if _v01018_place_cross_regression(row)]
    bad_release = [row for row in replay if _v01018_invalid_place_release(row)]

    # Build exact skill-removal events from replay evidence.
    removals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bad_approach:
        agent = str(row.get("agent") or "").strip().upper()
        if not agent:
            continue
        obj_type = str(row.get("target_object_type") or "") or None
        steps = max(0, int(row.get("primitive_steps", 0)))
        removals[agent].append({"skill":"place_approach","success":False,"steps":steps,"object_type":obj_type,"level":4})
        removals[agent].append({"skill":"route_resume_target","success":False,"steps":steps,"object_type":obj_type,"level":4})
        cross = next((a for a in (row.get("auxiliary_skills") or []) if str(a.get("skill") or "") == "door_cross"), {})
        removals[agent].append({
            "skill":"door_cross","success":False,"steps":max(0,int(cross.get("primitive_steps",0))),
            "object_type":"Door","level":4,"recovery":bool(cross.get("recovery",False)),
            "recovery_success":bool(cross.get("recovery_success",False)),
            "recovery_exhausted":bool(cross.get("recovery_exhausted",False)),
        })

    for row in bad_release:
        agent = str(row.get("agent") or "").strip().upper()
        if not agent:
            continue
        verification = row.get("verification") or {}
        recovery = bool(verification.get("interaction_recovery_used"))
        removals[agent].append({
            "skill":"place_release","success":False,"steps":max(0,int(row.get("primitive_steps",0))),
            "object_type":str(row.get("target_object_type") or "") or None,"level":4,
            "recovery":recovery,
            "recovery_success":bool(verification.get("interaction_recovery_success")),
            "recovery_exhausted":bool(verification.get("interaction_recovery_exhausted")),
        })

    repaired: dict[str, Any] = {}
    for agent, events in sorted(removals.items()):
        checkpoint = paths.agent_skill_checkpoint(agent)
        if not checkpoint.exists():
            repaired[agent] = {"events": len(events), "checkpoint_found": False}
            continue
        payload = load_json(checkpoint, {})
        backup = checkpoint.with_name("embodied_skills.pre_v01019_place_repair.json")
        if not backup.exists():
            shutil.copy2(checkpoint, backup)
        for event in events:
            skill = event["skill"]
            kwargs = dict(
                success=bool(event["success"]),
                primitive_steps=int(event.get("steps",0)),
                recovery=bool(event.get("recovery",False)),
                recovery_success=bool(event.get("recovery_success",False)),
                recovery_exhausted=bool(event.get("recovery_exhausted",False)),
            )
            _decrement_stats_row((payload.get("skills") or {}).get(skill), **kwargs)
            obj_type = event.get("object_type")
            if obj_type:
                _decrement_stats_row(((payload.get("by_object_type") or {}).get(skill) or {}).get(str(obj_type)), **kwargs)
            _decrement_stats_row(((payload.get("by_level") or {}).get(skill) or {}).get(str(int(event.get("level",4)))), **kwargs)
        write_json_atomic(checkpoint, payload)
        repaired[agent] = {"events": len(events), "checkpoint_found": True, "backup": str(backup)}

    result = {
        "repair_id": _V01018_PLACE_REPAIR_ID,
        "applied_at": utc_now(),
        "state_dir": str(paths.root),
        "matched_place_approach_cross_regressions": len(bad_approach),
        "matched_invalid_place_releases": len(bad_release),
        "agents": repaired,
        "raw_replay_preserved": True,
        "already_applied": False,
    }
    write_json_atomic(marker, result)
    return result

_V01019_PLACE_TRANSPORT_REPAIR_ID = "v01020_v01019_place_transport_regression_repair"


def _v01019_place_transport_regression(row: dict[str, Any]) -> bool:
    """Match only v0.10.19 Level-4 place failures caused by carried-route geometry."""
    if row.get("schema") != "author.embodied.v01019":
        return False
    if str(row.get("curriculum_skill") or "") != "place" or int(row.get("curriculum_level") or -1) != 4:
        return False
    if row.get("stage_skill") != "place_approach" or bool(row.get("success")):
        return False
    verification = row.get("verification") or {}
    error = str(verification.get("last_action_error") or "")
    aux = list(row.get("auxiliary_skills") or [])
    cross = next((a for a in aux if str(a.get("skill") or "") == "door_cross"), None)
    held_collision = "a held item:" in error
    portal_collision = "PortalBoundary" in error
    primitive_exception = isinstance(cross, dict) and str(cross.get("verification_basis") or "") == "cross_primitive_exception"
    return bool(held_collision or portal_collision or primitive_exception)


def repair_v01019_place_transport_regression(state_dir: str | Path) -> dict[str, Any]:
    """Unscore known v0.10.19 carried-object doorway transport failures.

    v0.10.19 correctly fixed final PutObject contracts but still used a body-only
    path/rotation schedule while the agent was carrying the placement source. The
    resulting failures are identified from simulator evidence (held-item rotation
    collision, explicit PortalBoundary collision, or the corresponding cross
    primitive exception). Successful route/open/place evidence and raw replay are
    preserved. The operation is idempotent and backs up both agent checkpoints.
    """
    paths = LearningPaths.from_root(state_dir).ensure()
    migration_dir = paths.root / "migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    marker = migration_dir / f"{_V01019_PLACE_TRANSPORT_REPAIR_ID}.json"
    if marker.exists():
        payload = load_json(marker, {})
        payload["already_applied"] = True
        return payload

    replay = read_jsonl(paths.embodied_replay_file)
    matches = [row for row in replay if _v01019_place_transport_regression(row)]

    removals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        agent = str(row.get("agent") or "").strip().upper()
        if not agent:
            continue
        obj_type = str(row.get("target_object_type") or "") or None
        steps = max(0, int(row.get("primitive_steps", 0)))
        removals[agent].append({
            "skill": "place_approach", "success": False, "steps": steps,
            "object_type": obj_type, "level": 4,
        })
        removals[agent].append({
            "skill": "route_resume_target", "success": False, "steps": steps,
            "object_type": obj_type, "level": 4,
        })
        for aux in row.get("auxiliary_skills") or []:
            skill = str(aux.get("skill") or "")
            if skill not in {"door_approach", "door_open", "door_cross"} or bool(aux.get("success")):
                continue
            removals[agent].append({
                "skill": skill,
                "success": False,
                "steps": max(0, int(aux.get("primitive_steps", 0))),
                "object_type": aux.get("object_type") or "Door",
                "level": 4,
                "recovery": bool(aux.get("recovery", False)),
                "recovery_success": bool(aux.get("recovery_success", False)),
                "recovery_exhausted": bool(aux.get("recovery_exhausted", False)),
            })

    repaired: dict[str, Any] = {}
    for agent, events in sorted(removals.items()):
        checkpoint = paths.agent_skill_checkpoint(agent)
        if not checkpoint.exists():
            repaired[agent] = {"events": len(events), "checkpoint_found": False}
            continue
        payload = load_json(checkpoint, {})
        backup = checkpoint.with_name("embodied_skills.pre_v01020_place_transport_repair.json")
        if not backup.exists():
            shutil.copy2(checkpoint, backup)
        for event in events:
            skill = event["skill"]
            kwargs = dict(
                success=False,
                primitive_steps=int(event.get("steps", 0)),
                recovery=bool(event.get("recovery", False)),
                recovery_success=bool(event.get("recovery_success", False)),
                recovery_exhausted=bool(event.get("recovery_exhausted", False)),
            )
            _decrement_stats_row((payload.get("skills") or {}).get(skill), **kwargs)
            obj_type = event.get("object_type")
            if obj_type:
                _decrement_stats_row(((payload.get("by_object_type") or {}).get(skill) or {}).get(str(obj_type)), **kwargs)
            _decrement_stats_row(((payload.get("by_level") or {}).get(skill) or {}).get("4"), **kwargs)
        write_json_atomic(checkpoint, payload)
        repaired[agent] = {"events": len(events), "checkpoint_found": True, "backup": str(backup)}

    result = {
        "repair_id": _V01019_PLACE_TRANSPORT_REPAIR_ID,
        "applied_at": utc_now(),
        "state_dir": str(paths.root),
        "matched_place_transport_regressions": len(matches),
        "agents": repaired,
        "raw_replay_preserved": True,
        "already_applied": False,
    }
    write_json_atomic(marker, result)
    return result

_V01020_PLACE_CARRY_REPAIR_ID = "v01021_v01020_place_carry_route_regression_repair"


def _v01020_place_carry_route_regression(row: dict[str, Any]) -> bool:
    """Match only v0.10.20 Level-4 place failures tied to carried-route mechanics."""
    if row.get("schema") != "author.embodied.v01020":
        return False
    if str(row.get("curriculum_skill") or "") != "place" or int(row.get("curriculum_level") or -1) != 4:
        return False
    if row.get("stage_skill") != "place_approach" or bool(row.get("success")):
        return False
    verification = row.get("verification") or {}
    reason = str(verification.get("reason") or "")
    error = str(verification.get("last_action_error") or "")
    lowered = error.lower()
    aux = list(row.get("auxiliary_skills") or [])
    cross = next((a for a in aux if str(a.get("skill") or "") == "door_cross"), None)

    held_turn_collision = "a held item:" in lowered and "rotates" in lowered
    portal_collision = "portalboundary" in lowered
    physical_blocker = "blocking the agent" in lowered
    pinned_omission = reason == "pinned_pose_not_available" and not error
    true_cross_geometry = (
        isinstance(cross, dict)
        and not bool(cross.get("success"))
        and ("discoverydoor_" in lowered or portal_collision)
    )
    return bool(held_turn_collision or portal_collision or physical_blocker or pinned_omission or true_cross_geometry)


def repair_v01020_place_carry_route_regression(state_dir: str | Path) -> dict[str, Any]:
    """Unscore known v0.10.20 carried-object route implementation failures.

    The repair is deliberately replay-driven. It removes only failed v0.10.20
    Level-4 place approach evidence whose simulator trace matches the carried-turn,
    portal-jamb, blocker-wording, pinned-pose omission, or selected-door collision
    bugs corrected in v0.10.21. Successful door/open/cross/place evidence remains
    untouched. Raw replay is retained verbatim for auditability.
    """
    paths = LearningPaths.from_root(state_dir).ensure()
    migration_dir = paths.root / "migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    marker = migration_dir / f"{_V01020_PLACE_CARRY_REPAIR_ID}.json"
    if marker.exists():
        payload = load_json(marker, {})
        payload["already_applied"] = True
        return payload

    replay = read_jsonl(paths.embodied_replay_file)
    matches = [row for row in replay if _v01020_place_carry_route_regression(row)]

    removals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        agent = str(row.get("agent") or "").strip().upper()
        if not agent:
            continue
        obj_type = str(row.get("target_object_type") or "") or None
        steps = max(0, int(row.get("primitive_steps", 0)))
        removals[agent].append({
            "skill": "place_approach", "success": False, "steps": steps,
            "object_type": obj_type, "level": 4,
        })
        removals[agent].append({
            "skill": "route_resume_target", "success": False, "steps": steps,
            "object_type": obj_type, "level": 4,
        })
        for aux in row.get("auxiliary_skills") or []:
            skill = str(aux.get("skill") or "")
            if skill not in {"door_approach", "door_open", "door_cross"} or bool(aux.get("success")):
                continue
            removals[agent].append({
                "skill": skill,
                "success": False,
                "steps": max(0, int(aux.get("primitive_steps", 0))),
                "object_type": aux.get("object_type") or "Door",
                "level": 4,
                "recovery": bool(aux.get("recovery", False)),
                "recovery_success": bool(aux.get("recovery_success", False)),
                "recovery_exhausted": bool(aux.get("recovery_exhausted", False)),
            })

    repaired: dict[str, Any] = {}
    for agent, events in sorted(removals.items()):
        checkpoint = paths.agent_skill_checkpoint(agent)
        if not checkpoint.exists():
            repaired[agent] = {"events": len(events), "checkpoint_found": False}
            continue
        payload = load_json(checkpoint, {})
        backup = checkpoint.with_name("embodied_skills.pre_v01021_place_carry_repair.json")
        if not backup.exists():
            shutil.copy2(checkpoint, backup)
        for event in events:
            kwargs = dict(
                success=False,
                primitive_steps=int(event.get("steps", 0)),
                recovery=bool(event.get("recovery", False)),
                recovery_success=bool(event.get("recovery_success", False)),
                recovery_exhausted=bool(event.get("recovery_exhausted", False)),
            )
            skill = event["skill"]
            _decrement_stats_row((payload.get("skills") or {}).get(skill), **kwargs)
            obj_type = event.get("object_type")
            if obj_type:
                _decrement_stats_row(((payload.get("by_object_type") or {}).get(skill) or {}).get(str(obj_type)), **kwargs)
            _decrement_stats_row(((payload.get("by_level") or {}).get(skill) or {}).get("4"), **kwargs)
        write_json_atomic(checkpoint, payload)
        repaired[agent] = {"events": len(events), "checkpoint_found": True, "backup": str(backup)}

    result = {
        "repair_id": _V01020_PLACE_CARRY_REPAIR_ID,
        "applied_at": utc_now(),
        "state_dir": str(paths.root),
        "matched_place_carry_route_regressions": len(matches),
        "agents": repaired,
        "raw_replay_preserved": True,
        "already_applied": False,
    }
    write_json_atomic(marker, result)
    return result
