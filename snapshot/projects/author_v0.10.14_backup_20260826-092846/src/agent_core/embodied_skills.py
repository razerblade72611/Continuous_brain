from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .learning import load_json, utc_now, write_json_atomic


STAGED_SKILLS = (
    "point_navigation",
    "door_approach",
    "door_open",
    "door_cross",
    "pickup_approach",
    "pickup_grasp",
    "place_approach",
    "place_release",
    "toggle_approach",
    "toggle_interaction",
    "peer_approach",
    "route_resume_target",
)

# v0.10 buckets that remain useful as historical evidence.  Approach/navigation
# deliberately has no staged alias because the v0.10 statistic mixed incompatible
# tasks and impossible episodes.
LEGACY_READINESS_ALIASES = {
    "door_open": "open",
    "door_cross": "cross_door",
    "pickup_grasp": "pickup",
    "place_release": "place",
    "toggle_interaction": "toggle",
    "peer_approach": "peer_meet",
}

LEGACY_WRITE_ALIASES = {
    "navigate": "point_navigation",
    "open": "door_open",
    "cross_door": "door_cross",
    "pickup": "pickup_grasp",
    "place": "place_release",
    "toggle": "toggle_interaction",
    "peer_meet": "peer_approach",
}


@dataclass
class SkillStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    primitive_steps: int = 0
    recoveries: int = 0  # recovery attempts; retained for v0.10.3 compatibility
    recovery_successes: int = 0
    recovery_exhausted: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def mean_steps(self) -> float:
        return self.primitive_steps / self.attempts if self.attempts else 0.0

    @property
    def recovery_unclassified(self) -> int:
        return max(0, self.recoveries - self.recovery_successes - self.recovery_exhausted)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recovery_attempts"] = int(self.recoveries)
        data["recovery_unclassified"] = int(self.recovery_unclassified)
        data["success_rate"] = round(self.success_rate, 6)
        data["mean_steps"] = round(self.mean_steps, 3)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SkillStats":
        payload = payload or {}
        return cls(
            attempts=int(payload.get("attempts", 0)),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            primitive_steps=int(payload.get("primitive_steps", 0)),
            recoveries=int(payload.get("recovery_attempts", payload.get("recoveries", 0))),
            recovery_successes=int(payload.get("recovery_successes", 0)),
            recovery_exhausted=int(payload.get("recovery_exhausted", 0)),
        )

    def merge(self, other: "SkillStats") -> None:
        self.attempts += other.attempts
        self.successes += other.successes
        self.failures += other.failures
        self.primitive_steps += other.primitive_steps
        self.recoveries += other.recoveries
        self.recovery_successes += other.recovery_successes
        self.recovery_exhausted += other.recovery_exhausted


class EmbodiedSkillLibrary:
    """Persistent per-agent staged body-skill competence.

    v0.10.3 keeps the semantic/body boundary: the LLM/user-facing layer selects a
    semantic target; these statistics only describe execution competence after that
    target has already been grounded.  v0.10 checkpoints are migrated losslessly
    into ``legacy_v010`` on first save.
    """

    VERSION = 3

    def __init__(self, path: str | Path | None = None, *, agent_label: str = "agent"):
        self.path = None if path is None else Path(path)
        self.agent_label = str(agent_label)
        self.skills: dict[str, SkillStats] = {name: SkillStats() for name in STAGED_SKILLS}
        self.by_object_type: dict[str, dict[str, SkillStats]] = {}
        self.by_level: dict[str, dict[str, SkillStats]] = {}
        self.legacy_v010: dict[str, Any] | None = None
        self.updated_at: str | None = None
        # Transient trainer context; never persisted. Runtime sessions leave this None.
        self.training_level: int | None = None
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        value = str(skill).strip().lower()
        return LEGACY_WRITE_ALIASES.get(value, value)

    @staticmethod
    def _normalize_object_type(object_type: str | None) -> str | None:
        if object_type is None:
            return None
        value = "".join(ch for ch in str(object_type).strip() if ch.isalnum() or ch in {"_", "-"})
        return value or None

    def _load(self) -> None:
        payload = load_json(self.path, {})
        self.agent_label = str(payload.get("agent_label", self.agent_label))
        self.updated_at = payload.get("updated_at")
        version = int(payload.get("version", 1) or 1)

        if version <= 1:
            # Preserve the exact old aggregate/object buckets rather than pretending
            # the overloaded v0.10 navigation metric is comparable with v0.10.3.
            self.legacy_v010 = {
                "version": version,
                "skills": deepcopy(payload.get("skills") or {}),
                "by_object_type": deepcopy(payload.get("by_object_type") or {}),
            }
            return

        legacy = payload.get("legacy_v010")
        self.legacy_v010 = deepcopy(legacy) if isinstance(legacy, dict) else None
        for skill, row in (payload.get("skills") or {}).items():
            self.skills[str(skill)] = SkillStats.from_dict(row if isinstance(row, dict) else {})
        self.by_object_type = {}
        for skill, objects in (payload.get("by_object_type") or {}).items():
            if not isinstance(objects, dict):
                continue
            self.by_object_type[str(skill)] = {
                str(obj_type): SkillStats.from_dict(stats if isinstance(stats, dict) else {})
                for obj_type, stats in objects.items()
            }
        self.by_level = {}
        for skill, levels in (payload.get("by_level") or {}).items():
            if not isinstance(levels, dict):
                continue
            self.by_level[str(skill)] = {
                str(level): SkillStats.from_dict(stats if isinstance(stats, dict) else {})
                for level, stats in levels.items()
            }

    def record(self, skill: str, *, success: bool, primitive_steps: int = 0, object_type: str | None = None, recovery: bool = False, recovery_success: bool = False, recovery_exhausted: bool = False, level: int | None = None) -> None:
        name = self._normalize_skill(skill)
        stats = self.skills.setdefault(name, SkillStats())
        stats.attempts += 1
        stats.successes += int(bool(success))
        stats.failures += int(not bool(success))
        stats.primitive_steps += max(0, int(primitive_steps))
        stats.recoveries += int(bool(recovery))
        stats.recovery_successes += int(bool(recovery_success))
        stats.recovery_exhausted += int(bool(recovery_exhausted))

        obj_type = self._normalize_object_type(object_type)
        if obj_type:
            object_stats = self.by_object_type.setdefault(name, {}).setdefault(obj_type, SkillStats())
            object_stats.attempts += 1
            object_stats.successes += int(bool(success))
            object_stats.failures += int(not bool(success))
            object_stats.primitive_steps += max(0, int(primitive_steps))
            object_stats.recoveries += int(bool(recovery))
            object_stats.recovery_successes += int(bool(recovery_success))
            object_stats.recovery_exhausted += int(bool(recovery_exhausted))

        effective_level = self.training_level if level is None else level
        if effective_level is not None:
            level_stats = self.by_level.setdefault(name, {}).setdefault(str(int(effective_level)), SkillStats())
            level_stats.attempts += 1
            level_stats.successes += int(bool(success))
            level_stats.failures += int(not bool(success))
            level_stats.primitive_steps += max(0, int(primitive_steps))
            level_stats.recoveries += int(bool(recovery))
            level_stats.recovery_successes += int(bool(recovery_success))
            level_stats.recovery_exhausted += int(bool(recovery_exhausted))

    def stats(self, skill: str, object_type: str | None = None) -> SkillStats:
        name = self._normalize_skill(skill)
        obj_type = self._normalize_object_type(object_type)
        if obj_type:
            row = self.by_object_type.get(name, {}).get(obj_type)
            if row is not None:
                return row
        return self.skills.get(name, SkillStats())

    def level_stats(self, skill: str, level: int) -> SkillStats:
        name = self._normalize_skill(skill)
        return self.by_level.get(name, {}).get(str(int(level)), SkillStats())

    def _legacy_stats(self, skill: str, object_type: str | None = None) -> SkillStats:
        if not self.legacy_v010:
            return SkillStats()
        legacy_name = LEGACY_READINESS_ALIASES.get(self._normalize_skill(skill))
        if not legacy_name:
            return SkillStats()
        obj_type = self._normalize_object_type(object_type)
        if obj_type:
            objects = (self.legacy_v010.get("by_object_type") or {}).get(legacy_name) or {}
            row = objects.get(obj_type)
            if isinstance(row, dict):
                return SkillStats.from_dict(row)
        row = (self.legacy_v010.get("skills") or {}).get(legacy_name)
        return SkillStats.from_dict(row if isinstance(row, dict) else {})

    def confidence(self, skill: str, object_type: str | None = None) -> float:
        row = self.stats(skill, object_type)
        if row.attempts == 0:
            row = self._legacy_stats(skill, object_type)
        return (row.successes + 1.0) / (row.attempts + 2.0)

    def ready(self, skill: str, object_type: str | None = None, *, min_attempts: int = 5, min_success_rate: float = 0.65) -> bool:
        row = self.stats(skill, object_type)
        if row.attempts < int(min_attempts) and object_type is not None:
            row = self.stats(skill)
        if row.attempts < int(min_attempts):
            legacy = self._legacy_stats(skill, object_type)
            if legacy.attempts < int(min_attempts) and object_type is not None:
                legacy = self._legacy_stats(skill)
            row = legacy if legacy.attempts >= int(min_attempts) else row
        return row.attempts >= int(min_attempts) and row.success_rate >= float(min_success_rate)

    def merge(self, other: "EmbodiedSkillLibrary") -> None:
        for skill, stats in other.skills.items():
            self.skills.setdefault(skill, SkillStats()).merge(stats)
        for skill, objects in other.by_object_type.items():
            mine = self.by_object_type.setdefault(skill, {})
            for obj_type, stats in objects.items():
                mine.setdefault(obj_type, SkillStats()).merge(stats)
        for skill, levels in other.by_level.items():
            mine = self.by_level.setdefault(skill, {})
            for level, stats in levels.items():
                mine.setdefault(level, SkillStats()).merge(stats)
        if self.legacy_v010 is None and other.legacy_v010 is not None:
            self.legacy_v010 = deepcopy(other.legacy_v010)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "version": self.VERSION,
            "kind": "per_agent_embodied_skill_competence",
            "agent_label": self.agent_label,
            "updated_at": self.updated_at or utc_now(),
            "skills": {name: stats.as_dict() for name, stats in sorted(self.skills.items())},
            "by_object_type": {
                skill: {obj: stats.as_dict() for obj, stats in sorted(objects.items())}
                for skill, objects in sorted(self.by_object_type.items())
            },
            "by_level": {
                skill: {level: stats.as_dict() for level, stats in sorted(levels.items(), key=lambda row: int(row[0]))}
                for skill, levels in sorted(self.by_level.items())
            },
        }
        if self.legacy_v010 is not None:
            data["legacy_v010"] = deepcopy(self.legacy_v010)
        return data

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("EmbodiedSkillLibrary.save requires a path")
        self.updated_at = utc_now()
        write_json_atomic(target, self.as_dict())
        self.path = target

    def level_summary(self) -> dict[str, Any]:
        levels: dict[str, dict[str, Any]] = {}
        for skill, rows in sorted(self.by_level.items()):
            for level, stats in sorted(rows.items(), key=lambda row: int(row[0])):
                if not stats.attempts:
                    continue
                levels.setdefault(level, {})[skill] = {
                    "attempts": stats.attempts,
                    "success_rate_pct": round(100.0 * stats.success_rate, 2),
                    "mean_steps": round(stats.mean_steps, 2),
                    "recovery_attempts": stats.recoveries,
                    "recovery_successes": stats.recovery_successes,
                    "recovery_exhausted": stats.recovery_exhausted,
                    "recovery_unclassified": stats.recovery_unclassified,
                    "recovery_attempt_rate_pct": round(100.0 * stats.recoveries / max(1, stats.attempts), 2),
                    "recovery_success_rate_pct": round(100.0 * stats.recovery_successes / max(1, stats.recoveries), 2),
                }
        return levels

    def compact_summary(self) -> dict[str, Any]:
        staged = {
            skill: {
                "attempts": stats.attempts,
                "success_rate_pct": round(100.0 * stats.success_rate, 2),
                "mean_steps": round(stats.mean_steps, 2),
                "recovery_attempts": stats.recoveries,
                "recovery_successes": stats.recovery_successes,
                "recovery_exhausted": stats.recovery_exhausted,
                "recovery_unclassified": stats.recovery_unclassified,
                "recovery_attempt_rate_pct": round(100.0 * stats.recoveries / max(1, stats.attempts), 2),
                "recovery_success_rate_pct": round(100.0 * stats.recovery_successes / max(1, stats.recoveries), 2),
            }
            for skill, stats in sorted(self.skills.items())
            if stats.attempts
        }
        if self.legacy_v010:
            legacy = {}
            for skill, row in sorted((self.legacy_v010.get("skills") or {}).items()):
                stats = SkillStats.from_dict(row if isinstance(row, dict) else {})
                if stats.attempts:
                    legacy[skill] = {
                        "attempts": stats.attempts,
                        "success_rate_pct": round(100.0 * stats.success_rate, 2),
                        "mean_steps": round(stats.mean_steps, 2),
                    }
            if legacy:
                staged["legacy_v010"] = legacy
        return staged
