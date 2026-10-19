from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .learning import load_json, utc_now, write_json_atomic


SKILLS = (
    "navigate",
    "face_target",
    "open",
    "cross_door",
    "pickup",
    "place",
    "toggle",
    "peer_meet",
)


@dataclass
class SkillStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    primitive_steps: int = 0
    recoveries: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def mean_steps(self) -> float:
        return self.primitive_steps / self.attempts if self.attempts else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
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
            recoveries=int(payload.get("recoveries", 0)),
        )

    def merge(self, other: "SkillStats") -> None:
        self.attempts += other.attempts
        self.successes += other.successes
        self.failures += other.failures
        self.primitive_steps += other.primitive_steps
        self.recoveries += other.recoveries


class EmbodiedSkillLibrary:
    """Persistent per-agent competence statistics for embodied motor macros.

    This checkpoint never chooses semantic goals or targets.  It only records how
    reliable the body has become at executing a named skill after a target has
    already been selected by the agent/user-facing reasoning layer.
    """

    VERSION = 1

    def __init__(self, path: str | Path | None = None, *, agent_label: str = "agent"):
        self.path = None if path is None else Path(path)
        self.agent_label = str(agent_label)
        self.skills: dict[str, SkillStats] = {name: SkillStats() for name in SKILLS}
        self.by_object_type: dict[str, dict[str, SkillStats]] = {}
        self.updated_at: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        value = str(skill).strip().lower()
        return value if value in SKILLS else value

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

    def record(
        self,
        skill: str,
        *,
        success: bool,
        primitive_steps: int = 0,
        object_type: str | None = None,
        recovery: bool = False,
    ) -> None:
        name = self._normalize_skill(skill)
        stats = self.skills.setdefault(name, SkillStats())
        stats.attempts += 1
        stats.successes += int(bool(success))
        stats.failures += int(not bool(success))
        stats.primitive_steps += max(0, int(primitive_steps))
        stats.recoveries += int(bool(recovery))

        obj_type = self._normalize_object_type(object_type)
        if obj_type:
            object_stats = self.by_object_type.setdefault(name, {}).setdefault(obj_type, SkillStats())
            object_stats.attempts += 1
            object_stats.successes += int(bool(success))
            object_stats.failures += int(not bool(success))
            object_stats.primitive_steps += max(0, int(primitive_steps))
            object_stats.recoveries += int(bool(recovery))

    def stats(self, skill: str, object_type: str | None = None) -> SkillStats:
        name = self._normalize_skill(skill)
        obj_type = self._normalize_object_type(object_type)
        if obj_type:
            row = self.by_object_type.get(name, {}).get(obj_type)
            if row is not None:
                return row
        return self.skills.get(name, SkillStats())

    def confidence(self, skill: str, object_type: str | None = None) -> float:
        """Beta(1,1) posterior mean; avoids confidence=1 after a single success."""
        row = self.stats(skill, object_type)
        return (row.successes + 1.0) / (row.attempts + 2.0)

    def ready(
        self,
        skill: str,
        object_type: str | None = None,
        *,
        min_attempts: int = 5,
        min_success_rate: float = 0.65,
    ) -> bool:
        row = self.stats(skill, object_type)
        if row.attempts < int(min_attempts):
            # Fall back to the generic skill if the object-specific bucket is sparse.
            if object_type is not None:
                row = self.stats(skill)
        return row.attempts >= int(min_attempts) and row.success_rate >= float(min_success_rate)

    def merge(self, other: "EmbodiedSkillLibrary") -> None:
        for skill, stats in other.skills.items():
            self.skills.setdefault(skill, SkillStats()).merge(stats)
        for skill, objects in other.by_object_type.items():
            mine = self.by_object_type.setdefault(skill, {})
            for obj_type, stats in objects.items():
                mine.setdefault(obj_type, SkillStats()).merge(stats)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "kind": "per_agent_embodied_skill_competence",
            "agent_label": self.agent_label,
            "updated_at": self.updated_at or utc_now(),
            "skills": {name: stats.as_dict() for name, stats in sorted(self.skills.items())},
            "by_object_type": {
                skill: {obj: stats.as_dict() for obj, stats in sorted(objects.items())}
                for skill, objects in sorted(self.by_object_type.items())
            },
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("EmbodiedSkillLibrary.save requires a path")
        self.updated_at = utc_now()
        write_json_atomic(target, self.as_dict())
        self.path = target

    def compact_summary(self) -> dict[str, Any]:
        return {
            skill: {
                "attempts": stats.attempts,
                "success_rate_pct": round(100.0 * stats.success_rate, 2),
                "mean_steps": round(stats.mean_steps, 2),
            }
            for skill, stats in sorted(self.skills.items())
            if stats.attempts
        }
