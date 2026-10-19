from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .interaction_pose import InteractionPose
from .learning import load_json, utc_now, write_json_atomic
from .navigation import planar_distance


@dataclass
class PoseActionStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    recoveries: int = 0
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PoseActionStats":
        payload = payload or {}
        return cls(
            attempts=int(payload.get("attempts", 0)),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            recoveries=int(payload.get("recoveries", 0)),
            last_error=(None if payload.get("last_error") in (None, "") else str(payload.get("last_error"))),
        )

    def merge(self, other: "PoseActionStats") -> None:
        self.attempts += int(other.attempts)
        self.successes += int(other.successes)
        self.failures += int(other.failures)
        self.recoveries += int(other.recoveries)
        if other.last_error:
            self.last_error = other.last_error

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["success_rate"] = round(self.success_rate, 6)
        return row


class PoseActionabilityMemory:
    """Controller-private experience about whether an action works from a body pose.

    GetInteractablePoses means "visible/in range from here", not "every interaction is
    guaranteed to succeed from here".  This store learns the missing actionability
    layer without exposing coordinates to the LLM.  It is deliberately per-agent so
    A and B can accumulate different embodied experience.
    """

    VERSION = 1

    def __init__(self, path: str | Path | None, *, agent_label: str = "agent", grid_size: float = 0.25):
        self.path = None if path is None else Path(path)
        self.agent_label = str(agent_label)
        self.grid_size = float(grid_size)
        self.pose_stats: dict[str, PoseActionStats] = {}
        self.updated_at: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _safe(value: str) -> str:
        return str(value).replace("|", "_").replace("\n", " ").strip()

    def _pose_key(self, action_kind: str, object_id: str, pose: InteractionPose) -> str:
        gx, gz, yaw, horizon = pose.key(self.grid_size)
        return "|".join((
            self._safe(action_kind),
            self._safe(object_id),
            str(gx),
            str(gz),
            str(yaw),
            str(horizon),
        ))

    def _load(self) -> None:
        payload = load_json(self.path, {})
        self.agent_label = str(payload.get("agent_label", self.agent_label))
        self.updated_at = payload.get("updated_at")
        rows = payload.get("pose_stats") or {}
        if isinstance(rows, dict):
            self.pose_stats = {
                str(key): PoseActionStats.from_dict(value if isinstance(value, dict) else {})
                for key, value in rows.items()
            }

    def record(
        self,
        action_kind: str,
        object_id: str,
        pose: InteractionPose,
        *,
        success: bool,
        error: str | None = None,
        recovery: bool = False,
    ) -> None:
        key = self._pose_key(action_kind, object_id, pose)
        row = self.pose_stats.setdefault(key, PoseActionStats())
        row.attempts += 1
        row.successes += int(bool(success))
        row.failures += int(not bool(success))
        row.recoveries += int(bool(recovery))
        if error:
            row.last_error = str(error)[:500]
        self.updated_at = utc_now()
        self.save()

    def stats(self, action_kind: str, object_id: str, pose: InteractionPose) -> PoseActionStats:
        return self.pose_stats.get(self._pose_key(action_kind, object_id, pose), PoseActionStats())

    def penalty(
        self,
        action_kind: str,
        object_id: str,
        pose: InteractionPose,
        *,
        object_position: dict[str, float] | None = None,
        preferred_open_clearance_m: float = 1.0,
    ) -> float:
        """Return an additive ranking penalty; lower is better.

        Exact pose experience dominates.  For open/close actions only, a soft
        clearance prior prefers poses around >= preferred_open_clearance_m because
        an agent can be visible/in range while still blocking a moving door/drawer.
        It is a preference, never a hard exclusion, so narrow scenes still work.
        """

        row = self.stats(action_kind, object_id, pose)
        penalty = 0.0
        if row.attempts:
            # One observed all-failure pose should immediately become unattractive,
            # while repeated success should make it slightly preferable.
            posterior_failure = (row.failures + 1.0) / (row.attempts + 2.0)
            penalty += posterior_failure * 45.0
            if row.failures and row.successes == 0:
                penalty += 80.0
            elif row.successes:
                penalty -= min(8.0, row.successes * 0.5)

        if action_kind in {"OpenObject", "CloseObject"} and object_position is not None:
            distance = planar_distance(pose.position, object_position)
            preferred = max(0.0, float(preferred_open_clearance_m))
            if distance < preferred:
                penalty += (preferred - distance) * 80.0
        return penalty

    def merge(self, other: "PoseActionabilityMemory") -> None:
        for key, stats in other.pose_stats.items():
            self.pose_stats.setdefault(key, PoseActionStats()).merge(stats)
        if other.updated_at:
            self.updated_at = other.updated_at

    def delta_from(self, baseline: "PoseActionabilityMemory") -> "PoseActionabilityMemory":
        """Return only outcomes accumulated beyond a seeded baseline."""
        delta = PoseActionabilityMemory(None, agent_label=self.agent_label, grid_size=self.grid_size)
        for key, current in self.pose_stats.items():
            old = baseline.pose_stats.get(key, PoseActionStats())
            row = PoseActionStats(
                attempts=max(0, current.attempts - old.attempts),
                successes=max(0, current.successes - old.successes),
                failures=max(0, current.failures - old.failures),
                recoveries=max(0, current.recoveries - old.recoveries),
                last_error=current.last_error if current.failures > old.failures else None,
            )
            if row.attempts or row.successes or row.failures or row.recoveries:
                delta.pose_stats[key] = row
        delta.updated_at = self.updated_at
        return delta

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "kind": "per_agent_interaction_pose_actionability",
            "agent_label": self.agent_label,
            "grid_size": self.grid_size,
            "updated_at": self.updated_at or utc_now(),
            "pose_stats": {key: stats.as_dict() for key, stats in sorted(self.pose_stats.items())},
        }

    def save(self) -> None:
        if self.path is None:
            return
        write_json_atomic(self.path, self.as_dict())
