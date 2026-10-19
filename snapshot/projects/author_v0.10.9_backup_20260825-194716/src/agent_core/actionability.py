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
            recoveries=int(payload.get("recoveries", payload.get("recovery_attempts", 0))),
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
    """Per-agent controller-private interaction reliability memory.

    v0.10.3 keeps exact pose experience, and also learns an X/Z position-level
    summary.  This lets one collision-like failure generalize across yaw/horizon
    variants without discarding orientation-specific evidence needed by pickup,
    toggle, and placement tasks.
    """

    VERSION = 2

    def __init__(self, path: str | Path | None, *, agent_label: str = "agent", grid_size: float = 0.25):
        self.path = None if path is None else Path(path)
        self.agent_label = str(agent_label)
        self.grid_size = float(grid_size)
        self.pose_stats: dict[str, PoseActionStats] = {}
        self.position_stats: dict[str, PoseActionStats] = {}
        self.updated_at: str | None = None
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _safe(value: str) -> str:
        return str(value).replace("|", "_").replace("\n", " ").strip()

    def _pose_key(self, action_kind: str, object_id: str, pose: InteractionPose) -> str:
        gx, gz, yaw, horizon = pose.key(self.grid_size)
        return "|".join((self._safe(action_kind), self._safe(object_id), str(gx), str(gz), str(yaw), str(horizon)))

    def _position_key(self, action_kind: str, object_id: str, pose: InteractionPose) -> str:
        gx, gz = pose.position_key(self.grid_size)
        return "|".join((self._safe(action_kind), self._safe(object_id), str(gx), str(gz)))

    @staticmethod
    def _position_key_from_pose_key(key: str) -> str | None:
        parts = str(key).split("|")
        if len(parts) < 4:
            return None
        return "|".join(parts[:4])

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
        position_rows = payload.get("position_stats") or {}
        if isinstance(position_rows, dict) and position_rows:
            self.position_stats = {
                str(key): PoseActionStats.from_dict(value if isinstance(value, dict) else {})
                for key, value in position_rows.items()
            }
        else:
            # Lossless v0.10.2 migration: aggregate its exact-pose history by X/Z.
            for key, stats in self.pose_stats.items():
                pkey = self._position_key_from_pose_key(key)
                if pkey:
                    self.position_stats.setdefault(pkey, PoseActionStats()).merge(stats)

    def record(self, action_kind: str, object_id: str, pose: InteractionPose, *, success: bool, error: str | None = None, recovery: bool = False) -> None:
        for key, table in (
            (self._pose_key(action_kind, object_id, pose), self.pose_stats),
            (self._position_key(action_kind, object_id, pose), self.position_stats),
        ):
            row = table.setdefault(key, PoseActionStats())
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

    def position_stats_for(self, action_kind: str, object_id: str, pose: InteractionPose) -> PoseActionStats:
        return self.position_stats.get(self._position_key(action_kind, object_id, pose), PoseActionStats())

    @staticmethod
    def _bayes_penalty(row: PoseActionStats, *, scale: float, all_failure_bonus: float) -> float:
        if not row.attempts:
            return 0.0
        posterior_failure = (row.failures + 1.0) / (row.attempts + 2.0)
        value = (posterior_failure - 0.5) * scale
        if row.failures and row.successes == 0:
            value += all_failure_bonus
        elif row.successes:
            value -= min(10.0, row.successes * 0.6)
        return value

    def penalty(self, action_kind: str, object_id: str, pose: InteractionPose, *, object_position: dict[str, float] | None = None, preferred_open_clearance_m: float = 1.0) -> float:
        exact = self.stats(action_kind, object_id, pose)
        position = self.position_stats_for(action_kind, object_id, pose)
        penalty = self._bayes_penalty(exact, scale=70.0, all_failure_bonus=80.0)
        penalty += self._bayes_penalty(position, scale=80.0, all_failure_bonus=70.0)
        if action_kind in {"OpenObject", "CloseObject"} and object_position is not None:
            distance = planar_distance(pose.position, object_position)
            preferred = max(0.0, float(preferred_open_clearance_m))
            if distance < preferred:
                penalty += (preferred - distance) * 100.0
        return penalty

    def merge(self, other: "PoseActionabilityMemory") -> None:
        for table_name in ("pose_stats", "position_stats"):
            mine = getattr(self, table_name)
            theirs = getattr(other, table_name)
            for key, stats in theirs.items():
                mine.setdefault(key, PoseActionStats()).merge(stats)
        if other.updated_at:
            self.updated_at = other.updated_at

    def delta_from(self, baseline: "PoseActionabilityMemory") -> "PoseActionabilityMemory":
        delta = PoseActionabilityMemory(None, agent_label=self.agent_label, grid_size=self.grid_size)
        for table_name in ("pose_stats", "position_stats"):
            current_table = getattr(self, table_name)
            old_table = getattr(baseline, table_name)
            out_table = getattr(delta, table_name)
            for key, current in current_table.items():
                old = old_table.get(key, PoseActionStats())
                row = PoseActionStats(
                    attempts=max(0, current.attempts - old.attempts),
                    successes=max(0, current.successes - old.successes),
                    failures=max(0, current.failures - old.failures),
                    recoveries=max(0, current.recoveries - old.recoveries),
                    last_error=current.last_error if current.failures > old.failures else None,
                )
                if row.attempts or row.successes or row.failures or row.recoveries:
                    out_table[key] = row
        delta.updated_at = self.updated_at
        return delta

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "kind": "per_agent_interaction_actionability",
            "agent_label": self.agent_label,
            "grid_size": self.grid_size,
            "updated_at": self.updated_at or utc_now(),
            "pose_stats": {key: stats.as_dict() for key, stats in sorted(self.pose_stats.items())},
            "position_stats": {key: stats.as_dict() for key, stats in sorted(self.position_stats.items())},
        }

    def save(self) -> None:
        if self.path is None:
            return
        write_json_atomic(self.path, self.as_dict())
