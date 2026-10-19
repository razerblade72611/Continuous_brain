from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .learning import append_jsonl, read_jsonl, rewrite_jsonl_tail, utc_now


@dataclass
class ReplayTransition:
    scene: str
    agent: str
    episode: int
    action: str
    before_position: dict[str, float]
    before_yaw: float
    after_position: dict[str, float]
    after_yaw: float
    target_position: dict[str, float]
    success: bool
    reward: float
    peer_training: bool = False
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if not row["timestamp"]:
            row["timestamp"] = utc_now()
        return row


class ReplayBuffer:
    def __init__(self, path: str | Path, max_records: int = 250_000):
        self.path = Path(path)
        self.max_records = int(max_records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._since_trim = 0

    def add(self, transition: ReplayTransition | dict[str, Any]) -> None:
        row = transition.as_dict() if isinstance(transition, ReplayTransition) else dict(transition)
        row.setdefault("timestamp", utc_now())
        append_jsonl(self.path, row)
        self._since_trim += 1
        if self._since_trim >= 5000:
            rewrite_jsonl_tail(self.path, self.max_records)
            self._since_trim = 0

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return read_jsonl(self.path, limit=limit)

    def finalize(self) -> None:
        rewrite_jsonl_tail(self.path, self.max_records)
        self._since_trim = 0
