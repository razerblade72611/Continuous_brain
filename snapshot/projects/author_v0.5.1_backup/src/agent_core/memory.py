from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any


@dataclass
class MemoryRecord:
    timestamp: str
    kind: str
    data: dict[str, Any]


class EpisodicMemory:
    def __init__(self, path: str = "runs/memory.jsonl", max_records: int = 1000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.records: list[MemoryRecord] = []

    def add(self, kind: str, data: dict[str, Any]) -> None:
        record = MemoryRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            data=data,
        )
        self.records.append(record)
        self.records = self.records[-self.max_records:]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        return self.records[-limit:]

    def failure_signatures(self, limit: int = 100) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records[-limit:]:
            if rec.kind != "action_failure":
                continue
            signature = str(rec.data.get("signature", "unknown"))
            out[signature] = out.get(signature, 0) + 1
        return out
