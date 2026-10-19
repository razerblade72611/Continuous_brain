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
    """Append-only episodic memory with optional cross-run reload.

    Raw records remain controller-private. `safe_context()` deliberately exposes
    only self-authored/action/social summaries and never replays stored camera prose
    or simulator object IDs back into the embodied LLM prompt.
    """

    def __init__(
        self,
        path: str = "runs/memory.jsonl",
        max_records: int = 1000,
        load_existing: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self.records: list[MemoryRecord] = []
        if load_existing:
            self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        rows: list[MemoryRecord] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    data = raw.get("data")
                    if not isinstance(data, dict):
                        data = {}
                    rows.append(
                        MemoryRecord(
                            timestamp=str(raw.get("timestamp", "")),
                            kind=str(raw.get("kind", "unknown")),
                            data=data,
                        )
                    )
        except OSError:
            return
        self.records = rows[-self.max_records:]

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

    def safe_context(self, limit: int = 12) -> list[dict[str, Any]]:
        """Return compact cross-run memory safe for embodied LLM context.

        This intentionally excludes prior visual `observation` prose and internal
        grounding/verification metadata so persistence does not reintroduce the
        recursive-observation or omniscient-object-catalog problems fixed in v0.6.
        """
        out: list[dict[str, Any]] = []
        for record in self.records:
            kind = record.kind
            data = record.data
            row: dict[str, Any] | None = None
            if kind == "embodied_decision":
                actions = []
                for item in (data.get("semantic_actions") or [])[:6]:
                    if not isinstance(item, dict):
                        continue
                    actions.append({"kind": item.get("kind"), "target": item.get("target")})
                row = {
                    "kind": kind,
                    "task": data.get("task"),
                    "intention": data.get("intention"),
                    "attention": data.get("attention"),
                    "semantic_actions": actions,
                    "success": data.get("success"),
                }
            elif kind == "heard_speech":
                row = {
                    "kind": kind,
                    "speaker": data.get("speaker") or data.get("speaker_label"),
                    "text": data.get("text"),
                }
            elif kind == "user_task_complete":
                row = {"kind": kind, "task": data.get("task")}
            elif kind == "training_summary":
                row = {
                    "kind": kind,
                    "scene": data.get("scene"),
                    "episodes": data.get("episodes"),
                    "primitive_steps": data.get("primitive_steps"),
                    "map_cells_learned": data.get("map_cells_learned"),
                    "coverage_pct": data.get("coverage_pct"),
                    "peer_encounters": data.get("peer_encounters"),
                    "navigation_checkpoint_saved": data.get("navigation_checkpoint_saved"),
                }
            elif kind == "training_peer_encounter":
                row = {
                    "kind": kind,
                    "peer": data.get("peer"),
                    "distance_m": data.get("distance_m"),
                    "scene": data.get("scene"),
                }
            elif kind in {"sent_text", "received_text"}:
                row = {"kind": kind, **{k: data.get(k) for k in ("text", "to", "from", "channel") if k in data}}
            if row is not None:
                row["timestamp"] = record.timestamp
                out.append(row)
        return out[-max(0, int(limit)):]

    def failure_signatures(self, limit: int = 100) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records[-limit:]:
            if rec.kind != "action_failure":
                continue
            signature = str(rec.data.get("signature", "unknown"))
            out[signature] = out.get(signature, 0) + 1
        return out
