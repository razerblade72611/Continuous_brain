from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
from typing import Any, Iterable


STATE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scene_slug(scene: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(scene))
    return value or "scene"


@dataclass(frozen=True)
class LearningPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "LearningPaths":
        return cls(Path(root).expanduser())

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def messages_dir(self) -> Path:
        return self.root / "messages"

    @property
    def maps_dir(self) -> Path:
        return self.root / "maps"

    @property
    def replay_dir(self) -> Path:
        return self.root / "replay"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def worker_dir(self) -> Path:
        return self.root / "workers"

    def ensure(self) -> "LearningPaths":
        for path in (
            self.root,
            self.memory_dir,
            self.messages_dir,
            self.maps_dir,
            self.replay_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.worker_dir,
            self.datasets_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def agent_memory(self, label: str) -> Path:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_") or "agent"
        return self.memory_dir / safe / "memory.jsonl"

    @property
    def peer_text_log(self) -> Path:
        return self.messages_dir / "peer_texts.jsonl"

    def scene_map(self, scene: str) -> Path:
        return self.maps_dir / f"{scene_slug(scene)}.json"

    @property
    def navigation_checkpoint(self) -> Path:
        return self.checkpoints_dir / "navigation_skill.json"

    @property
    def replay_file(self) -> Path:
        return self.replay_dir / "navigation_replay.jsonl"

    @property
    def embodied_replay_file(self) -> Path:
        return self.replay_dir / "embodied_replay.jsonl"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def embodied_sft_file(self) -> Path:
        return self.datasets_dir / "embodied_sft.jsonl"

    def agent_skill_checkpoint(self, label: str) -> Path:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_") or "agent"
        return self.checkpoints_dir / safe / "embodied_skills.json"

    def agent_actionability_checkpoint(self, label: str) -> Path:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_") or "agent"
        return self.checkpoints_dir / safe / "interaction_actionability.json"

    @property
    def training_summary(self) -> Path:
        return self.metrics_dir / "training_summary.json"

    @property
    def training_history(self) -> Path:
        return self.metrics_dir / "training_history.jsonl"


def _guard_reset_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    dangerous = {
        Path("/").resolve(),
        home,
        home.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    }
    if resolved in dangerous:
        raise ValueError(f"Refusing to reset dangerous learning-state path: {resolved}")
    # A normal state directory has several path components. This also prevents
    # accidental inputs such as /tmp or /home.
    if len(resolved.parts) < 4:
        raise ValueError(f"Refusing to reset overly broad path: {resolved}")
    return resolved


def reset_learning_state(root: str | Path) -> Path:
    """Delete only the configured persistent learning state directory.

    Run logs and source files live elsewhere and are intentionally untouched.
    """
    resolved = _guard_reset_path(Path(root))
    if resolved.exists():
        shutil.rmtree(resolved)
    LearningPaths.from_root(resolved).ensure()
    return resolved


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    if limit is not None:
        rows = rows[-max(0, int(limit)):]
    return rows


def rewrite_jsonl_tail(path: str | Path, max_records: int) -> int:
    target = Path(path)
    if max_records <= 0 or not target.exists():
        return 0
    rows = read_jsonl(target)
    rows = rows[-int(max_records):]
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, target)
    return len(rows)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return dict(default or {})
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default or {})
    return value if isinstance(value, dict) else dict(default or {})


def merge_jsonl_files(sources: Iterable[str | Path], destination: str | Path, max_records: int = 250_000) -> int:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dest.open("a", encoding="utf-8") as out:
        for source in sources:
            path = Path(source)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    rewrite_jsonl_tail(dest, max_records=max_records)
    return count
