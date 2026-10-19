from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Deque

import numpy as np


@dataclass(slots=True)
class MemoryEvent:
    step: int
    reward: float
    surprise: float
    action: int
    note: str = ""


class WorkingMemory:
    def __init__(self, capacity: int = 128):
        self._events: Deque[MemoryEvent] = deque(maxlen=capacity)

    def append(self, event: MemoryEvent) -> None:
        self._events.append(event)

    def recent(self, n: int = 8) -> list[MemoryEvent]:
        return list(self._events)[-n:]


class EpisodicMemory:
    """Persistent SQLite memory with separate cognitive and visual latent stores.

    The original v0.1/v0.2 `episodes` schema is retained unchanged. v0.3 adds a
    second table for visual foundation-model embeddings, so old databases migrate
    automatically without rewriting prior memories.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                step INTEGER NOT NULL,
                reward REAL NOT NULL,
                surprise REAL NOT NULL,
                action INTEGER NOT NULL,
                latent BLOB NOT NULL,
                dim INTEGER NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                step INTEGER NOT NULL,
                reward REAL NOT NULL,
                novelty REAL NOT NULL,
                action INTEGER NOT NULL,
                backbone TEXT NOT NULL,
                latent BLOB NOT NULL,
                dim INTEGER NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def store(
        self,
        *,
        step: int,
        reward: float,
        surprise: float,
        action: int,
        latent: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(latent, dtype=np.float32).reshape(-1)
        self.db.execute(
            """
            INSERT INTO episodes(created, step, reward, surprise, action, latent, dim, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(), step, reward, surprise, action,
                vec.tobytes(), int(vec.size), json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self.db.commit()

    def store_visual(
        self,
        *,
        step: int,
        reward: float,
        novelty: float,
        action: int,
        backbone: str,
        latent: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(latent, dtype=np.float32).reshape(-1)
        self.db.execute(
            """
            INSERT INTO visual_episodes(created, step, reward, novelty, action, backbone, latent, dim, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(), step, reward, novelty, action, backbone,
                vec.tobytes(), int(vec.size), json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self.db.commit()

    def count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def visual_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM visual_episodes").fetchone()
        return int(row[0])

    @staticmethod
    def _cosine_rank(rows: list[tuple], q: np.ndarray, *, visual: bool, k: int) -> list[dict[str, Any]]:
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q)) + 1e-8
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if visual:
                # id, created, step, reward, novelty, action, backbone, latent, dim, metadata
                vec = np.frombuffer(row[7], dtype=np.float32, count=row[8])
                if vec.shape != q.shape:
                    continue
                score = float(np.dot(q, vec) / (q_norm * (np.linalg.norm(vec) + 1e-8)))
                item = {
                    "id": row[0], "created": row[1], "step": row[2], "reward": row[3],
                    "novelty": row[4], "action": row[5], "backbone": row[6],
                    "similarity": score, "metadata": json.loads(row[9]),
                }
            else:
                # id, created, step, reward, surprise, action, latent, dim, metadata
                vec = np.frombuffer(row[6], dtype=np.float32, count=row[7])
                if vec.shape != q.shape:
                    continue
                score = float(np.dot(q, vec) / (q_norm * (np.linalg.norm(vec) + 1e-8)))
                item = {
                    "id": row[0], "created": row[1], "step": row[2], "reward": row[3],
                    "surprise": row[4], "action": row[5], "similarity": score,
                    "metadata": json.loads(row[8]),
                }
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def retrieve(self, query: np.ndarray, k: int = 5, scan_limit: int = 2000) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT id, created, step, reward, surprise, action, latent, dim, metadata
            FROM episodes ORDER BY id DESC LIMIT ?
            """,
            (scan_limit,),
        ).fetchall()
        return self._cosine_rank(rows, query, visual=False, k=k)

    def retrieve_visual(self, query: np.ndarray, k: int = 5, scan_limit: int = 2000) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT id, created, step, reward, novelty, action, backbone, latent, dim, metadata
            FROM visual_episodes ORDER BY id DESC LIMIT ?
            """,
            (scan_limit,),
        ).fetchall()
        return self._cosine_rank(rows, query, visual=True, k=k)
