"""Transactional lifetime memory and explicit, modest self-model.

Checkpoint and newly appended memories commit in ONE SQLite transaction. A crash
therefore loses only the uncommitted interval, never mismatches memories and state.
SQLite on a local Linux/WSL filesystem is required; do not train on /mnt/c or NFS.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np
import torch
from .skills import SKILLS


class LifetimeStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "individual.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("This individual is already running; use a different state directory")
        try:
            self.db = sqlite3.connect(self.directory / "individual.sqlite3")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS checkpoint (
                    id INTEGER PRIMARY KEY CHECK(id=1), format INTEGER, digest TEXT, payload BLOB);
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL,
                    kind TEXT NOT NULL, key BLOB, value BLOB, metadata TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS experience_kind ON experiences(kind, id);
            ''')
            self.db.commit()
        except Exception:
            self.lock.close()
            raise

    def load(self):
        row = self.db.execute("SELECT format,digest,payload FROM checkpoint WHERE id=1").fetchone()
        if row is None:
            if self.count():
                raise RuntimeError("Memory exists without a checkpoint; refusing to create a replacement identity")
            return None
        version, digest, data = row
        if version != 1 or hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Checkpoint integrity/version failure; preserve state and restore a backup")
        # This is the user's own local checkpoint, not an untrusted downloaded model.
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    def save(self, payload):
        out = io.BytesIO()
        torch.save(payload, out)
        data = out.getvalue()
        self.db.execute("INSERT OR REPLACE INTO checkpoint VALUES (1,1,?,?)",
                        (hashlib.sha256(data).hexdigest(), data))
        self.db.commit()

    def append(self, tick, kind, key=None, value=None, **metadata):
        def pack(x):
            return None if x is None else np.asarray(x, dtype=np.float32).tobytes()
        self.db.execute("INSERT INTO experiences(tick,kind,key,value,metadata) VALUES (?,?,?,?,?)",
                        (int(tick), kind, pack(key), pack(value), json.dumps(metadata, allow_nan=False)))

    def retrieve(self, query, k, memory_dim, scan_limit=2000):
        """Stable sensory keys, not obsolete hidden states; no diagnostic retrieval."""
        recent = self.db.execute("SELECT id,key,value FROM experiences WHERE kind='experience' "
                                 "ORDER BY id DESC LIMIT ?", (max(k, 3*scan_limit//4),)).fetchall()
        formative = self.db.execute("SELECT id,key,value FROM experiences WHERE kind='experience' "
                                    "ORDER BY id ASC LIMIT ?", (max(k, scan_limit//4),)).fetchall()
        rows = list({row[0]: row for row in recent + formative}.values())
        q = np.asarray(query, dtype=np.float32)
        scored = []
        for event_id, key, value in rows:
            key = np.frombuffer(key, dtype=np.float32)
            vec = np.frombuffer(value, dtype=np.float32)
            if key.shape != q.shape or vec.size != memory_dim:
                continue
            score = float(np.dot(q, key) / (max(np.linalg.norm(q), 1e-8) * max(np.linalg.norm(key), 1e-8)))
            scored.append((score, event_id, vec))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result = np.zeros((k, memory_dim), dtype=np.float32)
        ids = []
        for i, (_, event_id, vec) in enumerate(scored[:k]):
            result[i] = vec
            ids.append(event_id)
        return result, ids

    def count(self, kind=None):
        if kind is None:
            return self.db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM experiences WHERE kind=?", (kind,)).fetchone()[0]

    def close(self):
        if getattr(self, "db", None) is not None:
            self.db.close()  # uncommitted experiments roll back intentionally
            self.db = None
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()


@dataclass
class SelfModel:
    individual_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: float = field(default_factory=time.time)
    name: str = "Continuous Brain"
    ticks: int = 0
    decisions: int = 0
    tasks: int = 0
    successes: int = 0
    faults: int = 0
    energy: float = 1.0
    surprise: float = 0.0
    uncertainty: float = 1.0
    learning_progress: float = 0.0
    boundary: bool = False
    last_success: bool = False
    # Empirical source-labelled estimates, not invented personality claims.
    skill_stats: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)

    def vector(self, intention_age=0):
        capabilities = [self.skill_stats.get(skill.name, {}).get("confidence", 0.5) for skill in SKILLS]
        costs = [min(1.0, self.skill_stats.get(skill.name, {}).get("mean_cost", 0.0)/8) for skill in SKILLS]
        return np.array([
            self.energy, np.clip(self.surprise, 0, 1), np.clip(self.uncertainty, 0, 1),
            np.clip(self.learning_progress, -1, 1), self.successes / max(1, self.tasks),
            self.faults / max(1, self.decisions), float(self.boundary), float(self.last_success),
            min(1.0, intention_age / 100), min(1.0, self.decisions / 10000),
            min(1.0, len(self.skill_stats) / 6), 1.0, *capabilities, *costs,
        ], dtype=np.float32)

    def observe_outcome(self, skill, success, cost, failure="timeout"):
        entry = self.skill_stats.setdefault(skill, {"attempts": 0, "successes": 0,
            "total_cost": 0, "failures": {}, "source": "observed_outcome"})
        entry["attempts"] += 1
        entry["successes"] += int(success)
        entry["total_cost"] += int(cost)
        if not success:
            entry["failures"][failure] = entry["failures"].get(failure, 0) + 1
        entry["confidence"] = (entry["successes"] + 1) / (entry["attempts"] + 2)
        entry["mean_cost"] = entry["total_cost"] / entry["attempts"]

    def state_dict(self):
        return asdict(self)
