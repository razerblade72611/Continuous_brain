"""Transactional autobiographical continuity store for v0.9.

The child model weights live in versioned checkpoint files.  Fast recurrent state
and each lived/teacher/imagined episode are committed together so an interrupted
run cannot advance one without the other.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


PROVENANCE = {"lived", "teacher", "imagined", "imported"}
KINDS = {"episodic", "semantic", "procedural"}


class ContinuityStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS episodes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created REAL NOT NULL,
            kind TEXT NOT NULL,
            provenance TEXT NOT NULL,
            action INTEGER,
            reward REAL,
            success INTEGER,
            content TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runtime_state(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            step INTEGER NOT NULL,
            hidden BLOB NOT NULL,
            workspace BLOB NOT NULL,
            model_ref TEXT NOT NULL,
            updated REAL NOT NULL
        );
        """)
        self.db.commit()

    @staticmethod
    def _tensor_blob(tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO(); torch.save(tensor.detach().cpu(), buf); return buf.getvalue()

    @staticmethod
    def _blob_tensor(blob: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)

    def commit_episode_and_state(self, *, kind: str, provenance: str, action: int,
                                 reward: float, success: bool, content: dict,
                                 step: int, hidden: torch.Tensor, workspace: torch.Tensor,
                                 model_ref: str) -> None:
        if kind not in KINDS: raise ValueError(kind)
        if provenance not in PROVENANCE: raise ValueError(provenance)
        with self.db:
            self.db.execute(
                "INSERT INTO episodes(created,kind,provenance,action,reward,success,content) VALUES(?,?,?,?,?,?,?)",
                (time.time(), kind, provenance, int(action), float(reward), int(bool(success)),
                 json.dumps(content, sort_keys=True)),
            )
            self.db.execute(
                "INSERT INTO runtime_state(singleton,step,hidden,workspace,model_ref,updated) VALUES(1,?,?,?,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET step=excluded.step,hidden=excluded.hidden," 
                "workspace=excluded.workspace,model_ref=excluded.model_ref,updated=excluded.updated",
                (int(step), self._tensor_blob(hidden), self._tensor_blob(workspace), str(model_ref), time.time()),
            )

    def runtime_state(self):
        row = self.db.execute("SELECT step,hidden,workspace,model_ref FROM runtime_state WHERE singleton=1").fetchone()
        if row is None: return None
        return {"step": row[0], "hidden": self._blob_tensor(row[1]), "workspace": self._blob_tensor(row[2]), "model_ref": row[3]}

    def recent_memory_vector(self, dim=64, limit=16) -> np.ndarray:
        rows = self.db.execute(
            "SELECT action,reward,success,provenance FROM episodes ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        out = np.zeros(int(dim), dtype=np.float32)
        if not rows: return out
        for i, (action, reward, success, provenance) in enumerate(rows):
            base = (i * 4) % max(4, dim - 4)
            out[base] += float((action or 0) / 4.0)
            out[base+1] += float(np.clip(reward or 0.0, -1, 1))
            out[base+2] += float(bool(success))
            out[base+3] += 1.0 if provenance == "lived" else 0.25
        out /= max(1, len(rows))
        return out

    def counts(self) -> dict:
        return {row[0]: row[1] for row in self.db.execute("SELECT provenance,COUNT(*) FROM episodes GROUP BY provenance")}

    def close(self):
        self.db.close()
