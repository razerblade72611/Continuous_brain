"""Append-only episodic memory (the hippocampus analogue).

Every backbone consultation is logged as an episode. Nothing is ever edited in
place: later information (an outcome, a correction, the fact that sleep
consolidated it) is appended as a separate record and merged on read.

Layout under <root>/episodes/:
  episodes.jsonl   one JSON record per line: "episode" | "annotation" | "consolidated"
  arrays/<id>.npz  the episode's tensors, written atomically before its record

An episode stores what sleep needs to replay it without re-running the core:
the pooled population features at consultation time, the retrieval key, the
write-back policy input and the current actually injected, and the input
currents the core received around the consultation.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..config import MemoryConfig

ARRAY_FIELDS = ("features", "key", "wb_summary", "wb_current", "core_inputs")


@dataclass
class Episode:
    id: str
    t_wall: float
    t_sim: float
    query: str          # what prompted the consultation (message or internal cue)
    prompt: str         # the full prompt actually given to the backbone (incl. recalled context)
    response: str
    cycle_s: float
    novelty: float
    target: str | None = None
    outcome: float | None = None
    explicit: bool = False
    salience: float = 0.0
    consolidated_in: list[str] = field(default_factory=list)
    arrays: dict[str, np.ndarray] | None = None   # loaded lazily


class EpisodicStore:
    def __init__(self, root: str | os.PathLike, config: MemoryConfig | None = None):
        self.cfg = config or MemoryConfig()
        self.dir = Path(root) / "episodes"
        self.arrays_dir = self.dir / "arrays"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "episodes.jsonl"
        self._episodes: dict[str, Episode] = {}
        self._order: list[str] = []
        self._keys: dict[str, np.ndarray] = {}
        if self.log_path.exists():
            self._replay_log()

    # ------------------------------------------------------------------ internals

    def _append(self, record: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _apply(self, rec: dict) -> None:
        kind = rec["type"]
        if kind == "episode":
            ep = Episode(id=rec["id"], t_wall=rec["t_wall"], t_sim=rec["t_sim"],
                         query=rec.get("query", rec["prompt"]), prompt=rec["prompt"],
                         response=rec["response"], cycle_s=rec["cycle_s"], novelty=rec["novelty"],
                         explicit=rec.get("explicit", False))
            self._episodes[ep.id] = ep
            self._order.append(ep.id)
            with np.load(self.arrays_dir / f"{ep.id}.npz") as z:
                self._keys[ep.id] = z["key"].astype(np.float32)
        elif kind == "annotation":
            ep = self._episodes[rec["id"]]
            for k in ("outcome", "target", "explicit"):
                if k in rec and rec[k] is not None:
                    setattr(ep, k, rec[k])
        elif kind == "consolidated":
            for eid in rec["ids"]:
                self._episodes[eid].consolidated_in.append(rec["sleep_id"])
        else:
            raise ValueError(f"unknown episode-log record type {kind!r}")
        for eid in ([rec["id"]] if "id" in rec else rec.get("ids", [])):
            self._rescore(self._episodes[eid])

    def _replay_log(self) -> None:
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._apply(json.loads(line))

    def _rescore(self, ep: Episode) -> None:
        c = self.cfg
        ep.salience = (c.w_outcome * abs(ep.outcome or 0.0) + c.w_novelty * ep.novelty
                       + c.w_explicit * float(ep.explicit))

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return b @ a

    # ------------------------------------------------------------------ writing

    def add(self, *, query: str, prompt: str, response: str, t_sim: float, cycle_s: float,
            features: np.ndarray, key: np.ndarray, wb_summary: np.ndarray, wb_current: np.ndarray,
            core_inputs: np.ndarray, explicit: bool = False) -> str:
        key = np.asarray(key, dtype=np.float32)
        if self._keys:
            sims = self._cos(key, np.stack(list(self._keys.values())))
            novelty = float(np.clip(1.0 - sims.max(), 0.0, 1.0))
        else:
            novelty = 1.0
        eid = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        arrays = {"features": np.asarray(features, np.float32), "key": key,
                  "wb_summary": np.asarray(wb_summary, np.float32),
                  "wb_current": np.asarray(wb_current, np.float32),
                  "core_inputs": np.asarray(core_inputs, np.float32)}
        tmp = self.arrays_dir / f".{eid}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self.arrays_dir / f"{eid}.npz")
        rec = {"type": "episode", "id": eid, "t_wall": time.time(), "t_sim": t_sim,
               "query": query, "prompt": prompt, "response": response, "cycle_s": cycle_s,
               "novelty": novelty, "explicit": explicit}
        self._append(rec)
        self._apply(rec)
        return eid

    def annotate(self, episode_id: str, *, outcome: float | None = None, target: str | None = None,
                 explicit: bool | None = None) -> None:
        if episode_id not in self._episodes:
            raise KeyError(episode_id)
        rec = {"type": "annotation", "id": episode_id, "outcome": outcome, "target": target,
               "explicit": explicit}
        self._append(rec)
        self._apply(rec)

    def mark_consolidated(self, ids: list[str], sleep_id: str) -> None:
        if ids:
            rec = {"type": "consolidated", "ids": list(ids), "sleep_id": sleep_id}
            self._append(rec)
            self._apply(rec)

    # ------------------------------------------------------------------ reading

    def __len__(self) -> int:
        return len(self._order)

    def all(self) -> list[Episode]:
        return [self._episodes[i] for i in self._order]

    def get(self, episode_id: str) -> Episode:
        ep = self._episodes[episode_id]
        with np.load(self.arrays_dir / f"{episode_id}.npz") as z:
            arrays = {k: z[k] for k in ARRAY_FIELDS}
        return replace(ep, arrays=arrays, consolidated_in=list(ep.consolidated_in))

    def search(self, key: np.ndarray, k: int, min_similarity: float = 0.0) -> list[tuple[Episode, float]]:
        if not self._keys or k <= 0:
            return []
        ids = list(self._keys)
        sims = self._cos(np.asarray(key, np.float32), np.stack([self._keys[i] for i in ids]))
        order = np.argsort(-sims)[:k]
        return [(self._episodes[ids[j]], float(sims[j])) for j in order if sims[j] >= min_similarity]

    def select_for_sleep(self, n: int) -> list[Episode]:
        pending = [e for e in self.all() if not e.consolidated_in]
        pending.sort(key=lambda e: e.salience, reverse=True)
        return [self.get(e.id) for e in pending[:n]]
