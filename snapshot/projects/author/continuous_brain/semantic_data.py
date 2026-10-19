"""Dataset utilities for v0.7 developmental semantic training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch

from .semantic import BASIC_CONCEPTS, CARE_KEYS, ByteCodec, pad_ids
from .vision import PixelStatsBackbone


@dataclass
class SemanticRecord:
    record_id: str
    image: str
    context: list[float]
    source: str
    provenance: str = "lived_training_environment"
    teacher: dict | None = None
    evaluator: dict | None = None

    def validate(self):
        if len(self.context) != 48 or not np.isfinite(self.context).all():
            raise ValueError("semantic record context must contain 48 finite values")
        if self.provenance not in {"lived_training_environment", "teacher_counterfactual", "replay"}:
            raise ValueError("unknown developmental provenance")
        return self


class SemanticDataset:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.records_path = self.directory / "records.jsonl"
        self.images = self.directory / "images"
        self.cache = self.directory / "dino_cache"
        self.images.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.records_path.exists():
            return []
        out = []
        for line in self.records_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                SemanticRecord(**{k: row.get(k) for k in SemanticRecord.__dataclass_fields__}).validate()
                out.append(row)
        return out

    def rewrite(self, rows: list[dict]):
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.records_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        tmp.replace(self.records_path)

    def append(self, row: SemanticRecord | dict):
        if isinstance(row, SemanticRecord):
            data = row.validate().__dict__
        else:
            data = dict(row)
            SemanticRecord(**{k: data.get(k) for k in SemanticRecord.__dataclass_fields__}).validate()
        with self.records_path.open("a") as f:
            f.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")

    def digest(self) -> str:
        h = hashlib.sha256()
        for row in self.records():
            stable = dict(row)
            h.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())
            image = self.directory / row["image"]
            if image.exists():
                h.update(hashlib.sha256(image.read_bytes()).digest())
        return h.hexdigest()

    def cache_path(self, record_id: str) -> Path:
        return self.cache / f"{record_id}.npy"

    def image_path(self, row: dict) -> Path:
        return self.directory / row["image"]


def neutral_self_vector(energy=1.0) -> np.ndarray:
    out = np.zeros(24, dtype=np.float32)
    out[0] = float(np.clip(energy, 0, 1))
    out[2] = 0.5
    out[11] = 1.0
    return out


def make_context(obs, *, self_vector=None, goal=0, last_skill=None, memory_summary=None) -> np.ndarray:
    """48-value student context with no simulator geometry.

    6 sensed values + 24 empirical self values + 4-way goal + 6-way last-skill
    + 8-value latest-memory summary.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size != 6:
        raise ValueError("context requires six structured sensory values")
    self_vector = neutral_self_vector() if self_vector is None else np.asarray(self_vector, dtype=np.float32)
    if self_vector.shape != (24,):
        raise ValueError("self vector must have 24 values")
    goal_onehot = np.zeros(4, dtype=np.float32)
    goal_onehot[int(goal) % 4] = 1.0
    skill = np.zeros(6, dtype=np.float32)
    if last_skill is not None and 0 <= int(last_skill) < 6:
        skill[int(last_skill)] = 1.0
    memory = np.zeros(8, dtype=np.float32) if memory_summary is None else np.asarray(memory_summary, dtype=np.float32)
    if memory.shape != (8,):
        raise ValueError("memory summary must have eight values")
    out = np.concatenate((obs, self_vector, goal_onehot, skill, memory)).astype(np.float32)
    if out.shape != (48,) or not np.isfinite(out).all():
        raise ValueError("invalid communication context")
    return out


def teacher_concept_vector(teacher: dict) -> np.ndarray:
    chosen = set(teacher.get("concepts", []))
    return np.asarray([1.0 if concept in chosen else 0.0 for concept in BASIC_CONCEPTS], dtype=np.float32)


def teacher_care_vector(teacher: dict) -> np.ndarray:
    care = teacher.get("care", {}) or {}
    return np.asarray([float(np.clip(care.get(key, 0.0), 0, 1)) for key in CARE_KEYS], dtype=np.float32)


def training_examples(dataset: SemanticDataset, *, split="train", validation_fraction=0.1,
                      seed=7) -> list[tuple[dict, dict]]:
    rows = [row for row in dataset.records() if row.get("teacher") and row["teacher"].get("qas")]
    rows.sort(key=lambda r: r["record_id"])
    rng = random.Random(seed)
    ids = list(range(len(rows)))
    rng.shuffle(ids)
    cut = max(1, int(round(len(ids) * validation_fraction))) if len(ids) > 1 else 0
    valid = set(ids[:cut])
    selected = [row for i, row in enumerate(rows) if (i in valid) == (split == "validation")]
    examples = []
    for row in selected:
        for qa in row["teacher"].get("qas", []):
            if qa.get("question") and qa.get("answer"):
                examples.append((row, qa))
    return examples


def collate_examples(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "dino": torch.tensor(np.stack([x["dino"] for x in items]), dtype=torch.float32),
        "context": torch.tensor(np.stack([x["context"] for x in items]), dtype=torch.float32),
        "question_ids": pad_ids([x["question_ids"] for x in items]),
        "answer_ids": pad_ids([x["answer_ids"] for x in items]),
        "concepts": torch.tensor(np.stack([x["concepts"] for x in items]), dtype=torch.float32),
        "care": torch.tensor(np.stack([x["care"] for x in items]), dtype=torch.float32),
        "pixel_target": torch.tensor(np.stack([x["pixel_target"] for x in items]), dtype=torch.float32),
    }


def load_example(dataset: SemanticDataset, row: dict, qa: dict) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Semantic training requires Pillow") from exc
    cache = dataset.cache_path(row["record_id"])
    if not cache.exists():
        raise FileNotFoundError(f"Missing DINO cache for {row['record_id']}; run cache-dino first")
    dino = np.load(cache).astype(np.float32)
    image = np.asarray(Image.open(dataset.image_path(row)).convert("RGB"), dtype=np.uint8)
    pixel = PixelStatsBackbone().encode(image)
    return {
        "dino": dino,
        "context": np.asarray(row["context"], dtype=np.float32),
        "question_ids": ByteCodec.encode(qa["question"], 192),
        "answer_ids": ByteCodec.encode(qa["answer"], 256),
        "concepts": teacher_concept_vector(row["teacher"]),
        "care": teacher_care_vector(row["teacher"]),
        "pixel_target": pixel,
    }
