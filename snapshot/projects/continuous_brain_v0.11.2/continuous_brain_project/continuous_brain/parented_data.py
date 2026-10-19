"""Nursery lesson storage for parented development."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np


@dataclass
class NurseryManifest:
    format: str = "continuous_brain.nursery.v090"
    vision_parent_id: str = "vision.dinov3.base"
    vision_parent_dim: int = 768
    records: int = 0
    source: str = ""


class NurseryDataset:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.records_path = self.directory / "lessons.jsonl"
        self.images = self.directory / "images"
        self.parent_vision = self.directory / "parent_vision"
        self.manifest_path = self.directory / "manifest.json"
        self.images.mkdir(parents=True, exist_ok=True)
        self.parent_vision.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.records_path.exists():
            return []
        return [json.loads(line) for line in self.records_path.read_text().splitlines() if line.strip()]


    def append(self, row: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def rewrite(self, rows: list[dict]):
        tmp = self.records_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        tmp.replace(self.records_path)

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def write_manifest(self, data: dict):
        self.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def digest(self) -> str:
        h = hashlib.sha256()
        if self.records_path.exists():
            h.update(self.records_path.read_bytes())
        for p in sorted(self.parent_vision.glob("*.npy")):
            h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()

    def import_semantic(self, source: str | Path, *, parent_id="vision.dinov3.legacy") -> dict:
        """Convert v0.7 developmental evidence into lessons; no child weights are imported."""
        source = Path(source)
        rows = [json.loads(x) for x in (source / "records.jsonl").read_text().splitlines() if x.strip()]
        out = []
        visual_dim = None
        for i, row in enumerate(rows):
            record_id = f"legacy-{i:07d}"
            src_img = source / row["image"]
            dst_img = self.images / f"{record_id}{src_img.suffix.lower()}"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            src_vec = source / "dino_cache" / f"{row['record_id']}.npy"
            if not src_vec.exists():
                raise FileNotFoundError(src_vec)
            vec = np.load(src_vec).astype(np.float32)
            visual_dim = int(vec.size) if visual_dim is None else visual_dim
            if int(vec.size) != visual_dim:
                raise ValueError("mixed parent visual dimensions")
            np.save(self.parent_vision / f"{record_id}.npy", vec)
            temporal = row.get("temporal") or {}
            qas = list((row.get("teacher") or {}).get("qas") or [])
            action = 0
            skill = str(temporal.get("previous_skill") or "")
            if skill in {"navigate"}: action = 1
            elif skill in {"orient"}: action = 2
            elif skill in {"inspect"}: action = 4
            success = float(bool(temporal.get("previous_success", False)))
            reward = float(temporal.get("previous_reward", success * 0.1))
            care = (row.get("teacher") or {}).get("care") or {}
            care_vec = [float(care.get(k, 0.0)) for k in (
                "social_presence", "help_needed", "harm_risk", "respect_request", "uncertainty", "care_priority"
            )]
            out.append({
                "record_id": record_id,
                "image": str(dst_img.relative_to(self.directory)),
                "context": row.get("context", [0.0] * 48),
                "qas": qas,
                "action": action,
                "motor_valid": False,
                "reward": reward,
                "success": success,
                "care": care_vec,
                "parent_vision": str((self.parent_vision / f"{record_id}.npy").relative_to(self.directory)),
                "world_valid": False,
                "source": row.get("source", "legacy-semantic"),
                "provenance": "parent_lesson_import",
            })
        self.rewrite(out)
        manifest = {
            "format": "continuous_brain.nursery.v090",
            "vision_parent_id": parent_id,
            "vision_parent_dim": visual_dim or 0,
            "records": len(out),
            "source": str(source),
            "digest": self.digest(),
        }
        self.write_manifest(manifest)
        return manifest

    def split(self, validation_fraction=0.1, seed=17):
        rows = self.records()
        ids = list(range(len(rows)))
        random.Random(seed).shuffle(ids)
        cut = max(1, int(len(ids) * validation_fraction)) if len(ids) > 1 else 0
        valid = set(ids[:cut])
        return ([r for i, r in enumerate(rows) if i not in valid],
                [r for i, r in enumerate(rows) if i in valid])


def reparent_vision(directory: str | Path, *, model_id: str, parent_id: str, device="cpu") -> dict:
    """Replace active visual teacher targets while preserving prior target provenance."""
    from .parent_clients import DinoVisionParent
    dataset = NurseryDataset(directory)
    rows = dataset.records()
    if not rows:
        raise RuntimeError("No nursery lessons to re-parent")
    parent = DinoVisionParent(model_id, device=device)
    sub = dataset.parent_vision / parent_id.replace(".", "__").replace("/", "_")
    sub.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        old = {
            "parent_id": dataset.manifest().get("vision_parent_id"),
            "path": row.get("parent_vision"),
        }
        history = list(row.get("parent_vision_history") or [])
        if old["path"]:
            history.append(old)
        out = sub / f"{row['record_id']}.npy"
        np.save(out, parent.embed(dataset.directory / row["image"]))
        row["parent_vision_history"] = history
        row["parent_vision"] = str(out.relative_to(dataset.directory))
        if i % 50 == 0:
            print(f"reparented_vision={i}")
    dataset.rewrite(rows)
    manifest = dataset.manifest()
    manifest.update({
        "vision_parent_id": parent_id,
        "vision_parent_model": model_id,
        "vision_parent_dim": int(parent.output_dim),
        "records": len(rows),
        "digest": dataset.digest(),
    })
    dataset.write_manifest(manifest)
    return manifest


def teach_language(directory: str | Path, *, endpoint: str, model: str, parent_id: str,
                   limit: int | None = None, timeout=300) -> dict:
    """Add grounded language lessons from a replaceable OpenAI-compatible parent service."""
    from .parent_clients import OpenAIChatParent
    dataset = NurseryDataset(directory)
    rows = dataset.records()
    if not rows:
        raise RuntimeError("No nursery lessons to teach")
    parent = OpenAIChatParent(endpoint, model, timeout=timeout)
    taught = 0
    for row in rows:
        history = list(row.get("language_parent_history") or [])
        if any(x.get("parent_id") == parent_id for x in history):
            continue
        answer = parent.ask(
            "You are teaching a developing embodied visual system. Describe only what is visibly present "
            "in this image in one short grounded sentence. Do not mention hidden state or speculate.",
            image_path=dataset.directory / row["image"], max_tokens=80,
        ).strip()
        if answer:
            qas = list(row.get("qas") or [])
            qas.append({
                "kind": "vision", "question": "What do you see?", "answer": answer,
                "parent_id": parent_id,
            })
            row["qas"] = qas
            history.append({"parent_id": parent_id, "model": model, "lesson": "grounded_visual_language"})
            row["language_parent_history"] = history
            taught += 1
            if taught % 10 == 0:
                print(f"language_parent_taught={taught}")
            if limit is not None and taught >= int(limit):
                break
    dataset.rewrite(rows)
    manifest = dataset.manifest()
    manifest.update({"language_parent_id": parent_id, "language_parent_model": model, "digest": dataset.digest()})
    dataset.write_manifest(manifest)
    return {"taught": taught, "records": len(rows), "parent_id": parent_id, "model": model, "digest": dataset.digest()}


def merge_nurseries(sources: list[str | Path], destination: str | Path) -> dict:
    """Merge compatible nurseries into a fresh destination, rewriting relative file paths.

    This is developmental replay, not model merging: every lesson retains its provenance
    while image/parent-target files are copied under source-specific prefixes.
    """
    destination = Path(destination)
    out = NurseryDataset(destination)
    if out.records():
        raise FileExistsError("Destination nursery already contains lessons")
    source_sets = [NurseryDataset(s) for s in sources]
    manifests = [d.manifest() for d in source_sets]
    if not manifests or any(not m for m in manifests):
        raise RuntimeError("Every source nursery must contain a manifest")
    active_ids = {m.get("vision_parent_id") for m in manifests}
    active_dims = {int(m.get("vision_parent_dim", 0)) for m in manifests}
    if len(active_ids) != 1 or len(active_dims) != 1:
        raise ValueError(f"Nurseries must share active vision parent id/dim; got ids={active_ids}, dims={active_dims}")

    merged = []
    for source_index, ds in enumerate(source_sets):
        prefix = f"s{source_index:02d}"
        for row_index, row0 in enumerate(ds.records()):
            row = json.loads(json.dumps(row0))
            new_id = f"{prefix}-{row.get('record_id', row_index)}"
            row["record_id"] = new_id

            def copy_rel(field: str, subdir: str):
                rel = row.get(field)
                if not rel:
                    return
                src = ds.directory / rel
                suffix = src.suffix
                dst = destination / subdir / f"{new_id}-{field}{suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                row[field] = str(dst.relative_to(destination))

            copy_rel("image", "images")
            copy_rel("next_image", "images")
            copy_rel("parent_vision", "parent_vision")
            copy_rel("next_parent_vision", "parent_vision")

            hist = []
            for h_index, item0 in enumerate(row.get("parent_vision_history") or []):
                item = dict(item0)
                rel = item.get("path")
                if rel:
                    src = ds.directory / rel
                    dst = destination / "parent_vision" / "history" / f"{new_id}-{h_index}{src.suffix}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    item["path"] = str(dst.relative_to(destination))
                hist.append(item)
            if hist:
                row["parent_vision_history"] = hist
            row["merged_from"] = str(ds.directory)
            merged.append(row)

    out.rewrite(merged)
    manifest = {
        "format": "continuous_brain.nursery.v092",
        "vision_parent_id": manifests[0]["vision_parent_id"],
        "vision_parent_dim": int(manifests[0]["vision_parent_dim"]),
        "records": len(merged),
        "source": "merged-developmental-replay",
        "sources": [str(Path(s)) for s in sources],
        "digest": out.digest(),
    }
    out.write_manifest(manifest)
    return manifest
