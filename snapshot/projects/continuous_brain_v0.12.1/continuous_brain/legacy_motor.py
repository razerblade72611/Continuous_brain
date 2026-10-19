"""Validate trusted local legacy motors without guessing from tensor shape alone."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil

import torch

from .model import ContinuousBrainNet


def read_legacy_motor(source, vision):
    source = Path(source).resolve()
    path = source / "brain.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path}")
    raw = path.read_bytes()
    # Only use this on checkpoints you created/trust: pickle is executable data.
    data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("Legacy checkpoint must be a dictionary")
    state = data.get("environment_state", {})
    if not isinstance(state, dict):
        raise ValueError("Legacy environment_state must be a dictionary")
    modern = isinstance(state.get("scene"), str) and bool(state["scene"])
    # v0.4.3 format 6 predates the scene marker. These are its embodied
    # curriculum fields; ordinary foraging state with the same 6/5 tensors is
    # not sufficient. Mastery is recorded as evidence, not re-certified here.
    fields = ("edge_failures", "legacy_v041_stats", "level_action_steps",
              "level_totals", "recent_outcomes_by_level")
    legacy = (data.get("version") == "0.4.3" and data.get("format_version") == 6
              and "format_version" in state
              and all(isinstance(state.get(key), dict) for key in fields))
    if not (modern or legacy):
        raise ValueError("Cannot establish embodied motor provenance: expected a scene marker "
                         "or the v0.4.3 format-6 curriculum schema; foraging/unknown state rejected")
    weights = data.get("model")
    if not isinstance(weights, dict):
        raise ValueError("Legacy checkpoint has no model state dictionary")
    try:
        visual_dim = int(weights["visual_adapter.0.weight"].shape[1]) if "visual_adapter.0.weight" in weights else 0
        metadata = {"embed_dim": int(weights["perception.0.weight"].shape[0]),
                    "hidden_dim": int(weights["core.weight_hh"].shape[1]), "visual_dim": visual_dim}
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError("Legacy motor tensor schema is incomplete") from exc
    if visual_dim not in (0, vision["dim"]):
        raise ValueError("Imported motor visual dimension differs; use its original vision backbone")
    old_vision = data.get("vision")
    if visual_dim and old_vision:
        if not isinstance(old_vision, dict) or old_vision.get("name") != vision["name"]:
            raise ValueError("Imported motor vision backbone differs despite compatible dimensions")
        if old_vision.get("output_dim", visual_dim) != visual_dim:
            raise ValueError("Legacy visual dimension metadata disagrees with tensors")
        if old_vision.get("model_id") != vision.get("model_id"):
            raise ValueError("Imported motor visual model identifier differs")
    if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError("Legacy motor has nonfinite or non-tensor weights")
    # Strictly verify every key and shape. Validation must not consume the
    # persistent executive's random stream or reset any of its parameters.
    with torch.random.fork_rng(devices=[]):
        motor = ContinuousBrainNet(6, 5, metadata["embed_dim"], metadata["hidden_dim"], visual_dim)
        motor.load_state_dict(weights, strict=True)
    provenance = {"source_version": data.get("version", "unknown"),
                  "legacy_steps": data.get("steps", 0), "legacy_goals": data.get("goals", 0),
                  "legacy_mastered_levels": data.get("mastered_levels", []),
                  "sha256": hashlib.sha256(raw).hexdigest(),
                  "validation": "v043_format6_curriculum" if legacy else "embodied_scene_marker",
                  "history_archive": "legacy_v053",
                  "legacy_memory_policy": "preserved_not_retrieved_unclassified"}
    return {"weights": weights, "metadata": metadata, "environment": state, "provenance": provenance}


def archive_legacy_motor(source, destination, expected_digest):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Legacy source and archive must be separate, non-nested directories")
    if destination.exists():
        raise FileExistsError("Legacy archive already exists; no files were overwritten")
    shutil.copytree(source, destination)
    for path in (source / "brain.pt", destination / "brain.pt"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise RuntimeError("Legacy checkpoint changed during archive; stop its writer before retrying")
