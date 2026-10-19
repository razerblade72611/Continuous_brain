"""Checkpointing and reproducibility manifest (DAV-36).

A checkpoint is a never-overwrite snapshot (see snapshot.py) holding:
  state.pt     engine state: membranes, E/I drive, refractory, thresholds,
               homeostatic rate estimates, traces, all connectivity slots,
               noise RNG state, step counter
  manifest     schema version, developmental stage, seed, full config,
               config hash, anatomy hash, connectivity hash, state hash,
               code revision, environment, Linear issue

Restoring rebuilds the anatomy from the manifest's config and refuses to load
if its hash differs from the recorded one, so a checkpoint can never be
silently attached to a different body. Continuing from a restored checkpoint
reproduces the original trajectory bit-exactly on the same device/backend.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .anatomy import build_anatomy
from .config import SCHEMA_VERSION, E0Config
from .engine import SNNEngine
from .provenance import code_revision, environment
from .snapshot import SnapshotStore


class CheckpointMismatchError(RuntimeError):
    pass


def save_checkpoint(engine: SNNEngine, store: SnapshotStore, label: str = "e0",
                    linear_issue: str | None = None) -> Path:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": engine.cfg.stage,
        "seed": engine.cfg.seed,
        "config": engine.cfg.to_dict(),
        "config_hash": engine.cfg.config_hash(),
        "anatomy_hash": engine.anatomy.anatomy_hash(),
        "connectivity_hash": engine.slots.connectivity_hash(),
        "state_hash": engine.state_hash(),
        "step": engine.steps,
        "time_s": engine.time_s,
        "batch_size": engine.batch_size,
        "device_saved_on": str(engine.device),
        "code": code_revision(),
        "env": environment(),
        "linear_issue": linear_issue,
    }
    return store.save_state({"state": engine.state_dict(), "manifest": manifest}, label=label,
                            meta={"stage": engine.cfg.stage, "step": engine.steps,
                                  "anatomy_hash": manifest["anatomy_hash"], "linear_issue": linear_issue})


def load_checkpoint(path: str | os.PathLike, store: SnapshotStore,
                    device: torch.device | str = "cpu") -> SNNEngine:
    manifest = store.load_state(path, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CheckpointMismatchError(f"schema {manifest['schema_version']} != {SCHEMA_VERSION}")
    cfg = E0Config.from_dict(manifest["config"])
    if cfg.config_hash() != manifest["config_hash"]:
        raise CheckpointMismatchError("config does not reproduce its recorded hash")
    anatomy = build_anatomy(cfg)
    if anatomy.anatomy_hash() != manifest["anatomy_hash"]:
        raise CheckpointMismatchError("rebuilt anatomy differs from the checkpoint's anatomy")
    engine = SNNEngine(cfg, device=device, batch_size=manifest["batch_size"], anatomy=anatomy)
    engine.load_state_dict(store.load_state(path, "state"))
    if engine.slots.connectivity_hash() != manifest["connectivity_hash"]:
        raise CheckpointMismatchError("restored connectivity differs from its recorded hash")
    if str(torch.device(device)) == manifest["device_saved_on"] and engine.state_hash() != manifest["state_hash"]:
        raise CheckpointMismatchError("restored state differs from its recorded hash")
    return engine
