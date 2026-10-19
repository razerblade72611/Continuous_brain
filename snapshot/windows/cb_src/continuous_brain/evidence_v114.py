"""Strict evidence and artifact helpers shared by the trainer and Colab runner."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil

BUILD_ID = "v114-protected-integration-r2"
EXACT_SOURCE_SHA = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
STAGES = ("shadow", "micro", "low", "moderate")
TARGETS = dict(zip(STAGES, (2000, 1000, 1000, 1000)))
GATES = dict(zip(STAGES, (0.0, 0.02, 0.05, 0.10)))
P4B_PROTOCOL = dict(paired_episodes=100, steps_per_episode=120,
                    seed_start=12001, seed_end=12100, option_depth=2,
                    selector_topk=2, beam_width=8, max_option_steps=12,
                    discount=0.95, min_commit=3, terminate_threshold=0.5,
                    training_performed_during_evaluation=False, parents_used=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def copy_verified(source, destination):
    """Replace one file only after its temporary copy verifies; never delete a tree."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return sha256(source)
    digest = sha256(source)
    if destination.exists() and sha256(destination) == digest:
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, tmp)
    if sha256(tmp) != digest:
        raise RuntimeError(f"Copy verification failed: {destination.name}")
    os.replace(tmp, destination)
    return digest


def extract_outer_final_json(text):
    """Decode complete outer objects once; skip their nested objects."""
    decoder = json.JSONDecoder()
    i, last = 0, None
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            last = obj
        i = end
    if last is None:
        raise ValueError("No complete outer JSON result found")
    return last


def valid_p4b(result, checkpoint_sha, *, require_pass=True):
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        return False
    if result.get("checkpoint_sha256") != checkpoint_sha:
        return False
    exp = result.get("experiment") or {}
    if any(k not in exp or exp[k] != v for k, v in P4B_PROTOCOL.items()):
        return False
    if not require_pass:
        return True
    h = result.get("hierarchical_option_planning") or {}
    r = result.get("reactive") or {}
    try:
        hc, rc = float(h["completion_rate"]), float(r["completion_rate"])
        collision_delta = float(h["mean_collisions"]) - float(r["mean_collisions"])
    except (KeyError, TypeError, ValueError):
        return False
    return (result.get("p4b_passed") is True and math.isfinite(hc)
            and hc >= .20 and hc - rc > .05 and collision_delta <= 2.0)


def valid_training(result, stage):
    gates = result.get("gate_report") or {}
    try:
        gate_ok = all(math.isclose(float(gates[k]), GATES[stage], rel_tol=0, abs_tol=1e-7)
                      for k in ("planning.growth", "options.growth"))
        gate_ok = gate_ok and all(float(gates[k]) == 0.0
                                 for k in ("target_vision.growth", "target_spatial.growth"))
    except (KeyError, ValueError, TypeError):
        return False
    return bool(result.get("build_id") == BUILD_ID and result.get("stage") == stage
                and result.get("stage_updates_after") == TARGETS[stage]
                and result.get("legacy_tensors_changed") == 0
                and result.get("legacy_sha256") == result.get("source_legacy_sha256")
                and isinstance(result.get("legacy_sha256"), str)
                and len(result["legacy_sha256"]) == 64
                and result.get("plasticity") == "growth-only-protected"
                and len(result.get("checkpoint_sha256", "")) == 64 and gate_ok)


def safe_extract(archive, destination):
    """Read only regular archive members into a fresh staging directory."""
    import tarfile
    import zipfile
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def checked(name):
        path = (destination / name).resolve()
        if path != destination and destination not in path.parents:
            raise ValueError(f"Archive path escapes destination: {name}")
        return path
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                checked(member.filename)
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Archive symlinks are not supported")
            z.extractall(destination)
    else:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                checked(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive links and special files are not supported")
            # All members validated above; compatible with Python 3.10+.
            tar.extractall(destination, members=members)
