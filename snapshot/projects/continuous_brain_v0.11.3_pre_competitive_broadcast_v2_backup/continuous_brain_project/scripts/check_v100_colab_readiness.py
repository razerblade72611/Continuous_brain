#!/usr/bin/env python3
"""Check the precommitted v0.10.0 gates for moving to major Colab training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-checkpoint", required=True,
                   help="Frozen v0.9.6 P4a source checkpoint used to create the v0.10 branch")
    p.add_argument("--current-checkpoint", required=True,
                   help="Trained v0.10 checkpoint")
    p.add_argument("--planning-result", required=True,
                   help="Fresh 100-episode P4b v0.10 JSON")
    p.add_argument("--p3-result", required=True,
                   help="P3 final blind qualification JSON rerun on v0.10")
    p.add_argument("--retention-result", required=True,
                   help="Bootstrap retention qualification JSON rerun on v0.10")
    args = p.parse_args()

    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    current = torch.load(args.current_checkpoint, map_location="cpu", weights_only=False)
    changed = []
    missing = []
    for key, before in source["model"].items():
        if key.startswith("planning."):
            continue
        if key not in current["model"]:
            missing.append(key)
        elif not torch.equal(before, current["model"][key]):
            changed.append(key)
    legacy_frozen = not changed and not missing

    planning = load_json(args.planning_result)
    p3 = load_json(args.p3_result)
    retention = load_json(args.retention_result)
    meta = dict(current.get("metadata") or {})

    planning_pass = bool(planning.get("p4b_passed"))
    p3_pass = bool(p3.get("passed")) and bool(p3.get("world_model_passed"))
    caps = retention.get("capability_scores") or {}
    metrics = retention.get("metrics") or {}
    historical = metrics.get("vision_retention_min")
    retention_pass = (
        bool(retention.get("passed"))
        and float(caps.get("vision", 0.0)) >= 0.90
        and float(caps.get("language", 0.0)) >= 0.85
        and (historical is None or float(historical) >= 0.90)
    )
    planner_trained = meta.get("planning_cortex_version") == "recurrent-latent-v1" and int(meta.get("planning_updates", 0)) > 0

    gates = {
        "p4b_blind_planning": planning_pass,
        "p3_world_model_retained": p3_pass,
        "bootstrap_retention": retention_pass,
        "legacy_tensors_bit_exact": legacy_frozen,
        "planning_cortex_trained": planner_trained,
    }
    result = {
        "format": "continuous_brain.colab_readiness.v100",
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_sha256": sha(args.source_checkpoint),
        "current_checkpoint": str(Path(args.current_checkpoint).resolve()),
        "current_sha256": sha(args.current_checkpoint),
        "changed_legacy_tensor_count": len(changed),
        "missing_legacy_tensor_count": len(missing),
        "changed_legacy_tensors_first_20": changed[:20],
        "missing_legacy_tensors_first_20": missing[:20],
        "gates": gates,
        "retention_scores": {
            "vision": float(caps.get("vision", 0.0)),
            "language": float(caps.get("language", 0.0)),
            "historical_vision_min": None if historical is None else float(historical),
        },
        "colab_ready": all(gates.values()),
        "meaning": (
            "Ready for the first major whole-brain Colab development run on the current v0.10 architecture. "
            "This is an engineering readiness gate, not evidence of consciousness or AGI."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["colab_ready"] else 2)


if __name__ == "__main__":
    main()
