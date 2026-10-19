"""Attach a qualified v0.7.1 grounded-dialogue cortex to one continuation copy."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .lifetime import LifetimeStore
from .semantic import load_semantic_bundle


def attach_semantic_cortex(source, destination, semantic_lab):
    source, destination, semantic_lab = map(lambda p: Path(p).resolve(), (source, destination, semantic_lab))
    paths = (source, destination, semantic_lab)
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Source, destination and semantic lab must be separate, non-nested directories")
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new path. Originals are never overwritten")
    if not (source / "individual.sqlite3").is_file():
        raise FileNotFoundError("Source individual checkpoint not found")
    if (source / "UPGRADE_INCOMPLETE").exists():
        raise ValueError("Source is an incomplete upgrade; use the untouched original")
    qualified = semantic_lab / "qualified_semantic.pt"
    if not qualified.is_file():
        raise FileNotFoundError("No qualified_semantic.pt; run v0.7.1 semantic evaluation and pass first")
    _, bundle = load_semantic_bundle(qualified, require_qualified=True)

    original = LifetimeStore(source)
    target = None
    try:
        state = original.load()
        if not state or state.get("version") not in ("0.6.2", "0.7.0", "0.7.1"):
            raise ValueError("Expected a v0.6.2/v0.7.0/v0.7.1 continuation source")
        destination.mkdir(parents=True, exist_ok=False)
        marker = destination / "UPGRADE_INCOMPLETE"
        marker.write_text("Incomplete v0.7.1 semantic continuation copy. Originals are untouched.\n")
        target = LifetimeStore(destination)
        original.db.backup(target.db)
        for name in ("legacy_v053", "motor_history", "semantic_cortex"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name)

        semantic_dir = destination / "semantic_cortex"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        attached = semantic_dir / "qualified_semantic.pt"
        if attached.exists():
            history = semantic_dir / "history"
            history.mkdir(exist_ok=True)
            index = len(list(history.glob("semantic_*.pt")))
            attached.replace(history / f"semantic_{index:03d}.pt")
        shutil.copy2(qualified, attached)
        if (semantic_lab / "semantic_evaluation.json").is_file():
            shutil.copy2(semantic_lab / "semantic_evaluation.json", semantic_dir / "semantic_evaluation.json")

        metadata = bundle["metadata"]
        proof = {
            "kind": "qualified_grounded_dialogue_cortex_attachment",
            "version": "0.7.1",
            "digest": bundle["digest"],
            "attached_at_decision": state["self_model"]["decisions"],
            "parameters": int(sum(v.numel() for v in bundle["model"].values())),
            "dino_model": metadata.get("dino_model"),
            "teacher_models": metadata.get("teacher_models", []),
            "training_updates": metadata.get("training_updates", 0),
            "qualification": metadata.get("qualification", {}),
            "dialogue_intents": bundle.get("dialogue_intents", []),
            "runtime_qwen_required": False,
            "identity_grounding": "persistent_self_model.name",
            "goal_grounding": "persistent_intention.goal",
            "temporal_grounding": "latest_lifetime_experience_metadata",
        }
        lineage = dict(state["self_model"].get("lineage", {}))
        attachments = list(lineage.get("semantic_attachments", []))
        attachments.append(proof)
        state["self_model"]["lineage"] = {**lineage, "semantic_attachments": attachments}
        state["semantic_cortex"] = proof
        state["version"] = "0.7.1"
        target.append(state["self_model"]["ticks"], "semantic_attachment", provenance=proof)
        target.save(state)
        target.load()

        report = {
            "version": "0.7.1",
            "individual_id": state["self_model"]["individual_id"],
            "name": state["self_model"].get("name", "Continuous Brain"),
            "decisions": state["self_model"]["decisions"],
            "memories": original.count("experience"),
            "semantic_digest": bundle["digest"],
            "semantic_parameters": proof["parameters"],
            "dino_model": proof["dino_model"],
            "teacher_models": proof["teacher_models"],
            "runtime_qwen_required": False,
            "preserved": ["identity", "executive_weights", "motor", "workspace", "optimizer",
                          "target_network", "replay", "memories", "intentions", "world_state", "rng"],
            "source": str(source),
            "destination": str(destination),
            "note": "Same persistent individual continued with a qualified v0.7.1 grounded-dialogue cortex.",
        }
        (destination / "upgrade_report.json").write_text(json.dumps(report, indent=2))
        target.close(); target = None
        marker.unlink()
        return report
    finally:
        if target is not None:
            target.close()
        original.close()
