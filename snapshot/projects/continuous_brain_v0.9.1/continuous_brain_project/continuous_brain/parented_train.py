"""Training and qualification for the v0.9 raw-cortex child."""
from __future__ import annotations

from dataclasses import asdict
import difflib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .infant import (
    InfantBrain, InfantConfig, BYTE_PAD, byte_encode, one_hot_action,
)
from .parented_data import NurseryDataset


def _pad(seqs: list[list[int]], pad=BYTE_PAD) -> torch.Tensor:
    width = max(len(x) for x in seqs)
    out = torch.full((len(seqs), width), pad, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def _load_images(dataset: NurseryDataset, rows: list[dict], size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Parented training requires Pillow") from exc
    images = []
    for row in rows:
        image = Image.open(dataset.directory / row["image"]).convert("RGB").resize((size, size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images.append(np.transpose(arr, (2, 0, 1)))
    return torch.tensor(np.stack(images), dtype=torch.float32)


def _batch(dataset: NurseryDataset, rows: list[dict], cfg: InfantConfig, rng: random.Random) -> dict:
    images = _load_images(dataset, rows, cfg.image_size)
    context = torch.tensor(np.stack([np.asarray(r["context"], np.float32) for r in rows]), dtype=torch.float32)
    parent_vision = torch.tensor(np.stack([
        np.load(dataset.directory / r["parent_vision"]).astype(np.float32) for r in rows
    ]), dtype=torch.float32)

    # Preserve old parent targets as replay scaffolding.  A re-parented child should
    # learn from the new parent without catastrophically forgetting mappings that
    # earlier parents already helped establish.
    historical: dict[str, dict] = {}
    for row_index, row in enumerate(rows):
        for item in row.get("parent_vision_history") or []:
            pid, rel = item.get("parent_id"), item.get("path")
            if not pid or not rel:
                continue
            vec = np.load(dataset.directory / rel).astype(np.float32).reshape(-1)
            slot = historical.setdefault(pid, {"dim": int(vec.size), "values": [None] * len(rows)})
            if int(vec.size) != int(slot["dim"]):
                raise ValueError(f"mixed historical parent dimension for {pid}")
            slot["values"][row_index] = vec
    historical_parent_vision = {}
    for pid, slot in historical.items():
        values = np.zeros((len(rows), int(slot["dim"])), dtype=np.float32)
        mask = np.zeros((len(rows),), dtype=np.float32)
        for i, vec in enumerate(slot["values"]):
            if vec is not None:
                values[i] = vec
                mask[i] = 1.0
        historical_parent_vision[pid] = {
            "targets": torch.tensor(values, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }
    actions = torch.tensor([int(r.get("action", 0)) for r in rows], dtype=torch.long)
    rewards = torch.tensor([float(r.get("reward", 0.0)) for r in rows], dtype=torch.float32)
    successes = torch.tensor([float(r.get("success", 0.0)) for r in rows], dtype=torch.float32)
    care = torch.tensor(np.stack([np.asarray(r.get("care", [0]*cfg.care_dim), np.float32) for r in rows]), dtype=torch.float32)
    qas = []
    for row in rows:
        options = [q for q in row.get("qas", []) if q.get("question") and q.get("answer")]
        qas.append(rng.choice(options) if options else {"question": "What do you see?", "answer": "I am still learning."})
    qids = _pad([byte_encode(q["question"], cfg.max_question_bytes) for q in qas])
    aids = _pad([byte_encode(q["answer"], cfg.max_answer_bytes) for q in qas])
    next_images = None
    world_valid = torch.tensor([float(bool(r.get("world_valid", False) and r.get("next_image"))) for r in rows], dtype=torch.float32)
    if bool(world_valid.any()):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Parented training requires Pillow") from exc
        loaded = []
        for r in rows:
            path = r.get("next_image") or r["image"]
            im = Image.open(dataset.directory / path).convert("RGB").resize((cfg.image_size, cfg.image_size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            loaded.append(np.transpose(arr, (2,0,1)))
        next_images = torch.tensor(np.stack(loaded), dtype=torch.float32)
    return {
        "images": images, "context": context, "parent_vision": parent_vision,
        "historical_parent_vision": historical_parent_vision,
        "actions": actions, "rewards": rewards, "successes": successes, "care": care,
        "question_ids": qids, "answer_ids": aids, "world_valid": world_valid, "next_images": next_images,
    }


def _normalize(x):
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-6)


def training_step(model: InfantBrain, batch: dict, parent_id: str, device: str) -> tuple[torch.Tensor, dict]:
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    visual = model.vision(b["images"])
    parent_pred = model.parent_visual_prediction(parent_id, visual)
    active_vision_loss = 1.0 - (_normalize(parent_pred) * _normalize(b["parent_vision"])).sum(-1).mean()
    replay_losses = []
    for historical_parent_id, payload in (batch.get("historical_parent_vision") or {}).items():
        targets = payload["targets"].to(device)
        mask = payload["mask"].to(device)
        if float(mask.sum()) <= 0.0:
            continue
        try:
            hist_pred = model.parent_visual_prediction(historical_parent_id, visual)
        except KeyError:
            continue
        per = 1.0 - (_normalize(hist_pred) * _normalize(targets)).sum(-1)
        replay_losses.append((per * mask).sum() / mask.sum().clamp_min(1.0))
    replay_vision_loss = torch.stack(replay_losses).mean() if replay_losses else torch.zeros((), device=device)
    vision_loss = active_vision_loss + 0.25 * replay_vision_loss

    memory = torch.zeros(visual.shape[0], model.cfg.memory_dim, device=device)
    prev_action = (b["actions"] + 1).clamp(0, model.cfg.action_dim)
    hidden = model.nucleus.initial_state(visual.shape[0], device)
    nucleus = model.nucleus(visual, b["context"], memory, prev_action, hidden)
    workspace = nucleus["workspace"]

    answer = b["answer_ids"]
    logits = model.language(workspace, b["question_ids"], answer[:, :-1])
    language_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), answer[:, 1:].reshape(-1), ignore_index=BYTE_PAD
    )
    motor_logits = model.motor(workspace)
    motor_loss = F.cross_entropy(motor_logits, b["actions"])
    value = model.values(workspace)
    value_loss = F.mse_loss(value, b["care"])

    action_oh = one_hot_action(b["actions"], model.cfg.action_dim)
    world = model.world(workspace, action_oh)
    reward_loss = F.smooth_l1_loss(world["reward"], b["rewards"])
    success_loss = F.binary_cross_entropy_with_logits(world["success_logit"], b["successes"])
    world_bus_loss = torch.zeros((), device=device)
    if b.get("next_images") is not None and bool(b["world_valid"].sum() > 0):
        with torch.no_grad():
            next_visual = model.vision(b["next_images"])
        per = F.smooth_l1_loss(world["next_bus"], next_visual, reduction="none").mean(-1)
        world_bus_loss = (per * b["world_valid"]).sum() / b["world_valid"].sum().clamp_min(1.0)

    # Keep router non-collapsed while the child is young. Later curricula can supervise routing directly.
    route_entropy = -(nucleus["router"] * torch.log(nucleus["router"] + 1e-8)).sum(-1).mean()
    route_target = torch.tensor(np.log(6.0), device=device)
    route_loss = (route_entropy - route_target).abs()

    total = (
        1.00 * vision_loss + 1.00 * language_loss + 0.50 * motor_loss +
        0.20 * value_loss + 0.20 * reward_loss + 0.20 * success_loss + 0.40 * world_bus_loss + 0.01 * route_loss
    )
    return total, {
        "loss": float(total.detach()), "vision": float(vision_loss.detach()),
        "vision_active": float(active_vision_loss.detach()), "vision_replay": float(replay_vision_loss.detach()),
        "language": float(language_loss.detach()), "motor": float(motor_loss.detach()),
        "values": float(value_loss.detach()), "reward": float(reward_loss.detach()),
        "success": float(success_loss.detach()), "world": float(world_bus_loss.detach()), "router": float(route_loss.detach()),
    }


def save_checkpoint(path: str | Path, model: InfantBrain, *, updates: int, parent_id: str,
                    dataset_digest: str, optimizer=None, metadata=None) -> None:
    path = Path(path)
    payload = model.checkpoint_payload(metadata={
        "updates": int(updates), "parent_id": parent_id, "dataset_digest": dataset_digest,
        **dict(metadata or {}),
    })
    payload["updates"] = int(updates)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[InfantBrain, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = InfantConfig(**state["config"])
    dims = state.get("vision_parent_dims", {})
    model = InfantBrain(cfg, vision_parent_dims=dims)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    return model, state



def _discover_vision_parent_dims(dataset: NurseryDataset) -> dict[str, int]:
    manifest = dataset.manifest()
    dims = {str(manifest["vision_parent_id"]): int(manifest["vision_parent_dim"])}
    for row in dataset.records():
        for item in row.get("parent_vision_history") or []:
            pid, rel = item.get("parent_id"), item.get("path")
            if pid and rel and pid not in dims:
                dims[pid] = int(np.load(dataset.directory / rel).size)
    return dims


def train(dataset_path: str | Path, state_dir: str | Path, *, profile="proof", updates=2000,
          batch_size=8, device="cpu", learning_rate=3e-4, seed=17, resume=True) -> dict:
    dataset = NurseryDataset(dataset_path)
    manifest = dataset.manifest()
    if not manifest:
        raise RuntimeError("Nursery manifest missing")
    parent_id = manifest["vision_parent_id"]
    parent_dim = int(manifest["vision_parent_dim"])
    parent_dims = _discover_vision_parent_dims(dataset)
    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    rng = random.Random(seed)

    if resume and checkpoint.exists():
        model, old = load_checkpoint(checkpoint, device)
        # Parent upgrades may change target dimension. Add only a detachable new
        # distillation head; the child cortex/bus and all existing child weights stay intact.
        added_parent_head = False
        for pid, dim in parent_dims.items():
            key = pid.replace(".", "__").replace("/", "_")
            if key not in model.vision_parent_heads:
                model.add_vision_parent_head(pid, dim)
                added_parent_head = True
        model.to(device)
        start = int(old.get("updates", old.get("metadata", {}).get("updates", 0)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer_reinitialized = bool(added_parent_head)
        if old.get("optimizer") and not added_parent_head:
            try:
                optimizer.load_state_dict(old["optimizer"])
                optimizer_reinitialized = False
            except (ValueError, RuntimeError):
                optimizer_reinitialized = True
    else:
        cfg = InfantConfig.for_profile(profile)
        model = InfantBrain(cfg, vision_parent_dims=parent_dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer_reinitialized = False
        start = 0

    train_rows, _ = dataset.split(seed=seed)
    if not train_rows:
        raise RuntimeError("No training lessons")
    model.train()
    latest = {}
    for local_step in range(1, int(updates) + 1):
        rows = [rng.choice(train_rows) for _ in range(int(batch_size))]
        batch = _batch(dataset, rows, model.cfg, rng)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_step(model, batch, parent_id, device)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        latest = metrics
        if local_step % 100 == 0 or local_step == updates:
            print(
                f"infant_updates={start+local_step} loss={metrics['loss']:.4f} "
                f"vision={metrics['vision']:.4f} language={metrics['language']:.4f} "
                f"motor={metrics['motor']:.4f} values={metrics['values']:.4f} grad={grad:.3f}"
            )
        if local_step % 500 == 0:
            save_checkpoint(checkpoint, model, updates=start+local_step, parent_id=parent_id,
                            dataset_digest=dataset.digest(), optimizer=optimizer)
    total_updates = start + int(updates)
    save_checkpoint(checkpoint, model, updates=total_updates, parent_id=parent_id,
                    dataset_digest=dataset.digest(), optimizer=optimizer)
    return {
        "format": "continuous_brain.parented_training.v091",
        "updates": total_updates,
        "parameters": model.parameter_count,
        "profile": model.cfg.profile,
        "checkpoint": str(checkpoint),
        "parent_id": parent_id,
        "vision_parent_ids": sorted(parent_dims),
        "optimizer_reinitialized": bool(optimizer_reinitialized),
        "latest": latest,
    }


def evaluate(dataset_path: str | Path, state_dir: str | Path, *, device="cpu", seed=17,
             max_generated=48) -> dict:
    dataset = NurseryDataset(dataset_path)
    manifest = dataset.manifest()
    parent_id = manifest["vision_parent_id"]
    model, state = load_checkpoint(Path(state_dir) / "infant_training.pt", device)
    _, rows = dataset.split(seed=seed)
    if not rows:
        raise RuntimeError("No validation lessons")
    model.eval()
    rng = random.Random(seed + 1)
    cosines, actions, token_ok, token_total, value_errors = [], [], 0, 0, []
    historical_cosines: dict[str, list[float]] = {}
    generated_sims = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start+16]
            batch = _batch(dataset, chunk, model.cfg, rng)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            visual = model.vision(b["images"])
            pred = model.parent_visual_prediction(parent_id, visual)
            cosines.extend((_normalize(pred) * _normalize(b["parent_vision"])).sum(-1).cpu().tolist())
            for historical_parent_id, payload in (batch.get("historical_parent_vision") or {}).items():
                try:
                    hist_pred = model.parent_visual_prediction(historical_parent_id, visual)
                except KeyError:
                    continue
                targets = payload["targets"].to(device)
                mask = payload["mask"].to(device).bool()
                if not bool(mask.any()):
                    continue
                vals = (_normalize(hist_pred) * _normalize(targets)).sum(-1)
                historical_cosines.setdefault(historical_parent_id, []).extend(vals[mask].cpu().tolist())
            memory = torch.zeros(len(chunk), model.cfg.memory_dim, device=device)
            hidden = model.nucleus.initial_state(len(chunk), device)
            nuc = model.nucleus(visual, b["context"], memory, (b["actions"]+1).clamp(0, model.cfg.action_dim), hidden)
            workspace = nuc["workspace"]
            actions.extend(model.motor(workspace).argmax(-1).eq(b["actions"]).float().cpu().tolist())
            values = model.values(workspace)
            value_errors.extend((values - b["care"]).abs().mean(-1).cpu().tolist())
            answer = b["answer_ids"]
            logits = model.language(workspace, b["question_ids"], answer[:, :-1]).argmax(-1)
            target = answer[:, 1:]
            mask = target.ne(BYTE_PAD)
            token_ok += int((logits.eq(target) & mask).sum().item())
            token_total += int(mask.sum().item())

            if len(generated_sims) < max_generated:
                for j in range(min(len(chunk), max_generated - len(generated_sims))):
                    generated = model.language.generate(workspace[j:j+1], b["question_ids"][j:j+1], max_new=96)
                    qas = [q for q in chunk[j].get("qas", []) if q.get("question") and q.get("answer")]
                    # batch sampled a QA internally; generated similarity is a conservative nearest-parent-answer metric.
                    expected = [str(q["answer"]) for q in qas] or ["I am still learning."]
                    generated_sims.append(max(difflib.SequenceMatcher(None, generated.lower(), e.lower()).ratio() for e in expected))

    historical_summary = {pid: float(np.mean(vals)) for pid, vals in historical_cosines.items() if vals}
    retention_min = min(historical_summary.values()) if historical_summary else None
    metrics = {
        "validation_records": len(rows),
        "vision_parent_cosine": float(np.mean(cosines)),
        "vision_historical_cosine_by_parent": historical_summary,
        "vision_retention_min": retention_min,
        "language_token_accuracy": float(token_ok / max(1, token_total)),
        "language_generated_similarity": float(np.mean(generated_sims)) if generated_sims else 0.0,
        "motor_accuracy": float(np.mean(actions)),
        "value_mae": float(np.mean(value_errors)),
    }
    active_vision_score = max(0.0, min(1.0, metrics["vision_parent_cosine"]))
    if retention_min is not None:
        active_vision_score = min(active_vision_score, max(0.0, min(1.0, retention_min)))
    capability_scores = {
        "vision": active_vision_score,
        "language": max(0.0, min(1.0, 0.7 * metrics["language_token_accuracy"] + 0.3 * metrics["language_generated_similarity"])),
        "motor": metrics["motor_accuracy"],
        "values": max(0.0, 1.0 - metrics["value_mae"]),
    }
    action_counts = {}
    for row in rows:
        a = int(row.get("action", 0)); action_counts[a] = action_counts.get(a, 0) + 1
    care_arr = np.asarray([r.get("care", [0.0] * model.cfg.care_dim) for r in rows], dtype=np.float32)
    qa_count = sum(1 for r in rows if any(q.get("question") and q.get("answer") for q in r.get("qas", [])))
    evidence_eligible = {
        "vision": len(rows) >= 50,
        "language": qa_count >= 50,
        "motor": len(action_counts) >= 2 and min(action_counts.values()) >= 10,
        "values": bool(care_arr.size and float(care_arr.std(axis=0).max()) > 1e-3),
    }
    evidence = {
        "action_counts": action_counts,
        "qa_records": qa_count,
        "care_max_std": float(care_arr.std(axis=0).max()) if care_arr.size else 0.0,
        "world_valid_records": sum(bool(r.get("world_valid", False) and r.get("next_image")) for r in rows),
    }
    thresholds = {"vision": 0.80, "language": 0.70, "motor": 0.70, "values": 0.80}
    evaluated_capabilities = [k for k in thresholds if evidence_eligible.get(k, False)]
    passed = bool(evaluated_capabilities) and all(capability_scores[k] >= thresholds[k] for k in evaluated_capabilities)
    result = {
        "format": "continuous_brain.parented_qualification.v091",
        "passed": bool(passed), "metrics": metrics, "capability_scores": capability_scores,
        "evidence": evidence, "evidence_eligible": evidence_eligible,
        "evaluated_capabilities": evaluated_capabilities,
        "thresholds": thresholds, "training_updates": int(state.get("updates", 0)),
        "dataset_digest": dataset.digest(),
    }
    out = Path(state_dir) / "infant_evaluation.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    if passed:
        # Qualified checkpoint deliberately excludes optimizer state.
        save_checkpoint(Path(state_dir) / "qualified_infant.pt", model,
                        updates=int(state.get("updates", 0)), parent_id=parent_id,
                        dataset_digest=dataset.digest(), metadata={"qualification": result})
    return result
