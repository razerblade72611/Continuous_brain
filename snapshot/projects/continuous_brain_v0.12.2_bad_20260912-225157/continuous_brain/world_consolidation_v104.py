from __future__ import annotations

"""v0.10.4 continual world-memory consolidation.

This module repairs cross-environment interference in the child-owned consequence
model without changing the rest of the child.  It intentionally does not train
planning, options, motor, language, vision, spatial, nucleus, or target cortices.

Training uses equal domain weighting and action-balanced sampling within each
training domain.  Fresh retention holdouts are raw child-observation transition
sets and do not require or query an external visual parent.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .environment import ForagingWorld, DetourWorld
from .infant import one_hot_action
from .nursery_collect import detour_collection_action
from .parented_data import NurseryDataset
from .parented_train import load_checkpoint, save_checkpoint


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def transition_digest(directory: str | Path) -> str:
    """Hash transition metadata and every referenced current/next image.

    NurseryDataset.digest historically emphasizes lessons + parent targets.  v104
    raw world holdouts deliberately contain no parent targets, so this digest
    additionally seals the actual pixels used by world-model evaluation.
    """
    ds = NurseryDataset(directory)
    rows = ds.records()
    h = hashlib.sha256()
    if ds.records_path.exists():
        h.update(ds.records_path.read_bytes())
    refs = set()
    for row in rows:
        for key in ("image", "next_image"):
            rel = row.get(key)
            if rel:
                refs.add(str(rel))
    for rel in sorted(refs):
        path = ds.directory / rel
        h.update(rel.encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def _save_rgb(path: Path, array: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("world holdout collection requires Pillow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _context(obs: np.ndarray) -> np.ndarray:
    out = np.zeros(48, dtype=np.float32)
    out[:6] = obs[:6]
    out[6] = float(obs[4])
    out[30] = 1.0
    return out


def _foraging_collection_action(obs: np.ndarray, rng: random.Random, exploration: float) -> int:
    """Parent-free behavior policy used only to generate diverse heldout transitions."""
    if rng.random() < float(exploration):
        return rng.randrange(5)
    dx, dy = float(obs[2]), float(obs[3])
    if abs(dx) > abs(dy):
        return 4 if dx > 0 else 3
    return 1 if dy > 0 else 2


def collect_world_holdout(directory: str | Path, *, domain: str, samples: int = 1200,
                          seed: int = 24017, exploration: float = 0.45) -> dict:
    """Collect a fresh, parent-free transition holdout for world retention.

    The collection policy is used only to obtain diverse lived transitions.  No
    parent model or child model is queried and the resulting data is explicitly
    marked training-prohibited.
    """
    directory = Path(directory)
    ds = NurseryDataset(directory)
    if ds.records():
        raise FileExistsError("Fresh v0.10.4 holdout collection requires an empty directory")
    rng = random.Random(int(seed))
    exploration = max(0.0, min(1.0, float(exploration)))
    domain = str(domain).strip().lower()
    if domain == "foraging":
        env = ForagingWorld(seed=int(seed), max_episode_steps=500)
    elif domain == "detour":
        env = DetourWorld(seed=int(seed), max_episode_steps=120)
    else:
        raise ValueError("domain must be 'foraging' or 'detour'")

    obs = env.reset()
    previous_action = -1
    episode_id = 0
    episode_step = 0
    for i in range(int(samples)):
        rid = f"v104-{domain}-{i:07d}"
        image_path = ds.images / f"{rid}.png"
        next_image_path = ds.images / f"{rid}-next.png"
        _save_rgb(image_path, env.render_rgb(96))

        if domain == "foraging":
            action = _foraging_collection_action(obs, rng, exploration)
        else:
            action = detour_collection_action(obs, env, rng, exploration)

        next_obs, reward, done, info = env.step(int(action))
        _save_rgb(next_image_path, env.render_rgb(96))
        ds.append({
            "episode_id": int(episode_id),
            "episode_step": int(episode_step),
            "previous_action": int(previous_action),
            "record_id": rid,
            "image": str(image_path.relative_to(directory)),
            "next_image": str(next_image_path.relative_to(directory)),
            "context": _context(obs).tolist(),
            "next_context": _context(next_obs).tolist(),
            "executed_action": int(action),
            "reward": float(reward),
            "success": float(bool(info.get("reached"))),
            "collision": bool(info.get("collision", False)),
            "world_valid": True,
            "motor_valid": False,
            "source": f"v104-{domain}-world-retention-holdout",
            "provenance": "lived_heldout_environment",
            "training_prohibited": True,
        })
        if done:
            obs = env.reset()
            previous_action = -1
            episode_id += 1
            episode_step = 0
        else:
            obs = next_obs
            previous_action = int(action)
            episode_step += 1
        if (i + 1) % 250 == 0:
            print(f"v104_{domain}_holdout={i+1}/{samples}")

    digest = transition_digest(directory)
    action_counts = defaultdict(int)
    for row in ds.records():
        action_counts[int(row["executed_action"])] += 1
    manifest = {
        "format": "continuous_brain.world_retention_holdout.v104",
        "domain": domain,
        "records": int(samples),
        "seed": int(seed),
        "exploration": float(exploration),
        "parent_models_used": False,
        "child_model_used": False,
        "training_prohibited": True,
        "transition_digest": digest,
        "executed_action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
    }
    ds.write_manifest(manifest)
    return manifest


def seal_world_holdouts(*, foraging_dataset: str | Path, detour_dataset: str | Path,
                        training_datasets: list[str | Path], output: str | Path) -> dict:
    holdouts = {}
    for name, path in (("foraging", foraging_dataset), ("detour", detour_dataset)):
        ds = NurseryDataset(path)
        manifest = ds.manifest()
        if manifest.get("format") != "continuous_brain.world_retention_holdout.v104":
            raise ValueError(f"{name} holdout is not a v0.10.4 raw world holdout")
        if manifest.get("domain") != name:
            raise ValueError(f"holdout domain mismatch: expected {name}")
        if not bool(manifest.get("training_prohibited")):
            raise ValueError(f"{name} holdout is not marked training-prohibited")
        digest = transition_digest(path)
        if digest != manifest.get("transition_digest"):
            raise RuntimeError(f"{name} holdout digest mismatch")
        holdouts[name] = {
            "path": str(Path(path).resolve()),
            "transition_digest": digest,
            "records": len(ds.records()),
            "seed": manifest.get("seed"),
            "training_prohibited": True,
        }

    training = []
    for path in training_datasets:
        ds = NurseryDataset(path)
        if not ds.records():
            raise RuntimeError(f"training dataset is empty: {path}")
        training.append({
            "path": str(Path(path).resolve()),
            "transition_digest": transition_digest(path),
            "records": len(ds.records()),
        })
    holdout_digests = {x["transition_digest"] for x in holdouts.values()}
    if any(x["transition_digest"] in holdout_digests for x in training):
        raise RuntimeError("a sealed holdout is also listed as training data")

    result = {
        "format": "continuous_brain.world_retention_seal.v104",
        "holdouts": holdouts,
        "training_datasets": training,
        "rule": "sealed holdouts must never be used by train-world-consolidation",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_seal(path: str | Path) -> dict:
    seal = json.loads(Path(path).read_text())
    if seal.get("format") != "continuous_brain.world_retention_seal.v104":
        raise ValueError("invalid v0.10.4 holdout seal")
    return seal


def _load_images(ds: NurseryDataset, rows: list[dict], size: int, field: str) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("world consolidation requires Pillow") from exc
    images = []
    for row in rows:
        rel = row.get(field)
        if not rel:
            raise ValueError(f"world row missing {field}")
        im = Image.open(ds.directory / rel).convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        images.append(np.transpose(arr, (2, 0, 1)))
    return torch.tensor(np.stack(images), dtype=torch.float32)


def _world_batch(ds: NurseryDataset, rows: list[dict], model, device: str) -> dict:
    return {
        "images": _load_images(ds, rows, model.cfg.image_size, "image").to(device),
        "next_images": _load_images(ds, rows, model.cfg.image_size, "next_image").to(device),
        "context": torch.tensor(np.stack([np.asarray(r["context"], np.float32) for r in rows]), dtype=torch.float32, device=device),
        "next_context": torch.tensor(np.stack([np.asarray(r["next_context"], np.float32) for r in rows]), dtype=torch.float32, device=device),
        "executed_actions": torch.tensor([int(r.get("executed_action", r.get("action", 0))) for r in rows], dtype=torch.long, device=device),
        "previous_action_tokens": torch.tensor([
            (int(r.get("previous_action", -1)) + 1) if int(r.get("previous_action", -1)) >= 0 else 0
            for r in rows
        ], dtype=torch.long, device=device),
        "rewards": torch.tensor([float(r.get("reward", 0.0)) for r in rows], dtype=torch.float32, device=device),
        "successes": torch.tensor([float(r.get("success", 0.0)) for r in rows], dtype=torch.float32, device=device),
    }


def _world_loss(model, batch: dict) -> tuple[torch.Tensor, dict]:
    b = batch
    n = b["images"].shape[0]
    with torch.no_grad():
        visual = model.vision(b["images"])
        spatial = model.spatial(b["images"])
        fused = model.fuse_vision(visual, spatial)
        memory = torch.zeros(n, model.cfg.memory_dim, device=b["images"].device)
        hidden = model.nucleus.initial_state(n, b["images"].device)
        nuc = model.nucleus(fused, b["context"], memory, b["previous_action_tokens"], hidden)
        workspace = nuc["workspace"].detach()
        base_spatial = model.target_spatial_bus(b["images"]).detach()
        next_spatial = model.target_spatial_bus(b["next_images"]).detach()

    action_oh = one_hot_action(b["executed_actions"], model.cfg.action_dim)
    pred = model.world(workspace, action_oh, base_bus=base_spatial, base_context=b["context"])

    reward_target = b["rewards"].unsqueeze(0).expand_as(pred["member_reward"])
    success_target = b["successes"].unsqueeze(0).expand_as(pred["member_success_logit"])
    reward_loss = F.smooth_l1_loss(pred["member_reward"], reward_target)
    success_loss = F.binary_cross_entropy_with_logits(pred["member_success_logit"], success_target)
    bus_target = next_spatial.unsqueeze(0).expand_as(pred["member_bus"])
    bus_loss = F.smooth_l1_loss(pred["member_bus"], bus_target)
    dim = model.cfg.dynamics_context_dim
    ctx_target = b["next_context"][:, :dim]
    ctx_member_target = ctx_target.unsqueeze(0).expand_as(pred["member_context"])
    context_loss = F.smooth_l1_loss(pred["member_context"], ctx_member_target)

    wrong_actions = (b["executed_actions"] + 1) % model.cfg.action_dim
    wrong = model.world(
        workspace, one_hot_action(wrong_actions, model.cfg.action_dim),
        base_bus=base_spatial, base_context=b["context"],
    )
    correct_bus_err = F.smooth_l1_loss(pred["next_bus"], next_spatial, reduction="none").mean(-1)
    wrong_bus_err = F.smooth_l1_loss(wrong["next_bus"], next_spatial, reduction="none").mean(-1)
    correct_ctx_err = F.smooth_l1_loss(pred["next_context"], ctx_target, reduction="none").mean(-1)
    wrong_ctx_err = F.smooth_l1_loss(wrong["next_context"], ctx_target, reduction="none").mean(-1)
    rank_signal = correct_bus_err - wrong_bus_err + 0.5 * (correct_ctx_err - wrong_ctx_err)
    rank_loss = F.relu(1e-3 + rank_signal).mean()

    total = (
        0.50 * reward_loss + 0.50 * success_loss +
        2.50 * bus_loss + 3.00 * context_loss + 0.75 * rank_loss
    )
    return total, {
        "loss": float(total.detach()),
        "reward": float(reward_loss.detach()),
        "success": float(success_loss.detach()),
        "world": float(bus_loss.detach()),
        "world_context": float(context_loss.detach()),
        "world_action_rank": float(rank_loss.detach()),
    }


def _configure_world_only(model) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.world.parameters():
        p.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _world_rows(ds: NurseryDataset, seed: int) -> list[dict]:
    train_rows, _ = ds.split(seed=seed)
    rows = [r for r in train_rows if bool(r.get("world_valid", False) and r.get("next_image") and r.get("next_context") is not None)]
    if not rows:
        raise RuntimeError(f"no usable world transitions in {ds.directory}")
    return rows


def _action_counts(rows: list[dict]) -> dict[int, int]:
    out = defaultdict(int)
    for r in rows:
        out[int(r.get("executed_action", r.get("action", 0)))] += 1
    return dict(sorted(out.items()))


def _balanced_sample(rows: list[dict], n: int, action_dim: int, rng: random.Random) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[int(row.get("executed_action", row.get("action", 0)))].append(row)
    available = [a for a in range(int(action_dim)) if buckets.get(a)]
    if not available:
        raise RuntimeError("no action buckets available")
    chosen = []
    for i in range(int(n)):
        action = available[i % len(available)]
        chosen.append(rng.choice(buckets[action]))
    rng.shuffle(chosen)
    return chosen


def _snapshot_non_world(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("world.")}


def _changed_against(snapshot: dict[str, torch.Tensor], model) -> list[str]:
    current = model.state_dict()
    changed = []
    for key, before in snapshot.items():
        if key not in current or not torch.equal(before, current[key].detach().cpu()):
            changed.append(key)
    return changed


def train_world_consolidation(training_datasets: list[str | Path], state_dir: str | Path, *,
                              holdout_seal: str | Path, updates: int = 3000,
                              batch_per_domain: int = 16, learning_rate: float = 1e-4,
                              seed: int = 10410, device: str = "cpu") -> dict:
    if len(training_datasets) < 2:
        raise ValueError("world consolidation requires at least two training domains")
    seal = _load_seal(holdout_seal)
    sealed_holdout_digests = {
        entry["transition_digest"] for entry in (seal.get("holdouts") or {}).values()
    }
    sealed_training = {
        str(Path(entry["path"]).resolve()): entry["transition_digest"]
        for entry in seal.get("training_datasets", [])
    }

    datasets = []
    domain_info = []
    for path0 in training_datasets:
        path = Path(path0).resolve()
        digest = transition_digest(path)
        if digest in sealed_holdout_digests:
            raise RuntimeError(f"sealed holdout cannot be used for training: {path}")
        if sealed_training and sealed_training.get(str(path)) != digest:
            raise RuntimeError(f"training dataset not present in seal or changed since sealing: {path}")
        ds = NurseryDataset(path)
        rows = _world_rows(ds, seed)
        datasets.append((ds, rows))
        domain_info.append({
            "path": str(path),
            "transition_digest": digest,
            "world_records": len(rows),
            "executed_action_counts": {str(k): int(v) for k, v in _action_counts(rows).items()},
        })

    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_sha = _sha256_file(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    metadata = dict(saved.get("metadata") or {})
    preserved_metadata = {k: v for k, v in metadata.items() if k not in {"updates", "parent_id", "dataset_digest"}}
    start = int(saved.get("updates", metadata.get("updates", 0)))
    parent_id = str(metadata.get("parent_id", "vision.dinov3.base"))
    before_non_world = _snapshot_non_world(model)
    trainable = _configure_world_only(model)
    if not trainable or any(not name.startswith("world.") for name in trainable):
        raise RuntimeError("world-only parameter mask is invalid")

    model.eval()
    model.world.train()
    optimizer = torch.optim.AdamW(model.world.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    rng = random.Random(int(seed))
    latest_by_domain = {}
    for local_step in range(1, int(updates) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        metrics_by_domain = {}
        for domain_index, (ds, rows) in enumerate(datasets):
            selected = _balanced_sample(rows, int(batch_per_domain), model.cfg.action_dim, rng)
            batch = _world_batch(ds, selected, model, device)
            loss, metrics = _world_loss(model, batch)
            losses.append(loss)
            metrics_by_domain[str(domain_index)] = metrics
        total = torch.stack(losses).mean()
        total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.world.parameters(), 1.0))
        optimizer.step()
        latest_by_domain = metrics_by_domain
        if local_step == 1 or local_step % 250 == 0 or local_step == int(updates):
            parts = []
            for i, metrics in metrics_by_domain.items():
                parts.append(f"d{i}:world={metrics['world']:.4f},ctx={metrics['world_context']:.4f},rank={metrics['world_action_rank']:.4f}")
            print(f"world_consolidation={local_step}/{updates} loss={float(total.detach()):.5f} grad={grad:.3f} " + " ".join(parts))

    changed_non_world = _changed_against(before_non_world, model)
    if changed_non_world:
        raise RuntimeError(f"world consolidation changed frozen tensors: {changed_non_world[:8]}")

    total_updates = start + int(updates)
    digests = [x["transition_digest"] for x in domain_info]
    save_checkpoint(
        checkpoint, model, updates=total_updates, parent_id=parent_id,
        dataset_digest=hashlib.sha256("|".join(digests).encode()).hexdigest(),
        optimizer=optimizer,
        metadata={
            **preserved_metadata,
            "world_dynamics_version": "dual-stream-state-conditioned-v2",
            "world_consolidation_version": "balanced-multidomain-action-rehearsal-v1",
            "world_consolidation_updates": int(updates),
            "world_consolidation_seed": int(seed),
            "world_consolidation_domains": domain_info,
            "world_consolidation_holdout_seal": _sha256_file(holdout_seal),
        },
    )
    trained_sha = _sha256_file(checkpoint)
    return {
        "format": "continuous_brain.world_consolidation_training.v104",
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": source_sha,
        "trained_checkpoint_sha256": trained_sha,
        "source_training_updates": start,
        "training_updates": total_updates,
        "world_consolidation_updates": int(updates),
        "world_only": True,
        "world_only_trainable_parameters": trainable,
        "changed_non_world_tensors": len(changed_non_world),
        "balanced_domain_weighting": True,
        "action_balanced_within_domain": True,
        "training_domains": domain_info,
        "holdout_seal_sha256": _sha256_file(holdout_seal),
        "training": {
            "updates": int(updates),
            "batch_per_domain": int(batch_per_domain),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
        "latest_by_domain": latest_by_domain,
    }


@torch.no_grad()
def _evaluate_world_rows(model, ds: NurseryDataset, rows: list[dict], *, device: str = "cpu") -> dict:
    model.eval()
    model_errors, baseline_errors, wrong_errors = [], [], []
    ctx_errors, ctx_baselines, ctx_wrong_errors = [], [], []
    ctx_dim_errors, ctx_dim_baselines = [], []
    reward_abs, success_correct, uncertainties = [], [], []
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        b = _world_batch(ds, chunk, model, device)
        n = len(chunk)
        visual = model.vision(b["images"])
        spatial = model.spatial(b["images"])
        fused = model.fuse_vision(visual, spatial)
        memory = torch.zeros(n, model.cfg.memory_dim, device=device)
        hidden = model.nucleus.initial_state(n, device)
        nuc = model.nucleus(fused, b["context"], memory, b["previous_action_tokens"], hidden)
        workspace = nuc["workspace"]
        current_spatial = model.target_spatial_bus(b["images"])
        next_spatial = model.target_spatial_bus(b["next_images"])
        action_oh = one_hot_action(b["executed_actions"], model.cfg.action_dim)
        pred = model.world(workspace, action_oh, base_bus=current_spatial, base_context=b["context"])
        wrong_actions = (b["executed_actions"] + 1) % model.cfg.action_dim
        wrong = model.world(
            workspace, one_hot_action(wrong_actions, model.cfg.action_dim),
            base_bus=current_spatial, base_context=b["context"],
        )
        model_errors.extend(F.smooth_l1_loss(pred["next_bus"], next_spatial, reduction="none").mean(-1).cpu().tolist())
        baseline_errors.extend(F.smooth_l1_loss(current_spatial, next_spatial, reduction="none").mean(-1).cpu().tolist())
        wrong_errors.extend(F.smooth_l1_loss(wrong["next_bus"], next_spatial, reduction="none").mean(-1).cpu().tolist())
        dim = model.cfg.dynamics_context_dim
        ctx_target = b["next_context"][:, :dim]
        current_ctx = b["context"][:, :dim]
        ctx_errors.extend(F.smooth_l1_loss(pred["next_context"], ctx_target, reduction="none").mean(-1).cpu().tolist())
        ctx_baselines.extend(F.smooth_l1_loss(current_ctx, ctx_target, reduction="none").mean(-1).cpu().tolist())
        ctx_wrong_errors.extend(F.smooth_l1_loss(wrong["next_context"], ctx_target, reduction="none").mean(-1).cpu().tolist())
        ctx_dim_errors.extend(F.smooth_l1_loss(pred["next_context"], ctx_target, reduction="none").cpu().tolist())
        ctx_dim_baselines.extend(F.smooth_l1_loss(current_ctx, ctx_target, reduction="none").cpu().tolist())
        reward_abs.extend((pred["reward"] - b["rewards"]).abs().cpu().tolist())
        success_correct.extend(torch.sigmoid(pred["success_logit"]).ge(0.5).eq(b["successes"].ge(0.5)).float().cpu().tolist())
        uncertainties.extend(pred["uncertainty"].cpu().tolist())

    model_error = float(np.mean(model_errors))
    baseline_error = float(np.mean(baseline_errors))
    wrong_error = float(np.mean(wrong_errors))
    ctx_error = float(np.mean(ctx_errors))
    ctx_baseline = float(np.mean(ctx_baselines))
    ctx_wrong = float(np.mean(ctx_wrong_errors))
    per_dim_err = np.asarray(ctx_dim_errors, dtype=np.float64).mean(axis=0)
    per_dim_base = np.asarray(ctx_dim_baselines, dtype=np.float64).mean(axis=0)
    action_counts = _action_counts(rows)
    supported = {a: c for a, c in action_counts.items() if c >= 3}
    eligible = len(rows) >= 50 and len(supported) >= 2
    spatial_passed = bool(model_error < 0.98 * baseline_error and wrong_error - model_error > 0.0)
    context_passed = bool(ctx_error < 0.80 * ctx_baseline and ctx_wrong - ctx_error > 0.0)
    success_accuracy = float(np.mean(success_correct))
    passed = bool(eligible and spatial_passed and context_passed and success_accuracy >= 0.70)
    return {
        "records": len(rows),
        "transition_digest": transition_digest(ds.directory),
        "executed_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "supported_executed_action_counts": {str(k): int(v) for k, v in supported.items()},
        "evidence_eligible": eligible,
        "metrics": {
            "world_next_spatial_error": model_error,
            "world_spatial_no_change_baseline_error": baseline_error,
            "world_spatial_wrong_action_error": wrong_error,
            "world_spatial_relative_improvement": float((baseline_error - model_error) / max(1e-9, baseline_error)),
            "world_spatial_action_advantage": float(wrong_error - model_error),
            "world_next_context_error": ctx_error,
            "world_context_no_change_baseline_error": ctx_baseline,
            "world_context_wrong_action_error": ctx_wrong,
            "world_context_relative_improvement": float((ctx_baseline - ctx_error) / max(1e-9, ctx_baseline)),
            "world_context_action_advantage": float(ctx_wrong - ctx_error),
            "world_context_error_by_dim": [float(x) for x in per_dim_err],
            "world_context_baseline_by_dim": [float(x) for x in per_dim_base],
            "world_context_relative_improvement_by_dim": [
                float((b - e) / max(1e-9, b)) for e, b in zip(per_dim_err, per_dim_base)
            ],
            "world_reward_mae": float(np.mean(reward_abs)),
            "world_success_accuracy": success_accuracy,
            "world_uncertainty_mean": float(np.mean(uncertainties)),
        },
        "world_spatial_passed": spatial_passed,
        "world_structured_state_passed": context_passed,
        "world_model_passed": passed,
    }


def evaluate_world_consolidation(checkpoint: str | Path, *, foraging_holdout: str | Path,
                                 detour_holdout: str | Path, holdout_seal: str | Path,
                                 device: str = "cpu", output: str | Path | None = None) -> dict:
    seal = _load_seal(holdout_seal)
    model, saved = load_checkpoint(checkpoint, device)
    domains = {}
    for name, path0 in (("foraging", foraging_holdout), ("detour", detour_holdout)):
        path = Path(path0).resolve()
        expected = (seal.get("holdouts") or {}).get(name) or {}
        digest = transition_digest(path)
        if str(path) != str(expected.get("path")) or digest != expected.get("transition_digest"):
            raise RuntimeError(f"{name} holdout does not match the pre-training seal")
        ds = NurseryDataset(path)
        rows = [r for r in ds.records() if bool(r.get("world_valid", False) and r.get("next_image") and r.get("next_context") is not None)]
        if any(not bool(r.get("training_prohibited", False)) for r in rows):
            raise RuntimeError(f"{name} holdout contains non-sealed records")
        domains[name] = _evaluate_world_rows(model, ds, rows, device=device)

    result = {
        "format": "continuous_brain.world_consolidation_qualification.v104",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "holdout_seal_sha256": _sha256_file(holdout_seal),
        "parents_used": False,
        "training_performed_during_evaluation": False,
        "domains": domains,
        "world_consolidation_passed": all(bool(v.get("world_model_passed")) for v in domains.values()),
        "thresholds": {
            "spatial_vs_no_change": 0.98,
            "structured_context_vs_no_change": 0.80,
            "positive_action_advantage": True,
            "success_accuracy": 0.70,
        },
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
