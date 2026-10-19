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
    # `action` is the parent/demonstrator motor target.  `executed_action` is what
    # actually caused the recorded transition.  Never use the target as a recurrent
    # previous-action cue or as the world-model action unless it was truly executed.
    actions = torch.tensor([int(r.get("action", 0)) for r in rows], dtype=torch.long)
    motor_valid = torch.tensor([
        float(bool(r.get("motor_valid", r.get("world_valid", False)))) for r in rows
    ], dtype=torch.float32)
    executed_actions = torch.tensor([int(r.get("executed_action", r.get("action", 0))) for r in rows], dtype=torch.long)
    previous_action_tokens = torch.tensor([
        (int(r.get("previous_action", -1)) + 1) if int(r.get("previous_action", -1)) >= 0 else 0
        for r in rows
    ], dtype=torch.long)
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
    context_valid = torch.tensor([float(bool(r.get("next_context") is not None)) for r in rows], dtype=torch.float32)
    next_context = torch.tensor(np.stack([
        np.asarray(r.get("next_context", r["context"]), np.float32) for r in rows
    ]), dtype=torch.float32)
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
        "actions": actions, "motor_valid": motor_valid, "executed_actions": executed_actions,
        "previous_action_tokens": previous_action_tokens,
        "rewards": rewards, "successes": successes, "care": care,
        "question_ids": qids, "answer_ids": aids,
        "world_valid": world_valid, "next_images": next_images,
        "context_valid": context_valid, "next_context": next_context,
    }


def _normalize(x):
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-6)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def training_step(model: InfantBrain, batch: dict, parent_id: str, device: str, *, world_focus: bool = False, world_only: bool = False) -> tuple[torch.Tensor, dict]:
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    visual = model.vision(b["images"])
    spatial = model.spatial(b["images"])
    fused_visual = model.fuse_vision(visual, spatial)
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
        replay_losses.append(_masked_mean(per, mask))
    replay_vision_loss = torch.stack(replay_losses).mean() if replay_losses else torch.zeros((), device=device)
    vision_loss = active_vision_loss + 0.25 * replay_vision_loss

    # Simulator/body-state grounding teaches the dorsal stream where things are.
    spatial_state_pred = model.spatial_state_head(spatial)
    spatial_grounding_loss = F.smooth_l1_loss(
        spatial_state_pred, b["context"][:, :model.cfg.dynamics_context_dim]
    )

    memory = torch.zeros(visual.shape[0], model.cfg.memory_dim, device=device)
    prev_action = b["previous_action_tokens"].clamp(0, model.cfg.action_dim)
    hidden = model.nucleus.initial_state(visual.shape[0], device)
    nucleus = model.nucleus(fused_visual, b["context"], memory, prev_action, hidden)
    workspace = nucleus["workspace"]

    answer = b["answer_ids"]
    logits = model.language(workspace, b["question_ids"], answer[:, :-1])
    language_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), answer[:, 1:].reshape(-1), ignore_index=BYTE_PAD
    )

    # Only genuine motor demonstrations supervise the motor cortex. Imported static
    # semantic lessons must not silently teach "idle" as a default action.
    motor_logits = model.motor(workspace)
    motor_per = F.cross_entropy(motor_logits, b["actions"], reduction="none")
    motor_loss = _masked_mean(motor_per, b["motor_valid"])

    value = model.values(workspace)
    value_loss = F.mse_loss(value, b["care"])

    world_mask = b["world_valid"].float()
    executed_action_oh = one_hot_action(b["executed_actions"], model.cfg.action_dim)
    # Predict in a slow child-owned target coordinate system. This is not a parent
    # representation: it is an EMA shadow of the child's own visual cortex.
    with torch.no_grad():
        target_spatial = model.target_spatial_bus(b["images"])
    world = model.world(
        workspace, executed_action_oh,
        base_bus=target_spatial,
        base_context=b["context"],
    )

    reward_loss = torch.zeros((), device=device)
    success_loss = torch.zeros((), device=device)
    world_bus_loss = torch.zeros((), device=device)
    world_context_loss = torch.zeros((), device=device)
    temporal_loss = torch.zeros((), device=device)
    action_rank_loss = torch.zeros((), device=device)

    if float(world_mask.sum()) > 0.0:
        # Reward/outcome learning is a world-model target and is therefore restricted
        # to actual transition evidence rather than legacy static lessons.
        reward_target = b["rewards"].unsqueeze(0).expand_as(world["member_reward"])
        success_target = b["successes"].unsqueeze(0).expand_as(world["member_success_logit"])
        reward_per = F.smooth_l1_loss(world["member_reward"], reward_target, reduction="none").mean(0)
        success_per = F.binary_cross_entropy_with_logits(
            world["member_success_logit"], success_target, reduction="none"
        ).mean(0)
        reward_loss = _masked_mean(reward_per, world_mask)
        success_loss = _masked_mean(success_per, world_mask)

    if b.get("next_images") is not None and float(world_mask.sum()) > 0.0:
        with torch.no_grad():
            next_target_spatial = model.target_spatial_bus(b["next_images"])
        member_target = next_target_spatial.unsqueeze(0).expand_as(world["member_bus"])
        per_member = F.smooth_l1_loss(world["member_bus"], member_target, reduction="none").mean(-1)
        world_bus_loss = (per_member * world_mask.unsqueeze(0)).sum() / (
            world_mask.sum().clamp_min(1.0) * per_member.shape[0]
        )

        context_mask = world_mask * b["context_valid"].float()
        if float(context_mask.sum()) > 0.0:
            dim = model.cfg.dynamics_context_dim
            context_target = b["next_context"][:, :dim]
            member_context_target = context_target.unsqueeze(0).expand_as(world["member_context"])
            per_context = F.smooth_l1_loss(
                world["member_context"], member_context_target, reduction="none"
            ).mean(-1)
            world_context_loss = (per_context * context_mask.unsqueeze(0)).sum() / (
                context_mask.sum().clamp_min(1.0) * per_context.shape[0]
            )

        temporal_state = torch.zeros(visual.shape[0], model.cfg.temporal_dim, device=device)
        temporal_bus, _ = model.dynamics_temporal(target_spatial, executed_action_oh, temporal_state)
        per_temporal = F.smooth_l1_loss(temporal_bus, next_target_spatial, reduction="none").mean(-1)
        temporal_loss = _masked_mean(per_temporal, world_mask)

        # Counterfactual ranking: the action that actually happened should explain the
        # observed transition better than a deliberately different action. This makes
        # action-dependence an explicit learning target rather than an accidental side effect.
        wrong_actions = (b["executed_actions"] + 1) % model.cfg.action_dim
        wrong_world = model.world(
            workspace, one_hot_action(wrong_actions, model.cfg.action_dim),
            base_bus=target_spatial, base_context=b["context"],
        )
        correct_bus_err = F.smooth_l1_loss(
            world["next_bus"], next_target_spatial, reduction="none"
        ).mean(-1)
        wrong_bus_err = F.smooth_l1_loss(
            wrong_world["next_bus"], next_target_spatial, reduction="none"
        ).mean(-1)
        rank_signal = correct_bus_err - wrong_bus_err
        if float(context_mask.sum()) > 0.0:
            dim = model.cfg.dynamics_context_dim
            ctx_target = b["next_context"][:, :dim]
            correct_ctx_err = F.smooth_l1_loss(
                world["next_context"], ctx_target, reduction="none"
            ).mean(-1)
            wrong_ctx_err = F.smooth_l1_loss(
                wrong_world["next_context"], ctx_target, reduction="none"
            ).mean(-1)
            rank_signal = rank_signal + 0.5 * (correct_ctx_err - wrong_ctx_err)
        # 1e-3 is intentionally small relative to the observed proof-scale bus errors.
        action_rank_loss = _masked_mean(F.relu(1e-3 + rank_signal), world_mask)

    # Keep router non-collapsed while the child is young. Later curricula can supervise routing directly.
    route_entropy = -(nucleus["router"] * torch.log(nucleus["router"] + 1e-8)).sum(-1).mean()
    route_target = torch.tensor(np.log(6.0), device=device)
    route_loss = (route_entropy - route_target).abs()

    if world_only:
        # P4 planning-lab adaptation: preserve the reactive child exactly and teach
        # only the consequence model.  The optimizer also freezes all non-world
        # parameters, so this weighting is a second guard against policy leakage.
        total = (
            0.50 * reward_loss + 0.50 * success_loss +
            2.50 * world_bus_loss + 3.00 * world_context_loss +
            0.75 * action_rank_loss
        )
    elif world_focus:
        # Developmental specialization phase: mature ventral/language/motor skills
        # remain under a small rehearsal gradient while causal/spatial dynamics
        # receive most of the capacity. This avoids global gradient clipping being
        # dominated by occasional motor/language spikes during world-model learning.
        total = (
            0.20 * vision_loss + 0.50 * spatial_grounding_loss +
            0.08 * language_loss + 0.08 * motor_loss +
            0.05 * value_loss + 0.25 * reward_loss + 0.25 * success_loss +
            2.00 * world_bus_loss + 2.00 * world_context_loss +
            0.75 * temporal_loss + 0.50 * action_rank_loss + 0.005 * route_loss
        )
    else:
        total = (
            1.00 * vision_loss + 0.50 * spatial_grounding_loss +
            1.00 * language_loss + 0.50 * motor_loss +
            0.20 * value_loss + 0.20 * reward_loss + 0.20 * success_loss +
            1.50 * world_bus_loss + 1.00 * world_context_loss +
            0.50 * temporal_loss + 0.25 * action_rank_loss + 0.01 * route_loss
        )
    return total, {
        "loss": float(total.detach()), "vision": float(vision_loss.detach()),
        "vision_active": float(active_vision_loss.detach()), "vision_replay": float(replay_vision_loss.detach()),
        "spatial": float(spatial_grounding_loss.detach()),
        "language": float(language_loss.detach()), "motor": float(motor_loss.detach()),
        "values": float(value_loss.detach()), "reward": float(reward_loss.detach()),
        "success": float(success_loss.detach()), "world": float(world_bus_loss.detach()),
        "world_context": float(world_context_loss.detach()),
        "world_action_rank": float(action_rank_loss.detach()),
        "temporal": float(temporal_loss.detach()), "router": float(route_loss.detach()),
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
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    allowed_missing = [
        k for k in missing
        if (
            k.startswith("target_vision.")
            or k.startswith("spatial.")
            or k.startswith("target_spatial.")
            or k.startswith("spatial_state_head.")
            or k.startswith("vision_fusion.")
            or k.startswith("dynamics_temporal.")
            or ".next_context." in k
            or ".state_bus." in k
            or ".state_context." in k
            or k.startswith("planning.")
        )
    ]
    disallowed_missing = sorted(set(missing) - set(allowed_missing))
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch; missing={disallowed_missing}, unexpected={list(unexpected)}"
        )
    # v0.9.3 and earlier had no slow target cortex.  Initialize it exactly from
    # the learned online child vision so migration preserves the child's current
    # perceptual coordinate system instead of introducing a random target.
    if any(k.startswith("target_vision.") for k in missing):
        model.sync_target_vision(tau=0.0)
    if any(k.startswith("target_spatial.") for k in missing):
        # New dorsal stream starts as a child-owned random cortex and its slow
        # target begins in exactly the same coordinate system.
        model._ema_module(model.target_spatial, model.spatial, 0.0)
        model.target_spatial.eval()
    model.to(device)
    state["_v094_missing_keys"] = list(missing)
    state["_v095_missing_keys"] = [
        k for k in missing if ".state_bus." in k or ".state_context." in k
    ]
    state["_v100_missing_keys"] = [k for k in missing if k.startswith("planning.")]
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


def _configure_world_only(model: InfantBrain) -> list[str]:
    """Freeze the child except its learned consequence model for P4 adaptation."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.world.parameters():
        p.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def train(dataset_path: str | Path, state_dir: str | Path, *, profile="proof", updates=2000,
          batch_size=8, device="cpu", learning_rate=3e-4, seed=17, resume=True,
          world_fraction=0.50, world_focus=False, world_only=False) -> dict:
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
        metadata = dict(old.get("metadata") or {})
        # v0.9.4 preserves the learned v0.9.3 residual visual dynamics, but adds
        # a slow target vision cortex and structured context-dynamics heads.
        # Missing new heads are zero-initialized by the model constructor; no learned
        # v0.9.3 visual residual weights are discarded.
        old_world_version = metadata.get("world_dynamics_version")
        migrated_world_residual = old_world_version not in {
            "dual-stream-spatial-v1", "dual-stream-state-conditioned-v2"
        }
        migrated_v094_keys = list(old.get("_v094_missing_keys") or [])
        migrated_v095_keys = list(old.get("_v095_missing_keys") or [])
        migrated_v100_keys = list(old.get("_v100_missing_keys") or [])
        if migrated_world_residual:
            # The predictive target changes from the ventral semantic bus to the
            # new dorsal spatial bus.  The old v0.9.3 bus residual head did not pass
            # qualification and is not coordinate-compatible, so reset only the
            # world residual outputs; reward/success and the rest of the child remain.
            model.world.reset_residual_heads()
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
        world_only_parameters = _configure_world_only(model) if world_only else []
        optim_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optim_params, lr=learning_rate, weight_decay=1e-4)
        optimizer_reinitialized = bool(
            world_only or added_parent_head or migrated_world_residual or migrated_v094_keys or migrated_v095_keys or migrated_v100_keys
        )
        if old.get("optimizer") and not (
            world_only or added_parent_head or migrated_world_residual or migrated_v094_keys or migrated_v095_keys or migrated_v100_keys
        ):
            try:
                optimizer.load_state_dict(old["optimizer"])
                optimizer_reinitialized = False
            except (ValueError, RuntimeError):
                optimizer_reinitialized = True
    else:
        cfg = InfantConfig.for_profile(profile)
        model = InfantBrain(cfg, vision_parent_dims=parent_dims).to(device)
        world_only_parameters = _configure_world_only(model) if world_only else []
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4)
        optimizer_reinitialized = bool(world_only)
        migrated_world_residual = False
        migrated_v094_keys = []
        migrated_v095_keys = []
        migrated_v100_keys = []
        start = 0

    train_rows, _ = dataset.split(seed=seed)
    if not train_rows:
        raise RuntimeError("No training lessons")
    model.train()
    # In --world-only mode, frozen BatchNorm/statistical state must remain immutable
    # as well as frozen parameters. Put the whole child in eval and reopen only the
    # consequence ensemble for training.
    if world_only:
        model.eval()
        model.world.train()
    latest = {}
    world_rows = [r for r in train_rows if bool(r.get("world_valid", False) and r.get("next_image"))]
    replay_rows = [r for r in train_rows if r not in world_rows]
    requested_world_fraction = max(0.0, min(1.0, float(world_fraction)))
    for local_step in range(1, int(updates) + 1):
        if world_rows and replay_rows and requested_world_fraction > 0.0:
            n_world = max(1, min(int(batch_size) - 1, int(round(int(batch_size) * requested_world_fraction))))
            n_replay = int(batch_size) - n_world
            rows = [rng.choice(world_rows) for _ in range(n_world)] + [rng.choice(replay_rows) for _ in range(n_replay)]
            rng.shuffle(rows)
        else:
            rows = [rng.choice(train_rows) for _ in range(int(batch_size))]
        batch = _batch(dataset, rows, model.cfg, rng)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_step(
            model, batch, parent_id, device, world_focus=bool(world_focus), world_only=bool(world_only)
        )
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        # Slow target cortex follows the child's online visual cortex after each
        # update. It never receives optimizer gradients of its own.
        if not world_only:
            model.sync_target_cortices()
        latest = metrics
        if local_step % 100 == 0 or local_step == updates:
            print(
                f"infant_updates={start+local_step} loss={metrics['loss']:.4f} "
                f"vision={metrics['vision']:.4f} spatial={metrics.get('spatial', 0.0):.4f} "
                f"language={metrics['language']:.4f} motor={metrics['motor']:.4f} "
                f"world={metrics.get('world', 0.0):.4f} ctx={metrics.get('world_context', 0.0):.4f} "
                f"values={metrics['values']:.4f} grad={grad:.3f}"
            )
        if local_step % 500 == 0:
            save_checkpoint(checkpoint, model, updates=start+local_step, parent_id=parent_id,
                            dataset_digest=dataset.digest(), optimizer=optimizer,
                            metadata={"world_dynamics_version": "dual-stream-state-conditioned-v2", "p4_world_only": bool(world_only)})
    total_updates = start + int(updates)
    save_checkpoint(checkpoint, model, updates=total_updates, parent_id=parent_id,
                    dataset_digest=dataset.digest(), optimizer=optimizer,
                    metadata={"world_dynamics_version": "dual-stream-state-conditioned-v2", "p4_world_only": bool(world_only)})
    return {
        "format": "continuous_brain.parented_training.v095",
        "updates": total_updates,
        "parameters": model.parameter_count,
        "profile": model.cfg.profile,
        "checkpoint": str(checkpoint),
        "parent_id": parent_id,
        "vision_parent_ids": sorted(parent_dims),
        "optimizer_reinitialized": bool(optimizer_reinitialized),
        "world_dynamics_migrated": bool(migrated_world_residual),
        "world_target_cortex_migrated": bool(
            any(k.startswith("target_spatial.") for k in migrated_v094_keys)
            or any(k.startswith("target_vision.") for k in migrated_v094_keys)
        ),
        "spatial_cortex_migrated": bool(any(
            k.startswith(("spatial.", "spatial_state_head.", "vision_fusion.", "dynamics_temporal."))
            for k in migrated_v094_keys
        )),
        "world_context_head_migrated": bool(any(".next_context." in k for k in migrated_v094_keys)),
        "world_target_ema_tau": float(model.cfg.world_ema_tau),
        "world_state_conditioning_migrated": bool(migrated_v095_keys),
        "planning_cortex_migrated": bool(migrated_v100_keys),
        "world_focus": bool(world_focus),
        "world_only": bool(world_only),
        "world_only_trainable_parameters": world_only_parameters if world_only else [],
        "sampling": {
            "world_fraction": requested_world_fraction,
            "world_records": len(world_rows),
            "replay_records": len(replay_rows),
        },
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
    cosines, motor_correct, token_ok, token_total, value_errors = [], [], 0, 0, []
    historical_cosines: dict[str, list[float]] = {}
    generated_sims = []
    world_model_errors, world_baseline_errors, world_wrong_action_errors = [], [], []
    world_context_errors, world_context_baselines, world_context_wrong_errors = [], [], []
    world_context_dim_errors, world_context_dim_baselines = [], []
    world_reward_abs, world_success_correct, world_uncertainties = [], [], []
    temporal_errors = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start+16]
            batch = _batch(dataset, chunk, model.cfg, rng)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            visual = model.vision(b["images"])
            spatial = model.spatial(b["images"])
            fused_visual = model.fuse_vision(visual, spatial)
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
            nuc = model.nucleus(
                fused_visual, b["context"], memory,
                b["previous_action_tokens"].clamp(0, model.cfg.action_dim), hidden
            )
            workspace = nuc["workspace"]
            motor_ok = model.motor(workspace).argmax(-1).eq(b["actions"]).float()
            motor_mask = b["motor_valid"].bool()
            if bool(motor_mask.any()):
                motor_correct.extend(motor_ok[motor_mask].cpu().tolist())
            values = model.values(workspace)
            value_errors.extend((values - b["care"]).abs().mean(-1).cpu().tolist())

            if b.get("next_images") is not None and bool(b["world_valid"].sum() > 0):
                valid = b["world_valid"].bool()
                target_spatial = model.target_spatial_bus(b["images"])
                next_target_spatial = model.target_spatial_bus(b["next_images"])
                executed_oh = one_hot_action(b["executed_actions"], model.cfg.action_dim)
                world = model.world(
                    workspace, executed_oh,
                    base_bus=target_spatial, base_context=b["context"],
                )
                model_err = F.smooth_l1_loss(
                    world["next_bus"], next_target_spatial, reduction="none"
                ).mean(-1)
                baseline_err = F.smooth_l1_loss(
                    target_spatial, next_target_spatial, reduction="none"
                ).mean(-1)
                wrong_actions = (b["executed_actions"] + 1) % model.cfg.action_dim
                wrong = model.world(
                    workspace, one_hot_action(wrong_actions, model.cfg.action_dim),
                    base_bus=target_spatial, base_context=b["context"],
                )
                wrong_err = F.smooth_l1_loss(
                    wrong["next_bus"], next_target_spatial, reduction="none"
                ).mean(-1)
                reward_abs = (world["reward"] - b["rewards"]).abs()
                success_ok = torch.sigmoid(world["success_logit"]).ge(0.5).eq(
                    b["successes"].ge(0.5)
                ).float()
                temporal_state = torch.zeros(len(chunk), model.cfg.temporal_dim, device=device)
                temporal_bus, _ = model.dynamics_temporal(target_spatial, executed_oh, temporal_state)
                temporal_err = F.smooth_l1_loss(
                    temporal_bus, next_target_spatial, reduction="none"
                ).mean(-1)
                for dest, src in [
                    (world_model_errors, model_err), (world_baseline_errors, baseline_err),
                    (world_wrong_action_errors, wrong_err), (world_reward_abs, reward_abs),
                    (world_success_correct, success_ok), (world_uncertainties, world["uncertainty"]),
                    (temporal_errors, temporal_err),
                ]:
                    dest.extend(src[valid].cpu().tolist())

                context_valid = valid & b["context_valid"].bool()
                if bool(context_valid.any()):
                    dim = model.cfg.dynamics_context_dim
                    current_ctx = b["context"][:, :dim]
                    next_ctx = b["next_context"][:, :dim]
                    ctx_err = F.smooth_l1_loss(
                        world["next_context"], next_ctx, reduction="none"
                    ).mean(-1)
                    ctx_base = F.smooth_l1_loss(
                        current_ctx, next_ctx, reduction="none"
                    ).mean(-1)
                    ctx_wrong = F.smooth_l1_loss(
                        wrong["next_context"], next_ctx, reduction="none"
                    ).mean(-1)
                    world_context_errors.extend(ctx_err[context_valid].cpu().tolist())
                    world_context_baselines.extend(ctx_base[context_valid].cpu().tolist())
                    world_context_wrong_errors.extend(ctx_wrong[context_valid].cpu().tolist())
                    per_dim_err = F.smooth_l1_loss(
                        world["next_context"], next_ctx, reduction="none"
                    )[context_valid]
                    per_dim_base = F.smooth_l1_loss(
                        current_ctx, next_ctx, reduction="none"
                    )[context_valid]
                    world_context_dim_errors.extend(per_dim_err.cpu().tolist())
                    world_context_dim_baselines.extend(per_dim_base.cpu().tolist())

            answer = b["answer_ids"]
            logits = model.language(workspace, b["question_ids"], answer[:, :-1]).argmax(-1)
            target = answer[:, 1:]
            mask = target.ne(BYTE_PAD)
            token_ok += int((logits.eq(target) & mask).sum().item())
            token_total += int(mask.sum().item())

            if len(generated_sims) < max_generated:
                for j in range(min(len(chunk), max_generated - len(generated_sims))):
                    generated = model.language.generate(
                        workspace[j:j+1], b["question_ids"][j:j+1], max_new=96
                    )
                    qas = [q for q in chunk[j].get("qas", []) if q.get("question") and q.get("answer")]
                    expected = [str(q["answer"]) for q in qas] or ["I am still learning."]
                    generated_sims.append(max(
                        difflib.SequenceMatcher(None, generated.lower(), e.lower()).ratio()
                        for e in expected
                    ))

    historical_summary = {pid: float(np.mean(vals)) for pid, vals in historical_cosines.items() if vals}
    retention_min = min(historical_summary.values()) if historical_summary else None
    metrics = {
        "validation_records": len(rows),
        "vision_parent_cosine": float(np.mean(cosines)),
        "vision_historical_cosine_by_parent": historical_summary,
        "vision_retention_min": retention_min,
        "language_token_accuracy": float(token_ok / max(1, token_total)),
        "language_generated_similarity": float(np.mean(generated_sims)) if generated_sims else 0.0,
        "motor_accuracy": float(np.mean(motor_correct)) if motor_correct else 0.0,
        "value_mae": float(np.mean(value_errors)),
    }
    if world_model_errors:
        model_error = float(np.mean(world_model_errors))
        baseline_error = float(np.mean(world_baseline_errors))
        wrong_error = float(np.mean(world_wrong_action_errors))
        spatial_rel = float((baseline_error - model_error) / max(1e-9, baseline_error))
        spatial_adv = float(wrong_error - model_error)
        metrics.update({
            # Backward-compatible bus names plus explicit dorsal/spatial names.
            "world_next_bus_error": model_error,
            "world_no_change_baseline_error": baseline_error,
            "world_wrong_action_error": wrong_error,
            "world_relative_improvement": spatial_rel,
            "world_action_advantage": spatial_adv,
            "world_next_spatial_error": model_error,
            "world_spatial_no_change_baseline_error": baseline_error,
            "world_spatial_wrong_action_error": wrong_error,
            "world_spatial_relative_improvement": spatial_rel,
            "world_spatial_action_advantage": spatial_adv,
            "world_reward_mae": float(np.mean(world_reward_abs)),
            "world_success_accuracy": float(np.mean(world_success_correct)),
            "world_uncertainty_mean": float(np.mean(world_uncertainties)),
            "temporal_next_bus_error": float(np.mean(temporal_errors)),
        })
    if world_context_errors:
        ctx_error = float(np.mean(world_context_errors))
        ctx_baseline = float(np.mean(world_context_baselines))
        ctx_wrong = float(np.mean(world_context_wrong_errors))
        metrics.update({
            "world_next_context_error": ctx_error,
            "world_context_no_change_baseline_error": ctx_baseline,
            "world_context_wrong_action_error": ctx_wrong,
            "world_context_relative_improvement": float(
                (ctx_baseline - ctx_error) / max(1e-9, ctx_baseline)
            ),
            "world_context_action_advantage": float(ctx_wrong - ctx_error),
        })
        if world_context_dim_errors:
            dim_err = np.asarray(world_context_dim_errors, dtype=np.float64).mean(axis=0)
            dim_base = np.asarray(world_context_dim_baselines, dtype=np.float64).mean(axis=0)
            metrics["world_context_error_by_dim"] = [float(x) for x in dim_err]
            metrics["world_context_baseline_by_dim"] = [float(x) for x in dim_base]
            metrics["world_context_relative_improvement_by_dim"] = [
                float((b - e) / max(1e-9, b)) for e, b in zip(dim_err, dim_base)
            ]

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
    executed_action_counts = {}
    for row in rows:
        if bool(row.get("motor_valid", row.get("world_valid", False))):
            a = int(row.get("action", 0)); action_counts[a] = action_counts.get(a, 0) + 1
        if bool(row.get("world_valid", False) and row.get("next_image")):
            ea = int(row.get("executed_action", row.get("action", 0)))
            executed_action_counts[ea] = executed_action_counts.get(ea, 0) + 1
    supported_executed_actions = {a: c for a, c in executed_action_counts.items() if c >= 3}
    care_arr = np.asarray([r.get("care", [0.0] * model.cfg.care_dim) for r in rows], dtype=np.float32)
    qa_count = sum(1 for r in rows if any(q.get("question") and q.get("answer") for q in r.get("qas", [])))
    world_valid_records = sum(bool(r.get("world_valid", False) and r.get("next_image")) for r in rows)
    context_valid_records = sum(bool(
        r.get("world_valid", False) and r.get("next_image") and r.get("next_context") is not None
    ) for r in rows)
    evidence_eligible = {
        "vision": len(rows) >= 50,
        "language": qa_count >= 50,
        "motor": len(action_counts) >= 2 and min(action_counts.values(), default=0) >= 10,
        "values": bool(care_arr.size and float(care_arr.std(axis=0).max()) > 1e-3),
        # Do not let a rare idle/no-op class block the entire causal test. We require
        # at least two meaningfully represented executed actions plus structured targets.
        "world": (
            world_valid_records >= 50 and context_valid_records >= 50
            and len(supported_executed_actions) >= 2
        ),
    }
    evidence = {
        "action_counts": action_counts,
        "executed_action_counts": executed_action_counts,
        "supported_executed_action_counts": supported_executed_actions,
        "qa_records": qa_count,
        "care_max_std": float(care_arr.std(axis=0).max()) if care_arr.size else 0.0,
        "world_valid_records": world_valid_records,
        "world_context_valid_records": context_valid_records,
    }
    thresholds = {"vision": 0.80, "language": 0.70, "motor": 0.70, "values": 0.80}
    evaluated_capabilities = [k for k in thresholds if evidence_eligible.get(k, False)]
    passed = bool(evaluated_capabilities) and all(capability_scores[k] >= thresholds[k] for k in evaluated_capabilities)

    world_visual_passed = None
    world_context_passed = None
    world_passed = None
    if world_model_errors:
        world_visual_passed = bool(
            metrics.get("world_next_bus_error", 1e9) < 0.98 * metrics.get("world_no_change_baseline_error", 0.0)
            and metrics.get("world_action_advantage", -1.0) > 0.0
        )
    if world_context_errors:
        world_context_passed = bool(
            metrics.get("world_next_context_error", 1e9) < 0.80 * metrics.get("world_context_no_change_baseline_error", 0.0)
            and metrics.get("world_context_action_advantage", -1.0) > 0.0
        )
    if evidence_eligible.get("world", False):
        world_passed = bool(
            world_visual_passed is True
            and world_context_passed is True
            and metrics.get("world_success_accuracy", 0.0) >= 0.70
        )
        passed = bool(passed and world_passed)

    result = {
        "format": "continuous_brain.parented_qualification.v095",
        "passed": bool(passed), "metrics": metrics, "capability_scores": capability_scores,
        "evidence": evidence, "evidence_eligible": evidence_eligible,
        "evaluated_capabilities": evaluated_capabilities,
        "world_visual_passed": world_visual_passed,
        "world_spatial_passed": world_visual_passed,
        "world_structured_state_passed": world_context_passed,
        "world_model_passed": world_passed,
        "thresholds": thresholds, "training_updates": int(state.get("updates", 0)),
        "dataset_digest": dataset.digest(),
    }
    out = Path(state_dir) / "infant_evaluation.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    if passed:
        save_checkpoint(
            Path(state_dir) / "qualified_infant.pt", model,
            updates=int(state.get("updates", 0)), parent_id=parent_id,
            dataset_digest=dataset.digest(), metadata={"qualification": result, "world_dynamics_version": "dual-stream-state-conditioned-v2"}
        )
    return result

