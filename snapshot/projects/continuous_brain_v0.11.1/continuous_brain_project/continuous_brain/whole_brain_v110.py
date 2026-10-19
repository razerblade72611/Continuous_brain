from __future__ import annotations

"""v0.11.0 whole-brain growth, rehearsal, and Colab-oriented training.

The release has two distinct stages:
1. explicit function-preserving capacity expansion from a v0.10.x child;
2. routed growth-first rehearsal where only new *.growth.* parameters learn.

Legacy tensors are snapshotted and compared bit-for-bit after every training run.
The trainer is resumable and writes periodic checkpoint snapshots.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .growth_v110 import is_growth_parameter
from .device_v111 import resolve_device
from .infant import InfantBrain, InfantConfig, byte_encode, one_hot_action
from .parented_data import NurseryDataset
from .parented_train import _batch as parented_batch, training_step, load_checkpoint, save_checkpoint
from .planning_v100 import (
    _complete_episode_rows, _split_episodes, _extract_episode_features,
    _sample_windows, _planner_loss,
)
from .world_consolidation_v104 import _world_batch, _world_loss, _world_rows, _balanced_sample
from .planning_v102 import (
    _discover_options, _label_features, _precompute_latents, _flatten,
    _batch as option_batch, _loss as option_loss,
)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _growth_keys(model: InfantBrain) -> set[str]:
    return {name for name, _ in model.named_parameters() if is_growth_parameter(name)}


def _legacy_snapshot(model: InfantBrain) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not is_growth_parameter(k)
    }


def _changed_legacy(snapshot: dict[str, torch.Tensor], model: InfantBrain) -> list[str]:
    current = model.state_dict()
    out = []
    for key, before in snapshot.items():
        if key not in current or not torch.equal(before, current[key].detach().cpu()):
            out.append(key)
    return out


def _parent_dims(saved: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in (saved.get("vision_parent_dims") or {}).items()}


def _maxdiff(a, b) -> float:
    if isinstance(a, dict):
        return max((_maxdiff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (tuple, list)):
        return max((_maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    if torch.is_tensor(a):
        return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())
    return 0.0


@torch.no_grad()
def equivalence_audit(before: InfantBrain, after: InfantBrain, *, seed: int = 11000, device: str = "cpu") -> dict:
    """Compare all major organ interfaces before and immediately after growth."""
    device = resolve_device(device)
    torch.manual_seed(int(seed))
    before.eval().to(device); after.eval().to(device)
    b = 2
    image = torch.rand(b, 3, before.cfg.image_size, before.cfg.image_size, device=device)
    audio = torch.rand(b, 4096, device=device) * 2 - 1
    context = torch.rand(b, before.cfg.context_dim, device=device)
    memory = torch.rand(b, before.cfg.memory_dim, device=device)
    prev_action = torch.tensor([0, 2], dtype=torch.long, device=device)
    action = torch.tensor([1, 4], dtype=torch.long, device=device)
    aoh = one_hot_action(action, before.cfg.action_dim)
    qseq = byte_encode("What is here?", before.cfg.max_question_bytes)
    aseq = byte_encode("I am learning.", before.cfg.max_answer_bytes)
    qids = torch.tensor([qseq, qseq], dtype=torch.long, device=device)
    aids = torch.tensor([aseq, aseq], dtype=torch.long, device=device)

    def probe(m: InfantBrain):
        v = m.vision(image); s = m.spatial(image); fused = m.fuse_vision(v, s)
        nuc = m.nucleus(fused, context, memory, prev_action, m.nucleus.initial_state(b, device))
        ws = nuc["workspace"]
        temporal_state = torch.zeros(b, m.cfg.temporal_dim, device=device)
        tb, ts = m.temporal(v, aoh, temporal_state)
        db, ds = m.dynamics_temporal(s, aoh, temporal_state)
        world = m.world(ws, aoh, base_bus=s, base_context=context)
        pz = m.planning.encode_state(s, context[:, :m.cfg.dynamics_context_dim])
        pstep = m.planning.step(pz, aoh)
        option = torch.tensor([0, min(1, m.cfg.option_count-1)], dtype=torch.long, device=device)
        return {
            "vision": v, "spatial": s, "target_vision": m.target_visual(image),
            "target_spatial": m.target_spatial_bus(image), "fusion": fused, "audio": m.audio(audio),
            "workspace": ws, "router": nuc["router"], "meta": nuc["metacognition"],
            "language": m.language(ws, qids, aids[:, :-1]),
            "temporal": (tb, ts), "dynamics_temporal": (db, ds),
            "world": {k: world[k] for k in ("next_bus","next_context","reward","success_logit","uncertainty")},
            "planning_latent": pz,
            "planning_step": {k: pstep[k] for k in ("latent","spatial","context","reward","success_logit","collision_logit","value","goal_value")},
            "option_selector": m.options.selector_logits(pz),
            "option_term": m.options.termination_logit(pz, option),
            "option_action": m.options.action_logits(pz, option),
            "option_q": m.options.action_value(pz, option, aoh),
            "values": m.values(ws), "motor": m.motor(ws),
        }
    x, y = probe(before), probe(after)
    per = {k: _maxdiff(x[k], y[k]) for k in x}
    return {
        "max_abs_difference": max(per.values(), default=0.0),
        "by_interface": per,
        "exact_function_preservation": all(v == 0.0 for v in per.values()),
        "seed": int(seed),
    }


def expand_capacity(source_state_dir: str | Path, state_dir: str | Path, *,
                    growth_profile: str = "balanced-v110", device: str = "cpu") -> dict:
    device = resolve_device(device)
    source_state_dir = Path(source_state_dir)
    source_checkpoint = source_state_dir / "infant_training.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    source_sha = _sha256_file(source_checkpoint)
    source, saved = load_checkpoint(source_checkpoint, device)
    if bool(source.cfg.growth_enabled):
        raise RuntimeError("source checkpoint is already growth-enabled")

    cfg = replace(source.cfg, growth_enabled=True, growth_profile=str(growth_profile))
    expanded = InfantBrain(cfg, vision_parent_dims=_parent_dims(saved)).to(device)
    missing, unexpected = expanded.load_state_dict(saved["model"], strict=False)
    bad_missing = [k for k in missing if not is_growth_parameter(k)]
    if bad_missing or unexpected:
        raise RuntimeError(f"growth migration mismatch; missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}")

    # Every legacy tensor must be embedded exactly.
    embedded_mismatch = []
    expanded_state = expanded.state_dict()
    for key, value in saved["model"].items():
        if key not in expanded_state or not torch.equal(value.detach().cpu(), expanded_state[key].detach().cpu()):
            embedded_mismatch.append(key)
    if embedded_mismatch:
        raise RuntimeError(f"legacy tensor embedding failed: {embedded_mismatch[:8]}")

    audit = equivalence_audit(source, expanded, device=device)
    if not audit["exact_function_preservation"]:
        raise RuntimeError(f"growth is not function-preserving: max diff {audit['max_abs_difference']}")

    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(saved.get("metadata") or {})
    preserved = {k: v for k, v in metadata.items() if k not in {"updates","parent_id","dataset_digest"}}
    updates = int(saved.get("updates", metadata.get("updates", 0)))
    parent_id = str(metadata.get("parent_id", "vision.dinov3.base"))
    save_checkpoint(
        state_dir / "infant_training.pt", expanded, updates=updates, parent_id=parent_id,
        dataset_digest=str(metadata.get("dataset_digest", "v110-growth-expansion")),
        metadata={
            **preserved,
            "growth_version": "zero-residual-whole-brain-v1",
            "growth_profile": str(growth_profile),
            "growth_source_sha256": source_sha,
            "growth_legacy_tensor_count": len(saved["model"]),
            "growth_new_state_keys": sorted(missing),
            "growth_equivalence_audit": audit,
        },
    )
    for name in ("parents.json", "identity.json"):
        src = source_state_dir / name
        if src.exists(): shutil.copy2(src, state_dir / name)
    out_sha = _sha256_file(state_dir / "infant_training.pt")
    growth_params = sum(p.numel() for n,p in expanded.named_parameters() if is_growth_parameter(n) and p.requires_grad)
    return {
        "format": "continuous_brain.function_preserving_growth.v110",
        "source_checkpoint_sha256": source_sha,
        "expanded_checkpoint_sha256": out_sha,
        "source_trainable_parameters": int(source.parameter_count),
        "expanded_trainable_parameters": int(expanded.parameter_count),
        "new_growth_parameters": int(growth_params),
        "legacy_tensor_mismatches": 0,
        "new_growth_state_keys": len(missing),
        "growth_profile": str(growth_profile),
        "equivalence": audit,
        "checkpoint": str(state_dir / "infant_training.pt"),
    }


def _active_parent_id(ds: NurseryDataset, model: InfantBrain, fallback: str) -> str:
    manifest = ds.manifest()
    pid = str(manifest.get("vision_parent_id") or fallback)
    key = pid.replace(".", "__").replace("/", "_")
    if key not in model.vision_parent_heads:
        # Imported bootstrap sets can have an older active parent; choose any compatible
        # head only when the row target dimension matches.
        rows = ds.records()
        if not rows:
            return fallback
        dim = int(np.load(ds.directory / rows[0]["parent_vision"]).size)
        for k, head in model.vision_parent_heads.items():
            if int(head.out_features) == dim:
                return k.replace("__", ".")
        raise KeyError(f"no child parent head matches dataset {ds.directory}")
    return pid


def _set_route_trainable(model: InfantBrain, route: str) -> list[str]:
    """Growth-first routing: legacy weights remain frozen in v0.11.0 phase A."""
    for p in model.parameters(): p.requires_grad_(False)
    route = str(route)
    prefixes = {
        "perception": ("vision.growth.", "spatial.growth.", "growth_fusion.", "nucleus.growth."),
        "language": ("language.growth.", "nucleus.growth."),
        "embodiment": ("motor.growth.", "values.growth.", "nucleus.growth."),
        "world": ("world.", "dynamics_temporal.growth."),
        "planning": ("planning.growth.",),
        "options": ("options.growth.",),
    }[route]
    names = []
    for name, p in model.named_parameters():
        if not is_growth_parameter(name):
            continue
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad_(True); names.append(name)
    if not names:
        raise RuntimeError(f"route {route} selected no growth parameters")
    return names


def _prepare_planning(model: InfantBrain, dataset_path: str | Path, device: str, seed: int):
    ds = NurseryDataset(dataset_path)
    episodes, dropped = _complete_episode_rows(ds.records())
    train_eps, _ = _split_episodes(episodes, int(seed))
    features = _extract_episode_features(model, ds, train_eps, device, .95, .97)
    return ds, features, dropped



def _make_legacy_anchor(model: InfantBrain, saved: dict, device: str) -> InfantBrain:
    """Reconstruct the exact pre-growth computation from the frozen legacy tensors."""
    cfg = replace(model.cfg, growth_enabled=False, growth_profile="none")
    anchor = InfantBrain(cfg, vision_parent_dims=_parent_dims(saved)).to(device)
    legacy = {k: v for k, v in model.state_dict().items() if not is_growth_parameter(k)}
    missing, unexpected = anchor.load_state_dict(legacy, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"legacy anchor mismatch; missing={missing[:8]} unexpected={unexpected[:8]}")
    for p in anchor.parameters(): p.requires_grad_(False)
    anchor.eval()
    return anchor


def _preservation_loss(model: InfantBrain, anchor: InfantBrain, batch: dict, device: str) -> torch.Tensor:
    """Distill stable pre-growth interfaces while new capacity learns.

    This protects old perception/language/control/planning interfaces without
    constraining the world model, whose purpose is to consolidate new physics.
    """
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    images=b["images"]; context=b["context"]; prev=b["previous_action_tokens"].clamp(0,model.cfg.action_dim)
    memory=torch.zeros(images.shape[0],model.cfg.memory_dim,device=device)
    def forward(m, grad: bool):
        cm = torch.enable_grad() if grad else torch.no_grad()
        with cm:
            v=m.vision(images); sp=m.spatial(images); fused=m.fuse_vision(v,sp)
            nuc=m.nucleus(fused,context,memory,prev,m.nucleus.initial_state(images.shape[0],device)); ws=nuc["workspace"]
            ans=b["answer_ids"][:, :-1]
            lang=m.language(ws,b["question_ids"],ans)
            motor=m.motor(ws); values=m.values(ws)
            stable=m.target_spatial_bus(images)
            pz=m.planning.encode_state(stable,context[:,:m.cfg.dynamics_context_dim])
            sel=m.options.selector_logits(pz)
            opt=sel.argmax(dim=-1)
            oact=m.options.action_logits(pz,opt)
            return {"vision":v,"spatial":sp,"fusion":fused,"workspace":ws,"language":lang,
                    "motor":motor,"values":values,"planning":pz,"selector":sel,"option_action":oact}
    cur=forward(model,True); ref=forward(anchor,False)
    weights={"vision":1.0,"spatial":1.0,"fusion":0.5,"workspace":1.0,"language":0.25,
             "motor":0.5,"values":0.25,"planning":0.5,"selector":0.25,"option_action":0.25}
    parts=[float(weights[k])*F.smooth_l1_loss(cur[k],ref[k]) for k in weights]
    return torch.stack(parts).sum()


def train_whole_brain(rehearsal_datasets: list[str | Path], planning_dataset: str | Path,
                      state_dir: str | Path, *, world_datasets: list[str | Path] | None = None,
                      updates: int = 6000, batch_size: int = 8,
                      learning_rate: float = 1e-4, anchor_weight: float = 0.10,
                      checkpoint_every: int = 500, seed: int = 11010,
                      device: str = "cpu") -> dict:
    """Routed growth-first whole-brain rehearsal with resume-safe checkpoints.

    Routes are deterministic and predeclared: perception, language, world,
    embodiment, planning, options. Planning/options use contiguous lived Detour
    sequences; the other routes use ordinary developmental nursery rehearsal.
    """
    if not rehearsal_datasets:
        raise ValueError("at least one rehearsal dataset is required")
    device = resolve_device(device)
    state_dir = Path(state_dir); checkpoint = state_dir / "infant_training.pt"
    model, saved = load_checkpoint(checkpoint, device)
    if not bool(model.cfg.growth_enabled):
        raise RuntimeError("run expand-capacity before whole-brain training")
    metadata = dict(saved.get("metadata") or {})
    start_route_step = int(metadata.get("whole_brain_route_steps", 0))
    expanded_capacity = int(model.parameter_count)
    growth_capacity = int(sum(
        p.numel() for n,p in model.named_parameters()
        if is_growth_parameter(n) and not n.startswith(("target_vision.", "target_spatial."))
    ))
    legacy_before = _legacy_snapshot(model)
    anchor = _make_legacy_anchor(model, saved, device)

    datasets = [NurseryDataset(p) for p in rehearsal_datasets]
    if any(not ds.records() for ds in datasets):
        raise RuntimeError("one or more rehearsal datasets are empty")
    fallback_parent = str(metadata.get("parent_id", "vision.dinov3.base"))
    parent_ids = [_active_parent_id(ds, model, fallback_parent) for ds in datasets]
    planning_ds, planning_features, dropped = _prepare_planning(model, planning_dataset, device, seed)
    raw_world = []
    for wp in (world_datasets or rehearsal_datasets):
        wds = NurseryDataset(wp)
        try:
            wrows = _world_rows(wds, seed)
        except RuntimeError:
            continue
        raw_world.append((wds, wrows))
    if not raw_world:
        raise RuntimeError("whole-brain training requires at least one world-valid dataset")
    # Reuse successful-sequence option discovery so option growth receives the same
    # semantics as the already-proven v0.10.2 organ rather than new hand-coded phases.
    centers, mean, std = _discover_options(planning_features, model.cfg.option_count, 6, model.cfg.action_dim, seed)
    labeled = _label_features(planning_features, centers, mean, std, 6, model.cfg.action_dim)
    option_rows = _flatten(_precompute_latents(model, labeled, device))
    if not option_rows:
        raise RuntimeError("planning dataset produced no option rehearsal rows")

    # One optimizer owns every growth parameter; per-route requires_grad masks decide
    # which organ receives each update while preserving optimizer momentum on resume.
    all_growth = [p for n,p in model.named_parameters() if is_growth_parameter(n)]
    optimizer = torch.optim.AdamW(all_growth, lr=float(learning_rate), weight_decay=1e-4, foreach=False)
    if saved.get("optimizer") and metadata.get("whole_brain_training_version") == "routed-growth-first-v1":
        try: optimizer.load_state_dict(saved["optimizer"])
        except Exception: pass

    rng = random.Random(int(seed) + start_route_step)
    schedule = ["perception", "language", "world", "embodiment", "planning", "perception", "language", "planning", "world", "options"]
    route_counts = {r: 0 for r in sorted(set(schedule))}
    latest = {}
    model.eval()  # keep legacy BatchNorm/statistical state immutable
    for local in range(1, int(updates)+1):
        global_step = start_route_step + local
        route = schedule[(global_step - 1) % len(schedule)]
        trainable = _set_route_trainable(model, route)
        optimizer.zero_grad(set_to_none=True)
        if route == "planning":
            windows = _sample_windows(planning_features, int(batch_size), 12, rng)
            loss, metrics = _planner_loss(model, windows, device)
        elif route == "options":
            idx = [rng.randrange(len(option_rows)) for _ in range(max(8, int(batch_size)*4))]
            b = option_batch(option_rows, idx, device)
            loss, metrics = option_loss(model, b, q_margin=.08)
        elif route == "world":
            wds, wrows = rng.choice(raw_world)
            selected = _balanced_sample(wrows, max(5, int(batch_size)), model.cfg.action_dim, rng)
            wb = _world_batch(wds, selected, model, device)
            loss, metrics = _world_loss(model, wb)
            # The consolidated world loss intentionally freezes dorsal temporal dynamics;
            # v0.11.0 reopens only its new growth reserve with a child-owned next-spatial target.
            with torch.no_grad():
                base_sp = model.target_spatial_bus(wb["images"])
                next_sp = model.target_spatial_bus(wb["next_images"])
            aoh = one_hot_action(wb["executed_actions"], model.cfg.action_dim)
            tstate = torch.zeros(base_sp.shape[0], model.cfg.temporal_dim, device=device)
            temporal_pred, _ = model.dynamics_temporal(base_sp, aoh, tstate)
            temporal_growth_loss = F.smooth_l1_loss(temporal_pred, next_sp)
            loss = loss + 0.50 * temporal_growth_loss
            metrics = {**metrics, "temporal_growth": float(temporal_growth_loss.detach())}
        else:
            di = rng.randrange(len(datasets)); ds = datasets[di]
            rows = ds.records(); selected = [rng.choice(rows) for _ in range(int(batch_size))]
            batch = parented_batch(ds, selected, model.cfg, rng)
            loss, metrics = training_step(model, batch, parent_ids[di], device, world_focus=False)
        anchor_loss = torch.zeros((), device=device)
        # Every fifth routed update uses a separate rehearsal minibatch to anchor
        # stable pre-growth interfaces. The reference is reconstructed only from
        # frozen legacy tensors; no external parent model is queried.
        if float(anchor_weight) > 0 and global_step % 5 == 0:
            adi = rng.randrange(len(datasets)); ads = datasets[adi]
            arows = ads.records(); aselected = [rng.choice(arows) for _ in range(int(batch_size))]
            abatch = parented_batch(ads, aselected, model.cfg, rng)
            anchor_loss = _preservation_loss(model, anchor, abatch, device)
            loss = loss + float(anchor_weight) * anchor_loss
        # Very small growth norm prevents unbounded residual scale without being a
        # substitute for the actual rehearsal anchor above.
        reg = torch.stack([(p.float().pow(2).mean()) for p in all_growth if p.requires_grad]).mean()
        loss = loss + 1e-6 * reg
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_([p for p in all_growth if p.requires_grad], 1.0, foreach=False))
        optimizer.step()
        route_counts[route] += 1
        latest = {"route": route, "loss": float(loss.detach()), "grad": grad, "trainable_count": len(trainable),
                  "anchor_loss": float(anchor_loss.detach()), **{k: float(v) for k,v in metrics.items() if isinstance(v,(int,float))}}
        if local == 1 or local % 100 == 0 or local == int(updates):
            print(f"whole_brain_step={global_step} route={route} loss={latest['loss']:.5f} grad={grad:.3f}")
        if int(checkpoint_every) > 0 and local % int(checkpoint_every) == 0:
            save_checkpoint(
                checkpoint, model, updates=int(saved.get("updates", 0)), parent_id=fallback_parent,
                dataset_digest="v110-routed-rehearsal", optimizer=optimizer,
                metadata={**{k:v for k,v in metadata.items() if k not in {"updates","parent_id","dataset_digest"}},
                          "whole_brain_training_version":"routed-growth-first-v1",
                          "whole_brain_route_steps":global_step,
                          "whole_brain_seed":int(seed),
                          "whole_brain_schedule":schedule},
            )
            snap = state_dir / f"infant_training.v110_route_{global_step:07d}.pt"
            shutil.copy2(checkpoint, snap)

    changed_legacy = _changed_legacy(legacy_before, model)
    if changed_legacy:
        raise RuntimeError(f"whole-brain growth training changed legacy tensors/buffers: {changed_legacy[:8]}")
    final_step = start_route_step + int(updates)
    save_checkpoint(
        checkpoint, model, updates=int(saved.get("updates", 0)), parent_id=fallback_parent,
        dataset_digest="v110-routed-rehearsal", optimizer=optimizer,
        metadata={**{k:v for k,v in metadata.items() if k not in {"updates","parent_id","dataset_digest"}},
                  "whole_brain_training_version":"routed-growth-first-v1",
                  "whole_brain_route_steps":final_step,
                  "whole_brain_seed":int(seed),
                  "whole_brain_schedule":schedule,
                  "whole_brain_planning_dataset":str(Path(planning_dataset).resolve()),
                  "whole_brain_rehearsal_datasets":[str(Path(p).resolve()) for p in rehearsal_datasets],
                  "whole_brain_world_datasets":[str(Path(p).resolve()) for p in (world_datasets or rehearsal_datasets)]},
    )
    return {
        "format":"continuous_brain.whole_brain_training.v111",
        "device":str(device),
        "device_backend":device.type,
        "checkpoint":str(checkpoint),
        "checkpoint_sha256":_sha256_file(checkpoint),
        "growth_profile":model.cfg.growth_profile,
        "trainable_parameters":expanded_capacity,
        "growth_parameters":growth_capacity,
        "legacy_tensors_changed":0,
        "plasticity":"growth-only",
        "route_steps_before":start_route_step,
        "route_steps_after":final_step,
        "route_counts":route_counts,
        "schedule":schedule,
        "dropped_incomplete_planning_episodes":dropped,
        "latest":latest,
    }
