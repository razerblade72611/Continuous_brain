from __future__ import annotations

"""v0.11.4 protected growth integration.

v0.11.0 proved that the child can add ~44M parameters while keeping every legacy
weight bit-exact, but its fully visible planning/option residuals degraded the
previously qualified P4b behavior (38% -> 13%) and did not yet generalize.

v0.11.4 keeps the same learned capacity and introduces *visibility protection*:
planning/option residual branches can learn behind a closed gate via the
explicit auxiliary shadow forward in :class:`ProtectedZeroResidualStack`, then become
visible only through a predeclared 0% -> 2% -> 5% -> 10% schedule.  Mature
planning interfaces are explicitly distilled against the exact pre-growth child
on every planning/option update.

No external parent model participates in protected growth training.  The old
child remains frozen.  Successful option rehearsal comes only from lived
successful sequences already present in the nursery.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from .device_v111 import resolve_device
from .dml_ops_v121 import safe_one_hot
from .growth_v110 import ProtectedZeroResidualStack, is_growth_parameter
from .infant import InfantBrain, one_hot_action
from .parented_data import NurseryDataset
from .parented_train import _batch as parented_batch, training_step, load_checkpoint, save_checkpoint
from .planning_v100 import _sample_windows, _planner_loss
from .planning_v102 import (
    _successful, _precompute_latents, _flatten,
    _batch as option_batch, _loss as option_loss, _option_q_all,
)
from .world_consolidation_v104 import _world_batch, _world_loss, _world_rows, _balanced_sample
from .whole_brain_v110 import (
    _active_parent_id,
    _make_legacy_anchor, _parent_dims, _prepare_planning, _preservation_loss,
    _set_route_trainable, equivalence_audit,
)


PROTECTED_GROWTH_VERSION = "protected-shadow-gates-v1"
TRAINING_VERSION = "protected-routed-growth-v2"
from .evidence_v114 import (BUILD_ID, copy_verified, write_json, valid_p4b)


def legacy_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if is_growth_parameter(name):
            continue
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@contextmanager
def shadow_forward(model, enabled=True):
    modules = [model.planning.growth, model.options.growth]
    old = [m.shadow_training for m in modules]
    try:
        for m in modules:
            m.shadow_training = enabled
        yield
    finally:
        for m, value in zip(modules, old):
            m.shadow_training = value


def _training_dataset(path):
    path = Path(path)
    if not (path / "lessons.jsonl").is_file():
        raise FileNotFoundError(f"Training nursery is missing: {path}")
    ds = NurseryDataset(path)
    manifest, records = ds.manifest(), ds.records()
    if manifest.get("training_prohibited") or any(r.get("training_prohibited") for r in records):
        raise RuntimeError(f"Training-prohibited holdout: {path}")
    # Evaluation ranges must never enter rehearsal through copied/generated rows.
    for row in [manifest, *records]:
        for key in ("seed", "episode_seed", "seed_start", "seed_end"):
            value = row.get(key)
            if isinstance(value, int) and (41001 <= value <= 41100 or 51001 <= value <= 51100):
                raise RuntimeError(f"Evaluation seeds are prohibited in training: {path}")
    return ds


@torch.no_grad()
def _teacher_option_rows(anchor, features, device):
    # The historical checkpoint did not save its k-means centers. Preserve option
    # identities using the frozen selector, rather than assigning fresh cluster IDs.
    rows = []
    for ep in features:
        if not _successful(ep):
            continue
        item = dict(ep)
        z = anchor.planning.encode_state(ep["spatial"].to(device), ep["context"].to(device))
        labels = anchor.options.selector_logits(z).argmax(-1).cpu()
        item["option"] = labels
        item["terminate"] = torch.cat(((labels[:-1] != labels[1:]).float(), torch.ones(1)))
        flattened = _flatten(_precompute_latents(anchor, [item], device))
        for t, row in enumerate(flattened):
            for key in ("spatial", "context", "next_spatial", "next_context"):
                row[key] = ep[key][t]
        rows.extend(flattened)
    return rows


def _live_option_batch(model, rows, rng, batch_size, device):
    inds = [rng.randrange(len(rows)) for _ in range(max(8, int(batch_size) * 4))]
    batch = option_batch(rows, inds, device)
    batch["ref_z"] = batch["z"]
    for key in ("spatial", "context", "next_spatial", "next_context"):
        batch[key] = torch.stack([rows[i][key] for i in inds]).to(device)
    batch["z"] = model.planning.encode_state(batch["spatial"], batch["context"])
    batch["next_z"] = model.planning.encode_state(batch["next_spatial"], batch["next_context"])
    return batch


def _rank_anchor(current, reference):
    # Only preserve strict teacher order; do not impose arbitrary ties.
    ref_diff = reference[:, :, None] - reference[:, None, :]
    cur_diff = current[:, :, None] - current[:, None, :]
    mask = ref_diff > 1e-6
    penalties = F.relu(ref_diff.clamp(max=.08) - cur_diff)
    return (penalties * mask).sum() / mask.sum().clamp(min=1)


STAGE_PLAN = {
    "shadow": {
        "planner_gate": 0.00,
        "option_gate": 0.00,
        "general_gate": 1.00,
        "target_gate": 0.00,
        "target_updates": 2000,
        "description": "Planning/options learn behind closed interfaces; visible behavior remains legacy-exact at those organs.",
    },
    "micro": {
        "planner_gate": 0.02,
        "option_gate": 0.02,
        "general_gate": 1.00,
        "target_gate": 0.00,
        "target_updates": 1000,
        "description": "Expose two percent of protected planning/option residual influence.",
    },
    "low": {
        "planner_gate": 0.05,
        "option_gate": 0.05,
        "general_gate": 1.00,
        "target_gate": 0.00,
        "target_updates": 1000,
        "description": "Expose five percent only after earlier regression gates pass.",
    },
    "moderate": {
        "planner_gate": 0.10,
        "option_gate": 0.10,
        "general_gate": 1.00,
        "target_gate": 0.00,
        "target_updates": 1000,
        "description": "Final predeclared protected-development exposure ceiling for v0.11.4.",
    },
}
STAGE_ORDER = tuple(STAGE_PLAN)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_gate_key(name: str) -> bool:
    return name.endswith(".gate") and (".growth." in f".{name}." or name.startswith("growth_"))


def _protected_modules(model: InfantBrain) -> dict[str, ProtectedZeroResidualStack]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ProtectedZeroResidualStack)
    }


def set_protected_gates(model: InfantBrain, *, general_gate: float,
                        planner_gate: float, option_gate: float,
                        target_gate: float = 0.0) -> dict[str, float]:
    """Set non-trainable visibility gates by functional role.

    The EMA target cortices stay at zero visible growth in v0.11.4 so planning
    coordinates and world targets remain anchored to the already-qualified child.
    """
    modules = _protected_modules(model)
    if not modules:
        raise RuntimeError("checkpoint has no protected growth modules; run protect-growth first")
    report: dict[str, float] = {}
    for name, module in modules.items():
        if name.startswith(("target_vision.growth", "target_spatial.growth")):
            value = float(target_gate)
        elif name.startswith("planning.growth"):
            value = float(planner_gate)
        elif name.startswith("options.growth"):
            value = float(option_gate)
        else:
            value = float(general_gate)
        module.set_gate(value)
        report[name] = float(module.gate.item())
    return report


def gate_report(model: InfantBrain) -> dict[str, float]:
    return {name: float(module.gate.item()) for name, module in _protected_modules(model).items()}


def protect_growth(source_state_dir: str | Path, state_dir: str | Path, *,
                   device: str = "cpu") -> dict:
    """Migrate an exact v0.11 growth-expanded child into protected v0.11.4 form.

    Existing model parameters/buffers are copied bit-for-bit.  The only new state
    entries are non-trainable ``*.gate`` buffers.  The migration intentionally
    starts from the exact expansion branch (before v0.11.0 Phase A interference).
    """
    state_dir = Path(state_dir)
    if state_dir.exists() and any(state_dir.iterdir()):
        raise FileExistsError("Protected destination must be new and empty; existing state is preserved")
    device = resolve_device(device)
    source_state_dir = Path(source_state_dir)
    source_checkpoint = source_state_dir / "infant_training.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    source_sha = _sha256_file(source_checkpoint)
    source, saved = load_checkpoint(source_checkpoint, device)
    if not bool(source.cfg.growth_enabled):
        raise RuntimeError("protect-growth requires an already expanded v0.11 growth checkpoint")
    if bool(getattr(source.cfg, "protected_growth_enabled", False)):
        raise RuntimeError("source is already protected")
    metadata = dict(saved.get("metadata") or {})
    if int(metadata.get("whole_brain_route_steps", 0)) != 0:
        raise RuntimeError(
            "v0.11.4 must branch from the exact pre-Phase-A expansion checkpoint; "
            "whole_brain_route_steps must be 0"
        )

    if saved.get("_v094_missing_keys"):
        raise RuntimeError("Exact expansion source has missing tensors")
    for key, value in saved["model"].items():
        if is_growth_parameter(key) and ".out_proj." in key and torch.count_nonzero(value).item():
            raise RuntimeError("Source contains learned growth residuals; use the untouched exact expansion")
    cfg = replace(source.cfg, protected_growth_enabled=True)
    protected = InfantBrain(cfg, vision_parent_dims=_parent_dims(saved)).to(device)
    missing, unexpected = protected.load_state_dict(saved["model"], strict=False)
    bad_missing = [k for k in missing if not _is_gate_key(k)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"protected migration mismatch; missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}"
        )

    # Verify every pre-v0.11.4 tensor is copied exactly, including the 44M growth
    # weights already present in the exact expansion checkpoint.
    current = protected.state_dict()
    mismatches = [
        key for key, value in saved["model"].items()
        if key not in current or not torch.equal(value.detach().cpu(), current[key].detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"protected migration altered source tensors: {mismatches[:8]}")

    gates = set_protected_gates(
        protected,
        general_gate=1.0,
        planner_gate=0.0,
        option_gate=0.0,
        target_gate=0.0,
    )
    audit = equivalence_audit(source, protected, device=device)
    if not audit["exact_function_preservation"]:
        raise RuntimeError(f"protected migration is not exact: {audit['max_abs_difference']}")

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    preserved = {k: v for k, v in metadata.items() if k not in {"updates", "parent_id", "dataset_digest"}}
    updates = int(saved.get("updates", metadata.get("updates", 0)))
    parent_id = str(metadata.get("parent_id", "vision.dinov3.base"))
    save_checkpoint(
        state_dir / "infant_training.pt", protected,
        updates=updates,
        parent_id=parent_id,
        dataset_digest=str(metadata.get("dataset_digest", "v114-protected-growth")),
        metadata={
            **preserved,
            "protected_growth_version": PROTECTED_GROWTH_VERSION,
            "protected_growth_build_id": BUILD_ID,
            "protected_growth_source_legacy_sha256": legacy_digest(source),
            "protected_growth_source_sha256": source_sha,
            "protected_growth_gate_plan": STAGE_PLAN,
            "protected_growth_gates": gates,
            "protected_growth_stage_updates": {},
            "protected_growth_stage_history": [],
            "protected_growth_equivalence_audit": audit,
            "protected_growth_new_gate_keys": sorted(missing),
        },
    )
    for name in ("parents.json", "identity.json"):
        src = source_state_dir / name
        if src.exists():
            shutil.copy2(src, state_dir / name)
    out_sha = _sha256_file(state_dir / "infant_training.pt")
    result = {
        "build_id": BUILD_ID,
        "source_legacy_sha256": legacy_digest(source),
        "format": "continuous_brain.protected_growth_migration.v114",
        "source_checkpoint_sha256": source_sha,
        "protected_checkpoint_sha256": out_sha,
        "source_tensor_mismatches": 0,
        "new_gate_state_keys": len(missing),
        "learned_parameters_added": 0,
        "exact_function_preservation": True,
        "max_abs_difference": 0.0,
        "gate_report": gates,
        "stage_plan": STAGE_PLAN,
        "checkpoint": str(state_dir / "infant_training.pt"),
    }

    write_json(state_dir / "V114_MIGRATION.json", result)
    return result


def _planner_compatibility(model: InfantBrain, anchor: InfantBrain, windows,
                           device: str) -> tuple[torch.Tensor, dict[str, float]]:
    sp = torch.cat([ep["spatial"][st:st+1] for ep, st, _ in windows], 0).to(device)
    ctx = torch.cat([ep["context"][st:st+1] for ep, st, _ in windows], 0).to(device)
    actions = torch.stack([ep["action"][st] for ep, st, _ in windows]).to(device)
    aoh = one_hot_action(actions, model.cfg.action_dim)

    cur_z = model.planning.encode_state(sp, ctx)
    with torch.no_grad():
        ref_z = anchor.planning.encode_state(sp, ctx)
    latent = F.smooth_l1_loss(cur_z, ref_z)

    b, a = cur_z.shape[0], model.cfg.action_dim
    all_actions = torch.arange(a, device=device)[None, :].expand(b, a)
    all_oh = safe_one_hot(all_actions, a)
    cur_rep = cur_z[:, None, :].expand(b, a, cur_z.shape[-1]).reshape(b*a, -1)
    ref_rep = ref_z[:, None, :].expand(b, a, ref_z.shape[-1]).reshape(b*a, -1)
    oh_flat = all_oh.reshape(b*a, a)
    cur_q = model.planning.action_value(cur_rep, oh_flat).reshape(b, a)
    with torch.no_grad():
        ref_q = anchor.planning.action_value(ref_rep, oh_flat).reshape(b, a)
    q_value = F.smooth_l1_loss(cur_q, ref_q)
    q_order = _rank_anchor(cur_q, ref_q)
    q_rank = F.kl_div(
        F.log_softmax(cur_q, dim=-1),
        F.softmax(ref_q, dim=-1),
        reduction="batchmean",
    )

    cur_next = model.planning.step(cur_z, aoh)["latent"]
    with torch.no_grad():
        ref_next = anchor.planning.step(ref_z, aoh)["latent"]
    transition = F.smooth_l1_loss(cur_next, ref_next)
    total = 1.5 * latent + 1.0 * transition + 0.75 * q_value + 0.75 * q_rank + q_order
    return total, {
        "compat_planning_latent": float(latent.detach()),
        "compat_planning_transition": float(transition.detach()),
        "compat_planning_q_order": float(q_order.detach()),
        "compat_planning_q": float(q_value.detach()),
        "compat_planning_q_rank": float(q_rank.detach()),
    }


def _option_compatibility(model: InfantBrain, anchor: InfantBrain, b: dict) -> tuple[torch.Tensor, dict[str, float]]:
    z = b["z"]
    option = b["option"].long()
    ref_z = b.get("ref_z", z)
    cur_sel = model.options.selector_logits(z)
    cur_action = model.options.action_logits(z, option)
    cur_term = model.options.termination_logit(z, option)
    cur_q = _option_q_all(model, z, option)
    with torch.no_grad():
        ref_sel = anchor.options.selector_logits(ref_z)
        ref_action = anchor.options.action_logits(ref_z, option)
        ref_term = anchor.options.termination_logit(ref_z, option)
        ref_q = _option_q_all(anchor, ref_z, option)

    selector = F.kl_div(F.log_softmax(cur_sel, -1), F.softmax(ref_sel, -1), reduction="batchmean")
    action = F.kl_div(F.log_softmax(cur_action, -1), F.softmax(ref_action, -1), reduction="batchmean")
    termination = F.smooth_l1_loss(cur_term, ref_term)
    q_value = F.smooth_l1_loss(cur_q, ref_q)
    q_rank = F.kl_div(F.log_softmax(cur_q, -1), F.softmax(ref_q, -1), reduction="batchmean")
    q_order = _rank_anchor(cur_q, ref_q)
    total = selector + 1.5 * action + 0.5 * termination + 0.75 * q_value + 0.75 * q_rank + q_order
    return total, {
        "compat_option_selector": float(selector.detach()),
        "compat_option_action": float(action.detach()),
        "compat_option_termination": float(termination.detach()),
        "compat_option_q_order": float(q_order.detach()),
        "compat_option_q": float(q_value.detach()),
        "compat_option_q_rank": float(q_rank.detach()),
    }


def _validate_stage_order(metadata: dict, stage: str) -> None:
    if stage not in STAGE_PLAN:
        raise ValueError(f"stage must be one of {', '.join(STAGE_ORDER)}")
    stage_updates = {str(k): int(v) for k, v in (metadata.get("protected_growth_stage_updates") or {}).items()}
    index = STAGE_ORDER.index(stage)
    for prior in STAGE_ORDER[:index]:
        required = int(STAGE_PLAN[prior]["target_updates"])
        if int(stage_updates.get(prior, 0)) < required:
            raise RuntimeError(
                f"stage {stage} cannot start until {prior} has completed its predeclared {required} updates"
            )


def train_protected_growth(rehearsal_datasets, planning_dataset, state_dir, *, stage,
                           planning_rehearsal_dataset, previous_p4b_result=None,
                           durable_state_dir=None, world_datasets=None,
                           target_stage_updates=None, batch_size=8, learning_rate=1e-4,
                           anchor_weight=.10, interface_anchor_weight=1.0,
                           checkpoint_every=250, seed=11410, device="cpu"):
    """Train a bounded stage; persist measured receipts and verify stage admission."""
    if not rehearsal_datasets:
        raise ValueError("At least one rehearsal dataset is required")
    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    device = resolve_device(device)
    model, saved = load_checkpoint(checkpoint, device)
    metadata = dict(saved.get("metadata") or {})
    if metadata.get("protected_growth_build_id") != BUILD_ID:
        raise RuntimeError("This build requires a new protected branch from the exact expansion")
    _validate_stage_order(metadata, stage)
    stage_updates = dict(metadata.get("protected_growth_stage_updates") or {})
    if any(stage_updates.get(s, 0) for s in STAGE_ORDER[STAGE_ORDER.index(stage)+1:]):
        raise RuntimeError("Cannot reopen an earlier stage after a later stage has started")
    spec = STAGE_PLAN[stage]
    target = spec["target_updates"] if target_stage_updates is None else int(target_stage_updates)
    if not 0 <= target <= spec["target_updates"]:
        raise ValueError("Target must be within the predeclared stage update ceiling")
    if batch_size < 1 or checkpoint_every < 1 or learning_rate <= 0:
        raise ValueError("Batch, checkpoint interval, and learning rate must be positive")
    if not np.isfinite([learning_rate, anchor_weight, interface_anchor_weight]).all():
        raise ValueError("Training coefficients must be finite")
    if anchor_weight <= 0 or interface_anchor_weight <= 0:
        raise ValueError("Protected training requires positive compatibility anchors")
    before_stage = int(stage_updates.get(stage, 0))
    if target < before_stage:
        raise RuntimeError("Target is behind current stage progress; refusing to rewind")
    legacy_sha = legacy_digest(model)
    source_legacy = metadata["protected_growth_source_legacy_sha256"]
    if legacy_sha != source_legacy:
        raise RuntimeError("Frozen legacy tensors differ from the migration fingerprint")
    if before_stage == target and metadata.get("protected_growth_last_measurement"):
        receipt = dict(metadata["protected_growth_last_measurement"])
        receipt.update(checkpoint_sha256=_sha256_file(checkpoint), checkpoint=str(checkpoint))
        write_json(state_dir / f"V114_{stage.upper()}_TRAINING.json", receipt)
        return receipt

    admission = dict(metadata.get("protected_growth_admission") or {})
    if before_stage == 0:
        if previous_p4b_result is None:
            raise RuntimeError("A matching baseline/previous-stage P4b result is required before opening a stage")
        prior = json.loads(Path(previous_p4b_result).read_text())
        expected = (metadata["protected_growth_source_sha256"] if stage == "shadow"
                    else _sha256_file(checkpoint))
        if not valid_p4b(prior, expected):
            raise RuntimeError("P4b evidence failed, uses different settings, or belongs to another checkpoint")
        admission[stage] = {"checkpoint_sha256": expected,
                            "result_sha256": _sha256_file(previous_p4b_result)}
    elif stage not in admission:
        raise RuntimeError("Resumed stage is missing its admission evidence")

    paths = dict(rehearsal=list(rehearsal_datasets),
                 world=list(world_datasets or rehearsal_datasets),
                 planning=[planning_dataset], planning_rehearsal=[planning_rehearsal_dataset])
    dataset_evidence = {}
    for group, group_paths in paths.items():
        dataset_evidence[group] = []
        for path in group_paths:
            ds = _training_dataset(path)
            h = hashlib.sha256()
            for file in sorted(Path(path).rglob("*")):
                if file.is_file():
                    h.update(str(file.relative_to(path)).encode())
                    h.update(bytes.fromhex(_sha256_file(file)))
            dataset_evidence[group].append(h.hexdigest())
    protocol = dict(build_id=BUILD_ID, datasets=dataset_evidence, batch_size=int(batch_size),
                    learning_rate=float(learning_rate), seed=int(seed),
                    anchor_weight=float(anchor_weight), interface_anchor_weight=float(interface_anchor_weight))
    if metadata.get("protected_growth_protocol") not in (None, protocol):
        raise RuntimeError("Resume training data/settings differ from the recorded protocol")

    set_protected_gates(model, general_gate=spec["general_gate"],
                        planner_gate=spec["planner_gate"], option_gate=spec["option_gate"], target_gate=0)
    anchor = _make_legacy_anchor(model, saved, device)
    datasets = [_training_dataset(p) for p in rehearsal_datasets]
    rows_by_dataset = [ds.records() for ds in datasets]
    if any(not rows for rows in rows_by_dataset):
        raise RuntimeError("Empty rehearsal nursery")
    fallback_parent = str(metadata.get("parent_id", "vision.dinov3.base"))
    parent_ids = [_active_parent_id(ds, model, fallback_parent) for ds in datasets]
    _, new_features, dropped_new = _prepare_planning(anchor, planning_dataset, device, seed)
    _, old_features, dropped_old = _prepare_planning(anchor, planning_rehearsal_dataset, device, seed)
    if not new_features or not old_features:
        raise RuntimeError("Both new and old planning nurseries need complete training episodes")
    new_options = _teacher_option_rows(anchor, new_features, device)
    old_options = _teacher_option_rows(anchor, old_features, device)
    if not new_options or not old_options:
        raise RuntimeError("Both planning nurseries need successful lived option trajectories")
    raw_world = []
    for path in paths["world"]:
        ds = _training_dataset(path)
        raw_world.append((ds, _world_rows(ds, seed)))
    if not raw_world:
        raise RuntimeError("No world training data")

    all_growth = [p for n, p in model.named_parameters() if is_growth_parameter(n)]
    optimizer = torch.optim.AdamW(all_growth, lr=learning_rate, weight_decay=1e-4, foreach=False)
    if metadata.get("protected_growth_total_updates", 0):
        if metadata.get("protected_growth_training_version") != TRAINING_VERSION or not saved.get("optimizer"):
            raise RuntimeError("Cannot resume without the matching optimizer state")
        optimizer.load_state_dict(saved["optimizer"])
    total_before = int(metadata.get("protected_growth_total_updates", 0))
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    if metadata.get("protected_growth_rng_state") is not None:
        rng.setstate(metadata["protected_growth_rng_state"])
        torch.set_rng_state(metadata["protected_growth_torch_rng"].cpu())
        if str(device).startswith("cuda") and metadata.get("protected_growth_cuda_rng"):
            torch.cuda.set_rng_state_all([t.cpu() for t in metadata["protected_growth_cuda_rng"]])
    schedule = ["perception", "language", "world", "embodiment", "planning",
                "perception", "language", "planning", "world", "options"]
    counts_by_stage = dict(metadata.get("protected_growth_route_counts") or {})
    counts = dict(counts_by_stage.get(stage) or {r: 0 for r in sorted(set(schedule))})
    latest = {}
    model.eval()
    remaining = target - before_stage

    def persist(stage_step, global_step):
        measured_sha = legacy_digest(model)
        if measured_sha != source_legacy:
            raise RuntimeError("Frozen legacy integrity failed; checkpoint was not saved")
        if any(not torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()):
            raise RuntimeError("Non-finite model state; checkpoint was not saved")
        measurement = dict(format="continuous_brain.protected_growth_training.v114", build_id=BUILD_ID,
                           stage=stage, stage_target_updates=spec["target_updates"],
                           stage_updates_after=stage_step, protected_growth_total_updates=global_step,
                           legacy_tensors_changed=0, legacy_sha256=measured_sha,
                           source_legacy_sha256=source_legacy, plasticity="growth-only-protected",
                           gate_report=gate_report(model), route_counts=dict(counts), schedule=schedule,
                           interface_anchor_weight=float(interface_anchor_weight), ordinary_anchor_weight=float(anchor_weight),
                           successful_option_rehearsal_rows=len(old_options), new_option_rows=len(new_options),
                           option_label_source="frozen-legacy-selector", training_protocol=protocol,
                           source_checkpoint_sha256=metadata["protected_growth_source_sha256"],
                           stage_admission=admission[stage], latest=latest,
                           dropped_incomplete_planning_episodes=dropped_new + dropped_old)
        stage_updates[stage] = stage_step
        counts_by_stage[stage] = dict(counts)
        saved_metadata = {**metadata, "protected_growth_training_version": TRAINING_VERSION,
                          "protected_growth_current_stage": stage,
                          "protected_growth_stage_updates": dict(stage_updates),
                          "protected_growth_total_updates": global_step,
                          "protected_growth_gates": gate_report(model),
                          "protected_growth_protocol": protocol, "protected_growth_admission": admission,
                          "protected_growth_route_counts": counts_by_stage,
                          "protected_growth_rng_state": rng.getstate(),
                          "protected_growth_torch_rng": torch.get_rng_state(),
                          "protected_growth_cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
                          "protected_growth_last_measurement": measurement}
        save_checkpoint(checkpoint, model, updates=int(saved.get("updates", 0)), parent_id=fallback_parent,
                        dataset_digest="v114-protected-routed-rehearsal", optimizer=optimizer,
                        metadata=saved_metadata)
        receipt = {**measurement, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint)}
        receipt_path = write_json(state_dir / f"V114_{stage.upper()}_TRAINING.json", receipt)
        if durable_state_dir:
            durable = Path(durable_state_dir)
            # A previous known-good file remains available through an interrupted transfer.
            if (durable / "infant_training.pt").exists():
                copy_verified(durable / "infant_training.pt", durable / "infant_training.previous.pt")
            copy_verified(checkpoint, durable / checkpoint.name)
            copy_verified(receipt_path, durable / receipt_path.name)
        print(f"protected_checkpoint_saved stage={stage} step={stage_step} sha256={receipt['checkpoint_sha256']}", flush=True)
        return receipt

    for local in range(1, remaining + 1):
        stage_step, global_step = before_stage + local, total_before + local
        route = schedule[(global_step - 1) % len(schedule)]
        names = _set_route_trainable(model, route)
        optimizer.zero_grad(set_to_none=True)
        compat = torch.zeros((), device=device)
        cm = {}
        if route in {"planning", "options"}:
            # Shadow losses evaluate the actual grown function. Checkpoint gates
            # remain at zero; only this explicit auxiliary pass is fully visible.
            with shadow_forward(model, stage == "shadow"):
                if route == "planning":
                    features = old_features if rng.random() < .5 else new_features
                    windows = _sample_windows(features, batch_size, 12, rng)
                    loss, metrics = _planner_loss(model, windows, device)
                else:
                    options = old_options if rng.random() < .5 else new_options
                    batch = _live_option_batch(model, options, rng, batch_size, device)
                    loss, metrics = option_loss(model, batch, q_margin=.08)
                old_windows = _sample_windows(old_features, batch_size, 12, rng)
                pc, pm = _planner_compatibility(model, anchor, old_windows, device)
                old_batch = _live_option_batch(model, old_options, rng, batch_size, device)
                oc, om = _option_compatibility(model, anchor, old_batch)
                compat = pc + oc
                cm = {**pm, **om}
                loss = loss + interface_anchor_weight * compat
        elif route == "world":
            ds, rows = rng.choice(raw_world)
            selected = _balanced_sample(rows, max(5, batch_size), model.cfg.action_dim, rng)
            batch = _world_batch(ds, selected, model, device)
            loss, metrics = _world_loss(model, batch)
            with torch.no_grad():
                base_sp = model.target_spatial_bus(batch["images"])
                next_sp = model.target_spatial_bus(batch["next_images"])
            aoh = one_hot_action(batch["executed_actions"], model.cfg.action_dim)
            tstate = torch.zeros(base_sp.shape[0], model.cfg.temporal_dim, device=device)
            prediction, _ = model.dynamics_temporal(base_sp, aoh, tstate)
            loss = loss + .5 * F.smooth_l1_loss(prediction, next_sp)
        else:
            index = rng.randrange(len(datasets))
            selected = [rng.choice(rows_by_dataset[index]) for _ in range(batch_size)]
            batch = parented_batch(datasets[index], selected, model.cfg, rng)
            loss, metrics = training_step(model, batch, parent_ids[index], device, world_focus=False)
        # The old modulo-5 schedule anchored only planning/options, because route
        # positions 5 and 10 were always those routes. Anchor every active route.
        index = rng.randrange(len(datasets))
        selected = [rng.choice(rows_by_dataset[index]) for _ in range(batch_size)]
        anchor_batch = parented_batch(datasets[index], selected, model.cfg, rng)
        anchor_loss = _preservation_loss(model, anchor, anchor_batch, device)
        loss = loss + anchor_weight * anchor_loss
        active = [p for p in all_growth if p.requires_grad]
        reg = torch.stack([p.float().square().mean() for p in active]).mean()
        loss = loss + 1e-6 * reg
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss; refusing optimizer update")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(active, 1.0, foreach=False))
        if not np.isfinite(grad):
            raise RuntimeError("Non-finite gradient; refusing optimizer update")
        optimizer.step()
        counts[route] += 1
        latest = {**{k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
                  **cm, "route": route, "loss": float(loss.detach()), "grad": grad,
                  "anchor_loss": float(anchor_loss.detach()), "interface_compatibility_loss": float(compat.detach()),
                  "trainable_count": len(names), "stage_step": stage_step, "global_step": global_step}
        if local == 1 or global_step % 100 == 0 or local == remaining:
            print(f"protected_growth stage={stage} step={stage_step}/{target} route={route} loss={latest['loss']:.5f}", flush=True)
        if stage_step % checkpoint_every == 0 or local == remaining:
            receipt = persist(stage_step, global_step)
    if not remaining:
        receipt = persist(before_stage, total_before)
    if target == spec["target_updates"]:
        endpoint = state_dir / f"infant_training.at_V114_{stage.upper()}.pt"
        if endpoint.exists() and _sha256_file(endpoint) != receipt["checkpoint_sha256"]:
            raise RuntimeError("Existing stage endpoint differs; refusing to overwrite it")
        copy_verified(checkpoint, endpoint)
        if durable_state_dir:
            copy_verified(endpoint, Path(durable_state_dir) / endpoint.name)
    return receipt


def protected_status(state_dir, *, device="cpu"):
    checkpoint = Path(state_dir) / "infant_training.pt"
    model, saved = load_checkpoint(checkpoint, resolve_device(device))
    md = dict(saved.get("metadata") or {})
    measured = legacy_digest(model)
    return dict(format="continuous_brain.protected_growth_status.v114", build_id=md.get("protected_growth_build_id"),
                checkpoint=str(checkpoint), checkpoint_sha256=_sha256_file(checkpoint),
                protected_growth_enabled=bool(model.cfg.protected_growth_enabled),
                protected_growth_version=md.get("protected_growth_version"),
                training_version=md.get("protected_growth_training_version"),
                source_checkpoint_sha256=md.get("protected_growth_source_sha256"),
                legacy_sha256=measured, legacy_integrity_passed=measured == md.get("protected_growth_source_legacy_sha256"),
                stage_updates=md.get("protected_growth_stage_updates") or {},
                total_updates=int(md.get("protected_growth_total_updates", 0)),
                current_stage=md.get("protected_growth_current_stage"), gates=gate_report(model), stage_plan=STAGE_PLAN)
