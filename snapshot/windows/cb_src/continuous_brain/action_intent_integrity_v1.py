"""Shared checkpoint and preservation checks for the v0.17.1 repair.

The executive tensor schema and v0.17.0 optimizer schedule are unchanged.
CPU is an explicit testing option; DirectML remains the default on David's PC.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
EXPECTED_BASE_SHA256 = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
EXPECTED_BASE_DIGEST = "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
EXPECTED_ADMITTED_SHA256 = "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
FORMAT = "action_intent_executive_training_v1"
INIT_SEED = 717172
SCHEDULE_SEED = 1173221
MAX_UPDATE = 4800


def select_device(name):
    if name == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("Activate the existing continuous_brain_v110_dml venv. No CPU fallback was used.") from exc
        return torch_directml.device()
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be directml or cpu")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clone(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def assert_exact(reference, model, label):
    current = model.state_dict()
    if set(reference) != set(current):
        raise AssertionError(f"{label}: state keys changed")
    for key, value in reference.items():
        other = current[key].detach().cpu()
        if value.dtype != other.dtype or not torch.equal(value, other):
            raise AssertionError(f"{label}: changed {key}")


def validate_payload(payload, executive=None, *, minimum_update=0, require_optimizer=True):
    expected = {
        "format": FORMAT,
        "source_admitted_checkpoint_sha256": EXPECTED_ADMITTED_SHA256,
        "historical_base_sha256": EXPECTED_BASE_SHA256,
        "historical_base_digest": EXPECTED_BASE_DIGEST,
        "schedule_seed": SCHEDULE_SEED,
        "init_seed": INIT_SEED,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    update = payload.get("global_update")
    if type(update) is not int or not minimum_update <= update <= MAX_UPDATE:
        raise ValueError("checkpoint update outside the fixed 0..4800 budget")
    if payload.get("update") != update:
        raise ValueError("checkpoint update fields disagree")
    state = payload.get("executive")
    if not isinstance(state, dict) or not state:
        raise ValueError("missing executive tensors")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
            raise ValueError(f"invalid executive tensor: {key}")
    if executive is not None:
        reference = executive.state_dict()
        if set(reference) != set(state):
            raise ValueError("executive tensor keys differ")
        for key, value in reference.items():
            if value.shape != state[key].shape or value.dtype != state[key].dtype:
                raise ValueError(f"executive tensor schema differs: {key}")
    if require_optimizer:
        opt = payload.get("optimizer", {})
        if opt.get("step_count") != update:
            raise ValueError("optimizer step count does not match checkpoint")
        if not 0 < float(opt.get("lr", 0)) < 1:
            raise ValueError("invalid optimizer learning rate")
        for key in ("beta1", "beta2"):
            value = float(opt.get(key, -1))
            if not 0 <= value < 1:
                raise ValueError(f"invalid optimizer {key}")
        if not math.isfinite(float(opt.get("eps", 0))) or float(opt.get("eps", 0)) <= 0:
            raise ValueError("invalid optimizer epsilon")
        moments = opt.get("state", [])
        parameters = list(state.values())
        if len(moments) != len(parameters):
            raise ValueError("optimizer parameter count differs")
        for moment, parameter in zip(moments, parameters):
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"invalid optimizer tensor: {key}")
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"nonfinite optimizer tensor: {key}")
            if torch.any(moment["exp_avg_sq"] < 0).item():
                raise ValueError("negative second moment")
    return update


def load_protected(device):
    # Check exact file identity BEFORE deserializing the legacy base payload.
    for path, expected in ((BASE_CHECKPOINT, EXPECTED_BASE_SHA256),
                           (ADMITTED_CHECKPOINT, EXPECTED_ADMITTED_SHA256)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required existing project checkpoint is missing: {path}")
        if sha(path) != expected:
            raise ValueError(f"Protected checkpoint SHA mismatch: {path}")
    from .evaluate_competitive_infant import _load_base_checkpoint
    from .working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
    from .action_intent_executive_v1 import ActionIntentExecutiveInfant

    base, cfg, _, base_sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if base_sha != EXPECTED_BASE_SHA256 or digest(clone(base)) != EXPECTED_BASE_DIGEST:
        raise ValueError("historical base reconstruction differs")
    payload = torch.load(ADMITTED_CHECKPOINT, map_location="cpu", weights_only=True)
    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    admitted.memory.load_state_dict(payload["memory"], strict=True)
    admitted.recall.load_state_dict(payload["recall"], strict=True)
    admitted = admitted.to(device)
    model = ActionIntentExecutiveInfant.grow_from_admitted(admitted).to(device)
    model.train()
    return model, cfg, clone(model.admitted), clone(model.base), model.trainable_parameters()


def load_executive(path, device, *, minimum_update=0):
    checkpoint_sha = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_payload(payload, minimum_update=minimum_update)
    model, cfg, _, _, _ = load_protected(device)
    validate_payload(payload, model.executive, minimum_update=minimum_update)
    model.executive.load_state_dict(payload["executive"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sha(path) != checkpoint_sha:
        raise RuntimeError("checkpoint changed while loading; stop the training process first")
    return model, cfg, payload


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    names = (
        "continuous_brain/action_intent_executive_v1.py",
        "continuous_brain/action_intent_integrity_v1.py",
        "continuous_brain/action_intent_gate_v1.py",
        "continuous_brain/action_intent_rollout_v1.py",
        "continuous_brain/canonical_executive_rule_world_v3.py",
        "evaluate_action_intent_executive_final_train_v1.py",
        "evaluate_action_intent_control_v1.py",
    )
    return {name: sha(root / name) for name in names}


def provenance(path, payload, device):
    return {
        "checkpoint_path": str(Path(path).resolve()),
        "checkpoint_sha256": sha(path),
        "global_update": payload["global_update"],
        "admitted_checkpoint_sha256": sha(ADMITTED_CHECKPOINT),
        "historical_base_sha256": sha(BASE_CHECKPOINT),
        "source_sha256": source_hashes(),
        "torch_version": str(torch.__version__),
        "device": str(device),
    }


def assert_files_unchanged(before, checkpoint):
    if before["checkpoint_sha256"] != sha(checkpoint):
        raise RuntimeError("executive checkpoint changed during evaluation")
    if before["admitted_checkpoint_sha256"] != sha(ADMITTED_CHECKPOINT):
        raise RuntimeError("admitted checkpoint changed during evaluation")
    if before["historical_base_sha256"] != sha(BASE_CHECKPOINT):
        raise RuntimeError("base checkpoint changed during evaluation")
    if before["source_sha256"] != source_hashes():
        raise RuntimeError("source files changed during evaluation")
