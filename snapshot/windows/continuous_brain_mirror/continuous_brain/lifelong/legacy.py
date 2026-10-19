"""Local read-only diagnostics for the original v0.11.4 child.

No v0.12 resource-world checkpoint is presented as the historical child. Gate
crossover copies change exactly two buffers and never resume legacy training.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from ..device_v111 import resolve_device, device_report
from ..evidence_v114 import sha256, write_json, P4B_PROTOCOL, valid_p4b
from . import BUILD

KNOWN = {"micro": "324bb845941926c4a6c2a30b9fd1c185784542445a782327837ffe6b614d8e98",
         "low": "fa0142e29cbd80648b00629c4effcaaf122161dad21440e371c22caa106b9825"}
GATE_KEYS = ("planning.growth.gate", "options.growth.gate")


@torch.no_grad()
def paired_action_trace(checkpoint, seed, device):
    """Trace both controllers from identical seeds and identify their first split.

    This localizes a decision for later intervention; it does not prove the first
    different action caused the eventual outcome.
    """
    from ..environment import DetourWorld
    from ..parented_train import load_checkpoint
    from ..planning_v100 import _env_context, _env_image_tensor
    from ..planning_v102 import _resolve_option
    from ..planning_v103 import _hierarchical_action
    from ..infant import one_hot_action
    if not 12001 <= seed <= 12100:
        raise ValueError("Trace only spent seeds 12001--12100; reserved blind seeds are excluded")
    digest = sha256(checkpoint)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    if (state.get("metadata") or {}).get("option_cortex_version") != "latent-temporal-options-v1":
        raise ValueError("Trace requires a trained temporal-option checkpoint")
    traces = {}
    for mode in ("direct_options", "hierarchical"):
        env = DetourWorld(seed=seed, max_episode_steps=120)
        obs = env.reset()
        current, age = None, 0
        rows = []
        for step in range(1, 121):
            image = _env_image_tensor(env, model.cfg.image_size, device)
            _, context = _env_context(obs, device)
            spatial = model.target_spatial_bus(image)
            root = model.planning.encode_state(spatial, context)
            if mode == "direct_options":
                current, age, _ = _resolve_option(model, root, current, age, 3, .5)
                option = torch.tensor([current], device=device)
                action = int(model.options.action_logits(root, option).argmax(-1)[0].cpu())
                decision = {"option": current, "action": action}
            else:
                decision = _hierarchical_action(model, spatial, context, current, age,
                    option_depth=2, selector_topk=2, beam_width=8, max_option_steps=12,
                    discount=.95, min_commit=3, terminate_threshold=.5)
                current, age, action = int(decision["option"]), int(decision["option_age"]), int(decision["action"])
            aoh = one_hot_action(torch.tensor([action], device=device), model.cfg.action_dim)
            forecast = model.planning.step(root, aoh)
            before = obs.tolist()
            obs, reward, done, info = env.step(action)
            rows.append({"step": step, "observation_before": before, "action": action, "option": current,
                         "predicted_reward": float(forecast["reward"][0].cpu()),
                         "predicted_collision": float(torch.sigmoid(forecast["collision_logit"])[0].cpu()),
                         "actual_reward": reward, "collision": bool(info.get("collision", False)),
                         "completed": bool(info.get("reached", False)), "decision": decision})
            age += 1
            if done:
                break
        traces[mode] = rows
    first = None
    for a, b in zip(traces["direct_options"], traces["hierarchical"]):
        if a["action"] != b["action"]:
            first = {"step": a["step"], "same_observation": a["observation_before"] == b["observation_before"],
                     "direct": a, "hierarchical": b}
            break
    if sha256(checkpoint) != digest:
        raise RuntimeError("Trace changed the source checkpoint")
    return {"checkpoint_sha256": digest, "seed": seed, "first_action_divergence": first,
            "trace": traces, "training_performed": False, "source_changed": False,
            "interpretation": "First different decision is a diagnostic lead, not a causal verdict"}


def gate_copy(source, destination, gate):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Gate crossover requires a new, separate destination file")
    if not 0 <= float(gate) <= .10:
        raise ValueError("This diagnostic is bounded to 0--10% gates")
    source_sha = sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not payload.get("config", {}).get("protected_growth_enabled"):
        raise ValueError("Gate crossover requires a protected v0.11.4 checkpoint")
    changed = copy.deepcopy(payload)
    for key in GATE_KEYS:
        if key not in payload["model"]:
            raise ValueError("Missing protected gate: " + key)
        changed["model"][key] = torch.full_like(payload["model"][key], float(gate))
    other_changes = [k for k in payload["model"] if k not in GATE_KEYS and
                     not torch.equal(payload["model"][k], changed["model"][k])]
    if other_changes:
        raise RuntimeError("Crossover altered learned tensors")
    changed.setdefault("metadata", {})["local_crossover_v120"] = {
        "source_sha256": source_sha, "gate": float(gate), "diagnostic_only": True,
        "not_a_training_endpoint": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f:
        torch.save(changed, f)
    if sha256(source) != source_sha:
        raise RuntimeError("Source checkpoint changed during the diagnostic")
    return {"source_sha256": source_sha, "copy_sha256": sha256(destination),
            "gate": float(gate), "changed_buffers": list(GATE_KEYS), "other_tensors_changed": 0}


def _source_fingerprint():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _evaluate(checkpoint, output, device):
    from ..planning_v103 import evaluate_hierarchical_options
    before = sha256(checkpoint)
    binding = {"checkpoint": before, "runtime": device_report(device),
               "source": _source_fingerprint(), "protocol": P4B_PROTOCOL, "build": BUILD}
    if output.is_file():
        old = json.loads(output.read_text())
        if old.get("local_binding") == binding and valid_p4b(old, before, require_pass=False):
            print("Reusing matching evaluation: " + output.name, flush=True)
            return old
    report = evaluate_hierarchical_options(checkpoint, episodes=100, steps=120,
                                           seed_start=12001, option_depth=2, selector_topk=2,
                                           beam_width=8, max_option_steps=12, discount=.95,
                                           min_commit=3, terminate_threshold=.5, device=device)
    if sha256(checkpoint) != before:
        raise RuntimeError("Evaluation changed the source checkpoint")
    report["local_binding"] = binding
    write_json(output, report)
    return report


def run_legacy_diagnostics(args):
    device = resolve_device(args.device)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if getattr(args, "trace_seed", None) is not None:
        if not args.checkpoint or args.micro or args.low:
            raise ValueError("A single-seed trace requires --checkpoint only")
        report = paired_action_trace(Path(args.checkpoint).expanduser(), args.trace_seed, device)
        path = out / f"trace_{args.trace_seed}_{report['checkpoint_sha256'][:12]}.json"
        write_json(path, report)
        return {"trace": str(path), "first_action_divergence": report["first_action_divergence"],
                "source_changed": False}
    if not args.checkpoint and not args.micro and not args.low:
        print("This runs the historical three-controller comparison on 100 spent seeds.")
        args.checkpoint = input("Path to your infant_training.pt (or frozen endpoint): ").strip().strip('"')
    cases = []
    if args.checkpoint:
        if args.micro or args.low:
            raise ValueError("Use either --checkpoint OR the --micro/--low pair")
        cases = [("baseline", Path(args.checkpoint).expanduser().resolve())]
    else:
        if not args.micro or not args.low:
            raise ValueError("Crossover requires both --micro and --low")
        for label in ("micro", "low"):
            source = Path(getattr(args, label)).expanduser().resolve()
            if sha256(source) != KNOWN[label]:
                raise ValueError(f"{label} is not the recorded frozen endpoint; use --checkpoint for a different child's evaluation")
            for gate in (.02, .05):
                case = f"{label}_gate_{round(gate*100):02d}"
                destination = out / "copies" / f"{case}.pt"
                receipt = destination.with_suffix(".json")
                if destination.exists():
                    saved = json.loads(receipt.read_text()) if receipt.is_file() else {}
                    if (saved.get("copy_sha256") != sha256(destination) or
                            saved.get("source_sha256") != KNOWN[label] or saved.get("gate") != gate):
                        raise RuntimeError("Existing crossover copy has no matching receipt; choose a new output directory")
                else:
                    write_json(receipt, gate_copy(source, destination, gate))
                cases.append((case, destination))
    summary = {"build": BUILD, "diagnostic_only": True, "blind_seeds_used": False,
               "new_training_performed": False, "cases": {}}
    for label, checkpoint in cases:
        case_dir = out / sha256(checkpoint)[:16]
        case_dir.mkdir(parents=True, exist_ok=True)
        print("Evaluating " + label + " on " + str(device), flush=True)
        report = _evaluate(checkpoint, case_dir / "P4B_12001_12100.json", device)
        summary["cases"][label] = {"checkpoint_sha256": report["checkpoint_sha256"],
             "hierarchical": report["hierarchical_option_planning"]["completion_rate"],
             "direct_options": report["option_direct_diagnostic"]["completion_rate"],
             "reactive": report["reactive"]["completion_rate"], "p4b_floor_passed": report["p4b_passed"]}
    write_json(out / "LOCAL_LEGACY_SUMMARY.json", summary)
    return summary
