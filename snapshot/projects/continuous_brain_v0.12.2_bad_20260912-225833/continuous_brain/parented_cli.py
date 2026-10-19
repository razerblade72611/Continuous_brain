"""CLI for Continuous Brain v0.9 parented development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .device_v111 import resolve_device, device_report

from .infant import InfantBrain, InfantConfig, byte_encode
from .parented_data import NurseryDataset, reparent_vision, teach_language, merge_nurseries
from .parented_train import train as train_infant, evaluate as evaluate_infant, load_checkpoint
from .parenting import ParentRegistry, ParentSpec, parent_summary
from .nursery_collect import collect_foraging, collect_detour
from .developmental_runtime import run_foraging
from .planning_v100 import train_planning_cortex, evaluate_planning
from .planning_v101 import train_conservative_bellman_q
from .planning_v102 import train_temporal_options, evaluate_temporal_options
from .planning_v103 import evaluate_hierarchical_options
from .world_consolidation_v104 import (
    collect_world_holdout, seal_world_holdouts, train_world_consolidation,
    evaluate_world_consolidation,
)
from .whole_brain_v110 import expand_capacity, train_whole_brain
from .curriculum_v110 import collect_generalized_curriculum, evaluate_generalized_planning
from .protected_growth_v114 import protect_growth, train_protected_growth, protected_status, STAGE_PLAN


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def init_state(args):
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    registry = ParentRegistry(state / "parents.json")
    identity = state / "identity.json"
    if not identity.exists():
        identity.write_text(json.dumps({
            "format": "continuous_brain.identity.v090",
            "name": args.name,
            "lineage": "fresh-parented-development",
            "architecture": "raw-cortices-with-external-parents",
            "claim": "persistent artificial individual; consciousness not asserted",
        }, indent=2, sort_keys=True))
    _print({"state_dir": str(state), "identity": json.loads(identity.read_text()), "parents": parent_summary(registry)})


def import_semantic(args):
    dataset = NurseryDataset(args.destination)
    if dataset.records() and not args.force:
        raise FileExistsError("Destination nursery already contains lessons; choose a new path or use --force")
    _print(dataset.import_semantic(args.source, parent_id=args.parent_id))


def collect_cmd(args):
    _print(collect_foraging(args.dataset, samples=args.samples, seed=args.seed,
                            dino_model=args.dino_model, device=args.vision_device,
                            exploration=args.exploration))


def collect_detour_cmd(args):
    _print(collect_detour(args.dataset, samples=args.samples, seed=args.seed,
                          dino_model=args.dino_model, device=args.vision_device,
                          exploration=args.exploration))


def merge_cmd(args):
    _print(merge_nurseries(args.sources, args.destination))

def reparent_cmd(args):
    _print(reparent_vision(args.dataset, model_id=args.model, parent_id=args.parent_id, device=args.vision_device))


def teach_language_cmd(args):
    _print(teach_language(args.dataset, endpoint=args.endpoint, model=args.model, parent_id=args.parent_id,
                          limit=args.limit, timeout=args.timeout))


def train_cmd(args):
    _print(train_infant(args.dataset, args.state_dir, profile=args.profile, updates=args.updates,
                        batch_size=args.batch, device=resolve_device(args.device), learning_rate=args.learning_rate,
                        seed=args.seed, resume=not args.no_resume, world_fraction=args.world_fraction,
                        world_focus=args.world_focus, world_only=args.world_only))


def eval_cmd(args):
    result = evaluate_infant(args.dataset, args.state_dir, device=resolve_device(args.device), seed=args.seed)
    if args.record_weaning:
        registry = ParentRegistry(Path(args.state_dir) / "parents.json")
        for capability, score in result["capability_scores"].items():
            if capability in registry.weaning and result.get("evidence_eligible", {}).get(capability, False):
                registry.record_score(capability, score, threshold=result["thresholds"].get(capability, 0.85))
        result["weaning"] = parent_summary(registry)
    _print(result)


def status(args):
    state = Path(args.state_dir)
    registry = ParentRegistry(state / "parents.json")
    ckpt = state / "infant_training.pt"
    info = {"state_dir": str(state), "parents": parent_summary(registry)}
    if ckpt.exists():
        model, saved = load_checkpoint(ckpt, "cpu")
        info["training"] = {
            "updates": saved.get("updates", 0),
            "trainable_child_parameters": model.parameter_count,
            "growth_enabled": bool(model.cfg.growth_enabled),
            "growth_profile": str(model.cfg.growth_profile),
            "protected_growth_enabled": bool(getattr(model.cfg, "protected_growth_enabled", False)),
            "protected_growth_version": (saved.get("metadata") or {}).get("protected_growth_version"),
            "protected_growth_stage_updates": (saved.get("metadata") or {}).get("protected_growth_stage_updates", {}),
        }
    if (state / "infant_evaluation.json").exists():
        info["evaluation"] = json.loads((state / "infant_evaluation.json").read_text())
    _print(info)


def upgrade_parent(args):
    registry = ParentRegistry(Path(args.state_dir) / "parents.json")
    spec = ParentSpec(args.parent_id, args.capability, args.model, args.interface,
                      endpoint=args.endpoint, notes=args.notes)
    registry.upgrade(spec)
    _print({"upgraded": args.capability, "active": registry.active_parent(args.capability).__dict__,
            "parents": parent_summary(registry)})


def live_foraging(args):
    _print(run_foraging(args.state_dir, steps=args.steps, device=resolve_device(args.device), seed=args.seed,
                         allow_parent=not args.child_only))


def train_planning_cmd(args):
    _print(train_planning_cortex(
        args.dataset, args.state_dir, updates=args.updates, batch_size=args.batch,
        unroll=args.unroll, gamma=args.gamma, goal_gamma=args.goal_gamma,
        learning_rate=args.learning_rate, seed=args.seed, device=resolve_device(args.device),
    ))


def evaluate_planning_cmd(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_planning(
        checkpoint, episodes=args.episodes, steps=args.steps, seed_start=args.seed_start,
        horizons=args.horizons, beam_width=args.beam_width, discount=args.discount,
        device=resolve_device(args.device), output=args.output,
    ))


def train_planning_q_cmd(args):
    _print(train_conservative_bellman_q(
        args.dataset, args.state_dir, updates=args.updates, batch_size=args.batch,
        gamma=args.gamma, learning_rate=args.learning_rate, cql_alpha=args.cql_alpha,
        target_tau=args.target_tau, seed=args.seed, device=resolve_device(args.device),
    ))



def train_options_cmd(args):
    _print(train_temporal_options(
        args.dataset, args.state_dir, updates=args.updates, batch_size=args.batch,
        lookahead=args.lookahead, gamma=args.gamma, goal_gamma=args.goal_gamma,
        learning_rate=args.learning_rate, q_margin=args.q_margin,
        seed=args.seed, device=resolve_device(args.device),
    ))


def evaluate_options_cmd(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_temporal_options(
        checkpoint, episodes=args.episodes, steps=args.steps, seed_start=args.seed_start,
        horizons=args.horizons, beam_width=args.beam_width, discount=args.discount,
        min_commit=args.min_commit, terminate_threshold=args.terminate_threshold,
        device=resolve_device(args.device), output=args.output,
    ))

def evaluate_hierarchical_options_cmd(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_hierarchical_options(
        checkpoint, episodes=args.episodes, steps=args.steps, seed_start=args.seed_start,
        option_depth=args.option_depth, selector_topk=args.selector_topk,
        beam_width=args.beam_width, max_option_steps=args.max_option_steps,
        discount=args.discount, min_commit=args.min_commit,
        terminate_threshold=args.terminate_threshold, device=resolve_device(args.device), output=args.output,
    ))

def evaluate_generalization_gate_cmd(args):
    from .generalization_gate_v110 import evaluate_generalization_gate
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_generalization_gate(
        checkpoint, replication_report=args.replication_report, kind=args.kind,
        episodes=args.episodes, steps=args.steps, seed_start=args.seed_start,
        evaluation_role=args.evaluation_role, option_depth=args.option_depth,
        selector_topk=args.selector_topk, beam_width=args.beam_width,
        max_option_steps=args.max_option_steps, discount=args.discount,
        min_commit=args.min_commit, terminate_threshold=args.terminate_threshold,
        device=args.device, output=args.output))


def collect_world_holdout_cmd(args):
    _print(collect_world_holdout(
        args.dataset, domain=args.domain, samples=args.samples, seed=args.seed,
        exploration=args.exploration,
    ))


def seal_world_holdouts_cmd(args):
    _print(seal_world_holdouts(
        foraging_dataset=args.foraging_dataset, detour_dataset=args.detour_dataset,
        training_datasets=args.training_datasets, output=args.output,
    ))


def train_world_consolidation_cmd(args):
    _print(train_world_consolidation(
        args.datasets, args.state_dir, holdout_seal=args.holdout_seal,
        updates=args.updates, batch_per_domain=args.batch_per_domain,
        learning_rate=args.learning_rate, seed=args.seed, device=resolve_device(args.device),
    ))


def evaluate_world_consolidation_cmd(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_world_consolidation(
        checkpoint, foraging_holdout=args.foraging_holdout,
        detour_holdout=args.detour_holdout, holdout_seal=args.holdout_seal,
        device=resolve_device(args.device), output=args.output,
    ))


def expand_capacity_cmd(args):
    _print(expand_capacity(
        args.source_state_dir, args.state_dir, growth_profile=args.growth_profile, device=resolve_device(args.device),
    ))


def train_whole_brain_cmd(args):
    _print(train_whole_brain(
        args.rehearsal_datasets, args.planning_dataset, args.state_dir,
        world_datasets=args.world_datasets, updates=args.updates, batch_size=args.batch, learning_rate=args.learning_rate,
        anchor_weight=args.anchor_weight, checkpoint_every=args.checkpoint_every,
        seed=args.seed, device=resolve_device(args.device),
    ))


def protect_growth_cmd(args):
    _print(protect_growth(
        args.source_state_dir, args.state_dir, device=resolve_device(args.device),
    ))


def train_protected_growth_cmd(args):
    _print(train_protected_growth(
        args.rehearsal_datasets, args.planning_dataset, args.state_dir,
        stage=args.stage, world_datasets=args.world_datasets,
        planning_rehearsal_dataset=args.planning_rehearsal_dataset,
        previous_p4b_result=args.previous_p4b_result,
        durable_state_dir=args.durable_state_dir,
        target_stage_updates=args.target_stage_updates, batch_size=args.batch,
        learning_rate=args.learning_rate, anchor_weight=args.anchor_weight,
        interface_anchor_weight=args.interface_anchor_weight,
        checkpoint_every=args.checkpoint_every, seed=args.seed,
        device=resolve_device(args.device),
    ))


def protected_growth_status_cmd(args):
    _print(protected_status(args.state_dir, device=resolve_device(args.device)))


def collect_generalized_cmd(args):
    _print(collect_generalized_curriculum(
        args.dataset, kind=args.kind, samples=args.samples, seed=args.seed,
        exploration=args.exploration, max_steps=args.max_steps,
    ))


def evaluate_generalized_cmd(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.state_dir) / "infant_training.pt"
    _print(evaluate_generalized_planning(
        checkpoint, kind=args.kind, episodes=args.episodes, steps=args.steps, seed_start=args.seed_start,
        option_depth=args.option_depth, selector_topk=args.selector_topk, beam_width=args.beam_width,
        max_option_steps=args.max_option_steps, discount=args.discount, min_commit=args.min_commit,
        terminate_threshold=args.terminate_threshold, device=resolve_device(args.device), output=args.output,
    ))


def chat(args):
    device = resolve_device(args.device)
    state = Path(args.state_dir)
    checkpoint = state / ("qualified_infant.pt" if (state / "qualified_infant.pt").exists() else "infant_training.pt")
    model, _ = load_checkpoint(checkpoint, device)
    model.eval()
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("chat requires Pillow") from exc
    if args.image:
        image = Image.open(args.image).convert("RGB").resize((model.cfg.image_size, model.cfg.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images = torch.tensor(np.transpose(arr, (2,0,1))[None], dtype=torch.float32, device=device)
    else:
        images = torch.zeros(1, 3, model.cfg.image_size, model.cfg.image_size, device=device)
    context = torch.zeros(1, model.cfg.context_dim, device=device)
    memory = torch.zeros(1, model.cfg.memory_dim, device=device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        vision = model.vision(images)
        spatial = model.spatial(images)
        fused = model.fuse_vision(vision, spatial)
        nuc = model.nucleus(fused, context, memory, prev_action, model.nucleus.initial_state(1, device))
    print("Continuous Brain v0.11.2 child-only chat. Parent models are not queried. Type /quit to exit.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q == "/quit": break
        qids = torch.tensor([byte_encode(q, model.cfg.max_question_bytes)], dtype=torch.long, device=device)
        print("brain>", model.language.generate(nuc["workspace"], qids))



def device_info_cmd(args):
    _print(device_report(args.device, compute_test=args.compute_test, matrix_size=args.matrix_size))


def main(argv=None):
    import os
    if os.environ.get("CB_V114_STRICT_NUMERICS") == "1":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    p = argparse.ArgumentParser(prog="continuous-parented")
    sub = p.add_subparsers(required=True)

    x = sub.add_parser("device-info")
    x.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, dml, dml:0, directml, privateuseone:0")
    x.add_argument("--compute-test", action="store_true")
    x.add_argument("--matrix-size", type=int, default=256)
    x.set_defaults(func=device_info_cmd)

    x = sub.add_parser("init")
    x.add_argument("--state-dir", required=True); x.add_argument("--name", default="Continuous Brain")
    x.set_defaults(func=init_state)

    x = sub.add_parser("import-semantic")
    x.add_argument("--source", required=True); x.add_argument("--destination", required=True)
    x.add_argument("--parent-id", default="vision.dinov3.legacy")
    x.add_argument("--force", action="store_true"); x.set_defaults(func=import_semantic)

    x = sub.add_parser("collect-foraging")
    x.add_argument("--dataset", required=True); x.add_argument("--samples", type=int, default=200)
    x.add_argument("--seed", type=int, default=17)
    x.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    x.add_argument("--vision-device", default="cpu")
    x.add_argument("--exploration", type=float, default=0.25,
                   help="Fraction of lived actions deliberately sampled away from the motor parent")
    x.set_defaults(func=collect_cmd)



    x = sub.add_parser("collect-detour")
    x.add_argument("--dataset", required=True); x.add_argument("--samples", type=int, default=3000)
    x.add_argument("--seed", type=int, default=1701)
    x.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    x.add_argument("--vision-device", default="cpu")
    x.add_argument("--exploration", type=float, default=0.45,
                   help="Physics-collection exploration; motor supervision remains disabled")
    x.set_defaults(func=collect_detour_cmd)

    x = sub.add_parser("merge-nurseries")
    x.add_argument("--sources", nargs="+", required=True)
    x.add_argument("--destination", required=True)
    x.set_defaults(func=merge_cmd)

    x = sub.add_parser("reparent-vision")
    x.add_argument("--dataset", required=True); x.add_argument("--model", required=True)
    x.add_argument("--parent-id", required=True); x.add_argument("--vision-device", default="cpu")
    x.set_defaults(func=reparent_cmd)

    x = sub.add_parser("teach-language")
    x.add_argument("--dataset", required=True); x.add_argument("--endpoint", required=True)
    x.add_argument("--model", required=True); x.add_argument("--parent-id", required=True)
    x.add_argument("--limit", type=int); x.add_argument("--timeout", type=int, default=300)
    x.set_defaults(func=teach_language_cmd)

    x = sub.add_parser("train")
    x.add_argument("--dataset", required=True); x.add_argument("--state-dir", required=True)
    x.add_argument("--profile", choices=["proof","balanced","scale"], default="proof")
    x.add_argument("--updates", type=int, default=2000); x.add_argument("--batch", type=int, default=8)
    x.add_argument("--device", default="cpu"); x.add_argument("--learning-rate", type=float, default=3e-4)
    x.add_argument("--world-fraction", type=float, default=0.50,
                   help="Fraction of each mixed minibatch reserved for lived/world-valid transitions")
    x.add_argument("--world-focus", action="store_true",
                   help="Downweight mature rehearsal losses so causal/spatial dynamics dominate this phase")
    x.add_argument("--world-only", action="store_true",
                   help="Freeze the reactive child and train only its world-model ensemble (P4 planning lab)")
    x.add_argument("--seed", type=int, default=17); x.add_argument("--no-resume", action="store_true")
    x.set_defaults(func=train_cmd)

    x = sub.add_parser("evaluate")
    x.add_argument("--dataset", required=True); x.add_argument("--state-dir", required=True)
    x.add_argument("--device", default="cpu"); x.add_argument("--seed", type=int, default=17)
    x.add_argument("--record-weaning", action="store_true"); x.set_defaults(func=eval_cmd)

    x = sub.add_parser("status")
    x.add_argument("--state-dir", required=True); x.set_defaults(func=status)

    x = sub.add_parser("upgrade-parent")
    x.add_argument("--state-dir", required=True); x.add_argument("--capability", required=True)
    x.add_argument("--parent-id", required=True); x.add_argument("--model", required=True)
    x.add_argument("--interface", required=True); x.add_argument("--endpoint")
    x.add_argument("--notes", default=""); x.set_defaults(func=upgrade_parent)


    x = sub.add_parser("train-planning")
    x.add_argument("--dataset", required=True); x.add_argument("--state-dir", required=True)
    x.add_argument("--updates", type=int, default=4000); x.add_argument("--batch", type=int, default=32)
    x.add_argument("--unroll", type=int, default=12); x.add_argument("--gamma", type=float, default=0.95)
    x.add_argument("--goal-gamma", type=float, default=0.97); x.add_argument("--learning-rate", type=float, default=3e-4)
    x.add_argument("--seed", type=int, default=10010); x.add_argument("--device", default="cpu")
    x.set_defaults(func=train_planning_cmd)

    x = sub.add_parser("train-planning-q")
    x.add_argument("--dataset", required=True); x.add_argument("--state-dir", required=True)
    x.add_argument("--updates", type=int, default=3000); x.add_argument("--batch", type=int, default=256)
    x.add_argument("--gamma", type=float, default=0.95); x.add_argument("--learning-rate", type=float, default=3e-4)
    x.add_argument("--cql-alpha", type=float, default=0.10)
    x.add_argument("--target-tau", type=float, default=0.01)
    x.add_argument("--seed", type=int, default=10110); x.add_argument("--device", default="cpu")
    x.set_defaults(func=train_planning_q_cmd)

    x = sub.add_parser("train-options")
    x.add_argument("--dataset", required=True); x.add_argument("--state-dir", required=True)
    x.add_argument("--updates", type=int, default=3000); x.add_argument("--batch", type=int, default=256)
    x.add_argument("--lookahead", type=int, default=6); x.add_argument("--gamma", type=float, default=0.95)
    x.add_argument("--goal-gamma", type=float, default=0.97); x.add_argument("--learning-rate", type=float, default=3e-4)
    x.add_argument("--q-margin", type=float, default=0.08)
    x.add_argument("--seed", type=int, default=10210); x.add_argument("--device", default="cpu")
    x.set_defaults(func=train_options_cmd)

    x = sub.add_parser("evaluate-options")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--episodes", type=int, default=10); x.add_argument("--steps", type=int, default=120)
    x.add_argument("--seed-start", type=int, default=9001); x.add_argument("--horizons", nargs="+", type=int, default=[1,3,6,12])
    x.add_argument("--beam-width", type=int, default=32); x.add_argument("--discount", type=float, default=0.95)
    x.add_argument("--min-commit", type=int, default=3); x.add_argument("--terminate-threshold", type=float, default=0.50)
    x.add_argument("--device", default="cpu"); x.add_argument("--output")
    x.set_defaults(func=evaluate_options_cmd)

    x = sub.add_parser("evaluate-hierarchical-options")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--episodes", type=int, default=10); x.add_argument("--steps", type=int, default=120)
    x.add_argument("--seed-start", type=int, default=9001)
    x.add_argument("--option-depth", type=int, default=2, help="Number of learned option macro-actions imagined per high-level branch")
    x.add_argument("--selector-topk", type=int, default=2, help="Selector-supported option candidates expanded at each high-level boundary")
    x.add_argument("--beam-width", type=int, default=8); x.add_argument("--max-option-steps", type=int, default=12)
    x.add_argument("--discount", type=float, default=0.95)
    x.add_argument("--min-commit", type=int, default=3); x.add_argument("--terminate-threshold", type=float, default=0.50)
    x.add_argument("--device", default="cpu"); x.add_argument("--output")
    x.set_defaults(func=evaluate_hierarchical_options_cmd)

    x = sub.add_parser("evaluate-generalization-gate")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--replication-report", required=True)
    x.add_argument("--kind", choices=["randomized_detour", "two_barrier"], default="randomized_detour")
    x.add_argument("--evaluation-role", choices=["diagnostic", "confirmation"], default="diagnostic")
    x.add_argument("--episodes", type=int, default=100); x.add_argument("--steps", type=int, default=160)
    x.add_argument("--seed-start", type=int, default=41001)
    x.add_argument("--option-depth", type=int, default=2); x.add_argument("--selector-topk", type=int, default=2)
    x.add_argument("--beam-width", type=int, default=8); x.add_argument("--max-option-steps", type=int, default=12)
    x.add_argument("--discount", type=float, default=.95); x.add_argument("--min-commit", type=int, default=3)
    x.add_argument("--terminate-threshold", type=float, default=.5)
    x.add_argument("--device", default="cpu"); x.add_argument("--output")
    x.set_defaults(func=evaluate_generalization_gate_cmd)

    x = sub.add_parser("collect-world-holdout")
    x.add_argument("--dataset", required=True)
    x.add_argument("--domain", choices=["foraging", "detour"], required=True)
    x.add_argument("--samples", type=int, default=1200)
    x.add_argument("--seed", type=int, default=24017)
    x.add_argument("--exploration", type=float, default=0.45)
    x.set_defaults(func=collect_world_holdout_cmd)

    x = sub.add_parser("seal-world-holdouts")
    x.add_argument("--foraging-dataset", required=True)
    x.add_argument("--detour-dataset", required=True)
    x.add_argument("--training-datasets", nargs="+", required=True)
    x.add_argument("--output", required=True)
    x.set_defaults(func=seal_world_holdouts_cmd)

    x = sub.add_parser("train-world-consolidation")
    x.add_argument("--datasets", nargs="+", required=True)
    x.add_argument("--state-dir", required=True)
    x.add_argument("--holdout-seal", required=True)
    x.add_argument("--updates", type=int, default=3000)
    x.add_argument("--batch-per-domain", type=int, default=16)
    x.add_argument("--learning-rate", type=float, default=1e-4)
    x.add_argument("--seed", type=int, default=10410)
    x.add_argument("--device", default="cpu")
    x.set_defaults(func=train_world_consolidation_cmd)

    x = sub.add_parser("evaluate-world-consolidation")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--foraging-holdout", required=True)
    x.add_argument("--detour-holdout", required=True)
    x.add_argument("--holdout-seal", required=True)
    x.add_argument("--device", default="cpu")
    x.add_argument("--output")
    x.set_defaults(func=evaluate_world_consolidation_cmd)

    x = sub.add_parser("expand-capacity")
    x.add_argument("--source-state-dir", required=True)
    x.add_argument("--state-dir", required=True)
    x.add_argument("--growth-profile", default="balanced-v110", choices=["balanced-v110"])
    x.add_argument("--device", default="cpu")
    x.set_defaults(func=expand_capacity_cmd)

    x = sub.add_parser("train-whole-brain")
    x.add_argument("--rehearsal-datasets", nargs="+", required=True)
    x.add_argument("--planning-dataset", required=True)
    x.add_argument("--world-datasets", nargs="+", default=None)
    x.add_argument("--state-dir", required=True)
    x.add_argument("--updates", type=int, default=6000)
    x.add_argument("--batch", type=int, default=8)
    x.add_argument("--learning-rate", type=float, default=1e-4)
    x.add_argument("--anchor-weight", type=float, default=0.10)
    x.add_argument("--checkpoint-every", type=int, default=500)
    x.add_argument("--seed", type=int, default=11010)
    x.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, dml, dml:0")
    x.set_defaults(func=train_whole_brain_cmd)

    x = sub.add_parser("protect-growth")
    x.add_argument("--source-state-dir", required=True, help="Exact v0.11.0 expansion branch; must be pre-Phase-A")
    x.add_argument("--state-dir", required=True)
    x.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, dml, dml:0")
    x.set_defaults(func=protect_growth_cmd)

    x = sub.add_parser("train-protected-growth")
    x.add_argument("--planning-rehearsal-dataset", required=True)
    x.add_argument("--previous-p4b-result", help="Matching baseline or previous-stage P4b JSON")
    x.add_argument("--durable-state-dir", help="Mirror each verified periodic checkpoint here")
    x.add_argument("--rehearsal-datasets", nargs="+", required=True)
    x.add_argument("--planning-dataset", required=True)
    x.add_argument("--world-datasets", nargs="+", default=None)
    x.add_argument("--state-dir", required=True)
    x.add_argument("--stage", choices=list(STAGE_PLAN), required=True)
    x.add_argument("--target-stage-updates", type=int, default=None,
                   help="Target total updates for this stage; reruns train only the remainder")
    x.add_argument("--batch", type=int, default=8)
    x.add_argument("--learning-rate", type=float, default=1e-4)
    x.add_argument("--anchor-weight", type=float, default=0.10)
    x.add_argument("--interface-anchor-weight", type=float, default=1.0)
    x.add_argument("--checkpoint-every", type=int, default=250)
    x.add_argument("--seed", type=int, default=11410)
    x.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, dml, dml:0")
    x.set_defaults(func=train_protected_growth_cmd)

    x = sub.add_parser("protected-growth-status")
    x.add_argument("--state-dir", required=True)
    x.add_argument("--device", default="cpu")
    x.set_defaults(func=protected_growth_status_cmd)

    x = sub.add_parser("collect-generalized-curriculum")
    x.add_argument("--dataset", required=True)
    x.add_argument("--kind", choices=["randomized_detour", "two_barrier"], default="randomized_detour")
    x.add_argument("--samples", type=int, default=8000)
    x.add_argument("--seed", type=int, default=31001)
    x.add_argument("--exploration", type=float, default=0.25)
    x.add_argument("--max-steps", type=int, default=160)
    x.set_defaults(func=collect_generalized_cmd)

    x = sub.add_parser("evaluate-generalized-planning")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--kind", choices=["randomized_detour", "two_barrier"], default="randomized_detour")
    x.add_argument("--episodes", type=int, default=100); x.add_argument("--steps", type=int, default=160)
    x.add_argument("--seed-start", type=int, default=41001)
    x.add_argument("--option-depth", type=int, default=2); x.add_argument("--selector-topk", type=int, default=2)
    x.add_argument("--beam-width", type=int, default=8); x.add_argument("--max-option-steps", type=int, default=12)
    x.add_argument("--discount", type=float, default=0.95); x.add_argument("--min-commit", type=int, default=3)
    x.add_argument("--terminate-threshold", type=float, default=0.50)
    x.add_argument("--device", default="cpu"); x.add_argument("--output")
    x.set_defaults(func=evaluate_generalized_cmd)

    x = sub.add_parser("evaluate-planning")
    src = x.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dir"); src.add_argument("--checkpoint")
    x.add_argument("--episodes", type=int, default=10); x.add_argument("--steps", type=int, default=120)
    x.add_argument("--seed-start", type=int, default=9001); x.add_argument("--horizons", nargs="+", type=int, default=[1,3,6,12])
    x.add_argument("--beam-width", type=int, default=32); x.add_argument("--discount", type=float, default=0.95)
    x.add_argument("--device", default="cpu"); x.add_argument("--output")
    x.set_defaults(func=evaluate_planning_cmd)

    x = sub.add_parser("live-foraging")
    x.add_argument("--state-dir", required=True); x.add_argument("--steps", type=int, default=200)
    x.add_argument("--device", default="cpu"); x.add_argument("--seed", type=int, default=17)
    x.add_argument("--child-only", action="store_true"); x.set_defaults(func=live_foraging)

    x = sub.add_parser("chat")
    x.add_argument("--state-dir", required=True); x.add_argument("--image")
    x.add_argument("--device", default="cpu"); x.set_defaults(func=chat)

    args = p.parse_args(argv); args.func(args)

if __name__ == "__main__":
    main()
