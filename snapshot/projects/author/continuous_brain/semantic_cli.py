"""Developmental semantic training CLI for Continuous Brain v0.7."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .development import DevelopmentRoom
from .lifetime import LifetimeStore, SelfModel
from .semantic import (BASIC_CONCEPTS, CARE_KEYS, ByteCodec, SemanticConfig, SemanticCortex,
                       load_semantic_bundle, save_semantic_bundle, semantic_losses)
from .semantic_data import (SemanticDataset, SemanticRecord, collate_examples, load_example,
                            make_context, neutral_self_vector, training_examples)
from .skills import GOALS
from .teacher import QwenTeacherClient
from .vision import build_vision_backbone


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Semantic curriculum requires Pillow: pip install pillow") from exc
    return Image


def _next_id(dataset: SemanticDataset) -> int:
    ids = []
    for row in dataset.records():
        try:
            ids.append(int(str(row["record_id"]).split("-")[-1]))
        except ValueError:
            pass
    return max(ids, default=-1) + 1


def _save_record(dataset, rgb, context, source, allowed_text, evaluator=None):
    Image = _pil()
    index = _next_id(dataset)
    record_id = f"sample-{index:07d}"
    relative = f"images/{record_id}.png"
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(dataset.directory / relative)
    row = SemanticRecord(record_id, relative, np.asarray(context, np.float32).tolist(), source,
                         evaluator=evaluator).__dict__
    row["allowed_text"] = allowed_text
    dataset.append(row)
    return record_id


def collect_development(args):
    dataset = SemanticDataset(args.dataset)
    env = DevelopmentRoom(args.seed, args.max_episode_steps)
    rng = random.Random(args.seed + 901)
    try:
        for i in range(args.samples):
            if i and i % max(1, args.episode_length) == 0:
                obs = env.reset()
            elif i == 0:
                obs = env.reset()
            goal = i % len(GOALS)
            self_vec = neutral_self_vector(env.energy)
            context = make_context(obs, self_vector=self_vec, goal=goal)
            _, _, distance, angle, visible = env._beacon()  # evaluator only; never teacher/student input
            evaluator = {"visible": bool(visible), "distance": float(distance),
                         "bearing_degrees": float(math.degrees(angle))}
            _save_record(dataset, env.render_rgb(args.image_size), context, "development-room",
                         f"current goal={GOALS[goal]}; this is a training nursery frame", evaluator)
            allowed = list(env.valid_action_indices())
            action = rng.choice([a for a in allowed if a in (0, 1, 2, 3)])
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
    finally:
        env.close()
    print(json.dumps({"dataset": str(dataset.directory), "records": len(dataset.records()),
                      "digest": dataset.digest()}, indent=2))


def collect_thor(args):
    from .embodied import AI2ThorConfig, AI2ThorEmbodiedEnv
    dataset = SemanticDataset(args.dataset)
    rng = random.Random(args.seed + 1701)
    scenes = [s.strip() for s in args.scene.split(",") if s.strip()]
    created = 0
    per_scene = max(1, math.ceil(args.samples / max(1, len(scenes))))
    for scene_index, scene in enumerate(scenes):
        cfg = AI2ThorConfig(scene=scene, executable=args.thor_executable, width=args.thor_width,
                            height=args.thor_height, max_episode_steps=args.max_episode_steps,
                            curriculum_level=2, goal_signal="internal-map", seed=args.seed + scene_index)
        env = AI2ThorEmbodiedEnv(cfg)
        local_created = 0
        try:
            obs = env.reset()
            while created < args.samples and local_created < per_scene:
                goal = created % len(GOALS)
                context = make_context(obs, self_vector=neutral_self_vector(env.energy), goal=goal)
                visible_objects = []
                metadata = getattr(getattr(env, "event", None), "metadata", {}) or {}
                for obj in metadata.get("objects", []):
                    if obj.get("visible"):
                        visible_objects.append(str(obj.get("objectType", "unknown")))
                evaluator = {"scene": scene, "visible_object_types": sorted(set(visible_objects))[:32]}
                _save_record(dataset, env.render_rgb(), context, "ai2thor-nursery",
                             f"current goal={GOALS[goal]}; egocentric 3D nursery frame", evaluator)
                created += 1
                local_created += 1
                if created >= args.samples:
                    break
                actions = [a for a in env.valid_action_indices() if a in (0, 1, 2, 3)]
                action = rng.choice(actions or [0])
                obs, _, done, info = env.step(action)
                if done or info.get("infrastructure_fault"):
                    obs = env.reset()
        finally:
            env.close()
        if created >= args.samples:
            break
    print(json.dumps({"dataset": str(dataset.directory), "records": len(dataset.records()),
                      "digest": dataset.digest()}, indent=2))


def label_dataset(args):
    dataset = SemanticDataset(args.dataset)
    rows = dataset.records()
    client = QwenTeacherClient(args.endpoint, args.teacher_model, args.api_key, args.timeout)
    labelled = 0
    for row in rows:
        if row.get("teacher") and not args.relabel:
            continue
        row["teacher"] = client.label(dataset.image_path(row), row["context"], row.get("allowed_text", ""))
        row["teacher_provenance"] = {"model": args.teacher_model, "endpoint_kind": "openai-compatible-vlm"}
        dataset.rewrite(rows)
        labelled += 1
        if labelled % 10 == 0:
            print(f"teacher_labelled={labelled}", flush=True)
        if args.limit and labelled >= args.limit:
            break
    print(json.dumps({"labelled": labelled, "total": len(rows), "digest": dataset.digest()}, indent=2))


def cache_dino(args):
    dataset = SemanticDataset(args.dataset)
    rows = dataset.records()
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=args.dino_model)
    Image = _pil()
    count = 0
    for row in rows:
        cache = dataset.cache_path(row["record_id"])
        if cache.exists() and not args.recache:
            continue
        rgb = np.asarray(Image.open(dataset.image_path(row)).convert("RGB"), dtype=np.uint8)
        vec = vision.encode(rgb)
        np.save(cache, np.asarray(vec, dtype=np.float32))
        count += 1
        if count % 25 == 0:
            print(f"dino_cached={count}", flush=True)
    metadata = {"model_id": getattr(vision, "model_id", args.dino_model),
                "output_dim": int(vision.output_dim), "records_cached": count}
    (dataset.directory / "dino_cache.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def _load_training_state(path, dino_dim, dataset_digest, device):
    cfg = SemanticConfig(dino_dim=dino_dim)
    model = SemanticCortex(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    updates = 0
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "continuous_brain.semantic_training.v070":
            raise ValueError("Unsupported semantic training checkpoint")
        if state["dataset_digest"] != dataset_digest:
            raise ValueError("Dataset changed since training checkpoint; use a new lab or --reset-training")
        cfg = SemanticConfig(**state["config"])
        if cfg.dino_dim != dino_dim:
            raise ValueError("Cached DINO dimension changed")
        model = SemanticCortex(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        optimizer.load_state_dict(state["optimizer"])
        updates = int(state["updates"])
    return model, optimizer, updates


def _save_training(path, model, optimizer, updates, dataset_digest, dino_model, teacher_models):
    state = {"format": "continuous_brain.semantic_training.v070", "config": asdict(model.cfg),
             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "updates": int(updates),
             "dataset_digest": dataset_digest, "dino_model": dino_model,
             "teacher_models": sorted(set(teacher_models))}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(args):
    dataset = SemanticDataset(args.dataset)
    cache_meta = json.loads((dataset.directory / "dino_cache.json").read_text())
    examples = training_examples(dataset, split="train", validation_fraction=args.validation_fraction, seed=args.seed)
    if not examples:
        raise ValueError("No labelled training examples; collect, label and cache DINO first")
    digest = dataset.digest()
    state_path = Path(args.lab) / "semantic_training.pt"
    Path(args.lab).mkdir(parents=True, exist_ok=True)
    if args.reset_training and state_path.exists():
        state_path.unlink()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, optimizer, updates = _load_training_state(state_path, int(cache_meta["output_dim"]), digest, device)
    rng = random.Random(args.seed + updates)
    teacher_models = [row.get("teacher_provenance", {}).get("model", "unknown") for row, _ in examples]
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    target = updates + args.updates
    try:
        while updates < target and not stopping:
            chosen = [rng.choice(examples) for _ in range(args.batch)]
            loaded = [load_example(dataset, row, qa) for row, qa in chosen]
            batch = {k: v.to(device) for k, v in collate_examples(loaded).items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = semantic_losses(model, batch, mimic_weight=args.mimic_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite semantic loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates % args.save_every == 0 or updates == target:
                _save_training(state_path, model, optimizer, updates, digest, cache_meta["model_id"], teacher_models)
                print(f"semantic_updates={updates} loss={metrics['loss']:.5f} answer={metrics['answer']:.5f} "
                      f"concept={metrics['concept']:.5f} care={metrics['care']:.5f} grad={float(grad):.4f}", flush=True)
    finally:
        _save_training(state_path, model, optimizer, updates, digest, cache_meta["model_id"], teacher_models)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    print(json.dumps({"updates": updates, "parameters": model.parameter_count,
                      "checkpoint": str(state_path), "device": str(device)}, indent=2))


def _f1(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def evaluate(args):
    dataset = SemanticDataset(args.dataset)
    state_path = Path(args.lab) / "semantic_training.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = SemanticCortex(SemanticConfig(**state["config"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    examples = training_examples(dataset, split="validation", validation_fraction=args.validation_fraction, seed=args.seed)
    if len(examples) < args.min_examples:
        raise ValueError(f"Need at least {args.min_examples} held-out Q/A examples for qualification")
    token_correct = token_total = 0
    concept_scores, care_mae, mimic_cos, text_similarity, nonempty = [], [], [], [], 0
    with torch.no_grad():
        for row, qa in examples:
            item = load_example(dataset, row, qa)
            batch = collate_examples([item])
            answer = batch["answer_ids"]
            result = model(batch["dino"], batch["context"], batch["question_ids"], answer[:, :-1])
            pred = result["answer_logits"].argmax(dim=-1)
            truth = answer[:, 1:]
            mask = truth != ByteCodec.PAD
            token_correct += int((pred[mask] == truth[mask]).sum())
            token_total += int(mask.sum())
            concept_scores.append(_f1((torch.sigmoid(result["concept_logits"])[0].numpy() >= 0.5),
                                      item["concepts"] >= 0.5))
            care_mae.append(float(np.abs(torch.sigmoid(result["care_logits"])[0].numpy() - item["care"]).mean()))
            mimic_cos.append(float(torch.nn.functional.cosine_similarity(result["visual_code"], batch["pixel_target"]).item()))
            generated = model.generate(batch["dino"], batch["context"], batch["question_ids"])
            if generated.strip():
                nonempty += 1
            text_similarity.append(difflib.SequenceMatcher(None, generated.lower().strip(), qa["answer"].lower().strip()).ratio())
    metrics = {
        "examples": len(examples),
        "answer_token_accuracy": token_correct / max(1, token_total),
        "concept_f1": float(np.mean(concept_scores)),
        "care_mae": float(np.mean(care_mae)),
        "legacy_visual_cosine": float(np.mean(mimic_cos)),
        "generated_similarity": float(np.mean(text_similarity)),
        "generated_nonempty_rate": nonempty / len(examples),
    }
    passed = (metrics["answer_token_accuracy"] >= args.min_token_accuracy and
              metrics["concept_f1"] >= args.min_concept_f1 and
              metrics["care_mae"] <= args.max_care_mae and
              metrics["legacy_visual_cosine"] >= args.min_mimic_cosine and
              metrics["generated_similarity"] >= args.min_generated_similarity and
              metrics["generated_nonempty_rate"] >= 0.95)
    report = {"format": "continuous_brain.semantic_qualification.v070", "passed": bool(passed),
              "metrics": metrics, "thresholds": {
                  "min_token_accuracy": args.min_token_accuracy, "min_concept_f1": args.min_concept_f1,
                  "max_care_mae": args.max_care_mae, "min_mimic_cosine": args.min_mimic_cosine,
                  "min_generated_similarity": args.min_generated_similarity},
              "training_updates": state["updates"], "dataset_digest": state["dataset_digest"]}
    lab = Path(args.lab)
    (lab / "semantic_evaluation.json").write_text(json.dumps(report, indent=2))
    if passed:
        model.load_state_dict(state["model"], strict=True)
        save_semantic_bundle(lab / "qualified_semantic.pt", model, metadata={
            "qualified": True, "qualification": report, "dino_model": state["dino_model"],
            "teacher_models": state.get("teacher_models", []), "training_updates": state["updates"],
            "dataset_digest": state["dataset_digest"],
        })
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def _attached_bundle(state_dir: Path, explicit=None):
    if explicit:
        return Path(explicit)
    path = state_dir / "semantic_cortex" / "qualified_semantic.pt"
    if not path.is_file():
        raise FileNotFoundError("No attached qualified semantic cortex; run the v0.7 upgrade/attachment step")
    return path


def _state_context(state_dir: Path):
    store = LifetimeStore(state_dir)
    try:
        state = store.load()
        if state is None:
            raise ValueError("No individual checkpoint")
        latest = store.db.execute("SELECT value,metadata FROM experiences WHERE kind='experience' ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        store.close()
    obs = np.asarray(state.get("observation", np.zeros(6)), dtype=np.float32)
    if obs.shape != (6,):
        raise ValueError("Checkpoint does not contain a six-value current observation")
    self_model = SelfModel(**state["self_model"])
    intention = state["intention"]
    self_vec = self_model.vector(self_model.decisions - int(intention.get("started", 0)))
    goal = GOALS.index(intention.get("goal", GOALS[0]))
    memory = np.zeros(8, dtype=np.float32)
    last_skill = None
    if latest and latest[0] is not None:
        values = np.frombuffer(latest[0], dtype=np.float32)
        if values.size >= 8:
            memory = values[-8:].copy()
            if memory[:6].max() > 0:
                last_skill = int(memory[:6].argmax())
    return state, make_context(obs, self_vector=self_vec, goal=goal, last_skill=last_skill, memory_summary=memory)


def talk(args):
    state_dir = Path(args.state_dir)
    state, context = _state_context(state_dir)
    model, bundle = load_semantic_bundle(_attached_bundle(state_dir, args.semantic_bundle), require_qualified=True)
    dino_id = bundle["metadata"]["dino_model"]
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=dino_id)
    if vision.output_dim != model.cfg.dino_dim:
        raise ValueError("Attached semantic cortex expects a different DINO feature dimension")
    if args.image:
        Image = _pil()
        rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    elif state["world_signature"]["type"] == "DevelopmentRoom":
        env = DevelopmentRoom(state["environment"].get("rng", None) and 7 or 7)
        env.load_state_dict(state["environment"])
        rgb = env.render_rgb(args.image_size)
        env.close()
    else:
        raise ValueError("A live/native simulator frame cannot be reconstructed safely; pass --image for this checkpoint")
    dino = torch.tensor(vision.encode(rgb), dtype=torch.float32)[None]
    question = torch.tensor([ByteCodec.encode(args.question, model.cfg.max_question_bytes)], dtype=torch.long)
    answer = model.generate(dino, torch.tensor(context, dtype=torch.float32)[None], question)
    print(answer)



def chat(args):
    state_dir = Path(args.state_dir)
    state, context = _state_context(state_dir)
    model, bundle = load_semantic_bundle(_attached_bundle(state_dir, args.semantic_bundle), require_qualified=True)
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=bundle["metadata"]["dino_model"])
    if vision.output_dim != model.cfg.dino_dim:
        raise ValueError("Attached semantic cortex expects a different DINO feature dimension")
    if args.image:
        Image = _pil()
        rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    elif state["world_signature"]["type"] == "DevelopmentRoom":
        env = DevelopmentRoom(7)
        env.load_state_dict(state["environment"])
        rgb = env.render_rgb(args.image_size)
        env.close()
    else:
        raise ValueError("Pass --image for native simulator checkpoints until live chat integration is enabled")
    dino = torch.tensor(vision.encode(rgb), dtype=torch.float32)[None]
    ctx = torch.tensor(context, dtype=torch.float32)[None]
    print("Continuous Brain v0.7 teacher-free chat. Type /quit to exit.")
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        q = torch.tensor([ByteCodec.encode(question, model.cfg.max_question_bytes)], dtype=torch.long)
        print("brain> " + model.generate(dino, ctx, q), flush=True)

def parser():
    p = argparse.ArgumentParser(description="Continuous Brain v0.7 developmental semantic lab")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("collect-development", help="collect deterministic nursery frames without loading the individual")
    d.add_argument("--dataset", default="state/semantic_dataset_v070")
    d.add_argument("--samples", type=int, default=1000); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--episode-length", type=int, default=50); d.add_argument("--max-episode-steps", type=int, default=300)
    d.add_argument("--image-size", type=int, default=224); d.set_defaults(func=collect_development)

    t = sub.add_parser("collect-thor", help="collect real 3D egocentric nursery frames; metadata is evaluator-only")
    t.add_argument("--dataset", default="state/semantic_dataset_v070"); t.add_argument("--samples", type=int, default=3000)
    t.add_argument("--scene", default="FloorPlan1Discovery_physics"); t.add_argument("--thor-executable", required=True)
    t.add_argument("--thor-width", type=int, default=320); t.add_argument("--thor-height", type=int, default=240)
    t.add_argument("--max-episode-steps", type=int, default=300); t.add_argument("--seed", type=int, default=7)
    t.set_defaults(func=collect_thor)

    l = sub.add_parser("label", help="Qwen VLM teacher creates structured labels; teacher is never runtime policy")
    l.add_argument("--dataset", default="state/semantic_dataset_v070")
    l.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    l.add_argument("--teacher-model", default="Qwen/Qwen3.8-27B"); l.add_argument("--api-key")
    l.add_argument("--timeout", type=int, default=180); l.add_argument("--limit", type=int, default=0)
    l.add_argument("--relabel", action="store_true"); l.set_defaults(func=label_dataset)

    c = sub.add_parser("cache-dino", help="freeze DINOv3 and cache one feature vector per real frame")
    c.add_argument("--dataset", default="state/semantic_dataset_v070")
    c.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    c.add_argument("--vision-device", default="auto"); c.add_argument("--recache", action="store_true")
    c.set_defaults(func=cache_dino)

    r = sub.add_parser("train", help="train only the student's semantic/communication cortex")
    r.add_argument("--dataset", default="state/semantic_dataset_v070"); r.add_argument("--lab", default="state/semantic_lab_v070")
    r.add_argument("--updates", type=int, default=10000); r.add_argument("--batch", type=int, default=8)
    r.add_argument("--device", default="auto"); r.add_argument("--seed", type=int, default=7)
    r.add_argument("--save-every", type=int, default=250); r.add_argument("--validation-fraction", type=float, default=0.1)
    r.add_argument("--mimic-weight", type=float, default=0.2); r.add_argument("--reset-training", action="store_true")
    r.set_defaults(func=train)

    e = sub.add_parser("evaluate", help="teacher-free held-out qualification; writes attachable snapshot only on pass")
    e.add_argument("--dataset", default="state/semantic_dataset_v070"); e.add_argument("--lab", default="state/semantic_lab_v070")
    e.add_argument("--seed", type=int, default=7); e.add_argument("--validation-fraction", type=float, default=0.1)
    e.add_argument("--min-examples", type=int, default=100); e.add_argument("--min-token-accuracy", type=float, default=0.55)
    e.add_argument("--min-concept-f1", type=float, default=0.60); e.add_argument("--max-care-mae", type=float, default=0.18)
    e.add_argument("--min-mimic-cosine", type=float, default=0.60); e.add_argument("--min-generated-similarity", type=float, default=0.35)
    e.set_defaults(func=evaluate)

    q = sub.add_parser("talk", help="teacher-free basic communication using an attached qualified cortex")
    q.add_argument("--state-dir", required=True); q.add_argument("--question", required=True)
    q.add_argument("--semantic-bundle"); q.add_argument("--image"); q.add_argument("--image-size", type=int, default=224)
    q.add_argument("--vision-device", default="auto"); q.set_defaults(func=talk)

    ch = sub.add_parser("chat", help="interactive teacher-free first-conversation shell")
    ch.add_argument("--state-dir", required=True); ch.add_argument("--semantic-bundle"); ch.add_argument("--image")
    ch.add_argument("--image-size", type=int, default=224); ch.add_argument("--vision-device", default="auto")
    ch.set_defaults(func=chat)
    return p


def main():
    args = parser().parse_args()
    if hasattr(args, "samples") and args.samples < 1:
        raise SystemExit("--samples must be positive")
    if hasattr(args, "updates") and args.updates < 1:
        raise SystemExit("--updates must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
