"""Developmental semantic/dialogue training CLI for Continuous Brain v0.7.1."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
import math
from pathlib import Path
import random
import shutil
import signal

import numpy as np
import torch

from .development import DevelopmentRoom
from .dialogue import (
    CARE_STOP_REPLY,
    DIALOGUE_INTENTS,
    UNKNOWN_REPLY,
    augment_teacher_for_dialogue,
    goal_from_allowed_text,
    identity_answer,
    intent_index,
    past_skill_answer,
    primitive_to_skill,
    reasoning_answer,
    state_answer,
    temporal_memory_summary,
)
from .lifetime import LifetimeStore, SelfModel
from .semantic import (
    BASIC_CONCEPTS,
    CARE_KEYS,
    ByteCodec,
    SemanticConfig,
    SemanticCortex,
    bootstrap_from_v070,
    load_semantic_bundle,
    save_semantic_bundle,
    semantic_losses,
)
from .semantic_data import (
    SemanticDataset,
    SemanticRecord,
    collate_examples,
    load_example,
    make_context,
    neutral_self_vector,
    training_examples,
)
from .skills import GOALS, SKILLS
from .teacher import QwenTeacherClient
from .teacher_guard import repair_teacher_label
from .vision import build_vision_backbone


OOD_PROBES = (
    "How many moons does Jupiter have?",
    "Can you derive the quadratic formula?",
    "What happened in the news today?",
    "Please translate this sentence into Japanese.",
    "What is your opinion of abstract algebra?",
    "Who was the first emperor of China?",
    "Can you write a database query for me?",
    "What is the weather tomorrow?",
    "Explain photosynthesis in detail.",
    "Which movie should I watch tonight?",
)


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


def _save_record(dataset, rgb, context, source, allowed_text, evaluator=None, temporal=None):
    Image = _pil()
    index = _next_id(dataset)
    record_id = f"sample-{index:07d}"
    relative = f"images/{record_id}.png"
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(dataset.directory / relative)
    row = SemanticRecord(
        record_id,
        relative,
        np.asarray(context, np.float32).tolist(),
        source,
        evaluator=evaluator,
        temporal=temporal,
    ).__dict__
    row["allowed_text"] = allowed_text
    dataset.append(row)
    return record_id


def _temporal_record(previous_skill=None, previous_reward=0.0, previous_success=False):
    return {
        "has_previous": previous_skill is not None,
        "previous_skill": previous_skill,
        "previous_reward": float(previous_reward),
        "previous_success": bool(previous_success),
    }


def collect_development(args):
    """Append grounded DevelopmentRoom frames with a runtime-compatible memory tail."""
    dataset = SemanticDataset(args.dataset)
    env = DevelopmentRoom(args.seed, args.max_episode_steps)
    rng = random.Random(args.seed + 901)
    prev_skill, prev_reward, prev_success = None, 0.0, False
    try:
        for i in range(args.samples):
            reset_now = False
            if i and i % max(1, args.episode_length) == 0:
                obs = env.reset(); reset_now = True
            elif i == 0:
                obs = env.reset(); reset_now = True
            if reset_now:
                prev_skill, prev_reward, prev_success = None, 0.0, False

            goal = i % len(GOALS)
            goal_name = GOALS[goal]
            memory = temporal_memory_summary(prev_skill, prev_reward, prev_success)
            self_vec = neutral_self_vector(env.energy)
            context = make_context(obs, self_vector=self_vec, goal=goal, memory_summary=memory)
            _, _, distance, angle, visible = env._beacon()  # evaluator-only truth
            evaluator = {
                "visible": bool(visible),
                "distance": float(distance),
                "bearing_degrees": float(math.degrees(angle)),
            }
            temporal = _temporal_record(prev_skill, prev_reward, prev_success)
            _save_record(
                dataset,
                env.render_rgb(args.image_size),
                context,
                "development-room-v071",
                f"current goal={goal_name}; this is a grounded dialogue nursery frame",
                evaluator,
                temporal,
            )

            allowed = list(env.valid_action_indices())
            action = rng.choice([a for a in allowed if a in (0, 1, 2, 3)])
            action_name = env.action_names[action]
            obs, reward, done, info = env.step(action)
            prev_skill = primitive_to_skill(action_name, goal_name)
            prev_reward = float(reward)
            prev_success = bool(info.get("last_action_success", getattr(env, "last_action_success", True)))
            if done:
                obs = env.reset()
                prev_skill, prev_reward, prev_success = None, 0.0, False
    finally:
        env.close()
    print(json.dumps({"dataset": str(dataset.directory), "records": len(dataset.records()),
                      "digest": dataset.digest()}, indent=2))


def collect_thor(args):
    """Append real 3D egocentric frames with last-skill/outcome memory context."""
    from .embodied import AI2ThorConfig, AI2ThorEmbodiedEnv

    dataset = SemanticDataset(args.dataset)
    rng = random.Random(args.seed + 1701)
    scenes = [s.strip() for s in args.scene.split(",") if s.strip()]
    created = 0
    per_scene = max(1, math.ceil(args.samples / max(1, len(scenes))))
    for scene_index, scene in enumerate(scenes):
        cfg = AI2ThorConfig(
            scene=scene,
            executable=args.thor_executable,
            width=args.thor_width,
            height=args.thor_height,
            max_episode_steps=args.max_episode_steps,
            curriculum_level=2,
            goal_signal="internal-map",
            seed=args.seed + scene_index,
        )
        env = AI2ThorEmbodiedEnv(cfg)
        local_created = 0
        prev_skill, prev_reward, prev_success = None, 0.0, False
        try:
            obs = env.reset()
            while created < args.samples and local_created < per_scene:
                goal = created % len(GOALS)
                goal_name = GOALS[goal]
                memory = temporal_memory_summary(prev_skill, prev_reward, prev_success)
                context = make_context(obs, self_vector=neutral_self_vector(env.energy), goal=goal,
                                       memory_summary=memory)
                visible_objects = []
                metadata = getattr(getattr(env, "event", None), "metadata", {}) or {}
                for obj in metadata.get("objects", []):
                    if obj.get("visible"):
                        visible_objects.append(str(obj.get("objectType", "unknown")))
                evaluator = {"scene": scene, "visible_object_types": sorted(set(visible_objects))[:32]}
                temporal = _temporal_record(prev_skill, prev_reward, prev_success)
                _save_record(
                    dataset,
                    env.render_rgb(),
                    context,
                    "ai2thor-dialogue-v071",
                    f"current goal={goal_name}; egocentric 3D grounded dialogue frame",
                    evaluator,
                    temporal,
                )
                created += 1
                local_created += 1
                if created >= args.samples:
                    break

                actions = [a for a in env.valid_action_indices() if a in (0, 1, 2, 3)]
                action = rng.choice(actions or [0])
                action_name = env.action_names[action]
                obs, reward, done, info = env.step(action)
                prev_skill = primitive_to_skill(action_name, goal_name)
                prev_reward = float(reward)
                prev_success = bool(info.get("last_action_success", getattr(env, "last_action_success", True)))
                if done or info.get("infrastructure_fault"):
                    obs = env.reset()
                    prev_skill, prev_reward, prev_success = None, 0.0, False
        finally:
            env.close()
        if created >= args.samples:
            break
    print(json.dumps({"dataset": str(dataset.directory), "records": len(dataset.records()),
                      "digest": dataset.digest()}, indent=2))


def clone_dataset(args):
    source, destination = Path(args.source), Path(args.destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError("Destination exists; preserve the source and choose a new dataset path")
    shutil.copytree(source, destination)
    print(json.dumps({"source": str(source), "destination": str(destination),
                      "records": len(SemanticDataset(destination).records())}, indent=2))


def label_dataset(args):
    dataset = SemanticDataset(args.dataset)
    rows = dataset.records()
    client = QwenTeacherClient(args.endpoint, args.teacher_model, args.api_key, args.timeout)
    labelled = 0
    for row in rows:
        if row.get("teacher") and not args.relabel:
            continue
        teacher = client.label(dataset.image_path(row), row["context"], row.get("allowed_text", ""))
        row["teacher"] = repair_teacher_label(teacher, row.get("allowed_text", ""))
        row["teacher_provenance"] = {"model": args.teacher_model, "endpoint_kind": "openai-compatible-vlm"}
        dataset.rewrite(rows)
        labelled += 1
        if labelled % 10 == 0:
            print(f"teacher_labelled={labelled}", flush=True)
        if args.limit and labelled >= args.limit:
            break
    print(json.dumps({"labelled": labelled, "total": len(rows), "digest": dataset.digest()}, indent=2))


def prepare_dialogue(args):
    dataset = SemanticDataset(args.dataset)
    rows = dataset.records()
    changed = 0
    qas = 0
    temporal_rows = 0
    for row in rows:
        if not row.get("teacher"):
            continue
        augment_teacher_for_dialogue(row, individual_name=args.individual_name)
        changed += 1
        qas += len(row["teacher"].get("qas", []))
        temporal_rows += int(bool((row.get("temporal") or {}).get("has_previous")))
    dataset.rewrite(rows)
    print(json.dumps({
        "format": "continuous_brain.dialogue_curriculum.v071",
        "records_augmented": changed,
        "qa_examples": qas,
        "temporal_records": temporal_rows,
        "individual_name": args.individual_name,
        "digest": dataset.digest(),
    }, indent=2))


def cache_dino(args):
    dataset = SemanticDataset(args.dataset)
    rows = dataset.records()
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=args.dino_model)
    Image = _pil()
    newly_cached = 0
    for row in rows:
        cache = dataset.cache_path(row["record_id"])
        if cache.exists() and not args.recache:
            continue
        rgb = np.asarray(Image.open(dataset.image_path(row)).convert("RGB"), dtype=np.uint8)
        vec = vision.encode(rgb)
        np.save(cache, np.asarray(vec, dtype=np.float32))
        newly_cached += 1
        if newly_cached % 25 == 0:
            print(f"dino_cached={newly_cached}", flush=True)
    total_cached = sum(dataset.cache_path(row["record_id"]).exists() for row in rows)
    metadata = {
        "model_id": getattr(vision, "model_id", args.dino_model),
        "output_dim": int(vision.output_dim),
        "records_cached_new": newly_cached,
        "records_cached_total": total_cached,
        "records_total": len(rows),
    }
    (dataset.directory / "dino_cache.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def _load_training_state(path, dino_dim, dataset_digest, device, bootstrap_semantic=None):
    cfg = SemanticConfig(dino_dim=dino_dim)
    model = SemanticCortex(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    updates = 0
    bootstrap = None
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "continuous_brain.semantic_training.v071":
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
        bootstrap = state.get("bootstrap")
    elif bootstrap_semantic:
        model, old_bundle = bootstrap_from_v070(bootstrap_semantic, dino_dim=dino_dim, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        bootstrap = {
            "source": str(Path(bootstrap_semantic).resolve()),
            "format": old_bundle.get("format"),
            "digest": old_bundle.get("digest"),
        }
    return model, optimizer, updates, bootstrap


def _save_training(path, model, optimizer, updates, dataset_digest, dino_model, teacher_models, bootstrap):
    state = {
        "format": "continuous_brain.semantic_training.v071",
        "config": asdict(model.cfg),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "updates": int(updates),
        "dataset_digest": dataset_digest,
        "dino_model": dino_model,
        "teacher_models": sorted(set(teacher_models)),
        "bootstrap": bootstrap,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(args):
    dataset = SemanticDataset(args.dataset)
    cache_meta = json.loads((dataset.directory / "dino_cache.json").read_text())
    examples = training_examples(dataset, split="train", validation_fraction=args.validation_fraction, seed=args.seed)
    if not examples:
        raise ValueError("No labelled training examples; collect, label, prepare-dialogue and cache DINO first")
    digest = dataset.digest()
    state_path = Path(args.lab) / "semantic_training.pt"
    Path(args.lab).mkdir(parents=True, exist_ok=True)
    if args.reset_training and state_path.exists():
        state_path.unlink()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, optimizer, updates, bootstrap = _load_training_state(
        state_path, int(cache_meta["output_dim"]), digest, device, args.bootstrap_semantic
    )
    rng = random.Random(args.seed + updates)
    teacher_models = [row.get("teacher_provenance", {}).get("model", "trusted-curriculum") for row, _ in examples]
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
            loss, metrics = semantic_losses(
                model,
                batch,
                mimic_weight=args.mimic_weight,
                intent_weight=args.intent_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite semantic loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates % args.save_every == 0 or updates == target:
                _save_training(state_path, model, optimizer, updates, digest, cache_meta["model_id"],
                               teacher_models, bootstrap)
                print(
                    f"semantic_updates={updates} loss={metrics['loss']:.5f} answer={metrics['answer']:.5f} "
                    f"intent={metrics['intent']:.5f} concept={metrics['concept']:.5f} "
                    f"care={metrics['care']:.5f} grad={float(grad):.4f}",
                    flush=True,
                )
    finally:
        _save_training(state_path, model, optimizer, updates, digest, cache_meta["model_id"], teacher_models, bootstrap)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    print(json.dumps({"updates": updates, "parameters": model.parameter_count,
                      "checkpoint": str(state_path), "device": str(device), "bootstrap": bootstrap}, indent=2))


def _f1(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def _classify_question(model, question: str):
    q = torch.tensor([ByteCodec.encode(question, model.cfg.max_question_bytes)], dtype=torch.long)
    idx, confidence = model.classify_intent(q)
    return DIALOGUE_INTENTS[int(idx[0])], float(confidence[0])


def evaluate(args):
    dataset = SemanticDataset(args.dataset)
    state_path = Path(args.lab) / "semantic_training.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "continuous_brain.semantic_training.v071":
        raise ValueError("Expected a v0.7.1 semantic training checkpoint")
    model = SemanticCortex(SemanticConfig(**state["config"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    examples = training_examples(dataset, split="validation", validation_fraction=args.validation_fraction, seed=args.seed)
    if len(examples) < args.min_examples:
        raise ValueError(f"Need at least {args.min_examples} held-out Q/A examples for qualification")

    token_correct = token_total = 0
    intent_correct = intent_total = 0
    unknown_correct = unknown_total = 0
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
            predicted_intent = int(result["intent_logits"].argmax(dim=-1)[0])
            true_intent = int(item["intent"])
            intent_correct += int(predicted_intent == true_intent)
            intent_total += 1
            if qa.get("kind") == "unknown":
                unknown_correct += int(predicted_intent == intent_index("unknown"))
                unknown_total += 1
            concept_scores.append(_f1((torch.sigmoid(result["concept_logits"])[0].numpy() >= 0.5),
                                      item["concepts"] >= 0.5))
            care_mae.append(float(np.abs(torch.sigmoid(result["care_logits"])[0].numpy() - item["care"]).mean()))
            mimic_cos.append(float(torch.nn.functional.cosine_similarity(result["visual_code"], batch["pixel_target"]).item()))
            generated = model.generate(batch["dino"], batch["context"], batch["question_ids"])
            if generated.strip():
                nonempty += 1
            text_similarity.append(difflib.SequenceMatcher(
                None, generated.lower().strip(), qa["answer"].lower().strip()).ratio())

    ood_correct = 0
    for question in OOD_PROBES:
        intent, confidence = _classify_question(model, question)
        ood_correct += int(intent == "unknown" or confidence < args.intent_confidence)

    metrics = {
        "examples": len(examples),
        "answer_token_accuracy": token_correct / max(1, token_total),
        "intent_accuracy": intent_correct / max(1, intent_total),
        "unknown_intent_accuracy": unknown_correct / max(1, unknown_total),
        "ood_unknown_rejection": ood_correct / len(OOD_PROBES),
        "concept_f1": float(np.mean(concept_scores)),
        "care_mae": float(np.mean(care_mae)),
        "legacy_visual_cosine": float(np.mean(mimic_cos)),
        "generated_similarity": float(np.mean(text_similarity)),
        "generated_nonempty_rate": nonempty / len(examples),
    }
    passed = (
        metrics["answer_token_accuracy"] >= args.min_token_accuracy
        and metrics["intent_accuracy"] >= args.min_intent_accuracy
        and metrics["unknown_intent_accuracy"] >= args.min_unknown_accuracy
        and metrics["ood_unknown_rejection"] >= args.min_ood_unknown
        and metrics["concept_f1"] >= args.min_concept_f1
        and metrics["care_mae"] <= args.max_care_mae
        and metrics["legacy_visual_cosine"] >= args.min_mimic_cosine
        and metrics["generated_similarity"] >= args.min_generated_similarity
        and metrics["generated_nonempty_rate"] >= 0.95
    )
    report = {
        "format": "continuous_brain.semantic_qualification.v071",
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": {
            "min_token_accuracy": args.min_token_accuracy,
            "min_intent_accuracy": args.min_intent_accuracy,
            "min_unknown_accuracy": args.min_unknown_accuracy,
            "min_ood_unknown": args.min_ood_unknown,
            "max_care_mae": args.max_care_mae,
            "min_concept_f1": args.min_concept_f1,
            "min_mimic_cosine": args.min_mimic_cosine,
            "min_generated_similarity": args.min_generated_similarity,
            "intent_confidence": args.intent_confidence,
        },
        "training_updates": state["updates"],
        "dataset_digest": state["dataset_digest"],
        "bootstrap": state.get("bootstrap"),
    }
    lab = Path(args.lab)
    (lab / "semantic_evaluation.json").write_text(json.dumps(report, indent=2))
    if passed:
        save_semantic_bundle(lab / "qualified_semantic.pt", model, metadata={
            "qualified": True,
            "qualification": report,
            "dino_model": state["dino_model"],
            "teacher_models": state.get("teacher_models", []),
            "training_updates": state["updates"],
            "dataset_digest": state["dataset_digest"],
            "bootstrap": state.get("bootstrap"),
            "dialogue_version": "0.7.1",
        })
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def _attached_bundle(state_dir: Path, explicit=None):
    if explicit:
        return Path(explicit)
    path = state_dir / "semantic_cortex" / "qualified_semantic.pt"
    if not path.is_file():
        raise FileNotFoundError("No attached qualified v0.7.1 semantic cortex; run the v0.7.1 attachment step")
    return path


def _state_context(state_dir: Path):
    store = LifetimeStore(state_dir)
    try:
        state = store.load()
        if state is None:
            raise ValueError("No individual checkpoint")
        latest = store.db.execute(
            "SELECT value,metadata FROM experiences WHERE kind='experience' ORDER BY id DESC LIMIT 1"
        ).fetchone()
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
    latest_meta = {}
    if latest:
        if latest[0] is not None:
            values = np.frombuffer(latest[0], dtype=np.float32)
            if values.size >= 8:
                memory = values[-8:].copy()
                if memory[:6].max() > 0:
                    last_skill = int(memory[:6].argmax())
        try:
            latest_meta = json.loads(latest[1] or "{}")
        except (TypeError, json.JSONDecodeError):
            latest_meta = {}
    context = make_context(obs, self_vector=self_vec, goal=goal, last_skill=last_skill, memory_summary=memory)
    return state, context, latest_meta


def _runtime_rgb(state, args):
    if args.image:
        Image = _pil()
        return np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    if state["world_signature"]["type"] == "DevelopmentRoom":
        env = DevelopmentRoom(7)
        env.load_state_dict(state["environment"])
        try:
            return env.render_rgb(args.image_size)
        finally:
            env.close()
    raise ValueError("Pass --image for native simulator checkpoints until live camera chat integration is enabled")


def _grounded_reply(model, dino, context, state, latest_meta, question, confidence_threshold=0.55):
    q = torch.tensor([ByteCodec.encode(question, model.cfg.max_question_bytes)], dtype=torch.long)
    idx, confidence = model.classify_intent(q)
    intent = DIALOGUE_INTENTS[int(idx[0])]
    confidence = float(confidence[0])
    goal = state.get("intention", {}).get("goal")

    if confidence < confidence_threshold or intent == "unknown":
        return UNKNOWN_REPLY, intent, confidence
    if intent == "identity":
        return identity_answer(state.get("self_model", {}).get("name")), intent, confidence
    if intent == "state":
        return state_answer(goal), intent, confidence
    if intent == "reasoning":
        return reasoning_answer(goal), intent, confidence
    if intent == "temporal":
        lower = question.lower()
        if not latest_meta:
            return "I don't have a grounded recent action to report yet.", intent, confidence
        if "succeed" in lower or "work" in lower:
            success = bool(latest_meta.get("outcome", latest_meta.get("task_success", False)))
            return ("Yes, my last action succeeded." if success
                    else "No, my last action did not succeed."), intent, confidence
        return past_skill_answer(latest_meta.get("skill")), intent, confidence

    answer = model.generate(dino, context, q)
    return answer, intent, confidence


def talk(args):
    state_dir = Path(args.state_dir)
    state, context, latest_meta = _state_context(state_dir)
    model, bundle = load_semantic_bundle(_attached_bundle(state_dir, args.semantic_bundle), require_qualified=True)
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=bundle["metadata"]["dino_model"])
    if vision.output_dim != model.cfg.dino_dim:
        raise ValueError("Attached semantic cortex expects a different DINO feature dimension")
    rgb = _runtime_rgb(state, args)
    dino = torch.tensor(vision.encode(rgb), dtype=torch.float32)[None]
    ctx = torch.tensor(context, dtype=torch.float32)[None]
    answer, intent, confidence = _grounded_reply(
        model, dino, ctx, state, latest_meta, args.question, args.intent_confidence
    )
    print(answer)
    if args.show_intent:
        print(f"[intent={intent} confidence={confidence:.3f}]")


def chat(args):
    state_dir = Path(args.state_dir)
    state, context, latest_meta = _state_context(state_dir)
    model, bundle = load_semantic_bundle(_attached_bundle(state_dir, args.semantic_bundle), require_qualified=True)
    vision = build_vision_backbone("dinov3", device=args.vision_device, model_id=bundle["metadata"]["dino_model"])
    if vision.output_dim != model.cfg.dino_dim:
        raise ValueError("Attached semantic cortex expects a different DINO feature dimension")
    dino = torch.tensor(vision.encode(_runtime_rgb(state, args)), dtype=torch.float32)[None]
    ctx = torch.tensor(context, dtype=torch.float32)[None]
    print("Continuous Brain v0.7.1 grounded dialogue. Type /quit to exit.")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        answer, intent, confidence = _grounded_reply(
            model, dino, ctx, state, latest_meta, question, args.intent_confidence
        )
        suffix = f" [intent={intent} confidence={confidence:.3f}]" if args.show_intent else ""
        print("brain> " + answer + suffix, flush=True)


def parser():
    p = argparse.ArgumentParser(description="Continuous Brain v0.7.1 grounded dialogue developmental lab")
    sub = p.add_subparsers(dest="command", required=True)

    cl = sub.add_parser("clone-dataset", help="preserve v0.7 data while creating a v0.7.1 working dataset")
    cl.add_argument("--source", required=True); cl.add_argument("--destination", required=True)
    cl.set_defaults(func=clone_dataset)

    d = sub.add_parser("collect-development", help="append deterministic grounded dialogue nursery frames")
    d.add_argument("--dataset", default="state/semantic_dataset_v071")
    d.add_argument("--samples", type=int, default=500); d.add_argument("--seed", type=int, default=17)
    d.add_argument("--episode-length", type=int, default=50); d.add_argument("--max-episode-steps", type=int, default=300)
    d.add_argument("--image-size", type=int, default=224); d.set_defaults(func=collect_development)

    t = sub.add_parser("collect-thor", help="append real 3D grounded dialogue frames with temporal context")
    t.add_argument("--dataset", default="state/semantic_dataset_v071"); t.add_argument("--samples", type=int, default=1000)
    t.add_argument("--scene", default="FloorPlan1Discovery_physics"); t.add_argument("--thor-executable", required=True)
    t.add_argument("--thor-width", type=int, default=320); t.add_argument("--thor-height", type=int, default=240)
    t.add_argument("--max-episode-steps", type=int, default=300); t.add_argument("--seed", type=int, default=17)
    t.set_defaults(func=collect_thor)

    l = sub.add_parser("label", help="VLM teacher labels visual evidence only; trusted guard supplies self/care truth")
    l.add_argument("--dataset", default="state/semantic_dataset_v071")
    l.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    l.add_argument("--teacher-model", default="Qwen/Qwen3.8-27B"); l.add_argument("--api-key")
    l.add_argument("--timeout", type=int, default=180); l.add_argument("--limit", type=int, default=0)
    l.add_argument("--relabel", action="store_true"); l.set_defaults(func=label_dataset)

    pd = sub.add_parser("prepare-dialogue", help="add identity, paraphrase, temporal and unknown-question curriculum")
    pd.add_argument("--dataset", default="state/semantic_dataset_v071")
    pd.add_argument("--individual-name", default="Continuous Brain")
    pd.set_defaults(func=prepare_dialogue)

    c = sub.add_parser("cache-dino", help="freeze DINOv3 and cache one feature vector per real frame")
    c.add_argument("--dataset", default="state/semantic_dataset_v071")
    c.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    c.add_argument("--vision-device", default="auto"); c.add_argument("--recache", action="store_true")
    c.set_defaults(func=cache_dino)

    r = sub.add_parser("train", help="train the v0.7.1 semantic/dialogue cortex")
    r.add_argument("--dataset", default="state/semantic_dataset_v071"); r.add_argument("--lab", default="state/semantic_lab_v071")
    r.add_argument("--updates", type=int, default=5000); r.add_argument("--batch", type=int, default=8)
    r.add_argument("--device", default="auto"); r.add_argument("--seed", type=int, default=17)
    r.add_argument("--save-every", type=int, default=250); r.add_argument("--validation-fraction", type=float, default=0.1)
    r.add_argument("--mimic-weight", type=float, default=0.2); r.add_argument("--intent-weight", type=float, default=0.3)
    r.add_argument("--bootstrap-semantic", help="qualified v0.7.0 semantic bundle to continue from")
    r.add_argument("--reset-training", action="store_true"); r.set_defaults(func=train)

    e = sub.add_parser("evaluate", help="teacher-free held-out dialogue qualification")
    e.add_argument("--dataset", default="state/semantic_dataset_v071"); e.add_argument("--lab", default="state/semantic_lab_v071")
    e.add_argument("--seed", type=int, default=17); e.add_argument("--validation-fraction", type=float, default=0.1)
    e.add_argument("--min-examples", type=int, default=200); e.add_argument("--min-token-accuracy", type=float, default=0.65)
    e.add_argument("--min-intent-accuracy", type=float, default=0.90); e.add_argument("--min-unknown-accuracy", type=float, default=0.90)
    e.add_argument("--min-ood-unknown", type=float, default=0.80); e.add_argument("--intent-confidence", type=float, default=0.55)
    e.add_argument("--min-concept-f1", type=float, default=0.60); e.add_argument("--max-care-mae", type=float, default=0.18)
    e.add_argument("--min-mimic-cosine", type=float, default=0.60); e.add_argument("--min-generated-similarity", type=float, default=0.45)
    e.set_defaults(func=evaluate)

    q = sub.add_parser("talk", help="teacher-free grounded communication using an attached v0.7.1 cortex")
    q.add_argument("--state-dir", required=True); q.add_argument("--question", required=True)
    q.add_argument("--semantic-bundle"); q.add_argument("--image"); q.add_argument("--image-size", type=int, default=224)
    q.add_argument("--vision-device", default="auto"); q.add_argument("--intent-confidence", type=float, default=0.55)
    q.add_argument("--show-intent", action="store_true"); q.set_defaults(func=talk)

    ch = sub.add_parser("chat", help="interactive grounded dialogue shell with honest unknown rejection")
    ch.add_argument("--state-dir", required=True); ch.add_argument("--semantic-bundle"); ch.add_argument("--image")
    ch.add_argument("--image-size", type=int, default=224); ch.add_argument("--vision-device", default="auto")
    ch.add_argument("--intent-confidence", type=float, default=0.55); ch.add_argument("--show-intent", action="store_true")
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
