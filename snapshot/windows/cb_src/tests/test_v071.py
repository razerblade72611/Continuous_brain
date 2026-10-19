from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import torch

from continuous_brain.dialogue import (
    DIALOGUE_INTENTS,
    UNKNOWN_REPLY,
    augment_teacher_for_dialogue,
    temporal_memory_summary,
)
from continuous_brain.semantic import (
    ByteCodec,
    SemanticConfig,
    SemanticCortex,
    bootstrap_from_v070,
    model_digest,
    semantic_losses,
)
from continuous_brain.semantic_data import collate_examples
from continuous_brain.semantic_cli import _grounded_reply


def tiny_model():
    return SemanticCortex(SemanticConfig(
        dino_dim=8,
        visual_dim=8,
        context_dim=48,
        question_hidden=12,
        decoder_hidden=16,
        byte_embed=8,
        adapter_hidden=12,
    ))


def test_dialogue_curriculum_repairs_state_and_adds_unknown_identity_temporal():
    row = {
        "record_id": "sample-0000012",
        "allowed_text": "current goal=rest; nursery",
        "teacher": {
            "summary": "I see a door.",
            "concepts": ["door"],
            "care": {"respect_request": 1.0},
            "qas": [
                {"kind": "vision", "question": "What do you see?", "answer": "I see a door."},
                {"kind": "state", "question": "What are you trying to do?", "answer": "Open the door"},
            ],
        },
        "temporal": {"has_previous": True, "previous_skill": "navigate", "previous_success": True},
    }
    out = augment_teacher_for_dialogue(row, individual_name="Continuous Brain")
    by_q = {qa["question"]: qa for qa in out["teacher"]["qas"]}
    assert by_q["What are you trying to do?"]["answer"] == "I am trying to rest."
    assert by_q["Who are you?"]["answer"] == "I am Continuous Brain."
    assert by_q["What did you do just before this?"]["answer"] == "I navigated."
    assert by_q["Did your last action succeed?"]["answer"] == "Yes, my last action succeeded."
    assert any(qa["kind"] == "unknown" and qa["answer"] == UNKNOWN_REPLY for qa in out["teacher"]["qas"])


def test_temporal_memory_tail_matches_lifetime_layout():
    vec = temporal_memory_summary("inspect", reward=0.25, success=True)
    assert vec.shape == (8,)
    assert np.isclose(vec[3], 1.0)  # SKILLS index 3 == inspect
    assert np.isclose(vec[6], 0.25)
    assert np.isclose(vec[7], 1.0)


def test_semantic_cortex_trains_intent_head():
    model = tiny_model()
    items = []
    for kind_index, question in [(0, "What do you see?"), (6, "Explain topology.")]:
        items.append({
            "dino": np.zeros(8, np.float32),
            "context": np.zeros(48, np.float32),
            "question_ids": ByteCodec.encode(question),
            "answer_ids": ByteCodec.encode("I don't understand that question yet." if kind_index == 6 else "I see a door."),
            "concepts": np.zeros(model.cfg.concept_count, np.float32),
            "care": np.zeros(model.cfg.care_dim, np.float32),
            "pixel_target": np.zeros(model.cfg.visual_dim, np.float32),
            "intent": kind_index,
        })
    batch = collate_examples(items)
    loss, metrics = semantic_losses(model, batch)
    assert torch.isfinite(loss)
    assert metrics["intent"] > 0
    result = model(batch["dino"], batch["context"], batch["question_ids"], batch["answer_ids"][:, :-1])
    assert result["intent_logits"].shape == (2, len(DIALOGUE_INTENTS))


def test_bootstrap_from_legacy_v070_bundle(tmp_path):
    model = tiny_model()
    old_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("intent_head.")}
    old_cfg = asdict(model.cfg)
    old_cfg.pop("intent_count")
    bundle = {
        "format": "continuous_brain.semantic.v070",
        "config": old_cfg,
        "concepts": list(__import__('continuous_brain.semantic', fromlist=['BASIC_CONCEPTS']).BASIC_CONCEPTS),
        "care_keys": list(__import__('continuous_brain.semantic', fromlist=['CARE_KEYS']).CARE_KEYS),
        "model": old_state,
        "digest": model_digest(old_state),
        "metadata": {"qualified": True},
    }
    path = tmp_path / "legacy.pt"
    torch.save(bundle, path)
    new_model, loaded = bootstrap_from_v070(path, dino_dim=8)
    assert loaded["format"] == "continuous_brain.semantic.v070"
    assert new_model.intent_head.out_features == len(DIALOGUE_INTENTS)
    assert torch.allclose(new_model.visual_adapter[0].weight, model.visual_adapter[0].weight)


def test_grounded_router_rejects_unknown_and_grounds_identity():
    model = tiny_model()
    # Force all questions to unknown.
    with torch.no_grad():
        model.intent_head.weight.zero_()
        model.intent_head.bias.fill_(-5)
        model.intent_head.bias[DIALOGUE_INTENTS.index("unknown")] = 5
    state = {"self_model": {"name": "Continuous Brain"}, "intention": {"goal": "inspect"}}
    dino = torch.zeros(1, 8)
    ctx = torch.zeros(1, 48)
    answer, intent, confidence = _grounded_reply(model, dino, ctx, state, {}, "What is calculus?", 0.55)
    assert answer == UNKNOWN_REPLY
    assert intent == "unknown"
    assert confidence > 0.9

    with torch.no_grad():
        model.intent_head.bias.fill_(-5)
        model.intent_head.bias[DIALOGUE_INTENTS.index("identity")] = 5
    answer, intent, _ = _grounded_reply(model, dino, ctx, state, {}, "Who are you?", 0.55)
    assert answer == "I am Continuous Brain."
    assert intent == "identity"


def test_small_local_teacher_accepts_terse_concept_list(tmp_path, monkeypatch):
    from PIL import Image
    from continuous_brain.teacher import QwenTeacherClient

    image = tmp_path / "frame.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    client = QwenTeacherClient(model="Qwen3-VL-8B-local")
    monkeypatch.setattr(client, "_call", lambda payload: "Door, chair")
    label = client.label(image, [0.0] * 48, "current goal=inspect")
    assert label["concepts"] == ["chair", "door"]
    assert "door" in label["summary"].lower()
    assert "chair" in label["summary"].lower()
