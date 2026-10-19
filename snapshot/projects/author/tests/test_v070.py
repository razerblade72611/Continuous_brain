import json

import numpy as np
import pytest
import torch

from continuous_brain.cognition import CognitiveConfig
from continuous_brain.development import DevelopmentRoom
from continuous_brain.individual import IndividualRuntime
from continuous_brain.semantic import (BASIC_CONCEPTS, CARE_KEYS, ByteCodec, SemanticConfig,
                                       SemanticCortex, load_semantic_bundle, save_semantic_bundle)
from continuous_brain.semantic_data import make_context, neutral_self_vector
from continuous_brain.semantic_upgrade import attach_semantic_cortex
from continuous_brain.teacher import validate_teacher_label
from continuous_brain.vision import PixelStatsBackbone


def runtime(path):
    vision = PixelStatsBackbone()
    cfg = CognitiveConfig.profile("tiny", 6 + vision.output_dim)
    return IndividualRuntime(DevelopmentRoom(seed=7), path, cfg, vision, device="cpu", seed=7)


def tiny_semantic():
    cfg = SemanticConfig(dino_dim=8, adapter_hidden=16, question_hidden=16,
                         decoder_hidden=32, byte_embed=8)
    return SemanticCortex(cfg)


def test_byte_codec_and_student_shapes():
    text = "I see a door."
    assert ByteCodec.decode(ByteCodec.encode(text)) == text
    model = tiny_semantic()
    question = torch.tensor([ByteCodec.encode("What do you see?")], dtype=torch.long)
    answer = torch.tensor([ByteCodec.encode("A door.")], dtype=torch.long)
    out = model(torch.randn(1, 8), torch.zeros(1, 48), question, answer[:, :-1])
    assert out["answer_logits"].shape[:2] == (1, answer.shape[1] - 1)
    assert out["concept_logits"].shape == (1, len(BASIC_CONCEPTS))
    assert out["care_logits"].shape == (1, len(CARE_KEYS))
    assert model.parameter_count < 1_000_000


def test_context_contract_is_48_and_finite():
    context = make_context(np.zeros(6, np.float32), self_vector=neutral_self_vector(), goal=1,
                           last_skill=2, memory_summary=np.arange(8, dtype=np.float32) / 8)
    assert context.shape == (48,)
    assert np.isfinite(context).all()
    with pytest.raises(ValueError):
        make_context(np.zeros(7, np.float32))


def test_teacher_label_is_restricted_and_care_bounded():
    label = validate_teacher_label({
        "summary": "A door is ahead.",
        "concepts": ["door", "ahead", "visible"],
        "qas": [{"kind": "vision", "question": "What do you see?", "answer": "A door."}],
        "care": {"respect_request": 4, "harm_risk": -2},
    })
    assert label["care"]["respect_request"] == 1
    assert label["care"]["harm_risk"] == 0
    bad = dict(label); bad["concepts"] = ["not-in-vocabulary"]
    with pytest.raises(ValueError):
        validate_teacher_label(bad)


def test_qualified_semantic_bundle_digest_and_attachment_preserve_identity(tmp_path):
    source = tmp_path / "source"
    r = runtime(source)
    individual_id = r.self_model.individual_id
    r.run(8, realtime=False, log_every=10000)
    decisions = r.self_model.decisions
    memories = r.store.count("experience")
    r.close()

    lab = tmp_path / "semantic_lab"
    model = tiny_semantic()
    save_semantic_bundle(lab / "qualified_semantic.pt", model, metadata={
        "qualified": True,
        "qualification": {"passed": True, "metrics": {"generated_similarity": 0.7}},
        "dino_model": "test/dino",
        "teacher_models": ["test/qwen"],
        "training_updates": 123,
        "dataset_digest": "abc",
    })
    loaded, bundle = load_semantic_bundle(lab / "qualified_semantic.pt", require_qualified=True)
    assert bundle["digest"]
    assert loaded.parameter_count == model.parameter_count

    destination = tmp_path / "continued"
    report = attach_semantic_cortex(source, destination, lab)
    assert report["individual_id"] == individual_id
    assert report["decisions"] == decisions
    assert report["memories"] == memories
    assert report["runtime_qwen_required"] is False
    assert (destination / "semantic_cortex/qualified_semantic.pt").is_file()

    resumed = runtime(destination)
    try:
        status = resumed.status()
        assert resumed.self_model.individual_id == individual_id
        assert status["version"] == "0.7.0"
        assert status["communication_ready"] is True
        assert status["semantic_cortex"]["dino_model"] == "test/dino"
    finally:
        resumed.close()


def test_attachment_refuses_unqualified(tmp_path):
    source = tmp_path / "source"
    r = runtime(source); r.close()
    lab = tmp_path / "semantic_lab"
    model = tiny_semantic()
    save_semantic_bundle(lab / "qualified_semantic.pt", model, metadata={
        "qualified": False, "qualification": {"passed": False}, "dino_model": "test/dino"
    })
    with pytest.raises(ValueError, match="not passed qualification"):
        attach_semantic_cortex(source, tmp_path / "continued", lab)
