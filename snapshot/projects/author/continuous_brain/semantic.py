"""Trainable semantic/communication cortex for Continuous Brain v0.7.

DINOv3 and Qwen are teachers.  This module is the student that remains attached
at runtime.  Qwen is not imported or called here.  The student learns a compact
visual code, grounded concepts, care/social estimates and short byte-level
responses.  This is an engineering component, not evidence of sentience.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Curated first-language ontology.  The teacher must choose from this list rather
# than continuously inventing labels, so the student has a stable neural target.
BASIC_CONCEPTS = (
    "person", "self", "door", "chair", "table", "ball", "cup", "bottle", "book",
    "lamp", "sofa", "bed", "cabinet", "drawer", "counter", "sink", "fridge", "wall",
    "floor", "ceiling", "window", "plant", "picture", "screen", "box", "shelf", "room",
    "hallway", "opening", "obstacle", "target", "near", "far", "left", "right", "ahead",
    "behind", "above", "below", "inside", "on", "under", "open", "closed", "moving",
    "still", "reachable", "blocked", "visible", "hidden", "red", "green", "blue", "yellow",
    "white", "black", "gray", "brown", "round", "square", "large", "small", "bright",
    "dark", "approach", "stop", "turn", "inspect", "search", "rest", "help", "request",
    "permission", "danger", "safe", "uncertain", "success", "failure", "remember",
    "before", "after", "same", "different", "one", "many",
)

CARE_KEYS = (
    "social_presence",
    "help_needed",
    "harm_risk",
    "respect_request",
    "uncertainty",
    "care_priority",
)


class ByteCodec:
    """Tiny deterministic UTF-8 codec: PAD/BOS/EOS + raw bytes.

    A byte vocabulary avoids importing a second tokenizer and keeps the native
    decoder small enough to train on ordinary GPUs.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    OFFSET = 3
    VOCAB = 259

    @classmethod
    def encode(cls, text: str, max_bytes: int = 256, *, add_bos=True, add_eos=True) -> list[int]:
        raw = text.encode("utf-8", errors="replace")[:max_bytes]
        ids = [cls.OFFSET + b for b in raw]
        if add_bos:
            ids.insert(0, cls.BOS)
        if add_eos:
            ids.append(cls.EOS)
        return ids

    @classmethod
    def decode(cls, ids: Iterable[int]) -> str:
        raw = bytearray()
        for token in ids:
            token = int(token)
            if token == cls.EOS:
                break
            if token >= cls.OFFSET:
                raw.append(token - cls.OFFSET)
        return bytes(raw).decode("utf-8", errors="replace")


@dataclass
class SemanticConfig:
    dino_dim: int = 384
    visual_dim: int = 48
    context_dim: int = 48
    question_hidden: int = 192
    decoder_hidden: int = 384
    byte_embed: int = 64
    adapter_hidden: int = 256
    concept_count: int = len(BASIC_CONCEPTS)
    care_dim: int = len(CARE_KEYS)
    max_question_bytes: int = 192
    max_answer_bytes: int = 256

    def __post_init__(self):
        if min(self.dino_dim, self.visual_dim, self.context_dim, self.question_hidden,
               self.decoder_hidden, self.byte_embed, self.adapter_hidden) < 1:
            raise ValueError("Semantic dimensions must be positive")
        if self.concept_count != len(BASIC_CONCEPTS):
            raise ValueError("v0.7 concept vocabulary is fixed for checkpoint compatibility")
        if self.care_dim != len(CARE_KEYS):
            raise ValueError("v0.7 care target dimensionality is fixed")


class SemanticCortex(nn.Module):
    """Small trainable student around frozen DINO features.

    Inputs available at runtime are deliberately limited to:
      * frozen DINO embedding from the egocentric RGB frame,
      * a 48-value source-labelled internal context vector,
      * the user's question.

    No simulator metadata, hidden Qwen state or teacher answer is accepted at
    inference.  Qwen is therefore removable after training/qualification.
    """

    def __init__(self, cfg: SemanticConfig | None = None):
        super().__init__()
        self.cfg = cfg or SemanticConfig()
        c = self.cfg
        self.visual_adapter = nn.Sequential(
            nn.Linear(c.dino_dim, c.adapter_hidden),
            nn.LayerNorm(c.adapter_hidden),
            nn.SiLU(),
            nn.Linear(c.adapter_hidden, c.visual_dim),
            nn.Tanh(),
        )
        self.byte_embedding = nn.Embedding(ByteCodec.VOCAB, c.byte_embed, padding_idx=ByteCodec.PAD)
        self.question_gru = nn.GRU(c.byte_embed, c.question_hidden, batch_first=True)
        self.visual_context = nn.Sequential(
            nn.Linear(c.visual_dim + c.context_dim, c.decoder_hidden),
            nn.LayerNorm(c.decoder_hidden),
            nn.SiLU(),
        )
        self.question_to_decoder = nn.Linear(c.question_hidden, c.decoder_hidden)
        self.decoder = nn.GRU(c.byte_embed, c.decoder_hidden, batch_first=True)
        self.answer_head = nn.Linear(c.decoder_hidden, ByteCodec.VOCAB)
        self.concept_head = nn.Linear(c.visual_dim, c.concept_count)
        self.care_head = nn.Sequential(
            nn.Linear(c.visual_dim + c.context_dim + c.question_hidden, c.decoder_hidden // 2),
            nn.SiLU(),
            nn.Linear(c.decoder_hidden // 2, c.care_dim),
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def visual_code(self, dino: torch.Tensor) -> torch.Tensor:
        if dino.ndim != 2 or dino.shape[-1] != self.cfg.dino_dim:
            raise ValueError(f"Expected [batch,{self.cfg.dino_dim}] DINO features")
        return self.visual_adapter(dino)

    def encode_question(self, question_ids: torch.Tensor) -> torch.Tensor:
        emb = self.byte_embedding(question_ids)
        _, h = self.question_gru(emb)
        return h[-1]

    def condition(self, visual_code: torch.Tensor, context: torch.Tensor,
                  question_state: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.cfg.context_dim:
            raise ValueError(f"Expected [batch,{self.cfg.context_dim}] communication context")
        base = self.visual_context(torch.cat((visual_code, context), dim=-1))
        return torch.tanh(base + self.question_to_decoder(question_state))

    def forward(self, dino: torch.Tensor, context: torch.Tensor, question_ids: torch.Tensor,
                answer_input_ids: torch.Tensor):
        visual = self.visual_code(dino)
        question = self.encode_question(question_ids)
        condition = self.condition(visual, context, question)
        decoder_in = self.byte_embedding(answer_input_ids)
        decoded, _ = self.decoder(decoder_in, condition.unsqueeze(0))
        return {
            "answer_logits": self.answer_head(decoded),
            "concept_logits": self.concept_head(visual),
            "care_logits": self.care_head(torch.cat((visual, context, question), dim=-1)),
            "visual_code": visual,
        }

    @torch.inference_mode()
    def generate(self, dino: torch.Tensor, context: torch.Tensor, question_ids: torch.Tensor,
                 max_new_bytes: int | None = None) -> str:
        self.eval()
        visual = self.visual_code(dino)
        question = self.encode_question(question_ids)
        hidden = self.condition(visual, context, question).unsqueeze(0)
        token = torch.full((dino.shape[0], 1), ByteCodec.BOS, dtype=torch.long, device=dino.device)
        out: list[int] = []
        limit = max_new_bytes or self.cfg.max_answer_bytes
        for _ in range(limit + 1):
            emb = self.byte_embedding(token[:, -1:])
            decoded, hidden = self.decoder(emb, hidden)
            next_token = self.answer_head(decoded[:, -1]).argmax(dim=-1)
            value = int(next_token[0].item())
            if value == ByteCodec.EOS:
                break
            out.append(value)
            token = next_token[:, None]
        return ByteCodec.decode(out)


def pad_ids(sequences: list[list[int]], pad=ByteCodec.PAD) -> torch.Tensor:
    width = max(len(s) for s in sequences)
    out = torch.full((len(sequences), width), pad, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def semantic_losses(model: SemanticCortex, batch: dict[str, torch.Tensor], *,
                    answer_weight=1.0, concept_weight=0.25, care_weight=0.10,
                    mimic_weight=0.20) -> tuple[torch.Tensor, dict[str, float]]:
    answer = batch["answer_ids"]
    result = model(batch["dino"], batch["context"], batch["question_ids"], answer[:, :-1])
    targets = answer[:, 1:]
    answer_loss = F.cross_entropy(result["answer_logits"].reshape(-1, ByteCodec.VOCAB),
                                  targets.reshape(-1), ignore_index=ByteCodec.PAD)
    concept_loss = F.binary_cross_entropy_with_logits(result["concept_logits"], batch["concepts"])
    care_loss = F.mse_loss(torch.sigmoid(result["care_logits"]), batch["care"])
    mimic_loss = 1 - F.cosine_similarity(result["visual_code"], batch["pixel_target"], dim=-1).mean()
    total = (answer_weight * answer_loss + concept_weight * concept_loss +
             care_weight * care_loss + mimic_weight * mimic_loss)
    metrics = {
        "loss": float(total.detach()),
        "answer": float(answer_loss.detach()),
        "concept": float(concept_loss.detach()),
        "care": float(care_loss.detach()),
        "mimic": float(mimic_loss.detach()),
    }
    return total, metrics


def model_digest(state_dict: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        h.update(key.encode())
        if torch.is_tensor(value):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def save_semantic_bundle(path: str | Path, model: SemanticCortex, *, metadata: dict) -> dict:
    path = Path(path)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    digest = model_digest(state)
    bundle = {
        "format": "continuous_brain.semantic.v070",
        "config": asdict(model.cfg),
        "concepts": list(BASIC_CONCEPTS),
        "care_keys": list(CARE_KEYS),
        "model": state,
        "digest": digest,
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, tmp)
    tmp.replace(path)
    return bundle


def load_semantic_bundle(path: str | Path, *, device="cpu", require_qualified=False):
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    if bundle.get("format") != "continuous_brain.semantic.v070":
        raise ValueError("Unsupported semantic-cortex bundle")
    if tuple(bundle.get("concepts", ())) != BASIC_CONCEPTS or tuple(bundle.get("care_keys", ())) != CARE_KEYS:
        raise ValueError("Semantic vocabulary does not match v0.7")
    if model_digest(bundle["model"]) != bundle.get("digest"):
        raise ValueError("Semantic-cortex digest mismatch")
    if require_qualified and not bundle.get("metadata", {}).get("qualified", False):
        raise ValueError("Semantic cortex has not passed qualification")
    cfg = SemanticConfig(**bundle["config"])
    model = SemanticCortex(cfg).to(device)
    model.load_state_dict(bundle["model"], strict=True)
    return model, bundle
