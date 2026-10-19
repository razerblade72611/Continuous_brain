"""Frozen pretrained backbone. Its weights never change; the hash is checked before and after."""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from .memory import module_hash

DEFAULT_PATH = os.environ.get("BEING_BACKBONE", "models/distilgpt2")


def load_backbone(path: str = DEFAULT_PATH, smoke: bool = False):
    """Return (model, tokenizer). smoke=True builds a tiny random GPT-2 for pipeline tests only."""
    if smoke:
        tok = AutoTokenizer.from_pretrained(path) if os.path.isdir(path) else None
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=tok.vocab_size if tok else 50257)
        model = GPT2LMHeadModel(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(True)
    return model, tok


def backbone_hash(model) -> str:
    return module_hash(model)
