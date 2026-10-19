"""Ordinary-language competence: mean per-token loss (and perplexity) on fixed public-domain sentences."""
from __future__ import annotations

import math
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parents[1] / "data"


def load_sentences(name: str) -> list[str]:
    return [s for s in (DATA / f"competence_{name}.txt").read_text().splitlines() if s.strip()]


@torch.no_grad()
def mean_loss(model, tok, sentences: list[str]) -> float:
    total, n = 0.0, 0
    for s in sentences:
        ids = tok(s, return_tensors="pt").input_ids[:, :64]
        out = model(ids, labels=ids)
        k = ids.shape[1] - 1
        total += out.loss.item() * k
        n += k
    return total / n


def perplexity(model, tok, sentences: list[str]) -> float:
    return math.exp(mean_loss(model, tok, sentences))
