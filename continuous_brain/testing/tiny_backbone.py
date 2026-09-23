"""A small, real causal transformer implementing `Backbone`, for tests only.

It lets the whole wake/sleep pipeline run end to end on a CPU with no model
download. It is not a substitute for the real backbone.
"""

from __future__ import annotations

import string

import torch
from torch import nn

from ..bridge.backbone import Backbone, BackboneOutput


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int, heads: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, std=0.02)   # GPT-style init: near-uniform predictions
        nn.init.normal_(self.pos.weight, std=0.02)
        self.max_len = max_len

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor):
        length = embeds.shape[1]
        if length > self.max_len:
            raise ValueError(f"sequence of {length} exceeds max_len {self.max_len}")
        x = embeds + self.pos(torch.arange(length, device=embeds.device))[None]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=embeds.device), 1)
        h = self.norm(self.blocks(x, mask=causal, src_key_padding_mask=mask == 0))
        return self.head(h), h


class TinyBackbone(Backbone):
    CHARS = string.printable

    def __init__(self, hidden: int = 64, layers: int = 2, heads: int = 4, max_len: int = 512,
                 seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.eos_token_id = len(self.CHARS)
        self.vocab = len(self.CHARS) + 1
        self.index = {c: i for i, c in enumerate(self.CHARS)}
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.hidden_size = hidden
        self.model = _TinyCausalLM(self.vocab, hidden, layers, heads, max_len).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.index[c] for c in text if c in self.index], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.CHARS[i] for i in ids.tolist() if i < len(self.CHARS))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.tok(ids.to(self.device))

    def forward_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput:
        logits, hidden = self.model(embeds, attention_mask)
        return BackboneOutput(logits, hidden)

    def embedding_rms(self) -> float:
        w = self.model.tok.weight
        return float(torch.sqrt((w * w).mean(dim=1)).mean())
