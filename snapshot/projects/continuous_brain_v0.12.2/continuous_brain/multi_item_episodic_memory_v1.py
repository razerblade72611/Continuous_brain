#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

QUERY_FIRST = 0
QUERY_SECOND = 1
QUERY_SAME = 2
QUERY_DIFFERENT = 3
QUERY_NAMES = ("first", "second", "same", "different")


@dataclass
class TwoSlotMemory:
    """
    Explicit append-only episodic order.

    Content is canonical recall128 from the already-admitted single-item
    memory path.  This sidecar never changes the historical 64-D writer.
    """
    slots: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def empty(cls, batch_size: int, content_dim: int, device="cpu"):
        return cls(
            slots=torch.zeros(
                int(batch_size), 2, int(content_dim), device=device
            ),
            valid=torch.zeros(
                int(batch_size), 2, dtype=torch.bool, device=device
            ),
        )

    def append(self, content: torch.Tensor) -> "TwoSlotMemory":
        if content.ndim != 2:
            raise ValueError("content must be [B,D]")
        if content.shape[0] != self.slots.shape[0]:
            raise ValueError("batch mismatch")
        new_slots = self.slots.clone()
        new_valid = self.valid.clone()
        for b in range(content.shape[0]):
            if not bool(new_valid[b, 0]):
                idx = 0
            elif not bool(new_valid[b, 1]):
                idx = 1
            else:
                raise RuntimeError("TwoSlotMemory capacity exceeded")
            new_slots[b, idx] = content[b]
            new_valid[b, idx] = True
        return TwoSlotMemory(new_slots, new_valid)

    def swap(self) -> "TwoSlotMemory":
        return TwoSlotMemory(
            self.slots[:, [1, 0]].clone(),
            self.valid[:, [1, 0]].clone(),
        )

    def zero_slot(self, slot: int) -> "TwoSlotMemory":
        slot = int(slot)
        out = self.slots.clone()
        valid = self.valid.clone()
        out[:, slot] = 0
        valid[:, slot] = False
        return TwoSlotMemory(out, valid)

    def replace_slot(
        self,
        slot: int,
        donor_content: torch.Tensor,
    ) -> "TwoSlotMemory":
        slot = int(slot)
        out = self.slots.clone()
        valid = self.valid.clone()
        out[:, slot] = donor_content
        valid[:, slot] = True
        return TwoSlotMemory(out, valid)


class EpisodicLateQueryReader(nn.Module):
    """
    Learned late-query reader over two ordered episodic slots.

    Query is supplied only after both events have already been stored.

    The reader has:
      - shared content encoder;
      - learned order embeddings;
      - learned query embeddings;
      - query-conditioned slot attention;
      - relation features over both slots;
      - binary answer head.

    Answer semantics:
      FIRST/SECOND: 0=LEFT, 1=RIGHT
      SAME:         0=no,   1=yes
      DIFFERENT:    0=no,   1=yes
    """
    def __init__(
        self,
        content_dim: int = 128,
        hidden_dim: int = 96,
        query_dim: int = 32,
    ):
        super().__init__()
        self.content_dim = int(content_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_dim = int(query_dim)

        self.content_encoder = nn.Sequential(
            nn.Linear(self.content_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.position_embedding = nn.Embedding(2, self.query_dim)
        self.query_embedding = nn.Embedding(4, self.query_dim)

        self.attention = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.query_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        relation_dim = 4 * self.hidden_dim
        self.answer = nn.Sequential(
            nn.Linear(
                self.hidden_dim + relation_dim + self.query_dim,
                2 * self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_dim, 2),
        )

    def forward(
        self,
        slots: torch.Tensor,
        valid: torch.Tensor,
        query: torch.Tensor,
    ):
        if slots.ndim != 3 or slots.shape[1] != 2:
            raise ValueError("slots must be [B,2,D]")
        if valid.shape != slots.shape[:2]:
            raise ValueError("valid must be [B,2]")

        b = slots.shape[0]
        encoded = self.content_encoder(slots)

        pos_ids = torch.arange(
            2, device=slots.device, dtype=torch.long
        ).unsqueeze(0).expand(b, 2)
        pos = self.position_embedding(pos_ids)

        q = self.query_embedding(query.long())
        q_expanded = q.unsqueeze(1).expand(-1, 2, -1)

        score_in = torch.cat(
            (encoded, pos, q_expanded),
            dim=-1,
        )
        scores = self.attention(score_in).squeeze(-1)
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1)

        selected = (
            weights.unsqueeze(-1) * encoded
        ).sum(dim=1)

        e0 = encoded[:, 0]
        e1 = encoded[:, 1]
        relation = torch.cat(
            (
                e0,
                e1,
                torch.abs(e0 - e1),
                e0 * e1,
            ),
            dim=-1,
        )

        logits = self.answer(
            torch.cat(
                (selected, relation, q),
                dim=-1,
            )
        )
        return {
            "logits": logits,
            "attention": weights,
            "encoded_slots": encoded,
        }


def save_reader(path, reader, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.multi_item_episodic_memory_v1",
            "content_dim": reader.content_dim,
            "hidden_dim": reader.hidden_dim,
            "query_dim": reader.query_dim,
            "state_dict": {
                k: v.detach().cpu()
                for k, v in reader.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_reader(path, device="cpu"):
    payload = torch.load(
        path, map_location="cpu", weights_only=True
    )
    reader = EpisodicLateQueryReader(
        int(payload["content_dim"]),
        int(payload["hidden_dim"]),
        int(payload["query_dim"]),
    ).to(device)
    reader.load_state_dict(payload["state_dict"])
    reader.eval()
    return reader, payload
