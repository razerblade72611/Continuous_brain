#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    RECORD_FEATURE_DIM,
)

PRIMARY_DIM = 2
DISCREPANCY_DIM = RECORD_FEATURE_DIM - PRIMARY_DIM
ROUTED_DIM = 2 * DISCREPANCY_DIM


class NeutralAutobiographicalSelfModel(nn.Module):
    """
    Autobiographical expected-vs-actual discrepancy model with structural
    neutrality.

    The first two record fields are A/B routing labels. They are NEVER admitted
    as standalone evidence. They only route discrepancy features:

        [primary_A * discrepancy, primary_B * discrepancy]

    Every learned layer is bias-free and zero-preserving.

    Therefore any history whose expected trajectories equal its actual
    trajectories produces exactly:
        reliability = [0.5, 0.5]

    regardless of A/B labels, chronology, or history length.
    """
    def __init__(
        self,
        record_dim: int = 64,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.record_dim = int(record_dim)
        self.hidden_dim = int(hidden_dim)

        self.record_encoder = nn.Sequential(
            nn.Linear(
                ROUTED_DIM,
                record_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                record_dim,
                record_dim,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.history_gru = nn.GRU(
            input_size=record_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bias=False,
        )

        self.evidence_head = nn.Linear(
            hidden_dim,
            2,
            bias=False,
        )

    @staticmethod
    def route_features(history_features: torch.Tensor):
        if history_features.shape[-1] != RECORD_FEATURE_DIM:
            raise ValueError(
                f"expected feature dim {RECORD_FEATURE_DIM}, "
                f"got {history_features.shape[-1]}"
            )

        primary = history_features[..., :2]
        discrepancy = history_features[..., 2:]

        a = discrepancy * primary[..., 0:1]
        b = discrepancy * primary[..., 1:2]
        return torch.cat((a, b), dim=-1)

    def forward(self, history_features, history_mask):
        if history_features.ndim != 3:
            raise ValueError("history_features must be [B,T,F]")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must be [B,T]")

        routed = self.route_features(history_features)
        encoded = self.record_encoder(routed)
        encoded = (
            encoded
            * history_mask.unsqueeze(-1).float()
        )

        lengths = history_mask.sum(dim=1).long()
        reliability = torch.full(
            (history_features.shape[0], 2),
            0.5,
            dtype=history_features.dtype,
            device=history_features.device,
        )

        nonzero = lengths > 0
        if bool(nonzero.any()):
            x = encoded[nonzero]
            l = lengths[nonzero]

            packed_rows = []
            for row, ln in zip(x, l.tolist()):
                valid = row[-ln:]
                pad = torch.zeros_like(row)
                pad[:ln] = valid
                packed_rows.append(pad)
            packed_x = torch.stack(packed_rows)

            packed = nn.utils.rnn.pack_padded_sequence(
                packed_x,
                l.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h = self.history_gru(packed)

            raw = self.evidence_head(h[-1])

            # Exact neutral at raw=0, while retaining nonzero derivative.
            reliability[nonzero] = (
                0.5
                + 0.5 * torch.tanh(raw)
            )

        return {
            "reliability": reliability,
            "preference":
                reliability.argmax(dim=-1),
            "history_count":
                lengths,
        }


def save_neutral_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "format":
                "continuous_brain.autobiographical_self_discrepancy_neutral_v1",
            "record_dim":
                int(model.record_dim),
            "hidden_dim":
                int(model.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v
                in model.state_dict().items()
            },
            "metadata":
                dict(metadata),
        },
        path,
    )


def load_neutral_model(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model = NeutralAutobiographicalSelfModel(
        record_dim=int(
            payload["record_dim"]
        ),
        hidden_dim=int(
            payload["hidden_dim"]
        ),
    ).to(device)
    model.load_state_dict(
        payload["state_dict"]
    )
    model.eval()
    return model, payload
