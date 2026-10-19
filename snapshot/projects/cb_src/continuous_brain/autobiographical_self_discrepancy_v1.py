#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import nn


RECORD_FEATURE_DIM = 27
HISTORY_LEN = 12


@dataclass
class TrajectoryMemoryRecord:
    episode_id: int
    primary: int  # 1=A, 2=B
    expected_self: torch.Tensor   # [T,6]
    actual_self: torch.Tensor     # [T,6]
    expected_world: torch.Tensor  # [T,4]
    actual_world: torch.Tensor    # [T,4]
    expected_modes: torch.Tensor  # [T]
    actual_modes: torch.Tensor    # [T]

    def feature(self) -> torch.Tensor:
        es = self.expected_self.float()
        ac = self.actual_self.float()
        ew = self.expected_world.float()
        aw = self.actual_world.float()

        if es.shape != ac.shape or es.ndim != 2 or es.shape[-1] != 6:
            raise ValueError("self trajectories must both be [T,6]")
        if ew.shape != aw.shape or ew.ndim != 2 or ew.shape[-1] != 4:
            raise ValueError("world trajectories must both be [T,4]")

        abs_self = (ac - es).abs()
        signed_self = ac - es
        abs_world = (aw - ew).abs()

        primary_oh = torch.zeros(2, dtype=torch.float32)
        if int(self.primary) == 1:
            primary_oh[0] = 1.0
        elif int(self.primary) == 2:
            primary_oh[1] = 1.0
        else:
            raise ValueError("primary must be A(1) or B(2)")

        mode_mismatch = (
            self.expected_modes.long()
            != self.actual_modes.long()
        ).float().mean().view(1)

        # 2 + 6 + 6 + 6 + 4 + 1 + 2 = 27
        # final two fields summarize trajectory scale and terminal discrepancy.
        feat = torch.cat(
            (
                primary_oh,
                abs_self.mean(dim=0),
                abs_self[-1],
                signed_self.mean(dim=0),
                abs_world.mean(dim=0),
                mode_mismatch,
                torch.tensor(
                    [
                        float(abs_self.mean().item()),
                        float(abs_self[-1].mean().item()),
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        if feat.numel() != RECORD_FEATURE_DIM:
            raise AssertionError(feat.numel())
        return feat


class AutobiographicalTrajectoryMemory:
    """
    Append-only expected-vs-actual personal trajectory history.
    """
    def __init__(self, records: List[TrajectoryMemoryRecord] | None = None):
        self.records = list(records or [])

    def append(self, record: TrajectoryMemoryRecord):
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def feature_tensor(self, history_len: int = HISTORY_LEN):
        records = self.records[-int(history_len):]
        x = torch.zeros(
            int(history_len),
            RECORD_FEATURE_DIM,
            dtype=torch.float32,
        )
        mask = torch.zeros(
            int(history_len),
            dtype=torch.bool,
        )
        start = int(history_len) - len(records)
        for j, record in enumerate(records):
            x[start + j] = record.feature()
            mask[start + j] = True
        return x, mask

    def discrepancy_ablation(self):
        """
        Preserve episode identity/primary/order while making expected==actual.
        This removes autobiographical self-discrepancy information.
        """
        records = []
        for r in self.records:
            records.append(
                TrajectoryMemoryRecord(
                    episode_id=r.episode_id,
                    primary=r.primary,
                    expected_self=r.actual_self.clone(),
                    actual_self=r.actual_self.clone(),
                    expected_world=r.actual_world.clone(),
                    actual_world=r.actual_world.clone(),
                    expected_modes=r.actual_modes.clone(),
                    actual_modes=r.actual_modes.clone(),
                )
            )
        return AutobiographicalTrajectoryMemory(records)


class AutobiographicalSelfModel(nn.Module):
    """
    Learns a self-reliability estimate from chronological autobiographical
    expected-vs-actual trajectory discrepancies.

    Inputs contain NO identity vector, current world, or current self. This
    forces the estimate to come from autobiographical discrepancy history.
    """
    def __init__(
        self,
        feature_dim: int = RECORD_FEATURE_DIM,
        record_dim: int = 64,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.record_dim = int(record_dim)
        self.hidden_dim = int(hidden_dim)

        self.record_encoder = nn.Sequential(
            nn.Linear(feature_dim, record_dim),
            nn.SiLU(),
            nn.Linear(record_dim, record_dim),
            nn.Tanh(),
        )
        self.history_gru = nn.GRU(
            input_size=record_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, history_features, history_mask):
        if history_features.ndim != 3:
            raise ValueError("history_features must be [B,T,F]")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must be [B,T]")

        encoded = self.record_encoder(history_features)
        encoded = encoded * history_mask.unsqueeze(-1).float()

        lengths = history_mask.sum(dim=1).long()

        # For zero-history rows, return exact neutral reliability 0.5.
        nonzero = lengths > 0
        reliability = torch.full(
            (history_features.shape[0], 2),
            0.5,
            dtype=history_features.dtype,
            device=history_features.device,
        )

        if bool(nonzero.any()):
            x = encoded[nonzero]
            l = lengths[nonzero]

            # Histories are right-aligned. Shift valid records to the front for
            # packed recurrent processing.
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
            raw = self.reliability_head(h[-1])
            reliability[nonzero] = torch.sigmoid(raw)

        return {
            "reliability": reliability,
            "preference": reliability.argmax(dim=-1),  # 0=A, 1=B
            "history_count": lengths,
        }


def save_autobiographical_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.autobiographical_self_discrepancy_v1",
            "feature_dim": int(model.feature_dim),
            "record_dim": int(model.record_dim),
            "hidden_dim": int(model.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_autobiographical_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = AutobiographicalSelfModel(
        feature_dim=int(payload["feature_dim"]),
        record_dim=int(payload["record_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def save_memory(path, memory: AutobiographicalTrajectoryMemory):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = []
    for r in memory.records:
        serial.append(
            {
                "episode_id": int(r.episode_id),
                "primary": int(r.primary),
                "expected_self": r.expected_self.detach().cpu(),
                "actual_self": r.actual_self.detach().cpu(),
                "expected_world": r.expected_world.detach().cpu(),
                "actual_world": r.actual_world.detach().cpu(),
                "expected_modes": r.expected_modes.detach().cpu(),
                "actual_modes": r.actual_modes.detach().cpu(),
            }
        )
    torch.save(
        {
            "format": "continuous_brain.autobiographical_trajectory_memory_v1",
            "records": serial,
        },
        path,
    )


def load_memory(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    records = []
    for r in payload["records"]:
        records.append(
            TrajectoryMemoryRecord(
                episode_id=int(r["episode_id"]),
                primary=int(r["primary"]),
                expected_self=r["expected_self"].float(),
                actual_self=r["actual_self"].float(),
                expected_world=r["expected_world"].float(),
                actual_world=r["actual_world"].float(),
                expected_modes=r["expected_modes"].long(),
                actual_modes=r["actual_modes"].long(),
            )
        )
    return AutobiographicalTrajectoryMemory(records), payload
