"""Raw trainable cortical architecture for the v0.9 parented-development line.

No pretrained weights are loaded here.  External foundation models are teachers.
The child's cortices, recurrent nucleus, value system and body world model all
start from ordinary random initialization and remain in the individual checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

BYTE_PAD = 0
BYTE_BOS = 1
BYTE_EOS = 2
BYTE_OFFSET = 3
BYTE_VOCAB = 259


def byte_encode(text: str, max_bytes: int, *, bos=True, eos=True) -> list[int]:
    ids = [BYTE_OFFSET + b for b in text.encode("utf-8", errors="replace")[:max_bytes]]
    if bos:
        ids.insert(0, BYTE_BOS)
    if eos:
        ids.append(BYTE_EOS)
    return ids


def byte_decode(ids) -> str:
    raw = bytearray()
    for item in ids:
        item = int(item)
        if item == BYTE_EOS:
            break
        if item >= BYTE_OFFSET:
            raw.append(item - BYTE_OFFSET)
    return bytes(raw).decode("utf-8", errors="replace")


@dataclass
class InfantConfig:
    profile: str = "proof"
    image_size: int = 96
    context_dim: int = 48
    memory_dim: int = 64
    action_dim: int = 5
    bus_dim: int = 128
    nucleus_dim: int = 256
    language_dim: int = 256
    language_heads: int = 8
    language_layers: int = 4
    language_ff: int = 768
    temporal_dim: int = 192
    world_hidden: int = 256
    world_ensemble: int = 3
    max_question_bytes: int = 192
    max_answer_bytes: int = 256
    care_dim: int = 6

    @classmethod
    def for_profile(cls, profile: str) -> "InfantConfig":
        p = str(profile).lower()
        if p == "proof":
            return cls(profile=p)
        if p == "balanced":
            return cls(
                profile=p, bus_dim=256, nucleus_dim=512, language_dim=512,
                language_heads=8, language_layers=8, language_ff=1536,
                temporal_dim=384, world_hidden=512, world_ensemble=4,
            )
        if p == "scale":
            return cls(
                profile=p, bus_dim=384, nucleus_dim=768, language_dim=768,
                language_heads=12, language_layers=12, language_ff=3072,
                temporal_dim=512, world_hidden=768, world_ensemble=5,
            )
        raise ValueError("profile must be proof, balanced, or scale")


class InfantVisionCortex(nn.Module):
    """Child-owned raw-pixel cortex. Parent embeddings supervise only projection heads."""
    def __init__(self, bus_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.project = nn.Sequential(
            nn.Flatten(), nn.Linear(192 * 4, bus_dim * 2), nn.LayerNorm(bus_dim * 2),
            nn.SiLU(), nn.Linear(bus_dim * 2, bus_dim), nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [B,3,H,W]")
        return self.project(self.features(images))


class InfantAudioCortex(nn.Module):
    """Child-owned waveform/log-mel front end. Whisper remains an external transcript parent."""
    def __init__(self, bus_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4, padding=4), nn.SiLU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3), nn.SiLU(),
            nn.Conv1d(64, 128, 5, stride=4, padding=2), nn.SiLU(),
            nn.Conv1d(128, 192, 5, stride=4, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.project = nn.Sequential(
            nn.Flatten(), nn.Linear(192 * 4, bus_dim * 2), nn.LayerNorm(bus_dim * 2),
            nn.SiLU(), nn.Linear(bus_dim * 2, bus_dim), nn.Tanh(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must be [B,T] or [B,1,T]")
        return self.project(self.features(audio))


class CorticalNucleus(nn.Module):
    """Persistent recurrent global workspace and router."""
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(cfg.context_dim, cfg.bus_dim), nn.SiLU())
        self.memory = nn.Sequential(nn.Linear(cfg.memory_dim, cfg.bus_dim), nn.SiLU())
        self.action = nn.Embedding(cfg.action_dim + 1, cfg.bus_dim)
        self.core = nn.GRUCell(cfg.bus_dim * 4, cfg.nucleus_dim)
        self.workspace = nn.Sequential(
            nn.Linear(cfg.nucleus_dim, cfg.bus_dim), nn.LayerNorm(cfg.bus_dim), nn.Tanh()
        )
        self.router = nn.Linear(cfg.nucleus_dim, 6)
        self.metacognition = nn.Sequential(nn.Linear(cfg.nucleus_dim, 64), nn.SiLU(), nn.Linear(64, 3))
        self.cfg = cfg

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.nucleus_dim, device=device)

    def forward(self, vision: torch.Tensor, context: torch.Tensor, memory: torch.Tensor,
                prev_action: torch.Tensor, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat((vision, self.context(context), self.memory(memory), self.action(prev_action)), -1)
        hidden = self.core(x, hidden)
        return {
            "hidden": hidden,
            "workspace": self.workspace(hidden),
            "router": torch.softmax(self.router(hidden), dim=-1),
            "metacognition": torch.sigmoid(self.metacognition(hidden)),
        }


class InfantLanguageCortex(nn.Module):
    """Randomly initialized byte-level transformer conditioned by the child's workspace."""
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        d = cfg.language_dim
        self.cfg = cfg
        self.embed = nn.Embedding(BYTE_VOCAB, d, padding_idx=BYTE_PAD)
        self.pos = nn.Embedding(max(cfg.max_question_bytes, cfg.max_answer_bytes) + 4, d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg.language_heads, cfg.language_ff,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.question_encoder = nn.TransformerEncoder(enc_layer, cfg.language_layers // 2 or 1)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.language_heads, cfg.language_ff,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.language_layers)
        self.workspace = nn.Linear(cfg.bus_dim, d)
        self.final_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, BYTE_VOCAB, bias=False)
        self.out.weight = self.embed.weight

    def _positions(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)

    def encode_question(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) + self.pos(self._positions(ids))
        pad = ids.eq(BYTE_PAD)
        return self.question_encoder(x, src_key_padding_mask=pad)

    def forward(self, workspace: torch.Tensor, question_ids: torch.Tensor,
                answer_input: torch.Tensor) -> torch.Tensor:
        q = self.encode_question(question_ids)
        w = self.workspace(workspace).unsqueeze(1)
        memory = torch.cat((w, q), dim=1)
        y = self.embed(answer_input) + self.pos(self._positions(answer_input))
        n = answer_input.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=answer_input.device), diagonal=1)
        decoded = self.decoder(y, memory, tgt_mask=mask)
        return self.out(self.final_norm(decoded)) / math.sqrt(float(self.cfg.language_dim))

    @torch.inference_mode()
    def generate(self, workspace: torch.Tensor, question_ids: torch.Tensor, max_new=160) -> str:
        self.eval()
        seq = torch.full((workspace.shape[0], 1), BYTE_BOS, dtype=torch.long, device=workspace.device)
        for _ in range(min(max_new, self.cfg.max_answer_bytes)):
            logits = self(workspace, question_ids, seq)
            token = logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat((seq, token), dim=1)
            if int(token[0, 0]) == BYTE_EOS:
                break
        return byte_decode(seq[0, 1:].tolist())


class TemporalCortex(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.cell = nn.GRUCell(cfg.bus_dim + cfg.action_dim, cfg.temporal_dim)
        self.to_bus = nn.Linear(cfg.temporal_dim, cfg.bus_dim)
        self.cfg = cfg

    def forward(self, vision: torch.Tensor, action_onehot: torch.Tensor,
                state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.cell(torch.cat((vision, action_onehot), -1), state)
        return torch.tanh(self.to_bus(state)), state


class WorldMember(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.bus_dim + cfg.action_dim, cfg.world_hidden), nn.SiLU(),
            nn.Linear(cfg.world_hidden, cfg.world_hidden), nn.SiLU(),
        )
        self.next_bus = nn.Linear(cfg.world_hidden, cfg.bus_dim)
        self.reward = nn.Linear(cfg.world_hidden, 1)
        self.success = nn.Linear(cfg.world_hidden, 1)

    def forward(self, workspace, action_onehot):
        h = self.net(torch.cat((workspace, action_onehot), -1))
        return self.next_bus(h), self.reward(h).squeeze(-1), self.success(h).squeeze(-1)


class WorldModelEnsemble(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.members = nn.ModuleList([WorldMember(cfg) for _ in range(cfg.world_ensemble)])

    def forward(self, workspace, action_onehot):
        predictions = [m(workspace, action_onehot) for m in self.members]
        buses = torch.stack([x[0] for x in predictions])
        rewards = torch.stack([x[1] for x in predictions])
        successes = torch.stack([x[2] for x in predictions])
        return {
            "next_bus": buses.mean(0),
            "reward": rewards.mean(0),
            "success_logit": successes.mean(0),
            "uncertainty": buses.var(0, unbiased=False).mean(-1),
            "member_bus": buses,
            "member_reward": rewards,
            "member_success_logit": successes,
        }


class InfantValueCortex(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.bus_dim, cfg.bus_dim), nn.SiLU(), nn.Linear(cfg.bus_dim, cfg.care_dim))

    def forward(self, workspace):
        return torch.sigmoid(self.net(workspace))


class InfantMotorCortex(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(cfg.bus_dim, cfg.bus_dim), nn.SiLU(), nn.Linear(cfg.bus_dim, cfg.action_dim))

    def forward(self, workspace):
        return self.policy(workspace)


class InfantBrain(nn.Module):
    """The child-owned neural system. Parent heads are detachable distillation scaffolds."""
    def __init__(self, cfg: InfantConfig | None = None, *, vision_parent_dims: Mapping[str, int] | None = None):
        super().__init__()
        self.cfg = cfg or InfantConfig()
        self.vision = InfantVisionCortex(self.cfg.bus_dim)
        self.audio = InfantAudioCortex(self.cfg.bus_dim)
        self.nucleus = CorticalNucleus(self.cfg)
        self.language = InfantLanguageCortex(self.cfg)
        self.temporal = TemporalCortex(self.cfg)
        self.world = WorldModelEnsemble(self.cfg)
        self.values = InfantValueCortex(self.cfg)
        self.motor = InfantMotorCortex(self.cfg)
        self.vision_parent_heads = nn.ModuleDict()
        for name, dim in (vision_parent_dims or {}).items():
            self.add_vision_parent_head(name, int(dim))

    def add_vision_parent_head(self, parent_id: str, dim: int) -> None:
        key = parent_id.replace(".", "__").replace("/", "_")
        if key not in self.vision_parent_heads:
            self.vision_parent_heads[key] = nn.Linear(self.cfg.bus_dim, int(dim))

    def parent_visual_prediction(self, parent_id: str, visual_bus: torch.Tensor) -> torch.Tensor:
        key = parent_id.replace(".", "__").replace("/", "_")
        if key not in self.vision_parent_heads:
            raise KeyError(f"No vision parent head for {parent_id}")
        return self.vision_parent_heads[key](visual_bus)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint_payload(self, metadata=None) -> dict:
        return {
            "format": "continuous_brain.infant.v090",
            "config": asdict(self.cfg),
            "vision_parent_dims": {
                k: v.out_features for k, v in self.vision_parent_heads.items()
            },
            "model": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "metadata": dict(metadata or {}),
        }


def one_hot_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.one_hot(action.long().clamp(0, action_dim - 1), num_classes=action_dim).float()
