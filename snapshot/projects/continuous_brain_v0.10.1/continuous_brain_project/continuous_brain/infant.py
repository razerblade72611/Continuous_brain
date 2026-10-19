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
    dynamics_context_dim: int = 7
    world_ema_tau: float = 0.995
    planning_dim: int = 192

    @classmethod
    def for_profile(cls, profile: str) -> "InfantConfig":
        p = str(profile).lower()
        if p == "proof":
            return cls(profile=p)
        if p == "balanced":
            return cls(
                profile=p, bus_dim=256, nucleus_dim=512, language_dim=512,
                language_heads=8, language_layers=8, language_ff=1536,
                temporal_dim=384, world_hidden=512, world_ensemble=4, planning_dim=384,
            )
        if p == "scale":
            return cls(
                profile=p, bus_dim=384, nucleus_dim=768, language_dim=768,
                language_heads=12, language_layers=12, language_ff=3072,
                temporal_dim=512, world_hidden=768, world_ensemble=5, planning_dim=512,
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


class InfantSpatialCortex(nn.Module):
    """Child-owned dorsal visual stream for geometry, motion and action.

    Unlike the ventral semantic stream, this cortex preserves a denser spatial grid
    before projection.  It is grounded by structured state/depth/temporal parents
    rather than forced to inherit global semantic invariances from DINO pooling.
    """
    def __init__(self, bus_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.project = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 16, bus_dim * 2), nn.LayerNorm(bus_dim * 2),
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
    """Action-conditioned residual dynamics member.

    v0.9.5 predicts both a slow visual target-bus delta and a structured
    body/world-state delta.  Both residual heads start at zero, so an untrained
    model is exactly the honest no-change baseline.
    """
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.bus_dim + cfg.action_dim, cfg.world_hidden), nn.SiLU(),
            nn.Linear(cfg.world_hidden, cfg.world_hidden), nn.SiLU(),
        )
        self.next_bus = nn.Linear(cfg.world_hidden, cfg.bus_dim)
        self.next_context = nn.Linear(cfg.world_hidden, cfg.dynamics_context_dim)
        # v0.9.5: direct state-conditioning adapters. They start at exactly zero so
        # loading a v0.9.4 checkpoint preserves its existing predictions bit-for-bit
        # until learning makes use of the current dorsal/context state explicitly.
        self.state_bus = nn.Linear(cfg.bus_dim, cfg.world_hidden, bias=False)
        self.state_context = nn.Linear(cfg.dynamics_context_dim, cfg.world_hidden, bias=False)
        nn.init.zeros_(self.next_bus.weight)
        nn.init.zeros_(self.next_bus.bias)
        nn.init.zeros_(self.next_context.weight)
        nn.init.zeros_(self.next_context.bias)
        nn.init.zeros_(self.state_bus.weight)
        nn.init.zeros_(self.state_context.weight)
        self.reward = nn.Linear(cfg.world_hidden, 1)
        self.success = nn.Linear(cfg.world_hidden, 1)

    def forward(self, workspace, action_onehot, *, base_bus=None, base_context=None):
        h = self.net(torch.cat((workspace, action_onehot), -1))
        if base_bus is not None:
            h = h + self.state_bus(base_bus)
        if base_context is not None:
            h = h + self.state_context(base_context[..., :self.cfg.dynamics_context_dim])
        return (
            self.next_bus(h), self.next_context(h),
            self.reward(h).squeeze(-1), self.success(h).squeeze(-1),
        )


class WorldModelEnsemble(nn.Module):
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.cfg = cfg
        self.members = nn.ModuleList([WorldMember(cfg) for _ in range(cfg.world_ensemble)])

    def reset_residual_heads(self) -> None:
        for member in self.members:
            nn.init.zeros_(member.next_bus.weight)
            nn.init.zeros_(member.next_bus.bias)
            nn.init.zeros_(member.next_context.weight)
            nn.init.zeros_(member.next_context.bias)

    def forward(self, workspace, action_onehot, *, base_bus=None, base_context=None):
        predictions = [
            m(workspace, action_onehot, base_bus=base_bus, base_context=base_context)
            for m in self.members
        ]
        bus_deltas = torch.stack([x[0] for x in predictions])
        context_deltas = torch.stack([x[1] for x in predictions])
        if base_bus is None:
            base_bus = torch.zeros_like(bus_deltas[0])
        if base_context is None:
            base_context = torch.zeros_like(context_deltas[0])
        base_context = base_context[..., :self.cfg.dynamics_context_dim]
        buses = base_bus.unsqueeze(0) + bus_deltas
        contexts = base_context.unsqueeze(0) + context_deltas
        rewards = torch.stack([x[2] for x in predictions])
        successes = torch.stack([x[3] for x in predictions])
        return {
            "next_bus": buses.mean(0),
            "delta_bus": bus_deltas.mean(0),
            "next_context": contexts.mean(0),
            "delta_context": context_deltas.mean(0),
            "reward": rewards.mean(0),
            "success_logit": successes.mean(0),
            "uncertainty": (
                bus_deltas.var(0, unbiased=False).mean(-1)
                + context_deltas.var(0, unbiased=False).mean(-1)
            ) * 0.5,
            "member_bus": buses,
            "member_delta_bus": bus_deltas,
            "member_context": contexts,
            "member_delta_context": context_deltas,
            "member_reward": rewards,
            "member_success_logit": successes,
        }


class PlanningCortex(nn.Module):
    """Dedicated recurrent imagination state introduced in v0.10.0.

    This organ is intentionally separate from the lived CorticalNucleus.  Real
    perception initializes a compact planning latent from the child-owned dorsal
    bus plus structured physical context.  Hypothetical actions then advance this
    latent recurrently without pretending that a new real image has been seen.
    """
    def __init__(self, cfg: InfantConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.planning_dim)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.bus_dim + cfg.dynamics_context_dim, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.Tanh(),
        )
        self.transition = nn.GRUCell(cfg.action_dim, d)
        self.spatial_decoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, cfg.bus_dim), nn.Tanh())
        self.context_decoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, cfg.dynamics_context_dim))
        self.reward_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.success_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.collision_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.goal_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.q_head = nn.Sequential(
            nn.Linear(d + cfg.action_dim, d), nn.SiLU(), nn.Linear(d, 1)
        )

    def encode_state(self, spatial_bus: torch.Tensor, context7: torch.Tensor) -> torch.Tensor:
        context7 = context7[..., :self.cfg.dynamics_context_dim]
        return self.encoder(torch.cat((spatial_bus, context7), dim=-1))

    def decode_state(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "spatial": self.spatial_decoder(latent),
            "context": self.context_decoder(latent),
        }

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.value_head(latent).squeeze(-1)

    def goal_value(self, latent: torch.Tensor) -> torch.Tensor:
        # Training target is in [0,2], derived from lived time-to-success.
        return 2.0 * torch.sigmoid(self.goal_head(latent).squeeze(-1))

    def action_value(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.q_head(torch.cat((latent, action_onehot), dim=-1)).squeeze(-1)

    def step(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> dict[str, torch.Tensor]:
        next_latent = self.transition(action_onehot, latent)
        decoded = self.decode_state(next_latent)
        return {
            "latent": next_latent,
            "spatial": decoded["spatial"],
            "context": decoded["context"],
            "reward": self.reward_head(next_latent).squeeze(-1),
            "success_logit": self.success_head(next_latent).squeeze(-1),
            "collision_logit": self.collision_head(next_latent).squeeze(-1),
            "value": self.value(next_latent),
            "goal_value": self.goal_value(next_latent),
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
        # Ventral semantic stream: what is present?
        self.vision = InfantVisionCortex(self.cfg.bus_dim)
        self.target_vision = InfantVisionCortex(self.cfg.bus_dim)
        for p in self.target_vision.parameters():
            p.requires_grad_(False)

        # Dorsal spatial stream: where is it and how can action change it?
        self.spatial = InfantSpatialCortex(self.cfg.bus_dim)
        self.target_spatial = InfantSpatialCortex(self.cfg.bus_dim)
        for p in self.target_spatial.parameters():
            p.requires_grad_(False)
        self.spatial_state_head = nn.Sequential(
            nn.Linear(self.cfg.bus_dim, self.cfg.bus_dim), nn.SiLU(),
            nn.Linear(self.cfg.bus_dim, self.cfg.dynamics_context_dim),
        )
        self.vision_fusion = nn.Linear(self.cfg.bus_dim * 2, self.cfg.bus_dim)
        with torch.no_grad():
            self.vision_fusion.weight.zero_()
            self.vision_fusion.bias.zero_()
            self.vision_fusion.weight[:, :self.cfg.bus_dim].copy_(torch.eye(self.cfg.bus_dim))

        self.audio = InfantAudioCortex(self.cfg.bus_dim)
        self.nucleus = CorticalNucleus(self.cfg)
        self.language = InfantLanguageCortex(self.cfg)
        # Existing temporal cortex is preserved as a semantic-sequence organ.
        self.temporal = TemporalCortex(self.cfg)
        # New dorsal temporal cortex is dedicated to action-conditioned spatial dynamics.
        self.dynamics_temporal = TemporalCortex(self.cfg)
        self.world = WorldModelEnsemble(self.cfg)
        self.planning = PlanningCortex(self.cfg)
        self.values = InfantValueCortex(self.cfg)
        self.motor = InfantMotorCortex(self.cfg)
        self.vision_parent_heads = nn.ModuleDict()
        for name, dim in (vision_parent_dims or {}).items():
            self.add_vision_parent_head(name, int(dim))
        self.sync_target_cortices(tau=0.0)


    def train(self, mode: bool = True):
        super().train(mode)
        # Target cortices are updated only by EMA, never by their own BatchNorm
        # observations or optimizer gradients.
        self.target_vision.eval()
        self.target_spatial.eval()
        return self

    @staticmethod
    @torch.no_grad()
    def _ema_module(target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_p, source_p in zip(target.parameters(), source.parameters()):
            target_p.data.mul_(tau).add_(source_p.data, alpha=1.0 - tau)
        for target_b, source_b in zip(target.buffers(), source.buffers()):
            if torch.is_floating_point(target_b):
                target_b.data.mul_(tau).add_(source_b.data, alpha=1.0 - tau)
            else:
                target_b.data.copy_(source_b.data)

    @torch.no_grad()
    def sync_target_cortices(self, tau: float | None = None) -> None:
        tau = self.cfg.world_ema_tau if tau is None else float(tau)
        tau = max(0.0, min(1.0, tau))
        self._ema_module(self.target_vision, self.vision, tau)
        self._ema_module(self.target_spatial, self.spatial, tau)
        self.target_vision.eval()
        self.target_spatial.eval()

    @torch.no_grad()
    def sync_target_vision(self, tau: float | None = None) -> None:
        # Backward-compatible helper used by older tests/tools.
        tau = self.cfg.world_ema_tau if tau is None else float(tau)
        tau = max(0.0, min(1.0, tau))
        self._ema_module(self.target_vision, self.vision, tau)
        self.target_vision.eval()

    @torch.no_grad()
    def target_visual(self, images: torch.Tensor) -> torch.Tensor:
        self.target_vision.eval()
        return self.target_vision(images)

    @torch.no_grad()
    def target_spatial_bus(self, images: torch.Tensor) -> torch.Tensor:
        self.target_spatial.eval()
        return self.target_spatial(images)

    def fuse_vision(self, semantic_bus: torch.Tensor, spatial_bus: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.vision_fusion(torch.cat((semantic_bus, spatial_bus), dim=-1)))

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
        # Report trainable child capacity; the EMA target cortex is a shadow copy,
        # not additional independently trainable intelligence.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def checkpoint_payload(self, metadata=None) -> dict:
        return {
            "format": "continuous_brain.infant.v100",
            "config": asdict(self.cfg),
            "vision_parent_dims": {
                k: v.out_features for k, v in self.vision_parent_heads.items()
            },
            "model": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "metadata": dict(metadata or {}),
        }


def one_hot_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.one_hot(action.long().clamp(0, action_dim - 1), num_classes=action_dim).float()
