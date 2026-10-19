#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib
import torch
from torch import nn
from continuous_brain.memory_guided_goal_option_v1 import EXPECTED_BASE_SHA, EXPECTED_MEMORY_SHA

GOAL_DECODER_PATH = Path("state/memory_guided_goal_option_v1/goal_decoder.pt")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024), b""): h.update(c)
    return h.hexdigest()

class AdditiveGoalOptionBank(nn.Module):
    def __init__(self, planning_dim:int, action_dim:int, goal_dim:int=32):
        super().__init__()
        self.planning_dim=int(planning_dim); self.action_dim=int(action_dim); self.goal_dim=int(goal_dim)
        self.goal_embedding=nn.Embedding(2,self.goal_dim)
        h=self.planning_dim
        self.trunk=nn.Sequential(
            nn.Linear(self.planning_dim+self.goal_dim,h), nn.SiLU(),
            nn.Linear(h,h), nn.SiLU(),
        )
        self.action_head=nn.Linear(h,self.action_dim)
        self.termination_head=nn.Linear(h,1)

    def features(self, latent, goal_side):
        emb=self.goal_embedding(goal_side.long().clamp(0,1))
        return self.trunk(torch.cat((latent,emb),dim=-1))

    def action_logits(self, latent, goal_side):
        return self.action_head(self.features(latent,goal_side))

    def termination_logit(self, latent, goal_side):
        return self.termination_head(self.features(latent,goal_side)).squeeze(-1)

def save_goal_options(path, bank, goal_decoder_sha256, metadata):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format":"continuous_brain.additive_goal_options_v1",
        "base_checkpoint_sha256":EXPECTED_BASE_SHA,
        "memory_checkpoint_sha256":EXPECTED_MEMORY_SHA,
        "goal_decoder_sha256":str(goal_decoder_sha256),
        "planning_dim":bank.planning_dim,
        "action_dim":bank.action_dim,
        "goal_dim":bank.goal_dim,
        "goal_option_count":2,
        "historical_option_count_preserved":6,
        "state_dict":{k:v.detach().cpu() for k,v in bank.state_dict().items()},
        "metadata":dict(metadata),
    }, path)

def load_goal_options(path, expected_goal_decoder_sha256, device="cpu"):
    p=torch.load(path,map_location="cpu",weights_only=True)
    if p["base_checkpoint_sha256"]!=EXPECTED_BASE_SHA: raise RuntimeError("base provenance mismatch")
    if p["memory_checkpoint_sha256"]!=EXPECTED_MEMORY_SHA: raise RuntimeError("memory provenance mismatch")
    if p["goal_decoder_sha256"]!=expected_goal_decoder_sha256: raise RuntimeError("goal decoder provenance mismatch")
    if int(p["historical_option_count_preserved"])!=6: raise RuntimeError("historical option preservation mismatch")
    bank=AdditiveGoalOptionBank(int(p["planning_dim"]),int(p["action_dim"]),int(p["goal_dim"])).to(device)
    bank.load_state_dict(p["state_dict"]); bank.eval()
    return bank,p
