#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
)

WORLD_DIM = 4
SELF_CONT_DIM = 6
PRIMARY_DIM = 3  # NONE / A / B
MODE_DIM = 2     # NORMAL / INVESTIGATE
ACTION_DIM = 4

PRIMARY_NONE = 0
PRIMARY_A = 1
PRIMARY_B = 2

MODE_NORMAL = 0
MODE_INVESTIGATE = 1


@dataclass
class ExplicitIntentionState:
    primary: int
    mode: int

    @property
    def active_goal(self) -> int:
        if int(self.mode) == MODE_INVESTIGATE:
            return GOAL_INVESTIGATE
        if int(self.primary) == PRIMARY_A:
            return GOAL_COMMITMENT_A
        if int(self.primary) == PRIMARY_B:
            return GOAL_COMMITMENT_B
        return GOAL_NONE


def intention_vector(primary, mode):
    p = F.one_hot(
        torch.as_tensor(primary).long(),
        num_classes=PRIMARY_DIM,
    ).float()
    m = F.one_hot(
        torch.as_tensor(mode).long(),
        num_classes=MODE_DIM,
    ).float()
    return torch.cat((p, m), dim=-1)


def active_goal_from_intention(primary, mode):
    if isinstance(primary, torch.Tensor):
        out = torch.zeros_like(primary.long())
        out[primary == PRIMARY_A] = GOAL_COMMITMENT_A
        out[primary == PRIMARY_B] = GOAL_COMMITMENT_B
        out[mode == MODE_INVESTIGATE] = GOAL_INVESTIGATE
        return out

    if int(mode) == MODE_INVESTIGATE:
        return GOAL_INVESTIGATE
    if int(primary) == PRIMARY_A:
        return GOAL_COMMITMENT_A
    if int(primary) == PRIMARY_B:
        return GOAL_COMMITMENT_B
    return GOAL_NONE


class RecurrentSelfTrajectoryModel(nn.Module):
    """
    Recurrent imagined trajectory model with EXPLICIT intention state.

    The model does not reconstruct intention from continuous self scalars.
    Primary commitment and temporary investigate mode are separate state
    variables carried through every imagined step.
    """
    def __init__(
        self,
        hidden_dim: int = 144,
        context_dim: int = 96,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)

        self.context_encoder = nn.Sequential(
            nn.Linear(
                WORLD_DIM
                + SELF_CONT_DIM
                + PRIMARY_DIM
                + MODE_DIM,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        step_dim = (
            WORLD_DIM
            + SELF_CONT_DIM
            + PRIMARY_DIM
            + MODE_DIM
            + ACTION_DIM
        )
        self.gru = nn.GRUCell(
            step_dim,
            hidden_dim,
        )

        self.world_delta = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, WORLD_DIM),
        )
        self.self_delta = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, SELF_CONT_DIM),
        )
        self.primary_head = nn.Linear(
            hidden_dim,
            PRIMARY_DIM,
        )
        self.mode_head = nn.Linear(
            hidden_dim,
            MODE_DIM,
        )
        self.agency_head = nn.Linear(
            hidden_dim,
            1,
        )

    def init_hidden(
        self,
        world,
        self_cont,
        primary,
        mode,
    ):
        x = torch.cat(
            (
                world,
                self_cont,
                intention_vector(primary, mode),
            ),
            dim=-1,
        )
        return self.context_encoder(x)

    def step(
        self,
        world,
        self_cont,
        primary,
        mode,
        action,
        hidden,
    ):
        a = F.one_hot(
            action.long(),
            num_classes=ACTION_DIM,
        ).float()
        intention = intention_vector(
            primary,
            mode,
        )
        x = torch.cat(
            (
                world,
                self_cont,
                intention,
                a,
            ),
            dim=-1,
        )
        hidden = self.gru(x, hidden)

        dw = 0.30 * torch.tanh(
            self.world_delta(hidden)
        )
        ds = 0.55 * torch.tanh(
            self.self_delta(hidden)
        )

        next_world = torch.clamp(
            world + dw,
            0.0,
            1.0,
        )
        next_self = torch.clamp(
            self_cont + ds,
            0.0,
            1.0,
        )

        return {
            "hidden": hidden,
            "next_world": next_world,
            "next_self_cont": next_self,
            "primary_logits":
                self.primary_head(hidden),
            "mode_logits":
                self.mode_head(hidden),
            "agency_logit":
                self.agency_head(hidden).squeeze(-1),
        }


@torch.no_grad()
def rollout(
    model: RecurrentSelfTrajectoryModel,
    world: torch.Tensor,
    self_cont: torch.Tensor,
    primary: int,
    mode: int,
    actions,
):
    if world.ndim == 1:
        world = world.unsqueeze(0)
    if self_cont.ndim == 1:
        self_cont = self_cont.unsqueeze(0)

    p = torch.tensor(
        [int(primary)],
        dtype=torch.long,
        device=world.device,
    )
    m = torch.tensor(
        [int(mode)],
        dtype=torch.long,
        device=world.device,
    )

    hidden = model.init_hidden(
        world,
        self_cont,
        p,
        m,
    )

    worlds = []
    selfs = []
    primaries = []
    modes = []
    agencies = []

    w = world
    s = self_cont

    for action in actions:
        a = torch.tensor(
            [int(action)],
            dtype=torch.long,
            device=world.device,
        )
        out = model.step(
            w,
            s,
            p,
            m,
            a,
            hidden,
        )
        hidden = out["hidden"]
        w = out["next_world"]
        s = out["next_self_cont"]
        p = out["primary_logits"].argmax(
            dim=-1
        )
        m = out["mode_logits"].argmax(
            dim=-1
        )

        worlds.append(w[0].clone())
        selfs.append(s[0].clone())
        primaries.append(int(p[0].item()))
        modes.append(int(m[0].item()))
        agencies.append(
            float(
                torch.sigmoid(
                    out["agency_logit"]
                )[0].item()
            )
        )

    return {
        "worlds": torch.stack(worlds),
        "selfs": torch.stack(selfs),
        "primaries": primaries,
        "modes": modes,
        "agencies": agencies,
        "final_primary": primaries[-1],
        "final_mode": modes[-1],
        "final_active_goal":
            active_goal_from_intention(
                primaries[-1],
                modes[-1],
            ),
    }



def update_explicit_intention(
    primary: int,
    mode: int,
    self_cont,
    *,
    enter_error: float = 0.60,
    enter_uncertainty: float = 0.86,
    exit_error: float = 0.45,
    exit_uncertainty: float = 0.72,
    completion_threshold: float = 0.05,
):
    """
    Continuity-preserving intention state machine.

    Primary commitment and temporary investigate mode are distinct.

    Hysteresis prevents small numerical fluctuations around an uncertainty/error
    threshold from repeatedly erasing/restoring the continuing intention.
    """
    if isinstance(self_cont, torch.Tensor):
        v = self_cont.detach().cpu().float().reshape(-1)
        a = float(v[0].item())
        b = float(v[1].item())
        uncertainty = float(v[3].item())
        pred_error = float(v[4].item())
    else:
        a = float(self_cont[0])
        b = float(self_cont[1])
        uncertainty = float(self_cont[3])
        pred_error = float(self_cont[4])

    primary = int(primary)
    mode = int(mode)

    # Primary commitment changes only when it is actually completed (or when
    # NONE sees a clearly unresolved commitment). Temporary investigate does
    # not overwrite it.
    if primary == PRIMARY_A and a <= completion_threshold:
        if b > completion_threshold:
            primary = PRIMARY_B
        else:
            primary = PRIMARY_NONE
    elif primary == PRIMARY_B and b <= completion_threshold:
        if a > completion_threshold:
            primary = PRIMARY_A
        else:
            primary = PRIMARY_NONE
    elif primary == PRIMARY_NONE:
        if max(a, b) > 0.10:
            if a > b:
                primary = PRIMARY_A
            elif b > a:
                primary = PRIMARY_B
            # exact ties preserve NONE

    if mode == MODE_NORMAL:
        if (
            pred_error >= enter_error
            or uncertainty >= enter_uncertainty
        ):
            mode = MODE_INVESTIGATE
    else:
        if (
            pred_error <= exit_error
            and uncertainty <= exit_uncertainty
        ):
            mode = MODE_NORMAL

    return ExplicitIntentionState(
        primary=primary,
        mode=mode,
    )


def trajectory_self_value(
    final_self_cont: torch.Tensor,
    *,
    primary: int,
    final_mode: int,
):
    """
    Scalar value of an imagined future self relative to its explicit primary
    commitment. Used only for the synthetic trajectory-planning assay.
    """
    s = final_self_cont
    if int(primary) == PRIMARY_A:
        unresolved = s[..., 0]
    elif int(primary) == PRIMARY_B:
        unresolved = s[..., 1]
    else:
        unresolved = 0.5 * (s[..., 0] + s[..., 1])

    confidence = s[..., 2]
    uncertainty = s[..., 3]
    pred_error = s[..., 4]

    value = (
        0.68 * (1.0 - unresolved)
        + 0.12 * confidence
        + 0.10 * (1.0 - uncertainty)
        + 0.10 * (1.0 - pred_error)
    )
    if int(final_mode) == MODE_INVESTIGATE:
        value = value - 0.08
    return value

def save_trajectory_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "format":
                "continuous_brain.recurrent_self_trajectory_v1",
            "hidden_dim":
                int(model.hidden_dim),
            "context_dim":
                int(model.context_dim),
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


def load_trajectory_model(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model = RecurrentSelfTrajectoryModel(
        hidden_dim=int(
            payload["hidden_dim"]
        ),
        context_dim=int(
            payload["context_dim"]
        ),
    ).to(device)
    model.load_state_dict(
        payload["state_dict"]
    )
    model.eval()
    return model, payload
