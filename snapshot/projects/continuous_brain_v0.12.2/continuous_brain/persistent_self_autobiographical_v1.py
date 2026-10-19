#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import uuid

import torch
from torch import nn
from torch.nn import functional as F

GOAL_NONE = 0
GOAL_COMMITMENT_A = 1
GOAL_COMMITMENT_B = 2
GOAL_INVESTIGATE = 3

GOAL_NAMES = (
    "none",
    "commitment_a",
    "commitment_b",
    "investigate",
)

ACTION_OBSERVE = 0
ACTION_WORK_A = 1
ACTION_WORK_B = 2
ACTION_WAIT = 3
ACTION_NAMES = ("observe", "work_a", "work_b", "wait")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


@dataclass
class PersistentSelfState:
    identity_token: str
    identity_embedding: torch.Tensor
    unresolved_a: float
    unresolved_b: float
    active_goal: int
    confidence: float
    uncertainty: float
    prediction_error_ema: float
    agency_ema: float
    cycle: int
    autobiographical_cursor: int

    @classmethod
    def create(cls, identity_seed: int = 28001) -> "PersistentSelfState":
        g = torch.Generator().manual_seed(int(identity_seed))
        identity = torch.randn(32, generator=g)
        identity = F.normalize(identity, dim=0)
        token = hashlib.sha256(
            identity.detach().cpu().numpy().tobytes()
        ).hexdigest()[:32]
        return cls(
            identity_token=token,
            identity_embedding=identity,
            unresolved_a=0.0,
            unresolved_b=0.0,
            active_goal=GOAL_NONE,
            confidence=0.50,
            uncertainty=0.25,
            prediction_error_ema=0.0,
            agency_ema=0.50,
            cycle=0,
            autobiographical_cursor=0,
        )

    def clone(self) -> "PersistentSelfState":
        return PersistentSelfState(
            identity_token=str(self.identity_token),
            identity_embedding=self.identity_embedding.detach().clone(),
            unresolved_a=float(self.unresolved_a),
            unresolved_b=float(self.unresolved_b),
            active_goal=int(self.active_goal),
            confidence=float(self.confidence),
            uncertainty=float(self.uncertainty),
            prediction_error_ema=float(self.prediction_error_ema),
            agency_ema=float(self.agency_ema),
            cycle=int(self.cycle),
            autobiographical_cursor=int(self.autobiographical_cursor),
        )

    def vector(self) -> torch.Tensor:
        goal = F.one_hot(
            torch.tensor(int(self.active_goal), dtype=torch.long),
            num_classes=4,
        ).float()
        numeric = torch.tensor(
            [
                float(self.unresolved_a),
                float(self.unresolved_b),
                float(self.confidence),
                float(self.uncertainty),
                float(self.prediction_error_ema),
                float(self.agency_ema),
            ],
            dtype=torch.float32,
        )
        return torch.cat((numeric, goal), dim=0)


@dataclass
class AutobiographicalEpisode:
    episode_id: int
    cycle: int
    event_type: str
    active_goal_before: int
    unresolved_a_before: float
    unresolved_b_before: float
    confidence_before: float
    uncertainty_before: float
    prediction_error_before: float
    action: int
    intended_change_a: float
    intended_change_b: float
    observed_change_a: float
    observed_change_b: float
    action_success: float
    prediction_error: float
    agency_probability: float
    completed_goal: int
    active_goal_after: int
    unresolved_a_after: float
    unresolved_b_after: float
    confidence_after: float
    uncertainty_after: float
    prediction_error_after: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**dict(d))


class AutobiographicalMemory:
    def __init__(self, capacity: int = 512):
        self.capacity = int(capacity)
        self.episodes: list[AutobiographicalEpisode] = []

    def append(self, episode: AutobiographicalEpisode):
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def clone(self) -> "AutobiographicalMemory":
        out = AutobiographicalMemory(self.capacity)
        out.episodes = [
            AutobiographicalEpisode.from_dict(e.to_dict())
            for e in self.episodes
        ]
        return out

    def erase_episode(self, episode_id: int):
        self.episodes = [
            e for e in self.episodes
            if int(e.episode_id) != int(episode_id)
        ]

    def replace_episode(
        self,
        episode_id: int,
        donor: AutobiographicalEpisode,
    ):
        replaced = False
        new = []
        for e in self.episodes:
            if int(e.episode_id) == int(episode_id):
                d = donor.to_dict()
                d["episode_id"] = int(episode_id)
                new.append(AutobiographicalEpisode.from_dict(d))
                replaced = True
            else:
                new.append(e)
        if not replaced:
            raise KeyError(episode_id)
        self.episodes = new

    def latest(self, n: int = 8):
        return self.episodes[-int(n):]

    def to_list(self):
        return [e.to_dict() for e in self.episodes]

    @classmethod
    def from_list(cls, rows, capacity=512):
        out = cls(capacity)
        out.episodes = [
            AutobiographicalEpisode.from_dict(r)
            for r in rows
        ]
        return out


class AgencySelfModel(nn.Module):
    """
    Small learned self-model.

    Input:
      persistent self vector (10)
      action one-hot (4)
      intended world change (2)
      observed world change (2)
      action-success signal (1)
      instantaneous prediction error (1)

    Outputs:
      agency logit
      next confidence
      next uncertainty
    """
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.self_dim = 10
        self.action_dim = 4
        self.event_dim = 6
        self.input_dim = self.self_dim + self.action_dim + self.event_dim
        self.hidden_dim = int(hidden_dim)

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.agency = nn.Linear(hidden_dim, 1)
        self.next_internal = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        self_vector: torch.Tensor,
        action: torch.Tensor,
        intended_change: torch.Tensor,
        observed_change: torch.Tensor,
        action_success: torch.Tensor,
        prediction_error: torch.Tensor,
    ):
        action_oh = F.one_hot(
            action.long(), num_classes=self.action_dim
        ).float()
        x = torch.cat(
            (
                self_vector,
                action_oh,
                intended_change,
                observed_change,
                action_success.unsqueeze(-1),
                prediction_error.unsqueeze(-1),
            ),
            dim=-1,
        )
        h = self.trunk(x)
        next_internal = torch.sigmoid(self.next_internal(h))
        return {
            "agency_logit": self.agency(h).squeeze(-1),
            "next_confidence": next_internal[:, 0],
            "next_uncertainty": next_internal[:, 1],
        }


def target_internal_update(
    confidence: float,
    uncertainty: float,
    *,
    agency_target: float,
    prediction_error: float,
):
    """
    Training target for the learned self-transition component.

    Self-caused, expected effects increase control-confidence and reduce
    world-uncertainty.  Unexpected effects attributed to the world increase
    uncertainty.  Unexpected self-caused effects reduce confidence more than
    they increase world uncertainty.
    """
    c = float(confidence)
    u = float(uncertainty)
    a = float(agency_target)
    pe = clip01(prediction_error)

    if a >= 0.5:
        if pe <= 0.25:
            c2 = c + 0.18 * (1.0 - c)
            u2 = u * 0.72
        else:
            c2 = c * (1.0 - 0.22 * pe)
            u2 = u + 0.05 * pe * (1.0 - u)
    else:
        c2 = c * (1.0 - 0.05 * pe)
        u2 = u + 0.32 * pe * (1.0 - u)

    return clip01(c2), clip01(u2)


def event_features(
    state: PersistentSelfState,
    *,
    action: int,
    intended_change,
    observed_change,
    action_success: float,
    prediction_error: float,
):
    return {
        "self_vector": state.vector().unsqueeze(0),
        "action": torch.tensor([int(action)], dtype=torch.long),
        "intended_change": torch.tensor(
            [list(map(float, intended_change))],
            dtype=torch.float32,
        ),
        "observed_change": torch.tensor(
            [list(map(float, observed_change))],
            dtype=torch.float32,
        ),
        "action_success": torch.tensor(
            [float(action_success)],
            dtype=torch.float32,
        ),
        "prediction_error": torch.tensor(
            [float(prediction_error)],
            dtype=torch.float32,
        ),
    }


class EndogenousGoalArbiter:
    """
    Minimal endogenous arbitration.

    Investigate is generated from internal prediction-error / uncertainty,
    not from an externally assigned task. Otherwise unresolved commitments
    compete for control.
    """
    def __init__(
        self,
        prediction_error_threshold: float = 0.55,
        uncertainty_threshold: float = 0.82,
    ):
        self.prediction_error_threshold = float(
            prediction_error_threshold
        )
        self.uncertainty_threshold = float(
            uncertainty_threshold
        )

    def choose(self, state: PersistentSelfState) -> int:
        if (
            float(state.prediction_error_ema)
            >= self.prediction_error_threshold
            or float(state.uncertainty)
            >= self.uncertainty_threshold
        ):
            return GOAL_INVESTIGATE

        a = float(state.unresolved_a)
        b = float(state.unresolved_b)
        if max(a, b) <= 1e-6:
            return GOAL_NONE
        if a >= b:
            return GOAL_COMMITMENT_A
        return GOAL_COMMITMENT_B


class PersistentSelfCore:
    def __init__(
        self,
        model: AgencySelfModel,
        state: PersistentSelfState | None = None,
        memory: AutobiographicalMemory | None = None,
        arbiter: EndogenousGoalArbiter | None = None,
    ):
        self.model = model
        self.model.eval()
        self.state = (
            state.clone()
            if state is not None
            else PersistentSelfState.create()
        )
        self.memory = (
            memory.clone()
            if memory is not None
            else AutobiographicalMemory()
        )
        self.arbiter = (
            arbiter
            if arbiter is not None
            else EndogenousGoalArbiter()
        )

    def choose_goal(self):
        goal = self.arbiter.choose(self.state)
        self.state.active_goal = int(goal)
        return int(goal)

    def add_commitment(self, goal: int, strength: float = 1.0):
        strength = clip01(strength)
        if int(goal) == GOAL_COMMITMENT_A:
            self.state.unresolved_a = max(
                self.state.unresolved_a, strength
            )
        elif int(goal) == GOAL_COMMITMENT_B:
            self.state.unresolved_b = max(
                self.state.unresolved_b, strength
            )
        else:
            raise ValueError("commitment goal must be A or B")
        self.choose_goal()

    def record_commitment_event(self, goal: int):
        before = self.state.clone()
        self.add_commitment(goal, 1.0)
        after = self.state.clone()
        ep = self._episode(
            before,
            after,
            event_type="commitment_created",
            action=ACTION_OBSERVE,
            intended_change=(0.0, 0.0),
            observed_change=(0.0, 0.0),
            action_success=1.0,
            prediction_error=0.0,
            agency_probability=1.0,
            completed_goal=GOAL_NONE,
        )
        self.memory.append(ep)
        self.state.autobiographical_cursor = len(self.memory.episodes)
        return ep

    def record_completion(self, goal: int):
        before = self.state.clone()
        if int(goal) == GOAL_COMMITMENT_A:
            self.state.unresolved_a = 0.0
        elif int(goal) == GOAL_COMMITMENT_B:
            self.state.unresolved_b = 0.0
        else:
            raise ValueError(goal)

        self.state.prediction_error_ema *= 0.5
        self.choose_goal()
        self.state.cycle += 1
        after = self.state.clone()

        ep = self._episode(
            before,
            after,
            event_type="commitment_completed",
            action=(
                ACTION_WORK_A
                if int(goal) == GOAL_COMMITMENT_A
                else ACTION_WORK_B
            ),
            intended_change=(
                (1.0, 0.0)
                if int(goal) == GOAL_COMMITMENT_A
                else (0.0, 1.0)
            ),
            observed_change=(
                (1.0, 0.0)
                if int(goal) == GOAL_COMMITMENT_A
                else (0.0, 1.0)
            ),
            action_success=1.0,
            prediction_error=0.0,
            agency_probability=1.0,
            completed_goal=int(goal),
        )
        self.memory.append(ep)
        self.state.autobiographical_cursor = len(self.memory.episodes)
        return ep

    @torch.no_grad()
    def observe_event(
        self,
        *,
        action: int,
        intended_change,
        observed_change,
        action_success: float,
        prediction_error: float,
        completed_goal: int = GOAL_NONE,
        forced_agency_probability: float | None = None,
    ):
        before = self.state.clone()
        features = event_features(
            before,
            action=action,
            intended_change=intended_change,
            observed_change=observed_change,
            action_success=action_success,
            prediction_error=prediction_error,
        )
        out = self.model(**features)
        agency = float(
            torch.sigmoid(out["agency_logit"])[0].item()
        )
        if forced_agency_probability is not None:
            agency = clip01(forced_agency_probability)

        if forced_agency_probability is None:
            next_conf = float(
                out["next_confidence"][0].item()
            )
            next_unc = float(
                out["next_uncertainty"][0].item()
            )
        else:
            # Intervention semantics use the same preregistered self-update
            # relation used to train the model.
            next_conf, next_unc = target_internal_update(
                before.confidence,
                before.uncertainty,
                agency_target=agency,
                prediction_error=prediction_error,
            )

        self.state.confidence = clip01(next_conf)
        self.state.uncertainty = clip01(next_unc)
        # Raw outcome error is not automatically treated as world surprise.
        # If the event is attributed to the self, most of the discrepancy is
        # interpreted as a self/competence error rather than evidence that the
        # external world changed independently.
        world_attributed_error = clip01(
            float(prediction_error)
            * (1.0 - 0.75 * float(agency))
        )
        self.state.prediction_error_ema = clip01(
            0.55 * float(before.prediction_error_ema)
            + 0.45 * world_attributed_error
        )
        self.state.agency_ema = clip01(
            0.80 * float(before.agency_ema)
            + 0.20 * agency
        )

        if int(completed_goal) == GOAL_COMMITMENT_A:
            self.state.unresolved_a = 0.0
        elif int(completed_goal) == GOAL_COMMITMENT_B:
            self.state.unresolved_b = 0.0

        self.state.cycle += 1
        self.choose_goal()
        after = self.state.clone()

        ep = self._episode(
            before,
            after,
            event_type="experienced_transition",
            action=int(action),
            intended_change=intended_change,
            observed_change=observed_change,
            action_success=float(action_success),
            prediction_error=float(prediction_error),
            agency_probability=float(agency),
            completed_goal=int(completed_goal),
        )
        self.memory.append(ep)
        self.state.autobiographical_cursor = len(self.memory.episodes)

        return {
            "agency_probability": float(agency),
            "next_confidence": float(self.state.confidence),
            "next_uncertainty": float(self.state.uncertainty),
            "active_goal": int(self.state.active_goal),
            "episode": ep,
        }

    def _episode(
        self,
        before,
        after,
        *,
        event_type,
        action,
        intended_change,
        observed_change,
        action_success,
        prediction_error,
        agency_probability,
        completed_goal,
    ):
        return AutobiographicalEpisode(
            episode_id=(
                max(
                    [e.episode_id for e in self.memory.episodes],
                    default=-1,
                )
                + 1
            ),
            cycle=int(before.cycle),
            event_type=str(event_type),
            active_goal_before=int(before.active_goal),
            unresolved_a_before=float(before.unresolved_a),
            unresolved_b_before=float(before.unresolved_b),
            confidence_before=float(before.confidence),
            uncertainty_before=float(before.uncertainty),
            prediction_error_before=float(before.prediction_error_ema),
            action=int(action),
            intended_change_a=float(intended_change[0]),
            intended_change_b=float(intended_change[1]),
            observed_change_a=float(observed_change[0]),
            observed_change_b=float(observed_change[1]),
            action_success=float(action_success),
            prediction_error=float(prediction_error),
            agency_probability=float(agency_probability),
            completed_goal=int(completed_goal),
            active_goal_after=int(after.active_goal),
            unresolved_a_after=float(after.unresolved_a),
            unresolved_b_after=float(after.unresolved_b),
            confidence_after=float(after.confidence),
            uncertainty_after=float(after.uncertainty),
            prediction_error_after=float(after.prediction_error_ema),
        )

    def reconcile_commitments_from_autobiography(
        self,
        *,
        seed_a: float,
        seed_b: float,
    ):
        a = clip01(seed_a)
        b = clip01(seed_b)
        for ep in self.memory.episodes:
            if ep.event_type == "commitment_created":
                if ep.active_goal_after == GOAL_COMMITMENT_A:
                    a = max(a, ep.unresolved_a_after)
                elif ep.active_goal_after == GOAL_COMMITMENT_B:
                    b = max(b, ep.unresolved_b_after)

            if int(ep.completed_goal) == GOAL_COMMITMENT_A:
                a = 0.0
            elif int(ep.completed_goal) == GOAL_COMMITMENT_B:
                b = 0.0

        self.state.unresolved_a = float(a)
        self.state.unresolved_b = float(b)
        return self.choose_goal()


def save_self_bundle(path, core: PersistentSelfCore, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = core.state
    torch.save(
        {
            "format":
                "continuous_brain.persistent_self_autobiographical_v1",
            "state": {
                "identity_token": s.identity_token,
                "identity_embedding":
                    s.identity_embedding.detach().cpu(),
                "unresolved_a": float(s.unresolved_a),
                "unresolved_b": float(s.unresolved_b),
                "active_goal": int(s.active_goal),
                "confidence": float(s.confidence),
                "uncertainty": float(s.uncertainty),
                "prediction_error_ema":
                    float(s.prediction_error_ema),
                "agency_ema": float(s.agency_ema),
                "cycle": int(s.cycle),
                "autobiographical_cursor":
                    int(s.autobiographical_cursor),
            },
            "memory": core.memory.to_list(),
            "memory_capacity": int(core.memory.capacity),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_self_bundle(path, model):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    s = payload["state"]
    state = PersistentSelfState(
        identity_token=str(s["identity_token"]),
        identity_embedding=s["identity_embedding"].clone(),
        unresolved_a=float(s["unresolved_a"]),
        unresolved_b=float(s["unresolved_b"]),
        active_goal=int(s["active_goal"]),
        confidence=float(s["confidence"]),
        uncertainty=float(s["uncertainty"]),
        prediction_error_ema=float(s["prediction_error_ema"]),
        agency_ema=float(s["agency_ema"]),
        cycle=int(s["cycle"]),
        autobiographical_cursor=int(
            s["autobiographical_cursor"]
        ),
    )
    memory = AutobiographicalMemory.from_list(
        payload["memory"],
        capacity=int(payload["memory_capacity"]),
    )
    return PersistentSelfCore(
        model=model,
        state=state,
        memory=memory,
    ), payload


def save_self_model(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.agency_self_model_v1",
            "hidden_dim": int(model.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_self_model(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model = AgencySelfModel(
        hidden_dim=int(payload["hidden_dim"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
