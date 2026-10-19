"""Mechanism boundary table for the six G0-requalified legacy mechanisms.

For each mechanism: the runtime slot it is hosted in, the legacy callables
whose every invocation crosses the canonical store (arguments and results
serialized through blob channels), the loaders whose checkpoints become one
runtime-owned shared instance, the stateless constructors whose instances
are shared, and a projection that keeps the slot's canonical channels
current.

Slot choices that are spec decisions, not settled facts (review D3/S3):
  * event_consistent_self_development_v0341 is hosted in `skills`, because
    its HistoryRelativeDevelopmentalController selects a developmental
    strategy (SCAFFOLD/DIRECT) - an option selector. g0_plan lists only
    `curriculum`, which v0361 also needs and a slot holds one component.
  * consequence_replanning_v0252 is hosted in `planner`; its consequence
    model is part of that host, so `world_model` stays unbound.
  * reliability_from_memory is defined in the v0.33 module but is the read
    API of the v0.32.2 neutral autobiographical model, so it is hosted by
    the autobiography host.
  * build_substrate and the v0.21/v0.23 memory path used by v0252 are not
    among the six mechanisms; they stay inside the frozen evaluator
    (reported, not routed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class API:
    module: str
    qualname: str              # "fn" or "Class.method"

    @property
    def is_method(self) -> bool:
        return "." in self.qualname


@dataclass(frozen=True)
class SharedLoader:
    module: str
    name: str
    paths: tuple[str, ...]     # pinned checkpoints (relative to the legacy root) loaded once at binding
    instance_prefix: str


@dataclass(frozen=True)
class SharedConstructor:
    module: str
    qualname: str
    instance_key: str


@dataclass(frozen=True)
class CanonicalChannel:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    slot: str
    apis: tuple[API, ...]
    loaders: tuple[SharedLoader, ...] = ()
    constructors: tuple[SharedConstructor, ...] = ()
    canonical: tuple[CanonicalChannel, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _f(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _vec(t: Any, n: int) -> np.ndarray:
    try:
        import torch
        if isinstance(t, torch.Tensor):
            a = t.detach().reshape(-1).to(dtype=__import__("torch").float64).numpy()
        else:
            a = np.asarray(t, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(n, np.nan)
    out = np.full(n, np.nan)
    out[:min(n, a.size)] = a[:n]
    return out


def _finite(a: np.ndarray, fill: float = -1.0) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a[~np.isfinite(a)] = fill
    return a


# ---- canonical projections: (api qualname, decoded args, result) -> {channel: value} or {} to hold

def project_self_state(qual: str, args: tuple, result: Any) -> dict:
    if qual == "rollout" and isinstance(result, dict) and "selfs" in result:
        selfs = result["selfs"]
        last = selfs[-1] if getattr(selfs, "ndim", 0) >= 2 else selfs
        return {"self_state.vector": _finite(_vec(last, 6))}
    return {}


def project_autobiography(qual: str, args: tuple, result: Any) -> dict:
    if qual == "reliability_from_memory" and len(args) >= 2:
        memory = args[1]
        out = {"autobiography.reliability": _finite(_vec(result, 2))}
        records = getattr(memory, "records", None)
        if records:
            try:
                feat = records[-1].feature()
                out["autobiography.record"] = _finite(_vec(feat, 27))
                out["autobiography.discrepancy"] = _finite(_vec(feat[-2:], 2))
            except Exception:
                pass
        return out
    return {}


def project_skills(qual: str, args: tuple, result: Any) -> dict:
    if qual.endswith(".strategy"):
        return {"skills.selected_option": np.array([_f(result)], dtype=np.float64)}
    if qual.endswith((".initialize", ".update")) and hasattr(result, "strategy"):
        return {"skills.selected_option": np.array([_f(result.strategy)], dtype=np.float64)}
    return {}


def project_goal_genesis(qual: str, args: tuple, result: Any) -> dict:
    if qual.endswith(".generate"):
        if result is None:
            return {"goal_genesis.active_goal": np.array([0.0, -1.0, -1.0, -1.0])}
        return {"goal_genesis.active_goal": _finite(np.array(
            [1.0, _f(getattr(result, "primary", -1)), _f(getattr(result, "baseline_reliability", -1)),
             _f(getattr(result, "target_reliability", -1))]))}
    return {}


def project_curriculum(qual: str, args: tuple, result: Any) -> dict:
    if qual in ("goal_is_advanced", "same_goal_semantics"):
        return {"curriculum.next_goal": np.array([1.0 if bool(result) else 0.0,
                                                  1.0 if qual == "goal_is_advanced" else 2.0])}
    return {}


def project_planner(qual: str, args: tuple, result: Any) -> dict:
    if qual in ("plan_route", "plan_route_cost_consistent"):
        seq = result.get("plan", result.get("actions")) if isinstance(result, dict) else result
        try:
            seq = list(seq)
            first = float(seq[0]) if seq else -1.0
            return {"planner.plan": _finite(np.array([float(len(seq)), first]))}
        except Exception:
            return {"planner.plan": np.array([-1.0, -1.0])}
    return {}


PROJECTIONS: dict[str, Callable[[str, tuple, Any], dict]] = {
    "self_state": project_self_state, "autobiography": project_autobiography, "skills": project_skills,
    "goal_genesis": project_goal_genesis, "curriculum": project_curriculum, "planner": project_planner,
}

TRAJ = "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
NEUTRAL = "state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt"
CONSEQUENCE = "state/consequence_replanning_v1/consequence_model.pt"

SPECS: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        "recurrent_self_trajectory_v031", "self_state",
        apis=(API("continuous_brain.recurrent_self_trajectory_v1", "rollout"),
              API("continuous_brain.recurrent_self_trajectory_v1", "update_explicit_intention"),
              API("continuous_brain.recurrent_self_trajectory_v1", "active_goal_from_intention"),
              API("continuous_brain.recurrent_self_trajectory_v1", "trajectory_self_value")),
        loaders=(SharedLoader("continuous_brain.recurrent_self_trajectory_v1", "load_trajectory_model",
                              (TRAJ,), "trajectory_model"),),
        canonical=(CanonicalChannel("self_state.vector", (6,)),),
        notes="frozen recurrent self-trajectory model; rollout re-initializes its hidden state per call"),
    MechanismSpec(
        "autobiographical_discrepancy_v0322", "autobiography",
        apis=(API("continuous_brain.self_development_loop_v1", "reliability_from_memory"),),
        loaders=(SharedLoader("continuous_brain.autobiographical_self_discrepancy_neutral_v1", "load_neutral_model",
                              (NEUTRAL,), "neutral_model"),),
        canonical=(CanonicalChannel("autobiography.record", (27,)),
                   CanonicalChannel("autobiography.discrepancy", (2,)),
                   CanonicalChannel("autobiography.reliability", (2,))),
        notes="frozen neutral autobiographical self-model; the trajectory memory itself is built by the evaluator"),
    MechanismSpec(
        "event_consistent_self_development_v0341", "skills",
        apis=(API("continuous_brain.self_development_hysteresis_v1", "HistoryRelativeDevelopmentalController.initialize"),
              API("continuous_brain.self_development_hysteresis_v1", "HistoryRelativeDevelopmentalController.update"),
              API("continuous_brain.self_development_hysteresis_v1", "HistoryRelativeDevelopmentalController.strategy")),
        constructors=(SharedConstructor("continuous_brain.self_development_hysteresis_v1",
                                        "HistoryRelativeDevelopmentalController", "developmental_controller"),),
        canonical=(CanonicalChannel("skills.selected_option", (1,)),),
        notes="fixed-parameter hysteresis controller; update() mutates the phase in place (post-state is returned)"),
    MechanismSpec(
        "endogenous_goal_genesis_v0353", "goal_genesis",
        apis=(API("continuous_brain.endogenous_goal_genesis_v1", "EndogenousGoalGenerator.generate"),),
        constructors=(SharedConstructor("continuous_brain.endogenous_goal_genesis_v1", "EndogenousGoalGenerator",
                                        "goal_generator"),),
        canonical=(CanonicalChannel("goal_genesis.active_goal", (4,)),)),
    MechanismSpec(
        "recursive_autobiographical_curriculum_v0361", "curriculum",
        apis=(API("continuous_brain.recursive_autobiographical_curriculum_v1", "goal_is_advanced"),
              API("continuous_brain.recursive_autobiographical_curriculum_v1", "same_goal_semantics")),
        canonical=(CanonicalChannel("curriculum.next_goal", (2,)),)),
    MechanismSpec(
        "consequence_replanning_v0252", "planner",
        apis=(API("continuous_brain.consequence_replanning_v1", "plan_route"),
              API("continuous_brain.consequence_replanning_v1", "predict_transition"),
              API("continuous_brain.consequence_replanning_v1", "state_mismatch"),
              API("continuous_brain.consequence_replanning_cost_repair_v1", "plan_route_cost_consistent")),
        loaders=(SharedLoader("continuous_brain.consequence_replanning_v1", "load_consequence_model",
                              (CONSEQUENCE,), "consequence_model"),),
        canonical=(CanonicalChannel("planner.plan", (2,)),),
        notes="consequence model + mismatch-aware replanning; world_model slot left unbound (part of this host)"),
)

SPECS_BY_ID = {s.mechanism_id: s for s in SPECS}


def spec_digest(specs: tuple[MechanismSpec, ...] = SPECS) -> str:
    return hashlib.sha256(json.dumps([s.as_dict() for s in specs], sort_keys=True).encode()).hexdigest()
