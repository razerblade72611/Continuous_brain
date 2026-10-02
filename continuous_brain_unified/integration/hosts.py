"""Stateful hosts: one persistent individual whose state is held by the canonical store.

Each host keeps its mechanism's persistent state in the blob channel <slot>.state.
At every tick it reads that channel as of the start of the tick (a delayed read,
so a store intervention applied after its previous write is what it sees), and
then:
  * idle tick: writes the same bytes back unchanged (the state persists as stored)
  * request tick: decodes the state, executes one API call with the frozen
    legacy code, encodes the new state and the result, and writes both

No host keeps mechanism state in Python between ticks. Whatever the store holds
is the individual's state, so ERASE / REPLACE / FREEZE / DONOR interventions on
<slot>.state change the individual itself, and runtime save/restore persists it.

Hosts and the legacy code each calls (all hash-pinned in the G0 dependency closure):
  autobiography  (v0.32.2)  AutobiographicalTrajectoryMemory; reliability_from_memory
                            with the frozen neutral autobiographical self-model
  skills         (v0.34.1)  HistoryRelativeDevelopmentalController.initialize / update /
                            strategy over per-commitment CommitmentDevelopmentPhase state
  goal_genesis   (v0.35.3)  EndogenousGoalGenerator.generate; GoalLifecycle begin /
                            target_met / complete_active
  curriculum     (v0.36.1)  goal_is_advanced; same_goal_semantics over the authored goals
The v0.31 recurrent self-trajectory model stays a dev6 MechanismHost (stateless API).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from ..channels import ChannelDecl
from ..components import Component, ComponentProvenance, ProvenanceKind, TickContext
from ..composition.boundary import request_channel, response_channel
from ..composition.codec import CodecError, decode, encode


def state_channel(slot: str) -> str:
    return f"{slot}.state"


class StatefulHost(Component):
    """Base: API calls over store-held state. Subclasses define initial_state(), apis, project()."""

    mechanism_id: str = ""
    canonical: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __init__(self, boundary, legacy: dict[str, Any]):
        self.name = f"stateful-host:{self.mechanism_id}"
        self.reads = (request_channel(self.slot),)
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (
            ChannelDecl(response_channel(self.slot), self.slot, None, "uint8", intervenable=True,
                        description=f"encoded responses of {self.mechanism_id}"),
            ChannelDecl(state_channel(self.slot), self.slot, None, "uint8", intervenable=True,
                        description=f"the individual's {self.slot} state (authoritative; encoded)"),
            *(ChannelDecl(n, self.slot, shape, "float64", neutral=-1.0,
                          description=f"canonical projection of {self.mechanism_id} state")
              for n, shape in self.canonical))
        self.boundary = boundary
        self.legacy = legacy
        self.calls: dict[str, int] = {}
        self.blocked = 0
        self._held = {n: np.full(shape, -1.0) for n, shape in self.canonical}
        self._proj_sha: str | None = None

    def _neutral_projection(self) -> dict[str, np.ndarray]:
        return {n: np.full(shape, -1.0) for n, shape in self.canonical}

    def _refresh_projection(self, raw_state: bytes) -> None:
        """Canonical channels are a pure function of the stored state bytes: recomputed whenever the
        bytes change (a write or a store intervention), never carried over from an earlier state."""
        sha = hashlib.sha256(raw_state).hexdigest()
        if sha == self._proj_sha:
            return
        self._proj_sha = sha
        proj = self._neutral_projection()
        b = self.boundary
        b.in_host = self.slot          # projection runs inside this host: its model forwards are host-side
        try:
            proj.update(self.project(self._decode_state(raw_state)))
        except CodecError:
            pass
        finally:
            b.in_host = None
        self._held = {k: np.asarray(v, dtype=np.float64) for k, v in proj.items()}

    # subclasses ---------------------------------------------------------------
    def initial_state(self) -> Any:            # the state an erased (empty) channel decodes to
        raise NotImplementedError

    def apis(self) -> dict[str, Callable]:     # name -> fn(state, *args) -> (new_state, result)
        raise NotImplementedError

    def project(self, state: Any) -> dict[str, np.ndarray]:
        return {}

    # component ----------------------------------------------------------------
    def provenance(self) -> ComponentProvenance:
        return ComponentProvenance(ProvenanceKind.TEST_FIXTURE, {
            "mechanism_id": self.mechanism_id, "engineering_only": True, "stateful": True,
            "note": "frozen legacy code over store-held state; verified-manifest binding required for science"})

    def state_dict(self) -> dict:
        return {"held": {k: np.array(v) for k, v in self._held.items()}}

    def load_state_dict(self, state: dict) -> None:
        held = state.get("held", {})
        if set(held) != set(self._held):
            raise ValueError(f"{self.name}: checkpoint holds canonical channels {sorted(held)}")
        self._held = {k: np.array(v, dtype=np.float64) for k, v in held.items()}

    def _decode_state(self, raw: bytes) -> Any:
        return self.initial_state() if len(raw) == 0 else decode(raw, self.boundary.instances)

    def step(self, ctx: TickContext):
        raw_state = ctx.delayed[state_channel(self.slot)].tobytes()
        req = ctx.inputs[request_channel(self.slot)]
        out_state, resp = raw_state, b""
        if req.size:
            b = self.boundary
            b.in_host = self.slot
            try:
                call = decode(req.tobytes(), b.instances)
                name, args = call["api"], list(call["args"])
                fn = self.apis().get(name)
                if fn is None:
                    raise CodecError(f"{name!r} is not an API of {self.mechanism_id}")
                state = self._decode_state(raw_state)
                new_state, result = fn(state, *args, **dict(call["kwargs"]))
                out_state = encode(new_state, b.instances).data
                resp = encode({"status": "ok", "result": result, "result_ref": None,
                               "post_args": {}, "post_kwargs": {}}, b.instances).data
                self.calls[name] = self.calls.get(name, 0) + 1
            except CodecError as e:
                self.blocked += 1
                resp = encode({"status": "blocked", "message": f"{self.mechanism_id}: {e}"}).data
            except Exception as e:            # a legacy exception travels back to the driver
                resp = encode({"status": "error", "exc_module": type(e).__module__,
                               "exc_type": type(e).__qualname__, "exc_args": [str(e)]}).data
            finally:
                b.in_host = None
        self._refresh_projection(out_state)
        return {response_channel(self.slot): resp, state_channel(self.slot): out_state,
                **{k: v for k, v in self._held.items()}}


class AutobiographyHost(StatefulHost):
    slot = "autobiography"
    mechanism_id = "autobiographical_discrepancy_v0322"
    canonical = (("autobiography.record", (27,)), ("autobiography.discrepancy", (2,)),
                 ("autobiography.reliability", (2,)))

    def initial_state(self):
        return self.legacy["AutobiographicalTrajectoryMemory"]()

    def apis(self):
        return {"imprint": self._imprint, "append": self._append, "reliability": self._reliability}

    def _imprint(self, state, memory):
        if len(state):
            raise RuntimeError("imprint is only allowed on an empty autobiography (birth)")
        return memory, len(memory)

    def _append(self, state, record):
        state.append(record)
        return state, len(state)

    def _reliability(self, state):
        rel = self.legacy["reliability_from_memory"](self.legacy["neutral_model"], state)
        return state, rel

    def project(self, state):
        recs = getattr(state, "records", [])
        if not recs:
            return {}
        feat = recs[-1].feature()
        rel = self.legacy["reliability_from_memory"](self.legacy["neutral_model"], state)
        return {"autobiography.record": feat.double().numpy(), "autobiography.discrepancy": feat[-2:].double().numpy(),
                "autobiography.reliability": rel.detach().double().numpy().reshape(2)}


class DevelopmentHost(StatefulHost):
    slot = "skills"
    mechanism_id = "event_consistent_self_development_v0341"
    canonical = (("skills.selected_option", (1,)),)

    def initial_state(self):
        return {}

    def apis(self):
        return {"initialize": self._initialize, "choose": self._choose}

    def _initialize(self, state, rel):
        if state:
            raise RuntimeError("developmental phases are initialized once (birth)")
        c = self.legacy["controller"]
        # the body of the frozen initialize_phases(), with the reliability supplied by the
        # autobiography host instead of recomputed from a private memory copy
        phases = {1: c.initialize(float(rel[0].item())), 2: c.initialize(float(rel[1].item()))}
        return phases, {p: int(c.strategy(ph)) for p, ph in phases.items()}

    def _choose(self, state, primary, current_reliability):
        if not state:
            raise RuntimeError("developmental phases are not initialized")
        c = self.legacy["controller"]
        state[int(primary)] = c.update(state[int(primary)], float(current_reliability))
        state["last_selected"] = int(c.strategy(state[int(primary)]))
        return state, state["last_selected"]

    def project(self, state):
        if not state or "last_selected" not in state:
            return {}
        return {"skills.selected_option": np.array([float(state["last_selected"])])}


class GoalHost(StatefulHost):
    slot = "goal_genesis"
    mechanism_id = "endogenous_goal_genesis_v0353"
    canonical = (("goal_genesis.active_goal", (4,)),)

    def initial_state(self):
        return self.legacy["GoalLifecycle"]()

    def apis(self):
        return {"generate_if_idle": self._generate, "check_target": self._check, "active": self._active,
                "status": self._status}

    def _status(self, state):
        return state, {"generation": int(state.generation), "active": state.active_goal,
                       "completed_goals": list(state.completed_goals)}

    def _generate(self, state, rel, cycle):
        if state.active_goal is not None:
            return state, None
        goal = self.legacy["generator"].generate(rel, generation=state.generation, cycle=int(cycle))
        if goal is not None:
            state.begin(goal)
        return state, goal

    def _check(self, state, rel, cycle):
        g = state.active_goal
        if g is None:
            return state, False
        if g.target_met(self.legacy["reliability_for_primary"](rel, g.primary)):
            state.complete_active(int(cycle))
            return state, True
        return state, False

    def _active(self, state):
        return state, state.active_goal

    def project(self, state):
        g = state.active_goal
        if g is None:
            return {"goal_genesis.active_goal": np.array([0.0, -1.0, -1.0, -1.0])}
        return {"goal_genesis.active_goal": np.array([1.0, float(g.primary), float(g.baseline_reliability),
                                                       float(g.target_reliability)])}


class CurriculumHost(StatefulHost):
    slot = "curriculum"
    mechanism_id = "recursive_autobiographical_curriculum_v0361"
    canonical = (("curriculum.next_goal", (2,)),)

    def initial_state(self):
        return {"goals": [], "last_advanced": None}

    def apis(self):
        return {"observe": self._observe}

    def _observe(self, state, goal):
        """Compare an authored goal with the previous authored goal (frozen v0.36 semantics);
        only authored goals (not None) join the curriculum history."""
        prev = state["goals"][-1] if state["goals"] else None
        out = {"previous_exists": prev is not None,
               "advanced": bool(self.legacy["goal_is_advanced"](prev, goal)) if prev is not None else None,
               "same_as_previous": bool(self.legacy["same_goal_semantics"](prev, goal)) if prev is not None else None}
        if goal is not None:
            state["goals"].append(goal)
        state["last_advanced"] = out["advanced"]
        return state, out

    def project(self, state):
        if not state["goals"]:
            return {}
        return {"curriculum.next_goal": np.array([1.0 if state.get("last_advanced") else 0.0,
                                                  float(len(state["goals"]))])}


HOST_TYPES = (AutobiographyHost, DevelopmentHost, GoalHost, CurriculumHost)


def state_digest_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
