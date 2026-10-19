"""I1 life driver: one individual's life inside one UnifiedRuntime (ENGINEERING_ONLY until approved).

The driver plays the environment. It runs the frozen v0.33 environment
functions (build_bootstrap_memory, initial_capabilities, execute_development_episode)
and calls the individual's mechanisms only through the store-mediated hosts.
The self-predictions inside those functions (v0.31 rollout and intention
updates) are routed to the v0.31 host by the dev6 redirects.

Life (the orchestration under test; everything it calls is frozen legacy code):
  birth      bootstrap history -> autobiography.imprint; capabilities (environment)
  init       reliability -> developmental phases initialized (v0.34.1 controller);
             first self-authored goal (v0.35.3 generator); curriculum observes it
  interrupt  one stable-commitment DIRECT episode (the admitted v0.35 interruption)
  pursue 1   up to H episodes on the active goal's commitment. NORMAL: the v0.34.1
             controller chooses the strategy from the current autobiographical
             reliability. FIXED_SCAFFOLD: SCAFFOLD, as in the frozen v0.35/v0.36
             drivers. The goal completes when its target reliability is met.
  second     the driver asks the goal host for the store-held lifecycle status; only if the
             store records a completed goal and no active goal is a second self-authored goal
             requested from the updated autobiography; curriculum observes it (advanced?)
  pursue 2   up to H episodes on the second goal (cycle indices 100+k, as v0.36; the
             completion cycle carries the same 100+k label)

Arms (each a fresh runtime per individual; interventions are store-level and
registered before the first tick, at ticks taken from the NORMAL run, which is
replayed bit-exactly up to that tick):
  NORMAL, FIXED_SCAFFOLD,
  ERASED    ERASE autobiography.state right after the birth imprint
  DONOR     REPLACE autobiography.state right after imprint with the paired
            individual's birth autobiography
  ROLLBACK  REPLACE autobiography.state just before the second-goal decision with
            this individual's own birth autobiography
  PERSIST   save the runtime (SnapshotStore) and the driver's environment state just
            before the second-goal decision; a new process restores and finishes
  RNG_PERTURBED
            (validity control) the NORMAL life with the process's global random generators
            (torch, numpy, random) re-seeded with unrelated values before birth and after every
            tick. It must reproduce NORMAL bit-exactly: no behaviour may depend on global
            randomness outside the store-held state and the per-episode seeded environment.
  AUTHORITY_SKILLS / AUTHORITY_GOAL / AUTHORITY_CURRICULUM
            (validity controls) ERASE skills.state / goal_genesis.state / curriculum.state just
            before the second-goal decision. If the store is authoritative the second half must
            change (or abort with the mechanism's own error); if a host kept a private copy it
            would not.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..composition.boundary import Boundary, MechanismHost, RouteBlocked
from ..composition.codec import encode
from .hosts import state_channel

AUTHORITY_ARMS = {"AUTHORITY_SKILLS": "skills", "AUTHORITY_GOAL": "goal_genesis", "AUTHORITY_CURRICULUM": "curriculum"}
ARMS = ("NORMAL", "FIXED_SCAFFOLD", "ERASED", "DONOR", "ROLLBACK", "PERSIST_SAVE", "PERSIST_RESUME", "RNG_PERTURBED",
        *AUTHORITY_ARMS)


def _never(*_a, **_k):
    raise RouteBlocked("stateful host APIs are never called nested")


class Client:
    """Driver-side calls to the stateful hosts (each call is one runtime tick through the store)."""

    def __init__(self, boundary: Boundary):
        self.b = boundary

    def __call__(self, slot: str, api: str, *args):
        return self.b.call(slot, api, _never, args, {})


@dataclass
class LifeRecord:
    sid: int
    arm: str
    marks: dict = field(default_factory=dict)          # name -> tick
    digests: dict = field(default_factory=dict)        # name -> store digest at that tick
    birth_state_b64: str | None = None
    evaluator_fragile_primary: int | None = None      # environment/evaluator-side label, never shown to hosts
    evaluator_stable_primary: int | None = None
    first_goal: Any = None
    initial_phase_strategies: Any = None
    first_strategies: list = field(default_factory=list)
    first_completed: bool = False
    first_completion_cycle: Any = None
    first_episodes: int = 0
    first_series: list = field(default_factory=list)   # per check k: goal-commitment reliability, capability, episode
    capability_after_first: Any = None                 # environment capabilities at the second-goal decision
    first_goal_store_status: Any = None                # the goal host's store-held lifecycle at that decision
    first_completed_store: bool | None = None
    second_goal: Any = None
    second_observation: Any = None
    second_strategies: list = field(default_factory=list)
    second_completed: bool = False
    second_completion_cycle: Any = None
    second_episodes: int = 0
    second_series: list = field(default_factory=list)
    capability_start: Any = None
    capability_end: Any = None
    final_tick: int | None = None
    final_store_digest: str | None = None
    final_autobiography_sha256: str | None = None
    life_trace_sha256: str | None = None


class Life:
    def __init__(self, L: dict, client: Client, runtime, sid: int, arm: str, horizon: int):
        self.L, self.c, self.rt, self.sid, self.arm, self.H = L, client, runtime, int(sid), arm, int(horizon)
        self.rec = LifeRecord(self.sid, arm)
        self.env = None
        self.traj = None
        self._trace_hex = hashlib.sha256(b"cb-i1-life-trace/1").hexdigest()
        self._last_digest: str | None = None
        self._prev_digest: str | None = None

    def on_tick(self, tick: int) -> None:
        """Boundary hook: keep the store digest after the latest and the previous tick."""
        self._prev_digest, self._last_digest = self._last_digest, self.rt.store.digest()

    # ---------------------------------------------------------------- helpers
    def _note(self, *items):
        """Chained life trace over what the driver observed (resumable from its hex digest)."""
        self._trace_hex = hashlib.sha256(bytes.fromhex(self._trace_hex) + encode(list(items)).data).hexdigest()

    def _mark(self, name: str):
        """Record the tick T of the call just made, the store digest after T and after T-1
        (a store intervention scheduled in window [T, T+1) acts after the host's write at T, so
        the digest after T-1 is what a replayed arm must reproduce exactly)."""
        self.rec.marks[name] = self.rt.tick
        self.rec.digests[name] = self._last_digest
        self.rec.digests[name + "_pre"] = self._prev_digest
        self.rec.digests[name + "_autobiography"] = hashlib.sha256(
            self.rt.store.read(state_channel("autobiography")).tobytes()).hexdigest()

    def _rel(self):
        return self.c("autobiography", "reliability")

    def _observe(self, series: list, k: int, rel, primary: int) -> dict:
        """Driver-side observation at check k (no store call, no RNG use): the goal commitment's
        self-model reliability (from the reliability the driver just received) and the environment's
        true capability on that commitment."""
        entry = {"k": k, "reliability": float(self.L["reliability_for_primary"](rel, primary)),
                 "capability": float(self.env.capability_for(primary))}
        series.append(entry)
        return entry

    def _pursue(self, cycle_base: int, strategies: list, series: list, authored_goal) -> tuple[bool, Any, int]:
        L = self.L
        episodes = 0
        target_primary = int(authored_goal.primary)        # an authored field: fixed for the goal's lifetime
        for k in range(self.H):
            rel = self._rel()
            entry = self._observe(series, k, rel, target_primary)
            if self.c("goal_genesis", "check_target", rel, cycle_base + k):
                self._note("complete", cycle_base + k)
                return True, cycle_base + k, episodes
            goal = self.c("goal_genesis", "active")
            primary = int(goal.primary)
            if self.arm == "FIXED_SCAFFOLD":
                strategy = int(L["STRATEGY_SCAFFOLD"])
            else:
                strategy = int(self.c("skills", "choose", primary, L["reliability_for_primary"](rel, primary)))
            strategies.append(strategy)
            record, metrics = L["execute_development_episode"](self.traj, self.sid, cycle_base + k, primary, strategy,
                                                                self.env)
            entry.update(strategy=strategy, work_successes=int(metrics["work_successes"]),
                         work_failures=int(metrics["work_failures"]))
            self.c("autobiography", "append", record)
            self.env.cycle += 1
            episodes += 1
            self._note("episode", k, primary, strategy)
        rel = self._rel()
        self._observe(series, self.H, rel, target_primary)
        if self.c("goal_genesis", "check_target", rel, cycle_base + self.H):
            self._note("complete", cycle_base + self.H)
            return True, cycle_base + self.H, episodes
        return False, None, episodes

    # ---------------------------------------------------------------- the life
    def run_first_half(self) -> bool:
        """Birth, initialization, interruption and first goal. Returns False if no goal was authored."""
        L, sid = self.L, self.sid
        self.traj = L["trajectory_model"]          # the runtime-owned shared instance (acquired once per process)
        bootstrap = L["build_bootstrap_memory"](self.traj, sid)
        self.c("autobiography", "imprint", bootstrap)
        self._mark("imprint")
        raw = self.rt.store.read(state_channel("autobiography")).tobytes()
        self.rec.birth_state_b64 = base64.b64encode(raw).decode()
        self.rec.evaluator_fragile_primary = int(L["actual_fragile_primary"](sid))
        self.rec.evaluator_stable_primary = int(L["stable_primary"](sid))
        cap_a, cap_b = L["initial_capabilities"](sid)
        self.env = L["DevelopmentalState"](capability_a=cap_a, capability_b=cap_b, cycle=0)
        self.rec.capability_start = [float(cap_a), float(cap_b)]
        rel = self._rel()
        self.rec.initial_phase_strategies = self.c("skills", "initialize", rel)
        goal = self.c("goal_genesis", "generate_if_idle", rel, 0)
        self.rec.first_goal = goal
        self._note("first_goal", goal)
        if goal is None:
            return False
        self.c("curriculum", "observe", goal)
        record, _ = L["execute_development_episode"](self.traj, sid, -1, L["stable_primary"](sid),
                                                      int(L["STRATEGY_DIRECT"]), self.env)
        self.c("autobiography", "append", record)
        self._note("interrupt")
        done, cyc, n = self._pursue(0, self.rec.first_strategies, self.rec.first_series, goal)
        self.rec.first_completed, self.rec.first_completion_cycle, self.rec.first_episodes = done, cyc, n
        self.rec.capability_after_first = [float(self.env.capability_a), float(self.env.capability_b)]
        self._mark("before_second")
        return True

    def run_second_half(self) -> None:
        """The second-goal decision is gated on the store-held goal lifecycle, not on the driver's
        own record of the first half."""
        st = self.c("goal_genesis", "status")
        self.rec.first_goal_store_status = st
        self.rec.first_completed_store = bool(len(st["completed_goals"]) >= 1 and st["active"] is None)
        self._note("status", st["generation"], len(st["completed_goals"]), st["active"] is None)
        if self.rec.first_completed_store:
            rel = self._rel()
            goal2 = self.c("goal_genesis", "generate_if_idle", rel, self.env.cycle)
            self.rec.second_goal = goal2
            self.rec.second_observation = self.c("curriculum", "observe", goal2)
            self._note("second_goal", goal2)
            if goal2 is not None and self.arm != "ROLLBACK":
                done, cyc, n = self._pursue(100, self.rec.second_strategies, self.rec.second_series, goal2)
                self.rec.second_completed, self.rec.second_completion_cycle, self.rec.second_episodes = done, cyc, n
        self.finish()

    def finish(self) -> None:
        self.rec.capability_end = [float(self.env.capability_a), float(self.env.capability_b)] if self.env else None
        self.rec.final_tick = self.rt.tick
        self.rec.final_store_digest = self.rt.store.digest()
        self.rec.final_autobiography_sha256 = hashlib.sha256(
            self.rt.store.read(state_channel("autobiography")).tobytes()).hexdigest()
        self.rec.life_trace_sha256 = self._trace_hex

    # persistence: the driver's environment-side state at the resume point
    def driver_state(self) -> dict:
        return {"env": self.env, "rec": dict(self.rec.__dict__), "trace": self._trace_hex,
                "last_digest": self._last_digest, "prev_digest": self._prev_digest}

    def load_driver_state(self, st: dict) -> None:
        self.env = st["env"]
        for k, v in st["rec"].items():
            setattr(self.rec, k, v)
        self.rec.arm = self.arm
        self._trace_hex = st["trace"]
        self._last_digest, self._prev_digest = st["last_digest"], st["prev_digest"]
        self.traj = self.L["trajectory_model"]
