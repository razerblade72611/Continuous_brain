"""I1 arm runner: runs inside a subprocess whose historical-core import hook is installed
(scripts/dev/i1_dev.py writes the bootstrap). One fresh UnifiedRuntime per individual.

Outputs (in out_dir):
  header.json           arm, seeds, frozen thresholds read from the legacy evaluator modules
                        (with their source SHA-256), the I1 manifest computed inside this arm
                        process (code by repo path, spec digest, codec, legacy pins, historical
                        core, environment), environment fingerprint
  lives.jsonl           one LifeRecord per individual (+ a per-life runtime summary)
  persist/<sid>/        PERSIST_SAVE: runtime snapshot (SnapshotStore) + encoded driver state
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import platform
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from continuous_brain_protocols import InterventionKind, InterventionSpec
from continuous_brain_snn.snapshot import SnapshotStore

from ..channels import ChannelDecl
from ..composition import boundary as boundary_mod
from ..composition import codec as codec_mod
from ..composition.boundary import Boundary, Census, MechanismHost, _resolve, install, request_channel
from ..composition.codec import decode, encode
from ..composition.spec import SPECS_BY_ID
from ..runtime import BindingPolicy, RuntimeConfig, UnifiedRuntime
from ..slots import DRIVER_OWNER
from . import hosts as hosts_mod
from . import life as life_mod
from .hosts import HOST_TYPES, state_channel
from .life import ARMS, AUTHORITY_ARMS, Client, Life

SELF_LINEAGE = ("recurrent_self_trajectory_v031", "autobiographical_discrepancy_v0322",
                "event_consistent_self_development_v0341", "endogenous_goal_genesis_v0353",
                "recursive_autobiographical_curriculum_v0361")
BOUND = "recurrent_self_trajectory_v031"        # the one dev6 MechanismHost; the other four are I1 stateful hosts
PLANNED_ARMS = ("ERASED", "DONOR", "ROLLBACK", *AUTHORITY_ARMS)

FORMAT = "cb-i1-arm/1"
HORIZON_SOURCE = ("evaluate_recursive_autobiographical_curriculum_v1", "MAX_FIRST_GOAL_EPISODES")
THRESHOLDS = {   # check -> (frozen module, constant)
    "first_goal_endogenous": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_FIRST_GOAL_GENESIS"),
    "first_goal_completes": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_FIRST_GOAL_COMPLETION"),
    "history_guides_initial_strategy": ("evaluate_self_development_hysteresis_v1", "MIN_HISTORY_GUIDED_STRATEGY"),
    "second_goal_authored": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_SECOND_GOAL_GENERATION"),
    "curriculum_advances": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_CURRICULUM_ADVANCE"),
    "erased_store_autobiography_no_goal": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_ERASED_NEUTRAL"),
    "donor_store_autobiography_flips_goal": ("evaluate_recursive_autobiographical_curriculum_v1",
                                             "MIN_DONOR_FIRST_GOAL_FLIP"),
    "rolled_back_autobiography_repeats_goal": ("evaluate_recursive_autobiographical_curriculum_v1",
                                               "MIN_FROZEN_HISTORY_REPEATS"),
    "updated_vs_rolled_back_diverges": ("evaluate_recursive_autobiographical_curriculum_v1",
                                        "MIN_UPDATED_VS_FROZEN_DIVERGENCE"),
    "persistence_bit_exact": ("evaluate_recursive_autobiographical_curriculum_v1", "MIN_PERSISTENCE"),
}


def load_legacy() -> dict[str, Any]:
    """Import the frozen legacy code (the historical-core hook must already be installed)."""
    abs_ = importlib.import_module("continuous_brain.autobiographical_self_discrepancy_v1")
    neu = importlib.import_module("continuous_brain.autobiographical_self_discrepancy_neutral_v1")
    loop = importlib.import_module("continuous_brain.self_development_loop_v1")
    hys = importlib.import_module("continuous_brain.self_development_hysteresis_v1")
    egg = importlib.import_module("continuous_brain.endogenous_goal_genesis_v1")
    cur = importlib.import_module("continuous_brain.recursive_autobiographical_curriculum_v1")
    traj = importlib.import_module("continuous_brain.recurrent_self_trajectory_v1")
    ev_egg = importlib.import_module("evaluate_endogenous_goal_genesis_v1")
    importlib.import_module("evaluate_recursive_autobiographical_curriculum_v1")
    importlib.import_module("evaluate_self_development_hysteresis_v1")
    return {
        "AutobiographicalTrajectoryMemory": abs_.AutobiographicalTrajectoryMemory,
        "load_neutral_model": neu.load_neutral_model, "reliability_from_memory": loop.reliability_from_memory,
        "DevelopmentalState": loop.DevelopmentalState,
        "HistoryRelativeDevelopmentalController": hys.HistoryRelativeDevelopmentalController,
        "STRATEGY_DIRECT": hys.STRATEGY_DIRECT, "STRATEGY_SCAFFOLD": hys.STRATEGY_SCAFFOLD,
        "EndogenousGoalGenerator": egg.EndogenousGoalGenerator, "GoalLifecycle": egg.GoalLifecycle,
        "SelfGeneratedGoal": egg.SelfGeneratedGoal,
        "goal_is_advanced": cur.goal_is_advanced, "same_goal_semantics": cur.same_goal_semantics,
        "load_trajectory_model": traj.load_trajectory_model,
        "TRAJECTORY_MODEL": ev_egg.TRAJECTORY_MODEL, "AUTOBIO_MODEL": ev_egg.AUTOBIO_MODEL,
        "build_bootstrap_memory": ev_egg.build_bootstrap_memory,
        "execute_development_episode": ev_egg.execute_development_episode,
        "initial_capabilities": ev_egg.initial_capabilities, "stable_primary": ev_egg.stable_primary,
        "actual_fragile_primary": ev_egg.actual_fragile_primary,
        "reliability_for_primary": ev_egg.reliability_for_primary,
    }


def frozen_constants(legacy_root: Path) -> dict:
    out = {}
    for name, (mod, const) in {**THRESHOLDS, "horizon": HORIZON_SOURCE}.items():
        m = sys.modules[mod]
        out[name] = {"value": getattr(m, const), "module": mod, "constant": const,
                     "source_sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "__dict__"):
        return {"__type__": type(o).__qualname__, **{k: _jsonable(v) for k, v in vars(o).items()}}
    return repr(o)


class Process:
    """Per-process setup.

    All five self-lineage specs are installed. Only v0.31 is bound as a dev6 MechanismHost; the
    APIs of the other four are redirected too, so any call to them goes through Boundary.call:
    from inside the owning I1 stateful host it runs the frozen code (counted as nested), and
    from the driver it reaches a stateful host that has no such API and is blocked. The driver
    therefore cannot run a private copy of a mechanism without the run recording it."""

    def __init__(self, legacy_root: Path, extra_modules: tuple = ()):
        self.root = legacy_root
        self.census = Census()
        self.b = Boundary()
        specs = tuple(SPECS_BY_ID[m] for m in SELF_LINEAGE)
        # evaluator-side originals, resolved before any redirect (used only outside the individual)
        importlib.import_module("continuous_brain.recursive_autobiographical_curriculum_v1")
        importlib.import_module("continuous_brain.autobiographical_self_discrepancy_neutral_v1")
        self.evaluator_side = {
            "same_goal_semantics": _resolve("continuous_brain.recursive_autobiographical_curriculum_v1",
                                            "same_goal_semantics"),
            "load_neutral_model": _resolve("continuous_brain.autobiographical_self_discrepancy_neutral_v1",
                                           "load_neutral_model")}
        hosts = install(self.b, specs, {BOUND}, legacy_root, self.census, extra_modules)
        self.L = load_legacy()      # after install, so every legacy name it captures is the redirected one
        v031 = SPECS_BY_ID[BOUND]
        self.v031_spec, self.v031_originals = v031, dict(hosts["self_state"].originals)
        # acquire the shared trajectory model once (verified bit-identical to a fresh load)
        self.L["trajectory_model"], _ = self.L["load_trajectory_model"](self.L["TRAJECTORY_MODEL"], "cpu")
        # the neutral autobiographical model is owned by the autobiography host (registered + watched)
        neutral, _ = self.evaluator_side["load_neutral_model"](self.L["AUTOBIO_MODEL"], "cpu")
        for p in neutral.parameters():
            p.requires_grad_(False)
        neutral.eval()
        self.b.instances.register("neutral_model:i1", neutral)
        self.b.watch_model("neutral_model:i1", neutral)
        self.legacy_for_hosts = {**self.L, "neutral_model": neutral,
                                 "controller": self.L["HistoryRelativeDevelopmentalController"](),
                                 "generator": self.L["EndogenousGoalGenerator"]()}

    def runtime(self, interventions: list[dict]) -> tuple[UnifiedRuntime, dict]:
        mh = MechanismHost(self.v031_spec, self.b)
        mh.originals = dict(self.v031_originals)
        hosts = {"self_state": mh}
        for T in HOST_TYPES:
            hosts[T.slot] = T(self.b, self.legacy_for_hosts)
        driver = tuple(ChannelDecl(request_channel(s), DRIVER_OWNER, None, "uint8", intervenable=True)
                       for s in sorted(hosts))
        rt = UnifiedRuntime(RuntimeConfig(binding_policy=BindingPolicy.TEST_FIXTURES, driver=driver))
        rt.bind_many(list(hosts.values()))
        for i, iv in enumerate(interventions):
            ref = None
            if iv.get("payload_b64") is not None:
                ref = f"i1_{i}"
                rt.interventions.add_payload(ref, np.frombuffer(base64.b64decode(iv["payload_b64"]), dtype=np.uint8))
            rt.interventions.register(InterventionSpec(InterventionKind(iv["kind"]), iv["target"], int(iv["start_tick"]),
                                                       iv.get("end_tick"), payload_ref=ref,
                                                       note=f"I1 {iv.get('why', '')}"))
        self.b.attach(rt, hosts)
        for name in ("nested", "unbound_calls", "content_flows", "forwards"):
            getattr(self.b, name).clear()
        self.b.calls = 0
        self.b._emitted.clear()
        self.b.first_responses.clear()
        return rt, hosts


def _tensor_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        t = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def frozen_digests(P: "Process") -> dict:
    """Digests of every process-level object the individual's mechanisms use but do not own in the store:
    the shared frozen models and the fixed-parameter controller and generator. They must be identical
    before and after every life, and identical across all lives and arms."""
    def obj(o):
        return hashlib.sha256(json.dumps({k: repr(v) for k, v in sorted(vars(o).items())}).encode()).hexdigest()
    H = P.legacy_for_hosts
    return {"trajectory_model": _tensor_digest(P.L["trajectory_model"]),
            "neutral_model": _tensor_digest(H["neutral_model"]),
            "controller": obj(H["controller"]), "generator": obj(H["generator"])}


def perturb_global_rng(sid: int, tick: int) -> None:
    """Re-seed torch, numpy and random with values unrelated to the life (RNG_PERTURBED control)."""
    v = int.from_bytes(hashlib.sha256(f"i1-rng-perturb:{sid}:{tick}".encode()).digest()[:8], "big")
    torch.manual_seed(v & 0x7FFFFFFFFFFFFFFF)
    np.random.seed(v & 0xFFFFFFFF)
    random.seed(v)


def _summary(rt: UnifiedRuntime, hosts: dict, b: Boundary) -> dict:
    reads = {(r["reader"], r["channel"]): r for r in rt.store.read_report()}
    return {"ticks": rt.tick + 1, "store_calls": b.calls,
            "host_blocked": {s: (h.blocked if isinstance(h, hosts_mod.StatefulHost) else h.stats.blocked)
                             for s, h in sorted(hosts.items())},
            "nested_by_route": dict(sorted(b.nested.items())),
            "model_forwards": dict(sorted(b.forwards.items())),
            "interventions": [{"tick": r.tick, "target": r.target, "kind": r.kind,
                               "before_sha256": r.before_sha256, "after_sha256": r.after_sha256}
                              for r in rt.interventions.records],
            "host_calls": {s: (dict(h.calls) if hasattr(h, "calls") and isinstance(h.calls, dict)
                               else dict(h.stats.apis)) for s, h in sorted(hosts.items())},
            "state_reads_nonempty": {s: reads.get((s, state_channel(s)), {}).get("nonempty_reads", 0)
                                     for s in sorted(hosts) if s != "self_state"},
            "interventions_applied": len(rt.interventions.records),
            "unbound_calls": dict(b.unbound_calls), "nested_calls": sum(b.nested.values()),
            "content_flows": dict(sorted(b.content_flows.items())),
            "config_hash": rt.config.config_hash()}


def run_arm(*, arm: str, sids: list[int], out_dir: str, legacy_root: str, plan_path: str | None,
            extra_modules: tuple = ()) -> int:
    if arm not in ARMS + ("LEGACY_REF",):
        raise ValueError(arm)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(legacy_root).resolve()
    plan = json.loads(Path(plan_path).read_text()) if plan_path else {}
    if arm == "LEGACY_REF":
        return _legacy_reference(sids, out, root)
    P = Process(root, extra_modules)
    horizon = int(getattr(sys.modules[HORIZON_SOURCE[0]], HORIZON_SOURCE[1]))
    from .runner import manifest            # computed inside the arm process, from the files it runs
    man = manifest(torch.get_num_threads())
    header = {"format": FORMAT, "status": "ENGINEERING_ONLY", "arm": arm, "sids": [sids[0], sids[-1]],
              "manifest_sha256": man["manifest_sha256"], "code_sha256": man["code_sha256"],
              "n": len(sids), "horizon": horizon, "frozen_constants": frozen_constants(root),
              "sources": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (codec_mod, boundary_mod, hosts_mod, life_mod, sys.modules[__name__])},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "threads": torch.get_num_threads(), "platform": platform.platform()}}
    (out / "header.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    with open(out / "lives.jsonl", "w") as f:
        for sid in sids:
            entry = plan.get(str(sid), [])
            ivs = entry["interventions"] if isinstance(entry, dict) else entry
            compare = entry.get("compare") if isinstance(entry, dict) else None
            row: dict[str, Any] = {"sid": sid, "arm": arm}
            try:
                if arm in PLANNED_ARMS and not ivs:
                    row["skipped"] = "no intervention planned for this individual"
                    f.write(json.dumps(row) + "\n"); continue
                rt, hosts = P.runtime(ivs)
                frozen_before = frozen_digests(P)
                life = Life(P.L, Client(P.b), rt, sid, arm, horizon)
                if arm == "RNG_PERTURBED":
                    def hook(t, life=life, sid=sid):
                        life.on_tick(t)
                        perturb_global_rng(sid, t)
                    perturb_global_rng(sid, -1)
                    P.b.after_tick = hook
                else:
                    P.b.after_tick = life.on_tick
                if arm == "PERSIST_RESUME":
                    d = out.parent / "PERSIST_SAVE" / "persist" / str(sid)
                    store = SnapshotStore(d / "snapshots")
                    rt.restore(store, store.latest("i1"))
                    life.load_driver_state(decode((d / "driver_state.bin").read_bytes(), P.b.instances))
                    life._last_digest = rt.store.digest()
                    life.run_second_half()
                else:
                    authored = life.run_first_half()
                    if arm == "PERSIST_SAVE":
                        d = out / "persist" / str(sid)
                        d.mkdir(parents=True, exist_ok=True)
                        snap = rt.save(SnapshotStore(d / "snapshots"), label="i1")
                        (d / "driver_state.bin").write_bytes(encode(life.driver_state(), P.b.instances).data)
                        row["snapshot"] = str(snap.name)
                        life.finish()
                    elif authored and arm in AUTHORITY_ARMS:
                        try:
                            life.run_second_half()
                        except Exception as e:     # the mechanism's own error on erased state = authority shown
                            row["aborted"] = f"{type(e).__name__}: {e}"
                            life.finish()
                    elif authored:
                        life.run_second_half()
                    else:
                        life.finish()
                if compare is not None:      # frozen v0.36 comparison of the updated vs rolled-back second goal
                    ng = compare.get("normal_second_goal")
                    normal_goal = None if ng is None else P.L["SelfGeneratedGoal"](**ng)
                    row["updated_vs_rolled_back_same_semantics"] = bool(
                        P.evaluator_side["same_goal_semantics"](normal_goal, life.rec.second_goal))
                row["record"] = _jsonable(life.rec.__dict__)
                row["runtime"] = _summary(rt, hosts, P.b)
                row["runtime"]["frozen_before"], row["runtime"]["frozen_after"] = frozen_before, frozen_digests(P)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc()[-4000:]
            finally:
                P.b.after_tick = None
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
    census = P.census.as_dict()
    (out / "census.json").write_text(json.dumps(census, indent=2, sort_keys=True, default=str) + "\n")
    return 0


def _legacy_reference(sids: list[int], out: Path, root: Path) -> int:
    """Plain frozen v0.36 complete_first_goal per individual (no runtime, no redirects)."""
    L = load_legacy()
    rac = sys.modules["evaluate_recursive_autobiographical_curriculum_v1"]
    traj, _ = L["load_trajectory_model"](L["TRAJECTORY_MODEL"], "cpu")
    neutral, _ = L["load_neutral_model"](L["AUTOBIO_MODEL"], "cpu")
    for m in (traj, neutral):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    gen = L["EndogenousGoalGenerator"]()
    with open(out / "lives.jsonl", "w") as f:
        for sid in sids:
            r = rac.complete_first_goal(traj, neutral, gen, sid)
            row = {"sid": sid, "arm": "LEGACY_REF"}
            if r is None:
                row["first_goal"] = None
            else:
                row.update({"first_goal": _jsonable(r["first_goal"]), "first_completed": bool(r["first_completed"]),
                            "memory_sha256": hashlib.sha256(encode(r["memory"]).data).hexdigest(),
                            "n_records": len(r["memory"]),
                            "capability_after_first": [float(r["state"].capability_a), float(r["state"].capability_b)]})
            f.write(json.dumps(row, sort_keys=True) + "\n")
    from .runner import manifest
    man = manifest(torch.get_num_threads())
    (out / "header.json").write_text(json.dumps({"format": FORMAT, "arm": "LEGACY_REF", "n": len(sids),
                                                 "manifest_sha256": man["manifest_sha256"],
                                                 "code_sha256": man["code_sha256"]}) + "\n")
    return 0
