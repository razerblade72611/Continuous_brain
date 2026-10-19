#!/usr/bin/env python3
"""CB2-H1 Phase 0 (DAV-97, protocol docs/explore/CB2_H1_THREAT_ESCAPE.md rev 2): census and headroom, measurement only.

  run      python scripts/explore/cb2_h1_phase0.py run --individuals <dir holding S2b's cb2-ind-0001> \\
               --seeds 77001-77120 --workers 2 --out runs/explore/cb2/h1/phase0
  report   python scripts/explore/cb2_h1_phase0.py report --out runs/explore/cb2/h1/phase0

For each development world (fresh; never a sealed world):
  pass 1  S2b lives one full life (greedy, unchanged stack); every step logs the individual's own perception-based
          eligibility (threat radii, felt pain, enclosure, awake) and the evaluator's privileged truth (hits, true
          threats, health) -- the latter never reaches the agent.
  pass 2  the life is replayed deterministically to the first hands-attributable onset (checked action for action);
          the individual's state and the world are copied there, and each script of the fixed family is branched
          to the end of life (censored 1,000 steps after the onset).  The control branch (S2b's own hands) must
          reproduce pass 1 action for action (fidelity).
Scripts act only on what S2b itself has (its perceived objects, believed map, body sense) through a tester hook in
the skill host (provenance 'phase0-tester-override'); the *choice* of the best script is made in hindsight from
privileged outcomes and is an optimistic ceiling.  Results are appended per world to worlds.jsonl (resumable).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import RESERVED, git_rev, vision  # noqa: E402
from cb2_evaluate import load_individual  # noqa: E402

Z_R, S_R, CLEAR, TIMEOUT, MERGE, POST = 3, 4, 5, 40, 5, 50       # protocol rev 2, section 2
BRANCH_CAP = 1000
WALK = {2, 4, 5}                                                 # grass, path, sand (vision MATERIALS ids)
WATER, LAVA = 1, 7
DIRS = {(-1, 0): "move_left", (1, 0): "move_right", (0, -1): "move_up", (0, 1): "move_down"}
SCRIPTS = ("control", "flee4", "flee8", "flee16", "water", "fight", "hold")


# -- perception-side helpers (only what the individual itself has) -----------------------------------------------
def _man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def perceived(memory) -> tuple[list, list]:
    zs = [tuple(o["pos"]) for o in memory.get("objects", []) if o["kind"] == "zombie"]
    ss = [tuple(o["pos"]) for o in memory.get("objects", []) if o["kind"] == "skeleton"]
    return zs, ss


def enclosed(memory, pos) -> bool:
    """At most 2 walkable tiles (own included) reachable within 3 steps on its believed map (unknown = open)."""
    m = memory["map"]
    seen, q = {tuple(pos)}, deque([(tuple(pos), 0)])
    while q:
        p, d = q.popleft()
        if d == 3:
            continue
        for u in DIRS:
            n = (p[0] + u[0], p[1] + u[1])
            if n in seen or not (0 <= n[0] < m.shape[0] and 0 <= n[1] < m.shape[1]):
                continue
            if int(m[n]) in WALK or int(m[n]) < 0:
                seen.add(n)
                q.append((n, d + 1))
                if len(seen) > 2:
                    return False
    return True


def threat_near(memory, pos) -> bool:
    zs, ss = perceived(memory)
    return any(_man(z, pos) <= Z_R for z in zs) or any(_man(s, pos) <= S_R for s in ss)


def _walkable(memory, n, occupied) -> bool:
    m = memory["map"]
    if not (0 <= n[0] < m.shape[0] and 0 <= n[1] < m.shape[1]) or n in occupied:
        return False
    return int(m[n]) in WALK


class Script:
    """One fixed escape script, consulted by the host hook each step after the onset (on S2b's own beliefs)."""

    def __init__(self, kind: str, t0: int):
        self.kind, self.t0 = kind, t0
        self.limit = {"flee4": 4, "flee8": 8, "flee16": 16, "water": 16, "fight": 16, "hold": 8}.get(kind, 0)
        self.used = 0

    def __call__(self, memory, body, step, name):
        if self.kind == "control" or step < self.t0 or self.used >= self.limit:
            return None
        pos = tuple(body["player_pos"])
        zs, ss = perceived(memory)
        if not threat_near(memory, pos) or body.get("sleeping"):
            return None
        occupied = {tuple(o["pos"]) for o in memory.get("objects", [])}
        act = None
        if self.kind.startswith("flee"):
            th = zs + ss
            best = min(_man(t, pos) for t in th)
            for u, mv in DIRS.items():
                n = (pos[0] + u[0], pos[1] + u[1])
                if _walkable(memory, n, occupied) and min(_man(t, n) for t in th) > best:
                    best, act = min(_man(t, n) for t in th), mv
        elif self.kind == "hold":
            act = "noop"
        elif self.kind == "fight":
            adj = [z for z in zs if _man(z, pos) == 1]
            if adj:
                u = (adj[0][0] - pos[0], adj[0][1] - pos[1])
                act = "do" if tuple(body.get("facing", (0, 0))) == u else DIRS[u]
            elif zs:
                z = min(zs, key=lambda z: _man(z, pos))
                for u, mv in DIRS.items():
                    n = (pos[0] + u[0], pos[1] + u[1])
                    if _walkable(memory, n, occupied - {z}) and _man(z, n) < _man(z, pos):
                        act = mv
                        break
        elif self.kind == "water":
            act = self._toward_water(memory, pos, occupied)
        if act is None:
            return None
        if act in DIRS.values():                                     # the executor's lava check
            u = next(k for k, v in DIRS.items() if v == act)
            n = (pos[0] + u[0], pos[1] + u[1])
            m = memory["map"]
            if 0 <= n[0] < m.shape[0] and 0 <= n[1] < m.shape[1] and int(m[n]) == LAVA:
                act = "noop"
        self.used += 1
        return act

    @staticmethod
    def _toward_water(memory, pos, occupied):
        m = memory["map"]
        prev, q = {pos: None}, deque([pos])
        while q:
            p = q.popleft()
            if any(0 <= p[0] + u[0] < m.shape[0] and 0 <= p[1] + u[1] < m.shape[1]
                   and int(m[p[0] + u[0], p[1] + u[1]]) == WATER for u in DIRS):
                if p == pos:
                    return None
                while prev[p] != pos:
                    p = prev[p]
                return DIRS[(p[0] - pos[0], p[1] - pos[1])]
            if len(prev) > 600:
                return None
            for u in DIRS:
                n = (p[0] + u[0], p[1] + u[1])
                if n not in prev and _walkable(memory, n, occupied):
                    prev[n] = p
                    q.append(n)
        return None


# -- one life (pass 1, replay, branches) -------------------------------------------------------------------------
class World:
    def __init__(self, home: str, weights: bytes, life0: int):
        from continuous_brain_cb2.consolidation import install
        from continuous_brain_individual.individual import Individual, IndividualConfig
        from continuous_brain_organs.language import LanguageOrgan
        src, self.meta = load_individual(Path(home), vision())
        self.new = lambda: Individual(IndividualConfig(lifelong=True, skills="cb2"), src.vision,
                                      LanguageOrgan.from_config({"backend": "dev"}))
        dst = self.new()
        install(src, dst, weights, {"note": "H1 Phase 0 measurement copy (never saved)"})
        self.base = dst.save()
        self.life0 = life0

    def fresh(self, gidx):
        ind = self.new()
        ind.restore(copy.deepcopy(self.base))
        ind.life = self.life0 + gidx
        return ind


def _step_record(ind, obs, info, a, info2, prev_health_drop):
    from continuous_brain_world import daylight as dl
    memory = ind.read_state("episodic_memory") or {"objects": [], "map": np.full((64, 64), -1, np.int8)}
    memory["map"] = np.asarray(memory["map"])
    pv = info["privileged"]
    pos, inv = tuple(pv["player_pos"]), pv["inventory"]
    zs, ss = perceived(memory)
    tz = [tuple(o["pos"]) for o in pv["objects"] if o["kind"] == "zombie"]
    ts = [tuple(o["pos"]) for o in pv["objects"] if o["kind"] == "skeleton"]
    host = ind.rt.components["skills"]
    ev = info2.get("events", {}) or {}
    step = int(obs["step"])
    return {"t": step, "a": int(a), "cmd": int(host.last["command"]) if getattr(host, "last", None) else -1,
            "pz": any(_man(z, pos) <= Z_R for z in zs), "ps": any(_man(s, pos) <= S_R for s in ss),
            "pain": bool(prev_health_drop), "encl": enclosed(memory, pos), "sleep": bool(pv["sleeping"]),
            "dark": bool(dl.is_dark(step)), "h": int(pv["health"]), "food": int(inv["food"]),
            "drink": int(inv["drink"]), "energy": int(inv["energy"]),
            "tz": any(_man(z, pos) <= Z_R for z in tz), "ts": any(_man(s, pos) <= S_R for s in ts),
            "zh": int(ev.get("zombie_hits", 0)), "ah": int(ev.get("arrow_hits", 0)),
            "h2": int(info2["privileged"]["health"])}


def live(ind, env, obs, info, steps_left, t_stop=None, override=None, ref=None, drop0=False):
    """Live from the current state; returns per-step records, actions, (died, cause).  Stops before deciding at
    step t_stop if given (returns obs/info there).  ref: actions to match (fidelity)."""
    host = ind.rt.components["skills"]
    host.override = override
    recs, acts, done, n, drop = [], [], False, 0, bool(drop0)
    mismatch = None
    while n < steps_left and not done:
        if t_stop is not None and int(obs["step"]) == t_stop:
            host.override = None
            return {"recs": recs, "acts": acts, "obs": obs, "info": info, "stopped": True, "mismatch": mismatch,
                    "drop": drop}
        a, u = ind.decide(obs, info)
        obs2, _, done, info2 = env.step(a, u)
        r = _step_record(ind, obs, info, a, info2, drop)
        inv = info2["privileged"]["inventory"]
        drop = r["h2"] < r["h"] and inv["food"] > 0 and inv["drink"] > 0 and inv["energy"] > 0
        recs.append(r)
        acts.append(int(a))
        if ref is not None and mismatch is None and len(acts) <= len(ref) and acts[-1] != ref[len(acts) - 1]:
            mismatch = len(acts) - 1
        obs, info, n = obs2, info2, n + 1
    inv = info["privileged"]["inventory"]
    died = bool(done) and int(info["privileged"]["health"]) <= 0
    cause = "alive" if not died else ("food" if inv["food"] == 0 else "drink" if inv["drink"] == 0
                                      else "energy" if inv["energy"] == 0 else "damage")
    host.override = None
    return {"recs": recs, "acts": acts, "died": died, "cause": cause, "stopped": False, "mismatch": mismatch}


def encounters(recs, died: bool, cause: str) -> list[dict]:
    """Segment eligible encounters (protocol rev 2, section 2) on per-step records."""
    out, i, n = [], 0, len(recs)
    elig = [(r["pz"] or r["ps"] or r["pain"]) and not r["sleep"] and not r["encl"] for r in recs]
    near = [r["pz"] or r["ps"] for r in recs]
    while i < n:
        if not elig[i]:
            i += 1
            continue
        s, j, quiet, end = i, i, 0, None
        while j < n:
            quiet = 0 if near[j] else quiet + 1
            if quiet >= CLEAR:
                end = ("cleared", j)
                break
            if j - s + 1 >= TIMEOUT:
                end = ("timeout", j)
                break
            j += 1
        if end is None:
            end = ("died" if died else "censored", n - 1)
        kind, e = end
        if out and s - out[-1]["end"] <= MERGE:                    # merge with the previous encounter
            prev = out.pop()
            s = prev["start"]
        r0 = recs[s]
        dmg = sum(max(0, r["h"] - r["h2"]) for r in recs[s:e + 1] if r["zh"] or r["ah"])
        death_in = died and (kind == "died" or (n - 1) - e <= POST)
        out.append({"start": s, "end": e, "t0": r0["t"], "kind": kind, "len": e - s + 1, "damage": int(dmg),
                    "zh": sum(r["zh"] for r in recs[s:e + 1]), "ah": sum(r["ah"] for r in recs[s:e + 1]),
                    "assoc_death": bool(death_in and cause == "damage"), "assoc_death_any": bool(death_in),
                    "after": (n - 1) - e,
                    "hands": r0["h"] >= 6 and min(r0["food"], r0["drink"], r0["energy"]) > 2
                             and not (r0["dark"] and not r0["encl"]),
                    "onset": {k: r0[k] for k in ("h", "food", "drink", "energy", "dark", "cmd", "pz", "ps", "pain")}})
        i = e + 1
    return out


def world_job(args):
    home, weights, life0, gidx, seed, max_steps = args
    import torch
    torch.set_num_threads(1)
    from continuous_brain_individual.individual import first_info
    from continuous_brain_world.env import W1Config, W1Env
    t_start = time.time()
    W = World(home, weights, life0)
    # pass 1
    ind, env = W.fresh(gidx), W1Env(W1Config())
    obs = env.reset(seed)
    p1 = live(ind, env, obs, first_info(env), max_steps)
    recs = p1["recs"]
    enc = encounters(recs, p1["died"], p1["cause"])
    missed = sum(1 for r in recs if (r["tz"] and not r["pz"]) or (r["ts"] and not r["ps"]))
    hits_outside = 0
    inside = set()
    for e in enc:
        inside.update(range(e["start"], e["end"] + 1))
    hits_outside = sum(r["zh"] + r["ah"] for i, r in enumerate(recs) if i not in inside)
    out = {"seed": seed, "steps": len(recs), "died": p1["died"], "cause": p1["cause"], "encounters": enc,
           "damage_total": int(sum(max(0, r["h"] - r["h2"]) for r in recs if r["zh"] or r["ah"])),
           "missed_threat_steps": missed, "hits_outside_encounters": hits_outside,
           "true_threat_steps": sum(1 for r in recs if r["tz"] or r["ts"])}
    onset = next((e for e in enc if e["hands"]), None)
    out["branched"] = None
    if onset is not None:
        t0 = onset["t0"]
        ind, env = W.fresh(gidx), W1Env(W1Config())
        obs = env.reset(seed)
        rep = live(ind, env, obs, first_info(env), max_steps, t_stop=t0, ref=p1["acts"])
        ok_prefix = rep["stopped"] and rep["mismatch"] is None and rep["acts"] == p1["acts"][:len(rep["acts"])]
        snap_i, snap_e = copy.deepcopy(ind.save()), env.snapshot()
        obs0, info0 = copy.deepcopy(rep["obs"]), copy.deepcopy(rep["info"])
        k0 = len(rep["acts"])
        br = {}
        for sc in SCRIPTS:
            bi = W.new()
            bi.restore(copy.deepcopy(snap_i))
            bi.life = ind.life
            be = W1Env(W1Config())
            be.restore(copy.deepcopy(snap_e))
            r = live(bi, be, copy.deepcopy(obs0), copy.deepcopy(info0), min(BRANCH_CAP, max_steps - k0),
                     override=Script(sc, t0), ref=p1["acts"][k0:] if sc == "control" else None, drop0=rep["drop"])
            e2 = encounters(r["recs"], r["died"], r["cause"])
            first = e2[0] if e2 and e2[0]["start"] == 0 else None
            br[sc] = {"steps_after_onset": len(r["recs"]), "died": r["died"], "cause": r["cause"],
                      "censored": not r["died"], "encounter": first,
                      "damage_total": int(sum(max(0, x["h"] - x["h2"]) for x in r["recs"] if x["zh"] or x["ah"])),
                      "n_encounters": len(e2)}
            if sc == "control":
                br[sc]["fidelity_mismatch_at"] = r["mismatch"]
                n_cmp = min(len(r["acts"]), len(p1["acts"]) - k0)
                br[sc]["fidelity_ok"] = r["mismatch"] is None and r["acts"][:n_cmp] == p1["acts"][k0:k0 + n_cmp]
        out["branched"] = {"t0": t0, "prefix_ok": ok_prefix, "branches": br}
    out["seconds"] = round(time.time() - t_start, 1)
    return out


def run(a) -> int:
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if any(r0 <= s <= r1 for s in seeds for r0, r1 in RESERVED if r0 >= 90000):
        raise SystemExit("refusing: a sealed world is in --seeds")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    home = Path(a.individuals) / a.individual
    meta = json.loads((home / "individual.json").read_text())
    if meta["weights_sha256"] != "8cffde798661d2661c469fa208799561008d03ae860d2d9be7f1c514a3c6853b":
        raise SystemExit("not S2b")
    weights = (home / "weights" / f"{meta['weights_sha256']}.pt").read_bytes()
    f = out / "worlds.jsonl"
    done = {json.loads(l)["seed"] for l in open(f)} if f.exists() else set()
    (out / "run.json").write_text(json.dumps({"format": "cb2-h1-phase0/1", **git_rev(), "seeds": a.seeds,
                                              "individual": {k: meta[k] for k in ("state_hash", "weights_sha256")},
                                              "scripts": SCRIPTS, "radii": {"zombie": Z_R, "skeleton": S_R},
                                              "clear": CLEAR, "timeout": TIMEOUT, "merge": MERGE, "post": POST,
                                              "branch_cap": BRANCH_CAP}, indent=1) + "\n")
    jobs = [(str(home), weights, int(meta["life"]) + 1, i, s, a.max_steps) for i, s in enumerate(seeds) if s not in done]
    print(f"{len(jobs)} worlds to do ({len(done)} done)", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(world_job, jobs):
            with open(f, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            b = r["branched"]
            print(json.dumps({"seed": r["seed"], "steps": r["steps"], "cause": r["cause"], "enc": len(r["encounters"]),
                              "t0": b and b["t0"], "fid": b and (b["prefix_ok"] and b["branches"]["control"]["fidelity_ok"]),
                              "s": r["seconds"]}), flush=True)
    return 0


def _lb(d, q=5):
    d = np.asarray(d, float)
    rng = np.random.default_rng(20261009)
    return float(np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), q))


def _ci(d):
    return [_lb(d, 2.5), _lb(d, 97.5)] if len(d) > 1 else [None, None]


def report(a) -> int:
    out = Path(a.out)
    W = [json.loads(l) for l in open(out / "worlds.jsonl")]
    W.sort(key=lambda w: w["seed"])
    E = [e for w in W for e in w["encounters"]]
    deaths = [w for w in W if w["died"]]
    assoc = [w for w in deaths if any(e["assoc_death"] for e in w["encounters"])]
    per_life = lambda key: [len(w["encounters"]) if key == "n" else w[key] for w in W]
    dd = [w for w in deaths if w["cause"] == "damage"]
    traced = {"hands": 0, "upstream": 0, "none": 0}
    for w in dd:
        last = w["encounters"][-1] if w["encounters"] else None
        traced["none" if last is None else "hands" if last["hands"] else "upstream"] += 1
    census = {
        "worlds": len(W), "deaths": len(deaths), "causes": {c: sum(w["cause"] == c for w in W) for c in
                                                           ("damage", "food", "drink", "energy", "alive")},
        "mean_survival": float(np.mean([w["steps"] for w in W])),
        "encounters_per_life": {"mean": float(np.mean(per_life("n"))), "ci95": _ci(per_life("n"))},
        "encounter_outcomes": {k: sum(e["kind"] == k for e in E) for k in ("cleared", "timeout", "died", "censored")},
        "escape_rate": sum(e["kind"] == "cleared" for e in E) / max(1, len(E)),
        "timeout_rate": sum(e["kind"] == "timeout" for e in E) / max(1, len(E)),
        "damage_per_encounter": {"mean": float(np.mean([e["damage"] for e in E])) if E else None,
                                 "ci95": _ci([e["damage"] for e in E])},
        "combat_damage_per_life": {"mean": float(np.mean(per_life("damage_total"))), "ci95": _ci(per_life("damage_total"))},
        "hands_attributable_share": sum(e["hands"] for e in E) / max(1, len(E)),
        "damage_deaths_traced_to_last_encounter": traced,
        "missed_threat_steps_per_life": float(np.mean(per_life("missed_threat_steps"))),
        "true_threat_steps_per_life": float(np.mean(per_life("true_threat_steps"))),
        "hits_outside_encounters_per_life": float(np.mean(per_life("hits_outside_encounters"))),
    }
    ind = np.array([1.0 if w in assoc else 0.0 for w in deaths])
    screen_a = {"numerator": "deaths that are encounter-associated (damage cause, during or <=50 steps after)",
                "denominator": "all deaths in the census", "value": float(ind.mean()) if len(ind) else None,
                "ci95": _ci(ind), "threshold": 0.30}
    B = [w["branched"] for w in W if w["branched"]]
    fid = all(b["prefix_ok"] and b["branches"]["control"]["fidelity_ok"] for b in B)

    def ed(br):                                                       # encounter-associated death in a branch
        e = br["encounter"]
        return 1.0 if (e and e["assoc_death"]) else 0.0
    ctrl = np.array([ed(b["branches"]["control"]) for b in B])
    per_script = {}
    for sc in SCRIPTS[1:]:
        x = np.array([ed(b["branches"][sc]) for b in B])
        ds = [b["branches"][sc]["steps_after_onset"] - b["branches"]["control"]["steps_after_onset"] for b in B]
        dmg = [((b["branches"][sc]["encounter"] or {}).get("damage", 0)) - ((b["branches"]["control"]["encounter"] or {}).get("damage", 0)) for b in B]
        per_script[sc] = {"assoc_death_rate": float(x.mean()), "paired_death_diff_ci95": _ci(x - ctrl),
                          "survival_diff": {"mean": float(np.mean(ds)), "ci95": _ci(ds)},
                          "encounter_damage_diff": {"mean": float(np.mean(dmg)), "ci95": _ci(dmg)}}

    def best(b):                                                      # hindsight: fewest assoc deaths, then damage, then longest
        return min(SCRIPTS, key=lambda s: (ed(b["branches"][s]), (b["branches"][s]["encounter"] or {}).get("damage", 0),
                                           -b["branches"][s]["steps_after_onset"]))
    bests = [best(b) for b in B]
    xb = np.array([ed(b["branches"][s]) for b, s in zip(B, bests)])
    dsb = [b["branches"][s]["steps_after_onset"] - b["branches"]["control"]["steps_after_onset"] for b, s in zip(B, bests)]
    rel = (ctrl.mean() - xb.mean()) / ctrl.mean() if ctrl.mean() > 0 else None
    rng = np.random.default_rng(20261009)
    rels = []
    for _ in range(10000):
        i = rng.integers(0, len(B), len(B))
        c = ctrl[i].mean()
        if c > 0:
            rels.append((c - xb[i].mean()) / c)
    screen_b = {"numerator": "relative reduction in encounter-associated deaths, best-of-family (hindsight) vs control",
                "denominator": "sampled hands-attributable onsets (first per world)", "onsets": len(B),
                "control_rate": float(ctrl.mean()) if len(B) else None, "best_rate": float(xb.mean()) if len(B) else None,
                "relative_reduction": rel, "ci95": [float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))] if rels else None,
                "paired_lower95_of_death_reduction": _lb(ctrl - xb) if len(B) > 1 else None,
                "overall_survival_change_best": {"mean": float(np.mean(dsb)) if dsb else None, "ci95": _ci(dsb)},
                "best_script_counts": {s: bests.count(s) for s in SCRIPTS}, "threshold": 0.25,
                "note": "hindsight choice from privileged outcomes: an optimistic ceiling, not an attainable improvement"}
    ok_a = screen_a["value"] is not None and screen_a["value"] >= 0.30
    ok_b = rel is not None and rel >= 0.25 and (screen_b["paired_lower95_of_death_reduction"] or 0) > 0
    rep = {"format": "cb2-h1-phase0-report/1", "status": "MEASUREMENT ONLY (DAV-97 Phase 0)",
           "run": json.loads((out / "run.json").read_text()), "fidelity_all_ok": fid, "census": census,
           "screen_a_opportunity": screen_a, "screen_b_headroom": screen_b, "per_script_fixed_effects": per_script,
           "screening": ("OPPORTUNITY SCREENED IN (a reason to propose Phase 1, not a result)" if ok_a and ok_b and fid
                         else "NOT SCREENED IN" if fid else "STOPPED: branch fidelity failed")}
    p = Path(a.report or REPO / "evidence/explore/cb2/h1-phase0-report.json")
    p.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("fidelity_all_ok", "screening")}, indent=1))
    print(json.dumps({"a": screen_a, "b": screen_b}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--individuals", required=True)
    r.add_argument("--individual", default="cb2-ind-0001")
    r.add_argument("--seeds", default="77001-77120")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--max-steps", type=int, default=10000)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--report", default=None)
    q.set_defaults(func=report)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
