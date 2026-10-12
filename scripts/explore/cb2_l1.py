#!/usr/bin/env python3
"""CB2-L1 (DAV-98): does the individual learn from its own lived experience, and can we tell?

Two twins are cloned from the same individual (S2b): LEARNING (sleeps) and FROZEN (never sleeps).  Each cycle:

  wake    both twins live the same practice worlds (run 9: 89,000,000 + cycle * 10,000 + n); the learning twin's
          lives go into its own archive (PRACTICE x twin-L, every step with the command it chose, sidecar).
  sleep   the learning twin only, from its own whole history (its lives so far, alternated 1:1 with the canonical
          individual's 592 lived lives that came before the split):
          L-A  its core continues (long windows; remembered-map head on its own past perception);
          re-anchor  its selector follows its new core (self-distillation: it keeps choosing as it did, now from the
               new core's state -- no teacher);
          L-C  its chooser learns its own hands (state, command -> action) and the value of its states in its own
               drives (reward, death), then chooses commands by imagined consequence (selector as prior);
          then one consolidation event in its home (hash-chained, exact rollback).
  probe   both twins (and, at the last cycle, the learning twin with its chooser off: the Gate 2 ablation) on probes
          they never learn from: FRESH worlds (new each cycle), TRANSFER worlds (a caretaker unlike its practice:
          twice the request rate, heavy on 'avoid' and 'maintain'), and SITUATIONS (fixed bank, every cycle: it
          wakes up in a saved world moment -- dusk outside, hungry, a zombie close -- with a fresh mind; the same
          bank every cycle also measures retention).

  run     python scripts/explore/cb2_l1.py run --twin <S2b candidate dir> --bootstrap <archive with lived-cb2-ind-0001>
              --cycles 10 --lives 20 --workers 30 --out runs/explore/cb2/l1
  report  python scripts/explore/cb2_l1.py report --out runs/explore/cb2/l1

Every number a claim rests on is a paired difference (learning - frozen) on the same probe worlds or situations.

L1b (docs/explore/CB2_L1B_PREREG_DRAFT.md) changes three things:
  YOKED twin (replaces L1's fixed-buffer reference): each cycle the FROZEN twin also lives the same practice worlds
          into archive-yoked; the yoked twin sleeps on that accumulating archive with identical budgets -- the same
          amount of matched-world experience, by another author.  "learning - yoked" is the own-experience contrast.
  cycle 0 instrument check: every arm must give identical probe lives before any sleep, or the run stops.
  final   the PRIMARY: the final individuals on a large battery of new worlds (default 1,200 fresh + 1,200 transfer):
              python scripts/explore/cb2_l1.py final --out <run> --workers <n>      (after the loop; any machine)
          the report applies the preregistered decision rules (verdict()).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import PRACTICE_BASE, RESERVED, git_rev, vision  # noqa: E402
from cb2_evaluate import _consolidate, load_individual  # noqa: E402

RUN = 11                   # L1b's practice run (seeds 80M + run * 1M + ...): 0-9 earlier runs (L1 = 9), 10 touched by
USED_RUNS = set(range(11))  # rehearsals -- never reuse
# development probe worlds (never practice, never sealed).  L1 used 78100 / 78600 / 79100, whose per-cycle blocks
# overlapped across families (fresh cycle 5 = transfer cycle 0; the situation bank = transfer cycle 5) -- Sol's audit.
# L1b: new blocks (Sol, S-003), disjoint by construction and checked at start (_check_worlds) against L1's actual worlds.
FRESH0, TRANSFER0, SITU0 = 88050, 92050, 94050
L1_USED = [(78100, 79639)]  # every probe/situation world L1 (b6a76ee) touched
TIMING0, TIMING_N = 95000, 100   # throughput measurement worlds (bench-lives): never scored
# L1b final battery (docs/explore/CB2_L1B_PREREG_DRAFT.md): new blocks, used only once, by the final individuals
FINAL_FRESH0, FINAL_TRANSFER0, FINAL_BLOCK = 84100, 86100, 1900
# every world a CPU rehearsal touched before the lock (exposure bookkeeping, Sol S-010); production families avoid them
TOUCHED = [(84000, 84002), (86000, 86002), (88000, 88001), (88100, 88101), (92000, 92001), (92100, 92101), (94000, 94000),
           (80100, 80101), (80200, 80201), (82100, 82100)]
# rehearsals use their own smoke blocks and practice run, and can never yield a claim (report: "rehearsal: no claim")
BLOCKS = {"production": dict(run=RUN, fresh0=None, transfer0=None, situ0=None, final_fresh0=None, final_transfer0=None,
                             final_block=None),
          "rehearsal": dict(run=99, fresh0=96000, transfer0=97000, situ0=98000, final_fresh0=98500, final_transfer0=99000,
                            final_block=400)}
CLAIMED = [(83090, 83195)]  # Sol's CPU memory diagnostic (timing 83090-83091, episodes 83100-83195): never ours
# capability safeguards for "shown" (prereg section 5): learning - frozen lower95 must stay above these, both families
SAFE_MARGIN = {"n_unlocked": -0.5, "requests_net": -0.5}
LIFE_STRIDE = 50_000       # life ids: family base (1,000,000 apart) + cycle * LIFE_STRIDE + index; counts < LIFE_STRIDE
# life-id family bases (one per kind of life; final-battery ids have no cycle part)
LID = {"fresh": 20_000_000, "transfer": 21_000_000, "situations": 22_000_000, "final-fresh": 23_000_000,
       "final-transfer": 24_000_000, "wake": 30_000_000, "heldout": 31_000_000, "yoked-wake": 32_000_000}
ARMS = ("learning", "yoked", "frozen")
TRANSFER = {"arrival_rate0": 2 / 110, "arrival_growth": 0.3, "arrival_rate_max": 1 / 20,
            "family_weights": {"collect": 0.10, "craft": 0.06, "place": 0.07, "maintain": 0.25, "avoid": 0.30,
                               "visit": 0.08, "report": 0.14}}
SITU_STEPS = 300
_BOOT: dict = {}


def _sealed(seed):
    return any(a <= seed <= b for a, b in RESERVED if a >= 90000)


def _blocks(a) -> dict:
    """The world blocks of this run: production (module constants) or the rehearsal's smoke blocks."""
    if getattr(a, "rehearsal", False):
        return dict(BLOCKS["rehearsal"])
    b = BLOCKS["production"]
    return {"run": b["run"], "fresh0": b["fresh0"] or FRESH0, "transfer0": b["transfer0"] or TRANSFER0,
            "situ0": b["situ0"] or SITU0, "final_fresh0": b["final_fresh0"] or FINAL_FRESH0,
            "final_transfer0": b["final_transfer0"] or FINAL_TRANSFER0, "final_block": b["final_block"] or FINAL_BLOCK}


# ================================================================================================ lives
def _actor(home: str, life: int, chooser_on: bool = True):
    import torch
    torch.set_num_threads(1)
    ind, meta = load_individual(Path(home), vision())
    ind.life = life
    host = ind.rt.components["skills"]
    if not chooser_on:
        host.chooser_off = True
    return ind, meta


def _cmd(ind):
    last = getattr(ind.rt.components["skills"], "last", None) or {}
    return int(last.get("command", -1)), bool((last.get("chooser") or {}).get("changed", False))


def _fallback(ind) -> int:
    """1 when the chooser fell back to the prior this tick because its imagination was non-finite."""
    last = getattr(ind.rt.components["skills"], "last", None) or {}
    return int(bool((last.get("chooser") or {}).get("nonfinite", False)))


def life_job(job):
    """One worker's share of lives: job = {home, seeds, kind, life0, archive?, stream?, chooser_on, config?}."""
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import first_info
    from continuous_brain_world.env import W1Config, W1Env
    out, cmds = [], []
    writer = None
    if job.get("archive"):
        writer = StreamWriter(job["archive"], job["stream"], "PRACTICE", job["who"],
                              {"role": "cb2-l1-wake", **git_rev(), "twin_state_hash": job["state_hash"]})
    for i, seed in enumerate(job["seeds"]):
        if job.get("archive"):
            _guard(job.get("run_dir"), job.get("run_id"), f"wake {job.get('stream')}")
        if _sealed(seed):
            raise SystemExit(f"refusing sealed world {seed}")
        life_id = job["lives"][i]
        ind, _ = _actor(job["home"], life_id, job.get("chooser_on", True))
        cfg = W1Config(**job.get("config", {}))
        env = W1Env(cfg)
        stats = {"changed": 0, "steps": 0, "dmg": 0, "fallback": 0}
        trace = hashlib.sha256()                            # every action and command of the life (instrument check)

        def on_step(n, action, _u, _o, _r, info2, ind=ind):
            c, ch = _cmd(ind)
            trace.update(f"{int(action)},{c};".encode())
            cmds.append(c)
            stats["changed"] += int(ch)
            stats["fallback"] += _fallback(ind)
            ev = info2.get("events", {}) or {}
            if ev.get("zombie_hits") or ev.get("arrow_hits"):
                stats["dmg"] += 1
        if job["kind"] == "situation":
            snap = job["situations"][i]
            env.restore(snap["env"])
            obs, info = snap["obs"], snap["info"]
            n, done = 0, False
            while n < SITU_STEPS and not done:
                a, u = ind.decide(obs, info)
                trace.update(f"{int(a)},{_cmd(ind)[0]};".encode())
                h0 = int(info["privileged"]["health"])
                obs, _, done, info = env.step(a, u)
                ev = info.get("events", {}) or {}
                stats["dmg"] += max(0, h0 - int(info["privileged"]["health"])) if (ev.get("zombie_hits") or ev.get("arrow_hits")) else 0
                stats["changed"] += int(_cmd(ind)[1])
                stats["fallback"] += _fallback(ind)
                n += 1
            out.append({"seed": seed, "life": life_id, "situation": snap["name"], "steps": n,
                        "died": bool(done) and n < SITU_STEPS, "combat_damage": stats["dmg"],
                        "chooser_changed": stats["changed"], "chooser_fallback": stats["fallback"],
                        "trace_sha": trace.hexdigest()})
            continue
        if writer is not None:
            rec = record_life(ind, env, seed, writer, steps=job.get("steps", 10000), who="SELF", lived=False, on_step=on_step)
            writer.close()
        else:
            obs = env.reset(seed)
            info = first_info(env)
            n, done = 0, False
            while n < job.get("steps", 10000) and not done:
                a, u = ind.decide(obs, info)
                obs, rw, done, info2 = env.step(a, u)
                on_step(n, a, u, obs, rw, info2)
                info = info2
                n += 1
            rec = {"steps": n, "done": bool(done)}
        s = env.episode_summary()
        out.append({"seed": seed, "life": life_id, "chooser_fallback": stats["fallback"],
                    "steps": rec["steps"], "died": bool(rec["done"]) and rec["steps"] < job.get("steps", 10000),
                    "n_unlocked": s["crafter"]["n_unlocked"], "unlocked": sorted(s["crafter"]["unlocked"]),
                    "fulfilled": s["obligations"]["fulfilled"], "failed": s["obligations"]["failed"],
                    "combat_hits": stats["dmg"], "chooser_changed": stats["changed"], "trace_sha": trace.hexdigest()})
    if writer is not None:
        p = Path(job["archive"]) / f"{job['stream']}.cmd.npy"
        np.save(p, np.asarray(cmds, np.int8))
        # (the sidecar is outside the archive's own checksums)
        (p.parent / f"{job['stream']}.cmd.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n")
    return out


def submit_lives(pool, home, seeds, kind, life0, workers, **kw):
    """Queue a set of lives on the pool now; returns gather() -> the sorted results.  Queuing every set of a cycle
    before gathering keeps all workers busy (one barrier per cycle instead of one per set); the chunking, seeds and
    life ids are exactly those of run_lives, so the results are the same."""
    chunks = [seeds[k::workers] for k in range(workers) if seeds[k::workers]]
    life_of = {s: life0 + n for n, s in enumerate(seeds)}   # one life id per world, whatever the worker split
    futs = []
    for k, ch in enumerate(chunks):
        j = {"home": str(home), "seeds": ch, "kind": kind, "lives": [life_of[s] for s in ch], **kw}
        if kw.get("archive"):
            j["stream"] = f"{kw['stream']}-w{k:02d}"
        if kind == "situation":
            j["situations"] = [kw["bank"][seeds.index(s)] for s in ch]
        futs.append(pool.submit(life_job, j))                     # a killed worker raises BrokenProcessPool
    t = time.time()

    def gather():
        res = [x for f in futs for x in f.result()]
        print(f"  {time.strftime('%H:%M:%S')} {kind} x{len(seeds)} gathered (+{time.time() - t:.0f}s)", flush=True)
        return sorted(res, key=lambda r: (r["seed"], r.get("situation", "")))
    gather.futures = futs
    return gather


def run_lives(pool, home, seeds, kind, life0, workers, **kw):
    return submit_lives(pool, home, seeds, kind, life0, workers, **kw)()


# ================================================================================================ situations
def build_bank(home, seeds, out: Path) -> list[dict]:
    """Saved world moments from the frozen twin's own lives on dedicated worlds (built once, before cycle 1):
    the first dusk it is outside, the first time food <= 3, the first zombie within 3.  Bank entries hold the
    world snapshot and the observation; the twin that wakes up there starts a fresh life (empty memory)."""
    import pickle
    from continuous_brain_individual.individual import first_info
    from continuous_brain_world import daylight as dl
    from continuous_brain_world.env import W1Config, W1Env
    bank = []
    for seed in seeds:
        ind, _ = _actor(str(home), 10_000_000 + seed)
        env = W1Env(W1Config())
        obs = env.reset(seed)
        info = first_info(env)
        # "dark_awake" is what L1 called "dusk_outside": it checks darkness and wakefulness, NOT being outside a shelter
        want = {"dark_awake": None, "hungry": None, "zombie_near": None}
        n, done = 0, False
        while n < 3000 and not done and any(v is None for v in want.values()):
            pv = info["privileged"]
            pos = pv["player_pos"]
            cond = {"dark_awake": dl.is_dark(int(obs["step"])) and not pv["sleeping"],
                    "hungry": pv["inventory"]["food"] <= 3,
                    "zombie_near": any(o["kind"] == "zombie" and abs(o["pos"][0] - pos[0]) + abs(o["pos"][1] - pos[1]) <= 3
                                       for o in pv["objects"])}
            for k, c in cond.items():
                if c and want[k] is None and n > 20:
                    want[k] = {"name": k, "seed": seed, "step": int(obs["step"]), "env": env.snapshot(),
                               "obs": copy.deepcopy(obs), "info": copy.deepcopy(info)}
            a, u = ind.decide(obs, info)
            obs, _, done, info = env.step(a, u)
            n += 1
        bank += [v for v in want.values() if v is not None]
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    return bank


# ================================================================================================ sleep
def _owns(run_dir, run_id) -> bool:
    """True while run_dir still belongs to the run that started this process (a process orphaned by a restart --
    the first L1 restart left two spawned sleeps alive -- must never write into the next run's directory)."""
    if not run_id:
        return True
    try:
        return json.loads((Path(run_dir) / "run.json").read_text()).get("run_id") == run_id
    except Exception:
        return False


def _guard(run_dir, run_id, what):
    if not _owns(run_dir, run_id):
        raise RuntimeError(f"{what}: {run_dir} no longer belongs to run {run_id} -- refusing to write")


def _cpu_quota(cg: str = "/sys/fs/cgroup") -> int:
    """The CPUs this process may really use (runbook rule 1: never nproc).  cgroup v2 cpu.max, then cgroup v1
    cfs_quota/period, then the pod script's own reading (L1_CPU_QUOTA).  The L1b bench pod was cgroup v1: the v2-only
    reader returned the host's 96 and each sleeper ran 48 torch threads on a 10-CPU quota."""
    n = os.cpu_count() or 1
    try:
        q, per = open(f"{cg}/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, min(n, int(int(q) / int(per))))
    except Exception:
        pass
    try:
        q = int(open(f"{cg}/cpu/cpu.cfs_quota_us").read().strip())
        per = int(open(f"{cg}/cpu/cpu.cfs_period_us").read().strip())
        if q > 0 and per > 0:
            return max(1, min(n, q // per))
    except Exception:
        pass
    env = os.environ.get("L1_CPU_QUOTA", "")
    return max(1, min(n, int(env))) if env.isdigit() else n


def _sleep_init(gpu, threads):
    """Spawned sleep process: its own GPU (when there are two) and its share of the CPU."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["L1_SLEEP_THREADS"] = str(threads)


def _n_gpus() -> int:
    import subprocess
    try:
        return len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
                    if l.startswith("GPU ")])
    except Exception:
        return 0


def sleep(twin_home: Path, own_archive: Path, bootstrap: Path, cycle: int, out: Path, a, twin: str = "learning") -> dict:
    """One sleep of the learning twin: L-A core, selector re-anchoring, L-C chooser, one consolidation."""
    import torch
    import torch.nn.functional as F
    from continuous_brain_cb2.archive import Archive
    from continuous_brain_cb2.chooser import Chooser, ChooserConfig
    from continuous_brain_cb2.data import SequenceSampler, train_mask
    from continuous_brain_cb2.skill_host import pack_weights, unpack_chooser, unpack_weights
    from continuous_brain_cb2.world_model import PredictiveCore
    rid = getattr(a, "run_id", None)
    _guard(out, rid, f"{twin} sleep {cycle}")
    torch.set_num_threads(int(os.environ.get("L1_SLEEP_THREADS", 0)) or _cpu_quota())   # nproc is the host's
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1000 + cycle)
    ind, meta = load_individual(twin_home, vision())
    blob = (twin_home / "weights" / f"{meta['weights_sha256']}.pt").read_bytes()
    core0, pol0 = unpack_weights(np.frombuffer(blob, np.uint8))
    ch0 = unpack_chooser(np.frombuffer(blob, np.uint8))
    # rebuild the core on its device: .to() leaves the RSSM's own _device (used for its initial state) on the CPU
    core0 = PredictiveCore.from_state(core0.state_dict_full(), dev)
    core0.eval()
    pol0.to(dev)
    pol0.device = dev
    if os.environ.get("L1_TF32") == "1":                   # engineering switches, measured by bench-sleep
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    import contextlib
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if os.environ.get("L1_AMP") == "1" and dev == "cuda" \
        else contextlib.nullcontext
    saved = core0.state_dict_full()
    core = PredictiveCore.from_state(saved, dev, **({} if core0.cfg.map_radius else {"map_radius": a.map_radius}))
    own = Archive(own_archive).shards(("PRACTICE",))
    boot = Archive(bootstrap).shards(("LIVED",)) or Archive(bootstrap).shards(("PRACTICE",))   # (engineering smoke: practice)
    T = a.T
    ns = ("student", "organ")
    s_own = SequenceSampler(Archive(own_archive), own, T=T, seed=cycle, namespaces=ns, map_crop=True, preload=dev == "cuda",
                            chain_short=True)
    key = (str(bootstrap), T)
    if key not in _BOOT:                                    # its past before the split does not change: load it once
        _BOOT[key] = SequenceSampler(Archive(bootstrap), boot, T=T, seed=77, namespaces=ns, map_crop=True,
                                     preload=dev == "cuda")
    s_boot = _BOOT[key]
    s_boot.rng = np.random.default_rng(77 + cycle)          # the same past batches for every twin's sleep this cycle
    import hashlib
    sidecar_sha = {}
    for s in s_own.streams:                                 # the command it chose at every row (wake sidecar)
        parts = []
        for name, rows in (s.components or ((s.stream, len(s)),)):
            p = Path(own_archive) / f"{name}.cmd.npy"
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            want = Path(own_archive) / f"{name}.cmd.sha256"
            if want.exists() and want.read_text().strip() != sha:
                raise RuntimeError(f"{twin} sleep {cycle}: command sidecar {name} does not match its checksum")
            sidecar_sha[name] = sha
            parts.append(np.load(p).astype(np.int64))
            assert len(parts[-1]) == rows, (name, len(parts[-1]), rows)
        s.cols["command"] = np.concatenate(parts)
        assert len(s.cols["command"]) == len(s), s.stream
        for col in ("reward", "done"):                      # bad numbers in its own lives would poison the value head
            if col in s.cols and not np.isfinite(np.asarray(s.cols[col], float)).all():
                raise RuntimeError(f"{twin} sleep {cycle}: non-finite {col} in {s.stream}")
    log = []
    t0 = time.time()
    prog = Path(out) / f"sleep-progress-{twin}-c{cycle:02d}.jsonl"    # appended as it happens, not only at the end

    def progress(rec):
        log.append(rec)
        with open(prog, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    # L-A: the core continues, alternating its recent lives with its whole past
    for u in range(1, a.core_updates + 1):
        with amp():
            m = core.train_step((s_own if u % 2 else s_boot).batch(a.batch))
        if u % 250 == 0 or u == a.core_updates:
            ev = core.evaluate(s_own.batch(a.batch))
            progress({"twin": twin, "cycle": cycle, "core_update": u, "s": round(time.time() - t0), "train": m,
                        "own_heldout_like": {k: ev[k] for k in ev if k.startswith(("map_", "risk_brier", "openloop", "copy_last", "image"))}})
    core.eval()
    burn = T // 4

    def feats(c, b):
        with torch.no_grad():
            post, _ = c.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
            return c.feat(post)

    def pol_in(b, f):
        tt = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        return (f[:, burn:].reshape(-1, f.shape[-1]), tt(b["body"][:, burn:]).reshape(-1, b["body"].shape[-1]),
                tt(b["features"][:, burn:]).reshape(-1, b["features"].shape[-1]),
                tt(b["map_crop"][:, burn:], torch.int64).reshape(-1, b["map_crop"].shape[-1]))
    # re-anchor: the selector keeps choosing as it did, now reading the new core
    pol = copy.deepcopy(pol0)
    pol.train()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    for u in range(1, a.anchor_updates + 1):
        b = (s_own if u % 2 else s_boot).batch(a.batch)
        with amp():
            with torch.no_grad():
                target = F.softmax(pol0(*pol_in(b, feats(core0, b)))[0].float(), -1)
            logits = pol(*pol_in(b, feats(core, b)))[0].float()
            loss = F.kl_div(F.log_softmax(logits, -1), target, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if u % 250 == 0 or u == a.anchor_updates:
            agree = float((logits.argmax(-1) == target.argmax(-1)).float().mean())
            progress({"twin": twin, "cycle": cycle, "anchor_update": u, "s": round(time.time() - t0), "kl": float(loss),
                        "agreement": agree})
    pol.eval()
    # L-C: its hands and the value of its states, from its own lives only
    ch = ch0 if ch0 is not None else Chooser(ChooserConfig(tau=a.tau, horizon=a.horizon), core.feat_size, dev)
    ch.to(dev); ch.device = dev
    ch.train()
    skipped, blk_vmax, blk_cap = 0, 0.0, 0.0
    for u in range(1, a.chooser_updates + 1):
        b = s_own.batch(a.batch)
        f = feats(core, b)
        tt = lambda x, dt=torch.int64: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        mask = tt(train_mask(b["life_start"], burn), torch.bool)
        # the window's "reward" is the previous step's (the core's input): the reward of transition t is at t + 1
        # the reward of each transition, terminal rewards included (L1 shifted the core's masked input instead and
        # lost every life's terminal reward -- Sol's audit)
        m = ch.train_step(f, tt(b["command"]), tt(b["action"]), tt(b["reward_out"], torch.float32),
                          (1 - tt(b["cont"], torch.float32)).bool(), tt(b["is_first"], torch.float32), mask)
        skipped += m.get("skipped", 0)
        blk_vmax, blk_cap = max(blk_vmax, m.get("value_raw_absmax", 0.0)), max(blk_cap, m.get("value_capped_frac", 0.0))
        if m.get("skipped"):
            progress({"twin": twin, "cycle": cycle, "chooser_update": u, "event": "skipped non-finite update", **m})
        if skipped > max(5, a.chooser_updates // 100):     # tripwire: stop loudly rather than learn from bad numbers
            progress({"twin": twin, "cycle": cycle, "chooser_update": u, "event": "TRIPWIRE", "skipped_so_far": skipped})
            raise RuntimeError(f"{twin} sleep {cycle}: {skipped} non-finite chooser updates by update {u}")
        if u % 250 == 0 or u == a.chooser_updates:
            # value_raw_absmax / value_capped_frac: this update's batch; block_*: the maximum over every update since
            # the previous record (Sol's review asked which one is logged)
            progress({"twin": twin, "cycle": cycle, "chooser_update": u, "s": round(time.time() - t0),
                      "skipped_so_far": skipped, **m, "block_value_raw_absmax": blk_vmax, "block_value_capped_frac": blk_cap})
            blk_vmax, blk_cap = 0.0, 0.0
    ch.eval()
    for x in (core, pol, ch):
        x.to("cpu")
    core.device = pol.device = ch.device = "cpu"
    for name, x in (("core", core), ("selector", pol), ("chooser", ch)):   # tripwire: never consolidate bad numbers
        badp = [n for n, p_ in x.named_parameters() if not bool(torch.isfinite(p_).all())]
        if badp:
            raise RuntimeError(f"{twin} sleep {cycle}: non-finite {name} parameters {badp[:5]} -- not consolidated")
    _guard(out, rid, f"{twin} sleep {cycle} (consolidation)")
    weights = pack_weights(core, pol, ch)
    data_sha = hashlib.sha256(json.dumps(sorted([r.stream, r.manifest["shard"], r.manifest["manifest_sha"]] for r in own)
                                         + sorted(sidecar_sha.items())).encode()).hexdigest()
    lineage = {"kind": "cb2-l1-sleep", "twin": twin, "cycle": cycle, "learned_from": {"own_rows": s_own.rows(), "bootstrap_rows": s_boot.rows(),
                                                                        "own_streams": sorted(s.stream for s in s_own.streams),
                                                                        "own_excluded_rows": s_own.excluded_rows,
                                                                        "own_data_sha256": data_sha,
                                                                        "author": {"learning": "self", "yoked": "frozen twin, same worlds (yoked)",
                                                                                   "reference": "self, fixed cycle-0 buffer"}.get(twin, twin)},
               "updates": {"core": a.core_updates, "anchor": a.anchor_updates, "chooser": a.chooser_updates}, "code": git_rev()}
    _consolidate(twin_home, ind, meta, vision(), weights, lineage, candidate=True)
    fd = os.open(out / "sleep.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)   # one append: twins may sleep at once
    try:
        os.write(fd, "".join(json.dumps(r) + "\n" for r in log).encode())
    finally:
        os.close(fd)
    # exposure (L1b compares arms with equal updates; unique rows differ with life lengths, so they are reported)
    return {"cycle": cycle, "seconds": round(time.time() - t0), "own_rows": s_own.rows(),
            "own_streams": len(s_own.streams), "own_excluded_rows": s_own.excluded_rows, "last": log[-6:]}


# ================================================================================================ bench
def _bench_one(twin, own, bootstrap, out, ns: dict) -> dict:
    """One benchmark sleep in its own process; returns its per-phase timing and peak GPU memory."""
    import torch
    out = Path(out)
    home = out / "twin" / "cb2-ind-0001"
    if home.parent.exists():
        shutil.rmtree(home.parent)
    shutil.copytree(Path(twin), home)
    a = argparse.Namespace(**ns)
    t = time.time()
    sleep(home, Path(own), Path(bootstrap), 1, out, a, "bench")
    log = [json.loads(l) for l in open(out / "sleep.jsonl")]
    ends = {k: max(x["s"] for x in log if k in x) for k in ("core_update", "anchor_update", "chooser_update")}
    per = {"core": ends["core_update"] / a.updates, "anchor": (ends["anchor_update"] - ends["core_update"]) / a.updates,
           "chooser": (ends["chooser_update"] - ends["anchor_update"]) / a.updates}
    full = 1500 * per["core"] + 1000 * per["anchor"] + 1500 * per["chooser"]
    return {"sec_per_update": {k: round(v, 3) for k, v in per.items()}, "wall_s": round(time.time() - t),
            "projected_full_sleep_min": round(full / 60, 1),
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
            "peak_host_rss_mb": round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024),
            "last_core": [x for x in log if "core_update" in x][-1]["train"]}


def bench_sleep(a) -> int:
    """Sleeps at full batch/T on scratch copies of a twin, few updates per phase, timed per phase.  For sizing a pod
    (and checking the GPU path) before renting the long run.  --concurrent 2 runs two sleeps AT ONCE on the same device,
    as the run does with one GPU and L1_PARALLEL_SLEEP=1: measured contention and peak memory per process (Sol, S-009).
    L1_TF32 / L1_AMP select precision."""
    from concurrent.futures import ProcessPoolExecutor
    out = Path(a.out)
    ns = {**{k: v for k, v in vars(a).items() if k != "func"}, "core_updates": a.updates, "anchor_updates": a.updates,
          "chooser_updates": a.updates}
    n = max(1, a.concurrent)
    th = max(1, _cpu_quota() // n)
    t = time.time()
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_sleep_init, initargs=(None, th)) as ex:
        futs = [ex.submit(_bench_one, a.twin, a.own, a.bootstrap, str(out / f"p{k}"), ns) for k in range(n)]
        res = [f.result() for f in futs]
    print(json.dumps({"tf32": os.environ.get("L1_TF32") == "1", "amp": os.environ.get("L1_AMP") == "1",
                      "concurrent": n, "updates_per_phase": a.updates, "wall_s": round(time.time() - t),
                      "processes": res,
                      "projected_cycle_sleep_min": max(r["projected_full_sleep_min"] for r in res)}, default=str, indent=1))
    return 0


# ================================================================================================ run
def _check_worlds(a) -> dict:
    """Every world family of the run is disjoint from every other and from the sealed worlds."""
    B = _blocks(a)
    reh = getattr(a, "rehearsal", False)
    fam = {}
    for c in range(a.cycles + 1):
        fam[f"fresh-c{c}"] = set(range(B["fresh0"] + 100 * c, B["fresh0"] + 100 * c + a.probe_worlds))
        fam[f"transfer-c{c}"] = set(range(B["transfer0"] + 100 * c, B["transfer0"] + 100 * c + a.probe_worlds))
        fam[f"wake-c{c}"] = {PRACTICE_BASE + B["run"] * 1_000_000 + c * 10_000 + n for n in range(a.lives)}
        fam[f"heldout-c{c}"] = {PRACTICE_BASE + B["run"] * 1_000_000 + c * 10_000 + 5000 + n for n in range(a.holdout_lives)}
    fam["situations"] = set(range(B["situ0"], B["situ0"] + a.situation_worlds))
    nf = getattr(a, "final_worlds", 0)
    fam["final-fresh"] = set(range(B["final_fresh0"], B["final_fresh0"] + nf))
    fam["final-transfer"] = set(range(B["final_transfer0"], B["final_transfer0"] + nf))
    fam["timing"] = set(range(TIMING0, TIMING0 + TIMING_N))
    assert reh or B["run"] not in USED_RUNS, f"practice run {B['run']} was used before (runs {sorted(USED_RUNS)}): use a new one"
    assert (a.cycles + 1) * 100 <= 1000 or not reh, "rehearsal per-cycle blocks would overlap"
    # (the yoked archive's lives use the wake worlds on purpose: same worlds, different author)
    assert a.probe_worlds <= 100 and a.lives <= 5000 and a.holdout_lives <= 5000, "world blocks would overlap"
    assert nf <= B["final_block"] and getattr(a, "chooser_off_worlds", 0) <= nf, "final battery larger than its block"
    n_situ = 3 * a.situation_worlds
    assert max(a.lives, a.holdout_lives, a.probe_worlds, n_situ) < LIFE_STRIDE and (a.cycles + 1) * LIFE_STRIDE <= 1_000_000, \
        "life ids would collide across cycles or families"   # (Sol's review: lives=1001 collided at stride 1000)
    assert nf < 1_000_000
    names = list(fam)
    for i, x in enumerate(names):
        assert not any(_sealed(w) for w in fam[x]), f"{x} touches sealed worlds"
        assert not any(lo <= w <= hi for lo, hi in CLAIMED for w in fam[x]), f"{x} touches a block claimed by Sol"
        assert not any(lo <= w <= hi for lo, hi in L1_USED for w in fam[x]), f"{x} touches a world L1 used"
        if not reh and x != "timing":
            assert not any(lo <= w <= hi for lo, hi in TOUCHED for w in fam[x]), f"{x} touches a world a rehearsal used"
        for y in names[i + 1:]:
            assert not (fam[x] & fam[y]), f"world families overlap: {x} / {y}"
    return {k: [min(v), max(v)] for k, v in fam.items() if v}


def wake_specs(c: int, a, homes: dict, archives: dict, run_dir, run_id, hashes: dict) -> dict:
    """The lives that become experience in cycle c (pure: tested without running lives).
    learning: its own lives -> archive-learning.  yoked-source: the FROZEN twin on the same practice worlds ->
    archive-yoked (the yoked twin's experience: same worlds and count, another author).  heldout: the learning twin's
    reserved lives -> archive-heldout, never slept on."""
    run_no = _blocks(a)["run"]
    seeds = [PRACTICE_BASE + run_no * 1_000_000 + c * 10_000 + n for n in range(a.lives)]
    hseeds = [PRACTICE_BASE + run_no * 1_000_000 + c * 10_000 + 5000 + n for n in range(a.holdout_lives)]
    common = dict(steps=a.max_steps, run_dir=str(run_dir), run_id=run_id)
    return {
        "learning": dict(home=homes["learning"], seeds=seeds, life0=LID["wake"] + c * LIFE_STRIDE,
                         archive=str(archives["learning"]), stream=f"c{c:02d}", who="twin-L-of-cb2-ind-0001",
                         state_hash=hashes["learning"], **common),
        "yoked_source": dict(home=homes["frozen"], seeds=seeds, life0=LID["yoked-wake"] + c * LIFE_STRIDE,
                             archive=str(archives["yoked"]), stream=f"y{c:02d}", who="twin-F-of-cb2-ind-0001",
                             state_hash=hashes["frozen"], **common),
        "heldout": dict(home=homes["learning"], seeds=hseeds, life0=LID["heldout"] + c * LIFE_STRIDE,
                        archive=str(archives["heldout"]), stream=f"h{c:02d}", who="twin-L-of-cb2-ind-0001",
                        state_hash=hashes["learning"], **common),
    }


def sleep_specs(homes: dict, archives: dict) -> dict:
    """Who sleeps on what: identical arguments otherwise (same a, cycle, bootstrap, update counts)."""
    return {"learning": (homes["learning"], archives["learning"]), "yoked": (homes["yoked"], archives["yoked"])}


def instrument_check(probes: dict, expected: dict) -> list:
    """Cycle 0 (before any sleep) every arm is the same individual.  Every expected arm and family must be present with
    exactly the expected rows (no duplicates), and every life must match the frozen arm's exactly: its action/command
    trace hash, length, death, unlocks and requests (Sol, S-009: totals alone can hide different lives)."""
    bad = []
    fields = ("trace_sha", "steps", "died", "n_unlocked", "fulfilled", "failed")
    for arm in ARMS:
        if arm not in probes:
            bad.append((arm, "*", "missing arm", None))
            continue
        for fam, keys in expected.items():
            lives = probes[arm].get(fam)
            got = [(r["seed"], r.get("situation")) for r in lives or []]
            if sorted(got) != sorted(keys) or len(set(got)) != len(got):
                bad.append((arm, fam, "rows differ from the expected set", None))
                continue
            if arm == "frozen":
                continue
            ref = {(r["seed"], r.get("situation")): tuple(r.get(f) for f in fields) for r in probes["frozen"][fam]}
            for r in lives:
                if tuple(r.get(f) for f in fields) != ref.get((r["seed"], r.get("situation"))):
                    bad.append((arm, fam, r["seed"], r.get("situation")))
    return bad


MUTABLE_ARGS = {"workers", "out", "twin", "bootstrap", "resume", "run_id", "func"}   # machine/path only (Sol, S-009)


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tree_sha(p: Path) -> dict:
    p = Path(p)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    return {str(x.relative_to(p.parent if p.is_file() else p)): _sha_file(x) for x in files}


def _archive_sha(root: Path) -> str:
    """Identity of the bootstrap archive: its shards' manifests (the sleep's own data hash, same recipe)."""
    from continuous_brain_cb2.archive import Archive
    arc = Archive(root)
    sh = arc.shards(("LIVED",)) or arc.shards(("PRACTICE",))
    return hashlib.sha256(json.dumps(sorted([r.stream, r.manifest["shard"], r.manifest["manifest_sha"]] for r in sh)).encode()).hexdigest()


def origin_manifest(twin: Path) -> dict:
    """Checksummed record of the origin individual (S2b): its metadata and its snapshot's files.  Lets the S2b rollback
    check run on any machine holding the run directory (Sol, S-009), without the original source path."""
    from cb2_collect import snapshot_path
    home = Path(twin) / "cb2-ind-0001"
    meta = json.loads((home / "individual.json").read_text())
    return {"individual": meta, "individual_json_sha256": _sha_file(home / "individual.json"),
            "snapshot_files": _tree_sha(snapshot_path(home, meta))}


def _experiment(a) -> dict:
    return {k: v for k, v in vars(a).items() if k not in MUTABLE_ARGS}


def validate_resume(out: Path, a, lock: dict) -> list:
    """Everything that defines the experiment must be what the lock says, BEFORE anything is written (Sol, S-009):
    code revision (clean), immutable arguments, input hashes, and the recorded progress against the twins' state."""
    probs = []
    g = git_rev()
    if g["git"] != lock["git"]:
        probs.append(f"code {g['git'][:8]} is not the run's {lock['git'][:8]}")
    if g["git_dirty"] and not lock.get("rehearsal"):
        probs.append("source tree is dirty")
    want = lock["experiment"]
    now = _experiment(a)
    for k in sorted(set(want) | set(now)):
        if want.get(k) != now.get(k):
            probs.append(f"argument {k}: locked {want.get(k)!r}, now {now.get(k)!r}")
    src = json.loads((Path(a.twin) / "cb2-ind-0001" / "individual.json").read_text())["state_hash"]
    if src != lock["twin_state_hash"]:
        probs.append("--twin is not the locked origin individual")
    if _archive_sha(Path(a.bootstrap)) != lock["bootstrap_sha256"]:
        probs.append("--bootstrap is not the locked bootstrap archive")
    if (out / "origin.json").exists() and _sha_file(out / "origin.json") != lock["origin_manifest_sha256"]:
        probs.append("origin.json changed since the lock")
    prior = [json.loads(l) for l in open(out / "cycles.jsonl")] if (out / "cycles.jsonl").exists() else []
    if [r["cycle"] for r in prior] != list(range(len(prior))):
        probs.append("cycles.jsonl is not a contiguous record from cycle 0")
    origin = json.loads((out / "origin.json").read_text())["individual"]
    slept = sum(1 for r in prior if r.get("sleep"))
    for k in ARMS:
        h = out / f"twins/{k}/cb2-ind-0001/individual.json"
        if not h.exists():
            probs.append(f"{k} twin missing")
            continue
        n = int(json.loads(h.read_text()).get("consolidations", 0))
        want_n = int(origin.get("consolidations", 0)) + (slept if k != "frozen" else 0)
        if n != want_n:
            probs.append(f"{k} twin has {n} consolidations, the record implies {want_n} (an unrecorded sleep?)")
    if prior and any(v != lock["twin_state_hash"] for v in prior[0]["twin_hashes"].values()):
        probs.append("cycle 0 twins were not the origin individual")
    return probs


def run(a) -> int:
    import pickle
    import uuid
    out = Path(a.out).resolve()
    worlds = _check_worlds(a)
    B = _blocks(a)
    if (out / "run.json").exists():
        if not a.resume:
            raise SystemExit(f"{out} exists (use --resume to continue it)")
        lock = json.loads((out / "run.json").read_text())         # the lock is never rewritten
        probs = validate_resume(out, a, lock)
        if probs:
            raise SystemExit("refusing to resume -- this is not the locked experiment:\n  " + "\n  ".join(probs))
        a.run_id = lock["run_id"]
    else:
        if a.resume:
            raise SystemExit(f"nothing to resume in {out}")
        g = git_rev()
        if g["git_dirty"] and not a.rehearsal:
            raise SystemExit("a production run needs a clean, committed source tree")
        out.mkdir(parents=True, exist_ok=True)
        om = origin_manifest(Path(a.twin))
        (out / "origin.json").write_text(json.dumps(om, indent=1, sort_keys=True) + "\n")
        a.run_id = uuid.uuid4().hex                                # this run's identity; every writer checks it
        lock = {"format": "cb2-l1b/2", "run_id": a.run_id, **g, "rehearsal": bool(a.rehearsal), "blocks": B,
                "twin_source": str(a.twin), "twin_state_hash": om["individual"]["state_hash"],
                "origin_manifest_sha256": _sha_file(out / "origin.json"),
                "bootstrap": str(a.bootstrap), "bootstrap_sha256": _archive_sha(Path(a.bootstrap)), "worlds": worlds,
                "experiment": _experiment(a), "args": {k: v for k, v in vars(a).items() if k != "func"}}
        (out / "run.json").write_text(json.dumps(lock, indent=1) + "\n")
        os.chmod(out / "run.json", 0o444)
        os.chmod(out / "origin.json", 0o444)
    # L1b (docs/explore/CB2_L1B_PREREG_DRAFT.md): learning sleeps on its own accumulating lives; yoked sleeps with
    # identical budgets on the lives the FROZEN twin lived on the same practice worlds (replaces L1's fixed-buffer
    # reference, which degraded by itself); frozen never sleeps
    twins = {k: out / f"twins/{k}" for k in ARMS}
    for t in twins.values():
        if not t.exists():
            shutil.copytree(Path(a.twin) / "cb2-ind-0001", t / "cb2-ind-0001")
    homes = {k: v / "cb2-ind-0001" for k, v in twins.items()}
    archives = {"learning": out / "archive-learning", "yoked": out / "archive-yoked", "heldout": out / "archive-heldout"}
    for d in archives.values():
        d.mkdir(exist_ok=True)
    ctx = mp.get_context("fork")
    bank_f = out / "situations.pkl"
    from concurrent.futures import ProcessPoolExecutor
    ng = _n_gpus()
    sleepers = []
    if ng >= 2 or os.environ.get("L1_PARALLEL_SLEEP") == "1":
        th = max(1, _cpu_quota() // 2)
        sleepers = [ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=_sleep_init,
                                        initargs=(g if ng >= 2 else None, th)) for g in (0, 1)]
    print(f"gpus: {ng}  sleeps: {'parallel' if sleepers else 'sequential'}", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
        if not bank_f.exists():
            seeds = list(range(B["situ0"], B["situ0"] + a.situation_worlds))
            ks = [k for k in range(a.workers) if seeds[k::a.workers]]
            parts = list(pool.map(build_bank, [str(homes["frozen"])] * len(ks), [seeds[k::a.workers] for k in ks],
                                  [out / f"bank-{k}.pkl" for k in ks]))
            bank = sorted([x for p in parts for x in p], key=lambda s: (s["seed"], s["name"]))
            with open(bank_f, "wb") as f:
                pickle.dump(bank, f)
        bank = pickle.load(open(bank_f, "rb"))
        situ_ids = [s["seed"] * 10 + ("dark_awake", "hungry", "zombie_near").index(s["name"]) for s in bank]
        prior = [json.loads(l) for l in open(out / "cycles.jsonl")] if (out / "cycles.jsonl").exists() else []
        start = len(prior)
        seen_lives = {r["life"] for p_ in prior for k in ("wake", "heldout") for v in (p_.get(k) or {}).values()
                      for r in v if "life" in r}
        for c in range(start, a.cycles + 1):
            t0 = time.time()
            hashes = {k: json.loads((h / "individual.json").read_text())["state_hash"] for k, h in homes.items()}
            rec = {"cycle": c, "twin_hashes": hashes}
            # probes (cycle 0 = before any sleep: every arm is the same individual -- the instrument check below)
            fresh = list(range(B["fresh0"] + 100 * c, B["fresh0"] + 100 * c + a.probe_worlds))
            trans = list(range(B["transfer0"] + 100 * c, B["transfer0"] + 100 * c + a.probe_worlds))
            arms = {k: dict(home=homes[k]) for k in ARMS}
            if c == a.cycles:
                arms["learning_chooser_off"] = dict(home=homes["learning"], chooser_on=False)
            # queue every probe set and the wake lives at once (one barrier per cycle), then gather in order
            pend = {}
            for arm, kw in arms.items():
                pend[arm] = {
                    "fresh": submit_lives(pool, kw["home"], fresh, "life", LID["fresh"] + c * LIFE_STRIDE, a.workers,
                                          chooser_on=kw.get("chooser_on", True), steps=a.max_steps),
                    "transfer": submit_lives(pool, kw["home"], trans, "life", LID["transfer"] + c * LIFE_STRIDE, a.workers,
                                             chooser_on=kw.get("chooser_on", True), config=TRANSFER, steps=a.max_steps)}
            woke = {}
            if c < a.cycles:
                for k, w in wake_specs(c, a, homes, archives, out, a.run_id, hashes).items():
                    w = dict(w)
                    if not w["seeds"]:
                        woke[k] = lambda: []
                        continue
                    woke[k] = submit_lives(pool, w.pop("home"), w.pop("seeds"), "life", w.pop("life0"), a.workers, **w)
            for arm, kw in arms.items():
                pend[arm]["situations"] = submit_lives(pool, kw["home"], situ_ids, "situation", LID["situations"] + c * LIFE_STRIDE,
                                                       a.workers, chooser_on=kw.get("chooser_on", True), bank=bank)
            rec["probes"] = {arm: {k: g() for k, g in p.items()} for arm, p in pend.items()}
            rec["probe_seconds"] = round(time.time() - t0)
            if c == 0:
                bad = instrument_check(rec["probes"], {"fresh": [(w, None) for w in fresh], "transfer": [(w, None) for w in trans],
                                                       "situations": [(s_["seed"] * 10 + ("dark_awake", "hungry", "zombie_near").index(s_["name"]), s_["name"])
                                                                      for s_ in bank]})
                rec["instrument_check"] = {"ok": not bad, "mismatches": bad[:20]}
                if bad:                                  # keep the evidence, then stop: the arms are not comparable
                    with open(out / "cycles.jsonl", "a") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                    raise RuntimeError(f"instrument check failed at cycle 0: {len(bad)} probe lives differ between arms")
            if c < a.cycles:
                got = {k: g() for k, g in woke.items()}
                rec["wake"] = {"learning": got["learning"], "yoked_source": got["yoked_source"]}
                rec["heldout"] = {"learning": got["heldout"]}
                for r in got["learning"] + got["yoked_source"] + got["heldout"]:   # every archived life has its own id
                    if r["life"] in seen_lives:
                        raise RuntimeError(f"life id {r['life']} reused (world {r['seed']}, cycle {c})")
                    seen_lives.add(r["life"])
                sp = sleep_specs(homes, archives)
                if sleepers:                                 # the two twins sleep at once, each on its own GPU
                    fut = {k: sleepers[i].submit(sleep, h, ar, Path(a.bootstrap), c + 1, out, a, k)
                           for i, (k, (h, ar)) in enumerate(sp.items())}
                    rec["sleep"] = {k: f.result() for k, f in fut.items()}
                else:
                    rec["sleep"] = {k: sleep(h, ar, Path(a.bootstrap), c + 1, out, a, k) for k, (h, ar) in sp.items()}
            rec["seconds"] = round(time.time() - t0)
            with open(out / "cycles.jsonl", "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps({"cycle": c, "seconds": rec["seconds"], **summary_line(rec)}), flush=True)
    for x in sleepers:
        x.shutdown()
    return 0


def _lb(d, q=5):
    d = np.asarray(d, float)
    if len(d) < 2:
        return None
    rng = np.random.default_rng(20261009)
    return float(np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), q))


def _keyed(lives, what: str, key: str) -> dict:
    ks = [(r["seed"], r.get("situation")) for r in lives]
    if len(set(ks)) != len(ks):
        raise ValueError(f"{what}: duplicate rows")
    for r in lives:
        v = r.get(key)
        if v is None or not np.isfinite(float(v)):
            raise ValueError(f"{what}: non-finite {key} at world {r['seed']}")
    return dict(zip(ks, lives))


def paired(L, F, key, upper=False, expect=None, what="paired"):
    """Paired differences L - F by world.  Strict (Sol, S-009): both sides must hold exactly the same rows, no
    duplicates, finite values -- and, when given, exactly the expected worlds.  Nothing is silently dropped."""
    mL, mF = _keyed(L, f"{what} (first arm)", key), _keyed(F, f"{what} (second arm)", key)
    if set(mL) != set(mF):
        raise ValueError(f"{what}: the arms hold different rows ({len(set(mL) ^ set(mF))} unmatched)")
    if expect is not None and sorted(k[0] for k in mL) != sorted(expect):
        raise ValueError(f"{what}: the rows are not the expected worlds")
    d = [float(mL[k][key]) - float(mF[k][key]) for k in sorted(mL)]
    out = {"n": len(d), "mean": float(np.mean(d)) if d else None, "lower95": _lb(d)}
    if upper:
        out["upper95"] = _lb(d, 95)
        out["sd"] = float(np.std(d, ddof=1)) if len(d) > 1 else None
    return out


def summary_line(rec) -> dict:
    p = rec["probes"]
    out = {}
    for other in ("frozen", "yoked", "reference"):          # "reference": L1's records
        if other not in p:
            continue
        o = out[f"vs_{other}"] = {}
        for fam in ("fresh", "transfer"):
            o[fam] = {"survival": paired(p["learning"][fam], p[other][fam], "steps"),
                      "unlocks": paired(p["learning"][fam], p[other][fam], "n_unlocked")}
        o["situations"] = {"survival": paired(p["learning"]["situations"], p[other]["situations"], "steps")}
    lt = p["learning"]["fresh"]
    out["chooser_changed_share"] = (sum(r["chooser_changed"] for r in lt) / max(1, sum(r["steps"] for r in lt)))
    return out


MDE = 31.0          # L1b prereg section 4: minimum detectable effect (80% power, N=1,200/family); the futility threshold
FAMS = ("fresh", "transfer")


def verdict(prim: dict, safe: dict | None = None, identity: dict | None = None, rehearsal: bool = False) -> dict:
    """L1b prereg section 5, exactly.  prim[vs_o][fam] has lower95 / upper95 (one-sided 95% bootstrap over worlds);
    safe[metric][fam] has learning - frozen lower95 for the capability safeguards; identity[arm]["ok"] the identity /
    provenance / rollback checks.  Direction notes are DECISIONS to take, with the cause unproven (Sol, S-003);
    none of them launches anything."""
    G = all(prim["vs_frozen"][f]["lower95"] > 0 for f in FAMS)
    O = all(prim["vs_yoked"][f]["lower95"] > 0 for f in FAMS)
    guard = {m: all(safe[m][f]["lower95"] is not None and safe[m][f]["lower95"] > SAFE_MARGIN[m] for f in FAMS)
             for m in SAFE_MARGIN} if safe else {m: False for m in SAFE_MARGIN}
    ident = bool(identity) and all(v.get("ok") for v in identity.values())
    ok = all(guard.values()) and ident
    failed = [k for k, v in guard.items() if not v] + ([] if ident else ["identity/provenance/rollback"])
    if not G:
        claim = "not shown"
    elif failed:                                  # a survival gain bought with capability or identity is not a claim
        claim = "not shown: safeguards failed (" + ", ".join(failed) + ")"
    elif O:
        claim = "shown"
    else:
        claim = "growth shown; own closed-loop adaptation beyond matched-world frozen-policy experience not shown"
    if rehearsal:                                  # rehearsal settings can never produce a claim (Sol, S-010)
        claim = "rehearsal: no claim"
    notes = []
    if prim["vs_frozen"]["fresh"]["upper95"] < MDE:
        notes.append("decision needed: on fresh worlds this implementation at this budget gained less than the detectable "
                     "size (cause unproven) -- e.g. investigate objectives/controls or a bounded baseline comparison")
    elif prim["vs_frozen"]["fresh"]["lower95"] > 0 and prim["vs_frozen"]["transfer"]["upper95"] < MDE:
        notes.append("decision needed: gain on fresh worlds, below detectable size on transfer (cause unproven; "
                     "memorisation is one hypothesis) -- investigate before scaling")
    if G and all(prim["vs_yoked"][f]["upper95"] < MDE for f in FAMS):
        notes.append("decision needed: the yoked twin gained about as much (cause unproven) -- the own-adaptation "
                     "question stays open for L2")
    if rehearsal:
        notes = []
    return {"rehearsal": rehearsal, "G_growth": G, "O_own_adaptation": O, "safeguards": guard, "identity_ok": ident, "claim": claim,
            "decisions_needed": notes, "mde": MDE, "safe_margin": SAFE_MARGIN}


def identity_check(home: Path, origin: dict, sleeps: int, snapshot_files: dict | None = None) -> dict:
    """Identity / provenance / rollback for one final twin (prereg section 5): its consolidation chain verifies, links
    from the origin individual's state to its present state through exactly `sleeps` new events, and its origin
    snapshot (still in its home) restores to the origin's exact state hash."""
    from continuous_brain_cb2.consolidation import ConsolidationLog
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    from cb2_evaluate import organ_vision
    from cb2_collect import snapshot_path
    meta = json.loads((home / "individual.json").read_text())
    out = {"state_hash": meta["state_hash"]}
    try:
        ev = ConsolidationLog(home / "consolidations.jsonl")
        n = ev.verify()
        new = ev.events()[int(origin["consolidations"]):]
        chain = [origin["state_hash"]] + [e["event"]["state_hash"] for e in new]
        links = all(e["event"]["parent_state_hash"] == chain[i] for i, e in enumerate(new))
        out.update(chain_records=n, new_events=len(new), chain_links=links, ends_at_present=chain[-1] == meta["state_hash"])
        om = {**meta, "snapshot": origin["snapshot"], "skills": origin.get("skills", "cb2")}
        if snapshot_files is not None:                 # the twin's copy of the origin snapshot is byte-identical
            out["origin_snapshot_files_match"] = _tree_sha(snapshot_path(home, om)) == snapshot_files
        ind = Individual(IndividualConfig(lifelong=True, skills=om["skills"]), organ_vision(home, om, vision()),
                         LanguageOrgan.from_config({"backend": "dev"}))
        ind.rt.restore(SnapshotStore(home), snapshot_path(home, om))
        out["rollback_hash"] = ind.state_hash()
        out["ok"] = bool(links and len(new) == sleeps and chain[-1] == meta["state_hash"]
                         and out["rollback_hash"] == origin["state_hash"] and out.get("origin_snapshot_files_match", True))
    except Exception as e:                                    # a broken chain is a failed check, reported
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def _canon_sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def final_plan(out: Path) -> dict:
    """The locked final battery: its lock, twins (verified unchanged since the loop) and every arm x family condition.
    Shared by final() and report() so results are bound and checked the same way (Sol, S-013)."""
    lock = json.loads((out / "run.json").read_text())
    ra = lock["args"]
    C = [json.loads(l) for l in open(out / "cycles.jsonl")]
    if not C or C[-1]["cycle"] != ra["cycles"]:
        raise SystemExit("the loop has not finished: the final battery tests the final individuals only")
    homes = {k: out / f"twins/{k}/cb2-ind-0001" for k in ARMS}
    hashes = {k: json.loads((h / "individual.json").read_text())["state_hash"] for k, h in homes.items()}
    for k in ARMS:                     # the final cycle's record holds the twins as they were after the last sleep
        if hashes[k] != C[-1]["twin_hashes"][k]:
            raise SystemExit(f"{k} twin changed since the loop finished ({hashes[k][:12]} != {C[-1]['twin_hashes'][k][:12]})")
    B = lock["blocks"]
    nf, noff = ra["final_worlds"], ra["chooser_off_worlds"]
    worlds = {"fresh": list(range(B["final_fresh0"], B["final_fresh0"] + nf)),
              "transfer": list(range(B["final_transfer0"], B["final_transfer0"] + nf))}
    sets = [(arm, fam, worlds[fam], True) for arm in ARMS for fam in FAMS]
    sets += [("learning_chooser_off", fam, worlds[fam][:noff], False) for fam in FAMS if noff]
    conds = {}
    for arm, fam, seeds, on in sets:
        src = "learning" if arm == "learning_chooser_off" else arm
        conds[(arm, fam)] = {"run_id": lock["run_id"], "git": lock["git"], "arm": arm, "family": fam, "chooser_on": on,
                             "config": TRANSFER if fam == "transfer" else {}, "max_steps": ra["max_steps"],
                             "state_hash": hashes[src], "seeds": seeds, "life0": LID[f"final-{fam}"]}
    return {"lock": lock, "cycles": C, "homes": homes, "hashes": hashes, "worlds": worlds, "noff": noff, "conds": conds}


def read_partial(part: Path, key: tuple, cond: dict):
    """A recorded arm x family set, verified against its expected condition and checksum; None if never recorded."""
    f = part / f"{key[0]}-{key[1]}.json"
    if not f.exists():
        return None
    got = json.loads(f.read_text())
    if got.get("condition") != cond or _canon_sha(got.get("lives")) != got.get("lives_sha256"):
        raise SystemExit(f"{f} is stale or corrupt (not this run/revision/condition, or its checksum fails)")
    return got["lives"]


def verified_final(out: Path) -> dict:
    """The final battery's results for scoring: rebuilt ONLY from bound, checksummed partials of this locked run, all
    present; final.json must agree with them (run id, revision, twin hashes, content) -- a foreign or edited final.json
    is refused, never scored (Sol, S-013)."""
    P = final_plan(out)
    res = {}
    for key, cond in P["conds"].items():
        lives = read_partial(out / "final", key, cond)
        if lives is None:
            raise SystemExit(f"final battery incomplete: {key[0]} x {key[1]} has no recorded result")
        res.setdefault(key[0], {})[key[1]] = lives
    fj = json.loads((out / "final.json").read_text())
    lock = P["lock"]
    if fj.get("run_id") != lock["run_id"] or fj.get("twin_hashes") != P["hashes"] or _canon_sha(fj.get("arms")) != _canon_sha(res) \
            or (fj.get("git") != lock["git"] and not lock.get("rehearsal")):
        raise SystemExit("final.json does not match this run's verified final results (foreign, stale or edited)")
    return res


def final(a) -> int:
    """The final battery (L1b primary): the three final individuals on FINAL worlds, every arm on the same worlds and
    life ids; plus the learning twin with its chooser off on the first chooser_off_worlds of each family (Gate 2).
    Runs after the loop, anywhere with the run directory (e.g. a many-core CPU pod).  Each arm x family is bound to the
    locked run, revision and condition, checksummed, and written as soon as it completes; a batch that was started but
    never recorded stops the battery (its worlds were run and are not silently repeated -- Sol, S-009)."""
    out = Path(a.out).resolve()
    P = final_plan(out)
    lock, ra, homes, hashes, worlds, noff = P["lock"], P["lock"]["args"], P["homes"], P["hashes"], P["worlds"], P["noff"]
    g = git_rev()
    if (g["git"] != lock["git"] or g["git_dirty"]) and not lock.get("rehearsal"):
        raise SystemExit(f"code {g['git'][:8]}{' (dirty)' if g['git_dirty'] else ''} is not the run's {lock['git'][:8]}")
    part = out / "final"
    part.mkdir(exist_ok=True)
    t0 = time.time()
    res, todo = {}, []
    for (arm, fam), cond in P["conds"].items():               # verify what exists before running anything
        lives = read_partial(part, (arm, fam), cond)
        if lives is not None:
            res.setdefault(arm, {})[fam] = lives
        elif (part / f"{arm}-{fam}.started.json").exists():
            raise SystemExit(f"{arm} x {fam} was started and never recorded: its worlds ran unrecorded. Not repeating "
                             f"them silently -- this needs an explicit, recorded decision")
        else:
            todo.append((arm, fam, cond["seeds"], cond["chooser_on"], cond))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as pool:
        pend = {}
        for arm, fam, seeds, on, cond in todo:
            (part / f"{arm}-{fam}.started.json").write_text(json.dumps(cond) + "\n")
            home = homes["learning" if arm == "learning_chooser_off" else arm]
            kw = dict(chooser_on=on, steps=ra["max_steps"])
            if fam == "transfer":
                kw["config"] = TRANSFER
            pend[(arm, fam)] = (submit_lives(pool, home, seeds, "life", LID[f"final-{fam}"], a.workers, **kw), cond)
        while pend:                                           # record each set the moment it completes
            for key in [k for k, (gth, _) in pend.items() if all(f_.done() for f_ in gth.futures)]:
                gth, cond = pend.pop(key)
                lives = gth()
                (part / f"{key[0]}-{key[1]}.json").write_text(
                    json.dumps({"condition": cond, "lives": lives, "lives_sha256": _canon_sha(lives)}) + "\n")
                res.setdefault(key[0], {})[key[1]] = lives
            time.sleep(2)
    rec = {"format": "cb2-l1b-final/2", "run_id": lock["run_id"], **git_rev(), "rehearsal": bool(lock.get("rehearsal")),
           "twin_hashes": hashes, "worlds": {k: [v[0], v[-1]] for k, v in worlds.items()}, "chooser_off_worlds": noff,
           "seconds": round(time.time() - t0), "arms": res}
    (out / "final.json").write_text(json.dumps(rec) + "\n")
    print(json.dumps({"final_seconds": rec["seconds"], **{f"n_{k}": {f: len(v) for f, v in d.items()} for k, d in res.items()}}))
    return 0


def bench_lives(a) -> int:
    """Measured throughput of probe lives on THIS machine (Sol, S-003/S-010: measure before committing a cost): one
    twin (run it on S2b AND on a retained learner with its chooser active), timing worlds only (never scored), at
    each worker count given (scaling is measured, not assumed); prints steps/s and the projected L1b workload time."""
    from concurrent.futures import ProcessPoolExecutor
    seeds = list(range(TIMING0, TIMING0 + min(a.lives, TIMING_N)))
    # L1b workload, in steps, from L1's measured mean life lengths (cycles 6-10, ~620 fresh / ~650 transfer)
    per_cycle = 3 * (40 * 620 + 40 * 650 + 72 * 250) + 2 * 20 * 620      # 3 arms' probes + learning & yoked-source wake
    final = 3 * 1200 * (620 + 650) + 300 * (620 + 650)
    rows = []
    for w in [int(x) for x in str(a.workers).split(",")]:
        t = time.time()
        with ProcessPoolExecutor(w, mp_context=mp.get_context("fork")) as pool:
            res = run_lives(pool, Path(a.twin) / "cb2-ind-0001", seeds, "life", 25_000_000, w, steps=a.max_steps)
        sec = time.time() - t
        steps = sum(r["steps"] for r in res)
        rate = steps / sec
        rows.append({"workers": w, "lives": len(res), "steps": steps, "seconds": round(sec, 1), "steps_per_s": round(rate, 1),
                     "steps_per_s_per_worker": round(rate / w, 2),
                     "projected_h": {"per_cycle_probe_and_wake": round(per_cycle / rate / 3600, 2),
                                     "final_battery": round(final / rate / 3600, 2)}})
    print(json.dumps({"twin": str(a.twin), "cpu_quota": _cpu_quota(), "runs": rows}, indent=1))
    return 0


def report(a) -> int:
    out = Path(a.out)
    C = [json.loads(l) for l in open(out / "cycles.jsonl")]
    ctrl = "yoked" if "yoked" in C[0]["probes"] else "reference"          # L1 records used "reference"
    rep = {"format": "cb2-l1b-report/1", "run": json.loads((out / "run.json").read_text()),
           "instrument_check_cycle0": C[0].get("instrument_check"),
           "cycles": [{"cycle": r["cycle"], **summary_line(r)} for r in C]}
    # primary (L1b prereg section 4/5): the final individuals on the final battery
    fj = out / "final.json"
    if fj.exists():
        F = verified_final(out)                       # bound partials only; final.json must agree (Sol, S-013)
        lock = rep["run"]
        B, nf, noff = lock["blocks"], lock["args"]["final_worlds"], lock["args"]["chooser_off_worlds"]
        exp = {"fresh": list(range(B["final_fresh0"], B["final_fresh0"] + nf)),
               "transfer": list(range(B["final_transfer0"], B["final_transfer0"] + nf))}
        for arm in ARMS + (("learning_chooser_off",) if noff else ()):
            if set(F.get(arm, {})) != set(FAMS):
                raise SystemExit(f"final battery incomplete: {arm} has {sorted(F.get(arm, {}))}")
        prim = {f"vs_{o}": {fam: paired(F["learning"][fam], F[o][fam], "steps", upper=True, expect=exp[fam],
                                        what=f"final learning-{o} {fam}") for fam in FAMS}
                for o in ("frozen", ctrl)}
        rep["primary_final"] = prim
        net = lambda L: [{**r, "requests_net": r["fulfilled"] - r["failed"]} for r in L]
        safe = {m: {fam: paired(net(F["learning"][fam]), net(F["frozen"][fam]), m, upper=True, expect=exp[fam],
                                what=f"safeguard {m} {fam}") for fam in FAMS} for m in SAFE_MARGIN}
        rep["safeguards_final"] = safe
        om_f = out / "origin.json"                    # checksummed at the lock: no dependence on the source path
        if not om_f.exists() or _sha_file(om_f) != lock.get("origin_manifest_sha256"):
            rep["identity"] = {"error": "origin manifest missing or changed since the lock"}
            ident = None
        else:
            om = json.loads(om_f.read_text())
            n = sum(1 for r in C if r.get("sleep"))
            ident = rep["identity"] = {k: identity_check(out / f"twins/{k}/cb2-ind-0001", om["individual"],
                                                         n if k != "frozen" else 0, om["snapshot_files"]) for k in ARMS}
        rep["verdict"] = verdict(prim, safe, ident, bool(lock.get("rehearsal"))) if ctrl == "yoked" else None
        if "learning_chooser_off" in F:
            sub = {fam: [r for r in F["learning"][fam] if r["seed"] in set(exp[fam][:noff])] for fam in FAMS}
            rep["gate2_final"] = {fam: paired(sub[fam], F["learning_chooser_off"][fam], "steps", upper=True,
                                              expect=exp[fam][:noff], what=f"gate2 {fam}") for fam in FAMS}
    else:
        rep["primary_final"] = rep["verdict"] = None
        rep["note"] = "final battery not run yet: no primary result"
    # secondary: pooled over cycles 6-10 (L1's primary, kept for continuity)
    late = [r for r in C if 6 <= r["cycle"] <= 10]
    sec = rep["secondary_pooled_cycles_6_10"] = {
        f"vs_{o}": {fam: paired([x for r in late for x in r["probes"]["learning"][fam]],
                                [x for r in late for x in r["probes"][o][fam]], "steps")
                    for fam in FAMS} for o in ("frozen", ctrl)}
    for o in ("frozen", ctrl):                              # situations: per situation, mean over cycles 6-10
        sid = sorted({(x["seed"], x["situation"]) for r in late for x in r["probes"]["learning"]["situations"]})
        mean = lambda arm, k: np.mean([x["steps"] for r in late for x in r["probes"][arm]["situations"]
                                       if (x["seed"], x["situation"]) == k])
        d = [mean("learning", k) - mean(o, k) for k in sid]
        sec[f"vs_{o}"]["situations"] = {"n": len(d), "mean": float(np.mean(d)) if d else None, "lower95": _lb(d)}
    by = {r["cycle"]: r for r in C}
    if 5 in by and 10 in by:
        rep["retention_situations_c10_minus_c5"] = paired(by[10]["probes"]["learning"]["situations"],
                                                          by[5]["probes"]["learning"]["situations"], "steps")
    rep["exposure_per_sleep"] = [{"cycle": r["cycle"], **{k: {x: v.get(x) for x in ("own_rows", "own_streams", "own_excluded_rows")}
                                                          for k, v in (r.get("sleep") or {}).items()}} for r in C if r.get("sleep")]
    sl = [json.loads(l) for l in open(out / "sleep.jsonl")] if (out / "sleep.jsonl").exists() else []
    last_u = max((x.get("chooser_update", 0) for x in sl), default=0)
    rep["hands_model_training_accuracy"] = {f"{r['twin']}-c{r['cycle']}": r["hands_acc"] for r in sl if "hands_acc" in r
                                            and r.get("chooser_update") == last_u}
    rep["learned_authored_primitive_share"] = 0.0           # the executor is still scripted (reported plainly)
    last = C[-1]
    if "learning_chooser_off" in last["probes"]:
        p = last["probes"]
        rep["gate2_last_cycle"] = {fam: paired(p["learning"][fam], p["learning_chooser_off"][fam], "steps")
                                   for fam in ("fresh", "transfer", "situations")}
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "primary_final": rep["primary_final"]}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--twin", required=True, help="directory holding the S2b individual (cb2-ind-0001)")
    r.add_argument("--bootstrap", required=True, help="archive root holding lived-cb2-ind-0001")
    r.add_argument("--out", required=True)
    r.add_argument("--cycles", type=int, default=10)
    r.add_argument("--lives", type=int, default=20)
    r.add_argument("--holdout-lives", type=int, default=2, help="per cycle, reserved for memory tests, never slept on")
    r.add_argument("--workers", type=int, default=30)
    r.add_argument("--probe-worlds", type=int, default=40)
    r.add_argument("--situation-worlds", type=int, default=24)
    r.add_argument("--core-updates", type=int, default=1500)
    r.add_argument("--anchor-updates", type=int, default=1000)
    r.add_argument("--chooser-updates", type=int, default=1500)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--T", type=int, default=256)
    r.add_argument("--map-radius", type=int, default=8)
    r.add_argument("--tau", type=float, default=1.0)
    r.add_argument("--horizon", type=int, default=15)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--max-steps", type=int, default=10000)
    r.add_argument("--rehearsal", action="store_true",
                   help="smoke blocks and practice run 99; the report can never make a claim (engineering checks only)")
    r.add_argument("--final-worlds", type=int, default=1200, help="final battery worlds per family (L1b primary)")
    r.add_argument("--chooser-off-worlds", type=int, default=300, help="of those, Gate 2 ablation worlds per family")
    r.set_defaults(func=run)
    fb = sub.add_parser("final", help="the final battery, after the loop (any machine holding the run directory)")
    fb.add_argument("--out", required=True)
    fb.add_argument("--workers", type=int, default=30)
    fb.set_defaults(func=final)
    bq = sub.add_parser("bench-sleep")
    bq.add_argument("--twin", required=True, help="a cb2-ind-0001 directory (copied, never changed)")
    bq.add_argument("--own", required=True, help="an archive of its own lives (with .cmd.npy sidecars)")
    bq.add_argument("--bootstrap", required=True)
    bq.add_argument("--out", required=True)
    bq.add_argument("--updates", type=int, default=100)
    bq.add_argument("--concurrent", type=int, default=1, help="sleeps at once on the same device (2 = one shared GPU)")
    for k, v in (("--batch", 16), ("--T", 256), ("--map-radius", 8), ("--horizon", 15)):
        bq.add_argument(k, type=int, default=v)
    bq.add_argument("--tau", type=float, default=1.0)
    bq.set_defaults(func=bench_sleep)
    bl = sub.add_parser("bench-lives", help="measure probe-life throughput on this machine (timing worlds, never scored)")
    bl.add_argument("--twin", required=True, help="directory holding cb2-ind-0001 (read only)")
    bl.add_argument("--workers", default="30", help="worker counts to measure, e.g. 8,16,30")
    bl.add_argument("--lives", type=int, default=60)
    bl.add_argument("--max-steps", type=int, default=10000)
    bl.set_defaults(func=bench_lives)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--report", default=str(REPO / "evidence/explore/cb2/l1b-report.json"))
    q.set_defaults(func=report)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
