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
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
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

RUN = 9
FRESH0, TRANSFER0, SITU0 = 78100, 78600, 79100        # development probe worlds (never practice, never sealed)
TRANSFER = {"arrival_rate0": 2 / 110, "arrival_growth": 0.3, "arrival_rate_max": 1 / 20,
            "family_weights": {"collect": 0.10, "craft": 0.06, "place": 0.07, "maintain": 0.25, "avoid": 0.30,
                               "visit": 0.08, "report": 0.14}}
SITU_STEPS = 300
_BOOT: dict = {}


def _sealed(seed):
    return any(a <= seed <= b for a, b in RESERVED if a >= 90000)


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
        ind, _ = _actor(job["home"], job["life0"] + i, job.get("chooser_on", True))
        cfg = W1Config(**job.get("config", {}))
        env = W1Env(cfg)
        stats = {"changed": 0, "steps": 0, "dmg": 0}

        def on_step(n, action, _u, _o, _r, info2, ind=ind):
            c, ch = _cmd(ind)
            cmds.append(c)
            stats["changed"] += int(ch)
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
                h0 = int(info["privileged"]["health"])
                obs, _, done, info = env.step(a, u)
                ev = info.get("events", {}) or {}
                stats["dmg"] += max(0, h0 - int(info["privileged"]["health"])) if (ev.get("zombie_hits") or ev.get("arrow_hits")) else 0
                stats["changed"] += int(_cmd(ind)[1])
                n += 1
            out.append({"seed": seed, "situation": snap["name"], "steps": n, "died": bool(done) and n < SITU_STEPS,
                        "combat_damage": stats["dmg"], "chooser_changed": stats["changed"]})
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
        out.append({"seed": seed, "steps": rec["steps"], "died": bool(rec["done"]) and rec["steps"] < job.get("steps", 10000),
                    "n_unlocked": s["crafter"]["n_unlocked"], "unlocked": sorted(s["crafter"]["unlocked"]),
                    "fulfilled": s["obligations"]["fulfilled"], "failed": s["obligations"]["failed"],
                    "combat_hits": stats["dmg"], "chooser_changed": stats["changed"]})
    if writer is not None:
        np.save(Path(job["archive"]) / f"{job['stream']}.cmd.npy", np.asarray(cmds, np.int8))
    return out


def submit_lives(pool, home, seeds, kind, life0, workers, **kw):
    """Queue a set of lives on the pool now; returns gather() -> the sorted results.  Queuing every set of a cycle
    before gathering keeps all workers busy (one barrier per cycle instead of one per set); the chunking, seeds and
    life ids are exactly those of run_lives, so the results are the same."""
    chunks = [seeds[k::workers] for k in range(workers) if seeds[k::workers]]
    futs = []
    for k, ch in enumerate(chunks):
        j = {"home": str(home), "seeds": ch, "kind": kind, "life0": life0 + k * 1000, **kw}
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
        want = {"dusk_outside": None, "hungry": None, "zombie_near": None}
        n, done = 0, False
        while n < 3000 and not done and any(v is None for v in want.values()):
            pv = info["privileged"]
            pos = pv["player_pos"]
            cond = {"dusk_outside": dl.is_dark(int(obs["step"])) and not pv["sleeping"],
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


def _cpu_quota() -> int:
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, min(os.cpu_count() or 1, int(int(q) / int(per))))
    except Exception:
        pass
    return os.cpu_count() or 1


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
    from continuous_brain_cb2.data import SequenceSampler
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
    s_own = SequenceSampler(Archive(own_archive), own, T=T, seed=cycle, namespaces=ns, map_crop=True, preload=dev == "cuda")
    key = (str(bootstrap), T)
    if key not in _BOOT:                                    # its past before the split does not change: load it once
        _BOOT[key] = SequenceSampler(Archive(bootstrap), boot, T=T, seed=77, namespaces=ns, map_crop=True,
                                     preload=dev == "cuda")
    s_boot = _BOOT[key]
    s_boot.rng = np.random.default_rng(77 + cycle)          # the same past batches for every twin's sleep this cycle
    for s in s_own.streams:                                 # the command it chose at every row (wake sidecar)
        s.cols["command"] = np.load(Path(own_archive) / f"{s.stream}.cmd.npy").astype(np.int64)
        assert len(s.cols["command"]) == len(s), s.stream
        for col in ("reward", "done"):                      # bad numbers in its own lives would poison the value head
            if col in s.cols and not np.isfinite(np.asarray(s.cols[col], float)).all():
                raise RuntimeError(f"{twin} sleep {cycle}: non-finite {col} in {s.stream}")
    log = []
    t0 = time.time()
    # L-A: the core continues, alternating its recent lives with its whole past
    for u in range(1, a.core_updates + 1):
        with amp():
            m = core.train_step((s_own if u % 2 else s_boot).batch(a.batch))
        if u % 250 == 0 or u == a.core_updates:
            ev = core.evaluate(s_own.batch(a.batch))
            log.append({"twin": twin, "cycle": cycle, "core_update": u, "s": round(time.time() - t0), "train": m,
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
            log.append({"twin": twin, "cycle": cycle, "anchor_update": u, "s": round(time.time() - t0), "kl": float(loss),
                        "agreement": agree})
    pol.eval()
    # L-C: its hands and the value of its states, from its own lives only
    ch = ch0 if ch0 is not None else Chooser(ChooserConfig(tau=a.tau, horizon=a.horizon), core.feat_size, dev)
    ch.to(dev); ch.device = dev
    ch.train()
    skipped = 0
    for u in range(1, a.chooser_updates + 1):
        b = s_own.batch(a.batch)
        f = feats(core, b)[:, burn:]
        tt = lambda x, dt=torch.int64: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        # the window's "reward" is the previous step's (the core's input): the reward of transition t is at t + 1
        r_next = np.concatenate([b["reward"][:, burn + 1:], np.zeros_like(b["reward"][:, :1])], 1)
        m = ch.train_step(f, tt(b["command"][:, burn:]), tt(b["action"][:, burn:]), tt(r_next, torch.float32),
                          (1 - tt(b["cont"][:, burn:], torch.float32)).bool(), tt(b["is_first"][:, burn:], torch.float32))
        skipped += m.get("skipped", 0)
        if skipped > max(5, a.chooser_updates // 100):     # tripwire: stop loudly rather than learn from bad numbers
            raise RuntimeError(f"{twin} sleep {cycle}: {skipped} non-finite chooser updates by update {u}")
        if u % 250 == 0 or u == a.chooser_updates:
            log.append({"twin": twin, "cycle": cycle, "chooser_update": u, "s": round(time.time() - t0),
                        "skipped_so_far": skipped, **m})
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
    lineage = {"kind": "cb2-l1-sleep", "twin": twin, "cycle": cycle, "learned_from": {"own_rows": s_own.rows(), "bootstrap_rows": s_boot.rows(),
                                                                        "own_streams": sorted(s.stream for s in s_own.streams)},
               "updates": {"core": a.core_updates, "anchor": a.anchor_updates, "chooser": a.chooser_updates}, "code": git_rev()}
    _consolidate(twin_home, ind, meta, vision(), weights, lineage, candidate=True)
    fd = os.open(out / "sleep.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)   # one append: twins may sleep at once
    try:
        os.write(fd, "".join(json.dumps(r) + "\n" for r in log).encode())
    finally:
        os.close(fd)
    return {"cycle": cycle, "seconds": round(time.time() - t0), "last": log[-6:]}


# ================================================================================================ bench
def bench_sleep(a) -> int:
    """One sleep at full batch/T on a scratch copy of a twin, with few updates per phase, timed per phase.  For
    sizing a pod (and checking the GPU path) before renting the long run.  L1_TF32 / L1_AMP select precision."""
    out = Path(a.out)
    home = out / "twin" / "cb2-ind-0001"
    if home.parent.exists():
        shutil.rmtree(home.parent)
    shutil.copytree(Path(a.twin), home)
    ns = argparse.Namespace(**{**vars(a), "core_updates": a.updates, "anchor_updates": a.updates,
                                "chooser_updates": a.updates})
    t = time.time()
    r = sleep(home, Path(a.own), Path(a.bootstrap), 1, out, ns, "bench")
    log = [json.loads(l) for l in open(out / "sleep.jsonl")]
    ends = {k: max(x["s"] for x in log if k in x) for k in ("core_update", "anchor_update", "chooser_update")}
    per = {"core": ends["core_update"] / a.updates, "anchor": (ends["anchor_update"] - ends["core_update"]) / a.updates,
           "chooser": (ends["chooser_update"] - ends["anchor_update"]) / a.updates}
    full = 1500 * per["core"] + 1000 * per["anchor"] + 1500 * per["chooser"]
    print(json.dumps({"tf32": os.environ.get("L1_TF32") == "1", "amp": os.environ.get("L1_AMP") == "1",
                      "updates_per_phase": a.updates, "sec_per_update": {k: round(v, 3) for k, v in per.items()},
                      "wall_s": round(time.time() - t), "projected_full_sleep_min": round(full / 60, 1),
                      "last_core": [x for x in log if "core_update" in x][-1]["train"]}, default=str))
    return 0


# ================================================================================================ run
def run(a) -> int:
    import pickle
    import uuid
    out = Path(a.out).resolve()
    if (out / "run.json").exists() and not a.resume:
        raise SystemExit(f"{out} exists (use --resume to continue it)")
    out.mkdir(parents=True, exist_ok=True)
    # learning: sleeps on its own accumulating lives; reference (Sol, DAV-98): the same sleeps, update counts and
    # budget, but on a FIXED buffer (the cycle-0 lives, never its later experience); frozen: never sleeps
    twins = {"learning": out / "twins/learning", "reference": out / "twins/reference", "frozen": out / "twins/frozen"}
    for t in twins.values():
        if not t.exists():
            shutil.copytree(Path(a.twin) / "cb2-ind-0001", t / "cb2-ind-0001")
    homes = {k: v / "cb2-ind-0001" for k, v in twins.items()}
    own = out / "archive-learning"
    own.mkdir(exist_ok=True)
    fixed_buf = out / "archive-reference"
    meta0 = json.loads((homes["frozen"] / "individual.json").read_text())
    prev = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    a.run_id = prev.get("run_id") or uuid.uuid4().hex          # this run's identity; every writer checks it
    (out / "run.json").write_text(json.dumps({"format": "cb2-l1/1", "run_id": a.run_id, **git_rev(), "twin_source": str(a.twin),
                                              "twin_state_hash": meta0["state_hash"], "bootstrap": str(a.bootstrap),
                                              "args": {k: v for k, v in vars(a).items() if k != "func"}}, indent=1) + "\n")
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
            seeds = list(range(SITU0, SITU0 + a.situation_worlds))
            ks = [k for k in range(a.workers) if seeds[k::a.workers]]
            parts = list(pool.map(build_bank, [str(homes["frozen"])] * len(ks), [seeds[k::a.workers] for k in ks],
                                  [out / f"bank-{k}.pkl" for k in ks]))
            bank = sorted([x for p in parts for x in p], key=lambda s: (s["seed"], s["name"]))
            with open(bank_f, "wb") as f:
                pickle.dump(bank, f)
        bank = pickle.load(open(bank_f, "rb"))
        situ_ids = [s["seed"] * 10 + ("dusk_outside", "hungry", "zombie_near").index(s["name"]) for s in bank]
        start = len([l for l in open(out / "cycles.jsonl")]) if (out / "cycles.jsonl").exists() else 0
        for c in range(start, a.cycles + 1):
            t0 = time.time()
            rec = {"cycle": c, "twin_hashes": {k: json.loads((h / "individual.json").read_text())["state_hash"] for k, h in homes.items()}}
            # probes (cycle 0 = before any sleep: both twins are the same individual, a sanity check of the instrument)
            fresh = list(range(FRESH0 + 100 * c, FRESH0 + 100 * c + a.probe_worlds))
            trans = list(range(TRANSFER0 + 100 * c, TRANSFER0 + 100 * c + a.probe_worlds))
            arms = {"learning": dict(home=homes["learning"]), "reference": dict(home=homes["reference"]),
                    "frozen": dict(home=homes["frozen"])}
            if c == a.cycles:
                arms["learning_chooser_off"] = dict(home=homes["learning"], chooser_on=False)
            # queue every probe set and the wake lives at once (one barrier per cycle), then gather in order
            pend = {}
            for arm, kw in arms.items():
                pend[arm] = {
                    "fresh": submit_lives(pool, kw["home"], fresh, "life", 20_000_000 + c * 1000, a.workers,
                                          chooser_on=kw.get("chooser_on", True), steps=a.max_steps),
                    "transfer": submit_lives(pool, kw["home"], trans, "life", 21_000_000 + c * 1000, a.workers,
                                             chooser_on=kw.get("chooser_on", True), config=TRANSFER, steps=a.max_steps)}
            if c < a.cycles:
                # wake: the learning twin lives new practice worlds; its lives become its experience
                seeds = [PRACTICE_BASE + RUN * 1_000_000 + c * 10_000 + n for n in range(a.lives)]
                st = json.loads((homes["learning"] / "individual.json").read_text())["state_hash"]
                wake = submit_lives(pool, homes["learning"], seeds, "life", 30_000_000 + c * 1000, a.workers,
                                    archive=str(own), stream=f"c{c:02d}", who="twin-L-of-cb2-ind-0001",
                                    state_hash=st, steps=a.max_steps, run_dir=str(out), run_id=a.run_id)
            for arm, kw in arms.items():
                pend[arm]["situations"] = submit_lives(pool, kw["home"], situ_ids, "situation", 22_000_000 + c * 1000,
                                                       a.workers, chooser_on=kw.get("chooser_on", True), bank=bank)
            rec["probes"] = {arm: {k: g() for k, g in p.items()} for arm, p in pend.items()}
            rec["probe_seconds"] = round(time.time() - t0)
            if c < a.cycles:
                rec["wake"] = {"learning": wake()}
                if c == 0 and not fixed_buf.exists():       # the reference twin's fixed buffer: the cycle-0 lives only
                    shutil.copytree(own, fixed_buf)
                if sleepers:                                 # the two twins sleep at once, each on its own GPU
                    fl = sleepers[0].submit(sleep, homes["learning"], own, Path(a.bootstrap), c + 1, out, a, "learning")
                    fr = sleepers[1].submit(sleep, homes["reference"], fixed_buf, Path(a.bootstrap), c + 1, out, a, "reference")
                    rec["sleep"] = {"learning": fl.result(), "reference": fr.result()}
                else:
                    rec["sleep"] = {"learning": sleep(homes["learning"], own, Path(a.bootstrap), c + 1, out, a, "learning"),
                                    "reference": sleep(homes["reference"], fixed_buf, Path(a.bootstrap), c + 1, out, a, "reference")}
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


def paired(L, F, key):
    m = {(r["seed"], r.get("situation")): r for r in F}
    d = [r[key] - m[(r["seed"], r.get("situation"))][key] for r in L if (r["seed"], r.get("situation")) in m]
    return {"n": len(d), "mean": float(np.mean(d)) if d else None, "lower95": _lb(d)}


def summary_line(rec) -> dict:
    p = rec["probes"]
    out = {}
    for other in ("frozen", "reference"):
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


def report(a) -> int:
    out = Path(a.out)
    C = [json.loads(l) for l in open(out / "cycles.jsonl")]
    rep = {"format": "cb2-l1-report/1", "run": json.loads((out / "run.json").read_text()),
           "cycles": [{"cycle": r["cycle"], **summary_line(r)} for r in C]}
    # primary (l1-prereg.json): pooled over cycles 6-10, learning minus frozen and learning minus reference
    late = [r for r in C if 6 <= r["cycle"] <= 10]
    rep["primary_pooled_cycles_6_10"] = {
        f"vs_{o}": {fam: paired([x for r in late for x in r["probes"]["learning"][fam]],
                                [x for r in late for x in r["probes"][o][fam]], "steps")
                    for fam in ("fresh", "transfer")} for o in ("frozen", "reference")}
    for o in ("frozen", "reference"):                       # situations: per situation, mean over cycles 6-10
        sid = sorted({(x["seed"], x["situation"]) for r in late for x in r["probes"]["learning"]["situations"]})
        mean = lambda arm, k: np.mean([x["steps"] for r in late for x in r["probes"][arm]["situations"]
                                       if (x["seed"], x["situation"]) == k])
        d = [mean("learning", k) - mean(o, k) for k in sid]
        rep["primary_pooled_cycles_6_10"][f"vs_{o}"]["situations"] = {"n": len(d), "mean": float(np.mean(d)) if d else None,
                                                                     "lower95": _lb(d)}
    by = {r["cycle"]: r for r in C}
    if 5 in by and 10 in by:
        rep["retention_situations_c10_minus_c5"] = paired(by[10]["probes"]["learning"]["situations"],
                                                          by[5]["probes"]["learning"]["situations"], "steps")
    sl = [json.loads(l) for l in open(out / "sleep.jsonl")] if (out / "sleep.jsonl").exists() else []
    rep["hands_model_training_accuracy"] = {f"{r['twin']}-c{r['cycle']}": r["hands_acc"] for r in sl if "hands_acc" in r
                                            and r["chooser_update"] == max(x.get("chooser_update", 0) for x in sl)}
    rep["learned_authored_primitive_share"] = 0.0           # the executor is still scripted in L1 (reported plainly)
    last = C[-1]
    if "learning_chooser_off" in last["probes"]:
        p = last["probes"]
        rep["gate2_ablation"] = {fam: paired(p["learning"][fam], p["learning_chooser_off"][fam], "steps")
                                 for fam in ("fresh", "transfer", "situations")}
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["cycles"][-1], indent=1))
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
    r.set_defaults(func=run)
    bq = sub.add_parser("bench-sleep")
    bq.add_argument("--twin", required=True, help="a cb2-ind-0001 directory (copied, never changed)")
    bq.add_argument("--own", required=True, help="an archive of its own lives (with .cmd.npy sidecars)")
    bq.add_argument("--bootstrap", required=True)
    bq.add_argument("--out", required=True)
    bq.add_argument("--updates", type=int, default=100)
    for k, v in (("--batch", 16), ("--T", 256), ("--map-radius", 8), ("--horizon", 15)):
        bq.add_argument(k, type=int, default=v)
    bq.add_argument("--tau", type=float, default=1.0)
    bq.set_defaults(func=bench_sleep)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--report", default=str(REPO / "evidence/explore/cb2/l1-report.json"))
    q.set_defaults(func=report)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
