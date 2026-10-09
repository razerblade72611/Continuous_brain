#!/usr/bin/env python3
"""Upgrade the canonical individual's skill from r3h-B to the sealed S2b-progress selector (DAV-91), on David's decision.

Run on David's machine, from the repository root, with the explore venv:

  python scripts/explore/cb2_canonical_upgrade.py --weights <8cffde79....pt> --backup-dir <a Windows folder> \\
      [--sealed-report evidence/explore/cb2/s2b-sealed-report.json] [--finish-life]

Stops at the first thing that is not as expected (nothing is changed before step 4):
 1. the weights file is the sealed S2b candidate's (8cffde79...), and the sealed report says S2-PROGRESS (or COMPLETE)
    for exactly those weights, under the locked s2b preregistration, with no deviations and not a rehearsal;
 2. the canonical individual restores to its recorded hash, currently runs r3h-B (ab2baa67...) with vision v1;
    if saved mid-life, --finish-life lives that life to its end first (current skill, archived), else it stops;
 3. rollback point: the pre-upgrade snapshot (name, state hash, per-file SHA-256s) and a .tgz of the whole home;
 4. the upgrade: one normal consolidation event chained in consolidations.jsonl, lineage = the sealed result.
    r3h-B stays recorded as the 'stable' rollback point (DAV-96); the new skill is recorded as 'current';
 5. verification from disk: restores; chain verifies; weights are 8cffde79...; every non-skill channel, the tick
    and the life unchanged; vision still v1.
Gate 4 (prediction / persistent memory) stays open.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_canonical_install import V1, channels, sha_file, stop  # noqa: E402
from cb2_collect import ARCHIVE, INDIVIDUALS, git_rev, vision  # noqa: E402
from cb2_evaluate import _consolidate, load_individual  # noqa: E402

NEW = {"weights_sha256": "8cffde798661d2661c469fa208799561008d03ae860d2d9be7f1c514a3c6853b",
       "candidate_state_hash": "cc8948e7685dac80debbc218f891c2ba2ac2f2f8647d2db85c98f71df811de1a",
       "candidate_event_sha": "299f06162c96b4c21a3c2826371132e94104b8172af68f81b1e3d20e4f02478f",
       "prereg_sha256": "7d22c4d742ab94d548daa5c172c7915b9cffe5de0ba0628b4c00276086ac5b22",
       "sealed_seeds": "90121-90200"}
OLD = "ab2baa674be37970d2f0016f1e10427ac3be785e9dd907e2097449ae95004ee5"     # r3h-B


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sealed-report", default=str(REPO / "evidence/explore/cb2/s2b-sealed-report.json"))
    ap.add_argument("--prereg", default=str(REPO / "evidence/explore/cb2/s2b-prereg.json"))
    ap.add_argument("--individuals", default=str(INDIVIDUALS))
    ap.add_argument("--individual", default="cb2-ind-0001")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--finish-life", action="store_true")
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/canonical-upgrade-s2b.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = Path(a.individuals) / a.individual
    rep = {"format": "cb2-canonical-upgrade/1", "decision": "David: r3h-B -> S2b (sealed S2-PROGRESS), vision v1",
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **git_rev(), "home": str(home),
           "gate4": "OPEN: a better controller does not solve prediction / persistent memory"}

    # 1. weights + sealed result ------------------------------------------------------------------------------------
    wpath = Path(a.weights)
    rep["weights_file_sha256"] = sha_file(wpath)
    if rep["weights_file_sha256"] != NEW["weights_sha256"]:
        return stop("the weights file is not the sealed S2b candidate (sha256 differs)", rep, out)
    rep["prereg_file_sha256"] = sha_file(Path(a.prereg))
    if rep["prereg_file_sha256"] != NEW["prereg_sha256"]:
        return stop("the s2b preregistration file is not the locked one", rep, out)
    s = json.loads(Path(a.sealed_report).read_text())
    c = s["candidate"]
    rep["sealed"] = {"verdict": s["verdict"], "seeds": s["prereg"]["sealed"]["seeds"], "paired": s["paired"],
                     "weights_sha256": c["weights_sha256"], "state_hash": c["state_hash"], "event_sha": c["event_sha"],
                     "deviations": s["deviations"], "REHEARSAL": s["REHEARSAL"]}
    ok_verdict = s["verdict"].startswith("S2-PROGRESS") or s["verdict"].startswith("S2 COMPLETE")
    if not (ok_verdict and not s["deviations"] and not s["REHEARSAL"] and s["prereg"]["sealed"]["seeds"] == NEW["sealed_seeds"]
            and c["weights_sha256"] == NEW["weights_sha256"] and c["state_hash"] == NEW["candidate_state_hash"]
            and c["event_sha"] == NEW["candidate_event_sha"]):
        return stop("the sealed report does not certify these weights", rep, out)

    # 2. the canonical individual --------------------------------------------------------------------------------
    v = vision()
    ind, meta = load_individual(home, v)
    rep["before"] = {k: meta.get(k) for k in ("snapshot", "state_hash", "life", "lives_lived", "skills", "weights_sha256")}
    rep["before"]["in_life"] = bool(meta.get("in_life"))
    if meta.get("skills") != "cb2" or meta.get("weights_sha256") != OLD:
        return stop(f"it does not currently run r3h-B ({meta.get('skills')}, {meta.get('weights_sha256')})", rep, out)
    if ind.vision.weights_sha256 != V1 or (meta.get("organs") or {}).get("vision", {}).get("weights_sha256", V1) != V1:
        return stop("its vision organ is not v1", rep, out)
    if meta.get("in_life"):
        if not a.finish_life:
            return stop("it was saved in the middle of a life: re-run with --finish-life (or finish it in the viewer)", rep, out)
        import torch
        torch.set_num_threads(1)
        from continuous_brain_individual.viewer import LiveSession
        sess = LiveSession(v, {"backend": "dev"}, REPO / "runs/explore/viewer/lives")
        sess.live_as(home, a.archive)
        n0 = int(sess.obs["step"])
        while sess.alive:
            sess.step()
        rep["finished_life"] = {"life": int(sess.ind.life), "world": int(sess.seed), "from_step": n0,
                                "to_step": int(sess.obs["step"]), "death": sess.death}
        print(json.dumps({"finished_life": rep["finished_life"]}), flush=True)
        ind, meta = load_individual(home, v)
        if meta.get("in_life") or meta.get("weights_sha256") != OLD:
            return stop("after finishing the life it is not as expected", rep, out)
    pre = {"snapshot": meta["snapshot"], "state_hash": meta["state_hash"], "life": meta["life"],
           "lives_lived": meta["lives_lived"], "tick": int(ind.rt.tick), "event_sha": meta.get("event_sha")}

    # 3. rollback point -------------------------------------------------------------------------------------------
    snapdir = home / "checkpoints" / meta["snapshot"]
    pre["snapshot_files_sha256"] = {p.name: sha_file(p) for p in sorted(snapdir.iterdir()) if p.is_file()}
    bdir = Path(a.backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    tgz = bdir / f"{a.individual}-preupgrade-s2b-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(home, arcname=a.individual)
    pre["backup"] = {"path": str(tgz), "sha256": sha_file(tgz), "bytes": tgz.stat().st_size}
    rep["rollback_point"] = pre
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"rollback_point": {k: pre[k] for k in ("snapshot", "state_hash", "backup")}}, indent=1), flush=True)
    before_channels, before_tick, before_life = channels(ind), int(ind.rt.tick), int(ind.life)
    stable = meta.get("stable")

    # 4. the upgrade ----------------------------------------------------------------------------------------------
    lineage = {"decision": rep["decision"], "sealed_result": rep["sealed"], "replaces": {"weights_sha256": OLD, "name": "r3h-B"},
               "candidate": {"state_hash": NEW["candidate_state_hash"], "event_sha": NEW["candidate_event_sha"]},
               "weights_file": wpath.name, "code": git_rev(),
               "note": "S2b: r3h-B selector improved by PPO + KL to r3h-B with a terminal death penalty; executor, r2 core, vision v1 unchanged"}
    rc = _consolidate(home, ind, meta, v, wpath.read_bytes(), lineage)
    if rc != 0:
        return stop("the consolidation did not complete", rep, out)
    meta2 = json.loads((home / "individual.json").read_text())
    meta2["stable"] = stable                                  # r3h-B stays the recorded rollback point
    meta2["current"] = {"name": "S2b", "weights_sha256": meta2["weights_sha256"], "event_sha": meta2.get("event_sha"),
                        "snapshot": meta2["snapshot"], "note": "sealed S2-PROGRESS (DAV-91); r3h-B remains 'stable'",
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (home / "individual.json").write_text(json.dumps(meta2, indent=1) + "\n")

    # 5. verification ---------------------------------------------------------------------------------------------
    from continuous_brain_cb2.consolidation import ConsolidationLog
    after, meta3 = load_individual(home, v)
    events = ConsolidationLog(home / "consolidations.jsonl").verify()
    changed = sorted(n for n, d in channels(after).items() if before_channels.get(n) != d)
    missing = sorted(set(before_channels) - set(channels(after)))
    rep["after"] = {"snapshot": meta3["snapshot"], "state_hash": meta3["state_hash"], "event_sha": meta3.get("event_sha"),
                    "prev_event_sha": pre["event_sha"], "weights_sha256": meta3.get("weights_sha256"),
                    "consolidation_events": events, "stable": meta3.get("stable"), "current": meta3.get("current"),
                    "non_skill_channels": len(before_channels), "non_skill_channels_changed": changed,
                    "non_skill_channels_missing": missing, "tick_kept": int(after.rt.tick) == before_tick,
                    "life_kept": int(after.life) == before_life, "vision": after.vision.weights_sha256}
    ok = (meta3.get("weights_sha256") == NEW["weights_sha256"] and events >= 2 and not changed and not missing
          and rep["after"]["tick_kept"] and rep["after"]["life_kept"] and after.vision.weights_sha256 == V1
          and (meta3.get("stable") or {}).get("weights_sha256") == OLD)
    rep["verdict"] = "UPGRADED and verified" if ok else "UPGRADED but verification FAILED: roll back (see rollback_point)"
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "after": rep["after"]}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
