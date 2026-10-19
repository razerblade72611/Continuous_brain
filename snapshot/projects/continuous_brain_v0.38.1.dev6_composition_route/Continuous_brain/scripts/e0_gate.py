#!/usr/bin/env python3
"""E0 exit-gate runner (DAV-5): an INFRASTRUCTURE check, not a scientific claim.

Checks, in order:
  1. genesis      anatomy + connectivity build, semantic firewall, sparse invariants
  2. stability    synthetic physical input for --stability-s; no NaN/Inf; every
                  region's rate inside [--min-hz, --max-hz]
  3. determinism  two fresh engines, same seed -> bit-identical state and raster
  4. continuation checkpoint -> continue vs restore -> continue: identical
  5. structure    prune/activate cycle: shapes unchanged, invariants hold
  6. parity stats --parity-s of plasticity-off activity saved for cross-backend
                  comparison; with --compare-to, compared against another backend
  7. telemetry    every window record is complete (REQUIRED_KEYS)

Writes <out>/<run_id>/e0_gate_report.json, records its SHA-256 in the run
registry, and saves a never-overwrite checkpoint.

Examples (run inside WSL, from the repo root):
  python scripts/e0_gate.py --device cpu --out runs/e0_gate
  python scripts/e0_gate.py --device dml --out runs/e0_gate \
      --compare-to runs/e0_gate/<cpu_run_id>/parity_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuous_brain_snn.anatomy import REGIONS, validate_physical  # noqa: E402
from continuous_brain_snn.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from continuous_brain_snn.config import E0Config, small_config  # noqa: E402
from continuous_brain_snn.engine import NonFiniteStateError, SNNEngine  # noqa: E402
from continuous_brain_snn.inputs import SyntheticDrive  # noqa: E402
from continuous_brain_snn.parity import SpikeRecorder, SpikeStatistics, compare, compute_statistics  # noqa: E402
from continuous_brain_snn.provenance import RunRecord, RunRegistry  # noqa: E402
from continuous_brain_snn.snapshot import SnapshotStore  # noqa: E402
from continuous_brain_snn.telemetry import JsonlLogger, WindowStats, collect  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device(name)


def raster_hash(rec: SpikeRecorder) -> str:
    import hashlib
    steps, neurons = rec.events()
    return hashlib.sha256(np.ascontiguousarray(np.stack([steps, neurons])).tobytes()).hexdigest()


def run_window(engine: SNNEngine, n_steps: int, drive) -> tuple[str, str]:
    rec = SpikeRecorder()
    engine.run(n_steps, input_fn=drive, recorder=rec)
    return engine.state_hash(), raster_hash(rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | dml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--small", action="store_true", help="reduced organism for quick smoke runs")
    ap.add_argument("--stability-s", type=float, default=10.0)
    ap.add_argument("--determinism-steps", type=int, default=500)
    ap.add_argument("--parity-s", type=float, default=60.0)
    ap.add_argument("--min-hz", type=float, default=0.1)
    ap.add_argument("--max-hz", type=float, default=100.0)
    ap.add_argument("--compare-to", type=str, default=None)
    ap.add_argument("--linear-issue", default="DAV-5")
    ap.add_argument("--out", default="runs/e0_gate")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    cfg = small_config(args.seed) if args.small else E0Config(seed=args.seed)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=False)
    store = SnapshotStore(Path(args.out))
    logger = JsonlLogger(out / "telemetry.jsonl")
    checks: dict[str, dict] = {}

    def check(name: str, passed: bool, **detail):
        checks[name] = {"passed": bool(passed), **detail}
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {json.dumps(detail, default=str)[:300]}")

    # 1. genesis
    t0 = time.perf_counter()
    engine = SNNEngine(cfg, device=device)
    problems = validate_physical(engine.anatomy) + engine.slots.check_invariants(cfg.connectivity)
    check("genesis", not problems, n_neurons=engine.n, build_s=round(time.perf_counter() - t0, 2),
          anatomy_hash=engine.anatomy.anatomy_hash(), config_hash=cfg.config_hash(), problems=problems)

    # 2. stability + 7. telemetry
    drive = SyntheticDrive(engine, seed=args.seed)
    window = WindowStats(engine)
    steps_per_window = int(round(1.0 / engine.dt))
    records, stable_err = [], None
    try:
        for _ in range(max(1, int(round(args.stability_s)))):
            engine.run(steps_per_window, input_fn=drive, recorder=window)
            engine.check_finite()
            records.append(collect(engine, window))
            logger.log(records[-1])
            window.reset()
    except (NonFiniteStateError, AssertionError) as exc:
        stable_err = repr(exc)
    last = records[-1] if records else {}
    region_rates = {r: last.get(f"rate_hz/region/{r}") for r in REGIONS}
    in_band = bool(records) and all(args.min_hz <= v <= args.max_hz for v in region_rates.values())
    check("stability", stable_err is None and in_band, seconds=len(records), region_rates_hz=region_rates,
          error=stable_err, ms_per_step=last.get("timing/ms_per_step"))
    check("telemetry", bool(records) and stable_err is None, n_records=len(records))

    # 3. determinism
    a, b = SNNEngine(cfg, device=device), SNNEngine(cfg, device=device)
    ha = run_window(a, args.determinism_steps, SyntheticDrive(a, seed=args.seed))
    hb = run_window(b, args.determinism_steps, SyntheticDrive(b, seed=args.seed))
    check("determinism", ha == hb, state_hash=ha[0], raster_hash=ha[1])

    # 4. checkpoint continuation
    c = SNNEngine(cfg, device=device)
    dc = SyntheticDrive(c, seed=args.seed)
    run_window(c, args.determinism_steps, dc)
    ckpt = save_checkpoint(c, store, label="e0-gate", linear_issue=args.linear_issue)
    gen_state = dc.gen.get_state()
    cont = run_window(c, args.determinism_steps, dc)
    r = load_checkpoint(ckpt, store, device=device)
    dr = SyntheticDrive(r, seed=args.seed)
    dr.gen.set_state(gen_state)
    rest = run_window(r, args.determinism_steps, dr)
    check("continuation", cont == rest, checkpoint=str(ckpt), state_hash=cont[0])

    # 5. structure
    s = SNNEngine(cfg, device=device)
    shapes = tuple(s.slots.src.shape)
    g = torch.Generator().manual_seed(args.seed + 5)
    act = torch.nonzero(s.slots.active.cpu())
    drop = act[torch.randperm(act.shape[0], generator=g)[: act.shape[0] // 20]]
    s.slots.set_active(drop[:, 0].to(s.device), drop[:, 1].to(s.device), False)
    dormant = torch.nonzero((s.slots.candidate & ~s.slots.active).cpu())
    add = dormant[torch.randperm(dormant.shape[0], generator=g)[: drop.shape[0]]]
    s.slots.set_active(add[:, 0].to(s.device), add[:, 1].to(s.device), True)
    s.run(100)
    inv = s.slots.check_invariants(cfg.connectivity)
    check("structure", tuple(s.slots.src.shape) == shapes and not inv, pruned=int(drop.shape[0]),
          activated=int(add.shape[0]), problems=inv)

    # 6. parity statistics (plasticity off)
    pcfg = E0Config.from_dict(cfg.to_dict())
    pcfg.neuron.homeostasis = False
    p = SNNEngine(pcfg, device=device)
    prec = SpikeRecorder()
    n_parity = int(round(args.parity_s / p.dt))
    p.run(n_parity, input_fn=SyntheticDrive(p, seed=args.seed), recorder=prec)
    stats = compute_statistics(p, prec, n_parity)
    (out / "parity_stats.json").write_text(json.dumps(stats.as_json()))
    detail = {"saved": str(out / "parity_stats.json"), "n_correlations": int(stats.correlations.size)}
    if args.compare_to:
        other = SpikeStatistics.from_json(json.loads(Path(args.compare_to).read_text()))
        res = compare(other, stats)
        check("parity", res.passed, against=args.compare_to, ks=res.ks, worst_rate_rel=res.worst_rate_rel,
              worst_cv_abs=res.worst_cv_abs, failures=res.failures[:10])
    else:
        checks["parity"] = {"passed": None, "note": "stats recorded; compare across backends with --compare-to",
                            **detail}
        print(f"[INFO] parity statistics recorded: {detail}")

    gated = [k for k, v in checks.items() if v["passed"] is not None]
    passed = all(checks[k]["passed"] for k in gated)
    report = {"kind": "E0 exit gate (INFRASTRUCTURE)", "run_id": run_id, "device": str(device),
              "config": cfg.to_dict(), "config_hash": cfg.config_hash(),
              "anatomy_hash": engine.anatomy.anatomy_hash(), "checks": checks,
              "result": "INFRASTRUCTURE PASS" if passed else "FAIL",
              "note": "An infrastructure pass establishes no cognitive claim."}
    report_path = out / "e0_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    rec = RunRecord(run_id=run_id, kind="e0_gate", linear_issue=args.linear_issue, seed=cfg.seed,
                    config_hash=cfg.config_hash(), anatomy_hash=engine.anatomy.anatomy_hash(),
                    device=str(device), checkpoint=str(ckpt), status="PASS" if passed else "FAIL")
    digest = rec.attach(report_path)
    rec.attach(out / "parity_stats.json")
    rec.attach(out / "telemetry.jsonl")
    RunRegistry(Path(args.out)).register(rec)
    print(f"\n{report['result']}  report: {report_path}  sha256: {digest}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
