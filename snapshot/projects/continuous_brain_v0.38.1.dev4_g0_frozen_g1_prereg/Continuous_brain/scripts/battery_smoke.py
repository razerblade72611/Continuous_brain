#!/usr/bin/env python3
"""Run battery items B1-B6 on both substrates and record content-addressed runs.

ENGINEERING VALIDATION ONLY. Results carry status ENGINEERING_ONLY (or
CAPABILITY_ABSENT / BLOCKED) and metric status UNREGISTERED_METRIC; nothing
here is evidence for H1 or H2. Seeds come from the v0.38-ENGINEERING family.

Examples (inside WSL, repo root, venv active):
  python scripts/battery_smoke.py --size small --out runs/battery_smoke
  python scripts/battery_smoke.py --size full --items B3 B6 --out runs/battery_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuous_brain_eval.battery import ITEMS, BatteryConfig  # noqa: E402
from continuous_brain_eval.runs import RunStore  # noqa: E402
from continuous_brain_eval.streams import StreamConfig  # noqa: E402
from continuous_brain_protocols import sha256_json  # noqa: E402
from continuous_brain_snn.config import E0Config, small_config  # noqa: E402
from continuous_brain_snn.engine import SNNEngine  # noqa: E402
from continuous_brain_snn.provenance import code_revision  # noqa: E402
from continuous_brain_substrates import (DenseRNNConfig, DenseRNNSubstrate, EncoderConfig,  # noqa: E402
                                         SNNSubstrate)

SIZES = {
    "small": dict(develop_frames=300, eval_frames=600, b4_eval_frames=3000, b4_cue_interval_s=(0.4, 0.6),
                  b4_delays_frames=(3, 10, 30), b6_recovery_frames=300, b3_continue_frames=100),
    "full": dict(),
}


def resolve_device(name: str):
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--items", nargs="*", default=sorted(ITEMS))
    ap.add_argument("--substrates", nargs="*", default=["snn", "dense"], choices=["snn", "dense"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda | dml (the SNN engine device; dense runs on cpu/cuda)")
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--replicate", type=int, default=0, help="part of the run spec: change it to rerun a spec")
    ap.add_argument("--out", default="runs/battery_smoke")
    args = ap.parse_args(argv)

    e0 = (lambda s: small_config(s)) if args.size == "small" else (lambda s: E0Config(seed=s))
    device = resolve_device(args.device)
    cfg = BatteryConfig(seed=args.seed, **SIZES[args.size])
    enc = EncoderConfig.from_anatomy_config(e0(args.seed).anatomy)
    stream = StreamConfig(encoder=enc)

    def snn(seed):
        return SNNSubstrate(SNNEngine(e0(seed), device=device))

    target = snn(args.seed).budget().active_synapses
    dense_device = "cuda" if args.device == "cuda" else "cpu"

    def dense(seed):
        return DenseRNNSubstrate(enc, DenseRNNConfig(seed=seed, target_learned_scalars=target, device=dense_device))

    factories = {"snn": snn, "dense": dense}
    store = RunStore(args.out)
    summary = []
    for name in args.substrates:
        for item in args.items:
            spec = {"kind": "battery", "item": item, "substrate": name, "size": args.size, "seed": args.seed,
                    "battery_config_hash": cfg.config_hash(), "stream_config_hash": stream.config_hash(),
                    "code": code_revision(), "device": str(args.device), "replicate": args.replicate}
            run = store.create(spec)
            t0 = time.time()
            fn = ITEMS[item]
            rec = fn(factories[name], cfg) if item == "B1" else fn(factories[name], stream, cfg)
            run.artifact("metrics.json").write_text(json.dumps(rec.metrics, indent=1, default=str))
            manifest = store.finalize(run, rec)
            summary.append({"item": item, "substrate": name, "status": rec.status.value, "run_id": run.run_id,
                            "seconds": round(time.time() - t0, 1)})
            print(json.dumps(summary[-1]))
    print(f"registry: {store.registry}  (spec digest of this invocation {sha256_json(vars(args))[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
