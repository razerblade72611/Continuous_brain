"""Gate v3 DEV runner (seeds 0-9 only). Sealed seeds and the sealed runner come later, with PREREG_v3.md.

  python scripts/run_v3_dev.py --arm T --seeds 0,1 --set lr=3e-3 night_passes=4 --out results/dev3/T_a.json
  python scripts/run_v3_dev.py --arm T --seeds 0 --stranger --out results/dev3/T_stranger.json

Per seed: wiped baseline (the untouched backbone answering the same probes), then one full life stream.
--stranger: the being lives a DIFFERENT stream (seed + 5000) but is probed on this seed's facts.
Rows are written atomically after every seed; nothing is ever overwritten across files.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.competence import load_sentences, mean_loss  # noqa: E402
from being.v3.core import V3Config, backbone_hash, load_qwen  # noqa: E402
from being.v3.lifegen import make_stream  # noqa: E402
from being.v3 import run as r3  # noqa: E402

DEV = set(range(10))


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def write_atomic(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["T", "S", "L", "G"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--lives", type=int, default=100)
    ap.add_argument("--stranger", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if not set(seeds) <= DEV:
        sys.exit("dev runner accepts seeds 0-9 only")
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = V3Config(arm=a.arm, **{k: parse_val(v) for k, v in (s.split("=", 1) for s in a.set)})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model, tok = load_qwen(threads=a.threads)
    comp = {k: load_sentences(k) for k in ("eval", "modern")}
    base = {k: mean_loss(model, tok, v) for k, v in comp.items()}
    meta = {"which": "dev", "commit": commit, "config": asdict(cfg), "lives": a.lives, "stranger": a.stranger,
            "device": str(next(model.parameters()).device), "backbone_sha256": backbone_hash(model),
            "base_ppl_loss": base, "per_seed": []}
    write_atomic(out, {**meta, "status": "in_progress"})
    for seed in seeds:
        t0 = time.time()
        st = make_stream(seed, n_lives=a.lives)
        probe_lives = [p for p in (1, 10, 50) if p <= a.lives]
        wiped = {f"life{p}": r3.probe_facts(_Wiped(model, tok, cfg), st.lives[p - 1], pools=st.pools) for p in probe_lives}
        row = r3.live_stream(model, tok, cfg, seed + 5000 if a.stranger else seed, n_lives=a.lives,
                             checkpoints=tuple(c for c in (1, 2, 3, 5, 10, 25, 50, 100) if c <= a.lives),
                             probe_lives=probe_lives, competence=comp,
                             probe_stream=st if a.stranger else None)
        row["wiped"] = wiped
        row["probe_seed"] = seed
        row["seconds"] = round(time.time() - t0, 1)
        meta["per_seed"].append(row)
        write_atomic(out, {**meta, "status": "in_progress"})
    write_atomic(out, {**meta, "status": "complete"})


class _Wiped:
    """The untouched backbone, answering through the same probe code (no memory, no hook)."""
    def __init__(self, model, tok, cfg):
        from being.v3.core import Being3
        self.model, self.tok, self.cfg = model, tok, cfg
        self._ids = lambda t: tok(t, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        self.generate = lambda p, n=None: Being3.generate(self, p, n)
        self.answer_logprob = lambda p, a: Being3.answer_logprob(self, p, a)
        self.candidates_logprob = lambda p, c: Being3.candidates_logprob(self, p, c)
        self.dev = next(model.parameters()).device


if __name__ == "__main__":
    main()
