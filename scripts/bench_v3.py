"""Gate v3 throughput benchmark: seconds per training step and per probe, for each arm, on this device.
Prints one JSON line with per-arm timings and the projected GPU-hours for the dev plan in lab/DECISIONS_v3.md.

  python scripts/bench_v3.py [--steps 30]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from being.v3.core import Being3, V3Config, backbone_hash, load_qwen  # noqa: E402
from being.v3.run import background_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30)
a = ap.parse_args()
model, tok = load_qwen()
dev = next(model.parameters()).device
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
bg = background_pool()
text = "In free time, Symneth Piask enjoys yythcraft."
res = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
       "backbone_sha256": backbone_hash(model)}
for arm, extra in [("T", {}), ("S", {"lr": 200.0, "s_fit": True}), ("L", {"lr": 1e-2})]:
    b = Being3(model, tok, V3Config(arm=arm, **extra), birth_seed=0)
    b.birth(bg[:100])
    for _ in range(3):
        b.step(text, teacher_text=bg[0])
    sync(); t = time.time()
    for i in range(a.steps):
        b.step(text if i % 2 == 0 else bg[i], teacher_text=bg[i + 1])
    sync(); step = (time.time() - t) / a.steps
    t = time.time()
    for _ in range(5):
        b.generate("Symneth Piask's hobby is")
    sync(); gen = (time.time() - t) / 5
    b.close()
    res[arm] = {"step_s": round(step, 4), "gen12_s": round(gen, 3)}
# projection: steps per 100-life stream = 1000 facts x 4 phrasings x (1 + night 4) x 2 (rehearsal);
# probes per stream ~ 5 checkpoints x (30 facts x 3 gens + 40 comp + 20 neg) generations
steps = 1000 * 4 * 5 * 2
gens = 5 * (30 * 3 + 40 + 20)
for arm in ("T", "S", "L"):
    res[arm]["stream_min"] = round((steps * res[arm]["step_s"] + gens * res[arm]["gen12_s"]) / 60, 1)
mean = sum(res[x]["stream_min"] for x in ("T", "S", "L")) / 3
res["dev_gpu_hours"] = round((36 * mean / 4 + 80 * mean) / 60, 1)   # stage A at 25 lives, stage B at 100
res["sealed_gpu_hours"] = round(70 * mean / 60, 1)
print(json.dumps(res))
