"""Dev-only tuning sweep for Gate v2. Uses dev seeds (0-9) and the DEV competence set only; never sealed seeds.

  python scripts/tune_v2.py --seed 0 --grid '{"lr":[1e-4,3e-4,1e-3]}' --out results/dev/tune_lr.json
Reports lived recall (forced/full) and DEV perplexity ratio after awake and after every `every` sleep passes.
"""
import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.backbone import load_backbone  # noqa: E402
from being.competence import load_sentences, mean_loss  # noqa: E402
from being.gate_v2 import (SUBJECT, Being, V2Config, make_life, pool_ids, ppl_ratio, prefix,  # noqa: E402
                           probe, sentence, sleep, live_block)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--grid", required=True)
ap.add_argument("--base", default=str(ROOT / "configs/v2.json"))
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert 0 <= a.seed <= 9, "dev seeds only"

model, tok = load_backbone(str(ROOT / "models/distilgpt2"))
pool = pool_ids(tok)
dev = load_sentences("dev")[:50]
base_dev = mean_loss(model, tok, dev)
modern = load_sentences("modern")[:50]
base_modern = mean_loss(model, tok, modern)
base = V2Config(**json.loads(Path(a.base).read_text()))
grid = json.loads(a.grid)
rows = []
for combo in itertools.product(*grid.values()):
    cfg = replace(base, **dict(zip(grid.keys(), combo)))
    t0 = time.time()
    life = make_life(a.seed, SUBJECT, cfg.facts, pool)
    texts = [sentence(SUBJECT, at, pool[v][0]) for at, v in life]
    rng = random.Random(a.seed * 7 + 1)
    b = Being(model, tok, cfg, birth_seed=a.seed)
    traj = []
    live_block(b, texts, cfg.awake_passes, cfg, rng, rehearse=cfg.rehearse_awake)
    p = probe(b, life, pool, SUBJECT, prefix)
    traj.append({"passes": cfg.awake_passes, "forced": p["forced"], "full": p["full"], "dev_ppl": ppl_ratio(b, dev, base_dev), "modern_ppl": ppl_ratio(b, modern, base_modern)})
    done = 0
    while done < cfg.sleep_passes:
        k = min(a.every, cfg.sleep_passes - done)
        sleep(b, texts, replace(cfg, sleep_passes=k), rng)
        done += k
        p = probe(b, life, pool, SUBJECT, prefix)
        traj.append({"passes": cfg.awake_passes + done, "forced": p["forced"], "full": p["full"],
                     "dev_ppl": ppl_ratio(b, dev, base_dev), "modern_ppl": ppl_ratio(b, modern, base_modern)})
    b.close()
    rows.append({"cfg": asdict(cfg), "trajectory": traj, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({k: v for k, v in rows[-1]["cfg"].items() if k in grid}), traj, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
