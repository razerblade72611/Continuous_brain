"""Fresh-session probe in its own process: frozen backbone + config + memory weights file, questions only.

No life text, no replay, no cached activations, no momentum. Prints one JSON line: question-aligned answers,
counts, and the hashes of the loaded memory and backbone.
  python scripts/fresh_probe.py --config configs/v2.json --weights results/weights/dev/seed0_A.pt --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.backbone import backbone_hash, load_backbone  # noqa: E402
from being.gate_v2 import SUBJECT, Being, V2Config, make_life, pool_ids, prefix, probe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = V2Config(**json.loads(Path(a.config).read_text()))
model, tok = load_backbone(str(ROOT / "models/distilgpt2"))
pool = pool_ids(tok)
questions = make_life(a.seed, SUBJECT, cfg.facts, pool)  # answer key for scoring only; nothing is lived
b = Being(model, tok, cfg, birth_seed=a.seed)
b.load_weights(Path(a.weights))
print(json.dumps({"probe": probe(b, questions, pool, SUBJECT, prefix),
                  "memory_sha256": b.memory.weight_hash(), "backbone_sha256": backbone_hash(model)}))
