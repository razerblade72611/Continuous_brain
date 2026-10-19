"""Run Claude's reimplementation of Grok's v1 gate: original seeds 0-4, fresh seeds 100-119, plus stranger arm."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.backbone import load_backbone  # noqa: E402
from being.gate_v1 import run  # noqa: E402

model, tok = load_backbone()
res = {"original_seeds_0_4": run(model, tok, range(5)), "fresh_seeds_100_119": run(model, tok, range(100, 120))}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/v1_repro.json").write_text(json.dumps(res, indent=2))
for k, v in res.items():
    print(k, v["totals"], "of", v["probes"], "passed_grok_rule:", v["passed_grok_rule"],
          "competence", v["competence_loss_before"], v["competence_loss_after"])
