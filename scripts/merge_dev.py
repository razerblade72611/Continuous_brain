"""Pool per-seed rows from several dev result files (same config) and apply the preregistered summary checks.

  python scripts/merge_dev.py --config configs/v2_dev_candidate2.json --out results/dev/v2_cand2_summary.json \
      results/dev/v2_cand2_s0-4.json results/dev/v2_cand2_s5-9.json results/dev/v2_cand2_s3-4.json results/dev/v2_cand2_s8-9.json
Refuses mixed configs, duplicate seeds with different results, and non-dev seeds.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.gate_v2 import V2Config, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("files", nargs="+")
a = ap.parse_args()
cfg_path = ROOT / a.config
cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
cfg = V2Config(**json.loads(cfg_path.read_text()))
rows, sources = {}, []
for f in a.files:
    d = json.loads((ROOT / f).read_text())
    assert d["config_sha256"] == cfg_sha, f"{f}: config differs"
    assert d["which"] == "dev", f"{f}: not a dev run"
    sources.append({"file": f, "status": d.get("status"), "commit": d.get("commit"),
                    "seeds": [r["seed"] for r in d["per_seed"]]})
    for r in d["per_seed"]:
        assert 0 <= r["seed"] <= 9
        if r["seed"] in rows:
            assert rows[r["seed"]] == r, f"seed {r['seed']} differs between files"
        rows[r["seed"]] = r
ordered = [rows[s] for s in sorted(rows)]
summ = summarize(ordered, cfg)
backbones = sorted({r["backbone_sha256"] for r in ordered})
summ["checks"]["backbone_unchanged"] = len(backbones) == 1
summ["passed"] = all(summ["checks"].values())
out = {"config_path": a.config, "config_sha256": cfg_sha, "seeds": sorted(rows), "sources": sources,
       "backbone_sha256": backbones, "summary": summ, "per_seed": ordered}
(ROOT / a.out).write_text(json.dumps(out, indent=2))
print(json.dumps({"seeds": sorted(rows), "passed": summ["passed"], "checks": summ["checks"]}, indent=2))
