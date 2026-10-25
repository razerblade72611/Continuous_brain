"""Stage A launcher logic with fake job results (no model): phases, selection rule, guard, band, summary."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tmp):
    spec = importlib.util.spec_from_file_location("stage_v3", ROOT / "scripts/stage_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT, m.OUT = tmp, tmp / "results/dev3/stageA"
    return m


def fake_result(arm, cfg, rec10, rec25, ppl):
    seeds = []
    for _ in range(2):
        cps = [{"after_life": 10, "life1": [{"template": i < rec10} for i in range(10)], "ppl_loss": {"eval": 1.0, "modern": 1.0}},
               {"after_life": 25, "life1": [{"template": i < rec25} for i in range(10)], "ppl_loss": {"eval": 1.0 + ppl, "modern": 1.0}}]
        seeds.append({"checkpoints": cps})
    return {"status": "complete", "config": {"arm": arm, **cfg}, "base_ppl_loss": {"eval": 1.0, "modern": 1.0}, "per_seed": seeds}


def test_stage_a_selection(tmp_path, monkeypatch, capsys):
    m = load(tmp_path)

    def fake_run(jobs, parallel, lives, seeds):
        m.OUT.mkdir(parents=True, exist_ok=True)
        paths = {}
        for arm, cfg in jobs:
            n = m.job_name(arm, cfg)
            p = m.OUT / f"{n}.json"
            layer, lr = cfg.get("layer", 0), float(cfg.get("lr", 0))
            if arm == "T":   # layer 14 best; lr 1e-2 learns more but forgets (guard) ; layer 21 out of band
                r = fake_result(arm, cfg, 9 if lr > 5e-3 else 6, (4 if lr > 5e-3 else 7) + (layer == 14), 0.5 if layer == 21 else 0.0)
            elif arm == "S":
                r = fake_result(arm, cfg, 5, 6 if cfg.get("top_t") == 256 else 5, 0.0)
            elif arm == "L":
                r = fake_result(arm, cfg, 5, 8 if cfg["kl_weight"] == 1.0 and lr > 5e-3 else 5, 0.0)
            else:
                r = fake_result(arm, cfg, 5, 8 if cfg.get("replay_cues") == 16 else 6, 0.0)
            p.write_text(json.dumps(r))
            paths[n] = p
        return paths

    monkeypatch.setattr(m, "run_jobs", fake_run)
    monkeypatch.setattr(sys, "argv", ["stage_v3.py", "A"])
    m.main()
    s = json.loads((tmp_path / "results/dev3/stageA_summary.json").read_text())
    fz = s["frozen_for_stage_B"]
    assert fz["T"]["layer"] == 14 and fz["T"]["lr"] == 0.003          # high-lr config rejected by the forgetting guard
    assert fz["S"]["top_t"] == 256 and fz["L"]["kl_weight"] == 1.0 and fz["L"]["lr"] == 0.01
    assert s["phases"]["A3"]["G_base"] == "T" and fz["G"]["replay_cues"] == 16
    assert "STAGE A DONE" in capsys.readouterr().out
