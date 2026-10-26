"""Stage A launcher logic with fake job results (no model): phases, selection rule, guard, band, summary;
plus the 2026-10-25 runner repairs (R1-R5) and the frozen A2-A0 triage rule (Claude + Sol)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(tmp):
    spec = importlib.util.spec_from_file_location("stage_v3", ROOT / "scripts/stage_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT, m.OUT, m.OUT_A0 = tmp, tmp / "results/dev3/stageA", tmp / "results/dev3/stageA2_A0"
    return m


def fake_result(arm, cfg, acq, margin, ppl):
    seeds = []
    for _ in range(2):
        f = lambda k, m=0.0: [{"template": i < k, "fc": i < k, "margin": m, "margin_best": m} for i in range(10)]
        cps = [{"after_life": 1, "life1": f(acq), "ppl_loss": {"eval": 1.0, "modern": 1.0}},
               {"after_life": 10, "life1": f(0, margin + 1), "life10": f(acq), "ppl_loss": {"eval": 1.0, "modern": 1.0}},
               {"after_life": 25, "life1": f(0, margin), "life10": f(0), "ppl_loss": {"eval": 1.0 + ppl, "modern": 1.0}}]
        seeds.append({"checkpoints": cps})
    return {"status": "complete", "config": {"arm": arm, **cfg}, "base_ppl_loss": {"eval": 1.0, "modern": 1.0}, "per_seed": seeds}


def test_stage_a_selection(tmp_path, monkeypatch, capsys):
    m = load(tmp_path)

    def fake_run(jobs, parallel, lives, seeds, out_dir, backbone=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for arm, cfg in jobs:
            n = m.job_name(arm, cfg)
            p = out_dir / f"{n}.json"
            layer, lr = cfg.get("layer", 0), float(cfg.get("lr", 0))
            if arm == "T":   # layer 14 best margin; lr 3e-2 has the biggest margin but does not acquire; layer 21 out of band
                r = fake_result(arm, cfg, 3 if lr > 2e-2 else 8, (9 if lr > 2e-2 else 2) + (layer == 14), 0.5 if layer == 21 else 0.0)
            elif arm == "S":
                r = fake_result(arm, cfg, 7, 3 if cfg.get("top_t") == 256 else 2, 0.0)
            elif arm == "L":
                r = fake_result(arm, cfg, 9, 4 if cfg["kl_weight"] == 1.0 and lr > 2e-4 else 1, 0.0)
            else:
                r = fake_result(arm, cfg, 8, 5 if cfg.get("replay_cues") == 16 else 2, 0.0)
            p.write_text(json.dumps(r))
            paths[n] = p
        return paths, {}

    monkeypatch.setattr(m, "run_jobs", fake_run)
    monkeypatch.setattr(m, "check_clean_source", lambda: None)
    monkeypatch.setattr(m, "preflight_backbone", lambda accept=None: m.EXPECTED_BACKBONE)
    monkeypatch.setattr(sys, "argv", ["stage_v3.py", "A"])
    m.main()
    s = json.loads((tmp_path / "results/dev3/stageA2_summary.json").read_text())
    fz = s["frozen_for_stage_B"]
    assert fz["T"]["layer"] == 14 and fz["T"]["lr"] == 0.01           # lr 3e-2 rejected by the acquisition gate
    assert fz["S"]["top_t"] == 256 and fz["L"]["kl_weight"] == 1.0 and fz["L"]["lr"] == 0.0003
    assert s["phases"]["A3"]["G_base"] == "T" and fz["G"]["replay_cues"] == 16
    assert "STAGE A DONE" in capsys.readouterr().out


def test_stage_a_stops_without_scoring_on_failure(tmp_path, monkeypatch, capsys):
    m = load(tmp_path)
    monkeypatch.setattr(m, "run_jobs", lambda *a, **k: ({}, {"S_x": "exit 1"}))
    monkeypatch.setattr(m, "check_clean_source", lambda: None)
    monkeypatch.setattr(m, "preflight_backbone", lambda accept=None: m.EXPECTED_BACKBONE)
    monkeypatch.setattr(sys, "argv", ["stage_v3.py", "A"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 2
    s = json.loads((tmp_path / "results/dev3/stageA2_summary.json").read_text())
    assert s["failed"]["phase"] == "A1" and "phases" in s and not s["phases"]
    assert "nothing was scored" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------- A0 decision rule
def _m(acq, ret, band=True, seeds=(0, 1), problems=()):
    return {"seeds": {s: {} for s in seeds}, "problems": list(problems), "band": band, "acq_mean": acq, "ret_mean": ret,
            "ret_min_seed": ret, "ret_max_seed": ret}


@pytest.mark.parametrize("override,expected", [
    ({0: _m(7, 3)}, "GO"),
    ({0: _m(7, 1)}, "STOP"),
    ({}, "INCONCLUSIVE"),                                    # nobody acquires
    ({0: _m(7, 1), 5: _m(2, 2)}, "INCONCLUSIVE"),             # another competent config retains 2/10
    ({0: _m(7, 2)}, "INCONCLUSIVE"),                         # acquirer retains 2, not 3
    ({0: _m(7, 3, band=False)}, "INCONCLUSIVE"),             # vacuous-STOP clarification: no competent acquirer
    ({0: _m(7, 3, seeds=(0,))}, "INCONCLUSIVE"),             # seed coverage
    ({0: _m(7, 3, problems=["non-finite"])}, "INCONCLUSIVE"),
])
def test_a0_verdict(tmp_path, override, expected):
    m = load(tmp_path)
    names = [m.job_name(a, c) for a, c in m.A1_CONFIGS]
    metrics = {n: _m(0, 0) for n in names}
    for i, v in override.items():
        metrics[names[i]] = v
    assert m.a0_verdict(metrics, [0, 1])[0] == expected


# ------------------------------------------------------------------------------- R2 reuse binding / R1 R3 R5 runner
def git_repo(tmp, stub):
    (tmp / "being").mkdir(parents=True)
    (tmp / "being/x.py").write_text("X = 1\n")
    (tmp / "scripts").mkdir()
    (tmp / "scripts/run_v3_dev.py").write_text(stub)
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp, check=True, capture_output=True)
    run("init", "-q")
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return run


STUB = r'''
import argparse, json, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--arm"); ap.add_argument("--seeds"); ap.add_argument("--lives", type=int)
ap.add_argument("--out"); ap.add_argument("--set", nargs="*"); a = ap.parse_args()
print("stub", a.arm, a.set)
if "boom=1" in a.set:
    print("torch.OutOfMemoryError: CUDA out of memory (simulated)"); sys.exit(1)
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
cfg = {"arm": a.arm, **{k: v for k, v in (x.split("=", 1) for x in a.set)}}
Path(a.out).write_text(json.dumps({"commit": commit, "config": cfg, "lives": a.lives, "backbone_sha256": "%s",
    "per_seed": [{"probe_seed": int(s), "backbone_start": "b", "backbone_end": "b"} for s in a.seeds.split(",")],
    "status": "complete"}))
'''


def test_result_problem(tmp_path):
    m = load(tmp_path)
    run = git_repo(tmp_path, "pass\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    good = {"commit": head, "config": {"arm": "T", "layer": 7, "lr": 0.01}, "lives": 5, "backbone_sha256": "bb",
            "per_seed": [{"probe_seed": 0, "backbone_start": "b", "backbone_end": "b"},
                         {"probe_seed": 1, "backbone_start": "b", "backbone_end": "b"}], "status": "complete"}
    p = tmp_path / "r.json"
    chk = lambda d, arm="T", cfg={"layer": 7, "lr": 1e-2}, lives=5, seeds="0,1", bb="bb": (
        p.write_text(d if isinstance(d, str) else json.dumps(d)), m.result_problem(p, arm, cfg, lives, seeds, bb))[1]
    assert chk(good) is None
    assert "arm" in chk(good, arm="S")
    assert "lives" in chk(good, lives=25)
    assert "seeds" in chk(good, seeds="0")
    assert "config" in chk(good, cfg={"layer": 14, "lr": 1e-2})
    assert "backbone" in chk(good, bb="other")
    assert "unreadable" in chk("{not json")
    assert "status" in chk({**good, "status": "in_progress"})
    bad = json.loads(json.dumps(good)); bad["per_seed"][0]["backbone_end"] = "c"
    assert "changed during" in chk(bad)
    (tmp_path / "being/x.py").write_text("X = 2\n")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "change experiment source")
    assert "source changed" in chk(good)


def test_run_jobs_failure_reuse_logs(tmp_path, monkeypatch):
    m = load(tmp_path)
    git_repo(tmp_path, STUB % m.EXPECTED_BACKBONE)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    jobs = [("T", {"layer": 7}), ("S", {"layer": 7, "boom": 1}), ("T", {"layer": 14}), ("T", {"layer": 21})]
    out = tmp_path / "results/dev3/stageA2_A0"
    paths, fails = m.run_jobs(jobs, 1, 5, "0,1", out)                    # parallel 1 => strict order
    names = [m.job_name(a, c) for a, c in jobs]
    assert names[0] not in fails and paths[names[0]].exists()
    assert fails[names[1]].startswith("exit 1")
    assert all("not started" in fails[n] for n in names[2:])               # fail fast after the first failure
    jobs[1] = ("S", {"layer": 7, "boom": 1})
    (tmp_path / "scripts/run_v3_dev.py").write_text((STUB % m.EXPECTED_BACKBONE).replace('"boom=1" in a.set', "False"))
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "fix stub"], cwd=tmp_path, check=True)
    paths, fails = m.run_jobs(jobs, 1, 5, "0,1", out)
    assert not fails
    assert list(out.glob(names[1] + ".log.*")), "previous attempt's log must be kept (R3)"
    assert "CUDA out of memory" in next(out.glob(names[1] + ".log.*")).read_text()
    # results from the first run were produced by an older commit whose experiment source (being/, run_v3_dev.py)
    # differs -> set aside as mismatch, never silently reused
    assert list(out.glob(names[0] + ".mismatch.*.json"))


# ------------------------------------------------------------------------------------------------ R4 stop safety
def test_finish_stop_protection(tmp_path, monkeypatch):
    m = load(tmp_path)
    bundle = tmp_path / "results/dev3/stageA2_A0"
    bundle.mkdir(parents=True)
    (bundle / "a.json").write_text("{}")
    summary = tmp_path / "results/dev3/stageA2_A0_summary.json"
    summary.write_text("{}")
    calls, real = [], subprocess.run

    def spy(cmd, *a, **k):
        if cmd and cmd[0] == "runpodctl":
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)
    monkeypatch.setattr(m.subprocess, "run", spy)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    monkeypatch.setattr(m, "PERSIST_DIR", tmp_path)
    assert m.finish(bundle, summary, stop_pod=True) is False and not calls          # local disk: refuse to stop
    assert (tmp_path / "results_stageA2_A0.tar.gz.sha256").exists()
    monkeypatch.setattr(m, "_mount_of", lambda p: (str(tmp_path.resolve()), "mfs#ca-mtl-3:9421", "fuse"))
    assert m.finish(bundle, summary, stop_pod=True) is True and calls == [["runpodctl", "stop", "pod", "pod123"]]
    calls.clear()
    monkeypatch.setattr(m, "PERSIST_DIR", tmp_path / "elsewhere")                   # archive outside the mount
    assert m.finish(bundle, summary, stop_pod=True) is False and not calls


# ------------------------------------------------------------------ A0 evidence validity (fail closed, Sol 4dbf4e8)
def a0_result(acq=7, ret=4, mutate=None):
    probes = lambda k: [{"template": i < acq, "fc": i < k, "margin_best": 0.5 if i < k else -0.5} for i in range(10)]
    cps = [{"after_life": L, "life1": probes(10 if L == 1 else ret), "ppl_loss": {"eval": 1.0, "modern": 1.0}}
           for L in (1, 2, 3, 5)]
    d = {"status": "complete", "base_ppl_loss": {"eval": 1.0, "modern": 1.0},
         "per_seed": [{"probe_seed": s, "backbone_start": "b", "backbone_end": "b", "wiped": {"life1": []},
                       "checkpoints": json.loads(json.dumps(cps))} for s in (0, 1)]}
    if mutate:
        mutate(d)
    return d


@pytest.mark.parametrize("label,mutate", [
    ("empty base monitors", lambda d: d.update(base_ppl_loss={})),
    ("missing base monitor", lambda d: d["base_ppl_loss"].pop("modern")),
    ("missing checkpoint-5 monitor", lambda d: d["per_seed"][1]["checkpoints"][3]["ppl_loss"].pop("modern")),
    ("non-finite monitor", lambda d: d["per_seed"][0]["checkpoints"][3]["ppl_loss"].update(eval=float("nan"))),
    ("incomplete probes", lambda d: d["per_seed"][0]["checkpoints"][2]["life1"].pop()),
    ("probe missing margin_best", lambda d: d["per_seed"][0]["checkpoints"][3]["life1"][0].pop("margin_best")),
    ("missing checkpoint", lambda d: d["per_seed"][1]["checkpoints"].pop(1)),
    ("no wiped baseline", lambda d: d["per_seed"][0].pop("wiped")),
])
def test_a0_metrics_fail_closed(tmp_path, label, mutate):
    m = load(tmp_path)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(a0_result()))
    ok = m.a0_metrics(p)
    assert not ok["problems"] and ok["band"] and ok["acq_mean"] == 7 and ok["ret_mean"] == 4
    p.write_text(json.dumps(a0_result(mutate=mutate)))
    bad = m.a0_metrics(p)
    assert bad["problems"], label
    names = [m.job_name(a, c) for a, c in m.A1_CONFIGS]
    metrics = {n: m.a0_metrics(p) for n in names}                           # every config would otherwise be GO
    assert m.a0_verdict(metrics, [0, 1])[0] == "INCONCLUSIVE", label
