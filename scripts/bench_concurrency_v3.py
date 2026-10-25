"""Gate v3 concurrency benchmark (A2-A0 hardware step, agreed by Claude + Sol 2026-10-25).

Runs N copies of scripts/bench_v3.py at the same time for each N in --levels, records each copy's per-arm seconds per
training step, aggregate throughput, peak GPU memory (nvidia-smi, polled every second), failures, and wall time, and
recommends the level with the lowest cost per training step that kept >= --reserve-gb of memory free.

  python scripts/bench_concurrency_v3.py --levels 1,3,6 --price 0.59 [--steps 30] [--reserve-gb 6]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    return used, total


def run_level(n, steps):
    peak, stop = [0], threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_mem()[0])
            time.sleep(1)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, str(ROOT / "scripts/bench_v3.py"), "--steps", str(steps)], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(n)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.time() - t0
    stop.set()
    th.join()
    ok, failed = [], []
    for p, o in zip(procs, outs):
        line = next((ln for ln in reversed(o.splitlines()) if ln.startswith("{")), None)
        if p.returncode == 0 and line:
            ok.append(json.loads(line))
        else:
            failed.append({"exit": p.returncode, "tail": o.splitlines()[-3:]})
    arms = {}
    for arm in ("T", "S", "L"):
        ss = [r[arm]["step_s"] for r in ok]
        if ss:
            mean = sum(ss) / len(ss)
            arms[arm] = {"step_s_mean": round(mean, 4), "agg_steps_per_s": round(len(ss) / mean, 2)}
    return {"n": n, "ok": len(ok), "failed": failed, "wall_s": round(wall, 1), "peak_mib": peak[0], "arms": arms,
            "gpu": ok[0].get("gpu") if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,3,6")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--price", type=float, required=True, help="GPU $/hour")
    ap.add_argument("--reserve-gb", type=float, default=6.0)
    a = ap.parse_args()
    total_mib = gpu_mem()[1]
    res = {"price_per_hr": a.price, "total_mib": total_mib, "levels": []}
    for n in (int(x) for x in a.levels.split(",")):
        r = run_level(n, a.steps)
        r["free_at_peak_gb"] = round((total_mib - r["peak_mib"]) / 1024, 2)
        agg = [v["agg_steps_per_s"] for v in r["arms"].values()]
        r["usd_per_1k_steps"] = round(a.price / 3600 / (sum(agg) / len(agg)) * 1000, 4) if agg else None
        r["eligible"] = not r["failed"] and r["ok"] == n and r["free_at_peak_gb"] >= a.reserve_gb
        res["levels"].append(r)
        print(json.dumps(r), flush=True)
    good = [r for r in res["levels"] if r["eligible"] and r["usd_per_1k_steps"]]
    res["recommended_parallel"] = min(good, key=lambda r: r["usd_per_1k_steps"])["n"] if good else None
    out = ROOT / "results/dev3/bench_concurrency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("RECOMMENDED --parallel", res["recommended_parallel"], flush=True)


if __name__ == "__main__":
    main()
