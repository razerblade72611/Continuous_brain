#!/usr/bin/env python3
"""Print a RunPod plan: commands, resource and cost estimates. NEVER launches anything.

Paid cloud work is started by David, by hand, after reading this output
(Joint Contract §10: no generated script automatically launches paid work).
This script makes no network calls and holds no credentials.

Principle: replication and falsification first, model size second. E0-E2 SNN
workloads are small, so many independent replicates on mid-range GPUs beat a
few large ones. 48-80 GB GPUs only pay off once a coprocessor is loaded (E5).

Example:
  python scripts/cloud/runpod_plan.py --job battery --replicates 10 --gpu-hourly 0.44
"""

from __future__ import annotations

import argparse
import math

JOBS = {
    # battery sim_s: full BatteryConfig consumes ~23,200 frames x 10 ms per substrate (B2-B6 summed).
    # measured: full-size E0 engine 3.9 ms per 1 ms step on a 2-core cloud CPU (README); GPU figures are
    # planning assumptions to be replaced by a measured smoke run on the pod before any batch
    "e0_gate": {"sim_s": 70, "ms_per_step_gpu": 1.0, "vram_gb": 2, "cmd": "python scripts/e0_gate.py --device cuda --out runs/e0_gate"},
    "battery": {"sim_s": 240, "ms_per_step_gpu": 1.0, "vram_gb": 3,
                "cmd": "python scripts/battery_smoke.py --size full --device cuda --replicate {rep} --out runs/battery_full"},
    "a1_pilot": {"sim_s": 0, "cpu_hours": 3.0, "vram_gb": 0,
                 "cmd": "python scripts/a1_pilot.py --pairs 100 --out runs/a1_pilot"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=sorted(JOBS), required=True)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--gpu-hourly", type=float, required=True, help="USD per GPU-hour quoted by RunPod for the chosen GPU")
    ap.add_argument("--parallel-per-gpu", type=int, default=4, help="independent replicates packed on one GPU")
    ap.add_argument("--budget-usd", type=float, default=1000.0)
    ap.add_argument("--repo-url", default="https://github.com/razerblade72611/Continuous_brain.git")
    args = ap.parse_args(argv)
    job = JOBS[args.job]
    if "cpu_hours" in job:
        hours_each = job["cpu_hours"]
    else:
        hours_each = job["sim_s"] * 1000 * job["ms_per_step_gpu"] / 1000 / 3600 * 1.5     # 1.5x overhead
    gpus = math.ceil(args.replicates / max(args.parallel_per_gpu, 1))
    wall_h = hours_each * math.ceil(args.replicates / (gpus * args.parallel_per_gpu))
    cost = gpus * wall_h * args.gpu_hourly
    print("# RunPod plan (nothing has been launched)")
    print(f"# job={args.job} replicates={args.replicates} gpus={gpus} est_wall_h={wall_h:.2f} "
          f"est_cost_usd={cost:.2f} (budget {args.budget_usd:.0f}; {100 * cost / args.budget_usd:.1f}%)")
    print(f"# VRAM per replicate ~{job['vram_gb']} GB (assumption until a measured smoke run)")
    if cost > 0.25 * args.budget_usd:
        print("# WARNING: above a quarter of the budget: David approves before launching")
    print("\n# 1. Create the pod yourself in the RunPod console (PyTorch 2.x template), then open its terminal.")
    print("# 2. On the pod:")
    print(f"git clone {args.repo_url} ~/Continuous_brain && cd ~/Continuous_brain")
    print("bash scripts/cloud/pod_setup.sh")
    print("# 3. Measure before batching (replaces the planning assumptions above):")
    print("python scripts/battery_smoke.py --size small --items B3 --device cuda --out runs/pod_smoke")
    print("# 4. The job:")
    for rep in range(args.replicates):
        print(job["cmd"].format(rep=rep) + (" &" if args.parallel_per_gpu > 1 else ""))
    print("wait")
    print("# 5. Bring results home (from your machine, WSL):")
    print("#   scp -P <pod ssh port> -r root@<pod ip>:~/Continuous_brain/runs ~/projects/Continuous_brain/runs_pod")
    print("# 6. Stop the pod in the console when the job finishes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
