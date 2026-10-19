"""CB2-L1b loop + final-battery price from the measured bench (cb_exchange/l1b/l1b-bench-*.json, 2026-10-12).
Mean life lengths from L1 (cycles 6-10): fresh ~620, transfer ~650, situations ~250 steps.  Arms with an active chooser
(learning, yoked after sleep 1) run at the learner rate; S2b-like arms (frozen; every arm at cycle 0; chooser-off) at
the S2b rate.  Sleeps: the measured two-sleepers-on-one-GPU projection (an upper bound: measured with the CPU-quota bug)."""
import json, sys
B = {"s2b": 214.8, "learner": 60.5}            # steps/s with 9 workers on the 1x4090 secure pod (10-CPU quota)
SLEEP_MIN = 92.8                               # both sleeps at once on one 4090, per cycle (upper bound)
F, T, S = 620, 650, 250
probe = 40 * F + 40 * T + 72 * S               # one arm's per-cycle probe battery (steps)
wake = 20 * F
def secs(steps, kind): return steps / B[kind]
c0 = secs(3 * probe + wake + 2 * F, "s2b") + secs(wake, "s2b")                         # everyone is S2b at cycle 0
cyc = secs(2 * probe + wake + 2 * F, "learner") + secs(probe + wake, "s2b")            # learning+yoked chooser; frozen
last = secs(2 * probe, "learner") + secs(2 * probe, "s2b")                             # + chooser-off arm, no wake/sleep
loop_h = (c0 + 9 * cyc + last) / 3600 + 10 * SLEEP_MIN / 60
fb_chooser = 2 * 1200 * (F + T); fb_fast = 1200 * (F + T) + 300 * (F + T)
gpu_final_h = (secs(fb_chooser, "learner") + secs(fb_fast, "s2b")) / 3600
# 32-vCPU CPU pod: 31 workers; per-worker rate scaled by the measured 4->9 worker efficiency (28.15 -> 23.86, -15%)
eff = 0.85
cpu_rate = {k: v / 9 * 31 * eff for k, v in B.items()}
cpu_final_h = (fb_chooser / cpu_rate["learner"] + fb_fast / cpu_rate["s2b"]) / 3600
out = {"per_cycle_lives_min": round(cyc / 60, 1), "cycle0_lives_min": round(c0 / 60, 1), "sleep_min": SLEEP_MIN,
       "loop_h": round(loop_h, 1), "loop_cost_1x4090_secure_0.89": round(loop_h * 0.89, 2),
       "final_on_gpu_pod_h": round(gpu_final_h, 1), "final_on_gpu_pod_cost": round(gpu_final_h * 0.89, 2),
       "final_on_32vcpu_cpu_pod_h": round(cpu_final_h, 1), "final_on_32vcpu_cost_0.96": round(cpu_final_h * 0.96, 2)}
print(json.dumps(out, indent=1))
