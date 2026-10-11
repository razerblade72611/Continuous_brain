# Pod runbook (lessons from the CB2-L1 launches, 2026-10-10)

About $15-20 of the L1 spend was avoidable. Each rule below comes from one of those losses.

1. **Size from the pod's real limits, never `nproc`.** Containers report the host's cores (96/64) while the
   cgroup allows 10-40. Read `/sys/fs/cgroup/cpu.max` and `memory.max` for worker counts AND torch threads.
   (Cost: first pod hung 3 h+ from 32 workers x 1.15 GB on 31 GB; sleeps thrashed on 64 threads / 10 cores.)
2. **Time every phase at full scale before renting the long run.** One life (0.13 s/step/core, 1.15 GB) was
   measured; one full sleep (1500+1000+1500 updates, T=256) was not, and it is the bottleneck (~40 min,
   GPU partly idle). A 10-minute benchmark of one sleep would have set the right pod and estimate.
3. **Buy the resource that is the bottleneck.** Lives scale with CPU cores; a sleep's sequential T=256 updates
   do not speed up with more GPUs. Two GPUs (parallel twin sleeps) give nearly all of the 4-GPU benefit here.
4. **Fail loudly, and show progress.** Pools must raise on a dead worker (ProcessPoolExecutor, not mp.Pool);
   logs must be line-buffered with a progress line per batch, so a stall is visible in minutes, not hours.
5. **Test scripts under the shell options they run with** (`set -euo pipefail`): a missing optional cgroup
   file killed a launch after setup.
6. **Billing hygiene.** List pods (`runpodctl pod list`) at every relaunch and delete anything not in use;
   a stuck pod billed ~7 h unnoticed.
7. **Estimates are ranges with their unmeasured input named** ("depends on sleep time, not yet measured"),
   and that input gets measured before money is committed.
8. **Rehearse on the hardware the run uses** (a GPU-only device bug crashed the first GPU sleep; every rehearsal was CPU).
9. **Kill whole process trees, verify, and give runs an identity.** Name-based pkill missed spawned children, which then wrote
   into the next run's directory. Runs carry a run_id that every writer checks; a pod refuses to start with leftovers alive.
10. **Large data stays on the WSL (ext4) side**; only small files cross to Windows folders (cross-OS bulk I/O is slow).
11. **Two agents share this machine and RunPod account:** claim pods/processes/folders in cb_exchange/COORDINATION.md,
    name pods by owner, select pods by exact name, and never stop or delete what the other agent owns.
