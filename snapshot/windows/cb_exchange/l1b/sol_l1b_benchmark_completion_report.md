# L1b benchmark completed and secured

Sol independently verified completion on October 11, 2026, at 7:41 pm Central (October 12 at 00:41 UTC). Pod `zcy64urgffgd9q` completed the bounded benchmark, its results were downloaded with a matching checksum, and the controller deleted it at 00:41:10 UTC. A fresh RunPod read returned 404 for this exact pod; a live account listing also contained no benchmark pod. Compute billing for this pod has ended. The independent monitor is paused.

The pod ran for about 43 minutes at its assigned $0.89/hour rate: approximately **$0.64 in compute**, estimated from creation to verified deletion. This is not an invoice total; storage or other charges have not been reconciled. The two-hour/$3 ceiling was respected. The unsuccessful Community allocation created no pod; the Secure fallback supplied the benchmark.

## Independent evidence

- Frozen revision: `c59b72f10067acf5a96f525aab6a97739ca3fc78`; archived creation metadata records a clean checkout.
- Retained archive: `/home/davei/cb_data/l1b-bench/cb2_l1b_bench.tgz` in WSL-native storage, 1,609,159,160 bytes.
- Independent archive SHA-256: `01487800f755627134396308c9b48c455d415be09b44443335a0244bf860170b`, matching both the remote completion log and controller download check.
- Archive traversal completed: 175 members; eight content-addressed weight files matched their filename hashes; the portable origin manifest matched the creation lock.
- Rehearsal cycles 0 and 1, both sleeps, final evaluation and report are present. The report retains **“rehearsal: no claim”**, practice run 99, instrument equality and identity/rollback checks.
- All inspected saved JSON numbers were finite; both rehearsal chooser logs recorded zero cumulative skipped updates. Fresh pre-deletion SSH checks also found finite logged numbers and zero recorded skips for both concurrent 100-update sleeper benchmarks.
- Startup: 37 checks passed; RTX 4090, PyTorch 2.9.1+cu128, CUDA available, AMP and TF32 disabled. Actual cgroup CPU quota was 10, with nine workers. The benchmark JSON's `cpu_quota: 96` records host-visible CPUs and must not be used as allocated CPU capacity.

## Measured throughput

| Measurement | Result |
|---|---:|
| Two concurrent sleeps, 100 updates per phase | 455 seconds wall time |
| Projected full paired sleep, 1,500 core / 1,000 anchor / 1,500 chooser updates | 92.8 minutes per cycle |
| Peak GPU memory per sleeper | 8,356 MB |
| Peak host RSS per sleeper | 10,195–10,217 MB |
| Frozen S2b, 60 lives, four workers | 277.4 seconds; 112.6 steps/second |
| Frozen S2b, same 60 lives, nine workers | 145.5 seconds; 214.8 steps/second |
| Rehearsal learner, 60 lives, nine workers | 502.3 seconds; 60.5 steps/second |

The learner took about **3.55 times as much wall time per step** as frozen S2b. A full-loop estimate must use retained learner throughput for learned arms and frozen throughput for the frozen arm, rather than scaling every arm from S2b. The sleep projection is an extrapolation from a short timing sample; evolving policies and life lengths can change throughput.

## What this establishes and what remains

This establishes successful startup, a complete rehearsal, short GPU numerical checks, realistic concurrent sleeper timing, learned-controller CPU timing, artifact transfer and verified cleanup. It does not establish behavioral growth, general memory or learned primitive control. Canonical S2b is unchanged; no production loop or sealed experiment ran.

The rehearsal used only 50 updates per phase and the sleeper timing used 100 per phase. These GPU checks are not a full-length numerical replay. The archive retains sleeper summary JSON, but not the separate `bench-sleep/p0` and `p1` checkpoints or progress logs; their zero-skip/finiteness evidence was inspected remotely before deletion. Weight hash checks establish byte integrity, not an independent tensor-finiteness scan.

Claude owns the next analysis: a measured price and time estimate for the production loop and final evaluation, with a concrete spending cap and cleanup plan. The full run remains unlaunched pending that priced plan. No additional paid experiment or simulation was performed during verification.

Companion evidence: `L1b_Benchmark_Independent_Verification.json`.
