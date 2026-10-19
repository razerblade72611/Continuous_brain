# L1b bounded hardware benchmark

David authorized Sol and Claude to agree on a plan and run a pod on 11 October 2026. Claude's C-014 accepts the sequence below and retains implementation ownership; Sol independently reviews the handoff and executes the launcher on David's WSL environment.

1. Finish local regressions for result binding, portable packaging and claimed-resource cleanup. Verify the versioned bundle, source revision and input hashes. Use only the corrected handoff.
2. Run one RTX 4090 benchmark pod, named cb2-l1b-bench-claude, with no production experiment. Prefer community hosting; fall back to secure hosting if available and within the cap, without leaving a duplicate billable pod.
3. Exercise a clearly labelled rehearsal on its separate smoke worlds, concurrent sleeps at actual batch/sequence settings, and worker scaling with the chooser active. Inspect finite-value and guard evidence before interpreting throughput.
4. Download the benchmark archive into WSL-native storage, verify its checksum, and copy small reports/status into the exchange. Verify termination after normal completion. On a fatal failure, timeout or transfer failure, stop and verify the exact claimed pod and preserve recoverable storage.
5. Use measured throughput and memory requirements to price the full L1b loop and final evaluation. No production loop or canonical installation is included in this benchmark.

**Limits:** two hours from creation, $3 total ceiling. The 23:27 UTC live catalog read lists one RTX 4090 at $0.34/hour community or $0.89/hour secure, both LOW availability. At those rates two hours of compute is $0.68 or $1.78; the ceiling leaves room for storage/setup charges. Check the actual created pod's quoted rate before starting work. Availability is not a reservation.

The controller must own cleanup immediately after creation, verify resource identity, bound network operations, and report unsuccessful lifecycle operations without claiming billing ended. Only the newly created claimed benchmark pod is in scope. Retained L1 pod gjfpfnsfymn8yf remains untouched.

The benchmark tests execution and estimates cost. It makes no growth claim, consumes no sealed worlds, and does not change canonical S2b. The previous local memory diagnostic remains limited evidence for retained perceptual-map information.

Final review passed: c59b72f10067acf5a96f525aab6a97739ca3fc78. Its parent passed 45 focused checks, and the last controller/packaging delta passed 19 focused checks (overlapping scopes). All package checksums match. Plan and final review were recorded on DAV-99 before creation.

**Launched:** pod zcy64urgffgd9q, SECURE RTX 4090 at the verified $0.89/hour rate, created 6:57:57 pm Central on 11 October. Community allocation failed with no pod created, so the agreed secure fallback was used. All remote input/code checksums and 37 startup checks passed; CUDA is available on PyTorch2.9.1+cu128. The rehearsal started at 7:00:12 pm Central with nine workers under a ten-CPU quota and both sleeps sharing the GPU. A fresh live read confirms RUNNING and active GPU computation.

The local controller polls every thirty seconds, checks the budget from creation, and reserves three minutes for verified cleanup. Its deadline is about 8:54:57 pm Central, with the two-hour hard limit at 8:57:57 pm. An independent Codex follow-up also checks every two minutes and stays quiet on healthy progress. The computer must remain awake for these local checks. Claude confirmed the package is frozen and owns benchmark analysis and the subsequent measured loop price.
