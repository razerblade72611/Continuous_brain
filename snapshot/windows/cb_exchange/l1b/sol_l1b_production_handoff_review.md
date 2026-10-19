# L1b production handoff review

Sol reviewed Claude's exact revision `e4a82d35040559bf8c0dce533a7aa0189d3ab850` locally on October 11, 2026. No pod or simulation was launched. The versioned bundle SHA-256 is `8bb24753544f3d8434ff28ef959262f121629ff4749037ef31343200ec2b36c2`; all four handoff manifest checksums pass. The WSL-native review checkout remains isolated from canonical S2b and Claude's source.

**Status: corrected production launch revision 53363a4 accepted following David's direct “Approve Claude” instruction.** The lifecycle correction below now passes independent checks. The scientific design remains accepted. Thirty-eight preceding focused integrity/controller/packaging tests passed in 13.15 seconds. The cgroup-v1/v2 CPU quota fix and exported fallback are accepted.

The new watcher correctly stops failed production loop/final pods and preserves their volumes. However, `launch()` still deletes a pod on setup or budget failure even when `--pod-id` adopted a previously claimed pod containing saved work. Independent fixtures reproduced this for both `run` and `final`: unavailable SSH caused a call to `delete_verified(existing-evidence-pod)`. Every cloud and SSH operation was mocked; no resource was touched. Evidence is in `L1b_Loop_Lifecycle_Review_Evidence.json`.

**Closed in 53363a4.** Setup/budget failure now uses verified stop for an adopted pod, retaining its evidence; a genuinely new empty pod still uses verified deletion. The original independent run/final counterexamples now preserve their volumes. The corrected revision's controller suite passed 23 tests in 13.30 seconds, including both adoption cases and fresh-pod failure. Evidence: `L1b_Loop_Lifecycle_Closure_Evidence.json`. No resource was touched by these fixtures.

## Agreed planning estimates

The pricing calculation was independently reproduced using the measured nine-worker learner rate of 60.5 steps/second, frozen rate of 214.8 steps/second and paired sleep projection of 92.8 minutes. The actual benchmark CPU quota was 10, rather than the host-visible count of 96. The quota correction may improve sleeper speed; no unmeasured speedup is credited in these estimates.

| Phase | Point estimate | With Claude's allowances | Proposed compute limit |
|---|---:|---:|---:|
| Loop on one Secure RTX 4090 at $0.89/hour | 23.8 hours; $21.16 | About 26.4 hours; $23.50 | 30 hours / $27 |
| Final on one Secure RTX 4090 | 16.5 hours; $14.65 | About 20.6 hours; $18.30 | 22 hours / $20 |
| Final on 32-vCPU CPU pod, conditional option | 5.6 hours; $5.40 at assumed $0.96/hour | About 8 hours; $7.70 | Decide after loop and connectivity/throughput verification |

The expected compute total is about $31–42, with proposed aggregate compute caps of $47. The already completed benchmark cost about $0.64 in compute. These figures are estimates rather than invoice totals; storage and retained stopped-pod storage must be identified separately. CPU final throughput is extrapolated and its price/connectivity require live verification. Community GPU capacity may lower costs; it is not reserved.

The short sleeper benchmark is a conservative planning measurement, not a proved runtime upper bound or a full numerical replay. Canonical S2b remains unchanged. David has now approved the fresh loop under the concrete spending plan; final evaluation is a separate phase decision.

## Final fresh-launch acceptance

Reviewed locked revision: `4c7eec92e47948c4f11531a48ac5928d44b49b63`, bundle SHA-256 `391cb28fe06320e3772b1494ec35199648cea6f7715320d49127cea61f281fdd`. Locked protocol SHA-256: `25a18b8abd12d8474b54b8790442edd67734411879864036b31f253ad4f6384d`. All three versioned manifest entries pass. The delta from e4a82d3 locks/renames the protocol and updates comment references without executable scientific changes. Sixteen L1b checks passed at this exact revision in 0.36 seconds; those overlap the preceding 38 checks and are not added as independent test coverage.

The preceding 4c7eec9 acceptance is superseded by corrected revision `53363a4ddc5378e8e5a1f0005e3ec21722dad114`, bundle SHA-256 `ac09807b30d0951755e3416c81b641f8d3c094faa8d91944c5ae9ebeed9a8b02`, locked protocol SHA-256 `8a2679fa63594b49b76e2364f459f663c670c17cacebfa3bab5f6c7afd946edd`. All five versioned manifest entries pass. Scientific parameters are unchanged; the cost language now explicitly distinguishes compute caps from storage/invoice totals.

Claude owns the claim and launch decisions for one new `cb2-l1b-loop-claude` pod; Sol is the single WSL executor on his behalf after Claude publishes the promoted package and Sol verifies it. Approval: **30 hours / $27 compute**, expected about 26.4 hours and $23.50 at the measured Secure rate, subject to actual rate verification. No simultaneous creation, silent rerun, live code change, candidate installation or automatic final-stage launch is included. Production failures stop and preserve storage; successful completion requires verified artifact transfer before cleanup. The exact new pod identity and rate will scope independent monitoring.
