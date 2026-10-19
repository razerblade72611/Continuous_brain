# Continuous Brain v0.6.2 — Release validation

Date: 2026-09-04. Source-only release; no experimental model weights are bundled.

## Automated checks

Full suite: **141 passed in 113.00 seconds** (118 existing cases and 23 new/
parameterized cases). The new motor fixture was trained from fresh weights as part
of the tests; qualification was not mocked for successful attachment tests.

- Teacher-labelled learning changes actual motor weights and reduces full-dataset
  imitation loss to less than 20% of initial loss.
- Held-out motor evaluation succeeds with the reference teacher replaced by a
  function that raises if called. Untrained baseline fails navigation tests.
- Model/optimizer/data/sampler resume produces exact uninterrupted-training state.
- Bad contracts, tensor/digest corruption, teacher-assisted reports, inadequate
  sample sizes, failing scores, reused training seeds and stale qualifications
  are rejected. Report trust is local: these are not adversarial certificates.
- Motor labs refuse individual/legacy directories and concurrent writers.
- Attachment preserves executive/target/optimizer/replay/workspace/intentions/
  memories/world/RNG/identity for both tiny and 32.1M standard profiles. Original
  individual databases remain byte-identical and old motor tensors/hidden state
  are archived. Legacy files remain intact.
- Actual execution uses neural motor predictions, not reference calls. Planner
  diagnostic and plain executive evaluation leave the saved database unchanged.
- New-controller learned-world reranking waits for new-controller updates even
  when lifetime updates are already 899. Learned weights/optimizer are retained.
- Motor history is included in backups. Existing/running/nested/unqualified/wrong-
  body sources and incomplete destinations are rejected without overwriting sources.
- Injected attachment failure leaves the original intact and the partial copy
  marked non-runnable.
- Split versus uninterrupted executive runs with the learned motor produce exact
  final model, learner, workspace, world, intention, motor and RNG states on CPU.
- Motor-lab SIGTERM saves trained weights and optimizer and releases its lock.
- Prior THOR mock-controller, sensor recovery, memory, dashboard, checkpoint and
  learner tests remain passing. No native Unity changes were made.

## Executed command-line experiments

Motor training: seed 7, 256 cases, 3,132 labelled sensor/mode/action examples,
5,060 parameters, 2,000 class-balanced optimizer updates on CPU. Final sampled
cross-entropy 0.0259449; that loss is not a held-out behavioral metric.

Teacher-free qualification: evaluation seed 1,000,010, distinct from training.

| Skill | Successful episodes | Executed primitive evidence |
|---|---|---|
| Orient | 100/100 | 61 left turns, 57 right turns, 36 idles |
| Navigate | 100/100 | 762 successful advances, 79 left turns, 90 right turns |

Zero teacher actions and zero infrastructure faults. Test cases begin with a
visible goal and a clear direct corridor; these results do not certify general
navigation, obstacle avoidance, exploration, unseen bodies or real robots.

Exact tested motor digest:
`facc597e4e57b35aa13c123fdb94a4574409ebfdb9ad5ce24421dda9a51118e6`.
Users train their own model; numeric results/digests can differ across PyTorch and
hardware. Their own qualification must pass before attachment.

End-to-end CLI checks:

- Standard-profile continuation copy attached successfully, preserving an existing
  8-decision/1-update executive. Planner-guided, read-only room diagnostic: 200
  additional decisions, 24/26 new tasks successful, zero faults. This tests the
  motor under the authored planner, not independent learned executive competence.
- Tiny-profile continuation from a 300-decision/74-update executive: 300 further
  training decisions, 33/37 new tasks successful, 600 cumulative memories, 148
  cumulative updates and 74 new-controller updates; zero faults. The executive
  remained substantially planner-assisted. The motor snapshot stayed fixed.

These are short, single-seed development checks, not statistically established
reliability or evidence of consciousness. The user's actual 3,800-decision state
was not accessible here; preservation tests use analogous locally generated states.

## Environment and limits

Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.2, pytest 9.1.1. Existing isolated test
dependencies were reused; no installation or GPU changes were attempted. An earlier
optional package-install check was blocked and was not retried. Bundled-source CLI
execution is tested; native custom Unity, ROCm/RX 6600 and physical robots are not.

This release supplies staged supervised motor learning and fixed-snapshot executive
execution, not joint online motor learning or a continually expanding demonstration
set. Semantic foundation cognition and autonomous skill discovery remain future work.

Packaging generates one fresh per-file SHA-256 manifest and validates archive CRC
and every digest. Runtime state, trained weights, caches and virtual environments
are excluded. RUN_v0.6.2.md supersedes historical run instructions.
