# Continuous Brain v0.9.1 — Parent Succession Hotfix

v0.9.1 is a compatibility and scientific-rigor hotfix for the v0.9 parented-development line.
It preserves v0.9.0 child checkpoints and nursery datasets.

## Changes

- Safe optimizer reinitialization when a new parent projection head changes the trainable parameter list.
  Child neural weights are preserved; only optimizer moment estimates are reset when required.
- Multi-parent visual replay: archived visual-parent targets now remain active developmental replay evidence
  through their own detachable heads instead of being provenance-only archives.
- Evaluation reports active visual-parent fit, historical-parent retention by parent ID, and the minimum
  historical retention score. Vision qualification uses the weaker of current-parent fit and retained old-parent fit.
- Evidence-aware qualification/weaning: motor and value scores do not advance weaning when the nursery lacks
  sufficient target diversity. World-model evidence is reported separately.
- Parent upgrades reset parent-specific competency streaks while preserving the child's current authority level.

This release does not claim that imported v0.7 semantic lessons prove motor, world-model, temporal, speech,
reasoning, or value maturity. Those capabilities require dedicated developmental datasets and gates.
