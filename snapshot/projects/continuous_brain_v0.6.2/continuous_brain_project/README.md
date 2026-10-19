# Continuous Brain v0.6.2 — Persistent Individual Core

An experimental, trainable lifetime executive above the existing embodied motor
controller. This is an early cognitive-core research release, not a claim of
consciousness, personhood, frontier reasoning, or demonstrated open-ended learning.

**Start with [RUN_v0.6.2.md](RUN_v0.6.2.md).** This release trains a fresh, small
development motor separately, qualifies its actual movement without teacher
actions, then attaches a fixed snapshot to a continuation copy of the existing
executive. No executive reset is required. Old motors and their history remain
archived. No trained weights or Unity builds are bundled.

The new motor has 5,060 parameters, uses the existing local sensors plus an
orient/navigate mode, and does not call the reference motor at execution time.
It learns elementary steering by supervised imitation, not general navigation.
Search/inspection and the early executive planner remain explicitly authored.
See [RELEASE_NOTES_v0.6.2.md](RELEASE_NOTES_v0.6.2.md) and the current validation
report [RELEASE_VALIDATION_v062.md](RELEASE_VALIDATION_v062.md).

## Implemented

- Recurrent attention workspace: four persistent slots; cross-attention to current
  perception, self-state, active intention and four retrieved memory records.
- 24-value explicit self-state, including bounded energy, predictive uncertainty,
  lifetime performance and empirical per-skill confidence/cost. These are numeric
  estimates, not asserted feelings or a narrative identity.
- Persistent intention records, causal outcome history, named parameter slots,
  generic precondition/effect skill composition, six initial authored options.
- Sequence learning with burn-in, option-value TD, a small planner imitation loss,
  up to three-step predictive rollouts, reward/outcome prediction, learned variance,
  and target-network stabilization. Consequence-based option reranking activates
  after 200 optimizer updates; it remains a small, uncertainty-penalized term.
- Direct episodic retrieval into the neural workspace on each decision, using
  stable sensory keys rather than obsolete recurrent embeddings.
- Bounded recent replay plus protected formative sequences; replay-only
  consolidation without extra environment actions.
- Transactional persistence of network/optimizer/target, recurrent workspace,
  memories, intentions, self-model, RNG, replay, motor state and world state.
- A deterministic egocentric 2-D development room, an optional existing THOR
  adapter, source-labelled dashboard and evaluation with no persistent writes.
- Strict preservation and frozen reuse of imported v0.4.3/v0.5.x learned motor weights.
  Old replay, episodic databases and snapshots are copied intact to a legacy archive.
- Separate CPU motor lab with persistent optimizer/demonstrations/RNG, teacher-free
  local qualification, exact-weight qualification binding and non-overwriting
  motor attachment. Development-specific motors are refused for other bodies.
- `run_window` and per-skill primitive counts distinguish current-run behavior
  from lifetime totals. `--evaluate --planner-diagnostic` isolates motor behavior
  without saving any executive changes; it is not an autonomous-executive test.

## Profiles

With the default 48-value pixel-statistics visual adapter:

| Profile | Workspace width / layers | Use |
|---|---|---|
| tiny | 64 / 1 | Fast correctness and development checks |
| compact | 256 / 2 | Lower-cost experiments |
| standard | 768 / 3 | Approximately 32.1M trainable parameters; intended main branch |

The target network is a training copy, not additional independent cognitive capacity.
Frozen pretrained vision, when explicitly selected, is separate. GPU memory and
throughput on your RX 6600 have not been benchmarked here. Keep your working ROCm
PyTorch installation. CPU fallback works, but standard training can be slower.
Profiles cannot be switched in-place; dimensional expansion requires a future
explicit migration. A tiny smoke individual is not a smaller checkpoint of a
standard individual.

## What this release does not yet provide

There is no natural-language conversation interface, semantic VLM executive,
learned social model, free-form goal generation, autonomous new-skill invention,
network growth, or generative replay. Initial skills and the bounded goal scheduler
are authored. Target parameters currently bind to the environment's sensed target
or current waypoint, not arbitrary language-described objects. The world model
predicts sensor features over short horizons, not full physical scenes.

Without a motor import the development room uses an explicitly labelled analytical
motor teacher. Early executive choices also use a decaying planner teacher; the
planner supplies no private coordinates. Assisted task success is not autonomous
mastery. `--evaluate` removes executive teacher forcing and exploration, but does
not convert the analytical motor teacher into a learned controller. The v0.6.2
workflow actually trains and qualifies a neural motor before installing it.
Motor updates occur in the separate lab, not automatically inside the executive
run. Its initial demonstration set is fixed; repeated lab updates refine that
dataset, not unrestricted lifetime motor learning or autonomous skill discovery.

All episodic records remain on disk. Retrieval is bounded to recent and formative
candidate windows; intermediate history is retained but needs a future semantic
index for efficient lifetime-wide search. Legacy latent memories are archived,
not silently inserted into a differently shaped semantic space. No memory
summarization or deletion is performed.

The default pixel-statistics encoder is not pretrained visual intelligence. DINOv3
is an optional frozen feature provider inherited from v0.5.x; rich semantic and
language cognition remains the v0.7 milestone.

## State and safety

Use a new `state/individual_*` directory on the WSL Linux filesystem, not `/mnt/c`.
`individual.sqlite3` is the authority. SQLite commits its checkpoint and new
memories together. Abrupt power loss discards the uncommitted interval (default
at most 100 executive decisions); Ctrl+C commits it. A process lock rejects
simultaneous writers to one directory. Do not run restored copies of one identity
concurrently. Use `scripts/snapshot_individual.py --fork` for an explicit new branch.

`individual_status.json` is a human-readable derived report, not the checkpoint.
Copy the whole stopped state directory for backups; do not copy a live SQLite
file without its transaction state. Use the included snapshot tool instead.

Sensor faults are diagnostics, not task failures or replay examples. An entire
interrupted option is quarantined. Unrecoverable faults save and stop. Invalid
core state raises rather than sending actions. Simulator action masks remain
outside learned cognition. No physical robot actuation is implemented.

The development room restores exact state and CPU continuation is tested. THOR
restores the lifetime mind and map/statistics but must reacquire a fresh physical
scene; exact Unity physics continuation is not claimed.

## Architecture and compatibility

Read [ARCHITECTURE_v0.6.0.md](ARCHITECTURE_v0.6.0.md) and
[ARCHITECTURE_CONTRACT.md](ARCHITECTURE_CONTRACT.md).

`continuous-individual` / `python -m continuous_brain.individual_cli` run v0.6.
`continuous-brain` / the original scripts remain the unchanged v0.5.3-compatible
motor/curriculum path. Their format-9 checkpoints and version labels are preserved
deliberately; do not confuse that legacy entry point with the new core.

```bash
python -m pytest -q
python -m continuous_brain.individual_cli --profile tiny --state-dir state/smoke_v062 --decisions 30 --fast
```
