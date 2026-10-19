# v0.6.0 — Persistent Individual Core

This release changes the cognitive architecture while preserving the v0.5.3
sensorimotor branch. It implements a persistent recurrent attention executive,
direct memory retrieval, an empirical self-model, intention history, compositional
options, sequence learning and a short-horizon predictive model.

The standard executive contains **32,114,036 trainable parameters** with the default
pixel-statistics encoder. Compact: 2,704,756. Tiny: 126,260. The standard profile
is the intended main research branch; tiny is only a fast smoke-test configuration.

New entry point: `continuous-individual` (`python -m continuous_brain.individual_cli`).
New state: `individual.sqlite3`, checkpoint format 1, package version 0.6.0.
Old entry point: `continuous-brain`, format-9 motor checkpoints, retained compatibility.

The development room supports exact checkpoint continuation. Existing THOR motor
weights and full historical state can be imported into a new individual without
changing the original files. Motor weights remain frozen. Old latent memories are
preserved as an explicitly unclassified archive; no fabricated semantic migration
is claimed.

The architecture is runnable and trainable, but does not yet supply conversational
language, social reasoning, self-authored goals, autonomous skill discovery, dynamic
neural expansion, or evidence of subjective consciousness. It is not a frontier
LLM/VLM replacement. The authored teacher/controller contribution is visible in
every training run and separated from unassisted executive evaluation.

See RUN_v0.6.0.md for the first commands, ARCHITECTURE_v0.6.0.md for actual learning
mechanics, ARCHITECTURE_CONTRACT.md for privilege boundaries, and RELEASE_VALIDATION.txt
for measured tests and limitations. No trained cognitive checkpoint is included.
