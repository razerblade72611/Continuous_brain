# Upgrade / integration — v0.12.0 or v0.11.3 to v0.12.1

Use v0.12.1 as a fresh extracted source tree. Do not unzip over an older release.

The v0.12 lifelong individual's authoritative state is outside the release tree (`~/ContinuousBrain/individual_v120` under WSL), so a normal v0.12.1 launch continues it without copying files. Its runtime schema/build are intentionally unchanged.

The 14,000-update InfantBrain checkpoint is already bundled under `state/child_v113_dml_smoke/`; do not overwrite it with a partially trained competitive checkpoint.

Existing v0.11.3 and v0.12 source directories can remain as backups. Competitive v2 partial/probe directories are historical diagnostics; start v3 training in a new state directory.
