# Lanes: one protected, one fast, and known-good points to fall back to

The project runs in two lanes at once, so that pushing forward never puts the evidence at risk.

## 1. The protected lane (I1, and later every scientific claim)

- **Branch:** `v0.38.1-dev7-i1`, until I1's scientific run is done. Each later scientific result gets its own
  protected branch.
- **What happens here:** only preregistered work. Changes are patches that ChatGPT reviews before anything
  single-use is spent, and every run is evidence-gated (manifest, validity checks, sealed preflight, David's
  approval).
- **Checkout:** a dedicated one on David's WSL (`~/projects/continuous_brain_v0.38.1.dev7_i1`). Nothing
  exploratory is ever run, copied or pulled there.
- **Seed families:** only the ones its preregistration names (`i1-v1-preflight-eng-b`, `i1-v1-dev`), plus the
  engineering families for dev checks.

## 2. The exploration lane (I2 and beyond: move fast)

- **Branches:** `explore/<topic>`, starting with `explore/i2-goal-aware-controller`. Each starts from the latest
  `verified/...` tag of the protected lane (or from the protected branch's head, when that is newer and has
  passed the full suite).
- **What happens here:** new mechanisms, a richer world, larger learned components. Results are engineering
  results until they are preregistered and moved to a protected branch.
- **Checkout:** a separate one (`~/projects/continuous_brain_explore`), never the protected one. Its ledger is
  its own copy, so nothing done here can spend, register or alter a protected family.
- **Seed families:** its own engineering families, never an I1 family.
- **Guard:** on an `explore/*` branch, `tests/test_lane_guard.py` fails if any file the protected lane froze
  (listed in `LANES.json`) differs from that lane's base commit. New work goes into new modules, not into the
  frozen I1 code. To build on a newer protected state, merge it from the protected branch; the guard then
  checks against the new base.

## 3. Known-good points and how to fall back

- **Tags:** every state that an independent audit (ChatGPT, or a fresh verifier agent) reproduced gets an
  annotated tag `verified/<what>`. The tag message records what was verified (hashes, tests, evidence) and any
  open finding. Tags never move.
- **Evidence:** each tag has its evidence bundle (`evidence/...` in the tree, and the delivered `.tgz` with its
  SHA-256).
- **Falling back** in a checkout: `git switch -c recover/<name> verified/<tag>`, then run the full suite. The
  manifest and the design digest in the tag message say whether the state is the one verified. A failed
  exploration branch is simply abandoned or reset to its last good tag; the protected lane is untouched.
- **Individuals:** a running individual's state lives in its store snapshots (`PERSIST_SAVE/persist/...` for
  dev runs). Restoring a snapshot taken under a verified tag, with that tag's code, restores the individual
  bit-exactly (validity C10).

## Current tags

| tag | commit | what was independently verified | open finding at that point |
|---|---|---|---|
| `verified/i1-dev7-64h` | `0a0ae40` | ChatGPT: zip/bundle/evidence hashes, 531/531 sums, 283/283 tests, 21/21 tamper, 56/56 gate | approval authority, ledger freeze, derived ids at spend (fixed in `e323623`) |
| `verified/i1-dev7-56i` | `c627c93` | ChatGPT: hashes, 475/475 sums, 305/305 tests, 21/21 tamper, 59/59 gate | concurrent openings (fixed in `82646be`) |
| `verified/i1-dev7-64j` | `db4fa48` | ChatGPT: hashes, 531/531 sums, fsck clean, 311/311 tests, 21/21 tamper, race and control reproduced | a failure after the spend left no record (fixed in `6438563`) |
