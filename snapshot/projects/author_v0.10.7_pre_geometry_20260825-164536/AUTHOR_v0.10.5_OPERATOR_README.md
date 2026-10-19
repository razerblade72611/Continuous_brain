# AUTHOR v0.10.5 Operator Guide

## Purpose

v0.10.5 repairs a release regression discovered by the real Level-1 door curriculum run.

v0.10.4 successfully fixed pose-authoritative short-range approach. In the new 150-episode Level-1 run, A's new approach batch was 75/75 and B's was 74/75; every reached-door `OpenObject` also succeeded. But all new `CrossThreshold` attempts failed because the v0.10.4 refactor accidentally omitted `_cross_threshold_for_adapter` from `thor_adapter.py`. The same refactor also omitted `_face_target_for_adapter`, although Level 1 did not expose that path.

v0.10.5 restores both helpers and adds regression tests that dispatch the pseudo-actions through the same live adapter route used by training/runtime.

## State policy

Do **not** reset A or B. Copy the existing v0.10.4 `state/` into v0.10.5 exactly as with prior upgrades.

The raw v0.10.4 replay contains the false `door_cross` failures. Keep it: it is useful audit evidence. Then run the included repair utility once. It finds only replay rows matching all of these conditions:

- schema `author.embodied.v0104`
- `stage_skill == door_cross`
- `action == CrossThreshold`
- `success == false`
- exception contains `_cross_threshold_for_adapter` and `not defined`

It backs up each affected `embodied_skills.json`, subtracts only those synthetic failures from aggregate/object/Level-1 counters, writes an idempotent migration marker, and leaves the replay untouched.

## Upgrade from v0.10.4

```bash
cd ~/projects

mv author author_v0.10.4_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.5_cross_threshold_restoration.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.4_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate

python -m pip install -e ".[dev]"
pytest -q
```

Expected test count for this release is recorded in `RELEASE_VALIDATION.txt`.

Verify version:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.10.5
```

## Repair the v0.10.4 synthetic cross failures

Run exactly once after copying state:

```bash
python scripts/repair_v0104_cross_bug.py
```

The command is idempotent. Re-running it reports `already_applied: true` and does not subtract anything twice.

For the observed v0.10.4 run, the expected repair is approximately:

```text
A: remove 75 synthetic Level-1 door_cross failures
B: remove 74 synthetic Level-1 door_cross failures
```

Those numbers should be derived from replay by the script, not hard-coded.

Then inspect:

```bash
python scripts/inspect_skills.py
```

The restored historical Level-1 cross record should again reflect the real successful v0.10.3 crossing experience rather than the v0.10.4 missing-function regression.

## Repeat Level 1

Take a checkpoint first:

```bash
cp -a state state_v0105_pre_level1_repeat
```

Then:

```bash
EPISODES=150 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 1
```

Afterward:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py --worst 30
python -m json.tool state/metrics/training_summary.json
```

Do not start Level 2 until this Level-1 repeat is reviewed.

## What should happen

The v0.10.4 approach correction is preserved:

```text
trainer pins exact interaction pose P
        ↓
body reaches P
        ↓
controller pose verification is authoritative
        ↓
OpenObject with actionability recovery
        ↓
CrossThreshold re-queries reachable topology
        ↓
select reachable point beyond same open door
        ↓
physically walk across
        ↓
authoritative crossing verifier
```

The new regression suite includes live adapter dispatch for both `CrossThreshold` and `FaceTarget`, specifically preventing the missing-helper class of release failure from recurring.

## Release identity

```text
Project:       AUTHOR
Release:       v0.10.5
Phase:         Phase 9
Focus:         CrossThreshold restoration + synthetic-state repair
Previous:      v0.10.4
State:         Preserved and repaired, not reset
Agents:        A + B retain separate persistent experience
```
