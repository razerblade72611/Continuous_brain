# AUTHOR v0.10.15 — Route-Safe Semantic Continuation

This release is a focused Level-4 reliability patch on top of v0.10.14.

## Why
Observed Level-4 pickup failures showed authoritative Unity collisions from:
- `Discovery_Book is blocking Agent ...`
- `Agent 0 is blocking Agent 1 ...`

These are physical interference failures, not semantic-target loss.

## Changes
- Restores displaced `Discovery_*` props to their worker-start canonical positions before each curriculum setup. Vertical falls are detected, not only X/Z drift.
- Parks the passive Level-4 peer away from the verified active-agent door approach corridor. Peer interference remains a Level-5 concern.
- Adds bounded dynamic-obstacle replanning for pose navigation. If Unity reports `is blocking Agent`, AUTHOR blacklists the failed next grid cell and replans to the same interaction pose up to three times.
- Replay schema is `author.embodied.v01015` and SFT schema is `author.embodied_sft.v01015`.
- Replay rows include `run_id` and `worker`.
- Training summary reports `discovery_prop_restores`.

## Upgrade
Preserve `state/`. Do not use `--reset-memory`. No Unity rebuild is required.

## Validation
Run:

```bash
pytest -q
```

Expected: `205 passed`.

Then run the Level-4 pickup pilot:

```bash
EPISODES=60 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum pickup --level 4
```
