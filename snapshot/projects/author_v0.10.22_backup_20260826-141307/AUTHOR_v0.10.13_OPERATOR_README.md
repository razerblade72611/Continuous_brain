# AUTHOR v0.10.13 — Semantic Route Qualification

v0.10.13 is a narrow Level-4/5 curriculum hardening release on top of v0.10.12.

## Why this release exists

`--curriculum door --level 4` is a door-infrastructure drill. It does **not** prove preservation of a non-door semantic target across a blocked route. True Level-4 semantic continuation is exercised by `pickup`, `place`, and `toggle`: the trainer discovers the target with doors open, closes one route door so the target is inaccessible from the start component, and then calls `NavigateTo` on the original target. The motor layer may privately approach/open/cross the blocking door and must then resume the original target.

## Changes

- blocked-route Level-4/5 starts now require local maneuverability (at least two cardinal reachable neighbors);
- once the blocking door is selected, the setup yaw is aligned to the first verified path segment toward its interaction pose;
- replay schema is `author.embodied.v01013` and stores controller-private `auxiliary_skills` route evidence;
- successful Level-4/5 semantic target approaches require verified `door_approach`, `door_open`, and `door_cross` auxiliary stages plus final arrival at the original target pose;
- a new staged skill, `route_resume_target`, records that strict continuation result;
- SFT schema is `author.embodied_sft.v01013`;
- v0.10.12 FIFO timeout/transport recovery is unchanged;
- no Unity rebuild is required and persistent A/B state must be preserved.

## Install

Do not use `--reset-memory`.

After installing, run `pytest -q`, then verify `agent_core.__version__ == 0.10.13`.

## Level-4 qualification order

Start with pickup because it is the simplest non-door semantic target:

```bash
EPISODES=60 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum pickup --level 4
```

A valid Level-4 run must show `pickup_approach`, `pickup_grasp`, internal door skill increments, and `route_resume_target` Level-4 attempts matching the scored semantic episodes. Only after pickup is clean should place/toggle be qualified.
