# AI2-THOR Agent v0.9.3 — Physical Doorway Repair + Anti-Echo Peer Text

v0.9.3 is a focused hotfix for the v0.9.2 Discovery world and visible peer channel.

## What changed

The two stock FloorPlan1 architectural doors were successfully converted to openable simulator objects, but the moving leaf was rotating about its own center. v0.9.3 inserts a dedicated `DoorSwing` transform at the hinge, clears static flags on the moving hierarchy, and clears any stock decorative collision that physically occupies the doorway prism. Invisible boundary segments are rebuilt around the opening so the only traversable gap is the actual door opening.

Peer texting remains visible in terminal/dashboard/logs, but periodic text now suppresses direct or near-direct echoes of recent messages. The dedicated composer may return `NO_MESSAGE` when there is nothing new to say. Explicit text tasks still force a real delivery attempt.

## Quick upgrade

Preserve `state/`, install editable, reapply/sync the Unity patch, rebuild the Discovery player in Windows Unity, copy it back to WSL, then run `scripts/smoke_test_discovery_world.py`.

See `AUTHOR_v0.9.3_OPERATOR_README.md` for exact commands.
