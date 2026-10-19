# v0.9.3 — Physical Doorway Repair + Anti-Echo Peer Text

- Rebuilds Discovery doors around a true hinge/swing child so the leaf collider actually swings out of the doorway instead of rotating in place around its own center.
- Clears runtime static flags from the moving door hierarchy.
- Adds build-time doorway-prism collision clearance for stock decorative wall / DoorFrame colliders, then restores invisible boundary collision on both sides of each opening.
- Keeps the closed door leaf as the physical blocker; opening it exposes a real corridor with no teleport.
- Keeps v0.9.2 exact `Floor` walkable threshold/floor naming.
- Enhances the Discovery smoke test to report post-open `isOpen` state and perform a physics command boundary before reachability.
- Adds anti-echo peer text: periodic A↔B messages can return `NO_MESSAGE`, exact/near duplicates of recent dialogue are suppressed, and repeated status-relay spam is suppressed.
- Explicit user-requested text tasks remain reliable and auditable.
- Preserves verified task completion and the v0.9.1 interaction motor handoff.
