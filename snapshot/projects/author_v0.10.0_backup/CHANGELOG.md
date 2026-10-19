# v0.10.0 — Accelerated Embodied Skill Curriculum

- Rewrites the accelerated trainer around a weighted curriculum: navigation, door-open-and-cross, pickup, place, toggle, and peer meeting.
- Adds persistent independent A/B body-skill checkpoints under `state/checkpoints/a|b/embodied_skills.json`.
- Adds `CrossThreshold`, a body-only pseudo-action that crosses the same open doorway already selected by the LLM; it never chooses a semantic target.
- Normal task mode can use trained Navigate/Face/Cross macros after the model has grounded the requested target.
- Keeps direct verified interactions available even before training; learned skills improve the precondition and crossing mechanics.
- Adds `state/replay/embodied_replay.jsonl` with simulator-authoritative action/result transitions.
- Adds `state/datasets/embodied_sft.jsonl` containing successful target-type/action-sequence examples suitable for later SFT/LoRA preparation.
- Training writes compact per-agent competence summaries into that agent's own persistent episodic memory.
- Keeps shared map/local-navigation learning and separate private A/B memory.
- Default training makes zero LLM calls.
- Adds curriculum selection via `--curriculum` and `--no-sft-export`.
- Adds skill inspection utility.

# v0.9.3 — Physical Doorway Repair + Anti-Echo Peer Text

- Rebuilds Discovery doors around a true hinge/swing child so the leaf collider actually swings out of the doorway instead of rotating in place around its own center.
- Clears runtime static flags from the moving door hierarchy.
- Adds build-time doorway-prism collision clearance for stock decorative wall / DoorFrame colliders, then restores invisible boundary collision on both sides of each opening.
- Keeps the closed door leaf as the physical blocker; opening it exposes a real corridor with no teleport.
- Keeps v0.9.2 exact `Floor` walkable threshold/floor naming.
- Enhances the Discovery smoke test to report post-open `isOpen` state and perform a physics command boundary before reachability.
- Adds anti-echo peer text and preserves explicit user-requested text delivery.
