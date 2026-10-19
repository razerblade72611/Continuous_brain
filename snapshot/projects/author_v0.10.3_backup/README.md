# AUTHOR v0.10.3 — Spatial Actionability + Reliable Recovery

AUTHOR is a multi-agent AI2-THOR embodied-agent research system with local LLM/VLM planning, persistent A/B experience, simulator-authoritative verification, staged motor curricula, and controller-private motor state.

## v0.10.3 focus

v0.10.3 generalizes interaction failures from an exact `(x,z,yaw,horizon)` pose to the underlying X/Z physical region, then makes recovery choose a materially different, clearance-safe pose before retrying the same semantic action.

Key changes:

- exact-pose + X/Z position actionability memory;
- automatic migration of v0.10.2 actionability history;
- moving-door/open-close failed-region exclusion;
- safe-clearance-first recovery ranking;
- materially different recovery positioning;
- explicit recovery attempted/succeeded/exhausted metrics;
- historical v0.10.2 recoveries preserved as unclassified;
- intermediate route-door opening uses the same recovery system;
- per-agent A/B actionability remains independent;
- multi-worker delta merging covers exact and spatial statistics;
- current-run coverage remains bounded to 100%;
- replay schema updated to `author.embodied.v0103`.

See `AUTHOR_v0.10.3_OPERATOR_README.md` for exact upgrade, calibration, curriculum, and live-session instructions.
