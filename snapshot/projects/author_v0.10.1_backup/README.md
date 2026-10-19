# AUTHOR v0.10.1 — Verified Interaction-Pose Curriculum

AUTHOR is a two-agent embodied AI research/runtime stack for AI2-THOR. Qwen remains the semantic decision layer; controller-private body skills turn an already-selected semantic target into verified physical execution.

## What changed in v0.10.1

- Interaction approach targets AI2-THOR `GetInteractablePoses` rather than an object's transform center.
- Returned position, rotation, camera horizon, and stance stay private inside the motor layer.
- A `NavigateTo` can replan among valid interaction poses and can continue through a reachable closed interior Door without changing Qwen's semantic target.
- Training setup is trainer/admin infrastructure and is never scored as agent competence.
- Invalid/impossible setups are skipped, not counted as failures.
- Skills are staged: `door_approach`, `door_open`, `door_cross`, `pickup_approach`, `pickup_grasp`, `place_approach`, `place_release`, `toggle_approach`, `toggle_interaction`, `point_navigation`, and `peer_approach`.
- Existing v0.10 A/B checkpoints migrate losslessly into `legacy_v010`. Legacy `navigate` is retained as history but is not trusted as v0.10.1 approach competence.
- Verified outcomes from live sessions update A/B's independent staged checkpoints.
- Peer text has transport IDs, delivery persistence, and fail-closed filtering for `NO_MESSAGE` and malformed JSON.
- Only successful, simulator-verified trajectories are exported to `state/datasets/embodied_sft.jsonl`. This is motor-demonstration data; v0.10.1 does not modify Qwen's weights.

## Training

Start deliberately at level 0 instead of launching thousands of mixed episodes:

```bash
EPISODES=100 WORKERS=1 LEVEL=0 ./scripts/run_training_wsl.sh --curriculum door
python scripts/inspect_skills.py
```

Then graduate levels 1, 2, 3, and 4 only after the lower level is reliable. `open_cross` remains accepted as a backward-compatible alias for `door`.

A safe dry run (no Unity launch) is:

```bash
EPISODES=10 LEVEL=0 ./scripts/run_training_wsl.sh --curriculum door --dry-run
```

Before training against a custom Unity build, query its actual pose support:

```bash
python scripts/smoke_test_interaction_poses.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

See `AUTHOR_v0.10.1_OPERATOR_README.md` for the full upgrade, validation, training gates, and live `/a Escape.` test.
