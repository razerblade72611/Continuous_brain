# AUTHOR v0.10.0 — Accelerated Embodied Skill Curriculum

AUTHOR v0.10.0 turns accelerated training into a real body-skill curriculum instead of a navigation-only loop.

The LLM still chooses semantic goals and targets. Training does **not** create hidden goals. The learned layer only becomes competent at executing a target that the LLM has already selected.

## What learns now

Each agent gets an independent persistent skill checkpoint:

- `navigate`
- `face_target`
- `open`
- `cross_door`
- `pickup`
- `place`
- `toggle`
- `peer_meet`

The shared world map and tabular local-navigation checkpoint remain in use.

Normal sessions load the per-agent skill checkpoint. Once enough successful teacher episodes have been accumulated, an explicit user task can use the trained body sequence after the model has selected its own target. Example:

`LLM selects door -> learned NavigateTo -> learned FaceTarget -> OpenObject -> learned CrossThreshold -> simulator verifies region change`

## Accelerated training

No LLM calls are required by the default trainer. It uses simulator-authoritative teacher trajectories at maximum simulator speed and records both successes and failures.

```bash
EPISODES=1000 WORKERS=1 ./scripts/run_training_wsl.sh
```

Targeted training is supported:

```bash
EPISODES=200 ./scripts/run_training_wsl.sh --curriculum open_cross
EPISODES=300 ./scripts/run_training_wsl.sh --curriculum pickup,place,toggle
```

Successful trajectories are also exported to `state/datasets/embodied_sft.jsonl` for a later optional LoRA/SFT phase. v0.10.0 does **not** modify Qwen's weights.

See `AUTHOR_v0.10.0_OPERATOR_README.md` for the upgrade and operating sequence.
