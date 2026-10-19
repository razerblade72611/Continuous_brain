# AUTHOR v0.10.2 — Learned Interaction Actionability + Recovery

AUTHOR is a two-agent embodied AI research/runtime stack for AI2-THOR. Qwen remains the semantic decision layer; controller-private body skills turn an already-selected semantic target into verified physical execution.

## What changed in v0.10.2

v0.10.1 correctly replaced object-center navigation with `GetInteractablePoses`, but live probing showed an important simulator distinction: those poses are visibility/range candidates, not a guarantee that every physical interaction will succeed. A custom East door, for example, returned hundreds of poses while a repeatable subset inside its opening envelope failed `OpenObject`.

v0.10.2 adds the missing actionability layer:

- Per-agent persistent pose/action outcomes under `state/checkpoints/a|b/interaction_actionability.json`.
- Bounded alternate-pose recovery for `OpenObject`, `CloseObject`, `PickupObject`, `PutObject`, and toggle actions while preserving the original semantic target.
- Failed exact poses become strongly disfavored; successful poses become preferred.
- Open/close candidate ranking has a soft clearance prior. It is never a hard exclusion, so constrained scenes can still use closer poses.
- Every interaction attempt records `lastActionSuccess`, error text, recovery status, pose key, and distance-to-object in controller-private replay telemetry.
- Runtime skill records mark successful interactions that required internal recovery.
- Training workers are seeded from persistent A/B actionability experience, keep private copies, and merge only their new deltas so multi-worker runs cannot duplicate old statistics.
- `inspect_actionability.py` reports aggregate and worst exact-pose outcomes.
- Training coverage reporting now separates current-run coverage from the persistent map and cannot report bogus >100% run coverage.

All v0.10.1 behavior remains: staged curricula, authoritative verification, door-aware route continuation, lossless `legacy_v010` checkpoint preservation, online A/B skill learning, and fail-closed peer messaging.

## Upgrade without losing A/B state

```bash
cd ~/projects
mv author author_v0.10.1_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.2_actionability_recovery.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.1_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Then inspect preserved skills and the new actionability store:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py
```

Before resuming harder curricula, rerun Level 0 doors so the body can learn and exercise recovery on previously bad East-door poses:

```bash
EPISODES=100 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 0

python scripts/inspect_skills.py
python scripts/inspect_actionability.py
```

See `AUTHOR_v0.10.2_OPERATOR_README.md` for the complete gates and live-session instructions.
