# AUTHOR v0.10.14 Operator Guide

## Purpose

v0.10.14 fixes Level-4/5 setup starvation observed when 18 scored pickup episodes consumed 538 raw attempts. v0.10.13 selected a semantic target first and then randomly searched for a start/door combination. v0.10.14 searches the complete lesson tuple inside trainer setup: semantic target + blocking Door + opposite-side maneuverable start.

## Preserved architecture

- The semantic target remains a non-Door object.
- Door IDs/poses/geometry remain controller-private and are not exposed to Qwen.
- `door_approach`, `door_open`, `door_cross`, and `route_resume_target` remain authoritative Level-4/5 evidence.
- v0.10.12 timeout/transport recovery remains enabled.
- Existing A/B checkpoints and replay history must be preserved. Do not use `--reset-memory`.

## Upgrade

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.13_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.14_component_verified_routes.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

No Unity rebuild is required.

## Level-4 pickup validation

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum pickup \
  --level 4
```

Expected startup trainer: `component_verified_semantic_route_curriculum`. Setup skips should be dramatically lower than the v0.10.13 run; `route_resume_target` must be present for successful pickup approaches.
