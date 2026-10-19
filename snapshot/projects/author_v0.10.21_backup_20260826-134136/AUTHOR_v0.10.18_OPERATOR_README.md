# AUTHOR v0.10.18 — Topology-Authoritative Cross + Placement Preconditions

v0.10.18 fixes the three failure classes isolated by the Level-4 place replay:

1. Door crossing is verified by transition from the closed-door source component into the newly connected component, not by an oblique door-vector projection.
2. CrossThreshold targets the actual NavMesh component boundary revealed by opening the door, which centers the physical crossing on the real portal seam.
3. Openable placement receptacles are explicitly opened before PutObject and scored as `place_receptacle_open`.
4. Source/receptacle pairs are trainer-preflighted for physical PutObject feasibility and cached; impossible pairs are rejected before a scored lesson.
5. Replay/SFT schemas are `author.embodied.v01018` / `author.embodied_sft.v01018`.

No Unity rebuild is required. Preserve `state/`; do not use `--reset-memory`.

## Validate

```bash
python -m pip install -e ".[dev]"
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

## Level-4 place pilot

```bash
EPISODES=60 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum place --level 4
```

Expected startup trainer: `topology_cross_place_precondition_curriculum`.
