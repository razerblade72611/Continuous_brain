# v0.11.4 CLI protocol

For the complete workflow, use the included notebook and `START_HERE_v0.11.4.md`.
These commands document the training interface. Dataset and state variables must
point to your existing local staged assets; never point them at evaluation holdouts.

```bash
python -m continuous_brain.parented_cli protect-growth \
  --source-state-dir "$SOURCE_V110_EXACT_STATE" \
  --state-dir "$V114_STATE" --device cuda

python -m continuous_brain.parented_cli evaluate-hierarchical-options \
  --state-dir "$SOURCE_V110_EXACT_STATE" \
  --episodes 100 --steps 120 --seed-start 12001 \
  --option-depth 2 --selector-topk 2 --beam-width 8 --max-option-steps 12 \
  --discount 0.95 --min-commit 3 --terminate-threshold 0.50 \
  --device cuda --output "$RESULTS/V114_BASELINE_EXACT_P4B_12001_12100.json"

python -m continuous_brain.parented_cli train-protected-growth \
  --rehearsal-datasets "$BOOTSTRAP" "$OLD_WORLD" \
  --world-datasets "$OLD_WORLD" "$DETOUR" "$RANDOMIZED" "$TWO_BARRIER" \
  --planning-dataset "$RANDOMIZED" --planning-rehearsal-dataset "$DETOUR" \
  --state-dir "$V114_STATE" --durable-state-dir "$DRIVE_V114_STATE" \
  --stage shadow \
  --previous-p4b-result "$RESULTS/V114_BASELINE_EXACT_P4B_12001_12100.json" \
  --batch 8 --learning-rate 0.0001 --anchor-weight 0.10 \
  --interface-anchor-weight 1.0 --checkpoint-every 250 --seed 11410 --device cuda

python -m continuous_brain.parented_cli evaluate-hierarchical-options \
  --state-dir "$V114_STATE" \
  --episodes 100 --steps 120 --seed-start 12001 \
  --option-depth 2 --selector-topk 2 --beam-width 8 --max-option-steps 12 \
  --discount 0.95 --min-commit 3 --terminate-threshold 0.50 \
  --device cuda --output "$RESULTS/V114_SHADOW_P4B_12001_12100.json"
```

The baseline must match the exact source checkpoint and pass the original P4b
criteria. The next stage is `micro`, then `low`, then `moderate`; for each, pass
the preceding stage's matching P4b JSON through `--previous-p4b-result`. The trainer
checks checkpoint identity, evaluation settings, actual metrics, and completion
of prior stage targets. Each successful endpoint is frozen automatically.

`--target-stage-updates` is a cumulative target for a partial run, bounded by the
stage ceiling. It cannot add extra training or rewind a completed stage.

Protected migration refuses a nonempty destination and verifies every source
tensor, zero-initialized growth outputs, and exact interface equivalence. The
Colab wrapper additionally requires the known exact source file hash. Training
refuses mismatched optimizer/protocol state, non-finite loss/gradients, legacy
tensor drift, or explicitly training-prohibited/development/blind seed records.

CLI invocations alone do not create the complete Colab qualification envelopes.
Use the notebook to produce the final checkpoint/runtime-bound readiness report.
