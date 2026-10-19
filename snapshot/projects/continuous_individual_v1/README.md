# Continuous Individual v1

Survival-focused rewrite of the Continuous Brain learning path.

## Goal alignment
David’s goal is consciousness, with authenticity first. Persistent memory, self-modeling, and learning across lives are engineering targets, not substitutes for that goal. This package keeps the canonical-store authority invariant and adds a measurable experiment on the bottleneck that has blocked progress: need-management deaths under partial observability.

## What this is
- `CanonicalStore`: single-writer blob store; all controller weights live in it.
- `SurvivalWorld`: grid world with thirst, hunger, scattered water/food, partial observability (resources known only after visit).
- `Individual`: routine policy + small learned MLP controller for seek-water / seek-food / explore. Weights are written to the store after every update.
- Training uses a simple policy-gradient update on episode return.
- Evaluation compares routine vs learned on identical held-out seeds and reports mean survival, mean score, and need-death rate.
- Rebuild test: load the store into a fresh `Individual` and confirm the state hash matches.

## Setup and run

```bash
# From the package root
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pytest

# Run the experiment
python scripts/train_eval.py

# Tests
python -m pytest tests/ -q
```

Results land in `results/summary.json`.

## What would reverse this approach
If the learned controller does not reduce need-death rate or raise survival relative to the routine on the held-out set, the bottleneck is deeper than a thin need-management head and a larger change (learned low-level policy, better credit assignment, richer memory) is justified.
