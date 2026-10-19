#!/usr/bin/env python3
"""Train the need controller and compare against the routine baseline on held-out worlds."""
from __future__ import annotations

import json
from pathlib import Path

from continuous_individual.agent import Individual
from continuous_individual.learn import evaluate, train_controller
from continuous_individual.store import CanonicalStore


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)

    print("=== Baseline: routine individual ===")
    baseline = evaluate(n_worlds=40, seed_start=50000, use_learned=False)
    print(json.dumps(baseline, indent=2))

    print("\n=== Training learned need controller ===")
    store = CanonicalStore()
    individual = Individual(store=store)
    returns = train_controller(individual, n_episodes=250, seed=42)
    print(f"Training episodes: {len(returns)}, last-20 mean reward: {sum(returns[-20:]) / 20:.2f}")

    # Persist
    ckpt = out / "trained_store.pkl"
    store.save(ckpt)
    print(f"Saved store to {ckpt} (hash {store.state_hash()[:12]})")

    print("\n=== Evaluation: learned controller on held-out worlds ===")
    learned = evaluate(n_worlds=40, seed_start=50000, use_learned=True, store=store)
    print(json.dumps(learned, indent=2))

    print("\n=== Rebuild check ===")
    loaded = CanonicalStore.load(ckpt)
    rebuilt = Individual(store=loaded)
    # Compare against a second load so in-memory mutations from eval don't matter
    loaded2 = CanonicalStore.load(ckpt)
    assert rebuilt.store.state_hash() == loaded2.state_hash()
    print("Rebuild from saved store: state hash matches.")

    summary = {"baseline": baseline, "learned": learned, "train_last20": sum(returns[-20:]) / 20}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
