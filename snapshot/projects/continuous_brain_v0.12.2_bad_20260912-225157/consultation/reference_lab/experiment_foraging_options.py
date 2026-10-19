"""
Directed foraging: random walk vs. reactive goal-choice vs. hand-specified
options that persist a goal until completion or termination.

The three-way design is deliberate: conditions 2 and 3 have IDENTICAL
information (both can see where food and water are, in the same grid
channels every condition already receives) and an identical movement
rule (move_toward, below). The only difference between them is whether
the choice of what to move toward is recomputed every single step
(reactive) or held until a reason to change it arises (options). If
options do better, that difference is attributable to persistence
specifically, not to "knowing where resources are," which both
non-random conditions already have equally.

Both goal-directed controllers are HAND-SPECIFIED: the movement rule is a
fixed greedy Manhattan-distance rule, not learned, and the options
controller's termination logic (reach the target, a step-count timeout,
or the other resource crossing a fixed critical threshold) is a fixed
if/else rule, not learned either. What a LEARNED version would add: (a)
a learned movement policy in place of move_toward, so approach behavior
adapts to obstacles or collision risk rather than following a fixed
geometric rule; (b) a learned initiation/termination function in place of
the fixed threshold and timeout, so when to start or abandon an option is
fit to what actually predicts good outcomes rather than guessed; (c) a
learned option-selection policy in place of "whichever body variable is
lower," so goal choice could account for e.g. distance to each resource,
not just urgency. None of that is implemented here -- this milestone asks
whether persistence helps at all, hand-specified, before asking whether
persistence can be learned.

Scope: the random-walk diagnosis motivating this comparison describes the
heuristic used in every experiment in THIS project's mock simulator. It
is not a claim about the real project, which remains uninspected.

Same 200 predefined, paired episode seeds as every prior round.
Run: python experiment_foraging_options.py
"""
from __future__ import annotations
import numpy as np
from mock_env import MockGridWorld

N_ACTIONS = 5
TEST_SEEDS = list(range(3000, 3200))     # retained seeds -- exploratory only, per this round's instruction
CONFIRMATION_SEEDS = list(range(5000, 5200))  # fresh -- never used in any prior round
CRITICAL_THRESHOLD = 0.15
MAX_OPTION_DURATION = 60


def heuristic_action(rng):
    """The random-walk baseline used throughout this project -- unchanged."""
    if rng.random() < 0.1:
        return 0
    return int(rng.integers(1, N_ACTIONS))


def extract_positions(grid):
    self_pos = np.argwhere(grid[1] == 1)[0]
    food_pos = np.argwhere(grid[2] == 1)[0]
    water_pos = np.argwhere(grid[3] == 1)[0]
    return self_pos, food_pos, water_pos


def move_toward(pos, target):
    """Hand-specified greedy rule: reduce row distance first, then column.
    Not learned, not obstacle-aware beyond the grid boundary the
    environment itself enforces."""
    dr, dc = int(target[0]) - int(pos[0]), int(target[1]) - int(pos[1])
    if dr < 0:
        return 1
    if dr > 0:
        return 2
    if dc < 0:
        return 3
    if dc > 0:
        return 4
    return 0


def reactive_controller(rng, grid, body, state):
    """Condition 2: chooses its goal fresh every step -- no memory of what
    it was doing last step beyond what's needed to report a switch."""
    self_pos, food_pos, water_pos = extract_positions(grid)
    energy, hydration, _integrity = body
    goal = "food" if energy < hydration else "water"
    target = food_pos if goal == "food" else water_pos
    action = move_toward(self_pos, target)
    switched = state.get("goal") != goal
    return action, {"goal": goal}, switched


def options_controller(rng, grid, body, state):
    """Condition 3: retains a goal until it's reached, times out, or the
    OTHER resource becomes critically low -- otherwise ignores step-to-step
    fluctuation in which need looks marginally larger right now.

    BUG FIXED HERE: completion was checked by re-extracting food_pos/
    water_pos from the CURRENT observation and comparing to the agent's
    position. But the environment relocates a resource to a new cell in
    the same step it's collected, before the next observation is built --
    so by the time the agent's position matches where the resource WAS,
    the observation already shows where it respawned TO, and the
    equality check never passes. Fix: remember the specific cell being
    approached (target_pos) at the moment a goal is chosen, and check
    arrival against that stored cell, not against whatever the live
    observation currently shows. Movement rule and thresholds unchanged.
    """
    self_pos, food_pos, water_pos = extract_positions(grid)
    energy, hydration, _integrity = body
    goal = state.get("goal")
    duration = state.get("duration", 0)
    target_pos = state.get("target_pos")

    terminate = goal is None
    if goal is not None and np.array_equal(self_pos, target_pos):
        terminate = True  # actually arrived at the cell being approached
    if goal == "food" and hydration < CRITICAL_THRESHOLD:
        terminate = True
    if goal == "water" and energy < CRITICAL_THRESHOLD:
        terminate = True
    if duration >= MAX_OPTION_DURATION:
        terminate = True

    switched = False
    if terminate:
        new_goal = "food" if energy < hydration else "water"
        switched = new_goal != goal
        goal, duration = new_goal, 0
        target_pos = food_pos if goal == "food" else water_pos  # fix the target for this commitment

    action = move_toward(self_pos, target_pos)
    return action, {"goal": goal, "duration": duration + 1, "target_pos": target_pos}, switched


def run_episode(env_seed, mode, action_rng_seed, max_steps=200):
    env = MockGridWorld(seed=env_seed)
    rng = np.random.default_rng(action_rng_seed)
    grid, body = env.reset()
    controller_state = {}
    total_reward, collisions, food, water, switches, steps = 0.0, 0, 0, 0, 0, 0

    for _ in range(max_steps):
        if mode == "random_walk":
            action = heuristic_action(rng)
        elif mode == "reactive":
            action, controller_state, switched = reactive_controller(rng, grid, body, controller_state)
            switches += int(switched)
        elif mode == "options":
            action, controller_state, switched = options_controller(rng, grid, body, controller_state)
            switches += int(switched)
        else:
            raise ValueError(mode)

        (grid, body), reward, done, info = env.step(action)
        total_reward += reward
        collisions += int(info["collided"])
        if info["resource"] == "food":
            food += 1
        elif info["resource"] == "water":
            water += 1
        steps += 1
        if done:
            break

    return dict(reward=total_reward, collisions=collisions, food=food, water=water,
                switches=switches, steps=steps, hit_cap=(steps >= max_steps))


def summarize(eps, mode):
    r = np.array([e["reward"] for e in eps])
    c = np.array([e["collisions"] for e in eps])
    f = np.array([e["food"] for e in eps])
    w = np.array([e["water"] for e in eps])
    s = np.array([e["steps"] for e in eps])
    sw = np.array([e["switches"] for e in eps])
    cap = np.array([e["hit_cap"] for e in eps])
    return {
        "mode": mode,
        "mean_reward": float(r.mean()),
        "total_collisions": int(c.sum()),
        "mean_collisions": float(c.mean()),
        "mean_food": float(f.mean()),
        "mean_water": float(w.mean()),
        "total_resources": int(f.sum() + w.sum()),
        "mean_episode_length": float(s.mean()),
        "min_episode_length": int(s.min()),
        "max_episode_length": int(s.max()),
        "episodes_hitting_200_cap": int(cap.sum()),
        "episodes_ending_early": int((~cap).sum()),
        "mean_goal_switches": float(sw.mean()) if mode != "random_walk" else float("nan"),
    }


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    means = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(seeds=None, label="episodes"):
    seeds = TEST_SEEDS if seeds is None else seeds
    results = {}
    per_episode = {}
    for mode in ["random_walk", "reactive", "options"]:
        eps = [run_episode(seed, mode, action_rng_seed=seed + 5000) for seed in seeds]
        per_episode[mode] = eps
        results[mode] = summarize(eps, mode)

    print(f"Paired comparison on {len(seeds)} {label} (seeds {seeds[0]}-{seeds[-1]}), 200-step cap "
          f"(actual lengths reported separately below):\n")
    header = (f"{'mode':>12s} {'reward':>9s} {'collisions':>11s} {'food':>6s} {'water':>6s} "
              f"{'total_res':>9s} {'switches':>9s} {'ep_len(mean/min/max)':>21s} {'hit_cap':>8s}")
    print(header)
    for mode in ["random_walk", "reactive", "options"]:
        s = results[mode]
        switches_str = "n/a" if np.isnan(s["mean_goal_switches"]) else f"{s['mean_goal_switches']:.2f}"
        print(f"{mode:>12s} {s['mean_reward']:>9.3f} {s['total_collisions']:>11d} "
              f"{s['mean_food']:>6.3f} {s['mean_water']:>6.3f} {s['total_resources']:>9d} "
              f"{switches_str:>9s} "
              f"{s['mean_episode_length']:>6.1f}/{s['min_episode_length']:>3d}/{s['max_episode_length']:>3d}     "
              f"{s['episodes_hitting_200_cap']:>4d}/{len(seeds)}")

    print(f"\nEpisodes ending before the 200-step cap (died: energy, hydration, or integrity hit zero):")
    for mode in ["random_walk", "reactive", "options"]:
        print(f"  {mode:>12s}: {results[mode]['episodes_ending_early']:4d} of {len(seeds)}")

    print("\n--- Paired comparisons (resources collected, per episode) ---")
    for cmp_label, (a, b) in {"reactive vs random_walk": ("random_walk", "reactive"),
                               "options vs random_walk": ("random_walk", "options"),
                               "options vs reactive": ("reactive", "options")}.items():
        res_a = np.array([e["food"] + e["water"] for e in per_episode[a]])
        res_b = np.array([e["food"] + e["water"] for e in per_episode[b]])
        diffs = res_b - res_a
        lo, hi = bootstrap_ci(diffs)
        print(f"  {cmp_label}: mean {diffs.mean():+.4f} more resources/ep, 95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes zero' if lo > 0 or hi < 0 else 'includes zero'}")

    print("\n--- Paired comparisons (collisions, per episode) ---")
    for cmp_label, (a, b) in {"reactive vs random_walk": ("random_walk", "reactive"),
                               "options vs random_walk": ("random_walk", "options"),
                               "options vs reactive": ("reactive", "options")}.items():
        col_a = np.array([e["collisions"] for e in per_episode[a]])
        col_b = np.array([e["collisions"] for e in per_episode[b]])
        diffs = col_a - col_b
        lo, hi = bootstrap_ci(diffs)
        print(f"  {cmp_label}: mean {diffs.mean():+.4f} fewer collisions/ep, 95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes zero' if lo > 0 or hi < 0 else 'includes zero'}")

    return results, per_episode


def test_option_completes_on_collection_then_respawn():
    """
    Focused regression test for the exact bug reported: the environment
    respawns a collected resource to a new cell before the next
    observation is built, so completion must be checked against the cell
    that was being approached, not against whatever the live observation
    shows after the respawn. Forces the agent directly onto a resource
    cell and asserts the option registers completion (a fresh goal/target
    chosen) rather than continuing to chase the old goal label with a
    silently-updated target.
    """
    env = MockGridWorld(seed=3000)
    grid, body = env.reset()
    food_pos_before = np.argwhere(grid[2] == 1)[0].copy()

    # Commit the controller to "food" with that exact cell as its target,
    # as it would be mid-episode.
    state = {"goal": "food", "duration": 5, "target_pos": food_pos_before}

    # Force the agent one cell below the food (row+1) so action 1 (up) lands exactly on it.
    approach_pos = food_pos_before.copy()
    approach_pos[0] += 1
    env.pos = approach_pos.clip(1, 7)

    grid, body = env._obs()
    action, new_state, switched = options_controller(np.random.default_rng(0), grid, body, state)
    assert action == 1, f"Setup error: expected action 1 (up) to reach the food cell, got {action}."

    (grid2, body2), reward, done, info = env.step(action)
    assert info["resource"] == "food", "Setup error: expected this step to actually collect food."
    new_food_pos = np.argwhere(grid2[2] == 1)[0]
    assert not np.array_equal(new_food_pos, food_pos_before), (
        "Setup error: expected the environment to respawn food to a new cell -- "
        "this test is meaningless if it doesn't."
    )

    # This is the actual check: does the controller, seeing this new
    # observation, recognize the PREVIOUS target was reached?
    self_pos_now = np.argwhere(grid2[1] == 1)[0]
    assert np.array_equal(self_pos_now, food_pos_before), "Setup error: agent should now be exactly where food was."

    _action2, state_after, switched_after = options_controller(np.random.default_rng(0), grid2, body2, new_state)
    assert state_after["duration"] == 1, "Completed option must reset duration"
    assert state_after["goal"] == ("food" if body2[0] < body2[1] else "water")
    assert state_after["target_pos"] is not None
    assert not np.array_equal(state_after["target_pos"], food_pos_before) or state_after["goal"] != "food" or switched_after, (
        "FAIL: controller did not register completion of the food option after actually "
        "reaching it -- it would keep chasing the respawned location under the old goal, "
        "which is the exact bug reported."
    )
    print(f"PASS  after physically reaching the food cell, the controller registered completion "
          f"(new target_pos={state_after['target_pos']}, goal={state_after['goal']}, duration reset to "
          f"{state_after['duration']}) rather than silently continuing toward the respawned cell under "
          f"the same uncompleted option.")


if __name__ == "__main__":
    print("=== Focused regression test: collection followed by respawn ===\n")
    test_option_completes_on_collection_then_respawn()

    print("\n=== EXPLORATORY (retained seeds 3000-3199) ===\n")
    run(seeds=TEST_SEEDS, label="retained exploratory episodes")

    print("\n\n=== CONFIRMATION (fresh seeds 5000-5199, never used before) ===\n")
    run(seeds=CONFIRMATION_SEEDS, label="fresh confirmation episodes")
