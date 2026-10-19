#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    AutobiographicalTrajectoryMemory,
    TrajectoryMemoryRecord,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_OBSERVE,
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    sha256,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    MODE_NORMAL,
    PRIMARY_A,
    PRIMARY_B,
    load_trajectory_model,
    rollout,
    update_explicit_intention,
)
from continuous_brain.self_development_loop_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
    DevelopmentalController,
    DevelopmentalState,
    capability_update,
    choose_strategy_from_memory,
    load_developmental_state,
    reliability_from_memory,
    save_developmental_state,
    strategy_failure_probability,
)
from train_evaluate_autobiographical_self_discrepancy_v1 import (
    build_record,
    episode_initial_state,
)
from train_evaluate_joint_self_world_prediction_v1 import transition

DEV_IDS = range(10000, 10128)
TEST_IDS = range(20000, 20128)

BOOTSTRAP_EPISODES = 12
DEVELOPMENT_CYCLES = 8
PLAN_LEN = 4

DIRECT_THRESHOLD = .50

MIN_PROFILE_IDENTIFICATION = .95
MIN_HISTORY_GUIDED_STRATEGY = .95
MIN_DONOR_FLIP = .95
MIN_BEHAVIOR_TO_HISTORY = .90
MIN_RELIABILITY_GAIN = .12
MIN_RELIABILITY_GAIN_RATE = .80
MIN_BEHAVIOR_SWITCH = .75
MIN_CAPABILITY_GAIN = .35
MIN_STABLE_DIRECT = .95
MAX_LATE_DISCREPANCY_RATIO = .75
MIN_FULL_LOOP = .70
MIN_PERSISTENCE = 1.0

AUTOBIO_MODEL = Path(
    "state/autobiographical_self_discrepancy_neutral_v1/"
    "neutral_autobiographical_self_model.pt"
)
AUTOBIO_TEST = Path(
    "state/autobiographical_self_discrepancy_neutral_test_v1/result.json"
)
TRAJECTORY_MODEL = Path(
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

PROTECTED = {
    "autobio_model": AUTOBIO_MODEL,
    "autobio_test": AUTOBIO_TEST,
    "trajectory_model": TRAJECTORY_MODEL,
    "trajectory_test": TRAJECTORY_TEST,
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
    "self_test":
        Path("state/persistent_self_autobiographical_test_v1/result.json"),
    "v029_test":
        Path("state/self_broadcast_centrality_test_v1/result.json"),
    "v030_result":
        Path("state/joint_self_world_prediction_v1/result.json"),
    "v031_test":
        Path("state/recurrent_self_trajectory_test_v1/result.json"),
    "v0320_result":
        Path("state/autobiographical_self_discrepancy_v1/result.json"),
    "v0321_result":
        Path("state/autobiographical_self_discrepancy_neutral_v1/result.json"),
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def fragile_primary(subject_id: int) -> int:
    return (
        PRIMARY_A
        if int(subject_id) % 2 == 0
        else PRIMARY_B
    )


def stable_primary(subject_id: int) -> int:
    return (
        PRIMARY_B
        if fragile_primary(subject_id) == PRIMARY_A
        else PRIMARY_A
    )


def primary_index(primary: int) -> int:
    return 0 if int(primary) == PRIMARY_A else 1


def bootstrap_profile(subject_id: int):
    rng = random.Random(
        0x330000 + 100003 * int(subject_id)
    )
    fragile_r = .06 + .08 * rng.random()
    stable_r = .90 + .05 * rng.random()

    if fragile_primary(subject_id) == PRIMARY_A:
        return fragile_r, stable_r
    return stable_r, fragile_r


def initial_capabilities(subject_id: int):
    rng = random.Random(
        0x330100 + 104729 * int(subject_id)
    )
    fragile = .18 + .09 * rng.random()
    stable = .87 + .08 * rng.random()

    if fragile_primary(subject_id) == PRIMARY_A:
        return fragile, stable
    return stable, fragile


def build_bootstrap_memory(
    trajectory_model,
    subject_id: int,
):
    ra, rb = bootstrap_profile(subject_id)
    memory = AutobiographicalTrajectoryMemory()

    for episode in range(BOOTSTRAP_EPISODES):
        # Alternate commitments while varying which commitment appears first
        # across subjects. This avoids a fixed chronology cue.
        choose_a = (
            (episode + int(subject_id)) % 2 == 0
        )
        primary = (
            PRIMARY_A if choose_a else PRIMARY_B
        )
        reliability = (
            ra if primary == PRIMARY_A else rb
        )
        memory.append(
            build_record(
                trajectory_model,
                int(subject_id) + 70000,
                episode,
                primary,
                reliability,
            )
        )
    return memory


def plan(primary: int, strategy: int):
    work = (
        ACTION_WORK_A
        if int(primary) == PRIMARY_A
        else ACTION_WORK_B
    )
    if int(strategy) == STRATEGY_SCAFFOLD:
        return [
            ACTION_OBSERVE,
            work,
            ACTION_OBSERVE,
            work,
        ]
    return [
        work,
        work,
        ACTION_OBSERVE,
        work,
    ]


def discrepancy_scalar(record: TrajectoryMemoryRecord) -> float:
    # Feature[-2] is total mean absolute self discrepancy by v0.32 contract.
    return float(record.feature()[-2].item())


def execute_development_episode(
    trajectory_model,
    subject_id: int,
    cycle: int,
    primary: int,
    strategy: int,
    state: DevelopmentalState,
    *,
    forced_uniforms=None,
):
    world, personal = episode_initial_state(
        int(subject_id) + 90000,
        100 + int(cycle) * 2 + int(primary),
        primary,
    )
    actions = plan(primary, strategy)

    expected = rollout(
        trajectory_model,
        world,
        personal.vector()[:6],
        int(primary),
        MODE_NORMAL,
        actions,
    )

    expected_self = expected["selfs"].detach().cpu()
    expected_world = expected["worlds"].detach().cpu()
    expected_modes = torch.tensor(
        expected["modes"],
        dtype=torch.long,
    )

    actual_self = []
    actual_world = []
    actual_modes = []

    actual_state = personal
    actual_w = world
    intent_primary = int(primary)
    intent_mode = MODE_NORMAL

    rng = random.Random(
        0x330200
        + 130363 * int(subject_id)
        + 4001 * int(cycle)
        + 19 * int(primary)
    )
    forced = list(forced_uniforms) if forced_uniforms is not None else None
    work_index = 0

    capability_before = state.capability_for(primary)
    work_successes = 0
    work_failures = 0

    for action in actions:
        is_work = action in (
            ACTION_WORK_A,
            ACTION_WORK_B,
        )

        surprise = False
        if is_work:
            capability = state.capability_for(primary)
            p_fail = strategy_failure_probability(
                capability,
                strategy,
            )
            if forced is not None:
                u = float(
                    forced[min(work_index, len(forced) - 1)]
                )
            else:
                u = rng.random()
            surprise = u < p_fail
            work_index += 1

        actual_w, actual_state, _, _ = transition(
            actual_w,
            actual_state,
            action,
            surprise=surprise,
        )

        if is_work:
            success = not surprise
            if success:
                work_successes += 1
            else:
                work_failures += 1

            state.set_capability_for(
                primary,
                capability_update(
                    state.capability_for(primary),
                    success=success,
                    strategy=strategy,
                ),
            )

        intent = update_explicit_intention(
            intent_primary,
            intent_mode,
            actual_state.vector()[:6],
        )
        intent_primary = int(intent.primary)
        intent_mode = int(intent.mode)

        actual_world.append(actual_w.clone())
        actual_self.append(
            actual_state.vector()[:6].clone()
        )
        actual_modes.append(intent_mode)

    record = TrajectoryMemoryRecord(
        episode_id=(
            int(subject_id) * 1000
            + int(cycle) * 10
            + int(primary)
        ),
        primary=int(primary),
        expected_self=expected_self,
        actual_self=torch.stack(actual_self),
        expected_world=expected_world,
        actual_world=torch.stack(actual_world),
        expected_modes=expected_modes,
        actual_modes=torch.tensor(
            actual_modes,
            dtype=torch.long,
        ),
    )

    return record, {
        "capability_before": capability_before,
        "capability_after": state.capability_for(primary),
        "work_successes": work_successes,
        "work_failures": work_failures,
        "discrepancy": discrepancy_scalar(record),
    }


def run_subject(
    trajectory_model,
    autobiographical_model,
    controller,
    subject_id: int,
):
    memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )
    initial_memory = copy.deepcopy(memory)

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )

    fp = fragile_primary(subject_id)
    sp = stable_primary(subject_id)

    initial_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    initial_fragile_r = float(
        initial_rel[primary_index(fp)].item()
    )
    initial_stable_r = float(
        initial_rel[primary_index(sp)].item()
    )

    # Bootstrap discrepancy baseline for fragile commitment.
    bootstrap_fragile = [
        discrepancy_scalar(r)
        for r in memory.records
        if int(r.primary) == int(fp)
    ]
    bootstrap_discrepancy = mean(
        bootstrap_fragile
    )

    fragile_strategies = []
    stable_strategies = []
    fragile_development_discrepancies = []
    reliability_trace = [
        initial_fragile_r
    ]

    for cycle in range(DEVELOPMENT_CYCLES):
        # Alternate execution order so chronology is not fixed by commitment.
        order = (
            [PRIMARY_A, PRIMARY_B]
            if (cycle + subject_id) % 2 == 0
            else [PRIMARY_B, PRIMARY_A]
        )

        for primary in order:
            strategy, _ = choose_strategy_from_memory(
                autobiographical_model,
                memory,
                primary,
                controller,
            )

            record, episode_metrics = (
                execute_development_episode(
                    trajectory_model,
                    subject_id,
                    cycle,
                    primary,
                    strategy,
                    state,
                )
            )
            memory.append(record)

            if primary == fp:
                fragile_strategies.append(
                    int(strategy)
                )
                fragile_development_discrepancies.append(
                    episode_metrics["discrepancy"]
                )
            else:
                stable_strategies.append(
                    int(strategy)
                )

        state.cycle += 1
        rel = reliability_from_memory(
            autobiographical_model,
            memory,
        )
        reliability_trace.append(
            float(
                rel[primary_index(fp)].item()
            )
        )

    final_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    final_fragile_r = float(
        final_rel[primary_index(fp)].item()
    )
    final_stable_r = float(
        final_rel[primary_index(sp)].item()
    )

    initial_fragile_cap = (
        cap_a if fp == PRIMARY_A else cap_b
    )
    final_fragile_cap = state.capability_for(fp)

    late_discrepancy = mean(
        fragile_development_discrepancies[-2:]
    )

    switched = (
        len(fragile_strategies) >= 2
        and fragile_strategies[0] == STRATEGY_SCAFFOLD
        and fragile_strategies[-1] == STRATEGY_DIRECT
    )

    full_loop = (
        initial_fragile_r < initial_stable_r
        and fragile_strategies[0] == STRATEGY_SCAFFOLD
        and final_fragile_r
            >= initial_fragile_r + MIN_RELIABILITY_GAIN
        and switched
        and final_fragile_cap
            >= initial_fragile_cap + MIN_CAPABILITY_GAIN
        and late_discrepancy
            <= MAX_LATE_DISCREPANCY_RATIO
            * max(bootstrap_discrepancy, 1e-6)
    )

    return {
        "subject_id": int(subject_id),
        "fragile_primary": int(fp),
        "initial_memory": initial_memory,
        "final_memory": memory,
        "state": state,
        "initial_fragile_reliability":
            initial_fragile_r,
        "initial_stable_reliability":
            initial_stable_r,
        "final_fragile_reliability":
            final_fragile_r,
        "final_stable_reliability":
            final_stable_r,
        "fragile_reliability_gain":
            final_fragile_r
            - initial_fragile_r,
        "initial_fragile_capability":
            initial_fragile_cap,
        "final_fragile_capability":
            final_fragile_cap,
        "fragile_capability_gain":
            final_fragile_cap
            - initial_fragile_cap,
        "bootstrap_fragile_discrepancy":
            bootstrap_discrepancy,
        "late_fragile_discrepancy":
            late_discrepancy,
        "late_to_bootstrap_discrepancy_ratio":
            late_discrepancy
            / max(bootstrap_discrepancy, 1e-6),
        "fragile_strategies":
            fragile_strategies,
        "stable_strategies":
            stable_strategies,
        "reliability_trace":
            reliability_trace,
        "switched_to_direct":
            bool(switched),
        "full_loop":
            bool(full_loop),
    }


def donor_history_assay(
    trajectory_model,
    autobiographical_model,
    controller,
):
    flips = []
    guided = []
    erased_direct = []

    empty = AutobiographicalTrajectoryMemory()
    erased_rel = reliability_from_memory(
        autobiographical_model,
        empty,
    )

    for sid in range(12000, 12128, 2):
        even_memory = build_bootstrap_memory(
            trajectory_model,
            sid,
        )
        odd_memory = build_bootstrap_memory(
            trajectory_model,
            sid + 1,
        )

        # Even subject: A fragile, B stable.
        # Odd subject:  A stable, B fragile.
        for primary in (PRIMARY_A, PRIMARY_B):
            own = (
                even_memory
                if primary == PRIMARY_A
                else odd_memory
            )
            donor = (
                odd_memory
                if primary == PRIMARY_A
                else even_memory
            )

            own_strategy, _ = (
                choose_strategy_from_memory(
                    autobiographical_model,
                    own,
                    primary,
                    controller,
                )
            )
            donor_strategy, _ = (
                choose_strategy_from_memory(
                    autobiographical_model,
                    donor,
                    primary,
                    controller,
                )
            )

            guided.append(
                float(
                    own_strategy
                    == STRATEGY_SCAFFOLD
                )
            )
            flips.append(
                float(
                    own_strategy
                    == STRATEGY_SCAFFOLD
                    and donor_strategy
                    == STRATEGY_DIRECT
                )
            )

            idx = primary_index(primary)
            erased_strategy = controller.choose(
                float(erased_rel[idx].item())
            )
            erased_direct.append(
                float(
                    erased_strategy
                    == STRATEGY_DIRECT
                )
            )

    return {
        "fragile_history_selects_scaffold":
            mean(guided),
        "donor_history_flips_strategy":
            mean(flips),
        "erased_history_selects_neutral_direct":
            mean(erased_direct),
    }


def behavior_to_history_assay(
    trajectory_model,
):
    effects = []

    for i in range(128):
        sid = 14000 + i
        primary = (
            PRIMARY_A
            if i % 2 == 0
            else PRIMARY_B
        )
        cap = .20 + .05 * (
            (i % 7) / 6.0
        )

        if primary == PRIMARY_A:
            direct_state = DevelopmentalState(
                cap, .92, 0
            )
            scaffold_state = DevelopmentalState(
                cap, .92, 0
            )
        else:
            direct_state = DevelopmentalState(
                .92, cap, 0
            )
            scaffold_state = DevelopmentalState(
                .92, cap, 0
            )

        # Shared exogenous uniforms. At low capability, .40 lies above the
        # scaffold failure probability and below the direct failure probability.
        u = [.40, .40, .40]

        direct_record, _ = (
            execute_development_episode(
                trajectory_model,
                sid,
                0,
                primary,
                STRATEGY_DIRECT,
                direct_state,
                forced_uniforms=u,
            )
        )
        scaffold_record, _ = (
            execute_development_episode(
                trajectory_model,
                sid,
                0,
                primary,
                STRATEGY_SCAFFOLD,
                scaffold_state,
                forced_uniforms=u,
            )
        )

        direct_d = discrepancy_scalar(
            direct_record
        )
        scaffold_d = discrepancy_scalar(
            scaffold_record
        )

        effects.append(
            float(
                direct_d
                >= scaffold_d + .05
            )
        )

    return {
        "strategy_changes_next_autobiographical_discrepancy":
            mean(effects),
    }


def persistence_assay(
    autobiographical_model,
    controller,
    subject_result,
    state_dir,
):
    path = state_dir / "developmental_state_probe.pt"

    state = subject_result["state"]
    memory = subject_result["final_memory"]

    before_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    fp = subject_result["fragile_primary"]
    before_strategy = controller.choose(
        float(
            before_rel[
                primary_index(fp)
            ].item()
        )
    )

    save_developmental_state(
        path,
        state,
        memory,
        metadata={
            "probe": True,
        },
    )
    loaded_state, loaded_memory, _ = (
        load_developmental_state(path)
    )

    after_rel = reliability_from_memory(
        autobiographical_model,
        loaded_memory,
    )
    after_strategy = controller.choose(
        float(
            after_rel[
                primary_index(fp)
            ].item()
        )
    )

    return {
        "capability_exact_after_reload":
            float(
                loaded_state.capability_a
                == state.capability_a
                and loaded_state.capability_b
                == state.capability_b
                and loaded_state.cycle
                == state.cycle
            ),
        "autobiographical_reliability_exact_after_reload":
            float(
                torch.equal(
                    before_rel,
                    after_rel,
                )
            ),
        "developmental_strategy_exact_after_reload":
            float(
                before_strategy
                == after_strategy
            ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_development_loop_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    autobio_test = json.loads(
        AUTOBIO_TEST.read_text()
    )
    if autobio_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.32.2 frozen TEST PASS required"
        )
    if (
        autobio_test.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.32 TEST must be spent/closed"
        )

    trajectory_test = json.loads(
        TRAJECTORY_TEST.read_text()
    )
    if trajectory_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.31.1 frozen TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_DEVELOPMENT_LOOP_V1")
    print("device: cpu")
    print("training_performed: false")
    print("goal: autobiography_changes_behavior_which_changes_future_autobiography")
    print("frozen_trajectory_model: true")
    print("frozen_autobiographical_model: true")
    print("developmental_controller_trainable: false")
    print("development_subject_ids: 10000-10127")
    print("development_test_subject_ids: 20000-20127")
    print("development_test_touched: false")
    print("bootstrap_episodes:", BOOTSTRAP_EPISODES)
    print("development_cycles:", DEVELOPMENT_CYCLES)
    print("episodes_per_cycle: 2")
    print("direct_threshold:", DIRECT_THRESHOLD)

    trajectory_model, _ = (
        load_trajectory_model(
            TRAJECTORY_MODEL,
            "cpu",
        )
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    autobiographical_model, _ = (
        load_neutral_model(
            AUTOBIO_MODEL,
            "cpu",
        )
    )
    for p in autobiographical_model.parameters():
        p.requires_grad_(False)
    autobiographical_model.eval()

    controller = DevelopmentalController(
        direct_threshold=DIRECT_THRESHOLD,
    )

    subjects = []
    for n, sid in enumerate(DEV_IDS, 1):
        subjects.append(
            run_subject(
                trajectory_model,
                autobiographical_model,
                controller,
                int(sid),
            )
        )
        if (
            n == 1
            or n % 32 == 0
            or n == len(list(DEV_IDS))
        ):
            print(
                f"development_subjects={n}/128"
            )

    donor = donor_history_assay(
        trajectory_model,
        autobiographical_model,
        controller,
    )
    behavior_history = (
        behavior_to_history_assay(
            trajectory_model,
        )
    )
    persistence = persistence_assay(
        autobiographical_model,
        controller,
        subjects[0],
        state_dir,
    )

    profile_identification = mean(
        float(
            s["initial_fragile_reliability"]
            < s["initial_stable_reliability"]
        )
        for s in subjects
    )
    initial_scaffold = mean(
        float(
            s["fragile_strategies"][0]
            == STRATEGY_SCAFFOLD
        )
        for s in subjects
    )
    stable_direct = mean(
        float(
            all(
                st == STRATEGY_DIRECT
                for st in s["stable_strategies"]
            )
        )
        for s in subjects
    )
    reliability_gain_mean = mean(
        s["fragile_reliability_gain"]
        for s in subjects
    )
    reliability_gain_rate = mean(
        float(
            s["fragile_reliability_gain"]
            >= MIN_RELIABILITY_GAIN
        )
        for s in subjects
    )
    behavior_switch_rate = mean(
        float(
            s["switched_to_direct"]
        )
        for s in subjects
    )
    capability_gain_mean = mean(
        s["fragile_capability_gain"]
        for s in subjects
    )
    discrepancy_ratio_mean = mean(
        s["late_to_bootstrap_discrepancy_ratio"]
        for s in subjects
    )
    discrepancy_reduction_rate = mean(
        float(
            s["late_to_bootstrap_discrepancy_ratio"]
            <= MAX_LATE_DISCREPANCY_RATIO
        )
        for s in subjects
    )
    full_loop_rate = mean(
        float(s["full_loop"])
        for s in subjects
    )

    closed_loop = {
        "initial_fragile_profile_identification":
            profile_identification,
        "initial_fragile_scaffold_rate":
            initial_scaffold,
        "stable_commitment_always_direct_rate":
            stable_direct,
        "mean_fragile_reliability_gain":
            reliability_gain_mean,
        "fragile_reliability_gain_rate":
            reliability_gain_rate,
        "fragile_behavior_switch_to_direct_rate":
            behavior_switch_rate,
        "mean_fragile_capability_gain":
            capability_gain_mean,
        "mean_late_to_bootstrap_discrepancy_ratio":
            discrepancy_ratio_mean,
        "discrepancy_reduction_rate":
            discrepancy_reduction_rate,
        "full_closed_loop_rate":
            full_loop_rate,
    }

    checks = {
        "initial_autobiography_identifies_fragility":
            profile_identification
            >= MIN_PROFILE_IDENTIFICATION,
        "autobiography_guides_initial_behavior":
            initial_scaffold
            >= MIN_HISTORY_GUIDED_STRATEGY,
        "stable_history_guides_direct_behavior":
            stable_direct
            >= MIN_STABLE_DIRECT,

        "donor_history_changes_behavior":
            donor[
                "donor_history_flips_strategy"
            ] >= MIN_DONOR_FLIP,
        "erased_history_removes_fragility_response":
            donor[
                "erased_history_selects_neutral_direct"
            ] >= MIN_DONOR_FLIP,

        "behavior_changes_next_autobiography":
            behavior_history[
                "strategy_changes_next_autobiographical_discrepancy"
            ] >= MIN_BEHAVIOR_TO_HISTORY,

        "new_autobiography_updates_self_estimate":
            reliability_gain_rate
            >= MIN_RELIABILITY_GAIN_RATE,
        "self_estimate_changes_later_behavior":
            behavior_switch_rate
            >= MIN_BEHAVIOR_SWITCH,
        "behavior_develops_capability":
            capability_gain_mean
            >= MIN_CAPABILITY_GAIN,
        "development_reduces_discrepancy":
            discrepancy_reduction_rate
            >= MIN_RELIABILITY_GAIN_RATE,
        "full_feedback_loop":
            full_loop_rate
            >= MIN_FULL_LOOP,

        "capability_persists":
            persistence[
                "capability_exact_after_reload"
            ] >= MIN_PERSISTENCE,
        "autobiography_persists":
            persistence[
                "autobiographical_reliability_exact_after_reload"
            ] >= MIN_PERSISTENCE,
        "behavior_policy_persists":
            persistence[
                "developmental_strategy_exact_after_reload"
            ] >= MIN_PERSISTENCE,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during v0.33"
        )

    result = {
        "format":
            "continuous_brain.self_development_loop_v1",
        "development_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "frozen_components": {
            "recurrent_self_trajectory_model":
                True,
            "autobiographical_self_model":
                True,
            "historical_brain":
                True,
            "persistent_self":
                True,
        },
        "split": {
            "dev": [10000, 10127],
            "test": [20000, 20127],
            "test_touched": False,
        },
        "architecture": {
            "bootstrap_episodes":
                BOOTSTRAP_EPISODES,
            "development_cycles":
                DEVELOPMENT_CYCLES,
            "episodes_per_cycle":
                2,
            "controller_direct_threshold":
                DIRECT_THRESHOLD,
            "history_to_behavior":
                True,
            "behavior_to_experience":
                True,
            "experience_to_autobiography":
                True,
            "autobiography_to_updated_self_model":
                True,
            "updated_self_model_to_later_behavior":
                True,
        },
        "closed_loop_metrics":
            closed_loop,
        "donor_history_metrics":
            donor,
        "behavior_to_history_metrics":
            behavior_history,
        "persistence_metrics":
            persistence,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "v0322_test_pass_preserved":
            True,
        "development_test_touched":
            False,
        "scope": {
            "claim":
                "a frozen autobiographical self-model changes strategy; strategy changes later expected-vs-actual experience; new autobiography changes the self-estimate and later strategy across repeated developmental cycles",
            "subjective_experience_claimed":
                False,
            "sentience_claimed":
                False,
            "consciousness_claimed":
                False,
        },
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nCLOSED_LOOP")
    for k, v in closed_loop.items():
        print(f"{k}: {v}")

    print("\nDONOR_HISTORY")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nBEHAVIOR_TO_HISTORY")
    for k, v in behavior_history.items():
        print(f"{k}: {v}")

    print("\nPERSISTENCE")
    for k, v in persistence.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("TRAINING_PERFORMED: false")
    print("PROTECTED_UNCHANGED: true")
    print("V0322_TEST_PASS_PRESERVED: true")
    print("DEVELOPMENT_TEST_TOUCHED: false")
    print(
        "SELF_DEVELOPMENT_LOOP_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_DEVELOPMENT_LOOP_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
