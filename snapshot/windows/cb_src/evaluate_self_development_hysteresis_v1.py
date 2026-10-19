#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    load_neutral_model,
)
from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.recurrent_self_trajectory_v1 import load_trajectory_model
from continuous_brain.self_development_hysteresis_v1 import (
    MEANINGFUL_SELF_ESTIMATE_GAIN,
    NEUTRAL_RELIABILITY,
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
    HistoryRelativeDevelopmentalController,
)
from continuous_brain.self_development_loop_v1 import (
    DevelopmentalState,
    load_developmental_state,
    reliability_from_memory,
    save_developmental_state,
)
from evaluate_self_development_loop_v1 import (
    AUTOBIO_MODEL,
    TRAJECTORY_MODEL,
    behavior_to_history_assay,
    bootstrap_profile,
    build_bootstrap_memory,
    discrepancy_scalar,
    execute_development_episode,
    fragile_primary,
    initial_capabilities,
    mean,
    primary_index,
    stable_primary,
)

DEV_IDS = range(10000, 10128)
TEST_IDS = range(20000, 20128)

DEVELOPMENT_CYCLES = 8

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
MIN_FROZEN_HISTORY_BLOCK = .95
MIN_PERSISTENCE = 1.0

SOURCE_V033_RESULT = Path(
    "state/self_development_loop_v1/result.json"
)
AUTOBIO_TEST = Path(
    "state/autobiographical_self_discrepancy_neutral_test_v1/result.json"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

PROTECTED = {
    "v0330_result": SOURCE_V033_RESULT,
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
    "v0321_result":
        Path("state/autobiographical_self_discrepancy_neutral_v1/result.json"),
}


def reliability_for_primary(rel, primary):
    return float(rel[primary_index(primary)].item())


def initialize_phases(
    autobiographical_model,
    memory,
    controller,
):
    rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    phases = {
        1: controller.initialize(float(rel[0].item())),
        2: controller.initialize(float(rel[1].item())),
    }
    return phases, rel


def run_subject_hysteresis(
    trajectory_model,
    autobiographical_model,
    controller,
    subject_id: int,
    *,
    freeze_decision_history: bool = False,
):
    memory = build_bootstrap_memory(
        trajectory_model,
        subject_id,
    )
    bootstrap_memory = copy.deepcopy(memory)

    cap_a, cap_b = initial_capabilities(subject_id)
    state = DevelopmentalState(
        capability_a=cap_a,
        capability_b=cap_b,
        cycle=0,
    )

    phases, initial_rel = initialize_phases(
        autobiographical_model,
        bootstrap_memory,
        controller,
    )

    fp = fragile_primary(subject_id)
    sp = stable_primary(subject_id)

    initial_fragile_r = reliability_for_primary(
        initial_rel,
        fp,
    )
    initial_stable_r = reliability_for_primary(
        initial_rel,
        sp,
    )

    bootstrap_fragile = [
        discrepancy_scalar(r)
        for r in bootstrap_memory.records
        if int(r.primary) == int(fp)
    ]
    bootstrap_discrepancy = mean(
        bootstrap_fragile
    )

    fragile_strategies = []
    stable_strategies = []
    fragile_discrepancies = []
    reliability_trace = [
        initial_fragile_r
    ]

    for cycle in range(DEVELOPMENT_CYCLES):
        order = (
            [1, 2]
            if (cycle + subject_id) % 2 == 0
            else [2, 1]
        )

        for primary in order:
            decision_memory = (
                bootstrap_memory
                if freeze_decision_history
                else memory
            )
            decision_rel = reliability_from_memory(
                autobiographical_model,
                decision_memory,
            )
            current = reliability_for_primary(
                decision_rel,
                primary,
            )

            phases[primary] = controller.update(
                phases[primary],
                current,
            )
            strategy = controller.strategy(
                phases[primary]
            )

            record, episode_metrics = execute_development_episode(
                trajectory_model,
                subject_id,
                cycle,
                primary,
                strategy,
                state,
            )
            memory.append(record)

            if primary == fp:
                fragile_strategies.append(
                    int(strategy)
                )
                fragile_discrepancies.append(
                    episode_metrics["discrepancy"]
                )
            else:
                stable_strategies.append(
                    int(strategy)
                )

        state.cycle += 1

        current_memory = (
            bootstrap_memory
            if freeze_decision_history
            else memory
        )
        rel = reliability_from_memory(
            autobiographical_model,
            current_memory,
        )
        reliability_trace.append(
            reliability_for_primary(
                rel,
                fp,
            )
        )

    final_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    final_fragile_r = reliability_for_primary(
        final_rel,
        fp,
    )
    final_stable_r = reliability_for_primary(
        final_rel,
        sp,
    )

    initial_fragile_cap = (
        cap_a if fp == 1 else cap_b
    )
    final_fragile_cap = state.capability_for(fp)

    late_discrepancy = mean(
        fragile_discrepancies[-2:]
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
            >= initial_fragile_r
            + MIN_RELIABILITY_GAIN
        and switched
        and final_fragile_cap
            >= initial_fragile_cap
            + MIN_CAPABILITY_GAIN
        and late_discrepancy
            <= MAX_LATE_DISCREPANCY_RATIO
            * max(bootstrap_discrepancy, 1e-6)
    )

    return {
        "subject_id": int(subject_id),
        "fragile_primary": int(fp),
        "initial_memory": bootstrap_memory,
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
        "freeze_decision_history":
            bool(freeze_decision_history),
    }


def donor_phase_assay(
    trajectory_model,
    autobiographical_model,
    controller,
):
    fragile_scaffold = []
    donor_direct = []
    erased_direct = []

    from continuous_brain.autobiographical_self_discrepancy_v1 import (
        AutobiographicalTrajectoryMemory,
    )

    empty = AutobiographicalTrajectoryMemory()
    empty_rel = reliability_from_memory(
        autobiographical_model,
        empty,
    )

    for sid in range(12000, 12128, 2):
        even = build_bootstrap_memory(
            trajectory_model,
            sid,
        )
        odd = build_bootstrap_memory(
            trajectory_model,
            sid + 1,
        )

        for primary in (1, 2):
            fragile_mem = (
                even if primary == 1 else odd
            )
            stable_mem = (
                odd if primary == 1 else even
            )

            fragile_rel = reliability_from_memory(
                autobiographical_model,
                fragile_mem,
            )
            stable_rel = reliability_from_memory(
                autobiographical_model,
                stable_mem,
            )

            f_phase = controller.initialize(
                reliability_for_primary(
                    fragile_rel,
                    primary,
                )
            )
            d_phase = controller.initialize(
                reliability_for_primary(
                    stable_rel,
                    primary,
                )
            )
            e_phase = controller.initialize(
                reliability_for_primary(
                    empty_rel,
                    primary,
                )
            )

            fragile_scaffold.append(
                float(
                    controller.strategy(f_phase)
                    == STRATEGY_SCAFFOLD
                )
            )
            donor_direct.append(
                float(
                    controller.strategy(f_phase)
                    == STRATEGY_SCAFFOLD
                    and controller.strategy(d_phase)
                    == STRATEGY_DIRECT
                )
            )
            erased_direct.append(
                float(
                    controller.strategy(e_phase)
                    == STRATEGY_DIRECT
                )
            )

    return {
        "fragile_history_selects_scaffold":
            mean(fragile_scaffold),
        "donor_history_flips_strategy":
            mean(donor_direct),
        "erased_history_selects_neutral_direct":
            mean(erased_direct),
    }


def persistence_assay(
    autobiographical_model,
    controller,
    subject_result,
    state_dir,
):
    state = subject_result["state"]
    memory = subject_result["final_memory"]
    fp = subject_result["fragile_primary"]

    baseline_rel = reliability_from_memory(
        autobiographical_model,
        subject_result["initial_memory"],
    )
    current_rel = reliability_from_memory(
        autobiographical_model,
        memory,
    )

    phase = controller.initialize(
        reliability_for_primary(
            baseline_rel,
            fp,
        )
    )
    phase = controller.update(
        phase,
        reliability_for_primary(
            current_rel,
            fp,
        ),
    )
    before_strategy = controller.strategy(
        phase
    )

    path = state_dir / "developmental_state_probe.pt"
    save_developmental_state(
        path,
        state,
        memory,
        metadata={
            "baseline_fragile_reliability":
                reliability_for_primary(
                    baseline_rel,
                    fp,
                ),
            "primary":
                int(fp),
        },
    )

    loaded_state, loaded_memory, payload = load_developmental_state(
        path
    )

    loaded_rel = reliability_from_memory(
        autobiographical_model,
        loaded_memory,
    )
    loaded_phase = controller.initialize(
        float(
            payload["metadata"][
                "baseline_fragile_reliability"
            ]
        )
    )
    loaded_phase = controller.update(
        loaded_phase,
        reliability_for_primary(
            loaded_rel,
            fp,
        ),
    )
    after_strategy = controller.strategy(
        loaded_phase
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
                    current_rel,
                    loaded_rel,
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
        default="state/self_development_hysteresis_v1",
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
            raise FileNotFoundError(f"{name}: {path}")

    source = json.loads(
        SOURCE_V033_RESULT.read_text()
    )
    if source.get("development_gate_passed") is not False:
        raise RuntimeError(
            "v0.33.0 historical DEV FAIL required"
        )
    if (
        source.get("split", {})
        .get("test_touched")
        is not False
    ):
        raise RuntimeError(
            "v0.33 TEST already touched"
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

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_DEVELOPMENT_HYSTERESIS_V1")
    print("device: cpu")
    print("training_performed: false")
    print("source_v0330_formal_status: FAIL")
    print("repair_scope: controller_semantics_only")
    print("frozen_trajectory_model: true")
    print("frozen_autobiographical_model: true")
    print("development_window_reused: subjects 10000-10127")
    print("new_development_window_consumed: false")
    print("development_test_subjects: 20000-20127")
    print("development_test_touched: false")
    print("neutral_reliability:", NEUTRAL_RELIABILITY)
    print(
        "graduation_self_estimate_gain:",
        MEANINGFUL_SELF_ESTIMATE_GAIN,
    )
    print(
        "graduation_threshold_source:",
        "v0.33.0 preregistered MIN_RELIABILITY_GAIN",
    )
    print("graduation_hysteresis: one_way_for_this_assay")

    trajectory_model, _ = load_trajectory_model(
        TRAJECTORY_MODEL,
        "cpu",
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    autobiographical_model, _ = load_neutral_model(
        AUTOBIO_MODEL,
        "cpu",
    )
    for p in autobiographical_model.parameters():
        p.requires_grad_(False)
    autobiographical_model.eval()

    controller = HistoryRelativeDevelopmentalController()

    subjects = []
    frozen_controls = []

    for n, sid in enumerate(DEV_IDS, 1):
        subjects.append(
            run_subject_hysteresis(
                trajectory_model,
                autobiographical_model,
                controller,
                int(sid),
                freeze_decision_history=False,
            )
        )
        frozen_controls.append(
            run_subject_hysteresis(
                trajectory_model,
                autobiographical_model,
                controller,
                int(sid),
                freeze_decision_history=True,
            )
        )
        if n == 1 or n % 32 == 0 or n == 128:
            print(
                f"repair_development_subjects={n}/128"
            )

    donor = donor_phase_assay(
        trajectory_model,
        autobiographical_model,
        controller,
    )
    behavior_history = behavior_to_history_assay(
        trajectory_model,
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
    switch_rate = mean(
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

    frozen_history_no_graduation = mean(
        float(
            not s["switched_to_direct"]
            and all(
                st == STRATEGY_SCAFFOLD
                for st in s["fragile_strategies"]
            )
        )
        for s in frozen_controls
    )

    closed = {
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
            switch_rate,
        "mean_fragile_capability_gain":
            capability_gain_mean,
        "mean_late_to_bootstrap_discrepancy_ratio":
            discrepancy_ratio_mean,
        "discrepancy_reduction_rate":
            discrepancy_reduction_rate,
        "full_closed_loop_rate":
            full_loop_rate,
        "frozen_history_blocks_graduation_rate":
            frozen_history_no_graduation,
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
            switch_rate
            >= MIN_BEHAVIOR_SWITCH,
        "frozen_autobiography_blocks_behavior_change":
            frozen_history_no_graduation
            >= MIN_FROZEN_HISTORY_BLOCK,

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
            "protected artifact changed during v0.33.1"
        )

    result = {
        "format":
            "continuous_brain.self_development_hysteresis_v1",
        "repair_dev_gate_passed":
            bool(passed),
        "training_performed":
            False,
        "source_v0330_formal_status":
            "FAIL",
        "diagnosis": {
            "failed_absolute_threshold":
                0.50,
            "why":
                "v0.32 reliability is historical calibration, not a developmental readiness boundary",
            "source_fragile_switch_rate":
                source[
                    "closed_loop_metrics"
                ][
                    "fragile_behavior_switch_to_direct_rate"
                ],
            "source_stable_direct_rate":
                source[
                    "closed_loop_metrics"
                ][
                    "stable_commitment_always_direct_rate"
                ],
        },
        "repair": {
            "baseline_relative":
                True,
            "one_way_hysteresis":
                True,
            "graduation_gain":
                MEANINGFUL_SELF_ESTIMATE_GAIN,
            "graduation_gain_source":
                "v0.33.0 preregistered MIN_RELIABILITY_GAIN",
            "neural_training":
                False,
            "hidden_capability_used_for_decision":
                False,
        },
        "split": {
            "reused_dev": [10000, 10127],
            "test": [20000, 20127],
            "test_touched": False,
        },
        "closed_loop_metrics":
            closed,
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
                "autobiographical improvement, relative to the commitment's own baseline self-estimate, causes a developmental strategy transition; freezing autobiography prevents that transition even while practice continues",
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

    print("\nSOURCE_V0330")
    print(
        "stable_direct_rate:",
        source["closed_loop_metrics"][
            "stable_commitment_always_direct_rate"
        ],
    )
    print(
        "fragile_switch_rate:",
        source["closed_loop_metrics"][
            "fragile_behavior_switch_to_direct_rate"
        ],
    )
    print(
        "full_loop_rate:",
        source["closed_loop_metrics"][
            "full_closed_loop_rate"
        ],
    )

    print("\nREPAIRED_CLOSED_LOOP")
    for k, v in closed.items():
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
        "SELF_DEVELOPMENT_HYSTERESIS_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_DEVELOPMENT_HYSTERESIS_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
