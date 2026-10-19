#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    load_self_model,
    sha256,
)
from continuous_brain.self_broadcast_centrality_v1 import (
    apply_workspace_broadcast,
    load_hub,
    self_input_from_state,
)
from continuous_brain.self_broadcast_value_priority_gradient_safe_v1 import (
    load_priority_head,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    build_substrate,
    collect_decision_snapshot,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)
from train_evaluate_self_broadcast_centrality_v1 import (
    TEST_IDS,
    build_dataset,
    make_state,
    targets_from_state,
)

SOURCE_RESULT = Path(
    "state/self_broadcast_centrality_v1/result.json"
)
SOURCE_HUB = Path(
    "state/self_broadcast_centrality_v1/self_broadcast_hub.pt"
)
REPAIR_RESULT = Path(
    "state/self_broadcast_value_priority_gradient_safe_v1/result.json"
)
PRIORITY_HEAD = Path(
    "state/self_broadcast_value_priority_gradient_safe_v1/value_priority_gradient_safe.pt"
)
SELF_MODEL = Path(
    "state/persistent_self_autobiographical_v1/agency_self_model.pt"
)
SELF_SEED = Path(
    "state/persistent_self_autobiographical_v1/persistent_self_seed.pt"
)
SELF_TEST = Path(
    "state/persistent_self_autobiographical_test_v1/result.json"
)
EPISODIC_READER = Path(
    "state/multi_item_episodic_memory_v1/episodic_reader.pt"
)

MIN_POLICY = .95
MIN_RETRIEVAL = .97
MIN_GOAL = .97
MIN_PLANNING = .97
MAX_DEPTH_MAE = .05
MAX_VALUE_MAE = .05
MIN_IDENTITY_COS = .95

MAX_OFF_POLICY = .40
MAX_OFF_GOAL = .30
MAX_OFF_RETRIEVAL = .35
MAX_OFF_INVESTIGATE_RECALL = .05

MIN_DONOR = .95
MIN_HISTORY = .95
MIN_EXISTING_RETRIEVAL = .95
MIN_BROADCAST_CONTINUITY = 1.0
MIN_PRIORITY_IMPROVEMENT = .010
MAX_TRAIN_DEV_TEST_SHIFT = .025

PROTECTED = {
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        EPISODIC_READER,
    "goal_decoder":
        Path("state/memory_guided_goal_option_v1/goal_decoder.pt"),
    "consequence_model":
        Path("state/consequence_replanning_v1/consequence_model.pt"),
    "self_model":
        SELF_MODEL,
    "self_seed":
        SELF_SEED,
    "self_test":
        SELF_TEST,
    "source_result":
        SOURCE_RESULT,
    "source_hub":
        SOURCE_HUB,
    "repair_result":
        REPAIR_RESULT,
    "priority_head":
        PRIORITY_HEAD,
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


@torch.no_grad()
def full_test_metrics(hub, priority_head, base, data):
    hub.eval()
    priority_head.eval()

    out = hub(
        data["self_input"],
        recruitment=1.0,
        enabled=True,
    )
    _, policy_logits = apply_workspace_broadcast(
        base,
        data["workspace"],
        out,
    )
    priority = priority_head(out["broadcast"])

    identity_cos = F.cosine_similarity(
        out["identity_reconstruction"],
        data["identity"],
        dim=-1,
    )

    return {
        "policy_accuracy":
            float(
                (
                    policy_logits.argmax(dim=-1)
                    == data["policy"]
                ).float().mean().item()
            ),
        "retrieval_accuracy":
            float(
                (
                    out["retrieval_logits"].argmax(dim=-1)
                    == data["retrieval"]
                ).float().mean().item()
            ),
        "goal_accuracy":
            float(
                (
                    out["goal_logits"].argmax(dim=-1)
                    == data["goal"]
                ).float().mean().item()
            ),
        "planning_mode_accuracy":
            float(
                (
                    out["planning_mode_logits"].argmax(dim=-1)
                    == data["planning_mode"]
                ).float().mean().item()
            ),
        "planning_depth_mae":
            float(
                (
                    out["planning_depth"]
                    - data["planning_depth"]
                ).abs().mean().item()
            ),
        "repaired_value_priority_mae":
            float(
                (
                    priority
                    - data["value_priority"]
                ).abs().mean().item()
            ),
        "original_linear_value_priority_mae":
            float(
                (
                    out["value_priority"]
                    - data["value_priority"]
                ).abs().mean().item()
            ),
        "identity_cosine":
            float(identity_cos.mean().item()),
        "priority_pred_mean":
            float(priority.mean().item()),
        "priority_target_mean":
            float(data["value_priority"].mean().item()),
    }


@torch.no_grad()
def disabled_metrics(hub, priority_head, base, data):
    out = hub(
        data["self_input"],
        recruitment=1.0,
        enabled=False,
    )
    combined, policy_logits = apply_workspace_broadcast(
        base,
        data["workspace"],
        out,
    )
    base_logits = base.motor(data["workspace"])
    repaired_priority = priority_head(
        out["broadcast"]
    )

    investigate = data["planning_mode"] == 1
    planning_pred = (
        out["planning_mode_logits"].argmax(dim=-1)
    )

    return {
        "workspace_residual_exact_zero":
            bool(
                torch.count_nonzero(
                    out["workspace_residual"]
                ).item() == 0
            ),
        "workspace_exact_base":
            bool(
                torch.equal(
                    combined,
                    data["workspace"],
                )
            ),
        "policy_logits_exact_base":
            bool(
                torch.equal(
                    policy_logits,
                    base_logits,
                )
            ),
        "repaired_priority_exact_zero":
            bool(
                torch.count_nonzero(
                    repaired_priority
                ).item() == 0
            ),
        "off_policy_accuracy":
            float(
                (
                    policy_logits.argmax(dim=-1)
                    == data["policy"]
                ).float().mean().item()
            ),
        "off_goal_accuracy":
            float(
                (
                    out["goal_logits"].argmax(dim=-1)
                    == data["goal"]
                ).float().mean().item()
            ),
        "off_retrieval_accuracy":
            float(
                (
                    out["retrieval_logits"].argmax(dim=-1)
                    == data["retrieval"]
                ).float().mean().item()
            ),
        "off_investigate_recall":
            float(
                (
                    planning_pred[investigate] == 1
                ).float().mean().item()
            )
            if bool(investigate.any())
            else 0.0,
        "all_consumer_logits_exact_zero":
            bool(
                torch.count_nonzero(
                    out["retrieval_logits"]
                ).item() == 0
                and torch.count_nonzero(
                    out["goal_logits"]
                ).item() == 0
                and torch.count_nonzero(
                    out["planning_mode_logits"]
                ).item() == 0
            ),
    }


@torch.no_grad()
def fresh_donor_self_assay(hub, priority_head, base, cfg):
    normal = []
    donor = []
    off = []
    priority_mae = []

    g = torch.Generator().manual_seed(293101)

    for i in range(256):
        aid = 90000 + 4 * i + GOAL_COMMITMENT_A
        bid = 90000 + 4 * i + GOAL_COMMITMENT_B

        sa = make_state(aid)
        sb = make_state(bid)
        ta = targets_from_state(sa)
        tb = targets_from_state(sb)

        hidden = .40 * torch.randn(
            1,
            int(cfg.nucleus_dim),
            generator=g,
        )
        ws = base.nucleus.growth(
            base.nucleus.workspace(hidden)
        ).detach()

        def run(state, target, enabled=True):
            inp = self_input_from_state(
                state
            ).unsqueeze(0)
            h = hub(
                inp,
                recruitment=1.0,
                enabled=enabled,
            )
            _, p = apply_workspace_broadcast(
                base, ws, h
            )
            value = float(
                priority_head(
                    h["broadcast"]
                )[0].item()
            )
            return {
                "policy":
                    int(p.argmax(dim=-1)[0].item()),
                "retrieval":
                    int(
                        h["retrieval_logits"]
                        .argmax(dim=-1)[0].item()
                    ),
                "goal":
                    int(
                        h["goal_logits"]
                        .argmax(dim=-1)[0].item()
                    ),
                "planning":
                    int(
                        h["planning_mode_logits"]
                        .argmax(dim=-1)[0].item()
                    ),
                "priority": value,
                "priority_target":
                    float(target["value_priority"]),
            }

        ra = run(sa, ta)
        rb = run(sb, tb)

        normal.append(
            float(
                ra["policy"] == GOAL_COMMITMENT_A
                and ra["retrieval"] == 0
                and ra["goal"] == GOAL_COMMITMENT_A
                and ra["planning"] == 0
                and rb["policy"] == GOAL_COMMITMENT_B
                and rb["retrieval"] == 1
                and rb["goal"] == GOAL_COMMITMENT_B
                and rb["planning"] == 0
            )
        )

        priority_mae.extend(
            [
                abs(
                    ra["priority"]
                    - ra["priority_target"]
                ),
                abs(
                    rb["priority"]
                    - rb["priority_target"]
                ),
            ]
        )

        # External workspace is held fixed.  Swapping only the self
        # representation must make all self-dependent categorical consumers
        # follow the donor state.
        donor_a = run(sb, tb)
        donor_b = run(sa, ta)
        donor.append(
            float(
                donor_a["policy"] == GOAL_COMMITMENT_B
                and donor_a["retrieval"] == 1
                and donor_a["goal"] == GOAL_COMMITMENT_B
                and donor_b["policy"] == GOAL_COMMITMENT_A
                and donor_b["retrieval"] == 0
                and donor_b["goal"] == GOAL_COMMITMENT_A
            )
        )

        off_a = run(sa, ta, enabled=False)
        off_b = run(sb, tb, enabled=False)
        off.append(
            float(
                off_a["goal"] == off_b["goal"]
                and off_a["retrieval"] == off_b["retrieval"]
                and off_a["planning"] == off_b["planning"]
                and off_a["priority"] == 0.0
                and off_b["priority"] == 0.0
            )
        )

    return {
        "paired_normal_all_consumers_correct":
            mean(normal),
        "donor_self_all_consumers_follow_donor":
            mean(donor),
        "broadcast_off_erases_a_b_internal_difference":
            mean(off),
        "paired_repaired_priority_mae":
            mean(priority_mae),
    }


@torch.no_grad()
def fresh_autobiographical_history_assay(
    hub,
    priority_head,
    base,
    cfg,
):
    model, _ = load_self_model(
        SELF_MODEL,
        "cpu",
    )

    normal = []
    donor = []
    erase = []
    priority_errors = []

    g = torch.Generator().manual_seed(293202)

    for i in range(128):
        subject = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(
                720000 + i
            ),
        )
        subject.state.unresolved_a = 1.0
        subject.state.unresolved_b = 1.0
        subject.choose_goal()

        completion = subject.record_completion(
            GOAL_COMMITMENT_A
        )
        normal_goal = (
            subject.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )

        donor_core = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(
                730000 + i
            ),
        )
        donor_core.state.unresolved_a = 1.0
        donor_core.state.unresolved_b = 1.0
        donor_core.choose_goal()
        donor_completion = (
            donor_core.record_completion(
                GOAL_COMMITMENT_B
            )
        )

        donor_subject = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        donor_subject.memory.replace_episode(
            completion.episode_id,
            donor_completion,
        )
        donor_goal = (
            donor_subject.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )

        erased = PersistentSelfCore(
            model,
            state=subject.state,
            memory=subject.memory,
        )
        erased.memory.erase_episode(
            completion.episode_id
        )
        erased_goal = (
            erased.reconcile_commitments_from_autobiography(
                seed_a=1.0,
                seed_b=1.0,
            )
        )

        hidden = .40 * torch.randn(
            1,
            int(cfg.nucleus_dim),
            generator=g,
        )
        ws = base.nucleus.growth(
            base.nucleus.workspace(hidden)
        ).detach()

        def route(core):
            inp = self_input_from_state(
                core.state
            ).unsqueeze(0)
            h = hub(
                inp,
                recruitment=1.0,
                enabled=True,
            )
            _, p = apply_workspace_broadcast(
                base, ws, h
            )
            value = float(
                priority_head(
                    h["broadcast"]
                )[0].item()
            )
            target = targets_from_state(
                core.state
            )
            priority_errors.append(
                abs(
                    value
                    - float(
                        target["value_priority"]
                    )
                )
            )
            return {
                "goal":
                    int(
                        h["goal_logits"]
                        .argmax(dim=-1)[0].item()
                    ),
                "retrieval":
                    int(
                        h["retrieval_logits"]
                        .argmax(dim=-1)[0].item()
                    ),
                "policy":
                    int(
                        p.argmax(dim=-1)[0].item()
                    ),
            }

        rn = route(subject)
        rd = route(donor_subject)
        re = route(erased)

        normal.append(
            float(
                normal_goal == GOAL_COMMITMENT_B
                and rn["goal"] == GOAL_COMMITMENT_B
                and rn["retrieval"] == 1
                and rn["policy"] == GOAL_COMMITMENT_B
            )
        )
        donor.append(
            float(
                donor_goal == GOAL_COMMITMENT_A
                and rd["goal"] == GOAL_COMMITMENT_A
                and rd["retrieval"] == 0
                and rd["policy"] == GOAL_COMMITMENT_A
            )
        )
        erase.append(
            float(
                erased_goal == GOAL_COMMITMENT_A
                and re["goal"] == GOAL_COMMITMENT_A
                and re["retrieval"] == 0
                and re["policy"] == GOAL_COMMITMENT_A
            )
        )

    return {
        "normal_history_broadcast_coherence":
            mean(normal),
        "donor_history_broadcast_coherence":
            mean(donor),
        "erased_history_broadcast_coherence":
            mean(erase),
        "history_repaired_priority_mae":
            mean(priority_errors),
    }


@torch.no_grad()
def fresh_existing_episodic_reader_assay(hub):
    reader, _ = load_reader(
        EPISODIC_READER,
        "cpu",
    )
    base2, cfg2, admitted = build_substrate(
        "cpu"
    )

    key0 = collect_decision_snapshot(
        base2,
        cfg2,
        admitted,
        37,
        "left",
        device="cpu",
    )["recall"].clone()
    key1 = collect_decision_snapshot(
        base2,
        cfg2,
        admitted,
        37,
        "right",
        device="cpu",
    )["recall"].clone()

    slots = torch.cat(
        (
            key0.unsqueeze(1),
            key1.unsqueeze(1),
        ),
        dim=1,
    )
    valid = torch.ones(
        slots.shape[0],
        2,
        dtype=torch.bool,
    )

    correct = []

    for i in range(256):
        goal = (
            GOAL_COMMITMENT_A
            if i % 2 == 0
            else GOAL_COMMITMENT_B
        )
        eid = 100000 + 4 * i + goal
        state = make_state(eid)
        out = hub(
            self_input_from_state(
                state
            ).unsqueeze(0),
            recruitment=1.0,
            enabled=True,
        )
        route = int(
            out["retrieval_logits"]
            .argmax(dim=-1)[0].item()
        )
        if route not in (0, 1):
            correct.append(0.0)
            continue

        query = (
            QUERY_FIRST
            if route == 0
            else QUERY_SECOND
        )
        r = reader(
            slots,
            valid,
            torch.tensor(
                [query],
                dtype=torch.long,
            ),
        )
        selected = int(
            r["attention"]
            .argmax(dim=-1)[0].item()
        )
        correct.append(
            float(
                selected
                == (
                    0
                    if goal == GOAL_COMMITMENT_A
                    else 1
                )
            )
        )

    return {
        "self_to_existing_episodic_reader_accuracy":
            mean(correct)
    }


@torch.no_grad()
def persistent_reload_assay(hub, priority_head):
    model, _ = load_self_model(
        SELF_MODEL,
        "cpu",
    )
    first, _ = load_self_bundle(
        SELF_SEED,
        model,
    )
    second, _ = load_self_bundle(
        SELF_SEED,
        model,
    )

    a = hub(
        self_input_from_state(
            first.state
        ).unsqueeze(0),
        recruitment=1.0,
        enabled=True,
    )
    b = hub(
        self_input_from_state(
            second.state
        ).unsqueeze(0),
        recruitment=1.0,
        enabled=True,
    )

    pa = priority_head(a["broadcast"])
    pb = priority_head(b["broadcast"])

    keys = (
        "broadcast",
        "workspace_residual",
        "retrieval_logits",
        "goal_logits",
        "planning_mode_logits",
        "planning_depth",
        "identity_reconstruction",
    )

    return {
        "broadcast_exact_after_persistent_reload":
            float(
                all(
                    torch.equal(
                        a[k], b[k]
                    )
                    for k in keys
                )
            ),
        "repaired_priority_exact_after_persistent_reload":
            float(torch.equal(pa, pb)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_broadcast_centrality_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    source = json.loads(
        SOURCE_RESULT.read_text()
    )
    repair = json.loads(
        REPAIR_RESULT.read_text()
    )
    self_test = json.loads(
        SELF_TEST.read_text()
    )

    if source.get("development_gate_passed") is not False:
        raise RuntimeError(
            "v0.29.0 historical FAIL must be preserved"
        )
    source_failed = sorted(
        k for k, v in source["checks"].items()
        if not bool(v)
    )
    if source_failed != ["value_priority"]:
        raise RuntimeError(
            f"unexpected v0.29.0 failures: {source_failed}"
        )
    if repair.get("repair_dev_gate_passed") is not True:
        raise RuntimeError(
            "v0.29.2 successful repair required"
        )
    if repair.get("broadcast_test_touched") is not False:
        raise RuntimeError(
            "broadcast TEST already touched"
        )
    if self_test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.28.1 frozen self TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_BROADCAST_CENTRALITY_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("source_v0290_formal_status: FAIL")
    print("v0291_repair_formal_status: FAIL")
    print("v0292_repair_dev_status: PASS")
    print("broadcast_test_ids: 30000-32999")
    print("broadcast_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("fresh_donor_example_ids: 90000+")
    print("fresh_history_subject_seeds: 720000-720127")
    print("fresh_history_donor_seeds: 730000-730127")
    print("fresh_existing_reader_condition: 37")

    base, cfg, _ = build_substrate(
        "cpu"
    )
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    hub, _ = load_hub(
        SOURCE_HUB,
        "cpu",
    )
    for p in hub.parameters():
        p.requires_grad_(False)
    hub.eval()

    priority_head, _ = load_priority_head(
        PRIORITY_HEAD,
        "cpu",
    )
    for p in priority_head.parameters():
        p.requires_grad_(False)
    priority_head.eval()

    test_data = build_dataset(
        base,
        cfg,
        TEST_IDS,
        "test",
        seed=293003,
    )

    test_metrics = full_test_metrics(
        hub,
        priority_head,
        base,
        test_data,
    )
    disabled = disabled_metrics(
        hub,
        priority_head,
        base,
        test_data,
    )
    donor = fresh_donor_self_assay(
        hub,
        priority_head,
        base,
        cfg,
    )
    history = fresh_autobiographical_history_assay(
        hub,
        priority_head,
        base,
        cfg,
    )
    episodic = fresh_existing_episodic_reader_assay(
        hub
    )
    continuity = persistent_reload_assay(
        hub,
        priority_head,
    )

    original_test_priority_mae = float(
        test_metrics[
            "original_linear_value_priority_mae"
        ]
    )
    repaired_test_priority_mae = float(
        test_metrics[
            "repaired_value_priority_mae"
        ]
    )
    repaired_dev_priority_mae = float(
        repair[
            "repaired_dev_metrics"
        ]["mae"]
    )

    priority_improvement = (
        original_test_priority_mae
        - repaired_test_priority_mae
    )
    dev_test_shift = abs(
        repaired_test_priority_mae
        - repaired_dev_priority_mae
    )

    checks = {
        "policy_consumer":
            test_metrics["policy_accuracy"]
            >= MIN_POLICY,
        "retrieval_consumer":
            test_metrics["retrieval_accuracy"]
            >= MIN_RETRIEVAL,
        "goal_consumer":
            test_metrics["goal_accuracy"]
            >= MIN_GOAL,
        "planning_consumer":
            test_metrics["planning_mode_accuracy"]
            >= MIN_PLANNING,
        "planning_depth":
            test_metrics["planning_depth_mae"]
            <= MAX_DEPTH_MAE,
        "repaired_value_priority":
            repaired_test_priority_mae
            <= MAX_VALUE_MAE,
        "priority_improves_original_test_head":
            priority_improvement
            >= MIN_PRIORITY_IMPROVEMENT,
        "priority_dev_test_stable":
            dev_test_shift
            <= MAX_TRAIN_DEV_TEST_SHIFT,
        "identity_in_broadcast":
            test_metrics["identity_cosine"]
            >= MIN_IDENTITY_COS,

        "broadcast_off_workspace_zero":
            disabled[
                "workspace_residual_exact_zero"
            ],
        "broadcast_off_workspace_exact_base":
            disabled[
                "workspace_exact_base"
            ],
        "broadcast_off_policy_exact_base":
            disabled[
                "policy_logits_exact_base"
            ],
        "broadcast_off_priority_exact_zero":
            disabled[
                "repaired_priority_exact_zero"
            ],
        "broadcast_off_all_internal_consumers_zero":
            disabled[
                "all_consumer_logits_exact_zero"
            ],
        "broadcast_off_policy_collapses":
            disabled["off_policy_accuracy"]
            <= MAX_OFF_POLICY,
        "broadcast_off_goal_collapses":
            disabled["off_goal_accuracy"]
            <= MAX_OFF_GOAL,
        "broadcast_off_retrieval_collapses":
            disabled["off_retrieval_accuracy"]
            <= MAX_OFF_RETRIEVAL,
        "broadcast_off_investigate_collapses":
            disabled["off_investigate_recall"]
            <= MAX_OFF_INVESTIGATE_RECALL,

        "paired_normal_coherence":
            donor[
                "paired_normal_all_consumers_correct"
            ] >= MIN_DONOR,
        "donor_self_controls_consumers":
            donor[
                "donor_self_all_consumers_follow_donor"
            ] >= MIN_DONOR,
        "off_erases_self_difference":
            donor[
                "broadcast_off_erases_a_b_internal_difference"
            ] >= MIN_DONOR,
        "donor_priority_accurate":
            donor[
                "paired_repaired_priority_mae"
            ] <= MAX_VALUE_MAE,

        "normal_history_controls_broadcast":
            history[
                "normal_history_broadcast_coherence"
            ] >= MIN_HISTORY,
        "donor_history_controls_broadcast":
            history[
                "donor_history_broadcast_coherence"
            ] >= MIN_HISTORY,
        "erased_history_controls_broadcast":
            history[
                "erased_history_broadcast_coherence"
            ] >= MIN_HISTORY,
        "history_priority_accurate":
            history[
                "history_repaired_priority_mae"
            ] <= MAX_VALUE_MAE,

        "self_routes_existing_episodic_reader":
            episodic[
                "self_to_existing_episodic_reader_accuracy"
            ] >= MIN_EXISTING_RETRIEVAL,

        "broadcast_survives_persistent_reload":
            continuity[
                "broadcast_exact_after_persistent_reload"
            ] >= MIN_BROADCAST_CONTINUITY,
        "priority_survives_persistent_reload":
            continuity[
                "repaired_priority_exact_after_persistent_reload"
            ] >= MIN_BROADCAST_CONTINUITY,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during frozen broadcast TEST"
        )

    result = {
        "format":
            "continuous_brain.self_broadcast_centrality_test_v1",
        "test_admission_passed":
            bool(passed),
        "training_performed":
            False,
        "historical_status": {
            "v0290": "FAIL",
            "v0291": "FAIL",
            "v0292_dev_repair": "PASS",
        },
        "split": {
            "test": [30000, 32999],
            "test_used": True,
            "test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "test_metrics":
            test_metrics,
        "broadcast_disabled_metrics":
            disabled,
        "fresh_donor_self_metrics":
            donor,
        "fresh_autobiographical_history_metrics":
            history,
        "fresh_existing_episodic_reader_metrics":
            episodic,
        "persistent_reload_metrics":
            continuity,
        "priority_comparison": {
            "original_linear_test_mae":
                original_test_priority_mae,
            "repaired_test_mae":
                repaired_test_priority_mae,
            "test_improvement":
                priority_improvement,
            "repaired_dev_mae":
                repaired_dev_priority_mae,
            "dev_test_shift":
                dev_test_shift,
        },
        "checks":
            checks,
        "protected_unchanged":
            True,
        "source_hub_unchanged":
            True,
        "priority_head_unchanged":
            True,
        "scope": {
            "functional_self_centrality_claimed":
                True,
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

    print("\nTEST")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print("\nPRIORITY_COMPARISON")
    for k, v in result[
        "priority_comparison"
    ].items():
        print(f"{k}: {v}")

    print("\nBROADCAST_DISABLED")
    for k, v in disabled.items():
        print(f"{k}: {v}")

    print("\nFRESH_DONOR_SELF")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nFRESH_AUTOBIOGRAPHICAL_HISTORY")
    for k, v in history.items():
        print(f"{k}: {v}")

    print("\nFRESH_EXISTING_EPISODIC_READER")
    for k, v in episodic.items():
        print(f"{k}: {v}")

    print("\nPERSISTENT_RELOAD")
    for k, v in continuity.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_HUB_UNCHANGED: true")
    print("PRIORITY_HEAD_UNCHANGED: true")
    print("BROADCAST_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "SELF_BROADCAST_CENTRALITY_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_BROADCAST_CENTRALITY_TEST_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
