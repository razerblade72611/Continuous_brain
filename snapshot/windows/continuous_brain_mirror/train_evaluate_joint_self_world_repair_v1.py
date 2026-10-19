#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_WORK_A,
    ACTION_WORK_B,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    sha256,
)
from continuous_brain.joint_self_world_prediction_v1 import (
    load_joint_model,
)
from continuous_brain.joint_self_world_repair_v1 import (
    FutureGoalRepairHead,
    corrected_choose_action,
    load_goal_head,
    save_goal_head,
)
from train_evaluate_joint_self_world_prediction_v1 import (
    DEV_IDS,
    TRAIN_IDS,
    make_dataset,
    make_state,
    surprise_self_error_assay,
    reload_action_assay,
)

SOURCE_RESULT = Path(
    "state/joint_self_world_prediction_v1/result.json"
)
SOURCE_MODEL = Path(
    "state/joint_self_world_prediction_v1/joint_self_world_model.pt"
)
V029_TEST = Path(
    "state/self_broadcast_centrality_test_v1/result.json"
)

SEED = 30101
EPOCHS = 80
BATCH_SIZE = 512
LR = 8e-4
WEIGHT_DECAY = 1e-6

MIN_GOAL_ACC = .98
MIN_GOAL_IMPROVEMENT = .002
MAX_GOAL_TRAIN_DEV_GAP = .010

MIN_COUNTERFACTUAL_SELF = .95
MAX_TRUE_SELF_BLIND = .35
MIN_DONOR_FLIP = .95

PROTECTED = {
    "source_result": SOURCE_RESULT,
    "source_model": SOURCE_MODEL,
    "v029_test": V029_TEST,
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
    "self_test":
        Path("state/persistent_self_autobiographical_test_v1/result.json"),
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
}


@torch.no_grad()
def frozen_features(model, data, label):
    chunks = []
    targets = []
    bs = 1024

    n = data["world"].shape[0]
    for start in range(0, n, bs):
        end = min(n, start + bs)
        out = model(
            data["world"][start:end],
            data["self"][start:end],
            data["action"][start:end],
        )
        x = torch.cat(
            (
                out["latent"],
                out["next_self_continuous"],
                out["next_goal_logits"],
            ),
            dim=-1,
        )
        chunks.append(x.detach().cpu())
        targets.append(
            data["next_goal"][start:end]
            .detach().cpu()
        )

    x = torch.cat(chunks, dim=0)
    y = torch.cat(targets, dim=0)
    print(
        f"{label}_goal_repair_examples={len(y)} "
        f"feature_dim={x.shape[-1]}"
    )
    return x.float(), y.long()


class TensorGoalRepairHead(FutureGoalRepairHead):
    def forward_tensor(self, x):
        return self.net(x)


@torch.no_grad()
def goal_metrics(head, x, y):
    logits = head.net(x)
    pred = logits.argmax(dim=-1)
    return {
        "accuracy":
            float(
                (pred == y).float().mean().item()
            ),
        "nll":
            float(
                F.cross_entropy(
                    logits, y
                ).item()
            ),
    }


def train_goal_head(
    head,
    train_x,
    train_y,
    dev_x,
    dev_y,
):
    torch.manual_seed(SEED)

    counts = torch.bincount(
        train_y,
        minlength=4,
    ).float()
    weights = counts.sum() / (
        4.0 * counts.clamp_min(1.0)
    )

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)

    n = train_x.shape[0]
    best = None
    best_acc = -1.0
    best_nll = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(
            n, generator=gen
        )

        for start in range(
            0, n, BATCH_SIZE
        ):
            idx = order[
                start:start + BATCH_SIZE
            ]
            logits = head.net(
                train_x[idx]
            )
            loss = F.cross_entropy(
                logits,
                train_y[idx],
                weight=weights,
                label_smoothing=.002,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        head.eval()
        train_m = goal_metrics(
            head, train_x, train_y
        )
        dev_m = goal_metrics(
            head, dev_x, dev_y
        )

        better = (
            dev_m["accuracy"] > best_acc
            or (
                dev_m["accuracy"] == best_acc
                and dev_m["nll"] < best_nll
            )
        )
        if better:
            best_acc = dev_m["accuracy"]
            best_nll = dev_m["nll"]
            best_epoch = epoch
            best = {
                k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"goal_repair_epoch={epoch:03d}/{EPOCHS} "
                f"train_acc={train_m['accuracy']:.4f} "
                f"dev_acc={dev_m['accuracy']:.4f} "
                f"dev_nll={dev_m['nll']:.5f}"
            )

        if (
            epoch >= 10
            and dev_m["accuracy"] >= .995
        ):
            print(
                f"goal_repair_early_stop_epoch={epoch}"
            )
            break

    if best is None:
        raise RuntimeError(
            "no best goal head"
        )
    head.load_state_dict(best)
    head.eval()

    return (
        goal_metrics(
            head, train_x, train_y
        ),
        goal_metrics(
            head, dev_x, dev_y
        ),
        best_epoch,
    )


@torch.no_grad()
def corrected_counterfactual_assay(model):
    full = []
    blind = []
    donor = []

    for i in range(256):
        # Saturated external progress makes A vs B externally equivalent.
        world = torch.tensor(
            [1.0, 1.0, .55, .35],
            dtype=torch.float32,
        )

        for goal in (
            GOAL_COMMITMENT_A,
            GOAL_COMMITMENT_B,
        ):
            eid = (
                110000
                + 8 * i
                + goal
            )
            state = make_state(eid)

            if goal == GOAL_COMMITMENT_A:
                state.unresolved_a = .95
                state.unresolved_b = .05
                expected = ACTION_WORK_A
            else:
                state.unresolved_a = .05
                state.unresolved_b = .95
                expected = ACTION_WORK_B

            state.uncertainty = .20
            state.prediction_error_ema = .10
            state.active_goal = goal

            r = corrected_choose_action(
                model,
                world,
                state.vector(),
                include_future_self=True,
                self_weight=2.25,
            )
            full.append(
                float(
                    r["action"] == expected
                )
            )

            # TRUE self-blind control:
            # no active-goal-dependent world term and no future-self term.
            z = corrected_choose_action(
                model,
                world,
                state.vector(),
                include_future_self=False,
            )
            blind.append(
                float(
                    z["action"] == expected
                )
            )

        # Same external world. Swap only the self.
        a = make_state(
            115000
            + 8 * i
            + GOAL_COMMITMENT_A
        )
        b = make_state(
            115000
            + 8 * i
            + GOAL_COMMITMENT_B
        )

        a.unresolved_a = .95
        a.unresolved_b = .05
        a.uncertainty = .20
        a.prediction_error_ema = .10
        a.active_goal = GOAL_COMMITMENT_A

        b.unresolved_a = .05
        b.unresolved_b = .95
        b.uncertainty = .20
        b.prediction_error_ema = .10
        b.active_goal = GOAL_COMMITMENT_B

        ra = corrected_choose_action(
            model,
            world,
            a.vector(),
            include_future_self=True,
            self_weight=2.25,
        )
        rb = corrected_choose_action(
            model,
            world,
            b.vector(),
            include_future_self=True,
            self_weight=2.25,
        )

        donor.append(
            float(
                ra["action"] == ACTION_WORK_A
                and rb["action"] == ACTION_WORK_B
            )
        )

    return {
        "future_self_guided_action_accuracy":
            sum(full) / len(full),
        "true_self_blind_world_only_accuracy":
            sum(blind) / len(blind),
        "donor_self_flips_action_with_world_fixed":
            sum(donor) / len(donor),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/joint_self_world_repair_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    head_path = state_dir / "future_goal_repair_head.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    source = json.loads(
        SOURCE_RESULT.read_text()
    )
    if source.get(
        "development_gate_passed"
    ) is not False:
        raise RuntimeError(
            "expected v0.30.0 DEV FAIL"
        )

    failed = sorted(
        k
        for k, v in source["checks"].items()
        if not bool(v)
    )
    expected = sorted(
        [
            "future_goal_prediction",
            "self_consequence_ablation_collapses",
        ]
    )
    if failed != expected:
        raise RuntimeError(
            f"unexpected source failures: {failed}"
        )

    if (
        source.get("split", {})
        .get("test_touched")
        is not False
    ):
        raise RuntimeError(
            "joint TEST already touched"
        )

    v029 = json.loads(
        V029_TEST.read_text()
    )
    if (
        v029.get("split", {})
        .get("test_spent")
        is not True
    ):
        raise RuntimeError(
            "v0.29 TEST status not sealed"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("JOINT_SELF_WORLD_REPAIR_V1")
    print("device: cpu")
    print("source_v0300_formal_status: FAIL")
    print("repair_scope: goal_decoder_plus_corrected_counterfactual_control")
    print("joint_model_training: false")
    print("world_dynamics_training: false")
    print("self_dynamics_training: false")
    print("development_window_reused: 40000-44999")
    print("new_development_window_consumed: false")
    print("joint_test_ids: 60000-64999")
    print("joint_test_touched: false")

    model, model_payload = load_joint_model(
        SOURCE_MODEL,
        "cpu",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train = make_dataset(
        TRAIN_IDS,
        "train",
    )
    dev = make_dataset(
        DEV_IDS,
        "repair_dev",
    )

    train_x, train_y = frozen_features(
        model, train, "train"
    )
    dev_x, dev_y = frozen_features(
        model, dev, "repair_dev"
    )

    torch.manual_seed(SEED)
    head = TensorGoalRepairHead(
        latent_dim=int(
            model_payload["latent_dim"]
        ),
        hidden_dim=128,
    )

    train_goal, dev_goal, best_epoch = (
        train_goal_head(
            head,
            train_x,
            train_y,
            dev_x,
            dev_y,
        )
    )

    save_goal_head(
        head_path,
        head,
        metadata={
            "source_model_sha256":
                sha256(SOURCE_MODEL),
            "source_result_sha256":
                sha256(SOURCE_RESULT),
            "train_ids": [0, 19999],
            "repair_dev_ids":
                [40000, 44999],
            "test_ids":
                [60000, 64999],
            "test_touched": False,
            "best_epoch": best_epoch,
            "repair_scope":
                "goal_calibration_only",
        },
    )
    head, _ = load_goal_head(
        head_path,
        "cpu",
    )

    counterfactual = (
        corrected_counterfactual_assay(
            model
        )
    )

    # Re-run source passing causal assays with the source model unchanged.
    surprise = surprise_self_error_assay(
        model
    )
    reload = reload_action_assay(
        model,
        state_dir,
    )

    source_goal = float(
        source["dev_metrics"][
            "next_goal_accuracy"
        ]
    )
    goal_improvement = (
        dev_goal["accuracy"]
        - source_goal
    )
    goal_gap = abs(
        train_goal["accuracy"]
        - dev_goal["accuracy"]
    )

    source_other_checks_preserved = all(
        bool(v)
        for k, v in source["checks"].items()
        if k not in (
            "future_goal_prediction",
            "self_consequence_ablation_collapses",
        )
    )

    checks = {
        "source_other_checks_preserved":
            source_other_checks_preserved,
        "repaired_future_goal_prediction":
            dev_goal["accuracy"]
            >= MIN_GOAL_ACC,
        "goal_improves_source":
            goal_improvement
            >= MIN_GOAL_IMPROVEMENT,
        "goal_train_dev_stable":
            goal_gap
            <= MAX_GOAL_TRAIN_DEV_GAP,

        "corrected_future_self_guides_action":
            counterfactual[
                "future_self_guided_action_accuracy"
            ] >= MIN_COUNTERFACTUAL_SELF,
        "true_self_blind_control_collapses":
            counterfactual[
                "true_self_blind_world_only_accuracy"
            ] <= MAX_TRUE_SELF_BLIND,
        "corrected_donor_self_flips_action":
            counterfactual[
                "donor_self_flips_action_with_world_fixed"
            ] >= MIN_DONOR_FLIP,

        "surprise_self_error_preserved":
            source["checks"][
                "surprise_creates_self_prediction_error"
            ]
            and surprise[
                "mean_surprise_self_prediction_error"
            ] >= .10,
        "surprise_investigate_preserved":
            surprise[
                "surprise_recruits_investigate"
            ] >= .95,
        "normal_no_investigate_preserved":
            surprise[
                "normal_transition_investigate_rate"
            ] <= .10,
        "zero_self_error_preserved":
            surprise[
                "self_error_state_zeroed_investigate_rate"
            ] <= .10,
        "reload_choice_preserved":
            reload[
                "action_choice_exact_after_self_reload"
            ] >= 1.0,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected/source artifact changed during v0.30.1 repair"
        )

    result = {
        "format":
            "continuous_brain.joint_self_world_repair_v1",
        "repair_dev_gate_passed":
            bool(passed),
        "source_v0300_formal_status":
            "FAIL",
        "diagnosis": {
            "future_goal":
                "near-threshold decoder miss on otherwise strong frozen joint dynamics",
            "counterfactual_ablation":
                "original world-only control leaked current active_goal into world utility",
        },
        "repair_scope": {
            "future_goal_calibration_head":
                True,
            "counterfactual_assay_logic":
                True,
            "joint_model_training":
                False,
            "world_dynamics_training":
                False,
            "self_dynamics_training":
                False,
        },
        "split": {
            "train": [0, 19999],
            "reused_dev": [40000, 44999],
            "test": [60000, 64999],
            "test_touched": False,
        },
        "source_goal_accuracy":
            source_goal,
        "train_goal_metrics":
            train_goal,
        "repaired_dev_goal_metrics":
            dev_goal,
        "goal_improvement":
            goal_improvement,
        "goal_train_dev_gap":
            goal_gap,
        "corrected_counterfactual_metrics":
            counterfactual,
        "surprise_self_error_metrics":
            surprise,
        "persistent_reload_metrics":
            reload,
        "checks":
            checks,
        "source_model_unchanged":
            True,
        "protected_unchanged":
            True,
        "joint_test_touched":
            False,
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nSOURCE")
    print(
        "source_future_goal_accuracy:",
        source_goal,
    )
    print(
        "source_bad_ablation_accuracy:",
        source[
            "counterfactual_self_metrics"
        ][
            "self_consequence_ablated_accuracy"
        ],
    )

    print("\nGOAL_REPAIR_TRAIN")
    for k, v in train_goal.items():
        print(f"{k}: {v}")

    print("\nGOAL_REPAIR_DEV")
    for k, v in dev_goal.items():
        print(f"{k}: {v}")
    print(
        "goal_improvement:",
        goal_improvement,
    )
    print(
        "goal_train_dev_gap:",
        goal_gap,
    )

    print("\nCORRECTED_COUNTERFACTUAL")
    for k, v in counterfactual.items():
        print(f"{k}: {v}")

    print("\nSURPRISE_SELF_ERROR")
    for k, v in surprise.items():
        print(f"{k}: {v}")

    print("\nPERSISTENT_RELOAD")
    for k, v in reload.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("SOURCE_MODEL_UNCHANGED: true")
    print("PROTECTED_UNCHANGED: true")
    print("JOINT_TEST_TOUCHED: false")
    print(
        "JOINT_SELF_WORLD_REPAIR_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "JOINT_SELF_WORLD_REPAIR_COMPLETE"
    )
    print("result:", result_path)
    print("goal_head:", head_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
