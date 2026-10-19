#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import sha256
from continuous_brain.self_broadcast_centrality_v1 import (
    load_hub,
    self_input_from_state,
)
from continuous_brain.self_broadcast_value_priority_gradient_safe_v1 import (
    GradientSafeValuePriorityHead,
    load_priority_head,
    save_priority_head,
)
from train_evaluate_self_broadcast_centrality_v1 import (
    DEV_IDS,
    TRAIN_IDS,
    make_state,
    targets_from_state,
)

SOURCE_RESULT = Path(
    "state/self_broadcast_centrality_v1/result.json"
)
SOURCE_HUB = Path(
    "state/self_broadcast_centrality_v1/self_broadcast_hub.pt"
)
FAILED_REPAIR_RESULT = Path(
    "state/self_broadcast_value_priority_repair_v1/result.json"
)
SELF_TEST = Path(
    "state/persistent_self_autobiographical_test_v1/result.json"
)

SEED = 29201
EPOCHS = 160
BATCH_SIZE = 512
LR = 7.5e-4
WEIGHT_DECAY = 1e-6

MAX_VALUE_MAE = .05
MIN_IMPROVEMENT_VS_SOURCE = .010
MAX_TRAIN_DEV_GAP = .020

PROTECTED = {
    "source_result": SOURCE_RESULT,
    "source_hub": SOURCE_HUB,
    "failed_repair_result": FAILED_REPAIR_RESULT,
    "self_test": SELF_TEST,
    "base":
        Path("state/child_v113_dml_smoke/infant_training.pt"),
    "working_memory":
        Path(
            "state/working_memory_recall_capacity_control_phase1_v1/"
            "latest.portable.pt"
        ),
    "episodic_reader":
        Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
    "goal_decoder":
        Path("state/memory_guided_goal_option_v1/goal_decoder.pt"),
    "consequence_model":
        Path("state/consequence_replanning_v1/consequence_model.pt"),
    "self_model":
        Path("state/persistent_self_autobiographical_v1/agency_self_model.pt"),
    "self_seed":
        Path("state/persistent_self_autobiographical_v1/persistent_self_seed.pt"),
}


def build_priority_dataset(hub, ids, label):
    xs = []
    ys = []

    hub.eval()
    with torch.no_grad():
        ids = list(ids)
        for n, eid in enumerate(ids, 1):
            state = make_state(int(eid))
            self_input = self_input_from_state(
                state
            ).unsqueeze(0)
            out = hub(
                self_input,
                recruitment=1.0,
                enabled=True,
            )
            xs.append(
                out["broadcast"][0].detach().cpu()
            )
            ys.append(
                float(
                    targets_from_state(state)[
                        "value_priority"
                    ]
                )
            )

            if n == 1 or n % 3000 == 0 or n == len(ids):
                print(
                    f"{label}_priority_examples={n}/{len(ids)}"
                )

    return (
        torch.stack(xs).float(),
        torch.tensor(ys, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(head, x, y):
    pred = head(x)
    err = (pred - y).abs()

    high = y >= .75
    medium = (y >= .45) & (y < .75)
    low = y < .45

    def slice_mae(mask):
        if not bool(mask.any()):
            return 0.0
        return float(err[mask].mean().item())

    return {
        "mae": float(err.mean().item()),
        "max_abs_error": float(err.max().item()),
        "high_priority_mae": slice_mae(high),
        "medium_priority_mae": slice_mae(medium),
        "low_priority_mae": slice_mae(low),
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(y.mean().item()),
        "pred_min": float(pred.min().item()),
        "pred_max": float(pred.max().item()),
    }


def train(head, train_x, train_y, dev_x, dev_y):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)
    n = train_x.shape[0]

    best_dev = float("inf")
    best_state = None
    best_epoch = None
    last_train = None
    last_dev = None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(
            n, generator=gen
        )
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            pred = head(train_x[idx])

            # MSE keeps gradient alive across the full zero-centered sigmoid
            # range and strongly penalizes large priority errors.
            loss = F.mse_loss(
                pred,
                train_y[idx],
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        head.eval()
        train_metrics = evaluate(
            head, train_x, train_y
        )
        dev_metrics = evaluate(
            head, dev_x, dev_y
        )
        train_metrics["epoch"] = epoch
        train_metrics["loss"] = total / max(1, seen)

        if dev_metrics["mae"] < best_dev:
            best_dev = dev_metrics["mae"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()
            }

        last_train = train_metrics
        last_dev = dev_metrics

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"priority_gradient_safe_epoch={epoch:03d}/{EPOCHS} "
                f"loss={train_metrics['loss']:.6f} "
                f"train_mae={train_metrics['mae']:.5f} "
                f"dev_mae={dev_metrics['mae']:.5f} "
                f"dev_high={dev_metrics['high_priority_mae']:.5f} "
                f"dev_medium={dev_metrics['medium_priority_mae']:.5f} "
                f"dev_low={dev_metrics['low_priority_mae']:.5f}"
            )

        if (
            epoch >= 20
            and train_metrics["mae"] <= .025
            and dev_metrics["mae"] <= .025
        ):
            print(
                f"priority_gradient_safe_early_stop_epoch={epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("no best state recorded")
    head.load_state_dict(best_state)
    head.eval()

    return (
        evaluate(head, train_x, train_y),
        evaluate(head, dev_x, dev_y),
        int(best_epoch),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_broadcast_value_priority_gradient_safe_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    head_path = state_dir / "value_priority_gradient_safe.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    source = json.loads(
        SOURCE_RESULT.read_text()
    )
    if source.get("development_gate_passed") is not False:
        raise RuntimeError(
            "expected v0.29.0 formal DEV FAIL"
        )

    failed_repair = json.loads(
        FAILED_REPAIR_RESULT.read_text()
    )
    if (
        failed_repair.get("repair_dev_gate_passed")
        is not False
    ):
        raise RuntimeError(
            "expected v0.29.1 repair FAIL"
        )

    checks = dict(source.get("checks", {}))
    failed = sorted(
        k for k, v in checks.items()
        if not bool(v)
    )
    if failed != ["value_priority"]:
        raise RuntimeError(
            f"v0.29.0 had unexpected failures: {failed}"
        )

    if (
        source.get("split", {})
        .get("test_touched") is not False
    ):
        raise RuntimeError(
            "broadcast TEST already touched"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_V1")
    print("device: cpu")
    print("repair_scope: value_priority_readout_only")
    print("shared_broadcast_training: false")
    print("source_v0290_formal_status: FAIL")
    print("v0291_repair_formal_status: FAIL")
    print("diagnosis: clamp_dead_gradient_repair")
    print("development_window_reused: 20000-22999")
    print("new_development_window_consumed: false")
    print("broadcast_test_ids: 30000-32999")
    print("broadcast_test_touched: false")

    hub, hub_payload = load_hub(
        SOURCE_HUB,
        "cpu",
    )
    for p in hub.parameters():
        p.requires_grad_(False)
    hub.eval()

    train_x, train_y = build_priority_dataset(
        hub, TRAIN_IDS, "train"
    )
    dev_x, dev_y = build_priority_dataset(
        hub, DEV_IDS, "repair_dev"
    )

    torch.manual_seed(SEED)
    head = GradientSafeValuePriorityHead(
        broadcast_dim=int(
            hub_payload["broadcast_dim"]
        ),
        hidden_dim=128,
    )

    train_metrics, dev_metrics, best_epoch = train(
        head,
        train_x,
        train_y,
        dev_x,
        dev_y,
    )

    save_priority_head(
        head_path,
        head,
        metadata={
            "source_hub_sha256":
                sha256(SOURCE_HUB),
            "source_result_sha256":
                sha256(SOURCE_RESULT),
            "failed_repair_result_sha256":
                sha256(FAILED_REPAIR_RESULT),
            "train_ids": [0, 11999],
            "repair_dev_ids": [20000, 22999],
            "test_ids": [30000, 32999],
            "test_touched": False,
            "best_epoch": best_epoch,
            "repair_scope":
                "gradient_safe_value_priority_only",
        },
    )

    head, _ = load_priority_head(
        head_path,
        "cpu",
    )

    # Structural neutral control and gradient-at-zero check.
    zero = torch.zeros(
        257,
        int(hub_payload["broadcast_dim"]),
        requires_grad=True,
    )
    zero_priority = head(zero)
    zero_exact = bool(
        torch.count_nonzero(
            zero_priority
        ).item() == 0
    )
    zero_priority.sum().backward()
    zero_grad_nonzero = bool(
        zero.grad is not None
        and torch.isfinite(zero.grad).all()
        and float(zero.grad.abs().max().item()) > 0.0
    )

    source_train_mae = float(
        source["train_metrics"][
            "value_priority_mae"
        ]
    )
    source_dev_mae = float(
        source["dev_metrics"][
            "value_priority_mae"
        ]
    )

    source_other_checks_preserved = all(
        bool(v)
        for k, v in checks.items()
        if k != "value_priority"
    )

    repaired_pass = (
        dev_metrics["mae"] <= MAX_VALUE_MAE
    )
    improves_train = (
        train_metrics["mae"]
        <= source_train_mae
        - MIN_IMPROVEMENT_VS_SOURCE
    )
    improves_dev = (
        dev_metrics["mae"]
        <= source_dev_mae
        - MIN_IMPROVEMENT_VS_SOURCE
    )
    train_dev_stable = (
        abs(
            dev_metrics["mae"]
            - train_metrics["mae"]
        )
        <= MAX_TRAIN_DEV_GAP
    )

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected/source artifact changed during repair"
        )

    passed = bool(
        source_other_checks_preserved
        and repaired_pass
        and improves_train
        and improves_dev
        and train_dev_stable
        and zero_exact
        and zero_grad_nonzero
    )

    result = {
        "format":
            "continuous_brain.self_broadcast_value_priority_gradient_safe_v1",
        "repair_dev_gate_passed": passed,
        "source_v0290_formal_status": "FAIL",
        "v0291_repair_formal_status": "FAIL",
        "diagnosis":
            "v0.29.1 clamp created dead-gradient regions",
        "repair_scope":
            "gradient_safe_nonlinear_value_priority_readout_only",
        "training": {
            "priority_repair_head": True,
            "shared_self_broadcast_hub": False,
            "historical_brain": False,
            "persistent_self": False,
            "episodic_reader": False,
        },
        "split": {
            "train": [0, 11999],
            "reused_dev": [20000, 22999],
            "test": [30000, 32999],
            "test_touched": False,
        },
        "source_metrics": {
            "train_value_priority_mae":
                source_train_mae,
            "dev_value_priority_mae":
                source_dev_mae,
            "failed_repair_train_mae":
                failed_repair[
                    "train_metrics"
                ]["mae"],
            "failed_repair_dev_mae":
                failed_repair[
                    "repaired_dev_metrics"
                ]["mae"],
        },
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "repaired_dev_metrics": dev_metrics,
        "zero_broadcast_priority_exact_zero":
            zero_exact,
        "zero_broadcast_has_nonzero_input_gradient":
            zero_grad_nonzero,
        "improves_train_over_original_linear":
            improves_train,
        "improves_dev_over_original_linear":
            improves_dev,
        "train_dev_gap_within_limit":
            train_dev_stable,
        "source_other_checks_preserved":
            source_other_checks_preserved,
        "protected_unchanged": True,
        "source_hub_unchanged": True,
        "broadcast_test_touched": False,
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nSOURCE")
    print("original_train_mae:", source_train_mae)
    print("original_dev_mae:", source_dev_mae)
    print(
        "v0291_failed_train_mae:",
        failed_repair["train_metrics"]["mae"],
    )
    print(
        "v0291_failed_dev_mae:",
        failed_repair[
            "repaired_dev_metrics"
        ]["mae"],
    )

    print("\nGRADIENT_SAFE_TRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nGRADIENT_SAFE_DEV")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nGATE")
    gates = {
        "source_other_checks_preserved":
            source_other_checks_preserved,
        "repaired_value_priority":
            repaired_pass,
        "improves_train_over_original":
            improves_train,
        "improves_dev_over_original":
            improves_dev,
        "train_dev_stable":
            train_dev_stable,
        "zero_priority_exact_neutral":
            zero_exact,
        "zero_point_gradient_alive":
            zero_grad_nonzero,
    }
    for k, v in gates.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_HUB_UNCHANGED: true")
    print("BROADCAST_TEST_TOUCHED: false")
    print(
        "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_COMPLETE"
    )
    print("result:", result_path)
    print("priority_head:", head_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
