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
from continuous_brain.self_broadcast_value_priority_repair_v1 import (
    ValuePriorityRepairHead,
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
SELF_TEST = Path(
    "state/persistent_self_autobiographical_test_v1/result.json"
)

SEED = 29101
EPOCHS = 120
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_VALUE_MAE = .05

PROTECTED = {
    "source_result": SOURCE_RESULT,
    "source_hub": SOURCE_HUB,
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

    # Important slices: priority is a max over several internal demands.
    high = y >= .75
    medium = (y >= .45) & (y < .75)
    low = y < .45

    def slice_mae(mask):
        if not bool(mask.any()):
            return 0.0
        return float(
            err[mask].mean().item()
        )

    return {
        "mae": float(err.mean().item()),
        "max_abs_error": float(err.max().item()),
        "high_priority_mae": slice_mae(high),
        "medium_priority_mae": slice_mae(medium),
        "low_priority_mae": slice_mae(low),
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(y.mean().item()),
    }


def train(head, x, y):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)
    n = x.shape[0]
    last = None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(
            n, generator=gen
        )
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            pred = head(x[idx])
            loss = F.smooth_l1_loss(
                pred,
                y[idx],
                beta=.05,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        head.eval()
        last = evaluate(head, x, y)
        last["epoch"] = epoch
        last["loss"] = total / max(1, seen)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"priority_repair_epoch={epoch:03d}/{EPOCHS} "
                f"loss={last['loss']:.6f} "
                f"mae={last['mae']:.5f} "
                f"high={last['high_priority_mae']:.5f} "
                f"medium={last['medium_priority_mae']:.5f} "
                f"low={last['low_priority_mae']:.5f}"
            )

        if epoch >= 20 and last["mae"] <= .020:
            print(
                f"priority_repair_early_stop_epoch={epoch}"
            )
            break

    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_broadcast_value_priority_repair_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    head_path = state_dir / "value_priority_repair.pt"
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

    checks = dict(source.get("checks", {}))
    failed = sorted(
        k for k, v in checks.items()
        if not bool(v)
    )
    if failed != ["value_priority"]:
        raise RuntimeError(
            f"repair is only authorized for value_priority; "
            f"source failures={failed}"
        )

    if (
        source.get("split", {})
        .get("test_touched") is not False
    ):
        raise RuntimeError(
            "broadcast TEST was already touched"
        )

    self_test = json.loads(
        SELF_TEST.read_text()
    )
    if self_test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.28.1 TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_BROADCAST_VALUE_PRIORITY_REPAIR_V1")
    print("device: cpu")
    print("training_scope: nonlinear_value_priority_readout_only")
    print("shared_broadcast_training: false")
    print("source_v0290_formal_status: FAIL")
    print("source_only_failure: value_priority")
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
        hub,
        TRAIN_IDS,
        "train",
    )
    dev_x, dev_y = build_priority_dataset(
        hub,
        DEV_IDS,
        "repair_dev",
    )

    torch.manual_seed(SEED)
    head = ValuePriorityRepairHead(
        broadcast_dim=int(
            hub_payload["broadcast_dim"]
        ),
        hidden_dim=96,
    )

    train_metrics = train(
        head,
        train_x,
        train_y,
    )
    dev_metrics = evaluate(
        head,
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
            "train_ids": [0, 11999],
            "repair_dev_ids": [20000, 22999],
            "test_ids": [30000, 32999],
            "test_touched": False,
            "repair_scope":
                "value_priority_only",
        },
    )

    head, _ = load_priority_head(
        head_path,
        "cpu",
    )

    # Structural neutral-control invariant.
    zero = torch.zeros(
        257,
        int(hub_payload["broadcast_dim"]),
    )
    zero_priority = head(zero)
    zero_exact = bool(
        torch.count_nonzero(
            zero_priority
        ).item() == 0
    )

    # All source passing gates remain preserved by construction because the
    # source hub is frozen and not rewritten.
    source_other_checks_preserved = all(
        bool(v)
        for k, v in checks.items()
        if k != "value_priority"
    )

    repaired_checks = dict(checks)
    repaired_checks["value_priority"] = (
        dev_metrics["mae"]
        <= MAX_VALUE_MAE
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
        and repaired_checks["value_priority"]
        and zero_exact
    )

    result = {
        "format":
            "continuous_brain.self_broadcast_value_priority_repair_v1",
        "repair_dev_gate_passed": passed,
        "source_v0290_formal_status":
            "FAIL",
        "source_failure_preserved":
            "value_priority",
        "repair_scope":
            "nonlinear_value_priority_readout_only",
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
                source["train_metrics"][
                    "value_priority_mae"
                ],
            "dev_value_priority_mae":
                source["dev_metrics"][
                    "value_priority_mae"
                ],
        },
        "train_metrics": train_metrics,
        "repaired_dev_metrics": dev_metrics,
        "zero_broadcast_priority_exact_zero":
            zero_exact,
        "source_other_checks_preserved":
            source_other_checks_preserved,
        "repaired_checks": repaired_checks,
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
    print(
        "train_value_priority_mae:",
        source["train_metrics"][
            "value_priority_mae"
        ],
    )
    print(
        "dev_value_priority_mae:",
        source["dev_metrics"][
            "value_priority_mae"
        ],
    )

    print("\nREPAIR_TRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nREPAIRED_DEV")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nGATE")
    print(
        "source_other_checks_preserved:",
        "PASS" if source_other_checks_preserved else "FAIL",
    )
    print(
        "repaired_value_priority:",
        "PASS"
        if repaired_checks["value_priority"]
        else "FAIL",
    )
    print(
        "zero_broadcast_priority_exact_zero:",
        "PASS" if zero_exact else "FAIL",
    )
    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_HUB_UNCHANGED: true")
    print("BROADCAST_TEST_TOUCHED: false")
    print(
        "SELF_BROADCAST_VALUE_PRIORITY_REPAIR_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_BROADCAST_VALUE_PRIORITY_REPAIR_COMPLETE"
    )
    print("result:", result_path)
    print("priority_head:", head_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
