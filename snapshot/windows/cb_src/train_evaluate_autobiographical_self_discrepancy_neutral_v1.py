#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    RECORD_FEATURE_DIM,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    NeutralAutobiographicalSelfModel,
    load_neutral_model,
    save_neutral_model,
)
from continuous_brain.persistent_self_autobiographical_v1 import (
    sha256,
)
from continuous_brain.recurrent_self_trajectory_v1 import (
    load_trajectory_model,
)
from train_evaluate_autobiographical_self_discrepancy_v1 import (
    DEV_IDS,
    TRAIN_IDS,
    build_dataset,
    causal_history_assays as source_causal_history_assays,
    persistence_assay as source_persistence_assay,
)

SOURCE_RESULT = Path(
    "state/autobiographical_self_discrepancy_v1/result.json"
)
SOURCE_MODEL = Path(
    "state/autobiographical_self_discrepancy_v1/autobiographical_self_model.pt"
)
TRAJECTORY_MODEL = Path(
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt"
)
TRAJECTORY_TEST = Path(
    "state/recurrent_self_trajectory_test_v1/result.json"
)

SEED = 32101
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-6

MAX_RELIABILITY_MAE = .10
MIN_PREFERENCE_ACCURACY = .95
MIN_NORMAL = .95
MIN_DONOR = .95
MIN_STRUCTURAL_NEUTRAL = 1.0
MIN_DISCREPANCY_ABLATION_NEUTRAL = .95
MIN_RELOAD = 1.0

PROTECTED = {
    "source_result": SOURCE_RESULT,
    "source_model": SOURCE_MODEL,
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
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


@torch.no_grad()
def metrics(model, data):
    out = model(
        data["features"],
        data["mask"],
    )
    reliability = out["reliability"]
    preference = out["preference"]

    return {
        "reliability_mae":
            float(
                (
                    reliability
                    - data["target"]
                ).abs().mean().item()
            ),
        "preference_accuracy":
            float(
                (
                    preference
                    == data["preference"]
                ).float().mean().item()
            ),
        "mean_preference_margin":
            float(
                (
                    reliability.max(dim=-1).values
                    - reliability.min(dim=-1).values
                ).mean().item()
            ),
    }


def train_model(model, train, dev):
    torch.manual_seed(SEED)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)

    n = train["features"].shape[0]
    best = None
    best_score = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(
            n,
            generator=gen,
        )

        total = 0.0
        seen = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):
            idx = order[
                start:start + BATCH_SIZE
            ]
            out = model(
                train["features"][idx],
                train["mask"][idx],
            )
            reliability = out["reliability"]

            loss_reg = F.mse_loss(
                reliability,
                train["target"][idx],
            )

            chosen = reliability.gather(
                1,
                train["preference"][idx]
                .unsqueeze(-1),
            ).squeeze(-1)
            other = reliability.gather(
                1,
                (
                    1
                    - train["preference"][idx]
                ).unsqueeze(-1),
            ).squeeze(-1)

            loss_rank = torch.relu(
                .35 - (chosen - other)
            ).mean()

            loss = (
                loss_reg
                + .20 * loss_rank
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )
            opt.step()

            total += (
                float(loss.item())
                * len(idx)
            )
            seen += len(idx)

        train_m = metrics(model, train)
        dev_m = metrics(model, dev)
        score = (
            dev_m["reliability_mae"]
            + 1.0
            - dev_m["preference_accuracy"]
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"neutral_autobio_epoch={epoch:03d}/{EPOCHS} "
                f"loss={total/max(1,seen):.6f} "
                f"train_mae={train_m['reliability_mae']:.4f} "
                f"dev_mae={dev_m['reliability_mae']:.4f} "
                f"dev_pref={dev_m['preference_accuracy']:.4f}"
            )

        if (
            epoch >= 15
            and dev_m["reliability_mae"] <= .060
            and dev_m["preference_accuracy"] >= .990
        ):
            print(
                f"neutral_autobio_early_stop_epoch={epoch}"
            )
            break

    if best is None:
        raise RuntimeError(
            "no best neutral autobiographical model"
        )

    model.load_state_dict(best)
    model.eval()

    return (
        metrics(model, train),
        metrics(model, dev),
        best_epoch,
    )


@torch.no_grad()
def causal_assays(model, dev):
    normal = []
    donor = []
    erased = []
    zeroed = []
    exact_zeroed = []
    label_only_exact_neutral = []

    for i in range(0, len(dev["ids"]), 2):
        ia = i
        ib = i + 1
        if ib >= len(dev["ids"]):
            break

        mem_a = dev["memories"][ia]
        mem_b = dev["memories"][ib]

        xa, ma = mem_a.feature_tensor(
            HISTORY_LEN
        )
        xb, mb = mem_b.feature_tensor(
            HISTORY_LEN
        )

        oa = model(
            xa.unsqueeze(0),
            ma.unsqueeze(0),
        )
        ob = model(
            xb.unsqueeze(0),
            mb.unsqueeze(0),
        )

        normal.append(
            float(
                int(
                    oa["preference"][0].item()
                )
                == int(
                    dev["preference"][ia].item()
                )
                and int(
                    ob["preference"][0].item()
                )
                == int(
                    dev["preference"][ib].item()
                )
            )
        )

        # Donor autobiography only.
        donor_a = model(
            xb.unsqueeze(0),
            mb.unsqueeze(0),
        )
        donor_b = model(
            xa.unsqueeze(0),
            ma.unsqueeze(0),
        )
        donor.append(
            float(
                int(
                    donor_a["preference"][0].item()
                )
                == int(
                    dev["preference"][ib].item()
                )
                and int(
                    donor_b["preference"][0].item()
                )
                == int(
                    dev["preference"][ia].item()
                )
            )
        )

        # Empty history is exact neutral.
        zx = torch.zeros(
            1,
            HISTORY_LEN,
            RECORD_FEATURE_DIM,
        )
        zm = torch.zeros(
            1,
            HISTORY_LEN,
            dtype=torch.bool,
        )
        ze = model(zx, zm)
        erased.append(
            float(
                torch.equal(
                    ze["reliability"][0],
                    torch.tensor([.5, .5]),
                )
            )
        )

        # Expected==actual: all discrepancy fields are zero, but primary labels,
        # chronological order, mask, and history length remain intact.
        ablated = (
            mem_a.discrepancy_ablation()
        )
        ax, am = ablated.feature_tensor(
            HISTORY_LEN
        )
        az = model(
            ax.unsqueeze(0),
            am.unsqueeze(0),
        )
        zeroed.append(
            float(
                abs(
                    float(
                        az["reliability"][0, 0].item()
                    )
                    - .5
                ) <= 1e-7
                and abs(
                    float(
                        az["reliability"][0, 1].item()
                    )
                    - .5
                ) <= 1e-7
            )
        )
        exact_zeroed.append(
            float(
                torch.equal(
                    az["reliability"][0],
                    torch.tensor([.5, .5]),
                )
            )
        )

        # Stronger structural probe: retain ONLY A/B label/order/mask.
        label_only = ax.clone()
        label_only[:, 2:] = 0.0
        lo = model(
            label_only.unsqueeze(0),
            am.unsqueeze(0),
        )
        label_only_exact_neutral.append(
            float(
                torch.equal(
                    lo["reliability"][0],
                    torch.tensor([.5, .5]),
                )
            )
        )

    return {
        "normal_history_forms_correct_commitment":
            mean(normal),
        "donor_history_flips_commitment":
            mean(donor),
        "erased_history_exact_neutral":
            mean(erased),
        "discrepancy_ablation_exact_neutral":
            mean(zeroed),
        "discrepancy_ablation_bit_exact_neutral":
            mean(exact_zeroed),
        "primary_labels_alone_exact_neutral":
            mean(label_only_exact_neutral),
    }


@torch.no_grad()
def zero_gradient_sanity(model):
    """
    Exact-zero evidence should be neutral but still have an alive local
    derivative, so neutrality is not achieved by dead clamping.
    """
    x = torch.zeros(
        1,
        HISTORY_LEN,
        RECORD_FEATURE_DIM,
        requires_grad=True,
    )
    mask = torch.ones(
        1,
        HISTORY_LEN,
        dtype=torch.bool,
    )

    # Populate only alternating primary labels.
    with torch.no_grad():
        for t in range(HISTORY_LEN):
            x[0, t, 0 if t % 2 == 0 else 1] = 1.0

    out = model(x, mask)
    neutral = torch.equal(
        out["reliability"][0],
        torch.tensor([.5, .5]),
    )

    # Differentiate wrt discrepancy channels only.
    diff = (
        out["reliability"][0, 0]
        - out["reliability"][0, 1]
    )
    diff.backward()

    grad = x.grad[..., 2:]
    alive = bool(
        grad is not None
        and torch.isfinite(grad).all()
        and float(
            grad.abs().max().item()
        ) > 0.0
    )

    return {
        "zero_discrepancy_exact_neutral":
            bool(neutral),
        "zero_discrepancy_gradient_alive":
            bool(alive),
    }


@torch.no_grad()
def reload_assay(model, model_path, dev):
    reloaded, _ = load_neutral_model(
        model_path,
        "cpu",
    )

    x = dev["features"][:32]
    m = dev["mask"][:32]

    a = model(x, m)
    b = reloaded(x, m)

    return {
        "reliability_exact_after_model_reload":
            float(
                torch.equal(
                    a["reliability"],
                    b["reliability"],
                )
            ),
        "preference_exact_after_model_reload":
            float(
                torch.equal(
                    a["preference"],
                    b["preference"],
                )
            ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/autobiographical_self_discrepancy_neutral_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_path = (
        state_dir
        / "neutral_autobiographical_self_model.pt"
    )
    result_path = (
        state_dir
        / "result.json"
    )

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
            "expected v0.32.0 formal DEV FAIL"
        )

    failed = sorted(
        k
        for k, v
        in source["checks"].items()
        if not bool(v)
    )
    if failed != [
        "expected_actual_discrepancy_load_bearing"
    ]:
        raise RuntimeError(
            f"unexpected v0.32.0 failures: {failed}"
        )

    if (
        source.get("split", {})
        .get("test_touched")
        is not False
    ):
        raise RuntimeError(
            "v0.32 TEST already touched"
        )

    trajectory_test = json.loads(
        TRAJECTORY_TEST.read_text()
    )
    if trajectory_test.get(
        "test_admission_passed"
    ) is not True:
        raise RuntimeError(
            "v0.31.1 TEST PASS required"
        )

    before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_V1")
    print("device: cpu")
    print("source_v0320_formal_status: FAIL")
    print("source_only_failure: expected_actual_discrepancy_load_bearing")
    print("repair_scope: structural_neutrality_of_autobiographical_evidence")
    print("source_model_training: false")
    print("trajectory_model_training: false")
    print("development_window_reused: subjects 3000-3255")
    print("new_development_window_consumed: false")
    print("autobio_test_subjects: 6000-6255")
    print("autobio_test_touched: false")

    trajectory_model, _ = (
        load_trajectory_model(
            TRAJECTORY_MODEL,
            "cpu",
        )
    )
    for p in trajectory_model.parameters():
        p.requires_grad_(False)
    trajectory_model.eval()

    train = build_dataset(
        trajectory_model,
        TRAIN_IDS,
        "train",
    )
    dev = build_dataset(
        trajectory_model,
        DEV_IDS,
        "repair_dev",
    )

    torch.manual_seed(SEED)
    model = NeutralAutobiographicalSelfModel(
        record_dim=64,
        hidden_dim=96,
    )

    train_m, dev_m, best_epoch = train_model(
        model,
        train,
        dev,
    )

    save_neutral_model(
        model_path,
        model,
        metadata={
            "source_result_sha256":
                sha256(SOURCE_RESULT),
            "source_model_sha256":
                sha256(SOURCE_MODEL),
            "trajectory_model_sha256":
                sha256(TRAJECTORY_MODEL),
            "train_subject_ids":
                [0, 1023],
            "reused_dev_subject_ids":
                [3000, 3255],
            "test_subject_ids":
                [6000, 6255],
            "test_touched":
                False,
            "best_epoch":
                best_epoch,
            "structural_neutrality":
                True,
        },
    )

    model, _ = load_neutral_model(
        model_path,
        "cpu",
    )

    causal = causal_assays(
        model,
        dev,
    )
    neutral = zero_gradient_sanity(
        model
    )
    reload = reload_assay(
        model,
        model_path,
        dev,
    )

    checks = {
        "reliability_prediction":
            dev_m["reliability_mae"]
            <= MAX_RELIABILITY_MAE,
        "commitment_formation":
            dev_m["preference_accuracy"]
            >= MIN_PREFERENCE_ACCURACY,
        "normal_history_causal":
            causal[
                "normal_history_forms_correct_commitment"
            ] >= MIN_NORMAL,
        "donor_history_causal":
            causal[
                "donor_history_flips_commitment"
            ] >= MIN_DONOR,
        "empty_history_exact_neutral":
            causal[
                "erased_history_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "discrepancy_ablation_exact_neutral":
            causal[
                "discrepancy_ablation_exact_neutral"
            ] >= MIN_DISCREPANCY_ABLATION_NEUTRAL,
        "discrepancy_ablation_bit_exact_neutral":
            causal[
                "discrepancy_ablation_bit_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "primary_labels_alone_exact_neutral":
            causal[
                "primary_labels_alone_exact_neutral"
            ] >= MIN_STRUCTURAL_NEUTRAL,
        "zero_discrepancy_structural_neutral":
            neutral[
                "zero_discrepancy_exact_neutral"
            ],
        "zero_discrepancy_gradient_alive":
            neutral[
                "zero_discrepancy_gradient_alive"
            ],
        "model_reload_reliability_exact":
            reload[
                "reliability_exact_after_model_reload"
            ] >= MIN_RELOAD,
        "model_reload_preference_exact":
            reload[
                "preference_exact_after_model_reload"
            ] >= MIN_RELOAD,
    }

    passed = all(checks.values())

    after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if before != after:
        raise RuntimeError(
            "protected/source artifact changed during v0.32.1"
        )

    result = {
        "format":
            "continuous_brain.autobiographical_self_discrepancy_neutral_v1",
        "repair_dev_gate_passed":
            bool(passed),
        "source_v0320_formal_status":
            "FAIL",
        "source_only_failure":
            "expected_actual_discrepancy_load_bearing",
        "diagnosis":
            "source GRU could convert A/B label/order pattern into non-neutral out-of-distribution default when discrepancy was zero",
        "repair": {
            "primary_labels_are_routing_only":
                True,
            "all_evidence_layers_bias_free":
                True,
            "zero_discrepancy_maps_exactly_to_half_half":
                True,
            "source_model_training":
                False,
            "trajectory_model_training":
                False,
        },
        "split": {
            "train": [0, 1023],
            "reused_dev": [3000, 3255],
            "test": [6000, 6255],
            "test_touched": False,
        },
        "train_metrics":
            train_m,
        "repaired_dev_metrics":
            dev_m,
        "causal_metrics":
            causal,
        "neutrality_metrics":
            neutral,
        "reload_metrics":
            reload,
        "checks":
            checks,
        "protected_unchanged":
            True,
        "source_model_unchanged":
            True,
        "autobio_test_touched":
            False,
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTRAIN")
    for k, v in train_m.items():
        print(f"{k}: {v}")

    print("\nREPAIRED_DEV")
    for k, v in dev_m.items():
        print(f"{k}: {v}")

    print("\nCAUSAL")
    for k, v in causal.items():
        print(f"{k}: {v}")

    print("\nSTRUCTURAL_NEUTRALITY")
    for k, v in neutral.items():
        print(f"{k}: {v}")

    print("\nRELOAD")
    for k, v in reload.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("SOURCE_MODEL_UNCHANGED: true")
    print("PROTECTED_UNCHANGED: true")
    print("AUTOBIO_TEST_TOUCHED: false")
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_COMPLETE"
    )
    print("result:", result_path)
    print("model:", model_path)

    raise SystemExit(
        0 if passed else 4
    )


if __name__ == "__main__":
    main()
