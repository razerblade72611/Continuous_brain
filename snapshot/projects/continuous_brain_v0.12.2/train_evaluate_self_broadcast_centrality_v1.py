#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    EndogenousGoalArbiter,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    sha256,
)
from continuous_brain.self_broadcast_centrality_v1 import (
    SelfBroadcastHub,
    apply_workspace_broadcast,
    load_hub,
    save_hub,
    self_input_from_state,
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

TRAIN_IDS = range(0, 12000)
DEV_IDS = range(20000, 23000)
TEST_IDS = range(30000, 33000)

SEED = 29001
EPOCHS = 80
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5

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
}

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


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def make_state(example_id: int):
    rng = random.Random(0x290000 + 100003 * int(example_id))
    state = PersistentSelfState.create(
        identity_seed=2000000 + int(example_id)
    )

    goal_class = int(example_id) % 4

    state.confidence = .20 + .70 * rng.random()
    state.agency_ema = .10 + .80 * rng.random()

    if goal_class == GOAL_NONE:
        state.unresolved_a = 0.0
        state.unresolved_b = 0.0
        state.uncertainty = .05 + .35 * rng.random()
        state.prediction_error_ema = .05 + .25 * rng.random()

    elif goal_class == GOAL_COMMITMENT_A:
        state.unresolved_a = .70 + .30 * rng.random()
        state.unresolved_b = .00 + .25 * rng.random()
        state.uncertainty = .05 + .40 * rng.random()
        state.prediction_error_ema = .05 + .25 * rng.random()

    elif goal_class == GOAL_COMMITMENT_B:
        state.unresolved_a = .00 + .25 * rng.random()
        state.unresolved_b = .70 + .30 * rng.random()
        state.uncertainty = .05 + .40 * rng.random()
        state.prediction_error_ema = .05 + .25 * rng.random()

    else:
        state.unresolved_a = .15 + .70 * rng.random()
        state.unresolved_b = .15 + .70 * rng.random()
        state.uncertainty = .83 + .16 * rng.random()
        state.prediction_error_ema = .60 + .39 * rng.random()

    arbiter = EndogenousGoalArbiter()
    state.active_goal = int(arbiter.choose(state))
    if state.active_goal != goal_class:
        raise AssertionError(
            f"state construction mismatch id={example_id}: "
            f"{state.active_goal} != {goal_class}"
        )

    return state


def targets_from_state(state):
    goal = int(state.active_goal)

    if goal == GOAL_COMMITMENT_A:
        retrieval = 0
    elif goal == GOAL_COMMITMENT_B:
        retrieval = 1
    else:
        retrieval = 2

    planning_mode = (
        1 if goal == GOAL_INVESTIGATE else 0
    )

    planning_depth = max(
        0.0,
        min(
            1.0,
            .10
            + .40 * float(state.uncertainty)
            + .35 * float(state.prediction_error_ema)
            + .15 * (1.0 - float(state.confidence)),
        ),
    )

    commitment_priority = max(
        float(state.unresolved_a),
        float(state.unresolved_b),
    )
    value_priority = max(
        0.0,
        min(
            1.0,
            max(
                commitment_priority,
                float(state.prediction_error_ema),
                .75 * float(state.uncertainty),
            ),
        ),
    )

    # Policy channels are an integration probe through the frozen historical
    # motor head, not physical semantic labels.
    policy_target = goal

    return {
        "goal": goal,
        "retrieval": retrieval,
        "planning_mode": planning_mode,
        "planning_depth": planning_depth,
        "value_priority": value_priority,
        "policy": policy_target,
    }


def build_reference_workspaces(base, cfg, ids, seed):
    ids = list(ids)
    g = torch.Generator().manual_seed(int(seed))
    chunks = []
    remaining = len(ids)

    while remaining > 0:
        n = min(1024, remaining)
        hidden = .40 * torch.randn(
            n,
            int(cfg.nucleus_dim),
            generator=g,
        )
        with torch.no_grad():
            ws = base.nucleus.growth(
                base.nucleus.workspace(hidden)
            )
        chunks.append(ws.detach().cpu().float())
        remaining -= n

    return torch.cat(chunks, dim=0)


def build_dataset(base, cfg, ids, label, seed):
    ids = list(ids)
    states = [make_state(i) for i in ids]
    targets = [targets_from_state(s) for s in states]
    workspace = build_reference_workspaces(
        base, cfg, ids, seed
    )

    self_input = torch.stack(
        [self_input_from_state(s) for s in states]
    ).float()
    identity = torch.stack(
        [s.identity_embedding for s in states]
    ).float()

    data = {
        "self_input": self_input,
        "workspace": workspace,
        "identity": identity,
        "policy": torch.tensor(
            [t["policy"] for t in targets],
            dtype=torch.long,
        ),
        "retrieval": torch.tensor(
            [t["retrieval"] for t in targets],
            dtype=torch.long,
        ),
        "goal": torch.tensor(
            [t["goal"] for t in targets],
            dtype=torch.long,
        ),
        "planning_mode": torch.tensor(
            [t["planning_mode"] for t in targets],
            dtype=torch.long,
        ),
        "planning_depth": torch.tensor(
            [t["planning_depth"] for t in targets],
            dtype=torch.float32,
        ),
        "value_priority": torch.tensor(
            [t["value_priority"] for t in targets],
            dtype=torch.float32,
        ),
    }

    print(
        f"{label}_broadcast_examples={len(ids)} "
        f"workspace_dim={workspace.shape[-1]}"
    )
    return data


@torch.no_grad()
def metrics(hub, base, data):
    hub.eval()
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
        "value_priority_mae":
            float(
                (
                    out["value_priority"]
                    - data["value_priority"]
                ).abs().mean().item()
            ),
        "identity_cosine":
            float(identity_cos.mean().item()),
    }


def train_hub(hub, base, train):
    torch.manual_seed(SEED)
    for p in base.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        hub.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(SEED)
    n = train["self_input"].shape[0]
    last = {}

    for epoch in range(1, EPOCHS + 1):
        hub.train()
        order = torch.randperm(n, generator=gen)
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            out = hub(
                train["self_input"][idx],
                recruitment=1.0,
                enabled=True,
            )
            _, policy_logits = apply_workspace_broadcast(
                base,
                train["workspace"][idx],
                out,
            )

            loss_policy = F.cross_entropy(
                policy_logits,
                train["policy"][idx],
            )
            loss_retrieval = F.cross_entropy(
                out["retrieval_logits"],
                train["retrieval"][idx],
            )
            loss_goal = F.cross_entropy(
                out["goal_logits"],
                train["goal"][idx],
            )
            loss_planning = F.cross_entropy(
                out["planning_mode_logits"],
                train["planning_mode"][idx],
            )
            loss_depth = F.mse_loss(
                out["planning_depth"],
                train["planning_depth"][idx],
            )
            loss_value = F.mse_loss(
                out["value_priority"],
                train["value_priority"][idx],
            )
            identity_cos = F.cosine_similarity(
                out["identity_reconstruction"],
                train["identity"][idx],
                dim=-1,
            )
            loss_identity = (
                1.0 - identity_cos
            ).mean()

            loss = (
                1.00 * loss_policy
                + 1.00 * loss_retrieval
                + 1.00 * loss_goal
                + .75 * loss_planning
                + .50 * loss_depth
                + .35 * loss_value
                + .25 * loss_identity
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(idx)
            seen += len(idx)

        last = metrics(hub, base, train)
        last["epoch"] = epoch
        last["loss"] = total / max(1, seen)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"self_broadcast_epoch={epoch:03d}/{EPOCHS} "
                f"loss={last['loss']:.6f} "
                f"policy={last['policy_accuracy']:.4f} "
                f"retrieval={last['retrieval_accuracy']:.4f} "
                f"goal={last['goal_accuracy']:.4f} "
                f"plan={last['planning_mode_accuracy']:.4f} "
                f"depth_mae={last['planning_depth_mae']:.4f} "
                f"id_cos={last['identity_cosine']:.4f}"
            )

        if (
            epoch >= 15
            and last["policy_accuracy"] >= .995
            and last["retrieval_accuracy"] >= .995
            and last["goal_accuracy"] >= .995
            and last["planning_mode_accuracy"] >= .995
            and last["planning_depth_mae"] <= .025
            and last["value_priority_mae"] <= .025
            and last["identity_cosine"] >= .985
        ):
            print(
                f"self_broadcast_early_stop_epoch={epoch}"
            )
            break

    hub.eval()
    return last


@torch.no_grad()
def disabled_metrics(hub, base, data):
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

    investigate = data["planning_mode"] == 1
    planning_pred = out["planning_mode_logits"].argmax(dim=-1)

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
                torch.count_nonzero(out["retrieval_logits"]).item() == 0
                and torch.count_nonzero(out["goal_logits"]).item() == 0
                and torch.count_nonzero(
                    out["planning_mode_logits"]
                ).item() == 0
            ),
    }


@torch.no_grad()
def paired_donor_assay(hub, base, cfg):
    normal = []
    donor = []
    off = []

    g = torch.Generator().manual_seed(291777)

    for i in range(256):
        aid = 40000 + 4 * i + GOAL_COMMITMENT_A
        bid = 40000 + 4 * i + GOAL_COMMITMENT_B
        sa = make_state(aid)
        sb = make_state(bid)

        hidden = .40 * torch.randn(
            1,
            int(cfg.nucleus_dim),
            generator=g,
        )
        ws = base.nucleus.growth(
            base.nucleus.workspace(hidden)
        ).detach()

        def run(state, enabled=True):
            inp = self_input_from_state(state).unsqueeze(0)
            h = hub(
                inp,
                recruitment=1.0,
                enabled=enabled,
            )
            _, p = apply_workspace_broadcast(
                base, ws, h
            )
            return {
                "policy": int(p.argmax(dim=-1)[0].item()),
                "retrieval": int(
                    h["retrieval_logits"]
                    .argmax(dim=-1)[0].item()
                ),
                "goal": int(
                    h["goal_logits"]
                    .argmax(dim=-1)[0].item()
                ),
                "planning": int(
                    h["planning_mode_logits"]
                    .argmax(dim=-1)[0].item()
                ),
            }

        ra = run(sa)
        rb = run(sb)
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

        # Same external workspace. Replacing only the self broadcast with the
        # donor's state must make every self-dependent consumer follow donor.
        donor_a = run(sb)
        donor_b = run(sa)
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

        off_a = run(sa, enabled=False)
        off_b = run(sb, enabled=False)
        off.append(
            float(
                off_a["goal"] == off_b["goal"]
                and off_a["retrieval"] == off_b["retrieval"]
                and off_a["planning"] == off_b["planning"]
            )
        )

    return {
        "paired_normal_all_consumers_correct":
            mean(normal),
        "donor_self_all_consumers_follow_donor":
            mean(donor),
        "broadcast_off_erases_a_b_internal_difference":
            mean(off),
    }


@torch.no_grad()
def autobiographical_history_assay(hub, base, cfg):
    history_normal = []
    history_donor = []
    history_erase = []

    g = torch.Generator().manual_seed(292222)

    # Agency model is not needed here; autobiographical reconstruction is
    # deterministic in the v0.28 persistent-self layer. The existing frozen
    # self model is loaded only to instantiate PersistentSelfCore.
    from continuous_brain.persistent_self_autobiographical_v1 import (
        load_self_model,
    )
    model, _ = load_self_model(SELF_MODEL, "cpu")

    for i in range(128):
        subject = PersistentSelfCore(
            model,
            state=PersistentSelfState.create(500000 + i),
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
            state=PersistentSelfState.create(510000 + i),
        )
        donor_core.state.unresolved_a = 1.0
        donor_core.state.unresolved_b = 1.0
        donor_core.choose_goal()
        donor_completion = donor_core.record_completion(
            GOAL_COMMITMENT_B
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

        def hub_goal(core):
            inp = self_input_from_state(
                core.state
            ).unsqueeze(0)
            out = hub(inp, recruitment=1.0, enabled=True)
            _, policy = apply_workspace_broadcast(
                base, ws, out
            )
            return {
                "goal": int(
                    out["goal_logits"]
                    .argmax(dim=-1)[0].item()
                ),
                "retrieval": int(
                    out["retrieval_logits"]
                    .argmax(dim=-1)[0].item()
                ),
                "policy": int(
                    policy.argmax(dim=-1)[0].item()
                ),
            }

        rn = hub_goal(subject)
        rd = hub_goal(donor_subject)
        re = hub_goal(erased)

        history_normal.append(
            float(
                normal_goal == GOAL_COMMITMENT_B
                and rn["goal"] == GOAL_COMMITMENT_B
                and rn["retrieval"] == 1
                and rn["policy"] == GOAL_COMMITMENT_B
            )
        )
        history_donor.append(
            float(
                donor_goal == GOAL_COMMITMENT_A
                and rd["goal"] == GOAL_COMMITMENT_A
                and rd["retrieval"] == 0
                and rd["policy"] == GOAL_COMMITMENT_A
            )
        )
        history_erase.append(
            float(
                erased_goal == GOAL_COMMITMENT_A
                and re["goal"] == GOAL_COMMITMENT_A
                and re["retrieval"] == 0
                and re["policy"] == GOAL_COMMITMENT_A
            )
        )

    return {
        "normal_history_broadcast_coherence":
            mean(history_normal),
        "donor_history_broadcast_coherence":
            mean(history_donor),
        "erased_history_broadcast_coherence":
            mean(history_erase),
    }


@torch.no_grad()
def existing_episodic_reader_assay(hub):
    reader, _ = load_reader(
        EPISODIC_READER,
        "cpu",
    )
    base, cfg, admitted = build_substrate("cpu")

    key0 = collect_decision_snapshot(
        base,
        cfg,
        admitted,
        29,
        "left",
        device="cpu",
    )["recall"].clone()
    key1 = collect_decision_snapshot(
        base,
        cfg,
        admitted,
        29,
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
        eid = 60000 + 4 * i + goal
        state = make_state(eid)
        inp = self_input_from_state(state).unsqueeze(0)
        out = hub(
            inp,
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
        q = torch.tensor(
            [query],
            dtype=torch.long,
        )
        r = reader(slots, valid, q)
        selected = int(
            r["attention"].argmax(dim=-1)[0].item()
        )
        correct.append(
            float(selected == (0 if goal == GOAL_COMMITMENT_A else 1))
        )

    return {
        "self_to_existing_episodic_reader_accuracy":
            mean(correct)
    }


@torch.no_grad()
def persistent_reload_broadcast_assay(hub):
    from continuous_brain.persistent_self_autobiographical_v1 import (
        load_self_model,
    )
    model, _ = load_self_model(SELF_MODEL, "cpu")
    core, _ = load_self_bundle(
        SELF_SEED,
        model,
    )
    before = hub(
        self_input_from_state(core.state).unsqueeze(0),
        recruitment=1.0,
        enabled=True,
    )

    # Reload independently from disk.
    again, _ = load_self_bundle(
        SELF_SEED,
        model,
    )
    after = hub(
        self_input_from_state(again.state).unsqueeze(0),
        recruitment=1.0,
        enabled=True,
    )

    keys = (
        "broadcast",
        "workspace_residual",
        "retrieval_logits",
        "goal_logits",
        "planning_mode_logits",
        "planning_depth",
        "value_priority",
        "identity_reconstruction",
    )
    return {
        "broadcast_exact_after_persistent_reload":
            float(
                all(
                    torch.equal(
                        before[k],
                        after[k],
                    )
                    for k in keys
                )
            )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/self_broadcast_centrality_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    hub_path = state_dir / "self_broadcast_hub.pt"
    result_path = state_dir / "result.json"

    for name, path in PROTECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    self_test = json.loads(SELF_TEST.read_text())
    if self_test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.28.1 frozen self TEST PASS required"
        )
    if (
        self_test.get("split", {})
        .get("self_test_spent") is not True
    ):
        raise RuntimeError(
            "v0.28.1 self TEST must be sealed/spent"
        )

    protected_before = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }

    print("SELF_BROADCAST_CENTRALITY_V1")
    print("device: cpu")
    print("goal: make_persistent_self_causally_central")
    print("trainable_component: self_broadcast_hub_only")
    print("historical_brain_training: false")
    print("working_memory_training: false")
    print("episodic_reader_training: false")
    print("persistent_self_training: false")
    print("historical_planner_training: false")
    print("broadcast_train_ids: 0-11999")
    print("broadcast_dev_ids: 20000-22999")
    print("broadcast_test_ids: 30000-32999")
    print("broadcast_test_touched: false")

    base, cfg, _ = build_substrate("cpu")
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    # Infer the true historical workspace dimension instead of hardcoding it.
    with torch.no_grad():
        hidden = torch.zeros(
            1,
            int(cfg.nucleus_dim),
        )
        workspace_probe = base.nucleus.growth(
            base.nucleus.workspace(hidden)
        )
        workspace_dim = int(
            workspace_probe.shape[-1]
        )
        action_dim = int(
            base.motor(workspace_probe).shape[-1]
        )

    if action_dim < 4:
        raise RuntimeError(
            f"historical motor head has only {action_dim} channels; need >=4"
        )

    print("historical_workspace_dim:", workspace_dim)
    print("historical_policy_channels:", action_dim)

    train = build_dataset(
        base,
        cfg,
        TRAIN_IDS,
        "train",
        seed=291001,
    )
    dev = build_dataset(
        base,
        cfg,
        DEV_IDS,
        "dev",
        seed=291002,
    )

    torch.manual_seed(SEED)
    hub = SelfBroadcastHub(
        workspace_dim=workspace_dim,
        hidden_dim=96,
        broadcast_dim=64,
    )

    train_metrics = train_hub(
        hub,
        base,
        train,
    )
    save_hub(
        hub_path,
        hub,
        metadata={
            "seed": SEED,
            "train_ids": [0, 11999],
            "dev_ids": [20000, 22999],
            "test_ids": [30000, 32999],
            "test_touched": False,
            "historical_workspace_dim":
                workspace_dim,
            "historical_policy_channels":
                action_dim,
            "self_test_required": True,
        },
    )
    hub, _ = load_hub(
        hub_path,
        "cpu",
    )

    dev_metrics = metrics(
        hub,
        base,
        dev,
    )
    disabled = disabled_metrics(
        hub,
        base,
        dev,
    )
    donor = paired_donor_assay(
        hub,
        base,
        cfg,
    )
    history = autobiographical_history_assay(
        hub,
        base,
        cfg,
    )
    episodic = existing_episodic_reader_assay(
        hub,
    )
    continuity = persistent_reload_broadcast_assay(
        hub,
    )

    checks = {
        "policy_consumer":
            dev_metrics["policy_accuracy"] >= MIN_POLICY,
        "retrieval_consumer":
            dev_metrics["retrieval_accuracy"] >= MIN_RETRIEVAL,
        "goal_consumer":
            dev_metrics["goal_accuracy"] >= MIN_GOAL,
        "planning_consumer":
            dev_metrics["planning_mode_accuracy"] >= MIN_PLANNING,
        "planning_depth":
            dev_metrics["planning_depth_mae"] <= MAX_DEPTH_MAE,
        "value_priority":
            dev_metrics["value_priority_mae"] <= MAX_VALUE_MAE,
        "identity_in_broadcast":
            dev_metrics["identity_cosine"] >= MIN_IDENTITY_COS,

        "broadcast_off_workspace_zero":
            disabled["workspace_residual_exact_zero"],
        "broadcast_off_workspace_exact_base":
            disabled["workspace_exact_base"],
        "broadcast_off_policy_exact_base":
            disabled["policy_logits_exact_base"],
        "broadcast_off_all_internal_consumers_zero":
            disabled["all_consumer_logits_exact_zero"],
        "broadcast_off_policy_collapses":
            disabled["off_policy_accuracy"] <= MAX_OFF_POLICY,
        "broadcast_off_goal_collapses":
            disabled["off_goal_accuracy"] <= MAX_OFF_GOAL,
        "broadcast_off_retrieval_collapses":
            disabled["off_retrieval_accuracy"] <= MAX_OFF_RETRIEVAL,
        "broadcast_off_investigate_collapses":
            disabled["off_investigate_recall"]
            <= MAX_OFF_INVESTIGATE_RECALL,

        "paired_normal_coherence":
            donor["paired_normal_all_consumers_correct"]
            >= MIN_DONOR,
        "donor_self_controls_consumers":
            donor["donor_self_all_consumers_follow_donor"]
            >= MIN_DONOR,
        "off_erases_self_difference":
            donor[
                "broadcast_off_erases_a_b_internal_difference"
            ] >= MIN_DONOR,

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

        "self_routes_existing_episodic_reader":
            episodic[
                "self_to_existing_episodic_reader_accuracy"
            ] >= MIN_EXISTING_RETRIEVAL,
        "broadcast_survives_persistent_reload":
            continuity[
                "broadcast_exact_after_persistent_reload"
            ] >= MIN_BROADCAST_CONTINUITY,
    }

    passed = all(checks.values())

    protected_after = {
        name: sha256(path)
        for name, path in PROTECTED.items()
    }
    if protected_before != protected_after:
        raise RuntimeError(
            "protected artifact changed during v0.29"
        )

    result = {
        "format":
            "continuous_brain.self_broadcast_centrality_v1",
        "development_gate_passed":
            bool(passed),
        "training": {
            "self_broadcast_hub": True,
            "historical_brain": False,
            "working_memory": False,
            "episodic_reader": False,
            "persistent_self": False,
            "historical_planner": False,
        },
        "split": {
            "train": [0, 11999],
            "dev": [20000, 22999],
            "test": [30000, 32999],
            "test_touched": False,
        },
        "architecture": {
            "shared_broadcast_dim": 64,
            "self_input_dim": 42,
            "identity_dims": 32,
            "persistent_state_dims": 10,
            "historical_workspace_dim":
                workspace_dim,
            "consumer_fanout": [
                "historical_workspace_policy",
                "episodic_retrieval_route",
                "goal_arbitration",
                "planning_mode",
                "planning_depth",
                "value_priority",
                "identity_reconstruction",
            ],
            "broadcast_off_exact_neutral":
                True,
        },
        "scope": {
            "claim":
                "persistent self broadcast is shared and causally load-bearing across multiple cognitive consumers while disabled mode preserves the historical policy path exactly",
            "subjective_experience_claimed": False,
            "sentience_claimed": False,
            "consciousness_claimed": False,
        },
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "broadcast_disabled_metrics": disabled,
        "donor_self_metrics": donor,
        "autobiographical_history_metrics":
            history,
        "existing_episodic_reader_metrics":
            episodic,
        "persistent_reload_metrics":
            continuity,
        "checks": checks,
        "protected_unchanged": True,
        "v0281_self_test_preserved": True,
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nTRAIN")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nDEV")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nBROADCAST_DISABLED")
    for k, v in disabled.items():
        print(f"{k}: {v}")

    print("\nDONOR_SELF")
    for k, v in donor.items():
        print(f"{k}: {v}")

    print("\nAUTOBIOGRAPHICAL_HISTORY")
    for k, v in history.items():
        print(f"{k}: {v}")

    print("\nEXISTING_EPISODIC_READER")
    for k, v in episodic.items():
        print(f"{k}: {v}")

    print("\nPERSISTENT_RELOAD")
    for k, v in continuity.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0281_SELF_TEST_PRESERVED: true")
    print("BROADCAST_TEST_TOUCHED: false")
    print(
        "SELF_BROADCAST_CENTRALITY_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "SELF_BROADCAST_CENTRALITY_RUN_COMPLETE"
    )
    print("result:", result_path)
    print("hub:", hub_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
