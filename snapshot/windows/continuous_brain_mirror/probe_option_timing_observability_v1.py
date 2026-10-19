#!/usr/bin/env python3
"""
v0.18 Option Timing Observability Diagnostic v1

No brain training. No v3 DEV/TEST access.

Question:
Can the protected state available at a post-instruction timestep distinguish
"continue lower-level behavior" from "response NOW"?

We compare fresh condition-heldout CPU probes over:
  raw observation/context
  recurrent hidden entering the current step
  recurrent hidden after the current protected historical step
  historical workspace
  frozen base logits
  historical workspace + base logits  (the actual v0.18 gate inputs)
  hidden_out + workspace + base logits

The v0.18 trained gate is also evaluated as a reference.

The classification dataset is balanced at the transition boundary:
for each condition/cue, exactly the final pre-response state and the response
state are used for fitting/evaluation.  The fitted reader is additionally
evaluated across ALL post-instruction pre-response states.

Condition IDs, not individual timesteps, define cross-validation folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch_directml
from torch import nn
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.option_intent_executive_v1 import (
    OptionIntentExecutiveInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    TRAIN_CONDITION_RANGE,
    CanonicalExecutiveRuleWorld,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
DEFAULT_OPTION_CHECKPOINT = (
    "state/option_intent_executive_v1/latest.portable.pt"
)

EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)
EXPECTED_ADMITTED_SHA256 = (
    "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c"
)
EXPECTED_OPTION_FORMAT = "option_intent_executive_training_v1"

FOLDS = 5
SEED = 923771
LINEAR_EPOCHS = 140
MLP_EPOCHS = 180
LR = 1e-2
WD = 1e-4

CANDIDATES = (
    "raw_obs",
    "context",
    "hidden_in",
    "hidden_out",
    "historical_workspace",
    "base_logits",
    "workspace_plus_logits",
    "hidden_out_workspace_logits",
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_model(option_checkpoint, device):
    if sha256_file(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("admitted checkpoint SHA mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, bsha = _load_base_checkpoint(BASE_CHECKPOINT)
    if bsha != EXPECTED_BASE_SHA256:
        raise RuntimeError("historical base SHA mismatch")
    if state_digest(clone_state(base)) != EXPECTED_BASE_DIGEST:
        raise RuntimeError("historical base digest mismatch")

    admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(base, cfg)
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = OptionIntentExecutiveInfant.grow_from_admitted(admitted)

    op = torch.load(
        option_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if op.get("format") != EXPECTED_OPTION_FORMAT:
        raise RuntimeError("option checkpoint format mismatch")
    if int(op.get("global_update", -1)) != 4800:
        raise RuntimeError("expected v0.18 update-4800 checkpoint")
    if (
        op.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("option checkpoint admitted provenance mismatch")

    model.executive.load_state_dict(op["executive"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, op


@torch.no_grad()
def collect_trajectory(model, cfg, device, cid, cue):
    env = CanonicalExecutiveRuleWorld(
        int(cid),
        cue=cue,
        rule="invert_lr",
    )
    obs = env.reset()
    hidden = model.initial_state(1, device)
    memory = model.initial_memory(1, device)
    prev = torch.zeros(1, dtype=torch.long, device=device)
    rows = []
    post_index = 0

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if env.instruction_visible():
            hist = model.admitted.historical_step(
                image,
                context,
                memory,
                prev,
                hidden,
            )
            recall, _ = model.admitted.recall(
                hist["hidden"],
                memory,
            )
            workspace = (
                hist["historical_workspace"]
                + recall
            )
            base_logits = model.base.motor(workspace)
            gate, gate_logit, _ = model.executive.compute_gate(
                hist["historical_workspace"],
                base_logits,
            )

            rows.append({
                "condition_id": int(cid),
                "cue": cue,
                "post_index": int(post_index),
                "response": bool(env.at_response_step()),
                "actual_gate": float(gate.detach().cpu().reshape(-1)[0]),
                "actual_gate_logit": float(
                    gate_logit.detach().cpu().reshape(-1)[0]
                ),
                "raw_obs": torch.tensor(
                    obs,
                    dtype=torch.float32,
                ).reshape(-1),
                "context": context.detach().float().cpu().reshape(-1),
                "hidden_in": hidden.detach().float().cpu().reshape(-1),
                "hidden_out": hist["hidden"].detach().float().cpu().reshape(-1),
                "historical_workspace":
                    hist["historical_workspace"].detach().float().cpu().reshape(-1),
                "base_logits":
                    base_logits.detach().float().cpu().reshape(-1),
            })
            post_index += 1

        if env.at_response_step():
            break

        teacher = int(env.teacher_action())
        out = model.admitted.step(
            image,
            context,
            memory,
            prev,
            hidden,
            use_memory=True,
            use_recall=True,
        )
        obs, _, _, _ = env.step(teacher)
        hidden = out["hidden"]
        memory = out["memory_next"]
        prev = torch.tensor(
            [teacher + 1],
            dtype=torch.long,
            device=device,
        )

    if len(rows) < 2 or not rows[-1]["response"]:
        raise AssertionError(
            f"condition {cid} cue {cue}: missing pre/response states"
        )

    for r in rows:
        r["workspace_plus_logits"] = torch.cat(
            [r["historical_workspace"], r["base_logits"]]
        )
        r["hidden_out_workspace_logits"] = torch.cat(
            [
                r["hidden_out"],
                r["historical_workspace"],
                r["base_logits"],
            ]
        )
    return rows


def collect_all(model, cfg, device):
    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.extend(
            collect_trajectory(model, cfg, device, int(cid), "left")
        )
        rows.extend(
            collect_trajectory(model, cfg, device, int(cid), "right")
        )
        if i == 1 or i % 200 == 0 or i == 1600:
            print(f"collected_conditions={i}/1600 rows={len(rows)}")
    return rows


def boundary_rows(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(
            (r["condition_id"], r["cue"]),
            [],
        ).append(r)

    out = []
    for key, group in grouped.items():
        group = sorted(group, key=lambda x: x["post_index"])
        pre = [x for x in group if not x["response"]]
        resp = [x for x in group if x["response"]]
        if not pre or len(resp) != 1:
            raise AssertionError(f"bad boundary group {key}")
        out.append(pre[-1])
        out.append(resp[0])
    return out


def standardize(train_x, eval_x):
    mu = train_x.mean(0, keepdim=True)
    sd = train_x.std(0, unbiased=False, keepdim=True)
    sd = torch.where(
        sd < 1e-6,
        torch.ones_like(sd),
        sd,
    )
    return (train_x - mu) / sd, (eval_x - mu) / sd, mu, sd


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).reshape(-1)


class MLPProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def fit_probe(kind, train_x, train_y, seed):
    torch.manual_seed(int(seed))
    if kind == "linear":
        model = LinearProbe(train_x.shape[1])
        epochs = LINEAR_EPOCHS
    else:
        model = MLPProbe(train_x.shape[1])
        epochs = MLP_EPOCHS

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_y,
        )
        loss.backward()
        opt.step()

        if epoch >= 40:
            with torch.no_grad():
                pred = (logits >= 0).to(train_y.dtype)
                acc = float((pred == train_y).float().mean())
            if acc >= 0.9995:
                break

    model.eval()
    return model


def make_matrix(rows, field):
    return torch.stack([r[field] for r in rows]).float()


def labels(rows):
    return torch.tensor(
        [1.0 if r["response"] else 0.0 for r in rows],
        dtype=torch.float32,
    )


def condition_fold(cid):
    # Deterministic condition-group split. Both cue members remain in one fold.
    return int(cid) % FOLDS


def cue_pair_gap(rows, probs):
    by_key = {}
    for r, p in zip(rows, probs):
        key = (
            int(r["condition_id"]),
            int(r["post_index"]),
        )
        by_key.setdefault(key, {})[r["cue"]] = float(p)

    gaps = []
    for pair in by_key.values():
        if "left" in pair and "right" in pair:
            gaps.append(abs(pair["left"] - pair["right"]))

    if not gaps:
        return float("nan"), float("nan")
    gaps.sort()
    p99 = gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))]
    return float(sum(gaps) / len(gaps)), float(p99)


def evaluate_candidate(all_rows, boundary, field, kind):
    fold_results = []

    for fold in range(FOLDS):
        tr = [
            r for r in boundary
            if condition_fold(r["condition_id"]) != fold
        ]
        te_boundary = [
            r for r in boundary
            if condition_fold(r["condition_id"]) == fold
        ]
        te_all = [
            r for r in all_rows
            if condition_fold(r["condition_id"]) == fold
        ]

        train_x = make_matrix(tr, field)
        train_y = labels(tr)
        boundary_x = make_matrix(te_boundary, field)
        boundary_y = labels(te_boundary)
        all_x = make_matrix(te_all, field)

        train_x, boundary_x, mu, sd = standardize(
            train_x,
            boundary_x,
        )
        all_x = (all_x - mu) / sd

        model = fit_probe(
            kind,
            train_x,
            train_y,
            seed=SEED + 101 * fold,
        )

        with torch.no_grad():
            bprob = torch.sigmoid(model(boundary_x))
            aprob = torch.sigmoid(model(all_x))

        bpred = bprob >= 0.5
        btarget = boundary_y >= 0.5
        boundary_acc = float(
            (bpred == btarget).float().mean()
        )

        pre_probs = [
            float(p)
            for r, p in zip(te_all, aprob)
            if not r["response"]
        ]
        response_probs = [
            float(p)
            for r, p in zip(te_all, aprob)
            if r["response"]
        ]

        pre_off = (
            sum(p < 0.5 for p in pre_probs) / len(pre_probs)
        )
        response_on = (
            sum(p >= 0.5 for p in response_probs)
            / len(response_probs)
        )
        gap_mean, gap_p99 = cue_pair_gap(
            te_all,
            [float(x) for x in aprob],
        )

        fold_results.append({
            "fold": fold,
            "boundary_accuracy": boundary_acc,
            "pre_off_rate_all": pre_off,
            "response_on_rate": response_on,
            "cue_pair_gap_mean": gap_mean,
            "cue_pair_gap_p99": gap_p99,
            "n_boundary": len(te_boundary),
            "n_all": len(te_all),
        })

    def avg(name):
        return sum(r[name] for r in fold_results) / len(fold_results)

    return {
        "kind": kind,
        "folds": fold_results,
        "mean_boundary_accuracy": avg("boundary_accuracy"),
        "mean_pre_off_rate_all": avg("pre_off_rate_all"),
        "mean_response_on_rate": avg("response_on_rate"),
        "mean_cue_pair_gap": avg("cue_pair_gap_mean"),
        "mean_cue_pair_gap_p99": avg("cue_pair_gap_p99"),
    }


def actual_gate_metrics(rows):
    boundary = boundary_rows(rows)
    b_correct = []
    pre = []
    resp = []

    for r in boundary:
        pred = r["actual_gate"] >= 0.5
        b_correct.append(pred == bool(r["response"]))

    for r in rows:
        if r["response"]:
            resp.append(r["actual_gate"])
        else:
            pre.append(r["actual_gate"])

    gap_mean, gap_p99 = cue_pair_gap(
        rows,
        [r["actual_gate"] for r in rows],
    )

    return {
        "boundary_accuracy": sum(b_correct) / len(b_correct),
        "pre_off_rate_all":
            sum(p < 0.5 for p in pre) / len(pre),
        "response_on_rate":
            sum(p >= 0.5 for p in resp) / len(resp),
        "pre_gate_mean": sum(pre) / len(pre),
        "response_gate_mean": sum(resp) / len(resp),
        "cue_pair_gap_mean": gap_mean,
        "cue_pair_gap_p99": gap_p99,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=DEFAULT_OPTION_CHECKPOINT,
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, checkpoint = load_model(
        Path(args.checkpoint),
        device,
    )
    admitted_ref = clone_state(model.admitted)

    print("OPTION_TIMING_OBSERVABILITY_PROBE_V1")
    print("brain_device:", device)
    print("reader_device: cpu")
    print("checkpoint_global_update:", checkpoint["global_update"])
    print("split: v3_train_only")
    print("condition_group_cv_folds:", FOLDS)
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    rows = collect_all(model, cfg, device)
    boundary = boundary_rows(rows)

    results = {}
    for field in CANDIDATES:
        print(f"probing={field}")
        results[field] = {
            "linear": evaluate_candidate(
                rows,
                boundary,
                field,
                "linear",
            ),
            "mlp": evaluate_candidate(
                rows,
                boundary,
                field,
                "mlp",
            ),
        }

    gate = actual_gate_metrics(rows)

    for k, v in admitted_ref.items():
        got = model.admitted.state_dict()[k].detach().cpu()
        if not torch.equal(v, got):
            raise AssertionError(f"admitted brain changed: {k}")

    payload = {
        "format": "option_timing_observability_probe_v1",
        "checkpoint_global_update": int(checkpoint["global_update"]),
        "actual_v018_gate": gate,
        "candidates": results,
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )

    print()
    print("ACTUAL_V018_GATE")
    print(
        "boundary={:.4f} preOff={:.4f} responseOn={:.4f} "
        "preMean={:.4f} responseMean={:.4f} cueGap={:.4f}".format(
            gate["boundary_accuracy"],
            gate["pre_off_rate_all"],
            gate["response_on_rate"],
            gate["pre_gate_mean"],
            gate["response_gate_mean"],
            gate["cue_pair_gap_mean"],
        )
    )

    print()
    print("PROBE_SUMMARY")
    for field in CANDIDATES:
        for kind in ("linear", "mlp"):
            r = results[field][kind]
            print(
                f"{field}/{kind}: "
                f"boundary={r['mean_boundary_accuracy']:.4f} "
                f"preOff={r['mean_pre_off_rate_all']:.4f} "
                f"responseOn={r['mean_response_on_rate']:.4f} "
                f"cueGap={r['mean_cue_pair_gap']:.4f} "
                f"cueGapP99={r['mean_cue_pair_gap_p99']:.4f}"
            )

    print("admitted_brain_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print("OPTION_TIMING_OBSERVABILITY_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
