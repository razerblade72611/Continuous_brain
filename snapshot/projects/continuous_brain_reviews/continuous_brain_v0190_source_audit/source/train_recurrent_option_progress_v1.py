#!/usr/bin/env python3
"""
TRAIN-only v0.19.0 recurrent option-progress training.

Frozen:
- historical 14k brain
- admitted writer/recall
- v0.18 WHAT/action-intent head

Trainable:
- recurrent option-progress controller
- scalar intent_gain

Loss:
  0.45 timing MSE
  0.20 pre-response action preservation
  0.20 own response
  0.10 coherent donor response
  0.05 left/right timing-invariance

No BCEWithLogits is used, avoiding the v0.18 DirectML CPU fallback.
No v3 DEV/TEST access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.working_memory_recall_capacity_control_v1 import (
    CapacityMatchedRecallWorkingMemoryInfant,
)
from continuous_brain.recurrent_option_progress_v1 import (
    RecurrentOptionProgressInfant,
)
from continuous_brain.canonical_executive_rule_world_v3 import (
    TRAIN_CONDITION_RANGE,
    CanonicalExecutiveRuleWorld,
    target_action,
    UP,
)

BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
ADMITTED_CHECKPOINT = (
    "state/working_memory_recall_capacity_control_phase1_v1/"
    "latest.portable.pt"
)
PARENT_CHECKPOINT = (
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
EXPECTED_PARENT_FORMAT = "option_intent_executive_training_v1"

FORMAT = "recurrent_option_progress_training_v1"
RULES = ("invert_lr", "vertical", "vertical_reverse")
EPOCH_SIZE = 1600
SCHEDULE_SEED = 1457911
INIT_SEED = 919192


class DMLSafeAdam:
    def __init__(self, params, lr=2e-4):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.b1 = .9
        self.b2 = .999
        self.eps = 1e-8
        self.n = 0
        self.state = [
            {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            for p in self.params
        ]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.n += 1
        c1 = 1 - self.b1 ** self.n
        c2 = 1 - self.b2 ** self.n
        for p, s in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all().item():
                raise RuntimeError("non-finite gradient")
            s["m"].mul_(self.b1).add_(g, alpha=1-self.b1)
            s["v"].mul_(self.b2).add_(g*g, alpha=1-self.b2)
            den = s["v"].sqrt()/math.sqrt(c2) + self.eps
            p.add_(s["m"]/den, alpha=-self.lr/c1)

    def portable_state(self):
        return {
            "step_count": self.n,
            "lr": self.lr,
            "state": [
                {
                    "exp_avg": s["m"].detach().cpu(),
                    "exp_avg_sq": s["v"].detach().cpu(),
                }
                for s in self.state
            ],
        }

    def load_portable_state(self, payload):
        self.n = int(payload["step_count"])
        if len(payload["state"]) != len(self.state):
            raise RuntimeError("optimizer state mismatch")
        for dst, src in zip(self.state, payload["state"]):
            dst["m"].copy_(
                src["exp_avg"].to(dst["m"].device)
            )
            dst["v"].copy_(
                src["exp_avg_sq"].to(dst["v"].device)
            )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8*1024*1024), b""):
            h.update(c)
    return h.hexdigest()


def clone(m):
    return {
        k: v.detach().cpu().clone()
        for k, v in m.state_dict().items()
    }


def digest(st):
    h = hashlib.sha256()
    for k in sorted(st):
        t = st[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_exact(ref, module, label):
    for k, v in ref.items():
        if not torch.equal(
            v,
            module.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"{label}: changed {k}")


def schedule(epoch):
    ids = list(TRAIN_CONDITION_RANGE)
    random.Random(
        SCHEDULE_SEED + 1_000_003*int(epoch)
    ).shuffle(ids)
    return ids


def load(device):
    parent_sha = sha(PARENT_CHECKPOINT)
    parent = torch.load(
        PARENT_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    if (
        parent.get("format") != EXPECTED_PARENT_FORMAT
        or int(parent.get("global_update", -1)) != 4800
        or parent.get("source_admitted_checkpoint_sha256")
        != EXPECTED_ADMITTED_SHA256
    ):
        raise RuntimeError("v0.18 parent provenance mismatch")

    ap = torch.load(
        ADMITTED_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    base, cfg, _, bsha = _load_base_checkpoint(BASE_CHECKPOINT)
    if (
        sha(ADMITTED_CHECKPOINT) != EXPECTED_ADMITTED_SHA256
        or bsha != EXPECTED_BASE_SHA256
        or digest(clone(base)) != EXPECTED_BASE_DIGEST
    ):
        raise RuntimeError("protected provenance mismatch")

    admitted = (
        CapacityMatchedRecallWorkingMemoryInfant
        .grow_from(base, cfg)
    )
    admitted.memory.load_state_dict(ap["memory"], strict=True)
    admitted.recall.load_state_dict(ap["recall"], strict=True)
    admitted = admitted.to(device)
    admitted.eval()
    for p in admitted.parameters():
        p.requires_grad_(False)

    model = (
        RecurrentOptionProgressInfant
        .grow_from_admitted(admitted)
        .to(device)
    )
    model.load_frozen_parent_executive(parent["executive"])
    model.train(True)
    model.admitted.eval()
    model.base.eval()
    model.frozen_parent_executive.eval()

    return (
        model,
        cfg,
        clone(admitted),
        clone(base),
        clone(model.frozen_parent_executive),
        model.trainable_parameters(),
        parent_sha,
    )


@torch.no_grad()
def protected_sequence(model, cfg, device, cid, cue):
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

    while not env.done:
        image = _image_tensor(env, cfg.image_size, device)
        context, _ = _context_tensor(obs, device, cfg)

        if env.instruction_visible():
            hist, recall, _, _, base_logits = model.components(
                image,
                context,
                memory,
                prev,
                hidden,
            )
            rows.append({
                "hidden": hist["hidden"].detach(),
                "workspace": hist["historical_workspace"].detach(),
                "recall": recall.detach(),
                "base_logits": base_logits.detach(),
                "response": bool(env.at_response_step()),
            })

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
        raise AssertionError("bad protected sequence")
    return rows


def roll_progress(model, device, seq):
    state = model.initial_option_state(1, device)
    active = torch.tensor(
        [1],
        dtype=torch.long,
        device=device,
    )
    rolled = []
    for snap in seq:
        gate, logit, state, _ = model.advance_progress(
            snap["hidden"],
            snap["workspace"],
            snap["base_logits"],
            state,
            active,
        )
        rolled.append((snap, gate, logit, state))
    return rolled


def update_loss(model, cfg, device, cid):
    seqs = {
        cue: protected_sequence(
            model,
            cfg,
            device,
            cid,
            cue,
        )
        for cue in ("left", "right")
    }
    if len(seqs["left"]) != len(seqs["right"]):
        raise AssertionError("paired sequence length mismatch")

    rolled = {
        cue: roll_progress(
            model,
            device,
            seqs[cue],
        )
        for cue in ("left", "right")
    }

    timing = []
    invariant = []
    pre = []
    own = []
    donor = []
    correct = {}
    gate_values = {"pre": [], "response": []}

    for cue in ("left", "right"):
        for snap, gate, _, _ in rolled[cue]:
            target = torch.tensor(
                [[1.0 if snap["response"] else 0.0]],
                device=device,
            )
            timing.append(((gate-target)**2).mean())
            gate_values[
                "response" if snap["response"] else "pre"
            ].append(float(gate.detach().cpu().reshape(-1)[0]))

    for i in range(len(rolled["left"])):
        invariant.append(
            (
                rolled["left"][i][1]
                - rolled["right"][i][1]
            ).pow(2).mean()
        )

    for cue in ("left", "right"):
        donor_cue = "right" if cue == "left" else "left"
        response_snap, response_gate, _, _ = rolled[cue][-1]
        donor_recall = rolled[donor_cue][-1][0]["recall"]

        for rid, rule in enumerate(RULES, 1):
            rule_id = torch.tensor(
                [rid],
                dtype=torch.long,
                device=device,
            )

            for snap, gate, _, _ in rolled[cue][:-1]:
                logits, _, _ = model.apply_intent(
                    snap["recall"],
                    rule_id,
                    snap["base_logits"],
                    gate,
                )
                pre.append(
                    F.cross_entropy(
                        logits,
                        torch.tensor(
                            [UP],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )

            logits, _, _ = model.apply_intent(
                response_snap["recall"],
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            target = target_action(cue, rule)
            own.append(
                F.cross_entropy(
                    logits,
                    torch.tensor(
                        [target],
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )
            correct[(cue, rule)] = (
                int(torch.argmax(logits, dim=-1)[0]) == target
            )

            donor_logits, _, _ = model.apply_intent(
                donor_recall,
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            donor_target = target_action(donor_cue, rule)
            donor.append(
                F.cross_entropy(
                    donor_logits,
                    torch.tensor(
                        [donor_target],
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )

    timing_l = torch.stack(timing).mean()
    pre_l = torch.stack(pre).mean()
    own_l = torch.stack(own).mean()
    donor_l = torch.stack(donor).mean()
    invariant_l = torch.stack(invariant).mean()

    total = (
        .45*timing_l
        + .20*pre_l
        + .20*own_l
        + .10*donor_l
        + .05*invariant_l
    )

    return (
        total,
        timing_l,
        pre_l,
        own_l,
        donor_l,
        invariant_l,
        correct,
        gate_values,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    update,
    parent_sha,
):
    torch.save(
        {
            "format": FORMAT,
            "global_update": int(update),
            "source_option_checkpoint_sha256": parent_sha,
            "source_option_checkpoint_update": 4800,
            "source_admitted_checkpoint_sha256":
                EXPECTED_ADMITTED_SHA256,
            "historical_base_sha256":
                EXPECTED_BASE_SHA256,
            "historical_base_digest":
                EXPECTED_BASE_DIGEST,
            "schedule_seed": SCHEDULE_SEED,
            "init_seed": INIT_SEED,
            "progress": {
                k: v.detach().cpu()
                for k, v in model.progress.state_dict().items()
            },
            "intent_gain":
                model.intent_gain.detach().cpu(),
            "optimizer": optimizer.portable_state(),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/recurrent_option_progress_v1",
    )
    ap.add_argument(
        "--target-update",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--lr",
        type=float,
        required=True,
    )
    args = ap.parse_args()

    if args.target_update % EPOCH_SIZE:
        raise ValueError("target must end on full epoch")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    latest = state_dir / "latest.portable.pt"
    log_path = state_dir / "training.jsonl"

    torch.manual_seed(INIT_SEED)
    device = torch_directml.device()
    (
        model,
        cfg,
        admitted_ref,
        base_ref,
        parent_ref,
        params,
        parent_sha,
    ) = load(device)

    optimizer = DMLSafeAdam(params, args.lr)
    completed = 0

    if latest.exists():
        payload = torch.load(
            latest,
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("format") != FORMAT:
            raise RuntimeError("latest format mismatch")
        if (
            payload.get("source_option_checkpoint_sha256")
            != parent_sha
        ):
            raise RuntimeError("parent checkpoint changed")
        model.progress.load_state_dict(
            payload["progress"],
            strict=True,
        )
        model.intent_gain.data.copy_(
            payload["intent_gain"].to(device)
        )
        optimizer.load_portable_state(
            payload["optimizer"]
        )
        completed = int(payload["global_update"])

    if completed > args.target_update:
        raise RuntimeError("latest exceeds requested target")
    if completed == args.target_update:
        print("already_at_target_update:", completed)
        return

    optimizer.lr = float(args.lr)

    print("RECURRENT_OPTION_PROGRESS_TRAINING_V1")
    print("device:", device)
    print("parent_v018_checkpoint_sha256:", parent_sha)
    print(
        "trainable_parameter_count:",
        sum(p.numel() for p in params),
    )
    print("what_path_frozen: true")
    print("timing_loss: dml_safe_mse")
    print("bptt_across_option_phase: true")
    print("start_global_update:", completed+1)
    print("target_global_update:", args.target_update)
    print("stage_lr:", args.lr)
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    started = time.time()

    with log_path.open(
        "a" if log_path.exists() else "w"
    ) as log:
        for update in range(
            completed+1,
            args.target_update+1,
        ):
            z = update-1
            epoch = z // EPOCH_SIZE
            cid = schedule(epoch)[z % EPOCH_SIZE]

            optimizer.zero_grad()
            (
                loss,
                timing_l,
                pre_l,
                own_l,
                donor_l,
                inv_l,
                correct,
                gates,
            ) = update_loss(
                model,
                cfg,
                device,
                cid,
            )
            loss.backward()
            optimizer.step()

            record = {
                "global_update": update,
                "epoch_index": epoch,
                "condition_id": int(cid),
                "loss": float(loss.detach().cpu()),
                "timing_loss":
                    float(timing_l.detach().cpu()),
                "pre_loss":
                    float(pre_l.detach().cpu()),
                "own_loss":
                    float(own_l.detach().cpu()),
                "donor_loss":
                    float(donor_l.detach().cpu()),
                "invariance_loss":
                    float(inv_l.detach().cpu()),
                "intent_gain":
                    float(model.intent_gain.detach().cpu()),
            }
            log.write(json.dumps(record)+"\n")
            log.flush()

            if (
                update == completed+1
                or update % 25 == 0
                or update == args.target_update
            ):
                pairs = {
                    r: int(
                        correct[("left", r)]
                        and correct[("right", r)]
                    )
                    for r in RULES
                }
                pre_mean = (
                    sum(gates["pre"])/len(gates["pre"])
                )
                resp_mean = (
                    sum(gates["response"])
                    / len(gates["response"])
                )
                print(
                    f"update={update:05d}/{args.target_update:05d} "
                    f"epoch={epoch+1} condition={cid:04d} "
                    f"loss={record['loss']:.6f} "
                    f"time={record['timing_loss']:.6f} "
                    f"pre={record['pre_loss']:.6f} "
                    f"own={record['own_loss']:.6f} "
                    f"donor={record['donor_loss']:.6f} "
                    f"inv={record['invariance_loss']:.6f} "
                    f"preGate={pre_mean:.4f} "
                    f"respGate={resp_mean:.4f} "
                    f"gain={record['intent_gain']:.4f} "
                    f"invert={pairs['invert_lr']} "
                    f"vertical={pairs['vertical']} "
                    f"vreverse={pairs['vertical_reverse']} "
                    f"six={int(all(correct.values()))}"
                )

            if (
                update % 100 == 0
                or update == args.target_update
            ):
                assert_exact(
                    admitted_ref,
                    model.admitted,
                    "admitted",
                )
                assert_exact(
                    base_ref,
                    model.base,
                    "base",
                )
                assert_exact(
                    parent_ref,
                    model.frozen_parent_executive,
                    "frozen v0.18 WHAT",
                )

                ckpt = state_dir / (
                    f"recurrent_option_progress_"
                    f"u{update:06d}.portable.pt"
                )
                save_checkpoint(
                    ckpt,
                    model,
                    optimizer,
                    update,
                    parent_sha,
                )
                latest.write_bytes(
                    ckpt.read_bytes()
                )
                print("saved:", ckpt)

    print(
        "elapsed_seconds:",
        round(time.time()-started, 3),
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("frozen_v018_what_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print(
        "RECURRENT_OPTION_PROGRESS_TRAINING_COMPLETE"
    )


if __name__ == "__main__":
    main()
