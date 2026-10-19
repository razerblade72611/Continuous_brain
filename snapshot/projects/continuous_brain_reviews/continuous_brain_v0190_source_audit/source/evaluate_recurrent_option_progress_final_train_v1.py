#!/usr/bin/env python3
"""Fixed-final TRAIN causal/timing evaluation for v0.19.0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_directml

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
    ACTION_NAMES,
    TRAIN_CONDITION_RANGE,
    RULE_NONE,
    CanonicalExecutiveRuleWorld,
    generate_canonical_rule_spec,
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
EXPECTED_FORMAT = "recurrent_option_progress_training_v1"

RULES = ("invert_lr", "vertical", "vertical_reverse")


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


def mean(xs):
    return sum(xs)/len(xs) if xs else float("nan")


def percentile(xs, q):
    if not xs:
        return float("nan")
    vals = sorted(float(x) for x in xs)
    idx = min(
        len(vals)-1,
        max(0, int(round(q*(len(vals)-1)))),
    )
    return vals[idx]


def load(progress_checkpoint, device):
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
        raise RuntimeError("parent provenance mismatch")

    pp = torch.load(
        progress_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if (
        pp.get("format") != EXPECTED_FORMAT
        or int(pp.get("global_update", -1)) < 1600
        or pp.get("source_option_checkpoint_sha256")
        != parent_sha
    ):
        raise RuntimeError("progress checkpoint mismatch")

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
    model.progress.load_state_dict(pp["progress"], strict=True)
    model.intent_gain.data.copy_(
        pp["intent_gain"].to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, pp, parent_sha


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


@torch.no_grad()
def roll_progress(model, device, seq):
    state = model.initial_option_state(1, device)
    active = torch.tensor(
        [1],
        dtype=torch.long,
        device=device,
    )
    rolled = []
    for snap in seq:
        gate, _, state, _ = model.advance_progress(
            snap["hidden"],
            snap["workspace"],
            snap["base_logits"],
            state,
            active,
        )
        rolled.append((snap, gate, state))
    return rolled


def classify(logits, target):
    pred = int(torch.argmax(logits).item())
    return {
        "pred": pred,
        "pred_name": ACTION_NAMES[pred],
        "correct": bool(pred == target),
    }


@torch.no_grad()
def eval_condition(model, cfg, device, cid):
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

    row = {
        "condition_id": int(cid),
        "members": {},
        "timing": [],
        "cue_gate_gaps": [],
    }

    for i in range(len(rolled["left"])):
        lg = float(
            rolled["left"][i][1]
            .detach().cpu().reshape(-1)[0]
        )
        rg = float(
            rolled["right"][i][1]
            .detach().cpu().reshape(-1)[0]
        )
        row["cue_gate_gaps"].append(abs(lg-rg))

    for cue in ("left", "right"):
        row["members"][cue] = {}

        for snap, gate, _ in rolled[cue]:
            gv = float(
                gate.detach().cpu().reshape(-1)[0]
            )
            record = {
                "cue": cue,
                "response": bool(snap["response"]),
                "gate": gv,
            }
            if not snap["response"]:
                record["preserve"] = {}
                for rid, rule in enumerate(RULES, 1):
                    rule_id = torch.tensor(
                        [rid],
                        dtype=torch.long,
                        device=device,
                    )
                    logits, _, _ = model.apply_intent(
                        snap["recall"],
                        rule_id,
                        snap["base_logits"],
                        gate,
                    )
                    record["preserve"][rule] = bool(
                        int(torch.argmax(logits, dim=-1)[0])
                        == UP
                    )
            row["timing"].append(record)

        response_snap, response_gate, _ = rolled[cue][-1]
        donor_cue = "right" if cue == "left" else "left"
        donor_recall = rolled[donor_cue][-1][0]["recall"]
        zero_recall = torch.zeros_like(response_snap["recall"])

        for rid, rule in enumerate(RULES, 1):
            rule_id = torch.tensor(
                [rid],
                dtype=torch.long,
                device=device,
            )
            target = target_action(cue, rule)
            donor_target = target_action(donor_cue, rule)

            normal, _, _ = model.apply_intent(
                response_snap["recall"],
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            donor, _, _ = model.apply_intent(
                donor_recall,
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            zero, _, _ = model.apply_intent(
                zero_recall,
                rule_id,
                response_snap["base_logits"],
                response_gate,
            )
            none, _, _ = model.apply_intent(
                response_snap["recall"],
                torch.tensor(
                    [RULE_NONE],
                    dtype=torch.long,
                    device=device,
                ),
                response_snap["base_logits"],
                response_gate,
            )

            row["members"][cue][rule] = {
                "normal":
                    classify(normal[0].cpu(), target),
                "donor":
                    classify(donor[0].cpu(), donor_target),
                "zero":
                    classify(zero[0].cpu(), target),
                "off":
                    classify(
                        response_snap["base_logits"][0].cpu(),
                        target,
                    ),
                "none_diff":
                    float(
                        (
                            none
                            - response_snap["base_logits"]
                        )
                        .abs().max().cpu()
                    ),
            }

    spec = generate_canonical_rule_spec(cid)
    row["factors"] = {
        "corridor_length": spec.corridor_length,
        "goal_x_offset": spec.goal_x_offset,
        "goal_y_offset": spec.goal_y_offset,
        "cue_steps": spec.cue_steps,
        "move_step": spec.move_step,
    }
    return row


def pair(row, rule, mode):
    return bool(
        row["members"]["left"][rule][mode]["correct"]
        and row["members"]["right"][rule][mode]["correct"]
    )


def six(row, mode):
    return bool(
        all(
            row["members"][cue][rule][mode]["correct"]
            for cue in ("left", "right")
            for rule in RULES
        )
    )


def summarize(rows):
    out = {}
    for mode in ("normal", "donor", "zero", "off"):
        out[mode] = {
            rule: mean([
                float(pair(r, rule, mode))
                for r in rows
            ])
            for rule in RULES
        }
        out[mode]["six_way"] = mean([
            float(six(r, mode))
            for r in rows
        ])

    out["none_max_diff"] = max(
        r["members"][cue][rule]["none_diff"]
        for r in rows
        for cue in ("left", "right")
        for rule in RULES
    )

    pre = [
        q
        for r in rows
        for q in r["timing"]
        if not q["response"]
    ]
    resp = [
        q
        for r in rows
        for q in r["timing"]
        if q["response"]
    ]

    out["pre_gate_off_rate"] = mean([
        float(abs(q["gate"]) < .5)
        for q in pre
    ])
    out["response_gate_on_rate"] = mean([
        float(q["gate"] >= .5)
        for q in resp
    ])
    out["pre_gate_abs_mean"] = mean([
        abs(q["gate"])
        for q in pre
    ])
    out["response_gate_mean"] = mean([
        q["gate"]
        for q in resp
    ])
    out["pre_response_action_preservation"] = mean([
        float(value)
        for q in pre
        for value in q["preserve"].values()
    ])

    gaps = [
        g
        for r in rows
        for g in r["cue_gate_gaps"]
    ]
    out["cue_gate_gap_mean"] = mean(gaps)
    out["cue_gate_gap_p99"] = percentile(gaps, .99)

    return out


def factor_summary(rows):
    out = {}
    fields = (
        "corridor_length",
        "goal_x_offset",
        "goal_y_offset",
        "cue_steps",
        "move_step",
    )
    for field in fields:
        out[field] = {}
        for value in sorted(
            {r["factors"][field] for r in rows}
        ):
            group = [
                r
                for r in rows
                if r["factors"][field] == value
            ]
            s = summarize(group)
            rec = {
                "n": len(group),
                "six_way_full":
                    s["normal"]["six_way"],
                "pre_response_action_preservation":
                    s["pre_response_action_preservation"],
                "pre_gate_off_rate":
                    s["pre_gate_off_rate"],
                "response_gate_on_rate":
                    s["response_gate_on_rate"],
                "pre_gate_abs_mean":
                    s["pre_gate_abs_mean"],
                "response_gate_mean":
                    s["response_gate_mean"],
                "cue_gate_gap_mean":
                    s["cue_gate_gap_mean"],
            }
            for rule in RULES:
                rec[f"{rule}_pair_full"] = (
                    s["normal"][rule]
                )
                rec[f"{rule}_donor"] = (
                    s["donor"][rule]
                )
            out[field][str(value)] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, payload, parent_sha = load(
        Path(args.checkpoint),
        device,
    )

    admitted_ref = clone(model.admitted)
    base_ref = clone(model.base)
    parent_ref = clone(model.frozen_parent_executive)

    print("RECURRENT_OPTION_PROGRESS_FINAL_TRAIN_V1")
    print("device:", device)
    print(
        "checkpoint_global_update:",
        payload["global_update"],
    )
    print(
        "parent_v018_checkpoint_sha256:",
        parent_sha,
    )
    print(
        "intent_gain:",
        float(model.intent_gain.detach().cpu()),
    )
    print("v3_dev_touched: false")
    print("v3_test_touched: false")

    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.append(
            eval_condition(
                model,
                cfg,
                device,
                int(cid),
            )
        )
        if (
            i == 1
            or i % 100 == 0
            or i == 1600
        ):
            s = summarize(rows)
            print(
                f"evaluated={i}/1600 "
                f"invert={s['normal']['invert_lr']:.4f} "
                f"vertical={s['normal']['vertical']:.4f} "
                f"vreverse={s['normal']['vertical_reverse']:.4f} "
                f"six={s['normal']['six_way']:.4f} "
                f"preKeep={s['pre_response_action_preservation']:.4f} "
                f"gateOff={s['pre_gate_off_rate']:.4f} "
                f"gateOn={s['response_gate_on_rate']:.4f}"
            )

    for k, v in admitted_ref.items():
        if not torch.equal(
            v,
            model.admitted.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"admitted changed {k}")
    for k, v in base_ref.items():
        if not torch.equal(
            v,
            model.base.state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"base changed {k}")
    for k, v in parent_ref.items():
        if not torch.equal(
            v,
            model.frozen_parent_executive
            .state_dict()[k].detach().cpu(),
        ):
            raise AssertionError(f"parent WHAT changed {k}")

    s = summarize(rows)
    result = {
        "format":
            "recurrent_option_progress_final_train_v1",
        "split": "train_only",
        "checkpoint_global_update":
            int(payload["global_update"]),
        "intent_gain":
            float(model.intent_gain.detach().cpu()),
        "summary": s,
        "factor_summary": factor_summary(rows),
        "v3_dev_touched": False,
        "v3_test_touched": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print()
    for rule in RULES:
        print(
            f"{rule}: "
            f"pair={s['normal'][rule]:.4f} "
            f"donor={s['donor'][rule]:.4f} "
            f"off={s['off'][rule]:.4f} "
            f"zero={s['zero'][rule]:.4f}"
        )
    print(
        f"six_way={s['normal']['six_way']:.4f} "
        f"preKeep={s['pre_response_action_preservation']:.4f} "
        f"gateOff={s['pre_gate_off_rate']:.4f} "
        f"gateOn={s['response_gate_on_rate']:.4f} "
        f"preGateAbs={s['pre_gate_abs_mean']:.4f} "
        f"responseGateMean={s['response_gate_mean']:.4f} "
        f"cueGap={s['cue_gate_gap_mean']:.4f} "
        f"cueGapP99={s['cue_gate_gap_p99']:.4f} "
        f"none_diff={s['none_max_diff']:.9g}"
    )
    print("admitted_brain_bit_exact: true")
    print("historical_base_bit_exact: true")
    print("frozen_v018_what_bit_exact: true")
    print("V3_DEV_TOUCHED: false")
    print("V3_TEST_TOUCHED: false")
    print(
        "RECURRENT_OPTION_PROGRESS_FINAL_TRAIN_COMPLETE"
    )


if __name__ == "__main__":
    main()
