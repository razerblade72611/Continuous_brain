#!/usr/bin/env python3
"""TRAIN-only closed-loop diagnostic; no training or v3 DEV/TEST evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from continuous_brain.action_intent_integrity_v1 import (
    select_device, load_executive, clone, assert_exact, provenance, assert_files_unchanged,
)
from continuous_brain.action_intent_rollout_v1 import run_rollout, summarize_rollouts
from continuous_brain.canonical_executive_rule_world_v3 import CanonicalExecutiveRuleWorld
from continuous_brain.evaluate_competitive_infant import _context_tensor

RULES = ("invert_lr", "vertical", "vertical_reverse")
PROBE_SEED = 170117


class BrainPolicy:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.hidden = model.initial_state(1, device)
        self.memory = model.initial_memory(1, device)

    @torch.no_grad()
    def __call__(self, observation, pixels, visible_rule_id, previous_action):
        image = torch.tensor(np.transpose(pixels.astype(np.float32) / 255.0, (2, 0, 1))[None],
                             dtype=torch.float32, device=self.device)
        context, _ = _context_tensor(observation, self.device, self.cfg)
        previous = torch.tensor([0 if previous_action is None else previous_action + 1],
                                dtype=torch.long, device=self.device)
        rid = torch.tensor([visible_rule_id], dtype=torch.long, device=self.device)
        output = self.model.step(image, context, self.memory, previous, self.hidden, rid)
        if not torch.isfinite(output["logits"]).all().item():
            raise ValueError("nonfinite policy logits")
        self.hidden, self.memory = output["hidden"], output["memory_next"]
        return int(output["logits"].argmax(dim=-1).item())


def condition_ids(count):
    if not 1 <= count <= 1600:
        raise ValueError("count must be 1..1600; only TRAIN may be evaluated")
    order = list(range(1600))
    random.Random(PROBE_SEED).shuffle(order)
    return sorted(order[:count])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("directml", "cpu"), default="directml")
    parser.add_argument("--conditions", type=int, default=64)
    args = parser.parse_args()
    ids = condition_ids(args.conditions)
    device = select_device(args.device)
    model, cfg, payload = load_executive(args.checkpoint, device)
    refs = (clone(model.admitted), clone(model.base), clone(model.executive))
    identity = provenance(args.checkpoint, payload, device)
    rows = []
    print("ACTION_INTENT_CONTROL_EVALUATION_V1")
    print("No optimizer steps. v3 TRAIN only. Episode cap: 160 environment steps.")
    print("A sampled diagnostic cannot authorize promotion.")
    for index, cid in enumerate(ids, 1):
        row = {"condition_id": cid, "members": {}}
        for cue in ("left", "right"):
            row["members"][cue] = {}
            for rule in RULES:
                member = {}
                for mode in ("teacher_until_instruction", "autonomous"):
                    env = CanonicalExecutiveRuleWorld(cid, cue=cue, rule=rule)
                    policy = BrainPolicy(model, cfg, device)
                    member[mode] = run_rollout(env, policy, mode=mode, image_size=cfg.image_size)
                row["members"][cue][rule] = member
        rows.append(row)
        if index == 1 or index % 16 == 0 or index == len(ids):
            scores = summarize_rollouts(rows)
            print(f"evaluated={index}/{len(ids)} prefix_six={scores['teacher_until_instruction']['six_way']:.4f} "
                  f"autonomous_six={scores['autonomous']['six_way']:.4f}", flush=True)
    for reference, module, label in zip(refs, (model.admitted, model.base, model.executive), ("admitted", "base", "executive")):
        assert_exact(reference, module, label)
    assert_files_unchanged(identity, args.checkpoint)
    scores = summarize_rollouts(rows)
    full = ids == list(range(1600))
    contract = full and all(scores[mode]["six_way"] >= .90 and
                            all(scores[mode][rule] >= .95 for rule in RULES)
                            for mode in ("teacher_until_instruction", "autonomous"))
    report = {
        "format": "action_intent_control_v1", "split": "train_only", "condition_ids": ids,
        "condition_count": len(ids), "probe_seed": PROBE_SEED, "summary": scores,
        "per_condition": rows, "provenance": identity, "full_train_coverage": full,
        "control_contract_passed": bool(contract), "mapping_gate_checked": False,
        "ready_for_fresh_v4": False, "generalization_established": False,
        "preservation": {"admitted_bit_exact": True, "base_bit_exact": True,
                         "executive_bit_exact": True, "checkpoint_files_unchanged": True},
        "v3_dev_touched": False, "v3_test_touched": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(scores, indent=2))
    print("CONTROL_SCOPE:", "full TRAIN" if full else "sampled TRAIN diagnostic")
    print("CONTROL_CONTRACT:", ("PASS" if contract else "FAIL") if full else "NOT_EVALUATED")
    print("ACTION_INTENT_CONTROL_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
