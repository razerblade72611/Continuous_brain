#!/usr/bin/env python3
"""
Train-only cue-erased frozen-substrate drift probe for Continuous Brain.

Purpose
-------
Test whether the protected 14k InfantBrain has a length-dependent directional
pull that exists independently of the new writable-memory writer.

This is deliberately NOT the ordinary use_memory=False baseline, because the
base recurrent hidden state can itself carry the visible cue.  Instead, for
every train condition:

1. instantiate exact LEFT/RIGHT counterfactual worlds;
2. teacher-force through the visible-cue interval;
3. at the first cue-hidden state, assert LEFT/RIGHT sensory equality;
4. RESET recurrent hidden to zero;
5. keep external memory exactly zero;
6. evolve only the frozen historical InfantBrain through the remaining
   cue-hidden, cue-independent UP actions;
7. read the frozen motor logits at the delayed decision.

After the reset, LEFT and RIGHT members have identical sensory inputs, action
history, hidden initialization, and zero external memory.  Therefore their
decision logits must be bit-identical.  Any systematic RIGHT-vs-LEFT preference
is a property of the frozen substrate under cue-erased dynamics, not learned
working-memory content.

Uses TRAIN conditions only.  DEV/TEST are never touched.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch_directml

from continuous_brain.evaluate_competitive_infant import (
    _context_tensor,
    _image_tensor,
    _load_base_checkpoint,
)
from continuous_brain.varied_cued_detour_world import (
    ACTION_NAMES,
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
    generate_task_spec,
)
from continuous_brain.working_memory_infant import WorkingMemoryInfant


BASE_CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
EXPECTED_BASE_SHA256 = (
    "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
)
EXPECTED_BASE_DIGEST = (
    "8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e"
)


def state_digest(module):
    h = hashlib.sha256()
    for name in sorted(module.state_dict()):
        t = module.state_dict()[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def rankdata(values):
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[pairs[k][0]] = rank
        i = j
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def build(device):
    base, cfg, _, sha = _load_base_checkpoint(BASE_CHECKPOINT)
    if sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"unexpected base checkpoint SHA256: {sha}"
        )

    digest = state_digest(base)
    if digest != EXPECTED_BASE_DIGEST:
        raise RuntimeError(
            f"unexpected historical base digest: {digest}"
        )

    model = WorkingMemoryInfant.grow_from(base, cfg).to(device)
    model.eval()
    model.base.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg, sha, digest


def assert_pair_sensor_equal(left, right, image_size):
    obs_l = left._obs()
    obs_r = right._obs()
    if not np.array_equal(obs_l, obs_r):
        raise AssertionError(
            f"cue-erased observations differ at condition "
            f"{left.condition_id}, step {left.steps}"
        )

    rgb_l = left.render_rgb(image_size)
    rgb_r = right.render_rgb(image_size)
    if not np.array_equal(rgb_l, rgb_r):
        raise AssertionError(
            f"cue-erased renders differ at condition "
            f"{left.condition_id}, step {left.steps}"
        )


@torch.no_grad()
def run_condition(model, cfg, device, condition_id):
    left = VariedCuedDetourWorld(
        condition_id,
        cue="left",
        max_episode_steps=160,
    )
    right = VariedCuedDetourWorld(
        condition_id,
        cue="right",
        max_episode_steps=160,
    )

    obs_l = left.reset()
    obs_r = right.reset()

    prev_action_l = torch.zeros(
        1, dtype=torch.long, device=device
    )
    prev_action_r = torch.zeros(
        1, dtype=torch.long, device=device
    )

    # Advance through the cue-visible interval using only the environment.
    # No recurrent state from this interval is retained.
    while left.steps < left.cue_steps:
        a_l = int(left.teacher_action())
        a_r = int(right.teacher_action())
        if a_l != a_r:
            raise AssertionError(
                "counterfactual predecision actions differ "
                "during cue-visible interval"
            )
        if a_l in (LEFT, RIGHT):
            raise AssertionError(
                "decision occurred before cue disappearance"
            )

        obs_l, _, done_l, _ = left.step(a_l)
        obs_r, _, done_r, _ = right.step(a_r)
        if done_l or done_r:
            raise AssertionError(
                "episode ended during cue-visible interval"
            )

        prev_action_l = torch.tensor(
            [a_l + 1],
            dtype=torch.long,
            device=device,
        )
        prev_action_r = torch.tensor(
            [a_r + 1],
            dtype=torch.long,
            device=device,
        )

    if left.steps != right.steps:
        raise AssertionError("pair step mismatch at cue erasure")

    # This is the causal erasure: all cue-bearing recurrent history is removed.
    hidden_l = model.initial_state(1, device)
    hidden_r = model.initial_state(1, device)
    memory_l = model.initial_memory(1, device)
    memory_r = model.initial_memory(1, device)

    if not torch.equal(hidden_l.detach().cpu(), hidden_r.detach().cpu()):
        raise AssertionError("hidden reset mismatch")
    if not torch.equal(memory_l.detach().cpu(), memory_r.detach().cpu()):
        raise AssertionError("memory reset mismatch")

    assert_pair_sensor_equal(left, right, cfg.image_size)

    hidden_transitions = 0

    while not left.done:
        a_l = int(left.teacher_action())
        a_r = int(right.teacher_action())

        if a_l in (LEFT, RIGHT) or a_r in (LEFT, RIGHT):
            if not (a_l == LEFT and a_r == RIGHT):
                raise AssertionError(
                    f"unexpected decision actions "
                    f"{ACTION_NAMES[a_l]} / {ACTION_NAMES[a_r]}"
                )

            # Current cue-erased sensory inputs must still be identical.
            assert_pair_sensor_equal(left, right, cfg.image_size)

            image_l = _image_tensor(left, cfg.image_size, device)
            image_r = _image_tensor(right, cfg.image_size, device)
            context_l, _ = _context_tensor(left._obs(), device, cfg)
            context_r, _ = _context_tensor(right._obs(), device, cfg)

            if not torch.equal(
                image_l.detach().cpu(), image_r.detach().cpu()
            ):
                raise AssertionError("decision images differ")
            if not torch.equal(
                context_l.detach().cpu(), context_r.detach().cpu()
            ):
                raise AssertionError("decision contexts differ")
            if not torch.equal(
                prev_action_l.detach().cpu(),
                prev_action_r.detach().cpu(),
            ):
                raise AssertionError("decision previous actions differ")
            if not torch.equal(
                hidden_l.detach().cpu(), hidden_r.detach().cpu()
            ):
                raise AssertionError(
                    "cue-erased recurrent states diverged"
                )

            out_l = model.base_step(
                image_l,
                context_l,
                torch.zeros_like(memory_l),
                prev_action_l,
                hidden_l,
            )
            out_r = model.base_step(
                image_r,
                context_r,
                torch.zeros_like(memory_r),
                prev_action_r,
                hidden_r,
            )

            logits_l = out_l["logits"][0].detach().float().cpu()
            logits_r = out_r["logits"][0].detach().float().cpu()

            if not torch.equal(logits_l, logits_r):
                diff = float((logits_l - logits_r).abs().max().item())
                raise AssertionError(
                    f"cue-erased pair logits differ; max_abs_diff={diff}"
                )

            d = float(
                logits_l[RIGHT].item() - logits_l[LEFT].item()
            )
            full_pred = int(torch.argmax(logits_l).item())
            side_pred = RIGHT if d > 0.0 else LEFT

            spec = generate_task_spec(condition_id)
            declared = tuple(spec.behavioral_signature())
            declared_hidden_delay = int(declared[1])

            if hidden_transitions != declared_hidden_delay:
                raise AssertionError(
                    f"hidden transition mismatch condition={condition_id}: "
                    f"observed={hidden_transitions} "
                    f"declared={declared_hidden_delay}"
                )

            return {
                "condition_id": int(condition_id),
                "hidden_delay_steps": int(hidden_transitions),
                "cue_steps": int(spec.cue_steps),
                "corridor_length": float(spec.corridor_length),
                "move_step": float(spec.move_step),
                "goal_x_offset": float(spec.goal_x_offset),
                "goal_y_offset": float(spec.goal_y_offset),
                "start_x": float(spec.start_x),
                "start_y": float(spec.start_y),
                "distractor_pattern": int(spec.distractor_pattern),
                "right_minus_left_logit": d,
                "side_pred": int(side_pred),
                "side_pred_name": ACTION_NAMES[side_pred],
                "full_pred": int(full_pred),
                "full_pred_name": ACTION_NAMES[full_pred],
                "all_logits": [float(x) for x in logits_l.tolist()],
            }

        if a_l != a_r:
            raise AssertionError(
                "cue-erased predecision teacher actions differ"
            )

        assert_pair_sensor_equal(left, right, cfg.image_size)

        image_l = _image_tensor(left, cfg.image_size, device)
        image_r = _image_tensor(right, cfg.image_size, device)
        context_l, _ = _context_tensor(left._obs(), device, cfg)
        context_r, _ = _context_tensor(right._obs(), device, cfg)

        out_l = model.base_step(
            image_l,
            context_l,
            torch.zeros_like(memory_l),
            prev_action_l,
            hidden_l,
        )
        out_r = model.base_step(
            image_r,
            context_r,
            torch.zeros_like(memory_r),
            prev_action_r,
            hidden_r,
        )

        if not torch.equal(
            out_l["hidden"].detach().cpu(),
            out_r["hidden"].detach().cpu(),
        ):
            diff = float(
                (
                    out_l["hidden"] - out_r["hidden"]
                ).detach().abs().max().cpu()
            )
            raise AssertionError(
                f"cue-erased hidden states diverged; diff={diff}"
            )

        obs_l, _, done_l, _ = left.step(a_l)
        obs_r, _, done_r, _ = right.step(a_r)
        if done_l or done_r:
            raise AssertionError(
                "episode ended before delayed decision"
            )

        prev_action_l = torch.tensor(
            [a_l + 1],
            dtype=torch.long,
            device=device,
        )
        prev_action_r = torch.tensor(
            [a_r + 1],
            dtype=torch.long,
            device=device,
        )
        hidden_l = out_l["hidden"]
        hidden_r = out_r["hidden"]

        # External writable memory is held at exact zero throughout.
        memory_l = torch.zeros_like(memory_l)
        memory_r = torch.zeros_like(memory_r)

        hidden_transitions += 1

        if hidden_transitions > 64:
            raise RuntimeError("hidden interval exceeded safety bound")

    raise RuntimeError("episode ended before decision")


def summarize(rows):
    ds = [float(r["right_minus_left_logit"]) for r in rows]
    right = [float(d > 0.0) for d in ds]
    full_counts = collections.Counter(
        r["full_pred_name"] for r in rows
    )
    return {
        "n": len(rows),
        "right_side_fraction": mean(right),
        "left_side_fraction": 1.0 - mean(right),
        "mean_right_minus_left_logit": mean(ds),
        "median_right_minus_left_logit": median(ds),
        "full_pred_counts": dict(sorted(full_counts.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=(
            "state/working_memory_m1/train_epoch1_v1/"
            "cue_erased_frozen_substrate_probe.json"
        ),
    )
    args = ap.parse_args()

    device = torch_directml.device()
    model, cfg, sha, digest = build(device)

    print("CUE_ERASED_FROZEN_SUBSTRATE_DRIFT_PROBE_V1")
    print("device:", device)
    print("base_checkpoint_sha256:", sha)
    print("base_digest:", digest)
    print(
        "train_condition_range:",
        f"{TRAIN_CONDITION_RANGE.start}:{TRAIN_CONDITION_RANGE.stop}",
    )
    print("external_memory_forced_zero: true")
    print("hidden_reset_at_cue_disappearance: true")
    print("dev_touched: false")
    print("test_touched: false")

    rows = []
    for i, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        rows.append(run_condition(model, cfg, device, int(cid)))
        if i == 1 or i % 100 == 0 or i == len(TRAIN_CONDITION_RANGE):
            print(f"evaluated={i}/{len(TRAIN_CONDITION_RANGE)}")

    by_delay = collections.defaultdict(list)
    for row in rows:
        by_delay[int(row["hidden_delay_steps"])].append(row)

    delay_summary = {
        str(delay): summarize(group)
        for delay, group in sorted(by_delay.items())
    }

    delays = [float(r["hidden_delay_steps"]) for r in rows]
    ds = [float(r["right_minus_left_logit"]) for r in rows]
    right_bits = [float(d > 0.0) for d in ds]

    per_condition_corr = {
        "hidden_delay_vs_right_minus_left_logit_pearson":
            pearson(delays, ds),
        "hidden_delay_vs_right_minus_left_logit_spearman":
            spearman(delays, ds),
        "hidden_delay_vs_right_side_indicator_spearman":
            spearman(delays, right_bits),
    }

    binned_delays = []
    binned_right_fraction = []
    binned_mean_d = []
    for delay, group in sorted(by_delay.items()):
        s = summarize(group)
        binned_delays.append(float(delay))
        binned_right_fraction.append(
            float(s["right_side_fraction"])
        )
        binned_mean_d.append(
            float(s["mean_right_minus_left_logit"])
        )

    binned_corr = {
        "delay_bin_vs_right_fraction_spearman":
            spearman(binned_delays, binned_right_fraction),
        "delay_bin_vs_mean_right_minus_left_logit_spearman":
            spearman(binned_delays, binned_mean_d),
    }

    print()
    print("=" * 94)
    print("CUE-ERASED FROZEN SUBSTRATE BY HIDDEN DELAY")
    print("=" * 94)
    print(
        f"{'delay':>6} {'n':>5} "
        f"{'RIGHT%':>9} {'LEFT%':>9} "
        f"{'mean_d':>11} {'median_d':>11} "
        f"{'full_pred_counts'}"
    )
    for delay, group in sorted(by_delay.items()):
        s = summarize(group)
        print(
            f"{delay:6d} {s['n']:5d} "
            f"{s['right_side_fraction']:9.4f} "
            f"{s['left_side_fraction']:9.4f} "
            f"{s['mean_right_minus_left_logit']:+11.5f} "
            f"{s['median_right_minus_left_logit']:+11.5f} "
            f"{s['full_pred_counts']}"
        )

    print()
    print("PER_CONDITION_CORRELATIONS")
    for k, v in per_condition_corr.items():
        print(f"{k}: {v:+.6f}")

    print()
    print("BINNED_CORRELATIONS")
    for k, v in binned_corr.items():
        print(f"{k}: {v:+.6f}")

    result = {
        "format": "cue_erased_frozen_substrate_drift_probe_v1",
        "base_checkpoint_sha256": sha,
        "base_digest": digest,
        "train_condition_range": [
            TRAIN_CONDITION_RANGE.start,
            TRAIN_CONDITION_RANGE.stop,
        ],
        "external_memory_forced_zero": True,
        "hidden_reset_at_cue_disappearance": True,
        "dev_touched": False,
        "test_touched": False,
        "summary": summarize(rows),
        "by_hidden_delay": delay_summary,
        "per_condition_correlations": per_condition_corr,
        "binned_correlations": binned_corr,
        "per_condition": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    print("wrote:", out)
    print("TRAIN_ONLY: true")
    print("DEV_TOUCHED: false")
    print("TEST_TOUCHED: false")
    print("CUE_ERASED_FROZEN_SUBSTRATE_DRIFT_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
