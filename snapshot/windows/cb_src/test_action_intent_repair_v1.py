#!/usr/bin/env python3
"""CPU regression tests. Synthetic fixtures are never admission evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from continuous_brain.action_intent_executive_v1 import ActionIntentExecutive, ActionIntentExecutiveInfant
from continuous_brain.action_intent_integrity_v1 import (
    FORMAT, INIT_SEED, SCHEDULE_SEED, EXPECTED_ADMITTED_SHA256,
    EXPECTED_BASE_SHA256, EXPECTED_BASE_DIGEST, BASE_CHECKPOINT,
    atomic_save, assert_exact, clone, digest, sha, validate_payload,
)
from continuous_brain.action_intent_gate_v1 import RULES, FACTORS, decide, summary, factors
from continuous_brain.action_intent_rollout_v1 import run_rollout
from continuous_brain.canonical_executive_rule_world_v3 import CanonicalExecutiveRuleWorld, generate_canonical_rule_spec, target_action
from continuous_brain.evaluate_competitive_infant import _load_base_checkpoint, _context_tensor, _image_tensor
from continuous_brain.working_memory_recall_capacity_control_v1 import CapacityMatchedRecallWorkingMemoryInfant
from train_action_intent_executive_v1 import DMLSafeAdam

torch.set_num_threads(1)


def synthetic_report():
    rows = []
    for cid in range(1600):
        spec = generate_canonical_rule_spec(cid)
        row = {"condition_id": cid, "factors": {f: getattr(spec, f) for f in FACTORS}, "members": {}}
        for cue in ("left", "right"):
            row["members"][cue] = {}
            donor = "right" if cue == "left" else "left"
            for rule in RULES:
                row["members"][cue][rule] = {
                    "normal": {"pred": target_action(cue, rule), "correct": True},
                    "donor": {"pred": target_action(donor, rule), "correct": True},
                    "off": {"pred": 0, "correct": False},
                    "zero": {"pred": 0, "correct": False}, "none_diff": 0.0,
                }
        rows.append(row)
    return {
        "format": "action_intent_executive_final_train_v1_1", "split": "train_only",
        "per_condition": rows, "summary": summary(rows), "factor_summary": factors(rows),
        "v3_dev_touched": False, "v3_test_touched": False,
        "preservation": dict.fromkeys(("admitted_bit_exact", "base_bit_exact", "executive_bit_exact", "checkpoint_files_unchanged"), True),
        "provenance": {"global_update": 1600, "checkpoint_sha256": "1" * 64,
                       "admitted_checkpoint_sha256": EXPECTED_ADMITTED_SHA256,
                       "historical_base_sha256": EXPECTED_BASE_SHA256,
                       "source_sha256": {str(i): "2" * 64 for i in range(7)}},
    }


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = synthetic_report()

    def test_complete_synthetic_evidence_is_accepted_with_limited_scope(self):
        result = decide(self.report)
        self.assertTrue(result["pass"])
        self.assertFalse(result["ready_for_fresh_v4"])

    def test_empty_or_duplicate_or_heldout_conditions_rejected(self):
        for alteration in ("empty", "duplicate", "heldout"):
            with self.subTest(alteration=alteration):
                report = copy.deepcopy(self.report)
                if alteration == "empty": report["per_condition"] = []
                if alteration == "duplicate": report["per_condition"][-1] = report["per_condition"][0]
                if alteration == "heldout": report["per_condition"][-1]["condition_id"] = 1600
                with self.assertRaises(ValueError): decide(report)

    def test_missing_factors_and_unmeasured_preservation_rejected(self):
        for field in ("factor_summary", "preservation", "provenance"):
            report = copy.deepcopy(self.report)
            report[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): decide(report)

    def test_forged_correctness_and_summary_rejected(self):
        report = copy.deepcopy(self.report)
        report["per_condition"][0]["members"]["left"]["vertical"]["normal"]["pred"] = 0
        with self.assertRaises(ValueError): decide(report)
        report = copy.deepcopy(self.report)
        report["summary"]["normal"]["vertical"] = .999
        with self.assertRaises(ValueError): decide(report)

    def test_noop_must_be_exact_and_nonfinite_is_rejected(self):
        for value in (1e-8, float("nan")):
            report = copy.deepcopy(self.report)
            report["per_condition"][0]["members"]["left"]["vertical"]["none_diff"] = value
            report["summary"] = summary(report["per_condition"])
            report["factor_summary"] = factors(report["per_condition"])
            if value == value:
                self.assertFalse(decide(report)["pass"])
            else:
                with self.assertRaises(ValueError): decide(report)


class RolloutTests(unittest.TestCase):
    def test_oracle_fixture_solves_world(self):
        # Deliberately privileged oracle validates the harness, not a model.
        for rule in RULES:
            for mode in ("autonomous", "teacher_until_instruction"):
                env = CanonicalExecutiveRuleWorld(0, cue="left", rule=rule)
                result = run_rollout(env, lambda *_: int(env.teacher_action()), mode=mode, image_size=32)
                self.assertTrue(result["success"])
                if mode == "autonomous": self.assertEqual(result["teacher_actions"], 0)

    def test_response_correct_policy_can_fail_closed_loop(self):
        # This policy picks the correct delayed response too early and stalls.
        # A teacher-prefix final-response test would conceal that failure.
        env = CanonicalExecutiveRuleWorld(0, cue="left", rule="vertical_reverse", max_episode_steps=30)
        def premature_policy(obs, pixels, rule_id, previous):
            return 1 if rule_id == 0 else 2
        response_target = target_action("left", "vertical_reverse")
        self.assertEqual(premature_policy(None, None, 3, None), response_target)
        result = run_rollout(env, premature_policy, image_size=32)
        self.assertFalse(result["success"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["steps"], 30)

    def test_previous_action_is_executed_action_and_inputs_exclude_labels(self):
        env = CanonicalExecutiveRuleWorld(0, cue="left", rule="invert_lr", max_episode_steps=12)
        seen = []
        def idle_policy(obs, pixels, rule_id, previous):
            seen.append(previous)
            self.assertEqual(obs.shape, (6,))
            self.assertEqual(pixels.shape, (32, 32, 3))
            return 0
        result = run_rollout(env, idle_policy, mode="teacher_until_instruction", image_size=32)
        self.assertIsNone(seen[0])
        self.assertEqual(seen[1], 1)  # Teacher executed UP, although policy said IDLE.
        self.assertGreater(result["teacher_actions"], 0)
        self.assertEqual(seen[-1], 0)


class NetworkAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(92)
        self.net = ActionIntentExecutive()
        self.opt = DMLSafeAdam(self.net.parameters())
        self.content = torch.randn(12, 128)
        self.rules = torch.tensor([1, 2, 3] * 4)
        self.base = torch.randn(12, 5)
        self.targets = torch.tensor([1, 3, 4] * 4)

    def step(self, net=None, opt=None):
        net = self.net if net is None else net
        opt = self.opt if opt is None else opt
        opt.zero_grad()
        delta, _ = net(self.content, self.rules, self.base)
        torch.nn.functional.cross_entropy(self.base + delta, self.targets).backward()
        opt.step()

    def payload(self):
        return {"format": FORMAT, "global_update": self.opt.step_count, "update": self.opt.step_count,
                "executive": clone(self.net), "optimizer": self.opt.portable_state(),
                "source_admitted_checkpoint_sha256": EXPECTED_ADMITTED_SHA256,
                "historical_base_sha256": EXPECTED_BASE_SHA256,
                "historical_base_digest": EXPECTED_BASE_DIGEST,
                "schedule_seed": SCHEDULE_SEED, "init_seed": INIT_SEED}

    def test_initial_noop_and_post_training_disabled_paths(self):
        delta, _ = self.net(self.content, self.rules, self.base)
        self.assertTrue(torch.equal(self.base + delta, self.base))
        for _ in range(3): self.step()
        learned, _ = self.net(self.content, self.rules, self.base)
        self.assertGreater(learned.abs().max().item(), 0)
        zero, _ = self.net(torch.zeros_like(self.content), self.rules, self.base)
        none, _ = self.net(self.content, torch.zeros_like(self.rules), self.base)
        self.assertEqual(zero.count_nonzero().item(), 0)
        self.assertEqual(none.count_nonzero().item(), 0)

    def test_resume_reproduces_next_optimizer_step_exactly(self):
        for _ in range(3): self.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "portable.pt"
            atomic_save(self.payload(), checkpoint)
            payload = torch.load(checkpoint, weights_only=True)
        restored = ActionIntentExecutive()
        validate_payload(payload, restored)
        restored.load_state_dict(payload["executive"])
        restored_opt = DMLSafeAdam(restored.parameters())
        restored_opt.load_portable_state(payload["optimizer"])
        self.step()
        self.step(restored, restored_opt)
        assert_exact(clone(self.net), restored, "continuation")

    def test_checkpoint_identity_and_corruption_fail_closed(self):
        self.step()
        for field, bad in (("source_admitted_checkpoint_sha256", "0" * 64),
                           ("schedule_seed", 7), ("global_update", 4900)):
            payload = self.payload()
            payload[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError): validate_payload(payload, self.net)
        payload = self.payload()
        next(iter(payload["executive"].values())).view(-1)[0] = float("nan")
        with self.assertRaises(ValueError): validate_payload(payload, self.net)
        payload = self.payload()
        payload["optimizer"]["step_count"] += 1
        with self.assertRaises(ValueError): validate_payload(payload, self.net)

    def test_preservation_check_detects_actual_mutation(self):
        reference = clone(self.net)
        with torch.no_grad(): self.net.output.weight.add_(.1)
        with self.assertRaises(AssertionError): assert_exact(reference, self.net, "test")


class RealBaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(BASE_CHECKPOINT).exists():
            raise unittest.SkipTest("real baseline checkpoint is not present")
        if sha(BASE_CHECKPOINT) != EXPECTED_BASE_SHA256:
            raise RuntimeError("unexpected baseline checkpoint")
        cls.base, cls.cfg, _, _ = _load_base_checkpoint(BASE_CHECKPOINT)
        if digest(clone(cls.base)) != EXPECTED_BASE_DIGEST:
            raise RuntimeError("baseline reconstruction changed")

    def test_real_base_forward_freezing_and_optimizer_preservation(self):
        # Random NEW memory/recall, not David's missing trained memory checkpoint.
        admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(self.base, self.cfg)
        model = ActionIntentExecutiveInfant.grow_from_admitted(admitted)
        model.train()
        self.assertFalse(model.admitted.training)
        self.assertTrue(all(not p.requires_grad for p in model.admitted.parameters()))
        reference = clone(model.admitted)
        env = CanonicalExecutiveRuleWorld(0, cue="left", rule="vertical")
        obs = env.reset()
        image = _image_tensor(env, self.cfg.image_size, "cpu")
        context, _ = _context_tensor(obs, "cpu", self.cfg)
        h = model.initial_state(1, "cpu")
        memory = model.initial_memory(1, "cpu")
        prev = torch.zeros(1, dtype=torch.long)
        output = model.step(image, context, memory, prev, h, torch.tensor([1]))
        plain = model.admitted.step(image, context, memory, prev, h, use_memory=True, use_recall=True)
        self.assertTrue(torch.equal(output["logits"], plain["logits"]))
        self.assertTrue(torch.equal(output["hidden"], plain["hidden"]))
        optimizer = DMLSafeAdam(model.trainable_parameters())
        optimizer.zero_grad()
        logits, _, _ = model.compose(torch.randn(1, 128), torch.tensor([1]), output["base_logits"].detach())
        torch.nn.functional.cross_entropy(logits, torch.tensor([4])).backward()
        optimizer.step()
        assert_exact(reference, model.admitted, "real base plus new memory")

    def test_real_policy_adapter_runs_an_episode_without_target_inputs(self):
        from evaluate_action_intent_control_v1 import BrainPolicy
        admitted = CapacityMatchedRecallWorkingMemoryInfant.grow_from(self.base, self.cfg)
        model = ActionIntentExecutiveInfant.grow_from_admitted(admitted).eval()
        env = CanonicalExecutiveRuleWorld(0, cue="left", rule="vertical_reverse", max_episode_steps=8)
        result = run_rollout(env, BrainPolicy(model, self.cfg, "cpu"), image_size=self.cfg.image_size)
        self.assertEqual(result["teacher_actions"], 0)
        self.assertGreater(result["steps"], 0)
        self.assertLessEqual(result["steps"], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
