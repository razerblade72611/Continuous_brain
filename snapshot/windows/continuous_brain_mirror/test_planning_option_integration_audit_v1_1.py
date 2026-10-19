"""Standard-library software checks; synthetic fixtures are not trained models."""
from collections import OrderedDict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_STORED

import audit_planning_option_integration_v1_1 as audit


def checkpoint_bytes(metadata=None, missing_storage=False):
    # Serialization-only fixture namespace. No neural model or tensor operations.
    torch_fixture = types.ModuleType("torch")
    utils_fixture = types.ModuleType("torch._utils")
    storage_class = type("FloatStorage", (), {"__module__": "torch"})
    torch_fixture.FloatStorage = storage_class
    exec("def _rebuild_tensor_v2(*args):\n    raise AssertionError('must never execute')\n", utils_fixture.__dict__)
    torch_fixture._utils = utils_fixture

    class StorageToken:
        pass
    storage = StorageToken()
    class TensorFixture:
        def __reduce__(self):
            return (utils_fixture._rebuild_tensor_v2,
                    (storage, 0, (2, 2), (2, 1), False, OrderedDict()))
    class FixturePickler(pickle.Pickler):
        def persistent_id(self, obj):
            return ("storage", storage_class, "0", "cpu", 4) if obj is storage else None
    payload = {"format": "continuous_brain.infant.v110", "updates": 14000,
               "config": {"planning_dim": 192, "option_count": 6},
               "metadata": metadata or {"option_cortex_version": "latent-temporal-options-v1", "option_updates": 3000},
               "model": OrderedDict((k, TensorFixture()) for k in (
                   "planning.encoder.0.weight", "options.selector.0.weight",
                   "world.members.0.net.0.weight", "target_spatial.test.weight"))}
    pickled = io.BytesIO()
    with patch.dict(sys.modules, {"torch": torch_fixture, "torch._utils": utils_fixture}):
        FixturePickler(pickled, protocol=2).dump(payload)
    result = io.BytesIO()
    with ZipFile(result, "w", ZIP_STORED) as archive:
        archive.writestr("fixture/data.pkl", pickled.getvalue())
        archive.writestr("fixture/version", "3\n")
        archive.writestr("fixture/byteorder", "little")
        if not missing_storage:
            archive.writestr("fixture/data/0", bytes(16))
    return result.getvalue()


def make_project(root):
    source = root / "continuous_brain/infant.py"
    source.parent.mkdir(parents=True)
    source.write_text("\n".join(f"class {name}:\n    def forward(self, x):\n        return x\n"
                                  for name in sorted(audit.REQUIRED_CLASSES)))
    cp = root / audit.KNOWN_CHECKPOINTS["historical_base"][0]
    cp.parent.mkdir(parents=True)
    cp.write_bytes(checkpoint_bytes())
    return cp


def p4b_report(checkpoint_hash):
    modes = {"reactive": 0, "option_direct": 61, "hierarchical_option_planning": 38}
    report = {"format": "continuous_brain.p4b_hierarchical_option_planning_gate.v103",
              "checkpoint_sha256": checkpoint_hash, "p4b_passed": True,
              "experiment": {"paired_episodes": 100, "steps_per_episode": 120,
                 "seed_start": 12001, "seed_end": 12100, "option_depth": 2,
                 "selector_topk": 2, "beam_width": 8, "max_option_steps": 12,
                 "discount": .95, "min_commit": 3, "terminate_threshold": .5,
                 "training_performed_during_evaluation": False, "parents_used": False},
              "episodes": {}}
    for mode, successes in modes.items():
        report["episodes"][mode] = [{"seed": 12001 + i, "completed": i < successes, "collisions": 0}
                                    for i in range(100)]
        key = "option_direct_diagnostic" if mode == "option_direct" else mode
        report[key] = {"completion_rate": successes / 100, "mean_collisions": 0.0}
    return report


class AuditTests(unittest.TestCase):
    def test_parse_torch_format_without_importing_torch(self):
        present = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights.pt"
            path.write_bytes(checkpoint_bytes())
            result = audit.describe_checkpoint(path)
            self.assertEqual(result["metadata_status"], "parsed")
            self.assertTrue(result["has_planning_and_options"])
            self.assertEqual(result["groups"]["planning"]["numel"], 4)
            self.assertFalse(result["parameter_presence_is_training_proof"])
        self.assertEqual("torch" in sys.modules, present)

    def test_missing_tensor_storage_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"
            path.write_bytes(checkpoint_bytes(missing_storage=True))
            self.assertEqual(audit.describe_checkpoint(path)["metadata_status"], "unavailable")

    def test_corrupt_archive_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"
            path.write_bytes(checkpoint_bytes()[:100])
            row = audit.describe_checkpoint(path)
            self.assertEqual(row["metadata_status"], "unavailable")
            self.assertIn("BadZipFile", row["error"])

    def test_unapproved_pickle_global_does_not_execute(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "must_not_exist"
            class BadObject:
                def __reduce__(self):
                    return (eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')",))
            data = pickle.dumps({"x": BadObject()}, protocol=2)
            with self.assertRaises(pickle.UnpicklingError):
                audit.decode_metadata(data)
            self.assertFalse(marker.exists())

    def test_static_source_inspection_does_not_import_source(self):
        raw = b"raise RuntimeError('must not run')\nclass OptionCortex:\n def selector_logits(self, latent): pass\n"
        record = audit.source_record(Path("source.py"), raw)
        self.assertEqual(record["classes"][0]["methods"][0]["signature"], "selector_logits(self, latent)")

    def test_p4b_recomputes_direct_option_comparison(self):
        result = audit.verify_p4b_report(p4b_report("a" * 64))
        self.assertTrue(result["complete_original_protocol_and_arithmetic"])
        self.assertTrue(result["original_gate_pass_recomputed"])
        self.assertAlmostEqual(result["hierarchical_minus_direct_percentage_points"], -23.)

    def test_p4b_rejects_summary_only_evidence(self):
        report = p4b_report("a" * 64)
        report["episodes"] = {}
        self.assertFalse(audit.verify_p4b_report(report)["complete_original_protocol_and_arithmetic"])

    def test_p4b_rejects_duplicate_seeds_and_forged_summary(self):
        report = p4b_report("a" * 64)
        report["episodes"]["reactive"][0]["seed"] = 12002
        report["hierarchical_option_planning"]["completion_rate"] = 1.0
        result = audit.verify_p4b_report(report)
        self.assertFalse(result["complete_original_protocol_and_arithmetic"])
        self.assertIn("reactive:seeds", result["errors"])
        self.assertIn("hierarchical_option_planning:completion arithmetic", result["errors"])

    def test_full_audit_preserves_inputs_and_bundles_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            cp = make_project(root)
            before = cp.read_bytes()
            (root / "P4b_result.json").write_text(json.dumps(p4b_report(audit.sha256(cp))))
            with redirect_stdout(io.StringIO()):
                code = audit.main(["--project-root", str(root)])
            self.assertEqual(code, 0)
            self.assertEqual(cp.read_bytes(), before)
            output = root / "state/planning_option_integration_audit_v1_1"
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["checkpoint_mutated"])
            self.assertFalse(report["training_performed"])
            self.assertTrue(report["planning_option_candidates"][0]["matching_verified_p4b_report"])
            with ZipFile(output / "audit_bundle.zip") as archive:
                manifest = json.loads(archive.read("MANIFEST_SHA256.json"))
                for name, sha in manifest.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), sha)

    def test_input_change_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            cp = make_project(root)
            real_sha = audit.sha256
            count = 0
            def changing_sha(path):
                nonlocal count
                if Path(path) == cp:
                    count += 1
                    if count == 2:
                        cp.write_bytes(cp.read_bytes() + b"external mutation fixture")
                return real_sha(path)
            with patch.object(audit, "sha256", side_effect=changing_sha), redirect_stdout(io.StringIO()):
                code = audit.main(["--project-root", str(root)])
            report = json.loads((root / "state/planning_option_integration_audit_v1_1/report.json").read_text())
            self.assertEqual(code, 2)
            self.assertTrue(report["checkpoint_mutated"])
            self.assertFalse(report["audit_complete"])

    def test_failed_run_replaces_stale_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            make_project(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["--project-root", str(root)]), 0)
                (root / "continuous_brain/infant.py").unlink()
                self.assertEqual(audit.main(["--project-root", str(root)]), 2)
            output = root / "state/planning_option_integration_audit_v1_1"
            with ZipFile(output / "audit_bundle.zip") as archive:
                report = json.loads(archive.read("report.json"))
                self.assertEqual(report["status"], "ERROR")
                self.assertFalse(report["audit_complete"])

    def test_runner_propagates_failure(self):
        with tempfile.TemporaryDirectory() as td:
            import os
            env = dict(os.environ, PROJECT_ROOT=td)
            script = Path(__file__).with_name("run_planning_option_integration_audit_v1_1.sh")
            proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 2)
            self.assertNotIn("AUDIT_RUN: PASS", proc.stdout)

    def test_sibling_legacy_nested_state_is_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "current"
            make_project(root)
            old = Path(td) / "continuous_brain_v0.10.3/continuous_brain_project/state/old.pt"
            old.parent.mkdir(parents=True)
            old.write_bytes(checkpoint_bytes())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["--project-root", str(root)]), 0)
            report = json.loads((root / "state/planning_option_integration_audit_v1_1/report.json").read_text())
            self.assertIn(str(old), [r["path"] for r in report["checkpoint_inventory"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
