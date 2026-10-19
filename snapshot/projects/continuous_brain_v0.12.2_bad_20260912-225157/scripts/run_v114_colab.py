"""Colab orchestration: immutable inputs, stage endpoints, measured evidence, Drive sync."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch
from continuous_brain.evidence_v114 import (BUILD_ID, EXACT_SOURCE_SHA, STAGES, TARGETS,
    copy_verified, sha256, write_json, extract_outer_final_json, valid_p4b, valid_training, safe_extract)

P4_ARGS = ["--episodes", "100", "--steps", "120", "--seed-start", "12001", "--option-depth", "2",
           "--selector-topk", "2", "--beam-width", "8", "--max-option-steps", "12", "--discount", ".95",
           "--min-commit", "3", "--terminate-threshold", ".50"]


class ColabRun:
    def __init__(self, project, inputs, work, drive_root, world_archive):
        self.project, self.inputs, self.work, self.drive = map(Path, (project, inputs, work, drive_root))
        self.state = self.work / "state" / "child_v114_protected_r2"
        self.durable_state = self.drive / "state" / "child_v114_protected_r2"
        self.results = self.work / "results"
        self.durable_results = self.drive / "results" / "protected_integration_r2"
        self.data = self.inputs / "payload" / "datasets"
        self.exact = self.inputs / "payload" / "state" / "child_v110_growth"
        self.world_archive = Path(world_archive)
        for path in (self.state.parent, self.results, self.durable_results):
            path.mkdir(parents=True, exist_ok=True)
        self.runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                            gpu=torch.cuda.get_device_name(0),
                            source_manifest_sha256=sha256(self.project / "MANIFEST_SHA256.json"),
                            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                            tf32_cudnn=torch.backends.cudnn.allow_tf32,
                            cudnn_benchmark=torch.backends.cudnn.benchmark)
        self.runtime_id = hashlib.sha256(json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()
        self.save("V114_RUNTIME.json", {**self.runtime, "runtime_id": self.runtime_id})

    def save(self, name, obj):
        file = write_json(self.results / name, obj)
        copy_verified(file, self.durable_results / name)
        return file

    def command(self, args, log_name, *, allow_failure=False):
        command = [sys.executable, "-u", *map(str, args)]
        log = self.results / log_name
        print("$", " ".join(command), flush=True)
        # Stream logs directly to Drive too; active training and dataset IO remain local.
        with log.open("w", buffering=1) as stream, (self.durable_results / log_name).open("w", buffering=1) as durable:
            proc = subprocess.Popen(command, cwd=self.project, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True); stream.write(line); durable.write(line)
                rc = proc.wait()
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                raise
        if rc and not allow_failure:
            raise RuntimeError(f"Command failed ({rc}); see {log.name}. Periodic saved checkpoints remain available.")
        return rc, log

    def cli(self, command, args, log_name):
        return self.command(["-m", "continuous_brain.parented_cli", command, *args], log_name)

    def status(self, state=None):
        _, log = self.cli("protected-growth-status", ["--state-dir", state or self.state, "--device", "cpu"],
                          "V114_STATUS.stdout.log")
        result = extract_outer_final_json(log.read_text())
        if result.get("build_id") != BUILD_ID or not result.get("legacy_integrity_passed"):
            raise RuntimeError("Protected branch version/integrity check failed")
        if result.get("source_checkpoint_sha256") != EXACT_SOURCE_SHA:
            raise RuntimeError("Protected branch came from another source checkpoint")
        return result

    def stage_inputs(self):
        assert sha256(self.exact / "infant_training.pt") == EXACT_SOURCE_SHA
        local, durable = self.state / "infant_training.pt", self.durable_state / "infant_training.pt"
        if local.exists() and durable.exists() and sha256(local) != sha256(durable):
            # Both are verified independently. Refuse divergent branches at equal progress.
            a, b = self.status(self.state), self.status(self.durable_state)
            if a["total_updates"] == b["total_updates"]:
                raise RuntimeError("Local and Drive branches differ at equal progress; preserve both for diagnosis")
            if b["total_updates"] > a["total_updates"]:
                copy_verified(durable, local)
        elif not local.exists() and durable.exists():
            copy_verified(durable, local)
        if not local.exists():
            self.cli("protect-growth", ["--source-state-dir", self.exact, "--state-dir", self.state, "--device", "cuda"],
                     "V114_MIGRATION.stdout.log")
        # Copy persistent receipts/endpoints individually; no state tree is ever deleted.
        for source_dir, dest_dir in ((self.durable_state, self.state), (self.state, self.durable_state)):
            if source_dir.exists():
                for file in source_dir.iterdir():
                    if file.is_file() and file.name != "infant_training.previous.pt" and not file.name.endswith(".partial"):
                        dest = dest_dir / file.name
                        if not dest.exists(): copy_verified(file, dest)
        copy_verified(local, durable)
        st = self.status()
        print("Protected progress:", st["stage_updates"])
        print("Working datasets:", self.data)
        return st

    def warm_cache(self):
        total, start = 0, time.time()
        for file in self.data.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): total += len(chunk)
        print(f"Warmed {total / 1024**3:.2f} GiB of local training data in {time.time()-start:.1f}s")

    def evaluate(self, name, command, args, state, *, stdout_result=False):
        checkpoint = Path(state) / "infant_training.pt"
        digest = sha256(checkpoint)
        protocol = [command, *map(str, args)]
        # Paths change when Colab restarts; state hash and source manifest are the identities.
        expected = dict(build_id=BUILD_ID, checkpoint_sha256=digest,
                        runtime_id=self.runtime_id, protocol=protocol)
        cached = self.durable_results / name
        if cached.exists():
            result = json.loads(cached.read_text())
            if result.get("_v114_evidence") == expected:
                print("Using verified result:", name)
                copy_verified(cached, self.results / name)
                return result
        out = self.results / name
        cli_args = ["--state-dir", state, *args, "--device", "cuda"]
        if not stdout_result: cli_args += ["--output", out]
        _, log = self.cli(command, cli_args, name.replace(".json", ".stdout.log"))
        result = (extract_outer_final_json(log.read_text()) if stdout_result else json.loads(out.read_text()))
        if sha256(checkpoint) != digest:
            raise RuntimeError("Evaluation modified its checkpoint")
        if result.get("checkpoint_sha256", digest) != digest:
            raise RuntimeError("Evaluator returned a different checkpoint hash")
        result["_v114_evidence"] = expected
        self.save(name, result)
        return result

    def baseline(self):
        result = self.evaluate("V114_BASELINE_EXACT_P4B_12001_12100.json",
                               "evaluate-hierarchical-options", P4_ARGS, self.exact)
        if not valid_p4b(result, EXACT_SOURCE_SHA):
            raise RuntimeError("Exact-expansion P4b baseline failed; stop before protected training")
        print("Baseline completion:", result["hierarchical_option_planning"]["completion_rate"])
        return result

    def endpoint_state(self, stage):
        filename = f"infant_training.at_V114_{stage.upper()}.pt"
        source = self.state / filename
        if not source.exists() and (self.durable_state / filename).exists():
            copy_verified(self.durable_state / filename, source)
        if not source.exists():
            st = self.status()
            if st["current_stage"] == stage and st["stage_updates"].get(stage) == TARGETS[stage]:
                # Recover a lost post-save endpoint copy from the measured active endpoint.
                copy_verified(self.state / "infant_training.pt", source)
                copy_verified(source, self.durable_state / filename)
            else:
                raise RuntimeError(f"Missing historical {stage} endpoint; cannot label a later checkpoint as that stage")
        dest = self.work / "evaluation" / stage
        copy_verified(source, dest / "infant_training.pt")
        return dest

    def train_stage(self, stage):
        st = self.status()
        done = st["stage_updates"].get(stage, 0)
        if done < TARGETS[stage]:
            prior = ("V114_BASELINE_EXACT_P4B_12001_12100.json" if stage == "shadow" else
                     f"V114_{STAGES[STAGES.index(stage)-1].upper()}_P4B_12001_12100.json")
            self.cli("train-protected-growth", [
                "--rehearsal-datasets", self.data / "nursery_v090_bootstrap", self.data / "nursery_v095_mixed_003",
                "--world-datasets", self.data / "nursery_v095_mixed_003", self.data / "nursery_v096_detour_train",
                    self.data / "nursery_v110_randomized_detour_train", self.data / "nursery_v110_two_barrier_train",
                "--planning-dataset", self.data / "nursery_v110_randomized_detour_train",
                "--planning-rehearsal-dataset", self.data / "nursery_v096_detour_train",
                "--state-dir", self.state, "--durable-state-dir", self.durable_state,
                "--previous-p4b-result", self.durable_results / prior,
                "--stage", stage, "--batch", "8", "--learning-rate", ".0001", "--anchor-weight", ".10",
                "--interface-anchor-weight", "1.0", "--checkpoint-every", "250", "--seed", "11410", "--device", "cuda"],
                f"V114_{stage.upper()}_TRAINING.stdout.log")
        elif done != TARGETS[stage]:
            raise RuntimeError("Stage exceeded its update ceiling")
        endpoint = self.endpoint_state(stage)
        # Recreate receipts only from an actual measured checkpoint record, never assumed zero drift.
        payload = torch.load(endpoint / "infant_training.pt", map_location="cpu", weights_only=False)
        measured = dict((payload.get("metadata") or {}).get("protected_growth_last_measurement") or {})
        del payload
        measured.update(checkpoint_sha256=sha256(endpoint / "infant_training.pt"), checkpoint=str(endpoint / "infant_training.pt"))
        if not valid_training(measured, stage):
            raise RuntimeError(f"Missing or invalid measured training integrity for {stage}")
        self.save(f"V114_{stage.upper()}_TRAINING.json", measured)
        p4 = self.evaluate(f"V114_{stage.upper()}_P4B_12001_12100.json", "evaluate-hierarchical-options", P4_ARGS, endpoint)
        if not valid_p4b(p4, measured["checkpoint_sha256"]):
            self.save("V114_STOP.json", dict(stage=stage, reason="P4b regression failed", checkpoint_sha256=measured["checkpoint_sha256"]))
            raise RuntimeError(f"STOP: {stage} failed P4b. Endpoint retained; next gate remains closed.")
        print(stage, "P4b:", p4["hierarchical_option_planning"]["completion_rate"])
        return p4

    def run_stages(self):
        return {stage: self.train_stage(stage) for stage in STAGES}

    def prepare_world(self):
        root = self.work / "world_holdouts"
        safe_extract(self.world_archive, root)
        original, foraging, detour = root / "original_seal.json", root / "foraging", root / "detour"
        seal_sha = "8f00579cad8e283fac4103da250b1c768b5ad92353d628f8bc50700db4d63a4b"
        if sha256(original) != seal_sha: raise RuntimeError("Original world seal hash mismatch")
        from continuous_brain.world_consolidation_v104 import transition_digest
        seal = json.loads(original.read_text())
        expected = {"foraging": "1797991caa73463ebbed25f35e5240dba339f35e007b64a574195672bf2b2b5e",
                    "detour": "a0ba1cb474601031626214b371e988afd187f345a0836a071056cc5764062551"}
        for name, path in (("foraging", foraging), ("detour", detour)):
            digest = transition_digest(path)
            if digest != expected[name] or seal["holdouts"][name]["transition_digest"] != digest:
                raise RuntimeError("World transition digest mismatch")
            seal["holdouts"][name]["path"] = str(path.resolve())
        # All original fields preserved except the two relocation paths.
        relocated = self.save("V114_WORLD_SEAL_COLAB_RELOCATED.json", seal)
        self.save("V114_WORLD_RELOCATION_PROVENANCE.json", dict(original_seal_sha256=seal_sha,
            relocated_seal_sha256=sha256(relocated), transition_digests=expected, changed_fields=[
                "holdouts.foraging.path", "holdouts.detour.path"]))
        return foraging, detour, relocated

    def require_stage_evidence(self):
        for stage in STAGES:
            tr_path = self.durable_results / f"V114_{stage.upper()}_TRAINING.json"
            p4_path = self.durable_results / f"V114_{stage.upper()}_P4B_12001_12100.json"
            if not tr_path.exists() or not p4_path.exists():
                raise RuntimeError("Stage evidence is incomplete; run the stage cell first")
            tr, p4 = json.loads(tr_path.read_text()), json.loads(p4_path.read_text())
            endpoint = self.endpoint_state(stage)
            digest = sha256(endpoint / "infant_training.pt")
            if (not valid_training(tr, stage) or tr.get("checkpoint_sha256") != digest
                or not valid_p4b(p4, digest)
                or (p4.get("_v114_evidence") or {}).get("runtime_id") != self.runtime_id):
                raise RuntimeError(f"Stage {stage} is unqualified on this runtime; run the stage cell first")

    def qualify(self):
        self.require_stage_evidence()
        # Requiring the measured moderate endpoint prevents running final gates early.
        state = self.endpoint_state("moderate")
        boot = self.evaluate("V114_BOOTSTRAP_RETENTION.json", "evaluate",
            ["--dataset", str(self.data / "nursery_v090_bootstrap"), "--seed", "17"], state, stdout_result=True)
        foraging, detour, seal = self.prepare_world()
        world = self.evaluate("V114_WORLD_RETENTION.json", "evaluate-world-consolidation",
            ["--foraging-holdout", str(foraging), "--detour-holdout", str(detour), "--holdout-seal", str(seal)], state)
        # Preserve every actual result; do not interpret missing retention evidence as permission to continue.
        if boot.get("passed") is not True or world.get("world_consolidation_passed") is not True:
            self.save("V114_STOP.json", dict(reason="Final retention failed", bootstrap=boot.get("passed"),
                                              world=world.get("world_consolidation_passed")))
            raise RuntimeError("STOP: retention failed. Saved results are ready for diagnosis.")
        args = ["--kind", "randomized_detour", *P4_ARGS]
        args[args.index("--steps")+1] = "160"
        args[args.index("--seed-start")+1] = "41001"
        generalized = self.evaluate("V114_RANDOMIZED_DETOUR_DEV_41001_41100.json", "evaluate-generalized-planning", args, state)
        return boot, world, generalized

    def readiness(self):
        self.require_stage_evidence()
        args = [self.project / "scripts" / "check_v114_protected_readiness.py",
                "--migration-result", self.durable_state / "V114_MIGRATION.json"]
        for stage in STAGES:
            args += [f"--{stage}-training-result", self.durable_results / f"V114_{stage.upper()}_TRAINING.json",
                     f"--{stage}-p4b-result", self.durable_results / f"V114_{stage.upper()}_P4B_12001_12100.json"]
        args += ["--bootstrap-result", self.durable_results / "V114_BOOTSTRAP_RETENTION.json",
                 "--world-retention-result", self.durable_results / "V114_WORLD_RETENTION.json",
                 "--generalized-result", self.durable_results / "V114_RANDOMIZED_DETOUR_DEV_41001_41100.json",
                 "--output", self.results / "V114_PROTECTED_READINESS.json"]
        rc, log = self.command(args, "V114_PROTECTED_READINESS.stdout.log", allow_failure=True)
        if rc not in (0, 2) or not (self.results / "V114_PROTECTED_READINESS.json").exists():
            raise RuntimeError("Readiness checker failed to produce a report")
        result = json.loads((self.results / "V114_PROTECTED_READINESS.json").read_text())
        self.save("V114_PROTECTED_READINESS.json", result)
        self.save("V114_FINAL_SUMMARY.json", dict(build_id=BUILD_ID, runtime=self.runtime,
            protected_growth_passed=result["protected_growth_passed"], gates=result["gates"],
            checkpoint_sha256=sha256(self.endpoint_state("moderate") / "infant_training.pt"),
            reserved_blind_evaluation_run=False))
        print("Qualification passed:", result["protected_growth_passed"])
        print("Results:", self.durable_results)
        return result
