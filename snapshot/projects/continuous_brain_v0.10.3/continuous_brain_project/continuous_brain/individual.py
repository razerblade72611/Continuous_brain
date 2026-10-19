"""Persistent Individual Core v0.6 runtime.

An executable research foundation: lifetime workspace, episodic retrieval,
empirical self-model, intentions, options, sequence learning and world prediction.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .cognition import CognitiveCore, CognitiveConfig, SequenceLearner, tensor_packet, SKILL_COUNT
from .lifetime import LifetimeStore, SelfModel
from .model import ContinuousBrainNet
from .legacy_motor import read_legacy_motor, archive_legacy_motor
from .motor import validated_motor, check_qualification
from .skills import GOALS, SKILLS, Intention, SearchController, sensed_facts, available, primitive_action, reference_motor


class IndividualRuntime:
    def __init__(self, env, state_dir, config, vision, *, device="cpu", seed=7,
                 import_v053=None, evaluate=False, visualizer=None, name="Continuous Brain",
                 planner_diagnostic=False):
        self.env, self.cfg, self.vision = env, config, vision
        self.device = torch.device(device)
        self.evaluate, self.visualizer = evaluate, visualizer
        self.planner_diagnostic = planner_diagnostic
        if planner_diagnostic and not evaluate:
            raise ValueError("Planner diagnostic requires read-only evaluation")
        if (Path(state_dir) / "UPGRADE_INCOMPLETE").exists():
            raise ValueError("Incomplete upgrade: use the untouched original, not this directory")
        self.store = LifetimeStore(state_dir)
        self.closed = False
        self.running = True
        self.rng = random.Random(seed)
        self.motor = None
        self.motor_hidden = None
        self.motor_prev = 0
        self.motor_metadata = None
        self.motor_digest = None
        self.servo_motor = None
        self.servo_bundle = None
        self.semantic_cortex_meta = None
        self.controller_updates = 0
        self.primitive_counts = {}
        self.search = SearchController()
        self.last_retrieved = []
        self.last_plan = []
        self.last_selection = "none"
        self.last_skill = "none"
        self.last_imagination = None
        self.total_reward = 0.0
        self.world_signature = self._world_signature()
        try:
            saved = self.store.load()
            if saved and import_v053:
                raise ValueError("This individual already exists; do not reimport old state")
            if evaluate and not saved:
                raise ValueError("Evaluation requires an existing trained individual")
            if saved and saved["config"] != asdict(config):
                raise ValueError("Profile/sensory/training configuration differs from checkpoint. "
                                 "Use its original flags; never silently reset learned tensors.")
            if saved and saved["world_signature"] != self.world_signature:
                raise ValueError("Environment/scene differs from this checkpoint; explicit transfer is not implemented")
            if saved and saved["vision"] != self._vision_signature():
                raise ValueError("Vision backbone differs despite matching dimensions")
            torch.manual_seed(seed)
            self.net = CognitiveCore(config).to(self.device)
            self.learner = SequenceLearner(self.net, seed)
            self.workspace = self.net.initial_state(device=self.device)
            self.self_model = SelfModel(name=name)
            self.intention = Intention(goal="inspect" if self._development() else "reach")
            self.obs = self.env.reset()
            if saved:
                self._restore(saved)
            elif import_v053:
                self._import_motor(Path(import_v053))
            self.control_revision = ("servo_v062:" if self.servo_motor else "search_v061:") + (self.motor_digest or
                ("legacy_imported" if self.motor else "development_reference"))
            self.learner.control_revision = self.control_revision
            if not self._development() and self.motor is None:
                raise ValueError("THOR v0.6 requires a learned motor state via --import-v053-state")
            self._sensory(self.obs)  # validate interface before first checkpoint
            self.evaluation_start = {"decisions": self.self_model.decisions,
                                     "tasks": self.self_model.tasks, "successes": self.self_model.successes}
            if not saved and not evaluate:
                self.save()
        except Exception:
            self.store.close()
            self.env.close()
            raise

    def _development(self):
        return bool(getattr(self.env, "exact_resume", False))

    def _world_signature(self):
        return {"type": type(self.env).__name__,
                "scene": getattr(getattr(self.env, "config", None), "scene", None),
                "actions": list(self.env.action_names), "obs_dim": self.env.observation_dim,
                "curriculum_level": getattr(getattr(self.env, "config", None), "curriculum_level", None),
                "goal_signal": getattr(getattr(self.env, "config", None), "goal_signal", None)}

    def _vision_signature(self):
        return {"name": self.vision.name, "dim": self.vision.output_dim,
                "model_id": getattr(self.vision, "model_id", None)}

    def _create_motor(self, metadata, weights):
        self.motor_metadata = metadata
        self.motor = ContinuousBrainNet(6, 5, metadata["embed_dim"], metadata["hidden_dim"],
                                       metadata["visual_dim"]).to(self.device)
        self.motor.load_state_dict(weights, strict=True)
        self.motor.eval()
        for p in self.motor.parameters():
            p.requires_grad_(False)
        self.motor_hidden = self.motor.initial_state(1, self.device)

    def _import_motor(self, source):
        source = source.resolve()
        destination = self.store.directory.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Import source must be a separate stopped legacy state directory")
        imported = read_legacy_motor(source, self._vision_signature())
        self._create_motor(imported["metadata"], imported["weights"])
        self.motor_digest = imported["provenance"]["sha256"]
        archive = self.store.directory / "legacy_v053"
        archive_legacy_motor(source, archive, self.motor_digest)
        self.self_model.lineage = {"kind": "new_executive_inheriting_legacy_motor", **imported["provenance"]}
        if not self._development():
            self.env.load_state_dict(imported["environment"])
            self.obs = self.env.reset()
        self.store.append(0, "lineage", provenance=self.self_model.lineage)

    def _restore(self, state):
        if state.get("version") not in ("0.6.0", "0.6.1", "0.6.2", "0.7.0", "0.7.1"):
            raise ValueError("Unsupported individual checkpoint version")
        self.net.load_state_dict(state["model"], strict=True)
        self.learner.load_state_dict(state["learner"])
        self.workspace = state["workspace"].to(self.device)
        self.self_model = SelfModel(**state["self_model"])
        self.intention = Intention(**state["intention"])
        self.total_reward = state["total_reward"]
        self.semantic_cortex_meta = state.get("semantic_cortex")
        self.search = SearchController(**state.get("search_controller", {}))
        self.motor_digest = state.get("motor_digest", self.self_model.lineage.get("sha256"))
        self.controller_updates = state.get("controller_updates", self.learner.updates)
        if state.get("servo_motor") is not None:
            if not self._development() or type(self.env).__name__ != "DevelopmentRoom":
                raise ValueError("Learned development motor cannot be transferred to an unqualified body")
            if state.get("motor") is not None:
                raise ValueError("Checkpoint contains conflicting active motor controllers")
            self.servo_bundle = check_qualification(state["servo_motor"])
            self.servo_motor = validated_motor(self.servo_bundle).to(self.device)
            for parameter in self.servo_motor.parameters():
                parameter.requires_grad_(False)  # lab-trainable; fixed during an executive experiment
            self.motor_digest = self.servo_bundle["digest"]
        if state["motor"] is not None:
            self._create_motor(state["motor_metadata"], state["motor"])
            self.motor_hidden = state["motor_hidden"].to(self.device)
            self.motor_prev = state["motor_prev"]
        self.env.load_state_dict(state["environment"])
        if self._development():
            self.obs = state["observation"]
        else:
            # THOR cannot reconstruct arbitrary Unity physics. Keep lifetime mind,
            # reacquire sensors and explicitly end only the motor/physical context.
            self.obs = self.env.reset()
            self.search = SearchController()
            self.self_model.boundary = True
            self.intention.history.append({"event": "physical_reacquisition", "tick": self.self_model.ticks})
            if self.motor:
                self.motor_hidden = self.motor.initial_state(1, self.device)
                self.motor_prev = 0
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def _sensory(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (6,) or not np.isfinite(obs).all():
            raise FloatingPointError("Invalid structured observation")
        vec = self.vision.encode(self.env.render_rgb()) if self.vision.output_dim else None
        if vec is None:
            vec = np.zeros(0, dtype=np.float32)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (self.vision.output_dim,) or not np.isfinite(vec).all():
            raise FloatingPointError("Invalid visual feature vector")
        return np.concatenate((obs, vec)).astype(np.float32)

    def packet(self, obs):
        sensory = self._sensory(obs)
        memories, ids = self.store.retrieve(sensory, self.cfg.memory_k, self.cfg.memory_dim)
        self.last_retrieved = ids
        return {"sensory": sensory,
                "self": self.self_model.vector(self.self_model.decisions-self.intention.started),
                "goal": GOALS.index(self.intention.goal), "memories": memories}

    def facts(self, obs):
        return sensed_facts(obs, interaction=4 in self.env.valid_action_indices())

    def _motor_action(self, obs, orient_only=False):
        if self.servo_motor is not None:
            return self.servo_motor.action(obs, orient_only, self.env.valid_action_indices())
        if self.motor is None:
            return reference_motor(obs, orient_only)
        sensory = self._sensory(obs)
        visual = torch.as_tensor(sensory[6:], device=self.device)[None] if self.motor.visual_dim else None
        prev = torch.nn.functional.one_hot(torch.tensor([self.motor_prev], device=self.device), 5).float()
        with torch.no_grad():
            self.motor_hidden = self.motor.encode_step(torch.as_tensor(obs, device=self.device)[None],
                                                      prev, self.motor_hidden, visual)
            q = self.motor.q_values(self.motor_hidden)[0]
        allowed = set(self.env.valid_action_indices())
        if orient_only:
            allowed &= {0, 2, 3}
        if not allowed:
            return 0
        return max(sorted(allowed), key=lambda i: float(q[i]))

    def teacher_probability(self):
        if self.planner_diagnostic:
            return 1.0
        if self.evaluate:
            return 0.0
        fraction = min(1.0, self.self_model.decisions / max(1, self.cfg.teacher_decay))
        return 1.0 - 0.90 * fraction

    def _select(self, q, mask, teacher, latent=None):
        self.last_imagination = None
        if self.planner_diagnostic:
            self.last_selection = "diagnostic_planner"
            return teacher
        allowed = [i for i, ok in enumerate(mask) if ok]
        if not self.evaluate and self.rng.random() < self.teacher_probability():
            self.last_selection = "planner_teacher"
            return teacher
        if not self.evaluate and self.rng.random() < 0.10:
            self.last_selection = "exploration"
            return self.rng.choice(allowed)
        self.last_selection = "learned_option_value"
        warmup_updates = self.controller_updates if self.servo_motor is not None else self.learner.updates
        if latent is not None and warmup_updates >= self.cfg.imagination_after_updates:
            # Shallow learned consequence reranking. Primitive validity is still
            # checked independently at execution, never delegated to imagination.
            with torch.no_grad():
                actions = torch.arange(SKILL_COUNT, device=self.device)
                _, _, logvar, rewards, _ = self.net.imagine(latent.expand(SKILL_COUNT, -1), actions)
                uncertainty = logvar.exp().mean(dim=-1).sqrt()
                consequence = rewards - 0.1 * uncertainty
                q = q + self.cfg.imagination_weight * consequence[None]
                self.last_imagination = {"rewards": rewards.cpu().tolist(),
                                         "uncertainty": uncertainty.cpu().tolist()}
        return max(allowed, key=lambda i: float(q[0, i]))

    def train_executive(self):
        result = self.learner.train()
        if result is not None:
            self.controller_updates += 1
        return result

    def motor_source(self):
        if self.servo_motor is not None:
            return "learned_development_motor"
        return "imported_learned" if self.motor else "development_reference_teacher"

    def _schedule_goal(self):
        # Explicit bounded scheduler, not a claim of learned free-form preferences.
        if self.self_model.energy < 0.25:
            goal = "rest"
        elif self._development() and self.self_model.decisions % 13 == 0:
            goal = "explore"
        else:
            goal = "inspect" if self._development() else "reach"
        self.intention = Intention(goal=goal, started=self.self_model.decisions,
                                   history=self.intention.history[-64:])

    def _fault(self, info):
        self.self_model.faults += 1
        self.self_model.boundary = True
        if not self.evaluate:
            self.store.append(self.self_model.ticks, "diagnostic", cause="infrastructure_fault",
                              details=str(info.get("sensor_health", {}))[:2000])
        recovery = self.env.recover_sensor_fault()
        if not recovery.get("recovered"):
            self.running = False
            return
        self.obs = self.env.reset()
        self.search = SearchController()
        if self.motor:
            self.motor_hidden = self.motor.initial_state(1, self.device)
            self.motor_prev = 0

    def decision(self):
        packet = self.packet(self.obs)
        old_workspace = self.workspace.detach().cpu().numpy()[0].copy()
        with torch.no_grad():
            proposed, latent, q = self.net(*tensor_packet(packet, self.device), self.workspace)
        if not torch.isfinite(proposed).all() or not torch.isfinite(q).all():
            raise FloatingPointError("Nonfinite executive state; refusing to act")
        facts = self.facts(self.obs)
        mask = available(facts)
        plan = self.intention.plan(facts)
        teacher = plan[0]
        if not mask[teacher]:
            raise RuntimeError("Composed skill violates its preconditions")
        skill_id = self._select(q, mask, teacher, latent)
        skill = SKILLS[skill_id]
        self.last_plan = [SKILLS[i].name for i in plan]
        self.last_skill = skill.name
        before_obs = self.obs.copy()
        duration, reward, done, success, fault = 0, 0.0, False, False, False
        info = {}
        for t in range(skill.max_steps):
            if self.motor is not None and skill.controller != "motor":
                # Even scripted scan/idle options must advance the imported
                # recurrent motor's sensory history once per primitive tick.
                self._motor_action(self.obs)
            # Sensor-health state is a supervisor input, never privileged geometry.
            depth_healthy = bool(getattr(self.env, "_sensor_depth_healthy", True))
            action = primitive_action(skill_id, self.obs, self._motor_action, self.rng,
                                      self.search, depth_healthy)
            valid = self.env.valid_action_indices()
            if action not in valid:
                action = 0 if 0 in valid else valid[0]
            if skill.name == "search":
                self.search.last_action = action
            else:
                self.search = SearchController()
            primitive_key = f"{skill.name}/{self.env.action_names[action]}"
            self.primitive_counts[primitive_key] = self.primitive_counts.get(primitive_key, 0) + 1
            self.obs, r, done, info = self.env.step(action)
            self.motor_prev = action
            duration += 1
            self.self_model.ticks += 1
            fault = bool(info.get("infrastructure_fault", info.get("sensor_fault", False)))
            if not np.isfinite(self.obs).all() or not math.isfinite(r):
                fault = True
            if fault:
                break
            reward += self.cfg.gamma ** t * float(r)
            success = bool(info.get("reached", False))
            new_facts = self.facts(self.obs)
            if done or success or (skill.effect in new_facts) or not self.running:
                break
        self.self_model.decisions += 1
        if fault:
            self.search = SearchController()
            self._fault(info)
            return {"fault": True}
        # Workspace commits only after a verified, healthy action/outcome interval.
        self.workspace = proposed.detach()
        previous_surprise = self.self_model.surprise
        physical_sensory = self._sensory(self.obs)
        with torch.no_grad():
            _, prediction, logvar, _, _ = self.net.imagine(latent, torch.tensor([skill_id], device=self.device))
        mse = float(np.mean((prediction[0].cpu().numpy()-physical_sensory)**2))
        self.self_model.surprise = 0.95 * previous_surprise + 0.05 * min(mse, 10)
        self.self_model.learning_progress = previous_surprise-self.self_model.surprise
        self.self_model.uncertainty = float(logvar.exp().mean().sqrt().cpu())
        self.self_model.energy = float(np.clip(getattr(self.env, "energy", 1.0), 0, 1))
        self.self_model.last_success = success
        facts_after = self.facts(self.obs)
        goal_complete = (success or (self.intention.goal == "reach" and "near" in facts_after)
                         or (self.intention.goal == "explore" and "visible" in facts_after)
                         or (self.intention.goal == "rest" and self.self_model.energy >= 0.8))
        skill_success = (skill.effect in facts_after or
                         (skill.name == "inspect" and success) or
                         skill.name in ("observe", "rest"))
        failure = ("blocked" if not bool(self.obs[4]) else
                   ("no_target" if skill.name == "search" else "lost_target")
                   if "visible" not in facts_after else "timeout")
        self.self_model.observe_outcome(skill.name, skill_success, duration, failure)
        self.intention.attempts += 1
        expired = self.intention.attempts >= self.intention.budget
        boundary = bool(done or goal_complete or expired)
        self.self_model.boundary = boundary
        next_packet = self.packet(self.obs)  # physical next input, BEFORE curriculum resets
        # Small bounded novelty drive. It is a design signal, not an asserted feeling.
        novelty = min(1.0, float(np.mean((physical_sensory-packet["sensory"])**2)))
        # Do not reward merely executing observe/rest successfully: that would
        # create an artificial incentive to idle forever.
        learning_reward = float(np.clip(reward + 0.02 * novelty + 0.10 * goal_complete, -2, 2))
        row = {"serial": self.self_model.decisions, "packet": packet, "workspace": old_workspace,
               "control_revision": self.control_revision,
               "action": skill_id, "teacher": teacher, "mask": mask, "next": next_packet,
               "next_mask": available(facts_after), "duration": duration,
               "reward": learning_reward, "boundary": boundary, "success": goal_complete}
        self.total_reward += reward
        if not self.evaluate:
            self.learner.append(row)
            key = packet["sensory"]
            value = np.concatenate((physical_sensory, np.eye(SKILL_COUNT, dtype=np.float32)[skill_id],
                                    [learning_reward, float(skill_success)])).astype(np.float32)
            self.store.append(self.self_model.ticks, "experience", key, value,
                goal=self.intention.goal, target=self.intention.target, skill=skill.name,
                outcome=bool(skill_success), task_success=bool(goal_complete),
                duration=duration, decision_source=self.last_selection,
                primitive_source=self.motor_source()
                    if skill.controller == "motor" else "authored_" + skill.controller,
                control_revision=self.control_revision)
            if self.self_model.decisions % self.cfg.train_every == 0:
                self.train_executive()
        if boundary:
            self.search = SearchController()
            self.self_model.tasks += 1
            self.self_model.successes += int(goal_complete)
            self.intention.history.append({"goal": self.intention.goal, "success": bool(goal_complete),
                                            "tick": self.self_model.ticks, "attempts": self.intention.attempts})
            if done:
                self.obs = self.env.reset()
                if self.motor:
                    self.motor_hidden = self.motor.initial_state(1, self.device)
                    self.motor_prev = 0
            elif self._development() and self.intention.goal != "rest":
                self.obs = self.env.finish_intention()
            self._schedule_goal()
        return {"fault": False, "success": goal_complete, "boundary": boundary,
                "skill": skill.name, "source": self.last_selection, "reward": reward}

    def status(self):
        return {"version": "0.7.1", "individual": self.self_model.state_dict(),
                "intention": self.intention.state_dict(), "parameters": self.net.parameter_count,
                "device": str(self.device), "model_config": asdict(self.cfg),
                "training": self.learner.last, "replay": len(self.learner.replay),
                "controller_updates": self.controller_updates,
                "active_replay": self.learner.active_replay_count(),
                "control_revision": self.control_revision, "search_controller": self.search.state_dict(),
                "memory_count": self.store.count("experience"), "memory_ids": self.last_retrieved,
                "plan": self.last_plan, "last_skill": self.last_skill,
                "selection_source": self.last_selection,
                "imagination": self.last_imagination,
                "teacher_probability": self.teacher_probability(), "evaluation": self.evaluate,
                "planner_diagnostic": self.planner_diagnostic,
                "primitive_actions_by_skill": dict(self.primitive_counts),
                "run_window": {key: getattr(self.self_model, key)-value
                    for key, value in self.evaluation_start.items()},
                "evaluation_window": ({key: getattr(self.self_model, key)-value
                    for key, value in self.evaluation_start.items()} if self.evaluate else None),
                "motor_source": self.motor_source(),
                "motor_parameters": self.servo_motor.parameter_count if self.servo_motor is not None else None,
                "motor_training_mode": "separate_supervised_lab" if self.servo_motor is not None else None,
                "total_reward": self.total_reward,
                "workspace_norm": float(self.workspace.norm().cpu()),
                "semantic_cortex": self.semantic_cortex_meta,
                "communication_ready": bool(self.semantic_cortex_meta and self.semantic_cortex_meta.get("qualification", {}).get("passed", False)),
                "limitations": ["not evidence of consciousness", "skills initially authored",
                                "semantic cortex is separate from executive reasoning", "no autonomous network growth"]}

    def save(self):
        if self.evaluate:
            return
        payload = {"version": "0.7.1", "config": asdict(self.cfg),
            "world_signature": self.world_signature, "vision": self._vision_signature(),
            "model": self.net.state_dict(), "learner": self.learner.state_dict(),
            "workspace": self.workspace.detach().cpu(), "self_model": self.self_model.state_dict(),
            "intention": self.intention.state_dict(), "observation": self.obs,
            "environment": self.env.state_dict(), "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "motor": None if self.motor is None else self.motor.state_dict(),
            "motor_metadata": self.motor_metadata,
            "motor_digest": self.motor_digest, "search_controller": self.search.state_dict(),
            "servo_motor": self.servo_bundle, "controller_updates": self.controller_updates,
            "motor_hidden": None if self.motor_hidden is None else self.motor_hidden.detach().cpu(),
            "motor_prev": self.motor_prev, "total_reward": self.total_reward,
            "semantic_cortex": self.semantic_cortex_meta}
        self.store.save(payload)
        path = self.store.directory / "individual_status.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.status(), indent=2, allow_nan=False))
        temporary.replace(path)

    def _dashboard(self):
        if self.visualizer is None:
            return
        status = self.status()
        snapshot = self.env.visual_snapshot()
        snapshot.update({"individual": status, "step": self.self_model.ticks,
            "goals": self.self_model.successes, "action_name": self.last_skill,
            "task_kind": self.intention.goal, "target_label": self.intention.target,
            "total_reward": self.total_reward, "memories": status["memory_count"],
            "replay": status["replay"], "loss": self.learner.last.get("loss"),
            "vision_backbone": self.vision.name, "vision_pretrained": self.vision.pretrained,
            "surprise": self.self_model.surprise, "hidden_sample": self.workspace[0,0,:32].cpu().tolist(),
            "action_names": [s.name for s in SKILLS], "goal_signal": "skill-composition"})
        self.visualizer.update(snapshot)

    def run(self, decisions=0, checkpoint_every=100, realtime=True, hz=5.0, log_every=20):
        if min(checkpoint_every, hz, log_every) <= 0 or decisions < 0:
            raise ValueError("Positive checkpoint/log frequencies and Hz required")
        stop = self.self_model.decisions + decisions if decisions else None
        handlers = {}
        def request_stop(*_):
            self.running = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        try:
            while self.running and (stop is None or self.self_model.decisions < stop):
                start = time.monotonic()
                if self.visualizer and self.visualizer.paused:
                    time.sleep(0.05)
                    continue
                result = self.decision()
                if self.self_model.decisions % checkpoint_every == 0:
                    self.save()
                if self.self_model.decisions % log_every == 0 or result["fault"]:
                    print(f"decision={self.self_model.decisions} ticks={self.self_model.ticks} "
                          f"goal={self.intention.goal} skill={self.last_skill} source={self.last_selection} "
                          f"tasks={self.self_model.successes}/{self.self_model.tasks} "
                          f"run_tasks={self.self_model.successes-self.evaluation_start['successes']}/"
                          f"{self.self_model.tasks-self.evaluation_start['tasks']} "
                          f"memory={self.store.count('experience')} updates={self.learner.updates} "
                          f"faults={self.self_model.faults}", flush=True)
                self._dashboard()
                if realtime:
                    speed = self.visualizer.speed if self.visualizer else 1.0
                    time.sleep(max(0.0, 1/max(0.1, hz*speed)-(time.monotonic()-start)))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            self.save()

    def close(self):
        if not self.closed:
            self.store.close()
            self.env.close()
            self.closed = True
