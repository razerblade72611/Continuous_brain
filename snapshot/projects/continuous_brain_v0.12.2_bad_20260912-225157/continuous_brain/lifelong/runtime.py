"""Persistent local agent with candidate learning and measured admission."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from . import BUILD, SCHEMA
from .learning import (demonstrations, experience, train_block, evaluate, promotion_gate,
                       forecast_features, measured_deltas)
from .memory import ExperienceMemory, BehaviorHistory
from .model import (DevelopmentBrain, DevelopmentForecaster, model_payload, load_model,
                    fingerprint, cpu_state, tensor, observation_tensors, finite_step)
from .storage import LifetimeStore
from .world import ResourceWorld, ACTIONS


@dataclass(frozen=True)
class RuntimeConfig:
    learning_every: int = 256
    updates_per_block: int = 16
    batch_size: int = 8
    sequence: int = 12
    learning_rate: float = .0002
    episode_horizon: int = 96
    evaluation_horizon: int = 64
    retention: float = .95
    save_every: int = 32

    def validate(self):
        if not 16 <= self.learning_every <= 100000:
            raise ValueError("learning_every must be between 16 and 100000")
        if not 1 <= self.updates_per_block <= 64:
            raise ValueError("updates_per_block must be between 1 and 64")
        if not 1 <= self.batch_size <= 64 or not 4 <= self.sequence <= 32:
            raise ValueError("Invalid batch or sequence size")
        if not 0 < self.learning_rate <= .001:
            raise ValueError("Learning rate must be in (0, 0.001]")
        if not 32 <= self.episode_horizon <= 512 or not 16 <= self.evaluation_horizon <= 256:
            raise ValueError("Episode/evaluation horizon is outside the supported bounds")
        if not .95 <= self.retention <= 1. or not 1 <= self.save_every <= 256:
            raise ValueError("Invalid retention or save interval")


def runtime_signature(device):
    from ..device_v111 import device_report
    return {"device": str(device), "torch": str(torch.__version__),
            "python": platform.python_version(), "platform": platform.system(),
            "numpy": str(np.__version__), "device_report": device_report(device)}


def save_model_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    tmp.replace(path)


class DevelopmentRuntime:
    def __init__(self, directory, device, *, cfg=None, starter=None, seed=None, name="Child",
                 notify=None):
        self.device = device
        self.notify = notify or (lambda message: print(message, flush=True))
        self.store = LifetimeStore(directory)
        self.pending = []
        self.status_message = "Initializing"
        try:
            saved = self.store.load()
            if saved:
                if saved.get("schema") != SCHEMA or saved.get("build") != BUILD:
                    raise ValueError("This state needs an explicit format migration")
                self.cfg = RuntimeConfig(**saved["config"])
                if cfg is not None and cfg != self.cfg:
                    raise ValueError("Settings differ from this individual's saved protocol")
                self._restore(saved)
            else:
                self.cfg = cfg or RuntimeConfig()
                self.cfg.validate()
                self._new(starter, seed, name)
            self.cfg.validate()
            self.notify("Preparing the fixed training demonstrations (never used for live actions).")
            self.teaching = demonstrations()
            signature = runtime_signature(device)
            # This short suite is remeasured each session, including after driver
            # changes that a high-level device name/version might not reveal.
            self.notify("Measuring the founder and active policies on this device.")
            self.founder_eval = evaluate(self.founder, horizon=self.cfg.evaluation_horizon)
            self.active_eval = self.founder_eval if fingerprint(self.model) == fingerprint(self.founder) else evaluate(
                self.model, horizon=self.cfg.evaluation_horizon)
            self._event("device_baseline", "evaluation", {"runtime": signature,
                                                          "summary": self.active_eval["summary"]})
            self.status_message = "Ready"
            self.commit()
        except BaseException:
            self.store.close()
            raise

    def _new(self, starter, seed, name):
        if not starter or not Path(starter).is_file():
            raise FileNotFoundError("The packaged starter is missing. Run the bootstrap command to create one.")
        payload = torch.load(starter, map_location="cpu", weights_only=False)
        if payload.get("build") != BUILD or payload.get("kind") != "development_starter":
            raise ValueError("This is not a v0.12 development starter; legacy checkpoints use the legacy command")
        self.model = load_model(payload, self.device).eval()
        self.founder = copy.deepcopy(self.model).eval()
        for p in self.founder.parameters():
            p.requires_grad_(False)
        self.seed = int(seed if seed is not None else random.SystemRandom().randrange(1_000_000_000))
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        self.forecaster = DevelopmentForecaster().to(self.device)
        self.forecast_optimizer = torch.optim.Adam(self.forecaster.parameters(), lr=.0005, foreach=False)
        self.identity = {"id": str(uuid.uuid4()), "name": str(name)[:80],
                         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "lineage": "new developmental individual from packaged starter",
                         "starter_model_sha256": payload["model_sha256"],
                         "inherits_v114_checkpoint": False}
        self.steps = self.episode = self.blocks = self.promotions = 0
        self.last_learning_step = 0
        self.memory, self.behavior = ExperienceMemory(), BehaviorHistory()
        self.replay, self.current_trace, self.workspace_history = [], [], []
        self.forecasts = []
        self.last_decision = self.last_gate = None
        self.env = self._new_world()
        self.hidden = self.model.initial(1, self.device)
        self._event("birth", "system", {"identity": self.identity, "seed": self.seed})

    def _restore(self, saved):
        self.model = load_model(saved["active"], self.device).eval()
        self.founder = load_model(saved["founder"], self.device).eval()
        for p in self.founder.parameters():
            p.requires_grad_(False)
        self.forecaster = DevelopmentForecaster().to(self.device)
        self.forecaster.load_state_dict(saved["forecaster"], strict=True)
        self.forecast_optimizer = torch.optim.Adam(self.forecaster.parameters(), lr=.0005, foreach=False)
        self.forecast_optimizer.load_state_dict(saved["forecast_optimizer"])
        self.identity, self.seed = saved["identity"], saved["seed"]
        self.rng = random.Random()
        self.rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        for name in ("steps", "episode", "blocks", "promotions", "last_learning_step", "replay",
                     "current_trace", "workspace_history", "forecasts", "last_decision", "last_gate",
                     "active_eval", "founder_eval"):
            setattr(self, name, saved[name])
        self.memory = ExperienceMemory(saved["memory"])
        self.behavior = BehaviorHistory(saved["behavior"])
        self.env = ResourceWorld.restore(saved["environment"])
        self.hidden = saved["hidden"].to(self.device)

    def _new_world(self):
        return ResourceWorld(2_000_000 + self.seed + self.episode,
                             kind="open" if self.episode % 3 == 0 else "detour",
                             horizon=self.cfg.episode_horizon)

    def _event(self, kind, provenance, content):
        self.pending.append({"step": getattr(self, "steps", 0), "kind": kind,
                             "provenance": provenance, "content": content})

    def _model_step(self):
        obs = self.env.observe()
        memory, style = self.memory.retrieve(obs["sensors"]), self.behavior.vector()
        with torch.no_grad():
            out = self.model(*observation_tensors(obs, memory, style, self.device), self.hidden)
            self.hidden = out["hidden"]
            scores = self.model.choose(out).detach().cpu()[0]
            proposed = int(scores.argmax())
            epsilon = max(.03, .15 * (1 - self.steps / 20000))
            exploratory = self.rng.random() < epsilon
            action = self.rng.randrange(5) if exploratory else proposed
            onehot = F.one_hot(torch.tensor([action]), 5).float().to(self.device)
            pred = self.model.predict(self.hidden, onehot)
            predicted = {"body": pred["body"][0].cpu().tolist(),
                         "reward": float(pred["reward"][0].cpu()),
                         "collision_probability": float(torch.sigmoid(pred["collision_logit"])[0].cpu())}
            # Independent hypothetical action forecasts never enter lived memory.
            all_preds = self.model.predict(self.hidden.expand(5, -1), torch.eye(5).to(self.device))
            shadow_scores = (all_preds["reward"] - .1 * torch.sigmoid(all_preds["collision_logit"]))
            shadow_action = int(shadow_scores.detach().cpu().argmax())
        nxt, reward, done, info = self.env.step(action)
        self.steps += 1
        self.current_trace.append(experience(obs, memory, style, action, reward, nxt, info, done))
        self.workspace_history.append({"grid": obs["grid"].astype(np.uint8),
                                       "sensors": obs["sensors"], "memory": memory, "style": style})
        self.memory.add(obs["sensors"], action, reward, info)
        self.behavior.observe(action, info)
        self.last_decision = {"action": ACTIONS[action], "action_index": action,
                              "policy_action": ACTIONS[proposed], "exploration": exploratory,
                              "predicted": predicted, "actual": info,
                              "shadow_prediction_action": ACTIONS[shadow_action],
                              "shadow_controls_agent": False, "teacher_controls_agent": False}
        self._event("experience", "lived_simulation", self.last_decision)
        if done:
            self._event("episode_end", "lived_simulation", {"episode": self.episode,
                        "harvests": self.env.harvests, "depleted": info["depleted"],
                        "return": self.env.total_return, "kind": self.env.kind,
                        "boundary": "new simulated episode; lifetime memory retained"})
            self.replay.append(self.current_trace)
            self.replay = self.replay[-64:]
            self.current_trace, self.workspace_history = [], []
            self.episode += 1
            self.env = self._new_world()
            self.hidden = self.model.initial(1, self.device)

    @torch.no_grad()
    def _reencode_workspace(self, model):
        hidden = model.initial(1, self.device)
        for row in self.workspace_history:
            hidden = model(*observation_tensors(row, row["memory"], row["style"], self.device), hidden)["hidden"]
        return hidden

    def learn(self):
        self.status_message = "Training a candidate; active controller is preserved"
        self.notify(f"Learning block {self.blocks + 1}: {self.cfg.updates_per_block} bounded candidate updates.")
        before_hash = fingerprint(self.model)
        features = forecast_features(self.active_eval, live_steps=self.steps, blocks=self.blocks,
                                     updates=self.cfg.updates_per_block, lr=self.cfg.learning_rate)
        with torch.no_grad():
            forecast = self.forecaster(tensor(features[None], self.device))[0].cpu().tolist()
        candidate = copy.deepcopy(self.model)
        training, _ = train_block(candidate, self.teaching, self.replay + [self.current_trace],
                                  updates=self.cfg.updates_per_block, batch_size=self.cfg.batch_size,
                                  sequence=self.cfg.sequence, lr=self.cfg.learning_rate,
                                  rng=self.rng, reference=self.model)
        if fingerprint(self.model) != before_hash:
            raise RuntimeError("Candidate training changed the active controller")
        self.status_message = "Evaluating the candidate in its own closed-loop episodes"
        after = evaluate(candidate, horizon=self.cfg.evaluation_horizon)
        gate = promotion_gate(self.active_eval, after, self.founder_eval, retention=self.cfg.retention)
        actual = measured_deltas(self.active_eval, after)
        forecast_row = {"block": self.blocks + 1, "predicted_before_training": forecast,
                        "measured_after_training": actual,
                        "absolute_error": float(np.mean(np.abs(np.array(forecast) - actual))),
                        "zero_change_baseline_error": float(np.mean(np.abs(actual))),
                        "used_to_authorize_promotion": False}
        # Online/prequential error is measured BEFORE fitting this new example.
        self.forecaster.train()
        target = tensor(np.array(actual)[None], self.device)
        f_loss = F.mse_loss(self.forecaster(tensor(features[None], self.device)), target)
        finite_step(f_loss, self.forecaster, self.forecast_optimizer)
        self.forecasts.append(forecast_row)
        self.forecasts = self.forecasts[-256:]
        endpoint = self.store.directory / "candidates" / f"block_{self.blocks+1:06d}_{after['model_sha256'][:12]}.pt"
        payload = {"build": BUILD, "kind": "development_candidate", **model_payload(candidate),
                   "gate": gate, "evaluation": after, "training": training,
                   "source_model_sha256": before_hash}
        if not endpoint.exists():
            save_model_file(endpoint, payload)
        if gate["accepted"]:
            new_hidden = self._reencode_workspace(candidate)
            self.model = candidate.eval()
            self.hidden = new_hidden
            self.active_eval = after
            self.promotions += 1
        self.blocks += 1
        self.last_learning_step = self.steps
        self.last_gate = {**gate, "candidate": endpoint.name, "candidate_summary": after["summary"],
                          "forecast": forecast_row}
        self._event("learning_block", "learning_and_evaluation", {"gate": self.last_gate, "training": training})
        self.status_message = "Candidate admitted" if gate["accepted"] else "Candidate retained for analysis; active policy kept"
        self.notify(self.status_message + ("." if gate["accepted"] else ": " + "; ".join(gate["reasons"])))
        self.commit()

    def run(self, steps, *, on_tick=None, stop_requested=None):
        if not 1 <= int(steps) <= 10_000_000:
            raise ValueError("steps must be between 1 and 10000000")
        target = self.steps + int(steps)
        try:
            while self.steps < target:
                if stop_requested and stop_requested():
                    break
                self._model_step()
                if self.steps - self.last_learning_step >= self.cfg.learning_every:
                    self.learn()
                elif self.steps % self.cfg.save_every == 0:
                    self.commit()
                if on_tick:
                    on_tick(self.summary())
                if self.steps % 64 == 0:
                    self.notify(f"Lived steps {self.steps}; episode {self.episode}; "
                                f"resources {self.behavior.counts['food'] + self.behavior.counts['water']}; "
                                f"admitted updates {self.promotions}/{self.blocks} blocks.")
        except KeyboardInterrupt:
            self.status_message = "Paused; lifetime saved"
            self.notify(self.status_message)
        else:
            self.status_message = ("Paused; lifetime saved" if self.steps < target else
                                   "Session complete; relaunch to continue this individual")
        finally:
            # Candidate learning never mutates the active model before admission.
            self.commit()
        return self.summary()

    def summary(self):
        return {"build": BUILD, "identity": self.identity, "steps": self.steps,
                "episodes": self.episode, "learning_blocks": self.blocks, "promotions": self.promotions,
                "device": str(self.device), "status": self.status_message,
                "body": dict(zip(("energy", "hydration", "integrity"), map(float, self.env.body))),
                "positive": self.env.last_positive, "negative": self.env.last_negative,
                "behavior": self.behavior.describe(), "resources": {"food": self.behavior.counts["food"],
                                                                     "water": self.behavior.counts["water"]},
                "memory": {"recent_retrieval": len(self.memory.recent), "formative_retrieval": len(self.memory.formative)},
                "active_evaluation": getattr(self, "active_eval", {}).get("summary"),
                "last_decision": self.last_decision, "last_gate": self.last_gate,
                "forecaster": {"examples": len(self.forecasts), "advisory_only": True,
                               "mean_absolute_error": float(np.mean([f["absolute_error"] for f in self.forecasts])) if self.forecasts else None,
                               "zero_change_baseline_error": float(np.mean([f["zero_change_baseline_error"] for f in self.forecasts])) if self.forecasts else None},
                "world": {"walls": self.env.walls.tolist(), "agent": list(self.env.agent),
                          "food": list(self.env.food), "water": list(self.env.water), "kind": self.env.kind},
                "state_directory": str(self.store.directory),
                "scope": "Synthetic resource-world research; no consciousness or general-language claim"}

    def commit(self):
        def cpu_tree(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().clone()
            if isinstance(obj, dict):
                return {k: cpu_tree(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [cpu_tree(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(cpu_tree(v) for v in obj)
            return obj
        payload = {"schema": SCHEMA, "build": BUILD, "config": asdict(self.cfg),
                   "active": model_payload(self.model), "founder": model_payload(self.founder),
                   "forecaster": cpu_state(self.forecaster),
                   "forecast_optimizer": cpu_tree(self.forecast_optimizer.state_dict()),
                   "identity": self.identity, "seed": self.seed, "python_rng": self.rng.getstate(),
                   "torch_rng": torch.get_rng_state(), "runtime": runtime_signature(self.device),
                   "memory": self.memory.state_dict(), "behavior": self.behavior.counts,
                   "environment": self.env.state_dict(), "hidden": self.hidden.detach().cpu(),
                   **{k: getattr(self, k) for k in ("steps", "episode", "blocks", "promotions",
                       "last_learning_step", "replay", "current_trace", "workspace_history", "forecasts",
                       "last_decision", "last_gate", "active_eval", "founder_eval")}}
        self.store.commit(payload, self.summary(), self.pending)
        self.pending.clear()

    def close(self):
        self.store.close()
