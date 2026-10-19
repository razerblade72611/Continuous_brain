"""Independent motor training lab: teacher-labelled real simulator observations.

No persistent individual is loaded, copied, acted through or modified here.
World coordinates are confined to curriculum setup/evaluation, not policy input.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from .development import DevelopmentRoom
from .motor import DevelopmentMotor, MOTOR_CONTRACT, motor_digest, validated_motor, check_qualification
from .skills import reference_motor, sensed_facts


class MotorLabLock:
    def __init__(self, directory, create=False):
        self.path = Path(directory).resolve()
        if any((self.path / name).exists() for name in ("individual.sqlite3", "brain.pt")):
            raise ValueError("Use a separate motor-lab directory, never an individual or legacy state")
        if not create and not (self.path / "motor_training.pt").is_file():
            raise FileNotFoundError("Motor training checkpoint not found")
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "motor.lock").open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError("Motor lab is already in use; stop training/evaluation first")

    def close(self):
        if not self.file.closed:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    if path.exists():
        shutil.copy2(path, path.with_suffix(".previous.pt"))
    temporary.replace(path)


def setup_case(env, rng):
    """Evaluator/teacher-only curriculum: visible target and clear direct corridor.

    This deliberately tests local servo skills, not exploration or route planning.
    """
    for _ in range(1000):
        env.reset()
        delta = env.target - env.position
        distance = float(np.linalg.norm(delta))
        if not 0.8 <= distance <= 3.0:
            continue
        if any(env._blocked(*p, margin=0.16) for p in np.linspace(env.position, env.target, 48)):
            continue
        # Include difficult near-threshold bearings as well as broader turns.
        offset = rng.uniform(-59, 59)
        if rng.random() < 0.25:
            offset = rng.choice((-1, 1)) * rng.uniform(10, 20)
        env.yaw = (math.degrees(math.atan2(delta[0], delta[1])) - offset) % 360
        obs = env._obs()
        if "visible" in sensed_facts(obs):
            return obs
    raise RuntimeError("Cannot construct a valid local-motor curriculum case")


def collect_demonstrations(seed=7, cases=256):
    if cases < 8:
        raise ValueError("At least eight training cases required")
    env = DevelopmentRoom(seed=seed, max_episode_steps=80)
    rng = random.Random(seed)
    observations, modes, labels = [], [], []
    try:
        for case in range(cases):
            obs = setup_case(env, rng)
            orient_episode = case % 2 == 0
            for _ in range(48):
                if "visible" not in sensed_facts(obs):
                    break
                for orient in (False, True):
                    observations.append(obs.copy())
                    modes.append(orient)
                    labels.append(reference_motor(obs, orient_only=orient))
                facts = sensed_facts(obs)
                if (orient_episode and "aligned" in facts) or (not orient_episode and "near" in facts):
                    break
                action = reference_motor(obs, orient_only=orient_episode)
                obs, _, done, info = env.step(action)
                if done or info.get("infrastructure_fault"):
                    break
    finally:
        env.close()
    data = {"obs": torch.tensor(np.asarray(observations), dtype=torch.float32),
            "orient": torch.tensor(modes, dtype=torch.long), "actions": torch.tensor(labels, dtype=torch.long)}
    if set(labels) != {0, 1, 2, 3}:
        raise ValueError("Demonstrations must cover all four motor primitives")
    return data


class MotorTrainer:
    def __init__(self, seed=7, cases=256, saved=None):
        self.seed, self.cases = int(seed), int(cases)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = DevelopmentMotor()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.rng = np.random.default_rng(seed)
        self.updates, self.last_loss = 0, None
        if saved is not None:
            if (saved.get("kind") != "motor_lab_v1" or saved.get("seed") != seed
                    or saved.get("cases") != cases):
                raise ValueError("Motor lab configuration differs; use its original seed/case count")
            validated_motor(saved)
            self.model.load_state_dict(saved["model"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.rng.bit_generator.state = saved["rng"]
            self.updates, self.last_loss = saved["updates"], saved["last_loss"]
            self.data = saved["data"]
        else:
            self.data = collect_demonstrations(seed, cases)
        self.classes = [torch.where(self.data["actions"] == action)[0].numpy() for action in range(4)]
        if any(len(group) == 0 for group in self.classes):
            raise ValueError("Motor training data is missing a primitive class")

    def update(self):
        self.model.train()
        indices = np.concatenate([self.rng.choice(group, 32, replace=True) for group in self.classes])
        indices = torch.as_tensor(indices)
        logits = self.model(self.data["obs"][indices], self.data["orient"][indices])
        loss = torch.nn.functional.cross_entropy(logits, self.data["actions"][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite motor loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    def payload(self):
        return {"kind": "motor_lab_v1", "contract": MOTOR_CONTRACT, "model": self.model.state_dict(),
                "digest": motor_digest(self.model.state_dict()), "optimizer": self.optimizer.state_dict(),
                "data": self.data, "rng": self.rng.bit_generator.state, "seed": self.seed,
                "cases": self.cases, "updates": self.updates, "last_loss": self.last_loss,
                "teacher": "development_reference_motor", "origin": "fresh_motor_not_legacy_weights"}


def evaluate_motor(model, training_seed, episodes_per_skill=100):
    if episodes_per_skill < 1:
        raise ValueError("Episode count must be positive")
    evaluation_seed = training_seed + 1000003
    rng = random.Random(evaluation_seed)
    env = DevelopmentRoom(seed=evaluation_seed, max_episode_steps=80)
    report = {"protocol": "held_out_visible_clear_corridor_v1", "motor_digest": motor_digest(model.state_dict()),
              "training_seed": training_seed, "evaluation_seed": evaluation_seed,
              "teacher_actions": 0, "skills": {},
              "scope": "local orientation and clear-corridor movement, not arbitrary navigation"}
    model.eval()
    try:
        for skill in ("orient", "navigate"):
            orient = skill == "orient"
            counts = Counter()
            successes, successful_advances, faults = 0, 0, 0
            for _ in range(episodes_per_skill):
                obs = setup_case(env, rng)
                moved = 0
                for _ in range(48):
                    action = model.action(obs, orient_only=orient)
                    counts[env.action_names[action]] += 1
                    obs, _, done, info = env.step(action)
                    if info.get("infrastructure_fault") or not np.isfinite(obs).all():
                        faults += 1
                        break
                    if action == 1 and info["last_action_success"]:
                        moved += 1
                        successful_advances += 1
                    facts = sensed_facts(obs)
                    if ((orient and "aligned" in facts) or
                            (not orient and "near" in facts and moved > 0)):
                        successes += 1
                        break
                    if done:
                        break
            report["skills"][skill] = {"episodes": episodes_per_skill, "successes": successes,
                "success_rate": successes / episodes_per_skill, "actions": dict(counts),
                "successful_advances": successful_advances, "faults": faults}
    finally:
        env.close()
    report["passed"] = (episodes_per_skill >= 100 and all(
        score["success_rate"] >= 0.95 and score["faults"] == 0 for score in report["skills"].values())
        and report["skills"]["navigate"]["successful_advances"] > 0)
    return report


def load_qualified(directory):
    """Caller must hold MotorLabLock. Qualification is invalid after more training."""
    directory = Path(directory)
    candidate_path = directory / "qualified_motor.pt"
    if not candidate_path.is_file():
        raise ValueError("No qualified motor: run the teacher-free motor evaluation first")
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    check_qualification(candidate)
    current = torch.load(directory / "motor_training.pt", map_location="cpu", weights_only=False)
    if current.get("kind") != "motor_lab_v1" or motor_digest(current["model"]) != candidate["digest"]:
        raise ValueError("Qualification is stale after further training; evaluate the current weights again")
    return candidate
