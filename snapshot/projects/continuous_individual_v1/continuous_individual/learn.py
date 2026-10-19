"""Training and matched evaluation."""
from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim

from .agent import Individual, obs_to_tensor
from .store import CanonicalStore
from .world import SurvivalWorld


def run_episode(individual: Individual, world: SurvivalWorld, use_learned: bool, train: bool = False) -> Dict:
    # Fresh map for each life (world is new)
    individual.store.write("map", {"water": [], "food": None, "visited": []})
    obs = world.reset()
    total_reward = 0.0
    steps = 0
    transitions = []
    while True:
        action = individual.act(obs, use_learned=use_learned)
        next_obs, reward, done, info = world.step(action)
        total_reward += reward
        steps += 1
        if train:
            # Record the high-level choice that was made
            choice = individual.learned_choice(obs) if use_learned else "explore"
            transitions.append((obs, choice, reward))
        obs = next_obs
        if done:
            break
    return {
        "steps": steps,
        "reward": total_reward,
        "cause": info.get("cause"),
        "transitions": transitions,
    }


def train_controller(individual: Individual, n_episodes: int = 300, seed: int = 0) -> List[float]:
    opt = optim.Adam(individual.controller.parameters(), lr=3e-4)
    returns = []
    rng = random.Random(seed)
    for ep in range(n_episodes):
        world = SurvivalWorld(seed=rng.randint(0, 10_000_000))
        result = run_episode(individual, world, use_learned=True, train=True)
        returns.append(result["reward"])
        if not result["transitions"]:
            continue
        loss = torch.tensor(0.0, requires_grad=True)
        G = 0.0
        for obs, choice, reward in reversed(result["transitions"]):
            G = reward + 0.98 * G
            logits = individual.controller(obs_to_tensor(obs).unsqueeze(0))
            idx = {"explore": 0, "seek_water": 1, "seek_food": 2}[choice]
            logp = F.log_softmax(logits, dim=-1)[0, idx]
            loss = loss - logp * (G / 10.0)  # scale for stability
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(individual.controller.parameters(), 1.0)
        opt.step()
        individual._save_weights()
    return returns


def evaluate(n_worlds: int = 40, seed_start: int = 50000, use_learned: bool = False, store: CanonicalStore | None = None) -> Dict:
    individual = Individual(store=store)
    survivals, scores, need_deaths = [], [], 0
    for i in range(n_worlds):
        world = SurvivalWorld(seed=seed_start + i)
        result = run_episode(individual, world, use_learned=use_learned)
        survivals.append(result["steps"])
        scores.append(result["reward"])
        if result["cause"] in ("thirst", "hunger"):
            need_deaths += 1
    return {
        "mean_survival": sum(survivals) / len(survivals),
        "mean_score": sum(scores) / len(scores),
        "need_death_rate": need_deaths / n_worlds,
        "n": n_worlds,
    }
