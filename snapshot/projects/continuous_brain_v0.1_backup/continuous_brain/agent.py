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
from torch import nn

from .config import BrainConfig
from .environment import Environment
from .memory import EpisodicMemory, MemoryEvent, WorkingMemory
from .model import ContinuousBrainNet
from .ollama import OllamaPlanner
from .replay import ReplayBuffer, Transition


class ContinuousAgent:
    def __init__(
        self,
        env: Environment,
        config: BrainConfig,
        state_dir: str | Path,
        device: str = "auto",
        ollama_model: str | None = None,
        planner_every: int = 50,
    ):
        self.env = env
        self.cfg = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(device)
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.net = ContinuousBrainNet(
            env.observation_dim, env.action_dim, config.embed_dim, config.hidden_dim
        ).to(self.device)
        self.target_net = ContinuousBrainNet(
            env.observation_dim, env.action_dim, config.embed_dim, config.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.learning_rate)

        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
        self.working = WorkingMemory(128)
        self.episodic = EpisodicMemory(self.state_dir / "episodic.sqlite3")
        self.ollama = OllamaPlanner(ollama_model) if ollama_model else None
        self.planner_every = max(1, planner_every)

        self.total_steps = 0
        self.total_reward = 0.0
        self.goals = 0
        self.last_loss = float("nan")
        self.running = True
        self._load_checkpoint_if_present()

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _onehot(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(actions.long(), num_classes=self.env.action_dim).float()

    def epsilon(self) -> float:
        t = min(1.0, self.total_steps / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def _neural_action(self, hidden: torch.Tensor) -> int:
        if self.rng.random() < self.epsilon():
            return self.rng.randrange(self.env.action_dim)
        return int(torch.argmax(self.net.q_values(hidden), dim=-1).item())

    def _learn(self) -> None:
        batch = self.replay.sample(self.cfg.batch_size)
        if len(batch) < max(8, self.cfg.batch_size // 4):
            return
        obs = torch.as_tensor(np.stack([t.observation for t in batch]), device=self.device)
        prev_hidden = torch.as_tensor(np.stack([t.prev_hidden for t in batch]), device=self.device)
        prev_actions = torch.tensor([t.prev_action for t in batch], device=self.device, dtype=torch.long)
        next_obs = torch.as_tensor(np.stack([t.next_observation for t in batch]), device=self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)
        prev_a1 = self._onehot(prev_actions)
        a1 = self._onehot(actions)

        # Reconstruct the recurrent state with gradients. This means online replay
        # trains perception and the GRU core, not just the output heads.
        hidden = self.net.encode_step(obs, prev_a1, prev_hidden)
        pred_obs, pred_reward = self.net.predict(hidden, a1)
        world_loss = nn.functional.smooth_l1_loss(pred_obs, next_obs)
        reward_loss = nn.functional.smooth_l1_loss(pred_reward, rewards)

        q = self.net.q_values(hidden).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target_hidden = self.target_net.encode_step(next_obs, a1, hidden.detach())
            next_q = self.target_net.q_values(target_hidden).max(dim=1).values
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        q_loss = nn.functional.smooth_l1_loss(q, td_target)

        loss = world_loss + 0.5 * reward_loss + q_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu().item())

        if self.total_steps % 200 == 0:
            tau = 0.02
            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(), self.net.parameters()):
                    tp.mul_(1.0 - tau).add_(p, alpha=tau)

    def _consolidate(self) -> None:
        # Replay several extra minibatches at a slower timescale, analogous to memory consolidation.
        for _ in range(4):
            self._learn()

    def _checkpoint_path(self) -> Path:
        return self.state_dir / "brain.pt"

    def save(self) -> None:
        payload = {
            "model": self.net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.total_steps,
            "reward": self.total_reward,
            "goals": self.goals,
            "config": asdict(self.cfg),
        }
        torch.save(payload, self._checkpoint_path())
        (self.state_dir / "status.json").write_text(json.dumps({
            "steps": self.total_steps,
            "reward": self.total_reward,
            "goals": self.goals,
            "episodic_memories": self.episodic.count(),
            "last_loss": self.last_loss,
            "device": str(self.device),
        }, indent=2))

    def _load_checkpoint_if_present(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data["model"])
        self.target_net.load_state_dict(data.get("target", data["model"]))
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        self.total_steps = int(data.get("steps", 0))
        self.total_reward = float(data.get("reward", 0.0))
        self.goals = int(data.get("goals", 0))

    def stop(self, *_args) -> None:
        self.running = False

    def run(self, max_steps: int = 0, realtime: bool = True, log_every: int = 50) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        obs_np = self.env.reset()
        hidden = self.net.initial_state(1, self.device)
        prev_action = 0
        tick_period = 1.0 / max(0.1, self.cfg.brain_hz)
        episode_reward = 0.0

        try:
            while self.running and (max_steps <= 0 or self.total_steps < max_steps):
                tick_start = time.perf_counter()
                obs = torch.as_tensor(obs_np, device=self.device).float().unsqueeze(0)
                prev_a = self._onehot(torch.tensor([prev_action], device=self.device))
                with torch.no_grad():
                    next_hidden = self.net.encode_step(obs, prev_a, hidden)

                action = self._neural_action(next_hidden)
                if self.ollama and self.total_steps % self.planner_every == 0:
                    memories = self.episodic.retrieve(next_hidden.squeeze(0).cpu().numpy(), k=3)
                    planned = self.ollama.choose_action(self.env.describe(), self.env.action_names, memories)
                    if planned is not None:
                        action = planned

                action_t = torch.tensor([action], device=self.device)
                action_1h = self._onehot(action_t)
                with torch.no_grad():
                    predicted_obs, _ = self.net.predict(next_hidden, action_1h)

                actual_next_np, reward, done, info = self.env.step(action)
                actual_next = torch.as_tensor(actual_next_np, device=self.device).float().unsqueeze(0)
                surprise = float(nn.functional.mse_loss(predicted_obs, actual_next).cpu().item())

                next_obs_t = torch.as_tensor(actual_next_np, device=self.device).float().unsqueeze(0)
                with torch.no_grad():
                    next_next_hidden = self.net.encode_step(next_obs_t, action_1h, next_hidden)

                transition = Transition(
                    observation=obs_np.copy(),
                    prev_hidden=hidden.squeeze(0).cpu().numpy().copy(),
                    prev_action=prev_action,
                    action=action,
                    reward=reward,
                    next_observation=actual_next_np.copy(),
                    done=done,
                )
                self.replay.append(transition)
                self.working.append(MemoryEvent(self.total_steps, reward, surprise, action))
                if surprise >= self.cfg.memory_surprise_threshold or abs(reward) > 0.25:
                    self.episodic.store(
                        step=self.total_steps,
                        reward=reward,
                        surprise=surprise,
                        action=action,
                        latent=next_hidden.squeeze(0).cpu().numpy(),
                        metadata={"environment": self.env.describe(), "info": info},
                    )

                self.total_steps += 1
                self.total_reward += reward
                episode_reward += reward
                if bool(info.get("reached")):
                    self.goals += 1

                if self.total_steps % self.cfg.learn_every == 0:
                    self._learn()
                if self.total_steps % self.cfg.consolidate_every == 0:
                    self._consolidate()
                if self.total_steps % self.cfg.checkpoint_every == 0:
                    self.save()

                if self.total_steps % log_every == 0:
                    loss_text = "n/a" if math.isnan(self.last_loss) else f"{self.last_loss:.4f}"
                    print(
                        f"step={self.total_steps:7d} eps={self.epsilon():.3f} "
                        f"reward={self.total_reward:+.2f} episode={episode_reward:+.2f} "
                        f"goals={self.goals} replay={len(self.replay)} memories={self.episodic.count()} "
                        f"loss={loss_text} {self.env.describe()}",
                        flush=True,
                    )

                if done:
                    obs_np = self.env.reset()
                    hidden = self.net.initial_state(1, self.device)
                    prev_action = 0
                    episode_reward = 0.0
                else:
                    obs_np = actual_next_np
                    hidden = next_hidden.detach()
                    prev_action = action

                if realtime:
                    elapsed = time.perf_counter() - tick_start
                    time.sleep(max(0.0, tick_period - elapsed))
        finally:
            self.save()
            self.episodic.close()
