from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import signal
import shutil
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
from .visualizer import LiveVisualizer
from .vision import NullVisionBackbone, VisionBackbone


class ContinuousAgent:
    def __init__(
        self,
        env: Environment,
        config: BrainConfig,
        state_dir: str | Path,
        device: str = "auto",
        ollama_model: str | None = None,
        planner_every: int = 50,
        visualizer: LiveVisualizer | None = None,
        vision_backbone: VisionBackbone | None = None,
        vision_every: int = 1,
        bootstrap_checkpoint: str | Path | None = None,
    ):
        self.env = env
        self.cfg = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(device)
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.vision = vision_backbone or NullVisionBackbone()
        self.vision_every = max(1, int(vision_every))
        visual_dim = int(self.vision.output_dim)

        self.net = ContinuousBrainNet(
            env.observation_dim, env.action_dim, config.embed_dim, config.hidden_dim, visual_dim=visual_dim
        ).to(self.device)
        self.target_net = ContinuousBrainNet(
            env.observation_dim, env.action_dim, config.embed_dim, config.hidden_dim, visual_dim=visual_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.learning_rate)

        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
        self.working = WorkingMemory(128)
        self.episodic = EpisodicMemory(self.state_dir / "episodic.sqlite3")
        self.ollama = OllamaPlanner(ollama_model) if ollama_model else None
        self.planner_every = max(1, planner_every)
        self.visualizer = visualizer

        self.total_steps = 0
        self.total_reward = 0.0
        self.total_intrinsic_reward = 0.0
        self.goals = 0
        self.last_loss = float("nan")
        self.last_surprise = 0.0
        self.last_visual_novelty = 0.0
        self.last_curiosity = 0.0
        self.last_homeostatic_drive = 0.0
        self.mastered_levels: set[int] = set()
        self.running = True
        self._load_checkpoint_if_present()
        if self.total_steps == 0 and bootstrap_checkpoint is not None and not self._checkpoint_path().exists():
            self._bootstrap_from_checkpoint(Path(bootstrap_checkpoint))
        self._load_replay_if_present()

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _onehot(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(actions.long(), num_classes=self.env.action_dim).float()

    def _visual_tensor(self, vectors: list[np.ndarray | None]) -> torch.Tensor | None:
        if self.vision.output_dim <= 0:
            return None
        z = np.zeros(self.vision.output_dim, dtype=np.float32)
        arr = np.stack([z if v is None else np.asarray(v, dtype=np.float32) for v in vectors])
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

    def _encode_vision(self) -> np.ndarray | None:
        if self.vision.output_dim <= 0:
            return None
        frame = self.env.render_rgb()
        vec = self.vision.encode(frame)
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size != self.vision.output_dim:
            raise RuntimeError(
                f"vision backbone {self.vision.name} returned {vec.size} values; expected {self.vision.output_dim}"
            )
        return vec

    @staticmethod
    def _visual_novelty(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        an = float(np.linalg.norm(a)) + 1e-8
        bn = float(np.linalg.norm(b)) + 1e-8
        cosine = float(np.dot(a, b) / (an * bn))
        return float(np.clip(1.0 - cosine, 0.0, 2.0))

    def epsilon(self) -> float:
        t = min(1.0, self.total_steps / max(1, self.cfg.epsilon_decay_steps))
        global_epsilon = self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        try:
            curriculum_floor = float(self.env.exploration_floor())
        except Exception:
            curriculum_floor = 0.0
        return max(float(global_epsilon), max(0.0, curriculum_floor))

    def _valid_actions(self) -> tuple[int, ...]:
        try:
            valid = tuple(int(x) for x in self.env.valid_action_indices())
        except Exception:
            valid = tuple(range(self.env.action_dim))
        valid = tuple(x for x in valid if 0 <= x < self.env.action_dim)
        return valid or tuple(range(self.env.action_dim))

    def _masked_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        valid = set(self._valid_actions())
        if len(valid) >= self.env.action_dim:
            return q_values
        masked = q_values.clone()
        invalid = [i for i in range(self.env.action_dim) if i not in valid]
        if invalid:
            masked[..., invalid] = torch.finfo(masked.dtype).min
        return masked

    @torch.no_grad()
    def _neural_action(self, hidden: torch.Tensor) -> int:
        valid = self._valid_actions()
        if self.rng.random() < self.epsilon():
            return self.rng.choice(valid)
        q = self._masked_q_values(self.net.q_values(hidden))
        return int(torch.argmax(q, dim=-1).item())

    def _learn(self) -> None:
        try:
            current_level = self.env.current_skill_level()
        except Exception:
            current_level = None
        batch = self.replay.sample(
            self.cfg.batch_size, current_level=current_level,
            current_fraction=0.80 if current_level is not None and current_level > 0 else 1.0,
        )
        if len(batch) < max(8, self.cfg.batch_size // 4):
            return
        obs = torch.as_tensor(np.stack([t.observation for t in batch]), device=self.device)
        prev_hidden = torch.as_tensor(np.stack([t.prev_hidden for t in batch]), device=self.device)
        prev_actions = torch.tensor([t.prev_action for t in batch], device=self.device, dtype=torch.long)
        next_obs = torch.as_tensor(np.stack([t.next_observation for t in batch]), device=self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)
        visual = self._visual_tensor([t.visual_embedding for t in batch])
        next_visual = self._visual_tensor([t.next_visual_embedding for t in batch])
        prev_a1 = self._onehot(prev_actions)
        a1 = self._onehot(actions)

        hidden = self.net.encode_step(obs, prev_a1, prev_hidden, visual=visual)
        pred_obs, pred_reward = self.net.predict(hidden, a1)
        world_loss = nn.functional.smooth_l1_loss(pred_obs, next_obs)
        reward_loss = nn.functional.smooth_l1_loss(pred_reward, rewards)

        q = self.net.q_values(hidden).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target_hidden = self.target_net.encode_step(next_obs, a1, hidden.detach(), visual=next_visual)
            raw_next_q = self.target_net.q_values(target_hidden)
            masked_next_q = raw_next_q.clone()
            for row, transition in enumerate(batch):
                valid = transition.valid_actions
                if not valid:
                    try:
                        valid = self.env.valid_action_indices_for_level(transition.skill_level)
                    except Exception:
                        valid = tuple(range(self.env.action_dim))
                valid_set = {int(x) for x in valid}
                invalid = [i for i in range(self.env.action_dim) if i not in valid_set]
                if invalid:
                    masked_next_q[row, invalid] = torch.finfo(masked_next_q.dtype).min
            next_q = masked_next_q.max(dim=1).values
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
        for _ in range(4):
            self._learn()

    def _checkpoint_path(self) -> Path:
        return self.state_dir / "brain.pt"

    def _replay_checkpoint_path(self) -> Path:
        return self.state_dir / "replay.pt"

    @staticmethod
    def _atomic_torch_save(payload: object, path: Path, *, backup: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        if backup is not None and path.exists():
            try:
                shutil.copy2(path, backup)
            except OSError:
                pass
        tmp.replace(path)

    def _save_replay(self) -> None:
        limit = max(0, int(getattr(self.cfg, "replay_checkpoint_items", 0)))
        if limit <= 0:
            return
        payload = {
            "format_version": 1,
            "steps": int(self.total_steps),
            "signature": {
                "observation_dim": int(self.env.observation_dim),
                "action_dim": int(self.env.action_dim),
                "hidden_dim": int(self.cfg.hidden_dim),
                "vision_dim": int(self.vision.output_dim),
            },
            "buffer": self.replay.state_dict(limit=limit),
        }
        self._atomic_torch_save(payload, self._replay_checkpoint_path())

    def _load_replay_if_present(self) -> None:
        path = self._replay_checkpoint_path()
        if not path.exists():
            return
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            sig = dict(data.get("signature", {})) if isinstance(data, dict) else {}
            expected = {
                "observation_dim": int(self.env.observation_dim),
                "action_dim": int(self.env.action_dim),
                "hidden_dim": int(self.cfg.hidden_dim),
                "vision_dim": int(self.vision.output_dim),
            }
            if sig and any(int(sig.get(k, expected[k])) != expected[k] for k in expected):
                print(f"Replay restore: skipped incompatible replay signature {sig}; expected {expected}.", flush=True)
                return
            loaded = self.replay.load_state_dict(data.get("buffer", data))
            if loaded:
                print(f"Replay restore: loaded {loaded} persistent transitions.", flush=True)
        except Exception as exc:
            print(f"Replay restore: ignored unreadable replay checkpoint ({exc}).", flush=True)

    def _skill_snapshot_path(self, level: int) -> Path:
        return self.state_dir / "skills" / f"L{int(level)}_mastered.pt"

    def _save_skill_snapshot(self, level: int, stats: dict) -> None:
        level = int(level)
        path = self._skill_snapshot_path(level)
        payload = {
            "format_version": 6,
            "version": "0.4.3",
            "skill_level": level,
            "model": self.net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.total_steps,
            "reward": self.total_reward,
            "intrinsic_reward": self.total_intrinsic_reward,
            "goals": self.goals,
            "mastered_levels": sorted(self.mastered_levels | {level}),
            "curriculum_stats": dict(stats),
            "config": asdict(self.cfg),
            "vision": self.vision.describe(),
            "environment_state": self.env.state_dict(),
            "replay": self.replay.state_dict(limit=min(4000, max(0, int(getattr(self.cfg, "replay_checkpoint_items", 0))))),
        }
        self._atomic_torch_save(payload, path)
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({
            "version": "0.4.3",
            "skill_level": level,
            "steps": self.total_steps,
            "goals": self.goals,
            "curriculum_stats": stats,
        }, indent=2))
        self.mastered_levels.add(level)
        print(f"[skill-mastered] L{level} snapshot saved: {path}", flush=True)

    def _maybe_snapshot_mastered(self, info: dict) -> None:
        if not bool(info.get("graduation_ready", False)):
            return
        configured = info.get("configured_curriculum_level")
        episode = info.get("curriculum_level")
        if configured is None or episode is None or int(configured) != int(episode):
            return
        level = int(configured)
        if level in self.mastered_levels or self._skill_snapshot_path(level).exists():
            self.mastered_levels.add(level)
            return
        stats = info.get("configured_curriculum_stats", info.get("curriculum_stats", {})) or {}
        self._save_skill_snapshot(level, stats)

    def save(self) -> None:
        payload = {
            "format_version": 6,
            "version": "0.4.3",
            "model": self.net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.total_steps,
            "reward": self.total_reward,
            "intrinsic_reward": self.total_intrinsic_reward,
            "goals": self.goals,
            "mastered_levels": sorted(self.mastered_levels),
            "config": asdict(self.cfg),
            "vision": self.vision.describe(),
            "environment_state": self.env.state_dict(),
        }
        self._atomic_torch_save(
            payload, self._checkpoint_path(), backup=self.state_dir / "brain.previous.pt"
        )
        self._save_replay()
        (self.state_dir / "status.json").write_text(json.dumps({
            "version": "0.4.3",
            "steps": self.total_steps,
            "reward": self.total_reward,
            "intrinsic_reward": self.total_intrinsic_reward,
            "goals": self.goals,
            "mastered_levels": sorted(self.mastered_levels),
            "replay_persisted": min(len(self.replay), max(0, int(getattr(self.cfg, "replay_checkpoint_items", 0)))),
            "episodic_memories": self.episodic.count(),
            "visual_memories": self.episodic.visual_count(),
            "last_loss": self.last_loss,
            "last_surprise": self.last_surprise,
            "last_visual_novelty": self.last_visual_novelty,
            "last_curiosity": self.last_curiosity,
            "homeostatic_drive": self.last_homeostatic_drive,
            "device": str(self.device),
            "vision": self.vision.describe(),
            "visual_gate": self.net.visual_gate(),
            "environment_state": self.env.state_dict(),
        }, indent=2))

    def _load_checkpoint_if_present(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            backup = self.state_dir / "brain.previous.pt"
            if not backup.exists():
                raise
            print(f"Checkpoint recovery: brain.pt unreadable ({exc}); loading brain.previous.pt.", flush=True)
            data = torch.load(backup, map_location=self.device, weights_only=False)

        def load_compatible(module: nn.Module, incoming: dict[str, torch.Tensor]):
            current = module.state_dict()
            filtered = {}
            ignored = []
            for key, value in incoming.items():
                if key in current and tuple(current[key].shape) == tuple(value.shape):
                    filtered[key] = value
                else:
                    ignored.append(key)
            result = module.load_state_dict(filtered, strict=False)
            return result, ignored

        model_result, shape_ignored = load_compatible(self.net, data["model"])
        _target_result, target_ignored = load_compatible(
            self.target_net, data.get("target", data["model"])
        )
        ignored_count = len(set(shape_ignored + target_ignored))
        if model_result.missing_keys or ignored_count:
            print(
                "Checkpoint migration: preserved compatible learned weights; "
                f"new_or_changed={len(model_result.missing_keys)} ignored={ignored_count}",
                flush=True,
            )
        if "optimizer" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except (ValueError, KeyError):
                print("Checkpoint migration: optimizer state reset for changed parameters.", flush=True)
        self.total_steps = int(data.get("steps", 0))
        self.total_reward = float(data.get("reward", 0.0))
        self.total_intrinsic_reward = float(data.get("intrinsic_reward", 0.0))
        self.goals = int(data.get("goals", 0))
        self.mastered_levels = {int(x) for x in data.get("mastered_levels", [])}
        if "replay" in data and not self._replay_checkpoint_path().exists():
            try:
                loaded = self.replay.load_state_dict(data.get("replay"))
                if loaded:
                    print(f"Checkpoint recovery: restored {loaded} embedded replay transitions.", flush=True)
            except Exception:
                pass
        try:
            self.env.load_state_dict(data.get("environment_state", {}))
        except Exception as exc:
            print(f"Checkpoint migration: environment statistics reset ({exc}).", flush=True)

    def _bootstrap_from_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"bootstrap checkpoint not found: {path}")
        data = torch.load(path, map_location=self.device, weights_only=False)
        incoming = data.get("model", data)
        current = self.net.state_dict()
        filtered = {
            key: value for key, value in incoming.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        result = self.net.load_state_dict(filtered, strict=False)
        self.target_net.load_state_dict(self.net.state_dict())
        print(
            "Bootstrap checkpoint: imported compatible neural weights only; "
            f"loaded={len(filtered)} new_or_changed={len(result.missing_keys)}. "
            "Counters, optimizer, replay and episodic memory start fresh in this branch.",
            flush=True,
        )

    def stop(self, *_args) -> None:
        self.running = False

    def run(self, max_steps: int = 0, realtime: bool = True, log_every: int = 50) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        obs_np = self.env.reset()
        current_visual = self._encode_vision()
        hidden = self.net.initial_state(1, self.device)
        prev_action = 0
        tick_period = 1.0 / max(0.1, self.cfg.brain_hz)
        episode_reward = 0.0

        try:
            while self.running and (max_steps <= 0 or self.total_steps < max_steps):
                tick_start = time.perf_counter()
                obs = torch.as_tensor(obs_np, device=self.device).float().unsqueeze(0)
                prev_a = self._onehot(torch.tensor([prev_action], device=self.device))
                visual_t = self._visual_tensor([current_visual])
                with torch.no_grad():
                    next_hidden = self.net.encode_step(obs, prev_a, hidden, visual=visual_t)

                action = self._neural_action(next_hidden)
                if self.ollama and self.total_steps % self.planner_every == 0:
                    memories = self.episodic.retrieve(next_hidden.squeeze(0).cpu().numpy(), k=3)
                    valid = self._valid_actions()
                    planner_names = tuple(self.env.action_names[i] for i in valid)
                    planned_local = self.ollama.choose_action(self.env.describe(), planner_names, memories)
                    if planned_local is not None and 0 <= planned_local < len(valid):
                        action = valid[planned_local]

                action_t = torch.tensor([action], device=self.device)
                action_1h = self._onehot(action_t)
                with torch.no_grad():
                    predicted_obs, predicted_reward = self.net.predict(next_hidden, action_1h)
                    q_values = self.net.q_values(next_hidden)

                actual_next_np, extrinsic_reward, done, info = self.env.step(action)
                actual_next = torch.as_tensor(actual_next_np, device=self.device).float().unsqueeze(0)
                surprise = float(nn.functional.mse_loss(predicted_obs, actual_next).cpu().item())

                should_refresh_vision = self.vision.output_dim > 0 and ((self.total_steps + 1) % self.vision_every == 0)
                next_visual = self._encode_vision() if should_refresh_vision else current_visual
                visual_novelty = self._visual_novelty(current_visual, next_visual)
                curiosity_signal = min(self.cfg.curiosity_cap, max(0.0, surprise) + visual_novelty)
                intrinsic_reward = self.cfg.curiosity_weight * curiosity_signal
                energy = float(getattr(self.env, "energy", 1.0))
                homeostatic_drive = float(np.clip(1.0 - energy, 0.0, 1.0))
                homeostatic_cost = self.cfg.homeostasis_weight * homeostatic_drive
                learning_reward = float(extrinsic_reward + intrinsic_reward - homeostatic_cost)

                next_obs_t = torch.as_tensor(actual_next_np, device=self.device).float().unsqueeze(0)
                next_visual_t = self._visual_tensor([next_visual])
                with torch.no_grad():
                    next_next_hidden = self.net.encode_step(next_obs_t, action_1h, next_hidden, visual=next_visual_t)

                transition = Transition(
                    observation=obs_np.copy(),
                    prev_hidden=hidden.squeeze(0).cpu().numpy().copy(),
                    prev_action=prev_action,
                    action=action,
                    reward=learning_reward,
                    next_observation=actual_next_np.copy(),
                    done=done,
                    visual_embedding=None if current_visual is None else current_visual.copy(),
                    next_visual_embedding=None if next_visual is None else next_visual.copy(),
                    skill_level=int(info.get("curriculum_level")) if info.get("curriculum_level") is not None else None,
                    valid_actions=tuple(int(x) for x in info.get("allowed_actions", self._valid_actions())),
                )
                self.replay.append(transition)
                self.working.append(MemoryEvent(
                    self.total_steps, float(extrinsic_reward), surprise, action,
                    note=f"intrinsic={intrinsic_reward:.5f};drive={homeostatic_drive:.4f}",
                ))

                important = surprise >= self.cfg.memory_surprise_threshold or abs(extrinsic_reward) > 0.25
                metadata = {
                    "environment": self.env.describe(),
                    "info": info,
                    "intrinsic_reward": intrinsic_reward,
                    "homeostatic_drive": homeostatic_drive,
                    "vision_backbone": self.vision.name,
                }
                if important:
                    self.episodic.store(
                        step=self.total_steps,
                        reward=float(extrinsic_reward),
                        surprise=surprise,
                        action=action,
                        latent=next_hidden.squeeze(0).cpu().numpy(),
                        metadata=metadata,
                    )
                if next_visual is not None and (
                    important or visual_novelty >= self.cfg.visual_memory_threshold
                ):
                    self.episodic.store_visual(
                        step=self.total_steps,
                        reward=float(extrinsic_reward),
                        novelty=visual_novelty,
                        action=action,
                        backbone=self.vision.name,
                        latent=next_visual,
                        metadata=metadata,
                    )

                self.total_steps += 1
                self.total_reward += float(extrinsic_reward)
                self.total_intrinsic_reward += float(intrinsic_reward)
                episode_reward += float(extrinsic_reward)
                self.last_surprise = surprise
                self.last_visual_novelty = visual_novelty
                self.last_curiosity = intrinsic_reward
                self.last_homeostatic_drive = homeostatic_drive
                if bool(info.get("reached")):
                    self.goals += 1

                if self.total_steps % self.cfg.learn_every == 0:
                    self._learn()
                if self.total_steps % self.cfg.consolidate_every == 0:
                    self._consolidate()
                self._maybe_snapshot_mastered(info)
                if self.total_steps % self.cfg.checkpoint_every == 0:
                    self.save()

                if self.visualizer is not None:
                    predicted_np = predicted_obs.squeeze(0).detach().cpu().numpy()
                    hidden_np = next_hidden.squeeze(0).detach().cpu().numpy()
                    q_np = q_values.squeeze(0).detach().cpu().numpy()
                    snapshot = {
                        "step": self.total_steps,
                        "action": int(action),
                        "action_name": self.env.action_names[action],
                        "action_names": list(self.env.action_names),
                        "reward": float(extrinsic_reward),
                        "learning_reward": learning_reward,
                        "intrinsic_reward": float(intrinsic_reward),
                        "total_intrinsic_reward": float(self.total_intrinsic_reward),
                        "predicted_reward": float(predicted_reward.item()),
                        "total_reward": float(self.total_reward),
                        "episode_reward": float(episode_reward),
                        "goals": int(self.goals),
                        "epsilon": float(self.epsilon()),
                        "loss": None if math.isnan(self.last_loss) else float(self.last_loss),
                        "surprise": float(surprise),
                        "visual_novelty": float(visual_novelty),
                        "homeostatic_drive": float(homeostatic_drive),
                        "replay": len(self.replay),
                        "memories": self.episodic.count(),
                        "visual_memories": self.episodic.visual_count(),
                        "vision_backbone": self.vision.name,
                        "vision_pretrained": bool(self.vision.pretrained),
                        "visual_gate": self.net.visual_gate(),
                        "energy": energy,
                        "distance": float(info.get("distance", 0.0)),
                        "final_distance": float(info.get("final_distance", info.get("distance", 0.0))),
                        "bearing": float(info.get("bearing", 0.0)),
                        "final_bearing": float(info.get("final_bearing", info.get("bearing", 0.0))),
                        "task_kind": info.get("task_kind"),
                        "target_label": info.get("target_label"),
                        "curriculum_level": info.get("curriculum_level"),
                        "configured_curriculum_level": info.get("configured_curriculum_level", info.get("curriculum_level")),
                        "is_rehearsal": bool(info.get("is_rehearsal", False)),
                        "primitive_action": info.get("primitive_action"),
                        "allowed_actions": info.get("allowed_actions", list(self._valid_actions())),
                        "delta_bearing": float(info.get("delta_bearing", 0.0)),
                        "delta_distance": float(info.get("delta_distance", 0.0)),
                        "actual_move_distance": float(info.get("actual_move_distance", 0.0)),
                        "depth_center_m": float(info.get("depth_center_m", 0.0)),
                        "blocked_advance": bool(info.get("blocked_advance", False)),
                        "successful_translation": bool(info.get("successful_translation", False)),
                        "successful_translations_episode": int(info.get("successful_translations_episode", 0)),
                        "blocked_advances_episode": int(info.get("blocked_advances_episode", 0)),
                        "route_replanned": bool(info.get("route_replanned", False)),
                        "replans_episode": int(info.get("replans_episode", 0)),
                        "known_blocked_edges": int(info.get("known_blocked_edges", 0)),
                        "route_unavailable": bool(info.get("route_unavailable", False)),
                        "mastered_levels": sorted(self.mastered_levels),
                        "waypoint": info.get("waypoint"),
                        "final_target": info.get("final_target"),
                        "waypoint_index": int(info.get("waypoint_index", 0)),
                        "waypoint_count": int(info.get("waypoint_count", 0)),
                        "path_steps": int(info.get("path_steps", 0)),
                        "level_epsilon_floor": float(info.get("level_epsilon_floor", 0.0)),
                        "stagnation_steps": int(info.get("stagnation_steps", 0)),
                        "stagnation_limit": int(info.get("stagnation_limit", 0)),
                        "curriculum_stats": info.get("configured_curriculum_stats", info.get("curriculum_stats", {})),
                        "episode_curriculum_stats": info.get("curriculum_stats", {}),
                        "replay_level_counts": {str(k): int(v) for k, v in self.replay.level_counts().items()},
                        "graduation_ready": bool(info.get("graduation_ready", False)),
                        "q_values": [float(x) for x in q_np.tolist()],
                        "hidden_sample": [float(x) for x in hidden_np[:32].tolist()],
                        "hidden_norm": float(np.linalg.norm(hidden_np)),
                        "reached": bool(info.get("reached", False)),
                    }
                    try:
                        snapshot.update(self.env.visual_snapshot(predicted_np))
                    except Exception as exc:
                        snapshot["visual_snapshot_error"] = str(exc)
                    self.visualizer.update(snapshot)

                if self.total_steps % log_every == 0:
                    loss_text = "n/a" if math.isnan(self.last_loss) else f"{self.last_loss:.4f}"
                    action_name = self.env.action_names[action]
                    primitive = str(info.get("primitive_action", "–"))
                    if info.get("task_kind") == "orient":
                        delta_text = f"dBearing={float(info.get('delta_bearing', 0.0)):+.1f}"
                    else:
                        delta_text = (
                            f"dDist={float(info.get('delta_distance', 0.0)):+.3f} "
                            f"dBearing={float(info.get('delta_bearing', 0.0)):+.1f} "
                            f"move={float(info.get('actual_move_distance', 0.0)):.3f} "
                            f"depth={float(info.get('depth_center_m', 0.0)):.2f} "
                            f"ok={int(bool(info.get('last_action_success', True)))}"
                        )
                    stats = info.get("configured_curriculum_stats", info.get("curriculum_stats", {})) or {}
                    success_rate = float(stats.get("recent_success_rate", 0.0)) * 100.0
                    median_steps = stats.get("median_success_steps")
                    median_text = "n/a" if median_steps is None else f"{float(median_steps):.1f}"
                    configured_level = int(info.get("configured_curriculum_level", info.get("curriculum_level", 0)))
                    episode_level = int(info.get("curriculum_level", configured_level))
                    rehearsal_text = f" rehearse=L{episode_level}" if info.get("is_rehearsal") else ""
                    wp_count = int(info.get("waypoint_count", 0))
                    wp_text = f" wp={int(info.get('waypoint_index', 0))+1}/{wp_count}" if wp_count else ""
                    skill_goals = int(stats.get("goals_total", 0))
                    diag = ""
                    if configured_level == 1:
                        diag = (
                            f" L1goals={skill_goals} trans={int(stats.get('successful_translations_total', 0))} "
                            f"blocked={int(stats.get('blocked_advances_total', 0))} "
                            f"replans={int(stats.get('replans_total', 0))} "
                            f"edges={int(info.get('known_blocked_edges', 0))}"
                        )
                    print(
                        f"step={self.total_steps:7d} eps={self.epsilon():.3f} "
                        f"reward={self.total_reward:+.2f} episode={episode_reward:+.2f} "
                        f"goals={self.goals} replay={len(self.replay)} memories={self.episodic.count()} "
                        f"vmem={self.episodic.visual_count()} loss={loss_text} "
                        f"brain={action_name} thor={primitive} {delta_text}{wp_text} "
                        f"stagnation={int(info.get('stagnation_steps', 0))}/{int(info.get('stagnation_limit', 0))} "
                        f"L{configured_level}stats={success_rate:.1f}%/med{median_text}{diag}{rehearsal_text} "
                        f"cur={intrinsic_reward:.4f} drive={homeostatic_drive:.3f} vision={self.vision.name} "
                        f"{self.env.describe()}",
                        flush=True,
                    )

                if done:
                    obs_np = self.env.reset()
                    current_visual = self._encode_vision()
                    hidden = self.net.initial_state(1, self.device)
                    prev_action = 0
                    episode_reward = 0.0
                else:
                    obs_np = actual_next_np
                    current_visual = next_visual
                    hidden = next_hidden.detach()
                    prev_action = action

                if self.visualizer is not None:
                    while self.running and self.visualizer.paused:
                        time.sleep(0.05)

                if realtime:
                    elapsed = time.perf_counter() - tick_start
                    speed = self.visualizer.speed if self.visualizer is not None else 1.0
                    time.sleep(max(0.0, tick_period / max(0.1, speed) - elapsed))
        finally:
            self.save()
            self.episodic.close()
            try:
                self.env.close()
            except Exception:
                pass
