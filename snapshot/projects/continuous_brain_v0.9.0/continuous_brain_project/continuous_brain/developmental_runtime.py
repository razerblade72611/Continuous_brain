"""Live child/parent runtime used to test gradual weaning in a bounded nursery."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .continuity_v090 import ContinuityStore
from .environment import ForagingWorld
from .nursery_collect import foraging_parent_action
from .parented_train import load_checkpoint
from .parenting import ParentRegistry


def _image_tensor(env: ForagingWorld, size: int, device: str) -> torch.Tensor:
    arr = env.render_rgb(size).astype(np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2,0,1))[None], dtype=torch.float32, device=device)


def run_foraging(state_dir: str | Path, *, steps=200, device="cpu", seed=17,
                  allow_parent=True) -> dict:
    state_dir = Path(state_dir)
    ckpt = state_dir / ("qualified_infant.pt" if (state_dir / "qualified_infant.pt").exists() else "infant_training.pt")
    model, _ = load_checkpoint(ckpt, device)
    model.eval()
    registry = ParentRegistry(state_dir / "parents.json")
    continuity = ContinuityStore(state_dir / "continuity.sqlite3")
    env = ForagingWorld(seed=seed, max_episode_steps=max(steps, 500))
    rng = random.Random(seed)
    obs = env.reset()
    saved = continuity.runtime_state()
    hidden = saved["hidden"].to(device) if saved and tuple(saved["hidden"].shape) == (1, model.cfg.nucleus_dim) else model.nucleus.initial_state(1, device)
    total_reward = 0.0
    parent_steps = 0
    student_steps = 0
    successes = 0
    prev_action_token = torch.zeros(1, dtype=torch.long, device=device)
    for step in range(1, int(steps)+1):
        image = _image_tensor(env, model.cfg.image_size, device)
        context = np.zeros(48, dtype=np.float32); context[:6] = obs[:6]; context[6] = env.energy; context[30] = 1.0
        context_t = torch.tensor(context[None], dtype=torch.float32, device=device)
        memory = torch.tensor(continuity.recent_memory_vector(model.cfg.memory_dim)[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            vision = model.vision(image)
            nuc = model.nucleus(vision, context_t, memory, prev_action_token, hidden)
            hidden = nuc["hidden"]
            workspace = nuc["workspace"]
            student_action = int(model.motor(workspace).argmax(-1)[0].item())
        use_parent = bool(allow_parent and rng.random() < registry.guidance_probability("motor"))
        action = foraging_parent_action(obs) if use_parent else student_action
        parent_steps += int(use_parent); student_steps += int(not use_parent)
        next_obs, reward, done, info = env.step(action)
        successes += int(bool(info.get("reached")))
        total_reward += float(reward)
        continuity.commit_episode_and_state(
            kind="episodic", provenance="lived", action=action, reward=reward,
            success=bool(info.get("reached")),
            content={"parent_guided": use_parent, "student_action": student_action,
                     "executed_action": action, "distance": float(info.get("distance", 0.0))},
            step=step, hidden=hidden, workspace=workspace, model_ref=str(ckpt),
        )
        prev_action_token = torch.tensor([action + 1], dtype=torch.long, device=device)
        obs = env.reset() if done else next_obs
    result = {
        "steps": int(steps), "total_reward": total_reward, "successes": successes,
        "parent_guided_steps": parent_steps, "student_steps": student_steps,
        "motor_parent_authority": registry.guidance_probability("motor"),
        "memory_counts": continuity.counts(),
    }
    continuity.close()
    return result
