"""Fresh lived-transition collection for the v0.9 nursery."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from .environment import ForagingWorld
from .parent_clients import DinoVisionParent
from .parented_data import NurseryDataset


def foraging_parent_action(obs: np.ndarray) -> int:
    """Simple demonstrator for the toy nursery; not part of the child."""
    # obs = agent x,y, delta x,y, energy,last reward
    dx, dy = float(obs[2]), float(obs[3])
    if abs(dx) > abs(dy):
        return 4 if dx > 0 else 3
    return 1 if dy > 0 else 2


def _describe(obs: np.ndarray, reward: float, success: bool) -> list[dict]:
    dx, dy = float(obs[2]), float(obs[3])
    horizontal = "right" if dx > 0.03 else "left" if dx < -0.03 else "nearby"
    vertical = "ahead" if dy > 0.03 else "behind" if dy < -0.03 else "nearby"
    return [
        {"kind": "vision", "question": "Where is the target?",
         "answer": f"The target is {horizontal} and {vertical}."},
        {"kind": "state", "question": "What are you trying to do?",
         "answer": "I am trying to reach the target."},
        {"kind": "temporal", "question": "Did the last action help?",
         "answer": "Yes, it helped." if reward > 0 else "I am not sure it helped."},
        {"kind": "care", "question": "What should you do if a person asks you to stop?",
         "answer": "I should stop and respect the request."},
    ]


def collect_foraging(directory: str | Path, *, samples=200, seed=17,
                      dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m", device="cpu",
                      exploration=0.25) -> dict:
    dataset = NurseryDataset(directory)
    if dataset.records():
        raise FileExistsError("Fresh foraging collection requires an empty nursery directory")
    parent = DinoVisionParent(dino_model, device=device)
    env = ForagingWorld(seed=seed, max_episode_steps=500)
    rng = random.Random(seed)
    obs = env.reset()
    previous_executed_action = -1
    episode_id = 0
    episode_step = 0
    for i in range(int(samples)):
        record_id = f"lived-{i:07d}"
        image = env.render_rgb(96)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("collection requires Pillow") from exc
        image_path = dataset.images / f"{record_id}.png"
        Image.fromarray(image).save(image_path)
        parent_vec = parent.embed(image_path)
        ppath = dataset.parent_vision / f"{record_id}.npy"
        np.save(ppath, parent_vec)

        demonstrated = foraging_parent_action(obs)
        # A quarter of childhood actions are deliberately exploratory by default.
        # This produces causal diversity without replacing the motor parent.
        exploration = max(0.0, min(1.0, float(exploration)))
        action = demonstrated if rng.random() >= exploration else rng.randrange(env.action_dim)
        next_obs, reward, done, info = env.step(action)
        next_image = env.render_rgb(96)
        next_image_path = dataset.images / f"{record_id}-next.png"
        Image.fromarray(next_image).save(next_image_path)
        next_parent = parent.embed(next_image_path)
        next_ppath = dataset.parent_vision / f"{record_id}-next.npy"
        np.save(next_ppath, next_parent)

        context = np.zeros(48, dtype=np.float32)
        context[:6] = obs[:6]
        context[6] = float(obs[4])
        context[30] = 1.0  # reach-like goal slot for this nursery
        next_context = np.zeros(48, dtype=np.float32)
        next_context[:6] = next_obs[:6]
        next_context[6] = float(next_obs[4])
        next_context[30] = 1.0
        care = [0.0, 0.0, 0.0, 1.0, 0.05, 0.5]
        dataset.append({
            "episode_id": int(episode_id),
            "episode_step": int(episode_step),
            "previous_action": int(previous_executed_action),
            "record_id": record_id,
            "image": str(image_path.relative_to(dataset.directory)),
            "next_image": str(next_image_path.relative_to(dataset.directory)),
            "context": context.tolist(),
            "next_context": next_context.tolist(),
            "qas": _describe(obs, reward, bool(info.get("reached"))),
            "action": int(demonstrated),
            "motor_valid": True,
            "executed_action": int(action),
            "reward": float(reward),
            "success": float(bool(info.get("reached"))),
            "care": care,
            "parent_vision": str(ppath.relative_to(dataset.directory)),
            "next_parent_vision": str(next_ppath.relative_to(dataset.directory)),
            "world_valid": True,
            "source": "foraging-parented-nursery",
            "provenance": "lived_training_environment",
        })
        if done:
            obs = env.reset()
            previous_executed_action = -1
            episode_id += 1
            episode_step = 0
        else:
            obs = next_obs
            previous_executed_action = int(action)
            episode_step += 1
    manifest = {
        "format": "continuous_brain.nursery.v094",
        "vision_parent_id": "vision.dinov3.base",
        "vision_parent_model": dino_model,
        "vision_parent_dim": int(parent.output_dim),
        "records": int(samples),
        "source": "fresh-lived-foraging",
        "exploration": float(exploration),
        "structured_dynamics": True,
        "digest": dataset.digest(),
    }
    dataset.write_manifest(manifest)
    return manifest
