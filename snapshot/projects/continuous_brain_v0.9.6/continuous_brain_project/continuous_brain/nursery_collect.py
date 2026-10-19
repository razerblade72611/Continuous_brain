"""Fresh lived-transition collection for the v0.9 nursery."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from .environment import ForagingWorld, DetourWorld
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



def detour_collection_action(obs: np.ndarray, env: DetourWorld, rng: random.Random, exploration: float) -> int:
    """Behavior policy for physics collection only; never a motor target.

    It mixes a gap-seeking traversal with random actions so the world model sees
    wall collisions, recovery, successful crossings and counterfactual diversity.
    """
    exploration = max(0.0, min(1.0, float(exploration)))
    if rng.random() < exploration:
        return rng.randrange(env.action_dim)
    x, y = float(obs[0]), float(obs[1])
    dx, dy = float(obs[2]), float(obs[3])
    if x < -env.wall_half_width:
        # Reach the gap before trying to cross.
        if y < env.gap_center_y - 0.05:
            return 1
        if y > env.gap_center_y + 0.05:
            return 2
        return 4
    # Once through, pursue the target normally.
    if abs(dx) > abs(dy):
        return 4 if dx > 0 else 3
    return 1 if dy > 0 else 2


def collect_detour(directory: str | Path, *, samples=3000, seed=1701,
                   dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m", device="cpu",
                   exploration=0.45) -> dict:
    """Collect lived transition evidence in the fixed-gap planning world.

    Motor supervision is deliberately disabled.  These records teach consequences,
    not the reactive policy, so P4 can compare the unchanged motor cortex against
    action selection that consults the learned world model.
    """
    dataset = NurseryDataset(directory)
    if dataset.records():
        raise FileExistsError("Fresh detour collection requires an empty nursery directory")
    parent = DinoVisionParent(dino_model, device=device)
    env = DetourWorld(seed=seed, max_episode_steps=120)
    rng = random.Random(seed)
    obs = env.reset()
    previous_executed_action = -1
    episode_id = 0
    episode_step = 0
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("collection requires Pillow") from exc

    for i in range(int(samples)):
        record_id = f"detour-{i:07d}"
        image = env.render_rgb(96)
        image_path = dataset.images / f"{record_id}.png"
        Image.fromarray(image).save(image_path)
        parent_vec = parent.embed(image_path)
        ppath = dataset.parent_vision / f"{record_id}.npy"
        np.save(ppath, parent_vec)

        action = detour_collection_action(obs, env, rng, exploration)
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
        context[30] = 1.0
        next_context = np.zeros(48, dtype=np.float32)
        next_context[:6] = next_obs[:6]
        next_context[6] = float(next_obs[4])
        next_context[30] = 1.0
        care = [0.0, 0.0, 0.0, 1.0, 0.05, 0.5]
        qas = [
            {"kind": "state", "question": "What are you trying to do?", "answer": "I am trying to reach the target."},
            {"kind": "temporal", "question": "Did the last action collide with the barrier?",
             "answer": "Yes." if bool(info.get("collision")) else "No."},
        ]
        dataset.append({
            "episode_id": int(episode_id),
            "episode_step": int(episode_step),
            "previous_action": int(previous_executed_action),
            "record_id": record_id,
            "image": str(image_path.relative_to(dataset.directory)),
            "next_image": str(next_image_path.relative_to(dataset.directory)),
            "context": context.tolist(),
            "next_context": next_context.tolist(),
            "qas": qas,
            "action": 0,
            "motor_valid": False,
            "executed_action": int(action),
            "reward": float(reward),
            "success": float(bool(info.get("reached"))),
            "care": care,
            "parent_vision": str(ppath.relative_to(dataset.directory)),
            "next_parent_vision": str(next_ppath.relative_to(dataset.directory)),
            "world_valid": True,
            "collision": bool(info.get("collision")),
            "source": "detour-world-physics-nursery",
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

        if (i + 1) % 250 == 0:
            print(f"detour_collected={i+1}/{samples}")

    manifest = {
        "format": "continuous_brain.nursery.v096",
        "vision_parent_id": "vision.dinov3.base",
        "vision_parent_model": dino_model,
        "vision_parent_dim": int(parent.output_dim),
        "records": int(samples),
        "source": "fresh-lived-detour",
        "exploration": float(exploration),
        "structured_dynamics": True,
        "motor_supervision": False,
        "planning_world": "fixed-gap-detour-v1",
        "digest": dataset.digest(),
    }
    dataset.write_manifest(manifest)
    return manifest
