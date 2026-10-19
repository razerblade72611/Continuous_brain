import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from continuous_brain.parented_train import load_checkpoint

ckpt_path = Path("state/nucleus_regrowth_v1/infant_training.pt")
model, _ = load_checkpoint(ckpt_path, "cpu")
cfg = model.cfg

heads = {k: v.out_features for k, v in model.vision_parent_heads.items()}
assert heads, "checkpoint has no vision_parent_heads; this approach needs a different starting point"
parent_key, parent_dim = next(iter(heads.items()))
parent_id = parent_key.replace("__", ".")
print(f"parent head: {parent_key!r} -> parent_id {parent_id!r}, dim={parent_dim}")
print(f"cfg: context_dim={cfg.context_dim} image_size={cfg.image_size} action_dim={cfg.action_dim} care_dim={cfg.care_dim}")

data_dir = Path("data/NurseryDatasetFake")
images_dir = data_dir / "images"
parent_dir = data_dir / "parent_vision"
images_dir.mkdir(parents=True, exist_ok=True)
parent_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(0)
np.random.seed(0)

num_episodes = 8
steps_per_episode = 20  # well under the DetourWorld 120-step terminal-visibility limit
rows = []

def make_image(name: str) -> str:
    arr = (np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8)
    path = images_dir / f"{name}.png"
    Image.fromarray(arr).save(path)
    return str(path.relative_to(data_dir))

def make_parent_vision(name: str) -> str:
    vec = np.random.randn(parent_dim).astype(np.float32)
    path = parent_dir / f"{name}.npy"
    np.save(path, vec)
    return str(path.relative_to(data_dir))

for eid in range(num_episodes):
    # one extra frame per episode so every step has a real "next_image"
    frame_names = [f"ep{eid:03d}_s{s:03d}" for s in range(steps_per_episode + 1)]
    frame_images = [make_image(n) for n in frame_names]
    for step in range(steps_per_episode):
        is_last_step = step == steps_per_episode - 1
        action = rng.randrange(cfg.action_dim)
        rows.append({
            "record_id": f"ep{eid:03d}_s{step:03d}",
            "episode_id": eid,
            "episode_step": step,
            "image": frame_images[step],
            "next_image": frame_images[step + 1],
            "parent_vision": make_parent_vision(frame_names[step]),
            "context": [0.0] * cfg.context_dim,
            "next_context": [0.0] * cfg.context_dim,
            "action": action,
            "executed_action": action,
            "motor_valid": True,
            "world_valid": True,
            "reward": 0.0,
            "success": 1.0 if is_last_step else 0.0,
            "collision": False,
            "care": [0.0] * cfg.care_dim,
            "qas": [{"question": "What do you see?", "answer": "a synthetic smoke-test frame."}],
        })

with open(data_dir / "lessons.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")

(data_dir / "manifest.json").write_text(json.dumps({"vision_parent_id": parent_id}, indent=2))
print(f"wrote {len(rows)} rows across {num_episodes} episodes to {data_dir}")
