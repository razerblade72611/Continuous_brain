"""Dreams: the predictive core imagining forward from moments it really lived, written as DREAMED.

``dream(core, policy, sampler, writer, n, H)`` takes ``n`` windows of lived (or practised) experience, lets the core
observe each, then rolls its dynamics forward ``H`` steps under the policy with nothing observed, and archives
the decoded frames, predicted body sense, predicted rewards and the policy's actions.  A dream row has no store
hash and its stream is DREAMED, so it can train the core and the policy but can never be read as something that
happened to the individual (``Archive.lived_rows``).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .archive import INVENTORY, WHO


@torch.no_grad()
def dream(core, policy, sampler, writer, n: int, H: int = 32, life0: int = 0, batch: int = 8) -> int:
    if policy.cfg.arm != "P-B":
        raise ValueError("dreams need a P-B policy: a dream has no memory readout")
    rows = 0
    k = 0
    while k < n:
        B = min(batch, n - k)
        b = sampler.batch(B)
        post, _ = core.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
        start = {key: v[:, -1] for key, v in post.items()}

        def act(feat):
            logits, _ = policy(feat, core.heads["body"](feat).mode())     # it acts on the body it imagines
            return logits.argmax(-1)

        feats, acts, _ = core.imagine(start, act, H, sample=True)
        imgs = (core.image_dec(feats).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        body = core.heads["body"](feats).mode().cpu().numpy()
        rew = core.heads["reward"](feats).mode().squeeze(-1).cpu().numpy()
        cont = core.heads["cont"](feats).mean.squeeze(-1).cpu().numpy()
        acts = acts.cpu().numpy()
        for i in range(B):
            for t in range(H):
                inv = np.clip(np.round(body[i, t, :len(INVENTORY)]), 0, 127).astype(int).tolist()
                writer.append({"life": life0 + k + i, "step": t, "seed": -1,
                               "frame": np.ascontiguousarray(imgs[i, t]).tobytes(), "inventory": inv,
                               "pos": [0, 0], "facing": [0, 0], "sleeping": bool(body[i, t, len(INVENTORY)] > 0.5),
                               "messages": json.dumps([]), "action": int(acts[i, t]), "utterance": "",
                               "who": WHO.index("SELF"), "reward": float(rew[i, t]), "reward_caretaker": 0.0,
                               "done": bool(t == H - 1 or cont[i, t] < 0.5), "dying": False, "store_hash": ""})
                rows += 1
        k += B
    writer.close()
    return rows
