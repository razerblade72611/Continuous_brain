"""Reading the learned skill's mind for the viewer, without touching it.

``read_mind(ind)`` looks at what the individual's predictive core holds after the latest tick
(``skills.core_state``) and decodes it:

- what it sees: the core's own picture of the present moment (its decoded state), beside the real frame;
- what it imagines: the core rolled forward H steps under its own policy, decoded frame by frame, with nothing
  observed (the deterministic mode, so it uses no randomness and changes nothing);
- what it foresees: its 15 risk predictions (``data.RISKS`` x ``data.RISK_H``);
- what it would do and why: its policy's action probabilities and the routine option it believes it serves;
- what else it could do: for its choice and its next three likeliest actions, the consequences its world model
  imagines over ``H_ALT`` steps (that action first, then its own skill), sampled ``N_ALT`` times: imagined reward,
  risks and body at the end, with their spread across samples.  Its skill does not use these yet (imitation has no
  value function): they show what its world model predicts, which is what a planner or a value function will use.

Everything is computed from copies of store values; the store, the core's state and the life are untouched
(``tests/test_cb2_mind.py``).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

from continuous_brain_individual import codec, ledger
from continuous_brain_individual import memory as mem
from continuous_brain_individual.body import read_body
from continuous_brain_individual.features import belief_features
from continuous_brain_world.env import action_names

from .archive import INVENTORY
from .data import RISK_H, RISKS, body_vector
from .record import map_crop
from .skill_host import weights_sha

ACTIONS = action_names()
H_ALT, N_ALT, K_ALT = 50, 4, 4
OPTIONS = ("routine", "explore", "sleeping", "eat")


def _png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frames(core, feats) -> list[str]:
    imgs = (core.image_dec(feats).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return [_png(x) for x in imgs.reshape((-1,) + imgs.shape[-3:])]


@torch.no_grad()
def read_mind(ind, H: int = 15) -> dict | None:
    store = ind.rt.store
    w = np.asarray(store.read("skills.weights"))
    raw = np.asarray(store.read("skills.core_state"), np.uint8)
    if w.size == 0 or raw.size == 0:
        return None
    host = ind.rt.components["skills"]
    core, pol = host._models(store.read("skills.weights"))
    vec = np.frombuffer(raw.tobytes(), np.float32)
    d = core.cfg.deter
    state = {"deter": torch.from_numpy(vec[:d].copy())[None],
             "stoch": torch.from_numpy(vec[d:].copy()).reshape(1, core.cfg.stoch, core.cfg.discrete)}
    state["logit"] = torch.zeros_like(state["stoch"])
    feat = core.feat(state)
    body = read_body(store.read("driver.body"))
    inv = body["inventory"]
    bv = body_vector([[inv[k] for k in INVENTORY]], [body["sleeping"]], [body["facing"]], [int(body["step"])],
                     [body["player_pos"]])
    feats = crop = None
    if pol.cfg.arm == "P-A":
        st = codec.unblob(store.read("skills.state")) or {}
        memory = codec.unblob(store.read("episodic_memory.state")) or mem.empty()
        led = codec.unblob(store.read("goal_genesis.state")) or ledger.empty()
        feats = belief_features(memory, body, int(body["step"]), led, None, st.get("hist") or [])[None]
        feats = feats.astype(np.float16).astype(np.float32)
        crop = map_crop(memory, body["player_pos"]).reshape(1, -1)
    logits, opt_logits = pol(feat, bv, feats, crop)
    probs = torch.softmax(logits, -1)[0].numpy()
    top = np.argsort(-probs)[:5]
    risk = core.heads["risk"](feat).mean[0].numpy()

    def act(f):                                  # in imagination it feels the body it imagines; P-A keeps its
        b = core.heads["body"](f).mode()         # memory readout as it was when the dream began
        lg, _ = pol(f, b, feats, crop)
        return lg.argmax(-1)

    ifeats, iacts, _ = core.imagine(state, act, H, sample=False)
    irisk = core.heads["risk"](ifeats[:, -1]).mean[0].numpy()
    alternatives = imagine_alternatives(core, pol, state, [int(i) for i in top[:K_ALT]], feats, crop)
    lineage = (codec.unblob(store.read("skills.state")) or {}).get("lineage", [])
    return {"weights_sha256": weights_sha(w), "arm": pol.cfg.arm, "params": core.n_params(),
            "sees": _frames(core, feat[:, None])[0],
            "imagined": _frames(core, ifeats), "imagined_actions": [ACTIONS[int(a)] for a in iacts[0]],
            "risks": [f"{r} within {h}" for r in RISKS for h in RISK_H],
            "risk": [round(float(x), 4) for x in risk], "risk_after_imagining": [round(float(x), 4) for x in irisk],
            "policy": [{"action": ACTIONS[int(i)], "p": round(float(probs[i]), 4)} for i in top],
            "alternatives": alternatives, "alt_horizon": H_ALT, "alt_samples": N_ALT,
            "intent": OPTIONS[int(opt_logits.argmax(-1)[0])] if int(opt_logits.argmax(-1)[0]) < len(OPTIONS) else "?",
            "lineage": [{k: e.get(k) for k in ("kind", "time", "tick", "life", "weights_sha256", "event_sha")}
                        | {"note": (e.get("lineage") or {}).get("note")} for e in lineage]}


@torch.no_grad()
def imagine_alternatives(core, pol, state: dict, actions: list[int], feats=None, crop=None) -> list[dict]:
    """For each first action: N_ALT imagined continuations of H_ALT steps (that action, then its skill), batched.
    Sampling runs inside a forked RNG, so the process's random state, and the life, are untouched."""
    K, N = len(actions), N_ALT
    B = K * N
    st = {k: v.expand(B, *v.shape[1:]).clone() for k, v in state.items()}
    first = torch.tensor(actions).repeat_interleave(N)
    f_rep = None if feats is None else np.repeat(feats, B, 0)
    c_rep = None if crop is None else np.repeat(crop, B, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        rew = torch.zeros(B)
        a = first
        for t in range(H_ALT):
            st = core.rssm.img_step(st, core._onehot(a), sample=True)
            f = core.feat(st)
            rew += core.heads["reward"](f).mode().reshape(B)
            lg, _ = pol(f, core.heads["body"](f).mode(), f_rep, c_rep)
            a = lg.argmax(-1)
        risk = core.heads["risk"](f).mean.reshape(K, N, -1).numpy()
        body = core.heads["body"](f).mode()[:, :4].reshape(K, N, 4).numpy()     # health, food, drink, energy
    rew = rew.reshape(K, N).numpy()
    out = []
    for k, act in enumerate(actions):
        out.append({"action": ACTIONS[act], "reward": round(float(rew[k].mean()), 3),
                    "reward_spread": round(float(rew[k].std()), 3),
                    "risk": [round(float(x), 4) for x in risk[k].mean(0)],
                    "risk_spread": [round(float(x), 4) for x in risk[k].std(0)],
                    "vitals": [round(float(x), 2) for x in body[k].mean(0)]})
    base = out[0]
    for o in out:
        o["vs_choice"] = {"reward": round(o["reward"] - base["reward"], 3),
                          "vitals": [round(x - y, 2) for x, y in zip(o["vitals"], base["vitals"])]}
    return out
