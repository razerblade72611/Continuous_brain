"""Gate v3 core: a frozen Qwen3 backbone with one memory arm, the day/night life loop, and probe scoring.

Arms (lab/DECISIONS_v3.md section 4):
  T  Titans MLP in the residual stream after layer `layer` (v2 carried over).
  S  sparse product-key slots after layer `layer` (being/v3/slots.py), Meta TF-IDF top-t updates.
  L  LoRA (q,v projections, all layers) + frozen-teacher KL on background text (the Baseten recipe).
  G  T or S plus self-generated replay of OLD lives at night (replaces half the background rehearsal).
Day: each fact of the life in all training phrasings, 1 pass, 1 background sentence per life sentence.
Night: today's buffer replayed `night_passes` times, then discarded (no diary: past lives are never re-read).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from being.memory import TitansMemory
from being.v3.slots import SlotMemory

ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "models/qwen3-0.6b"


@dataclass
class V3Config:
    arm: str = "T"              # T | S | L | G
    g_base: str = "T"           # for G: which memory it replays into
    g_gate: bool = True         # G: template filter + self-consistency gate
    layer: int = 14
    lr: float = 3e-3
    momentum: float = 0.9
    hidden: int = 1024          # T
    n_keys: int = 48            # S
    slot_k: int = 16            # S
    top_t: int = 64             # S: slots updated per text
    s_fit: bool = False         # S: fit sub-keys to background queries before life (S-fitted) vs fixed random
    lora_r: int = 16            # L
    kl_weight: float = 1.0      # L
    night_passes: int = 4
    rehearsal: int = 1          # background sentences per life sentence
    replay_cues: int = 8        # G: old-life recalls attempted per night
    max_new: int = 12           # probe generation length


def load_qwen(path=QWEN, threads: int | None = None, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def backbone_hash(model) -> str:
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if "lora_" in n:
            continue
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Being3:
    """One being: the shared frozen backbone plus its own memory. close() removes hooks / adapters."""

    def __init__(self, model, tok, cfg: V3Config, birth_seed: int):
        self.model, self.tok, self.cfg, self.seed = model, tok, cfg, birth_seed
        d = model.config.hidden_size
        self.kind = cfg.g_base if cfg.arm == "G" else cfg.arm
        self.dev = next(model.parameters()).device
        self.hook = None
        self.opt = None
        if self.kind == "T":
            self.memory = TitansMemory(d, cfg.hidden, d, birth_seed=birth_seed)
        elif self.kind == "S":
            self.memory = SlotMemory(d, n_keys=cfg.n_keys, k=cfg.slot_k, birth_seed=birth_seed)
        elif self.kind == "L":
            self.memory = None
            self._attach_lora(birth_seed)
        else:
            raise ValueError(cfg.arm)
        if self.memory is not None:
            self.memory.to(self.dev)              # born on CPU from the birth seed, then moved: same weights on any device
            if self.kind == "T":
                self.memory.momentum_state = {n: torch.zeros_like(p) for n, p in self.memory.trainable()}
            else:
                self.memory.momentum_state = torch.zeros_like(self.memory.V)
            self.norm = nn.LayerNorm(d, elementwise_affine=False).to(self.dev)
            self.hook = model.model.layers[cfg.layer].register_forward_hook(self._hook)

    # --- wiring ---
    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + self.memory(self.norm(h))
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def _attach_lora(self, seed):
        torch.manual_seed(seed)
        self.lora = []
        for layer in self.model.model.layers:
            for name in ("q_proj", "v_proj"):
                lin = getattr(layer.self_attn, name)
                a = nn.Parameter((torch.randn(self.cfg.lora_r, lin.in_features) / lin.in_features ** 0.5).to(self.dev))
                b = nn.Parameter(torch.zeros(lin.out_features, self.cfg.lora_r, device=self.dev))
                self.lora += [a, b]
                lin._lora = (a, b)
                lin._orig_forward = lin.forward
                lin.forward = (lambda l: (lambda x: l._orig_forward(x) + (x @ l._lora[0].T) @ l._lora[1].T))(lin)
        self.opt = torch.optim.SGD(self.lora, lr=self.cfg.lr, momentum=self.cfg.momentum)

    def close(self):
        if self.hook is not None:
            self.hook.remove()
        if self.kind == "L":
            for layer in self.model.model.layers:
                for name in ("q_proj", "v_proj"):
                    lin = getattr(layer.self_attn, name)
                    lin.forward = lin._orig_forward
                    del lin._lora, lin._orig_forward

    def trainable_params(self) -> int:
        if self.kind == "L":
            return sum(p.numel() for p in self.lora)
        if self.kind == "S":
            return self.memory.trainable_params()
        return sum(p.numel() for _, p in self.memory.trainable())

    def memory_hash(self) -> str:
        h = hashlib.sha256()
        params = self.lora if self.kind == "L" else [p for _, p in sorted(self.memory.named_parameters())]
        for p in params:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def state(self):
        if self.kind == "L":
            return [p.detach().clone() for p in self.lora]
        return {n: p.detach().clone() for n, p in self.memory.named_parameters()}

    @torch.no_grad()
    def load_state(self, st):
        if self.kind == "L":
            for p, q in zip(self.lora, st):
                p.copy_(q)
        else:
            for n, p in self.memory.named_parameters():
                p.copy_(st[n])

    # --- learning ---
    def _ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.dev)

    def birth(self, background: list[str]):
        """S only: background slot-usage statistics from never-scored text (Meta's idf)."""
        if self.kind != "S":
            return
        cap = {}

        def grab(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            cap["h"] = h.detach()
        hk = self.model.model.layers[self.cfg.layer].register_forward_hook(grab)
        self.hook.remove()
        xs = []
        with torch.no_grad():
            for s in background:
                self.model(self._ids(s))
                xs.append(self.norm(cap["h"])[0])
        hk.remove()
        x = torch.cat(xs)
        if self.cfg.s_fit:
            self.memory.fit_keys(x, seed=self.seed)
        self.memory.observe_background(x)
        self.hook = self.model.model.layers[self.cfg.layer].register_forward_hook(self._hook)

    def step(self, text: str, teacher_text: str | None = None) -> float:
        """One surprise step on `text`: the backbone's own next-token loss, gradient to the memory only."""
        ids = self._ids(text)
        loss = self.model(ids, labels=ids).loss
        if self.kind == "L" and teacher_text is not None and self.cfg.kl_weight > 0:
            loss = loss + self.cfg.kl_weight * self._kl_to_frozen(teacher_text)
        loss.backward()
        if self.kind == "T":
            self.memory.titans_step(self.cfg.lr, self.cfg.momentum, 0.0)
        elif self.kind == "S":
            self.memory.sparse_step(self.cfg.top_t, self.cfg.lr, self.cfg.momentum)
        else:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return loss.item()

    def _kl_to_frozen(self, text):
        ids = self._ids(text)
        lin_states = []
        for layer in self.model.model.layers:            # frozen teacher = same backbone with LoRA switched off
            for name in ("q_proj", "v_proj"):
                lin = getattr(layer.self_attn, name)
                lin_states.append((lin, lin.forward))
                lin.forward = lin._orig_forward
        with torch.no_grad():
            t = F.log_softmax(self.model(ids).logits, -1)
        for lin, fwd in lin_states:
            lin.forward = fwd
        s = F.log_softmax(self.model(ids).logits, -1)
        return F.kl_div(s, t, log_target=True, reduction="batchmean") / ids.shape[1]

    # --- reading ---
    @torch.no_grad()
    def generate(self, prompt: str, max_new: int | None = None) -> str:
        ids = self._ids(prompt)
        out = self.model.generate(ids, max_new_tokens=max_new or self.cfg.max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def answer_logprob(self, prompt: str, answer: str) -> float:
        p = self._ids(prompt)
        full = self._ids(prompt + " " + answer)
        logits = self.model(full).logits[0, :-1]
        lp = F.log_softmax(logits, -1)
        tgt = full[0, 1:]
        n = p.shape[1] - 1
        return lp[n:].gather(-1, tgt[n:, None]).sum().item()


def first_answer(gen: str) -> str:
    """The first phrase of a greedy continuation, up to sentence punctuation or newline."""
    g = gen.strip()
    for stop in (".", "\n", ",", ";", "!", "?"):
        g = g.split(stop)[0]
    return g.strip().strip('"').strip()


def exact(gen: str, answer: str) -> bool:
    a = first_answer(gen).lower()
    ans = answer.lower()
    return a == ans or a.startswith(ans) or a.endswith(" " + ans) or a == "a " + ans or a == "an " + ans
