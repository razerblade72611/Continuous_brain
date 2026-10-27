"""Fast episodic memory for Continuous Brain: one independently written, sparse, masked low-rank adapter per experienced
episode, keyed for retrieval, read out by reconstruction (Claude + Sol, 2026-10-27).

Clean reimplementation FROM THE PAPER (no code copied): Pan, Hahami, Siegelmann & Sompolinsky (2026), "A complementary
learning system for continual episodic memory in large language models", bioRxiv 10.64898/2026.08.24.746712.
Published design reproduced here (CLS-0A defaults):
  - each episode gets its own masked rank-r LoRA on ONE module (default layer 10 mlp.down_proj), trained from the BASE
    model (no other episode active), so every episode is independently readable;
  - mask: a random subset of `mask` input and `mask` output dimensions; the adapter only reads those inputs and only
    writes those outputs (params = r * 2 * mask); LoRA weights in float32, scaling alpha/r with alpha = r*sqrt(r);
    A ~ kaiming-uniform(a=sqrt(5)), B = 0 (zero initial output);
  - objective: reconstruct the episode text after the fixed cue "Reconstruct the story:" (loss on the episode tokens
    only), full-batch AdamW (lr 0.05, no weight decay), up to `max_epochs`, exact greedy reconstruction checked every
    `check_every` epochs; on failure the mask grows x1.2 (up to `retries` times);
  - keys: internal = mean-pooled, L2-normalised input of a later MLP (default layer 24) on the plain episode text with
    all adapters off (first token excluded); an external embedding function can be supplied instead;
  - retrieval: winner-take-all by dot product, with an optional null-memory threshold (Sol: a persistent individual
    must be able to decline to recall).
Every stored record carries provenance (time, source) so self-acquired experience is distinguishable from anything else.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

CUE = "Reconstruct the story:"


@dataclass
class Episode:
    eid: str
    text: str
    time: int                       # when it was experienced (life / step index)
    source: str                     # provenance, e.g. "observed:life1"
    key: torch.Tensor | None = None
    mask: int = 0
    attempts: list = field(default_factory=list)


class _EpisodeAdapter(nn.Module):
    def __init__(self, in_f, out_f, r, mask, gen, device):
        super().__init__()
        self.in_idx = torch.randperm(in_f, generator=gen)[:mask].to(device)
        self.out_idx = torch.randperm(out_f, generator=gen)[:mask].to(device)
        self.A = nn.Parameter(torch.empty(r, mask, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(mask, r, device=device, dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
            a = torch.empty(r, mask)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        with torch.no_grad():
            self.A.copy_(a.to(device))
        self.scaling = math.sqrt(r)            # alpha / r with alpha = r * sqrt(r)

    def delta(self, x, like):
        h = x.index_select(-1, self.in_idx).to(torch.float32)
        y = (h @ self.A.T) @ self.B.T * self.scaling
        out = torch.zeros_like(like)
        return out.index_add(-1, self.out_idx, y.to(like.dtype))


class _MaskedSite(nn.Module):
    """Wraps the target linear; applies the deltas of the currently active episode adapters only."""
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active: list[str] = []

    def forward(self, x):
        y = self.base(x)
        for k in self.active:
            y = y + self.adapters[k].delta(x, y)
        return y


class EpisodicMemory:
    def __init__(self, model, tok, layer=10, key_layer=24, r=2, init_mask=45, growth=1.2, retries=10, lr=0.05,
                 max_epochs=1000, check_every=100, seed=72, system_prompt=None, key_fn=None):
        self.model, self.tok = model, tok
        self.r, self.init_mask, self.growth, self.retries = r, init_mask, growth, retries
        self.lr, self.max_epochs, self.check_every = lr, max_epochs, check_every
        self.gen = torch.Generator().manual_seed(seed)
        self.dev = next(model.parameters()).device
        layers = model.model.layers
        self.layer, self.key_layer = layer, min(key_layer, len(layers) - 1)
        mlp = layers[layer].mlp
        if not isinstance(mlp.down_proj, _MaskedSite):
            mlp.down_proj = _MaskedSite(mlp.down_proj)
        self.site: _MaskedSite = mlp.down_proj
        self.system_prompt = system_prompt
        self.key_fn = key_fn                 # external embedding: fn(list[str]) -> tensor [n, d] (normalised)
        self.episodes: dict[str, Episode] = {}

    # ---- formatting -------------------------------------------------------------------------------------------
    def _chat(self, user, assistant=None):
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + \
               [{"role": "user", "content": user}]
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            p = "".join(f"<{m['role']}>{m['content']}</{m['role']}>\n" for m in msgs) + "<assistant>"
        if assistant is None:
            return p, None
        end = getattr(self.tok, "eos_token", None) or ""
        return p, p + assistant + (self._end_text() or end)

    def _end_text(self):
        t = getattr(self.tok, "chat_template", None) or ""
        return "<|im_end|>" if "<|im_end|>" in t else None

    def _ids(self, text):
        x = self.tok(text, return_tensors="pt").input_ids
        return x.to(self.dev)

    # ---- keys ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def key(self, text):
        if self.key_fn is not None:
            return self.key_fn([text])[0].float().cpu()
        saved, self.site.active = self.site.active, []
        cap = {}
        h = self.model.model.layers[self.key_layer].mlp.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", i[0]))
        try:
            self.model(self._ids(text))
        finally:
            h.remove()
            self.site.active = saved
        x = cap["x"][0, 1:].float()           # exclude the first token
        v = x.mean(0)
        return (v / v.norm()).cpu()

    # ---- writing ------------------------------------------------------------------------------------------------
    def _train_once(self, ep, mask):
        name = ep.eid
        lin = self.site.base
        ad = _EpisodeAdapter(lin.in_features, lin.out_features, self.r, mask, self.gen, self.dev)
        self.site.adapters[name] = ad
        self.site.active = [name]              # written against the base: no other episode active
        prompt, full = self._chat(CUE, ep.text)
        ids = self._ids(full)
        n_prompt = self._ids(prompt).shape[1]
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        target = ids[0, n_prompt:].tolist()
        opt = torch.optim.AdamW(ad.parameters(), lr=self.lr, weight_decay=0.0)
        t0, ok, epochs = time.time(), False, 0
        for e in range(self.max_epochs):
            if (e + 1) % self.check_every == 0 and self._exact(prompt, target):
                ok, epochs = True, e
                break
            loss = self.model(ids, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            epochs = e + 1
        if not ok:
            ok = self._exact(prompt, target)
        self.site.active = []
        rec = {"mask": mask, "params": sum(p.numel() for p in ad.parameters()), "epochs": epochs, "exact": ok,
               "seconds": round(time.time() - t0, 2), "target_tokens": len(target), "final_loss": float(loss.item())}
        if not ok:
            del self.site.adapters[name]
        return ok, rec

    @torch.no_grad()
    def _exact(self, prompt, target):
        was = self.model.training
        self.model.eval()
        ids = self._ids(prompt)
        out = self.model.generate(ids, max_new_tokens=len(target) + 2, do_sample=False,
                                  pad_token_id=getattr(self.tok, "eos_token_id", 0))
        self.model.train(was)
        return out[0, ids.shape[1]:ids.shape[1] + len(target)].tolist() == target

    def write(self, eid, text, time_, source):
        """Store one experienced episode. Returns the Episode (exactly reconstructable) or raises after all retries."""
        assert eid not in self.episodes, f"episode {eid} already stored"
        ep = Episode(eid, text, time_, source)
        ep.key = self.key(text)
        mask = self.init_mask
        for _ in range(self.retries):
            m = min(int(math.ceil(mask)), self.site.base.in_features, self.site.base.out_features)
            ok, rec = self._train_once(ep, m)
            ep.attempts.append(rec)
            if ok:
                ep.mask = m
                self.episodes[eid] = ep
                return ep
            mask *= self.growth
        err = RuntimeError(f"episode {eid} not exactly reconstructable after {self.retries} attempts")
        err.attempts = ep.attempts
        raise err

    # ---- reading ------------------------------------------------------------------------------------------------
    def route(self, cue, threshold=None, k=1):
        """Winner-take-all (top-k) by dot product of normalised keys. Returns [(eid, sim)], [] if below threshold."""
        if not self.episodes:
            return []
        q = self.key(cue)
        eids = list(self.episodes)
        sims = torch.stack([self.episodes[e].key for e in eids]) @ q
        order = torch.argsort(sims, descending=True)[:k].tolist()
        out = [(eids[i], float(sims[i])) for i in order]
        return [x for x in out if threshold is None or x[1] >= threshold]

    @torch.no_grad()
    def reconstruct(self, eid, max_new=300):
        self.site.active = [eid]
        try:
            prompt, _ = self._chat(CUE)
            ids = self._ids(prompt)
            out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=getattr(self.tok, "eos_token_id", 0))
        finally:
            self.site.active = []
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def state_digest(self):
        h = hashlib.sha256()
        for k in sorted(self.site.adapters):
            ad = self.site.adapters[k]
            for t in (ad.in_idx, ad.out_idx, ad.A, ad.B):
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
