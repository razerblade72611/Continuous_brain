"""The frozen pretrained backbone, behind a small interface.

The backbone never trains. Gradients flow *through* it (to the bridge) during
sleep, never *into* it. `HFBackbone` wraps a Hugging Face causal LM
(e.g. a ~27B Qwen checkpoint in 4-bit on a 24 GB card, per the spec).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BackboneOutput:
    logits: torch.Tensor   # [B, L, V]
    hidden: torch.Tensor   # [B, L, H] final-layer hidden states


@dataclass
class Generation:
    text: str
    token_ids: torch.Tensor   # [L_new]
    hidden: torch.Tensor      # [L_new, H] final-layer hidden state that produced each token


class Backbone(ABC):
    hidden_size: int
    device: torch.device
    dtype: torch.dtype
    eos_token_id: int | None

    # -------------------------------------------------------------- required

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Text -> 1-D token ids (no padding, no special tokens)."""

    @abstractmethod
    def decode(self, ids: torch.Tensor) -> str: ...

    @abstractmethod
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """[..., L] ids -> [..., L, H] input embeddings."""

    @abstractmethod
    def forward_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput: ...

    @abstractmethod
    def embedding_rms(self) -> float:
        """Typical RMS of an input-embedding row; the bridge matches this scale."""

    @contextmanager
    def gradient_mode(self):
        """Context for passes that backprop through the frozen backbone."""
        yield

    # -------------------------------------------------------------- shared logic

    def _pack(self, prefixes: torch.Tensor | None, segments: list[list[torch.Tensor]]):
        """Build right-padded [B, L, H] embeddings with no internal padding.

        prefixes: [B, K, H] soft tokens or None. segments[b] is a list of 1-D id
        tensors concatenated after the prefix. Returns embeds, mask, and for each
        example the start index of each segment.
        """
        b = len(segments)
        k = 0 if prefixes is None else prefixes.shape[1]
        rows, starts = [], []
        for i in range(b):
            ids = [s.to(self.device) for s in segments[i]]
            parts = [] if prefixes is None else [prefixes[i].to(self.dtype)]
            pos, st = k, []
            for s in ids:
                st.append(pos)
                pos += int(s.shape[0])
                if s.shape[0]:
                    parts.append(self.embed(s))
            rows.append(torch.cat(parts, dim=0) if parts else
                        torch.zeros(0, self.hidden_size, device=self.device, dtype=self.dtype))
            starts.append(st)
        length = max(max(r.shape[0] for r in rows), 1)
        embeds = torch.zeros(b, length, self.hidden_size, device=self.device, dtype=self.dtype)
        mask = torch.zeros(b, length, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            embeds[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        return embeds, mask, starts

    def lm_loss(self, prefixes: torch.Tensor | None, prompts: list[torch.Tensor],
                targets: list[torch.Tensor]) -> torch.Tensor:
        """Per-example mean NLL of `targets` given [prefix, prompt]. Returns [B]."""
        embeds, mask, starts = self._pack(prefixes, [[p, t] for p, t in zip(prompts, targets)])
        out = self.forward_embeds(embeds, mask)
        logp = F.log_softmax(out.logits.float(), dim=-1)
        losses = []
        for i, t in enumerate(targets):
            t = t.to(self.device)
            n = int(t.shape[0])
            if n == 0:
                losses.append(torch.zeros((), device=self.device))
                continue
            s = starts[i][1]
            if s == 0:
                raise ValueError("a target needs a prefix or a non-empty prompt before it")
            # position j predicts token j+1, so target token at s+q is read from s+q-1
            pred = logp[i, s - 1: s - 1 + n]
            losses.append(-pred.gather(-1, t[:, None]).squeeze(-1).mean())
        return torch.stack(losses)

    def summarize(self, text: str, prefix: torch.Tensor | None = None) -> torch.Tensor:
        """Masked-mean final hidden state of `text` -> [H] float32 (a retrieval key)."""
        ids = self.encode(text)
        with torch.no_grad():
            embeds, mask, starts = self._pack(None if prefix is None else prefix[None], [[ids]])
            out = self.forward_embeds(embeds, mask)
        s = starts[0][0]
        h = out.hidden[0, s: s + max(int(ids.shape[0]), 1)].float()
        return h.mean(dim=0)

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor | None, prompt: str, max_new_tokens: int) -> Generation:
        """Greedy decoding, batch size 1. Subclasses may override with a cached version."""
        prompt_ids = self.encode(prompt)
        generated: list[int] = []
        hiddens: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            new = torch.tensor(generated, dtype=torch.long)
            embeds, mask, _ = self._pack(None if prefix is None else prefix[None], [[prompt_ids, new]])
            out = self.forward_embeds(embeds, mask)
            last = int(mask[0].sum()) - 1
            tok = int(out.logits[0, last].argmax())
            hiddens.append(out.hidden[0, last].float())
            if self.eos_token_id is not None and tok == self.eos_token_id:
                break
            generated.append(tok)
        ids = torch.tensor(generated, dtype=torch.long)
        hidden = torch.stack(hiddens) if hiddens else torch.zeros(0, self.hidden_size)
        return Generation(self.decode(ids), ids, hidden)


class HFBackbone(Backbone):
    """A frozen Hugging Face causal LM. Optional 4-bit loading via bitsandbytes."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.requires_grad_(False)
        self.model.eval()
        emb = self.model.get_input_embeddings()
        self.hidden_size = int(emb.weight.shape[1])
        self.device = emb.weight.device
        self.dtype = emb.weight.dtype if emb.weight.dtype.is_floating_point else torch.bfloat16
        self.eos_token_id = tokenizer.eos_token_id
        with torch.no_grad():
            w = emb.weight.float()
            self._emb_rms = float(torch.sqrt((w * w).mean(dim=1)).mean())

    @classmethod
    def from_pretrained(cls, name: str, load_in_4bit: bool = True,
                        dtype: torch.dtype = torch.bfloat16, device_map: str = "auto",
                        gradient_checkpointing: bool = True) -> "HFBackbone":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict = {"torch_dtype": dtype, "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(model, AutoTokenizer.from_pretrained(name))

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids.to(self.device))

    def forward_embeds(self, embeds: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput:
        out = self.model(inputs_embeds=embeds.to(self.dtype), attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        return BackboneOutput(out.logits, out.hidden_states[-1])

    def embedding_rms(self) -> float:
        return self._emb_rms

    @contextmanager
    def gradient_mode(self):
        # HF only applies gradient checkpointing in train mode. Parameters stay
        # frozen; LLM configs normally have dropout 0, so train mode is safe.
        self.model.train()
        try:
            yield
        finally:
            self.model.eval()

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor | None, prompt: str, max_new_tokens: int) -> Generation:
        prompt_ids = self.encode(prompt)
        embeds, mask, _ = self._pack(None if prefix is None else prefix[None], [[prompt_ids]])
        out = self.model.generate(inputs_embeds=embeds, attention_mask=mask,
                                  max_new_tokens=max_new_tokens, do_sample=False,
                                  output_hidden_states=True, return_dict_in_generate=True,
                                  pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id)
        seq = out.sequences[0]
        # With inputs_embeds, decoder-only models return only the new tokens.
        n_new = min(len(out.hidden_states), int(seq.shape[0]))
        hidden = (torch.stack([out.hidden_states[i][-1][0, -1].float().cpu() for i in range(n_new)])
                  if n_new else torch.zeros(0, self.hidden_size))
        ids = (seq[-n_new:] if n_new else seq[:0]).cpu()
        if self.eos_token_id is not None and n_new:
            keep = ids != self.eos_token_id
            ids, hidden = ids[keep], hidden[keep]
        return Generation(self.decode(ids), ids, hidden)
