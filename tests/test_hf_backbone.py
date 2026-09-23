"""Exercises HFBackbone with a tiny Qwen2 model built locally (no download)."""

import string

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from continuous_brain.bridge import HFBackbone  # noqa: E402


@pytest.fixture(scope="module")
def hf():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

    vocab = {c: i for i, c in enumerate(string.printable)}
    vocab["<eos>"] = len(vocab)
    vocab["<unk>"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    torch.manual_seed(0)
    return HFBackbone(AutoModelForCausalLM.from_config(cfg), tok)


def test_frozen(hf):
    assert all(not p.requires_grad for p in hf.model.parameters())


def test_gradient_flows_to_prefix_only(hf):
    prefix = torch.randn(2, 4, hf.hidden_size, requires_grad=True)
    with hf.gradient_mode():
        loss = hf.lm_loss(prefix, [hf.encode("Q: hi"), hf.encode("Q: there")],
                          [hf.encode(" ok"), hf.encode(" sure")])
    loss.sum().backward()
    assert prefix.grad is not None and prefix.grad.abs().sum() > 0
    assert all(p.grad is None for p in hf.model.parameters())
    assert not hf.model.training   # gradient_mode restores eval


def test_generate_with_prefix(hf):
    g = hf.generate(torch.randn(4, hf.hidden_size), "Q: hi", 6)
    assert g.hidden.shape == (g.token_ids.shape[0], hf.hidden_size)


def test_summarize_shape(hf):
    assert hf.summarize("some text").shape == (hf.hidden_size,)
