"""A tiny random Qwen3 + character tokenizer that stands in for load_qwen() in CPU end-to-end tests of experiment
runners (real Being3, real loops, real autograd; scores are meaningless). Usage: import tiny_backbone; tiny_backbone.install()"""
import torch


class CharTok:
    """Byte-level tokenizer with the slice of the HF interface the runners use."""
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        ids = [1] + [2 + (b % 254) for b in text.encode()]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return bytes([max(0, int(i) - 2) for i in ids if int(i) > 1]).decode(errors="ignore")


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                      tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def install():
    from being.v3 import core
    model, tok = tiny_model(), CharTok()
    core.load_qwen = lambda *a, **k: (model, tok)
    return model, tok
