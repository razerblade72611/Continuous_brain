"""Builds a tiny random Qwen3 checkpoint WITH a real HF byte-level BPE tokenizer in a directory, so scripts that call
AutoModelForCausalLM/AutoTokenizer.from_pretrained(path) can be exercised end to end on CPU (scores are meaningless)."""
import sys
from pathlib import Path


def build(path, hidden=64, inter=128, layers=2):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = ["The quick brown fox works as a Pemyiasnind by trade.", "Reconstruct the story:",
              "On the third morning of the thaw a compass was traded for pears at the river market."] * 20
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<eos>", "<pad>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", pad_token="<pad>")
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return path


if __name__ == "__main__":
    print(build(sys.argv[1]))
