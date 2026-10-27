"""External retrieval keys (published K1): Qwen3-Embedding with its documented last-token pooling and L2 normalisation.
Queries carry an instruction prefix; stored episodes are embedded without one (as the model card prescribes)."""
import torch
import torch.nn.functional as F

QUERY_INSTRUCTION = "Given a question or cue, retrieve the remembered experience that answers it"


class ExternalKeys:
    def __init__(self, path, device, dtype):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts, query=False):
        if query:
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{t}" for t in texts]
        b = self.tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.device)
        h = self.model(**b).last_hidden_state
        # left padding: the last position is every sequence's last real token
        v = h[:, -1].float()
        return F.normalize(v, dim=-1).cpu()
