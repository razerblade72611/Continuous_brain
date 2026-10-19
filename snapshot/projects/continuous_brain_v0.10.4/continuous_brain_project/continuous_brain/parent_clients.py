"""External parent adapters. These clients never belong to the child checkpoint."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.request

import numpy as np


class OpenAIChatParent:
    def __init__(self, endpoint: str, model: str, *, timeout=300):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def ask(self, prompt: str, *, image_path: str | Path | None = None, max_tokens=256) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            p = Path(image_path)
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())
        return str(data["choices"][0]["message"]["content"]).strip()


class DinoVisionParent:
    """Frozen DINOv3 embedding parent, lazily loaded to keep core dependencies clean."""
    def __init__(self, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", device="cpu"):
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINO parent requires the vision optional dependencies") from exc
        self.torch = torch
        self.Image = Image
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.model_id = model_id

    @property
    def output_dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

    def embed(self, image_path: str | Path) -> np.ndarray:
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model(**inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled[0].float().cpu().numpy().astype(np.float32)
