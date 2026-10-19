from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


class VisionBackbone(ABC):
    """Frozen visual feature provider.

    Backbones return a single normalized float32 embedding for an RGB uint8 frame.
    The Continuous Brain trains only its small adapter by default.
    """

    name: str = "vision"
    output_dim: int = 0
    pretrained: bool = False

    @abstractmethod
    def encode(self, rgb: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dim": int(self.output_dim),
            "pretrained": bool(self.pretrained),
        }


class NullVisionBackbone(VisionBackbone):
    name = "none"
    output_dim = 0
    pretrained = False

    def encode(self, rgb: np.ndarray) -> None:
        return None


class PixelStatsBackbone(VisionBackbone):
    """Dependency-free visual pipeline used to validate v0.3 without model downloads.

    This is deliberately *not* presented as pretrained intelligence. It spatially pools
    the rendered frame into a 4x4 RGB grid so the adapter/memory/curiosity plumbing can
    be exercised before a foundation model is installed.
    """

    name = "pixel-stats"
    output_dim = 48
    pretrained = False

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(rgb, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got {x.shape}")
        x = x / 255.0
        h, w, _ = x.shape
        ys = np.linspace(0, h, 5, dtype=int)
        xs = np.linspace(0, w, 5, dtype=int)
        pooled = []
        for yi in range(4):
            for xi in range(4):
                cell = x[ys[yi]:ys[yi + 1], xs[xi]:xs[xi + 1]]
                pooled.extend(cell.mean(axis=(0, 1)).tolist())
        v = np.asarray(pooled, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / max(norm, 1e-8)


@dataclass
class DinoV3Backbone(VisionBackbone):
    """Frozen Meta DINOv3 vision foundation model via Hugging Face Transformers."""

    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    device_name: str = "auto"

    name: str = "dinov3-vits16"
    pretrained: bool = True

    def __post_init__(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 requires the optional vision dependencies. Install with: "
                "python -m pip install 'transformers>=4.56' pillow"
            ) from exc

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("DINOv3 requires Pillow: python -m pip install pillow") from exc

        self._Image = Image
        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id)
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINOv3. The official Meta model is gated: accept its "
                "Hugging Face license/access terms and authenticate, then retry. "
                f"Model: {self.model_id}. Original error: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.output_dim = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "hidden_dim", 0)
            or 384
        )

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        image = self._Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("DINOv3 output did not contain pooler_output or last_hidden_state")
            pooled = hidden[:, 0]
        v = pooled[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        norm = float(np.linalg.norm(v))
        return v / max(norm, 1e-8)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"model_id": self.model_id, "device": str(self.device)})
        return out


def build_vision_backbone(name: str, *, device: str = "auto", model_id: str | None = None) -> VisionBackbone:
    key = (name or "none").strip().lower()
    if key in {"none", "off", "disabled"}:
        return NullVisionBackbone()
    if key in {"pixel", "pixel-stats", "stats"}:
        return PixelStatsBackbone()
    if key in {"dinov3", "dinov3-vits16", "dino-v3"}:
        return DinoV3Backbone(
            model_id=model_id or "facebook/dinov3-vits16-pretrain-lvd1689m",
            device_name=device,
        )
    raise ValueError(f"unknown vision backbone: {name!r}")
