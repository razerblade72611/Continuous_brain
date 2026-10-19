from __future__ import annotations

import base64
from typing import Any
import httpx


class VisionProvider:
    """
    Optional visual specialist. It does not define world truth.
    It asks a vision-capable OpenAI-compatible endpoint to inspect a JPEG/PNG frame.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 120.0, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def describe(self, image_bytes: bytes, prompt: str) -> str:
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "temperature": 0.0,
        }
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
