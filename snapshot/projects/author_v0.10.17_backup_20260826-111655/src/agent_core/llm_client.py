from __future__ import annotations

import base64
import os
from typing import Any

import httpx


class SyncLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("AGENT_LLM_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"

        def send(body: dict[str, Any]) -> httpx.Response:
            return httpx.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )

        response = send(payload)

        # Some llama.cpp/model combinations reject response_format even though the
        # OpenAI-compatible endpoint otherwise works. The roam prompt already
        # requests JSON and the parser extracts JSON defensively, so retry once
        # without response_format before falling back.
        if response.status_code == 400 and "response_format" in payload:
            retry_payload = dict(payload)
            retry_payload.pop("response_format", None)
            response = send(retry_payload)

        if response.is_error:
            body = response.text.strip()
            if len(body) > 1200:
                body = body[:1200] + "...[truncated]"
            raise RuntimeError(
                f"LLM HTTP {response.status_code} from {url}: {body or '<empty response body>'}"
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        return self._post(payload)

    def chat_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        return self._post(payload)
