from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaPlanner:
    """Optional slow-timescale planner. The neural policy keeps running if Ollama is unavailable."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434/api/generate", timeout: float = 2.0):
        self.model = model
        self.url = url
        self.timeout = timeout

    def choose_action(self, state_description: str, action_names: tuple[str, ...], memories: list[dict]) -> int | None:
        actions = ", ".join(f"{i}:{name}" for i, name in enumerate(action_names))
        memory_text = json.dumps(memories[:3], ensure_ascii=False)
        prompt = (
            "You are a slow planner inside a continuously running embodied agent. "
            "Choose exactly one action number. Do not explain.\n"
            f"State: {state_description}\nActions: {actions}\nRelevant memories: {memory_text}\nAction number:"
        )
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = str(data.get("response", "")).strip()
            first = text.split()[0].strip(".,:;[]()")
            idx = int(first)
            return idx if 0 <= idx < len(action_names) else None
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            return None
