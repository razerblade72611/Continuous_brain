from __future__ import annotations

from io import BytesIO
from typing import Any
import base64
import httpx

from .models import WorldState


def capture_frame_png(thor: Any, max_width: int = 512) -> bytes | None:
    """Capture the current RGB viewport without consulting object metadata."""
    event = thor.raw_event()
    frame = None if event is None else getattr(event, "frame", None)
    if frame is None:
        return None

    from PIL import Image

    image = Image.fromarray(frame)
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, round(image.height * ratio))))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def embodiment_snapshot(world: WorldState) -> dict[str, Any]:
    """
    Minimal state allowed into the autonomous LLM prompt.

    Object catalogs, object IDs, object types, visibility flags, distances, and
    receptacle metadata are intentionally excluded.
    """
    return {
        "pose": {
            "x": round(float(world.pose.position.get("x", 0.0)), 2),
            "z": round(float(world.pose.position.get("z", 0.0)), 2),
            "yaw": round(float(world.pose.rotation.get("y", 0.0)), 1),
            "horizon": round(float(world.pose.horizon), 1),
        },
        "holding_something": world.held_object_id is not None,
        "last_action": world.last_action,
        "last_action_success": world.last_action_success,
        # Do not pass raw simulator error text; AI2-THOR errors can contain object IDs/names.
        "last_error_present": bool(world.last_error),
        "step_index": world.step_index,
    }


class VisionProvider:
    """Backward-compatible direct visual description helper."""

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
