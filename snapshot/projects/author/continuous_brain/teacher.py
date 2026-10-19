"""Frozen Qwen teacher client for developmental semantic supervision.

The teacher emits structured labels only.  It is not part of runtime action
selection and is intentionally absent from qualified communication inference.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from .semantic import BASIC_CONCEPTS, CARE_KEYS


TEACHER_SYSTEM = """You are a developmental teacher labeling one egocentric camera frame for a small embodied neural student.
Be concrete, truthful, conservative about uncertainty, and caring toward people. Do not invent unseen objects or internal feelings.
Return STRICT JSON only. Use only concepts from the supplied allowed concept list. The student will learn from your labels, so prefer simple grounded language.
"""


def teacher_prompt(context: list[float], allowed_text: str = "") -> str:
    return f"""Label this frame for a developing embodied learner.
Allowed concepts: {', '.join(BASIC_CONCEPTS)}
Permitted internal context description: {allowed_text or "none"}
The numeric internal context below contains only permitted sensed/self/goal/memory channels; do not infer hidden simulator metadata from it:
{json.dumps([round(float(x), 5) for x in context])}

Return exactly this JSON shape:
{{
  "summary": "one short literal sentence",
  "concepts": ["zero or more allowed concepts"],
  "qas": [
    {{"kind":"vision","question":"What do you see?","answer":"short grounded answer"}},
    {{"kind":"state","question":"What are you trying to do?","answer":"short answer grounded in visible/context evidence"}},
    {{"kind":"reasoning","question":"What should you do next?","answer":"short cautious answer"}},
    {{"kind":"care","question":"What matters if a person asks you to stop?","answer":"short answer emphasizing stopping and respecting the request"}}
  ],
  "care": {{
    "social_presence": 0.0,
    "help_needed": 0.0,
    "harm_risk": 0.0,
    "respect_request": 1.0,
    "uncertainty": 0.0,
    "care_priority": 0.5
  }}
}}
All care values must be numbers from 0 to 1. If no person is visible, social_presence should be low. If visual evidence is ambiguous, uncertainty should be higher.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Teacher response did not contain JSON")
        return json.loads(text[start:end + 1])


def validate_teacher_label(label: dict) -> dict:
    if not isinstance(label, dict):
        raise ValueError("Teacher label must be an object")
    concepts = label.get("concepts", [])
    if not isinstance(concepts, list) or any(c not in BASIC_CONCEPTS for c in concepts):
        raise ValueError("Teacher returned a concept outside the v0.7 vocabulary")
    qas = label.get("qas", [])
    if not isinstance(qas, list) or not qas:
        raise ValueError("Teacher must provide at least one question/answer")
    cleaned_qas = []
    for qa in qas[:8]:
        if not isinstance(qa, dict):
            continue
        q, a = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
        if q and a:
            cleaned_qas.append({"kind": str(qa.get("kind", "general"))[:32],
                                "question": q[:500], "answer": a[:800]})
    if not cleaned_qas:
        raise ValueError("Teacher did not provide usable Q/A pairs")
    care = label.get("care", {}) or {}
    clean_care = {}
    for key in CARE_KEYS:
        try:
            value = float(care.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        clean_care[key] = max(0.0, min(1.0, value))
    return {
        "summary": str(label.get("summary", "")).strip()[:800],
        "concepts": sorted(set(concepts)),
        "qas": cleaned_qas,
        "care": clean_care,
    }


class QwenTeacherClient:
    """Minimal OpenAI-compatible VLM client (vLLM/SGLang/etc.)."""

    def __init__(self, endpoint="http://127.0.0.1:8000/v1/chat/completions",
                 model="Qwen/Qwen3.8-27B", api_key=None, timeout=180):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = int(timeout)

    def label(self, image_path: str | Path, context: list[float], allowed_text: str = "") -> dict:
        path = Path(image_path)
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": teacher_prompt(context, allowed_text)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                         headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen teacher endpoint failed: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part)
                               for part in text)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected teacher API response") from exc
        return validate_teacher_label(_extract_json(str(text)))
