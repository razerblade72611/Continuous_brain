"""Frozen VLM teacher clients for developmental semantic supervision.

Small local Qwen3-VL teachers are deliberately used only as visual concept
sensors.  Stronger OpenAI-compatible VLMs may emit richer JSON, but runtime
communication never calls the teacher.
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

Return one complete JSON object with:
- summary: one short literal visual sentence
- concepts: zero or more allowed concepts
- qas: at least one grounded visual question/answer
- care: numeric values from 0 to 1 for {', '.join(CARE_KEYS)}
Do not copy placeholder language. Keep answers short.
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
        raise ValueError("Teacher returned a concept outside the fixed vocabulary")
    qas = label.get("qas", [])
    if not isinstance(qas, list) or not qas:
        raise ValueError("Teacher must provide at least one question/answer")
    cleaned_qas = []
    for qa in qas[:8]:
        if not isinstance(qa, dict):
            continue
        q, a = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
        if q and a:
            cleaned_qas.append({"kind": str(qa.get("kind", "vision"))[:32], "question": q[:500], "answer": a[:800]})
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
    """OpenAI-compatible visual teacher with a robust small-local profile."""

    def __init__(self, endpoint="http://127.0.0.1:8000/v1/chat/completions",
                 model="Qwen/Qwen3.8-27B", api_key=None, timeout=180):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = int(timeout)

    def _call(self, payload: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen teacher endpoint failed: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in text)
            return str(text).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected teacher API response") from exc

    def _encoded_image(self, image_path):
        path = Path(image_path)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return mime, base64.b64encode(path.read_bytes()).decode("ascii")

    def _small_local(self) -> bool:
        name = self.model.lower()
        return (
            "qwen3-vl-8b" in name
            or "qwen2-vl-2b" in name
            or name.endswith("-local")
            or "2b-local" in name
        )

    def _label_small_local(self, image_path, context, allowed_text=""):
        mime, encoded = self._encoded_image(image_path)
        allowed = ", ".join(BASIC_CONCEPTS)
        prompt = f"""Look only at the supplied camera image.
Identify concepts that are visibly supported.
Allowed concepts: {allowed}
Use only words from that list. A short comma-separated answer is ideal. Do not infer goals or tell the learner what to do.
If genuinely uncertain, answer: uncertain
"""
        visible_entities = {
            "person", "door", "chair", "table", "ball", "cup", "bottle", "book", "lamp", "sofa", "bed",
            "cabinet", "drawer", "counter", "sink", "fridge", "wall", "floor", "ceiling", "window", "plant",
            "picture", "screen", "box", "shelf", "room", "hallway", "opening", "obstacle", "target",
        }
        for _ in range(3):
            payload = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 96,
                "messages": [
                    {"role": "system", "content": "You are a conservative visual concept detector. Report only image evidence."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ]},
                ],
            }
            text = self._call(payload).strip().lower()
            concepts = []
            for concept in BASIC_CONCEPTS:
                if re.search(r"(?<![A-Za-z0-9_])" + re.escape(concept.lower()) + r"(?![A-Za-z0-9_])", text):
                    concepts.append(concept)
            if len(concepts) > 16:
                continue
            if concepts:
                entities = [c for c in concepts if c in visible_entities][:6]
                if entities:
                    phrases = [("an " if c[:1] in "aeiou" else "a ") + c for c in entities]
                    if len(phrases) == 1:
                        joined = phrases[0]
                    elif len(phrases) == 2:
                        joined = " and ".join(phrases)
                    else:
                        joined = ", ".join(phrases[:-1]) + ", and " + phrases[-1]
                    summary = "I see " + joined + "."
                else:
                    summary = "I see visual evidence associated with " + ", ".join(concepts[:6]) + "."
                return validate_teacher_label({
                    "summary": summary,
                    "concepts": concepts,
                    "qas": [{"kind": "vision", "question": "What do you see?", "answer": summary}],
                    "care": {"social_presence": 1.0 if "person" in concepts else 0.0,
                             "help_needed": 0.0, "harm_risk": 0.0, "respect_request": 1.0,
                             "uncertainty": 0.35 if "uncertain" in concepts else 0.05,
                             "care_priority": 0.5},
                })
        return validate_teacher_label({
            "summary": "I am uncertain about what is visible.",
            "concepts": ["uncertain"],
            "qas": [{"kind": "vision", "question": "What do you see?", "answer": "I am uncertain about what is visible."}],
            "care": {"social_presence": 0.0, "help_needed": 0.0, "harm_risk": 0.0,
                     "respect_request": 1.0, "uncertainty": 1.0, "care_priority": 0.5},
        })

    def _label_rich(self, image_path, context, allowed_text=""):
        mime, encoded = self._encoded_image(image_path)
        last_error = None
        for attempt in range(3):
            prompt = teacher_prompt(context, allowed_text)
            if attempt:
                prompt += "\nPrevious output was invalid. Return one complete JSON object only."
            payload = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 900,
                "messages": [
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ]},
                ],
            }
            try:
                return validate_teacher_label(_extract_json(self._call(payload)))
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Teacher failed structured output after 3 attempts: {last_error}")

    def label(self, image_path: str | Path, context: list[float], allowed_text: str = "") -> dict:
        if self._small_local():
            return self._label_small_local(image_path, context, allowed_text)
        return self._label_rich(image_path, context, allowed_text)
