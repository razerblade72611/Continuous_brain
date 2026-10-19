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
    """OpenAI-compatible VLM teacher.

    Small local Qwen2-VL teachers use a deliberately compact visual-only
    protocol. Larger/cloud teachers retain the rich JSON protocol.
    """

    def __init__(
        self,
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="Qwen/Qwen3.8-27B",
        api_key=None,
        timeout=180,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = int(timeout)

    def _call(self, payload: dict) -> str:
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Qwen teacher endpoint failed: {exc}"
            ) from exc

        try:
            text = body["choices"][0]["message"]["content"]

            if isinstance(text, list):
                text = "".join(
                    str(part.get("text", ""))
                    if isinstance(part, dict)
                    else str(part)
                    for part in text
                )

            return str(text).strip()

        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected teacher API response"
            ) from exc

    def _encoded_image(self, image_path):
        path = Path(image_path)
        suffix = path.suffix.lower()

        mime = (
            "image/png"
            if suffix == ".png"
            else "image/jpeg"
        )

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")

        return mime, encoded

    def _is_small_local_teacher(self) -> bool:
        name = self.model.lower()

        return (
            "qwen2-vl-2b" in name
            or name.endswith("-local")
            or "2b-local" in name
        )

    def _label_small_local(
        self,
        image_path,
        context,
        allowed_text="",
    ):
        """Robust local Qwen2-VL visual teacher.

        Small local VLMs are used only as visual concept detectors.
        They are NOT trusted to infer goals, self-state, reasoning,
        values, or social policy.
        """

        mime, encoded = self._encoded_image(image_path)

        allowed = ", ".join(BASIC_CONCEPTS)

        prompt = f"""Look only at the supplied camera image.

Identify concepts that are visibly supported.

Allowed concepts:
{allowed}

Rules:
- Use only words from the allowed concept list.
- Do not infer the learner's goal.
- Do not tell the learner what action to take.
- Do not discuss morality or hidden simulator state.
- Do not repeat the entire allowed list.
- A short comma-separated answer is ideal.

Example:
door, chair, table

If genuinely uncertain, answer:
uncertain
"""

        last_text = ""

        # Object-like concepts that make sense in a natural visual sentence.
        visible_entities = {
            "person", "door", "chair", "table", "ball", "cup",
            "bottle", "book", "lamp", "sofa", "bed", "cabinet",
            "drawer", "counter", "sink", "fridge", "wall", "floor",
            "ceiling", "window", "plant", "picture", "screen", "box",
            "shelf", "room", "hallway", "opening", "obstacle", "target",
        }

        def article_phrase(word):
            if word in {"floor", "ceiling"}:
                return f"the {word}"
            article = "an" if word[:1].lower() in "aeiou" else "a"
            return f"{article} {word}"

        def natural_join(items):
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return ", ".join(items[:-1]) + f", and {items[-1]}"

        for attempt in range(3):

            payload = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 96,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative visual concept detector. "
                            "Report only things supported by the image."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{encoded}"
                                },
                            },
                        ],
                    },
                ],
            }

            text = self._call(payload).strip()
            last_text = text
            lowered = text.lower()

            concepts = []

            # Accept terse answers such as:
            #   Door, chair
            #
            # We ignore every word that is not in our fixed vocabulary.
            for concept in BASIC_CONCEPTS:
                pattern = (
                    r"(?<![A-Za-z0-9_])"
                    + re.escape(concept.lower())
                    + r"(?![A-Za-z0-9_])"
                )

                if re.search(pattern, lowered):
                    concepts.append(concept)

            # A response containing a huge fraction of our vocabulary is
            # probably the model parroting the prompt rather than observing.
            if len(concepts) > 16:
                continue

            if concepts:
                entities = [
                    c for c in concepts
                    if c in visible_entities
                ]

                if entities:
                    spoken = natural_join(
                        [article_phrase(c) for c in entities[:6]]
                    )
                    summary = f"I see {spoken}."
                else:
                    spoken = natural_join(concepts[:6])
                    summary = (
                        f"I see visual evidence associated with {spoken}."
                    )

                uncertainty = (
                    0.35
                    if "uncertain" in concepts
                    else 0.05
                )

                social = (
                    1.0
                    if "person" in concepts
                    else 0.0
                )

                label = {
                    "summary": summary,
                    "concepts": concepts,
                    "qas": [
                        {
                            "kind": "vision",
                            "question": "What do you see?",
                            "answer": summary,
                        }
                    ],
                    "care": {
                        "social_presence": social,
                        "help_needed": 0.0,
                        "harm_risk": 0.0,
                        "respect_request": 1.0,
                        "uncertainty": uncertainty,
                        "care_priority": 0.5,
                    },
                }

                return validate_teacher_label(label)

        # Do not abort a 3,500-frame run because the small model gives one
        # unusable answer. Store an explicit uncertainty lesson instead.
        label = {
            "summary": "I am uncertain about what is visible.",
            "concepts": ["uncertain"],
            "qas": [
                {
                    "kind": "vision",
                    "question": "What do you see?",
                    "answer": "I am uncertain about what is visible.",
                }
            ],
            "care": {
                "social_presence": 0.0,
                "help_needed": 0.0,
                "harm_risk": 0.0,
                "respect_request": 1.0,
                "uncertainty": 1.0,
                "care_priority": 0.5,
            },
        }

        return validate_teacher_label(label)

    def _label_rich(
        self,
        image_path,
        context,
        allowed_text="",
    ):
        """Rich structured teacher path for stronger cloud VLMs."""

        mime, encoded = self._encoded_image(image_path)

        last_error = None

        for attempt in range(3):
            prompt = teacher_prompt(
                context,
                allowed_text,
            )

            if attempt:
                prompt += (
                    "\nIMPORTANT: Your previous response was invalid "
                    "or incomplete. Return one complete JSON object only."
                )

            payload = {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 900,
                "messages": [
                    {
                        "role": "system",
                        "content": TEACHER_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{mime};base64,"
                                        f"{encoded}"
                                    )
                                },
                            },
                        ],
                    },
                ],
            }

            try:
                text = self._call(payload)

                return validate_teacher_label(
                    _extract_json(text)
                )

            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise ValueError(
            "Teacher failed to return valid structured output "
            f"after 3 attempts: {last_error}"
        )

    def label(
        self,
        image_path: str | Path,
        context: list[float],
        allowed_text: str = "",
    ) -> dict:

        if self._is_small_local_teacher():
            return self._label_small_local(
                image_path,
                context,
                allowed_text,
            )

        return self._label_rich(
            image_path,
            context,
            allowed_text,
        )
