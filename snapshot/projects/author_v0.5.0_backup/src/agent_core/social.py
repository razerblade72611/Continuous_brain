from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .models import WorldState


@dataclass(frozen=True)
class HeardSpeech:
    speaker_id: int
    speaker_label: str
    text: str
    distance_m: float

    def for_llm(self) -> dict:
        return {
            "speaker": self.speaker_label,
            "text": self.text,
            "distance_m": round(self.distance_m, 2),
        }


class SpeechBus:
    """
    Physical-range speech transport.

    It carries only utterances. It never copies another agent's camera, prompt,
    memory, semantic focus, body state, task, or model output.
    """

    def __init__(self, labels: Mapping[int, str], speech_range_m: float = 4.0):
        self.labels = {int(k): str(v) for k, v in labels.items()}
        self.speech_range_m = float(speech_range_m)
        self._inboxes: dict[int, list[HeardSpeech]] = {agent_id: [] for agent_id in self.labels}

    @staticmethod
    def _distance(a: WorldState, b: WorldState) -> float:
        ax = float(a.pose.position.get("x", 0.0))
        az = float(a.pose.position.get("z", 0.0))
        bx = float(b.pose.position.get("x", 0.0))
        bz = float(b.pose.position.get("z", 0.0))
        return math.hypot(ax - bx, az - bz)

    def speak(self, speaker_id: int, text: str, worlds: Mapping[int, WorldState]) -> list[HeardSpeech]:
        speaker_id = int(speaker_id)
        utterance = " ".join(str(text).split()).strip()[:500]
        if not utterance:
            return []
        if speaker_id not in worlds:
            return []
        delivered: list[HeardSpeech] = []
        for receiver_id in self.labels:
            if receiver_id == speaker_id or receiver_id not in worlds:
                continue
            distance = self._distance(worlds[speaker_id], worlds[receiver_id])
            if distance > self.speech_range_m:
                continue
            event = HeardSpeech(
                speaker_id=speaker_id,
                speaker_label=self.labels.get(speaker_id, f"agent-{speaker_id}"),
                text=utterance,
                distance_m=distance,
            )
            self._inboxes.setdefault(receiver_id, []).append(event)
            delivered.append(event)
        return delivered

    def consume(self, receiver_id: int) -> list[HeardSpeech]:
        receiver_id = int(receiver_id)
        items = list(self._inboxes.get(receiver_id, []))
        self._inboxes[receiver_id] = []
        return items

    def peek(self, receiver_id: int) -> list[HeardSpeech]:
        return list(self._inboxes.get(int(receiver_id), []))
