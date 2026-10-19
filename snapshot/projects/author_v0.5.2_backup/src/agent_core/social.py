from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .models import WorldState
from .navigation import facing_yaw


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


@dataclass(frozen=True)
class PeerPresence:
    """
    Geometry-only social sensing between embodied bodies.

    `in_camera_cone` means the peer body center falls inside the observer's
    horizontal camera field-of-view by pose geometry. It intentionally does not
    claim an unoccluded pixel-level detection. The RGB camera remains the source
    for appearance.
    """

    peer_id: int
    peer_label: str
    distance_m: float
    bearing_degrees: float
    in_camera_cone: bool
    within_hearing_range: bool

    def for_llm(self) -> dict:
        if not self.in_camera_cone:
            return {"another_embodied_agent_in_camera_cone": False}
        return {
            "another_embodied_agent_in_camera_cone": True,
            "distance_m": round(self.distance_m, 2),
            "bearing_degrees": round(self.bearing_degrees, 1),
            "sensor_note": (
                "geometry says another embodied body's center is inside the camera cone; "
                "use the attached RGB image for its appearance and possible occlusion"
            ),
        }

    def for_observer(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "peer_label": self.peer_label,
            "distance_m": round(self.distance_m, 2),
            "bearing_degrees": round(self.bearing_degrees, 1),
            "in_camera_cone": bool(self.in_camera_cone),
            "within_hearing_range": bool(self.within_hearing_range),
        }


def planar_agent_distance(a: WorldState, b: WorldState) -> float:
    ax = float(a.pose.position.get("x", 0.0))
    az = float(a.pose.position.get("z", 0.0))
    bx = float(b.pose.position.get("x", 0.0))
    bz = float(b.pose.position.get("z", 0.0))
    return math.hypot(ax - bx, az - bz)


def signed_bearing_to_peer(observer: WorldState, peer: WorldState) -> float:
    """Signed yaw error in degrees: negative=left, positive=right."""
    desired = facing_yaw(observer.pose.position, peer.pose.position)
    current = float(observer.pose.rotation.get("y", 0.0)) % 360.0
    return (desired - current + 180.0) % 360.0 - 180.0


def peer_presence_for(
    observer_id: int,
    worlds: Mapping[int, WorldState],
    labels: Mapping[int, str],
    *,
    field_of_view_degrees: float = 90.0,
    speech_range_m: float = 4.0,
    camera_cone_margin_degrees: float = 3.0,
) -> list[PeerPresence]:
    """Return neutral pose-geometry facts for the other embodied bodies."""
    observer_id = int(observer_id)
    if observer_id not in worlds:
        return []
    observer = worlds[observer_id]
    half_fov = max(1.0, float(field_of_view_degrees) / 2.0 + float(camera_cone_margin_degrees))
    out: list[PeerPresence] = []
    for peer_id in sorted(worlds):
        if int(peer_id) == observer_id:
            continue
        peer = worlds[peer_id]
        distance = planar_agent_distance(observer, peer)
        bearing = signed_bearing_to_peer(observer, peer)
        out.append(
            PeerPresence(
                peer_id=int(peer_id),
                peer_label=str(labels.get(int(peer_id), f"agent-{peer_id}")),
                distance_m=distance,
                bearing_degrees=bearing,
                in_camera_cone=abs(bearing) <= half_fov,
                within_hearing_range=distance <= float(speech_range_m),
            )
        )
    return out


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
        return planar_agent_distance(a, b)

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
