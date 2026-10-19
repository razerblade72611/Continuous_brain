from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
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
        return {"speaker": self.speaker_label, "text": self.text, "distance_m": round(self.distance_m, 2)}


@dataclass(frozen=True)
class TextMessage:
    sender_id: int
    sender_label: str
    recipient_id: int
    recipient_label: str
    text: str
    timestamp: str

    def for_llm(self) -> dict:
        return {"from": self.sender_label, "text": self.text, "channel": "text"}

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PeerPresence:
    peer_id: int
    peer_label: str
    distance_m: float
    bearing_degrees: float
    in_camera_cone: bool
    within_hearing_range: bool

    def for_llm(self) -> dict:
        base = {
            "peer_label": self.peer_label,
            "peer_is_another_independently_embodied_agent_like_you": True,
            "in_camera_cone": bool(self.in_camera_cone),
        }
        if not self.in_camera_cone:
            return base
        base.update({
            "distance_m": round(self.distance_m, 2),
            "bearing_degrees": round(self.bearing_degrees, 1),
            "within_hearing_range": bool(self.within_hearing_range),
            "sensor_note": "this geometry is the body of the named peer; use the attached RGB image for its appearance and possible occlusion",
        })
        return base

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
    ax = float(a.pose.position.get("x", 0.0)); az = float(a.pose.position.get("z", 0.0))
    bx = float(b.pose.position.get("x", 0.0)); bz = float(b.pose.position.get("z", 0.0))
    return math.hypot(ax - bx, az - bz)


def signed_bearing_to_peer(observer: WorldState, peer: WorldState) -> float:
    desired = facing_yaw(observer.pose.position, peer.pose.position)
    current = float(observer.pose.rotation.get("y", 0.0)) % 360.0
    return (desired - current + 180.0) % 360.0 - 180.0


def peer_presence_for(observer_id: int, worlds: Mapping[int, WorldState], labels: Mapping[int, str], *, field_of_view_degrees: float = 90.0, speech_range_m: float = 4.0, camera_cone_margin_degrees: float = 3.0) -> list[PeerPresence]:
    observer_id = int(observer_id)
    if observer_id not in worlds: return []
    observer = worlds[observer_id]
    half_fov = max(1.0, float(field_of_view_degrees) / 2.0 + float(camera_cone_margin_degrees))
    out=[]
    for peer_id in sorted(worlds):
        if int(peer_id)==observer_id: continue
        peer=worlds[peer_id]
        distance=planar_agent_distance(observer,peer); bearing=signed_bearing_to_peer(observer,peer)
        out.append(PeerPresence(int(peer_id), str(labels.get(int(peer_id), f"agent-{peer_id}")), distance, bearing, abs(bearing)<=half_fov, distance<=float(speech_range_m)))
    return out


class SpeechBus:
    def __init__(self, labels: Mapping[int, str], speech_range_m: float = 4.0):
        self.labels={int(k):str(v) for k,v in labels.items()}; self.speech_range_m=float(speech_range_m)
        self._inboxes={agent_id:[] for agent_id in self.labels}

    def speak(self, speaker_id: int, text: str, worlds: Mapping[int, WorldState]) -> list[HeardSpeech]:
        speaker_id=int(speaker_id); utterance=" ".join(str(text).split()).strip()[:500]
        if not utterance or speaker_id not in worlds: return []
        delivered=[]
        for receiver_id in self.labels:
            if receiver_id==speaker_id or receiver_id not in worlds: continue
            distance=planar_agent_distance(worlds[speaker_id], worlds[receiver_id])
            if distance>self.speech_range_m: continue
            event=HeardSpeech(speaker_id,self.labels.get(speaker_id,f"agent-{speaker_id}"),utterance,distance)
            self._inboxes.setdefault(receiver_id,[]).append(event); delivered.append(event)
        return delivered

    def consume(self, receiver_id: int) -> list[HeardSpeech]:
        receiver_id=int(receiver_id); items=list(self._inboxes.get(receiver_id,[])); self._inboxes[receiver_id]=[]; return items

    def peek(self, receiver_id: int) -> list[HeardSpeech]:
        return list(self._inboxes.get(int(receiver_id),[]))


class TextBus:
    """Distance-independent explicit text transport with persistent public message log.

    The bus carries only words deliberately emitted through the TextPeer action. It
    never transfers camera frames, hidden state, prompts, memory, or chain-of-thought.
    """
    def __init__(self, labels: Mapping[int,str], log_path: str|Path|None=None, max_chars: int=800):
        self.labels={int(k):str(v) for k,v in labels.items()}
        self._by_label={label.lower():aid for aid,label in self.labels.items()}
        self._inboxes={aid:[] for aid in self.labels}
        self.log_path=Path(log_path) if log_path is not None else None
        self.max_chars=max(32,int(max_chars))
        if self.log_path is not None: self.log_path.parent.mkdir(parents=True,exist_ok=True)

    def recipient_id(self, recipient: str|int, sender_id: int|None=None) -> int|None:
        if isinstance(recipient,int): rid=recipient
        else:
            text=str(recipient or '').strip().lower()
            rid=self._by_label.get(text)
            if rid is None and text.isdigit(): rid=int(text)
        if rid not in self.labels or (sender_id is not None and rid==int(sender_id)): return None
        return rid

    def send(self, sender_id:int, recipient:str|int, text:str) -> TextMessage|None:
        sender_id=int(sender_id); rid=self.recipient_id(recipient,sender_id=sender_id)
        body=" ".join(str(text).split()).strip()[:self.max_chars]
        if sender_id not in self.labels or rid is None or not body: return None
        event=TextMessage(sender_id,self.labels[sender_id],rid,self.labels[rid],body,datetime.now(timezone.utc).isoformat())
        self._inboxes.setdefault(rid,[]).append(event)
        if self.log_path is not None:
            with self.log_path.open('a',encoding='utf-8') as f: f.write(json.dumps(event.as_dict(),ensure_ascii=False)+"\n")
        return event

    def consume(self, receiver_id:int) -> list[TextMessage]:
        receiver_id=int(receiver_id); items=list(self._inboxes.get(receiver_id,[])); self._inboxes[receiver_id]=[]; return items

    def peek(self, receiver_id:int) -> list[TextMessage]:
        return list(self._inboxes.get(int(receiver_id),[]))
