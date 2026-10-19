from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Optional


class ActionKind(str, Enum):
    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    PICKUP = "PickupObject"
    PUT = "PutObject"
    OPEN = "OpenObject"
    CLOSE = "CloseObject"
    WAIT = "Pass"
    NAVIGATE_TO = "NavigateTo"  # Agent-level pseudo-action; never sent directly to Unity.
    FACE_TARGET = "FaceTarget"  # Agent-level pseudo-action; never sent directly to Unity.


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "reason": self.reason}


@dataclass
class ActionOutcome:
    action: Action
    success: bool
    error: Optional[str] = None
    event_metadata: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    primitive_steps: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.as_dict(),
            "success": self.success,
            "error": self.error,
            "event_metadata": self.event_metadata,
            "verification": self.verification,
            "attempts": self.attempts,
            "primitive_steps": self.primitive_steps,
        }


@dataclass
class Plan:
    goal: str
    actions: list[Action]
    rationale: str = ""
    needs_replan: bool = False


@dataclass
class ObjectState:
    object_id: str
    object_type: str
    name: str
    visible: bool
    position: dict[str, float] | None
    distance: float | None
    pickupable: bool = False
    receptacle: bool = False
    openable: bool = False
    is_open: bool | None = None
    is_picked_up: bool = False
    parent_receptacles: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "name": self.name,
            "visible": self.visible,
            "position": self.position,
            "distance": self.distance,
            "pickupable": self.pickupable,
            "receptacle": self.receptacle,
            "openable": self.openable,
            "is_open": self.is_open,
            "is_picked_up": self.is_picked_up,
            "parent_receptacles": self.parent_receptacles,
        }


@dataclass
class AgentPose:
    position: dict[str, float]
    rotation: dict[str, float]
    horizon: float = 0.0


@dataclass
class WorldState:
    scene: str
    pose: AgentPose
    objects: dict[str, ObjectState]
    held_object_id: str | None
    last_action: str | None
    last_action_success: bool | None
    last_error: str | None
    frame_available: bool
    step_index: int = 0
    # Hidden grounding-only viewport evidence. Never serialize these IDs into LLM prompts.
    viewport_object_ids: set[str] = field(default_factory=set)
    viewport_tracking_available: bool = False

    @staticmethod
    def _canonical(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def find(self, object_type_or_name: str) -> list[ObjectState]:
        needle = self._canonical(object_type_or_name)
        if not needle:
            return []

        exact: list[ObjectState] = []
        partial: list[ObjectState] = []
        for obj in self.objects.values():
            obj_type = self._canonical(obj.object_type)
            obj_name = self._canonical(obj.name)
            if needle in {obj_type, obj_name}:
                exact.append(obj)
            elif needle in obj_type or needle in obj_name:
                partial.append(obj)
        return exact or partial

    def visible_objects(self) -> list[ObjectState]:
        return [obj for obj in self.objects.values() if obj.visible]

    def held_object(self) -> ObjectState | None:
        if not self.held_object_id:
            return None
        return self.objects.get(self.held_object_id)

    def compact(self, max_objects: int = 100) -> dict[str, Any]:
        objs = list(self.objects.values())
        objs.sort(key=lambda obj: (obj.distance is None, obj.distance or 9999))
        return {
            "scene": self.scene,
            "pose": {
                "position": self.pose.position,
                "rotation": self.pose.rotation,
                "horizon": self.pose.horizon,
            },
            "held_object_id": self.held_object_id,
            "last_action": self.last_action,
            "last_action_success": self.last_action_success,
            "last_error": self.last_error,
            "step_index": self.step_index,
            "objects": [obj.compact() for obj in objs[:max_objects]],
        }
