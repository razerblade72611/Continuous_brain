from __future__ import annotations

from typing import Any

from .models import AgentPose, ObjectState, WorldState


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _xyz(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "x": _safe_float(raw.get("x")) or 0.0,
        "y": _safe_float(raw.get("y")) or 0.0,
        "z": _safe_float(raw.get("z")) or 0.0,
    }


def normalize_event(event: Any, scene: str, step_index: int = 0) -> WorldState:
    md = getattr(event, "metadata", {}) or {}
    agent = md.get("agent", {}) or {}

    pose = AgentPose(
        position=_xyz(agent.get("position")) or {"x": 0.0, "y": 0.0, "z": 0.0},
        rotation=_xyz(agent.get("rotation")) or {"x": 0.0, "y": 0.0, "z": 0.0},
        horizon=_safe_float(agent.get("cameraHorizon")) or 0.0,
    )

    objects: dict[str, ObjectState] = {}
    held: str | None = None

    for raw in md.get("objects", []) or []:
        oid = str(raw.get("objectId", ""))
        if not oid:
            continue
        obj = ObjectState(
            object_id=oid,
            object_type=str(raw.get("objectType", "")),
            name=str(raw.get("name", raw.get("objectType", oid))),
            visible=bool(raw.get("visible", False)),
            position=_xyz(raw.get("position")),
            distance=_safe_float(raw.get("distance")),
            pickupable=bool(raw.get("pickupable", False)),
            receptacle=bool(raw.get("receptacle", False)),
            openable=bool(raw.get("openable", False)),
            is_open=raw.get("isOpen"),
            is_picked_up=bool(raw.get("isPickedUp", False)),
            toggleable=bool(raw.get("toggleable", False)),
            is_toggled=raw.get("isToggled"),
            parent_receptacles=list(raw.get("parentReceptacles") or []),
            raw=raw,
        )
        objects[oid] = obj
        if obj.is_picked_up:
            held = oid

    detections = getattr(event, "instance_detections2D", None)
    masks = getattr(event, "instance_masks", None)
    viewport_tracking_available = detections is not None or masks is not None
    viewport_object_ids: set[str] = set()
    if isinstance(detections, dict):
        viewport_object_ids.update(str(key) for key in detections.keys())
    if isinstance(masks, dict):
        viewport_object_ids.update(str(key) for key in masks.keys())

    return WorldState(
        scene=str(md.get("sceneName") or scene),
        pose=pose,
        objects=objects,
        held_object_id=held,
        last_action=md.get("lastAction"),
        last_action_success=md.get("lastActionSuccess"),
        last_error=md.get("errorMessage") or None,
        frame_available=getattr(event, "frame", None) is not None,
        step_index=step_index,
        viewport_object_ids=viewport_object_ids,
        viewport_tracking_available=viewport_tracking_available,
    )
