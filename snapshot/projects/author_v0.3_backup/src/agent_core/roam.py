from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import re
from typing import Any

from .executor import ActionExecutor
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .models import Action, ActionKind, ObjectState, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter


ROAM_SYSTEM_PROMPT = """
You control an embodied agent exploring an AI2-THOR 3D environment.
There is no task to finish. Continue exploring until the external controller stops you.

At each decision, choose exactly ONE useful next action. Prefer discovering new areas,
viewpoints, rooms, and objects instead of repeating the same motion.

Return ONLY JSON in this form:
{
  "reason": "brief operational reason, not hidden chain-of-thought",
  "action": {
    "kind": "NavigateTo",
    "params": {"objectId": "exact object id from world state"}
  }
}

Supported navigation/look actions:
NavigateTo, MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight,
LookUp, LookDown, Pass.

If interactions are enabled, these are also supported:
PickupObject, PutObject, OpenObject, CloseObject.

Rules:
- Never invent object IDs.
- NavigateTo is the preferred way to move to a specific object or area anchor.
- Use RotateLeft/RotateRight/LookUp/LookDown to survey the current view.
- Avoid repeating recent actions or revisiting the same target without a reason.
- If carrying an object and interactions are enabled, you may place it on/in a valid receptacle.
- PutObject.objectId is the receptacle ID, not the held object ID.
- Keep the reason short. Always return one action.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Roam response was not JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def _candidate_objects(world: WorldState, limit: int = 60) -> list[dict[str, Any]]:
    objects = list(world.objects.values())
    objects.sort(key=lambda obj: (not obj.visible, obj.distance is None, obj.distance or 9999.0))
    result: list[dict[str, Any]] = []
    for obj in objects[:limit]:
        result.append(
            {
                "object_id": obj.object_id,
                "type": obj.object_type,
                "name": obj.name,
                "visible": obj.visible,
                "distance": obj.distance,
                "position": obj.position,
                "pickupable": obj.pickupable,
                "openable": obj.openable,
                "is_open": obj.is_open,
                "receptacle": obj.receptacle,
                "is_picked_up": obj.is_picked_up,
            }
        )
    return result


def frame_to_png(thor: ThorAdapter, max_width: int = 512) -> bytes | None:
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


@dataclass
class RoamDecision:
    action: Action
    reason: str
    used_vision: bool = False
    fallback: bool = False


@dataclass
class RoamResult:
    decisions: int
    primitive_steps: int
    reason: str


class RoamPlanner:
    def __init__(
        self,
        client: SyncLLMClient | None,
        interaction_distance: float = 1.25,
        allow_interactions: bool = False,
    ):
        self.client = client
        self.interaction_distance = interaction_distance
        self.allow_interactions = allow_interactions
        self._fallback_cursor = 0

    def _validate(self, data: dict[str, Any], world: WorldState) -> RoamDecision:
        row = data.get("action") or {}
        kind = str(row.get("kind", ""))
        params = dict(row.get("params") or {})
        reason = str(data.get("reason", row.get("reason", "Explore.")))[:300]

        navigation = {
            ActionKind.NAVIGATE_TO.value,
            ActionKind.MOVE_AHEAD.value,
            ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value,
            ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value,
            ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value,
            ActionKind.LOOK_DOWN.value,
            ActionKind.WAIT.value,
        }
        interactions = {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }
        allowed = navigation | (interactions if self.allow_interactions else set())
        if kind not in allowed:
            raise ValueError(f"Unsupported roam action: {kind}")

        object_actions = {
            ActionKind.NAVIGATE_TO.value,
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }
        if kind in object_actions:
            object_id = params.get("objectId")
            if not object_id or object_id not in world.objects:
                raise ValueError(f"Invalid objectId for {kind}: {object_id}")
            obj = world.objects[object_id]
            if kind == ActionKind.NAVIGATE_TO.value:
                params.setdefault("interactionDistance", self.interaction_distance)
            elif kind == ActionKind.PICKUP.value and not obj.pickupable:
                raise ValueError(f"Object is not pickupable: {object_id}")
            elif kind in {ActionKind.OPEN.value, ActionKind.CLOSE.value} and not obj.openable:
                raise ValueError(f"Object is not openable: {object_id}")
            elif kind == ActionKind.PUT.value:
                if world.held_object_id is None:
                    raise ValueError("PutObject requested while nothing is held.")
                if not obj.receptacle:
                    raise ValueError(f"PutObject target is not a receptacle: {object_id}")

        return RoamDecision(Action(kind, params, reason), reason)

    def _fallback(self, world: WorldState, recent_target_ids: list[str]) -> RoamDecision:
        candidates = [
            obj for obj in world.objects.values()
            if obj.position is not None and obj.object_id not in recent_target_ids
        ]
        # Prefer farther objects to produce actual roaming rather than tiny local moves.
        candidates.sort(key=lambda obj: (obj.distance is None, -(obj.distance or 0.0)))
        if candidates:
            chosen = candidates[self._fallback_cursor % len(candidates)]
            self._fallback_cursor += 1
            return RoamDecision(
                Action(
                    ActionKind.NAVIGATE_TO.value,
                    {"objectId": chosen.object_id, "interactionDistance": self.interaction_distance},
                    f"Fallback exploration toward {chosen.name}.",
                ),
                f"Fallback exploration toward {chosen.name}.",
                fallback=True,
            )
        return RoamDecision(
            Action(ActionKind.ROTATE_RIGHT.value, {}, "Fallback visual sweep."),
            "Fallback visual sweep.",
            fallback=True,
        )

    def decide(
        self,
        world: WorldState,
        recent: list[dict[str, Any]],
        image_bytes: bytes | None = None,
    ) -> RoamDecision:
        recent_target_ids = [
            str(item.get("objectId")) for item in recent if item.get("objectId")
        ][-12:]
        if self.client is None:
            return self._fallback(world, recent_target_ids)

        payload = {
            "mode": "free_roam",
            "scene": world.scene,
            "agent_pose": {
                "position": world.pose.position,
                "rotation": world.pose.rotation,
                "horizon": world.pose.horizon,
            },
            "held_object_id": world.held_object_id,
            "last_action": world.last_action,
            "last_action_success": world.last_action_success,
            "last_error": world.last_error,
            "interactions_enabled": self.allow_interactions,
            "recent_decisions": recent[-12:],
            "candidate_objects": _candidate_objects(world),
        }
        user = json.dumps(payload, ensure_ascii=False, indent=2)

        try:
            kwargs = {
                "temperature": 0.35,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
            }
            if image_bytes is not None:
                raw = self.client.chat_with_image(
                    ROAM_SYSTEM_PROMPT,
                    user + "\nThe attached image is the agent's current camera view.",
                    image_bytes,
                    **kwargs,
                )
                decision = self._validate(_extract_json(raw), world)
                decision.used_vision = True
                return decision
            raw = self.client.chat(ROAM_SYSTEM_PROMPT, user, **kwargs)
            return self._validate(_extract_json(raw), world)
        except Exception as exc:
            decision = self._fallback(world, recent_target_ids)
            decision.reason = f"LLM fallback after {type(exc).__name__}: {exc}"
            return decision


class FreeRoamAgent:
    def __init__(
        self,
        thor: ThorAdapter,
        planner: RoamPlanner,
        executor: ActionExecutor,
        memory: EpisodicMemory,
        recovery: RecoveryManager,
        max_decisions: int = 100,
        vision_every: int = 1,
        use_vision: bool = False,
        print_decisions: bool = True,
    ):
        self.thor = thor
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.recovery = recovery
        self.max_decisions = max_decisions
        self.vision_every = max(1, vision_every)
        self.use_vision = use_vision
        self.print_decisions = print_decisions

    def run(self) -> RoamResult:
        recent: list[dict[str, Any]] = []

        for index in range(1, self.max_decisions + 1):
            if self.executor.total_steps >= self.executor.max_steps:
                return RoamResult(index - 1, self.executor.total_steps, "Primitive-action budget exhausted.")

            world = self.thor.observe()
            image = None
            if self.use_vision and (index == 1 or (index - 1) % self.vision_every == 0):
                image = frame_to_png(self.thor)

            decision = self.planner.decide(world, recent, image_bytes=image)
            action = decision.action
            target_id = action.params.get("objectId")
            target = world.objects.get(target_id) if target_id else None

            if self.print_decisions:
                target_label = ""
                if target is not None:
                    target_label = f" -> {target.name} [{target.object_type}]"
                vision_label = " vision" if decision.used_vision else ""
                fallback_label = " fallback" if decision.fallback else ""
                print(
                    f"[roam {index:03d} | step {self.executor.total_steps:04d}] "
                    f"{action.kind}{target_label}{vision_label}{fallback_label}: {decision.reason}",
                    flush=True,
                )

            outcome = self.executor.execute(action)
            record = {
                "decision": index,
                "kind": action.kind,
                "objectId": target_id,
                "reason": decision.reason,
                "success": outcome.success,
                "error": outcome.error,
                "primitive_steps": outcome.primitive_steps,
                "used_vision": decision.used_vision,
                "fallback": decision.fallback,
            }
            recent.append(record)
            recent = recent[-16:]
            self.memory.add("roam_decision", record)

            if outcome.success:
                continue

            recovery = self.recovery.classify(action, outcome, self.thor.observe())
            self.memory.add(
                "roam_recovery",
                {
                    "failed_action": action.as_dict(),
                    "reason": recovery.reason,
                    "actions": [candidate.as_dict() for candidate in recovery.actions],
                },
            )
            for recovery_action in recovery.actions[: self.recovery.max_retries_per_action]:
                if self.executor.total_steps >= self.executor.max_steps:
                    break
                self.executor.execute(recovery_action)

        return RoamResult(self.max_decisions, self.executor.total_steps, "Decision budget exhausted.")
