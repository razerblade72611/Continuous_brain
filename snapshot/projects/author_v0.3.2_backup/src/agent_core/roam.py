from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import re
from typing import Any

from .executor import ActionExecutor
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .models import Action, ActionKind, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter


ROAM_SYSTEM_PROMPT = """
You control an embodied agent freely exploring an AI2-THOR 3D environment.
There is no task to finish. You are allowed to wander, turn, look around, revisit
interesting viewpoints later, and interact with objects when interactions are enabled.

Choose what YOU want to do next. Do not behave like a shortest-path task solver.

Return ONLY JSON:
{
  "reason": "brief operational reason",
  "pace": "slow|normal|fast",
  "actions": [
    {"kind": "MoveAhead", "params": {}},
    {"kind": "RotateLeft", "params": {}}
  ]
}

For an object-targeted action, use the candidate alias supplied in the user message:
{
  "kind": "NavigateTo",
  "target": "c07",
  "params": {}
}

Supported movement/look actions:
MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight,
LookUp, LookDown, Pass, NavigateTo.

If interactions are enabled:
PickupObject, PutObject, OpenObject, CloseObject.

Rules:
- Return 1 to MAX_BURST actions.
- Primitive movement bursts are encouraged for natural free roaming.
- You may repeat MoveAhead inside one burst when you want to keep walking.
- Do not invent AI2-THOR object IDs. Use only candidate aliases such as c03.
- Do not NavigateTo a target marked recent=true.
- Avoid selecting the same navigation target repeatedly.
- NavigateTo is optional; you can roam entirely with primitive movement/look actions.
- If you use NavigateTo, it may be followed by a look/rotate action.
- For Pickup/Open/Close/Put, target must be a supplied candidate alias.
- PutObject target must be a receptacle.
- Keep the reason short; do not provide hidden chain-of-thought.
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


def _recent_target_ids(recent: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for item in recent:
        for oid in item.get("objectIds", []) or []:
            if oid:
                result.append(str(oid))
        oid = item.get("objectId")
        if oid:
            result.append(str(oid))
    return result


def _candidate_catalog(
    world: WorldState,
    recent_target_ids: list[str],
    limit: int = 28,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    recent_set = set(recent_target_ids)
    objects = list(world.objects.values())

    # Visible and interaction-relevant objects first, then nearby anchors.
    objects.sort(
        key=lambda obj: (
            not obj.visible,
            obj.object_id in recent_set,
            obj.distance is None,
            obj.distance or 9999.0,
        )
    )

    rows: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for index, obj in enumerate(objects[:limit]):
        alias = f"c{index:02d}"
        aliases[alias] = obj.object_id
        rows.append(
            {
                "id": alias,
                "type": obj.object_type,
                "name": obj.name,
                "visible": obj.visible,
                "distance": None if obj.distance is None else round(float(obj.distance), 2),
                "pickupable": obj.pickupable,
                "openable": obj.openable,
                "open": obj.is_open,
                "receptacle": obj.receptacle,
                "recent": obj.object_id in recent_set,
            }
        )
    return rows, aliases


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
    actions: list[Action]
    reason: str
    pace: str = "normal"
    used_vision: bool = False
    fallback: bool = False

    @property
    def action(self) -> Action:
        # Backward-compatible convenience for older callers/tests.
        return self.actions[0]


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
        max_burst_actions: int = 6,
        target_cooldown: int = 6,
    ):
        self.client = client
        self.interaction_distance = interaction_distance
        self.allow_interactions = allow_interactions
        self.max_burst_actions = max(1, min(12, int(max_burst_actions)))
        self.target_cooldown = max(0, int(target_cooldown))
        self._fallback_cursor = 0

    def _resolve_target(
        self,
        row: dict[str, Any],
        params: dict[str, Any],
        aliases: dict[str, str],
        world: WorldState,
    ) -> str | None:
        target = row.get("target")
        if target is None:
            target = params.pop("target", None)

        # New preferred form: alias like c07.
        if isinstance(target, str) and target in aliases:
            return aliases[target]

        # Backward compatibility: exact objectId is accepted if it really exists.
        object_id = params.get("objectId")
        if isinstance(object_id, str) and object_id in world.objects:
            return object_id

        if target is not None:
            raise ValueError(f"Invalid target alias: {target}")
        if object_id is not None:
            raise ValueError(f"Invalid objectId: {object_id}")
        return None

    def _validate(
        self,
        data: dict[str, Any],
        world: WorldState,
        aliases: dict[str, str],
        recent_target_ids: list[str],
    ) -> RoamDecision:
        reason = str(data.get("reason", "Explore."))[:300]
        pace = str(data.get("pace", "normal")).lower()
        if pace not in {"slow", "normal", "fast"}:
            pace = "normal"

        rows = data.get("actions")
        if rows is None:
            # Backward compatibility with v0.3.0/0.3.1 single-action schema.
            single = data.get("action")
            rows = [] if single is None else [single]
        if not isinstance(rows, list) or not rows:
            raise ValueError("LLM must return at least one roam action.")
        rows = rows[: self.max_burst_actions]

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

        recent_nav = set(recent_target_ids[-self.target_cooldown :]) if self.target_cooldown else set()
        actions: list[Action] = []

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each roam action must be a JSON object.")
            kind = str(row.get("kind", ""))
            params = dict(row.get("params") or {})
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
                object_id = self._resolve_target(row, params, aliases, world)
                if not object_id:
                    raise ValueError(f"{kind} requires a candidate target alias.")
                params["objectId"] = object_id
                obj = world.objects[object_id]

                if kind == ActionKind.NAVIGATE_TO.value:
                    if object_id in recent_nav:
                        raise ValueError(
                            f"NavigateTo target was visited too recently: {obj.name}. Choose another target or primitive movement."
                        )
                    params.setdefault("interactionDistance", self.interaction_distance)
                elif kind == ActionKind.PICKUP.value and not obj.pickupable:
                    raise ValueError(f"Object is not pickupable: {obj.name}")
                elif kind in {ActionKind.OPEN.value, ActionKind.CLOSE.value} and not obj.openable:
                    raise ValueError(f"Object is not openable: {obj.name}")
                elif kind == ActionKind.PUT.value:
                    if world.held_object_id is None:
                        raise ValueError("PutObject requested while nothing is held.")
                    if not obj.receptacle:
                        raise ValueError(f"PutObject target is not a receptacle: {obj.name}")

            actions.append(Action(kind, params, reason))

        return RoamDecision(actions=actions, reason=reason, pace=pace)

    def _fallback(self, world: WorldState, recent_target_ids: list[str]) -> RoamDecision:
        # Short local wander instead of a long automatic path. This makes a model
        # outage visible without letting fallback take over the whole environment.
        patterns = [
            [Action(ActionKind.ROTATE_RIGHT.value, {}, "Fallback survey."),
             Action(ActionKind.MOVE_AHEAD.value, {}, "Fallback wander."),
             Action(ActionKind.MOVE_AHEAD.value, {}, "Fallback wander.")],
            [Action(ActionKind.ROTATE_LEFT.value, {}, "Fallback survey."),
             Action(ActionKind.MOVE_AHEAD.value, {}, "Fallback wander.")],
            [Action(ActionKind.LOOK_UP.value, {}, "Fallback look."),
             Action(ActionKind.ROTATE_RIGHT.value, {}, "Fallback survey.")],
        ]
        chosen = patterns[self._fallback_cursor % len(patterns)]
        self._fallback_cursor += 1
        return RoamDecision(
            actions=chosen,
            reason="Short fallback wander.",
            pace="normal",
            fallback=True,
        )

    def _call(
        self,
        user: str,
        image_bytes: bytes | None,
        correction: str | None = None,
    ) -> tuple[str, bool]:
        if self.client is None:
            raise RuntimeError("No LLM client configured.")

        if correction:
            user = (
                user
                + "\n\nCORRECTION: Your previous response was rejected by the controller: "
                + correction
                + "\nReturn a different valid choice. Prefer primitive movement if unsure."
            )

        kwargs = {
            "temperature": 0.55,
            "max_tokens": 420,
            "response_format": {"type": "json_object"},
        }
        if image_bytes is not None:
            raw = self.client.chat_with_image(
                ROAM_SYSTEM_PROMPT.replace("MAX_BURST", str(self.max_burst_actions)),
                user + "\nThe attached image is the agent's current camera view.",
                image_bytes,
                **kwargs,
            )
            return raw, True

        raw = self.client.chat(
            ROAM_SYSTEM_PROMPT.replace("MAX_BURST", str(self.max_burst_actions)),
            user,
            **kwargs,
        )
        return raw, False

    def decide(
        self,
        world: WorldState,
        recent: list[dict[str, Any]],
        image_bytes: bytes | None = None,
    ) -> RoamDecision:
        recent_target_ids = _recent_target_ids(recent)
        if self.client is None:
            return self._fallback(world, recent_target_ids)

        candidates, aliases = _candidate_catalog(world, recent_target_ids)
        recent_compact = []
        for item in recent[-8:]:
            recent_compact.append(
                {
                    "kinds": item.get("kinds") or [item.get("kind")] if item.get("kind") else [],
                    "targets": item.get("target_names") or [],
                    "success": item.get("success"),
                    "reason": str(item.get("reason", ""))[:90],
                }
            )

        payload = {
            "mode": "free_roam",
            "scene": world.scene,
            "pose": {
                "x": round(float(world.pose.position.get("x", 0.0)), 2),
                "z": round(float(world.pose.position.get("z", 0.0)), 2),
                "yaw": round(float(world.pose.rotation.get("y", 0.0)), 1),
                "horizon": round(float(world.pose.horizon), 1),
            },
            "held": world.held_object_id is not None,
            "last_action": world.last_action,
            "last_success": world.last_action_success,
            "interactions_enabled": self.allow_interactions,
            "recent": recent_compact,
            "candidates": candidates,
        }
        # Compact JSON dramatically reduces local-model prompt tokens.
        user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        first_error: Exception | None = None
        try:
            raw, used_vision = self._call(user, image_bytes)
            decision = self._validate(_extract_json(raw), world, aliases, recent_target_ids)
            decision.used_vision = used_vision
            return decision
        except (ValueError, json.JSONDecodeError) as exc:
            first_error = exc
            # One corrective re-prompt for hallucinated aliases, invalid actions,
            # repeated targets, or malformed JSON.
            try:
                raw, used_vision = self._call(user, image_bytes, correction=str(exc))
                decision = self._validate(_extract_json(raw), world, aliases, recent_target_ids)
                decision.used_vision = used_vision
                return decision
            except Exception as retry_exc:
                first_error = retry_exc
        except Exception as exc:
            first_error = exc

        decision = self._fallback(world, recent_target_ids)
        decision.reason = f"LLM fallback after {type(first_error).__name__}: {first_error}"
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
        llm_controls_pace: bool = True,
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
        self.llm_controls_pace = llm_controls_pace

    def _pace_delay(self, base_delay: float, pace: str) -> float:
        if not self.llm_controls_pace:
            return base_delay
        factor = {"slow": 2.0, "normal": 1.0, "fast": 0.5}.get(pace, 1.0)
        return max(0.0, base_delay * factor)

    def run(self) -> RoamResult:
        recent: list[dict[str, Any]] = []
        base_delay = float(self.thor.config.step_delay_seconds)

        for index in range(1, self.max_decisions + 1):
            if self.executor.total_steps >= self.executor.max_steps:
                return RoamResult(index - 1, self.executor.total_steps, "Primitive-action budget exhausted.")

            world = self.thor.observe()
            image = None
            if self.use_vision and (index == 1 or (index - 1) % self.vision_every == 0):
                image = frame_to_png(self.thor)

            decision = self.planner.decide(world, recent, image_bytes=image)
            self.thor.config.step_delay_seconds = self._pace_delay(base_delay, decision.pace)

            target_ids: list[str] = []
            target_names: list[str] = []
            for action in decision.actions:
                oid = action.params.get("objectId")
                if oid:
                    target_ids.append(str(oid))
                    obj = world.objects.get(str(oid))
                    if obj is not None:
                        target_names.append(obj.name)

            if self.print_decisions:
                action_summary = " -> ".join(action.kind for action in decision.actions)
                target_label = ""
                if target_names:
                    target_label = " targets=" + ",".join(target_names)
                vision_label = " vision" if decision.used_vision else ""
                fallback_label = " fallback" if decision.fallback else ""
                print(
                    f"[roam {index:03d} | step {self.executor.total_steps:04d}] "
                    f"pace={decision.pace} {action_summary}{target_label}"
                    f"{vision_label}{fallback_label}: {decision.reason}",
                    flush=True,
                )

            all_success = True
            errors: list[str] = []
            primitive_steps = 0

            for action in decision.actions:
                if self.executor.total_steps >= self.executor.max_steps:
                    break

                outcome = self.executor.execute(action)
                primitive_steps += outcome.primitive_steps
                if outcome.success:
                    continue

                all_success = False
                if outcome.error:
                    errors.append(outcome.error)

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
                # After a failed member of a burst, stop the burst and let the LLM
                # observe the new state rather than blindly executing stale actions.
                break

            record = {
                "decision": index,
                "kinds": [action.kind for action in decision.actions],
                "objectIds": target_ids,
                "target_names": target_names,
                "reason": decision.reason,
                "pace": decision.pace,
                "success": all_success,
                "error": "; ".join(errors) if errors else None,
                "primitive_steps": primitive_steps,
                "used_vision": decision.used_vision,
                "fallback": decision.fallback,
            }
            recent.append(record)
            recent = recent[-16:]
            self.memory.add("roam_decision", record)

        self.thor.config.step_delay_seconds = base_delay
        return RoamResult(self.max_decisions, self.executor.total_steps, "Decision budget exhausted.")
