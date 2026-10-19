from __future__ import annotations

import json
import re
from typing import Optional

from .goals import canonical_name, parse_goal
from .llm_client import SyncLLMClient
from .models import Action, ActionKind, ObjectState, Plan, WorldState


SYSTEM_PROMPT = """
You are the planning module of an embodied AI agent in AI2-THOR.

You receive a natural-language goal, compact structured world state, and recent memory.
Return ONLY a JSON object:
{
  "rationale": "short explanation",
  "actions": [
    {"kind": "NavigateTo", "params": {"objectId": "..."}, "reason": "..."},
    {"kind": "PickupObject", "params": {"objectId": "..."}, "reason": "..."}
  ]
}

Rules:
- Supported action kinds: NavigateTo, MoveAhead, MoveBack, MoveLeft, MoveRight,
  RotateLeft, RotateRight, LookUp, LookDown, PickupObject, PutObject,
  OpenObject, CloseObject, Pass.
- Prefer structured metadata over guesses.
- Never invent object IDs.
- Use NavigateTo before interaction when the target is not visible or is farther than ~1.25 m.
- For composite goals such as "Pick up the Apple and put it on the CounterTop",
  first acquire the source, then navigate to the receptacle, then PutObject.
- PutObject.objectId is the target receptacle's objectId.
- Keep plans short and state-driven; normally return one immediate subgoal at a time.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Planner response was not JSON: {text[:500]}")
    return json.loads(text[start:end + 1])


class RulePlanner:
    def __init__(self, world: WorldState, interaction_distance: float = 1.25):
        self.world = world
        self.interaction_distance = interaction_distance

    def _nearest(self, noun: str | None) -> ObjectState | None:
        if not noun:
            return None
        candidates = self.world.find(noun)
        if not candidates:
            return None
        candidates.sort(key=lambda obj: (obj.distance is None, obj.distance or 9999))
        return candidates[0]

    def _ready_to_interact(self, obj: ObjectState) -> bool:
        return bool(obj.visible and obj.distance is not None and obj.distance <= self.interaction_distance)

    def _navigate(self, obj: ObjectState, reason: str) -> Plan:
        return Plan(
            goal="",
            actions=[
                Action(
                    ActionKind.NAVIGATE_TO.value,
                    {"objectId": obj.object_id, "interactionDistance": self.interaction_distance},
                    reason,
                )
            ],
        )

    def plan(self, goal: str) -> Plan:
        spec = parse_goal(goal)

        # Placement has priority over the pickup clause in a composite goal.
        if spec.has_put:
            source = self._nearest(spec.source_object)
            target = self._nearest(spec.put_target)
            held = self.world.held_object()

            if source and target and any(parent in {target.object_id} for parent in source.parent_receptacles):
                return Plan(goal=goal, actions=[], rationale="Placement already verified in world metadata.")

            source_is_held = bool(
                held
                and (
                    (source is not None and held.object_id == source.object_id)
                    or canonical_name(held.object_type) == canonical_name(spec.source_object)
                    or canonical_name(held.name) == canonical_name(spec.source_object)
                )
            )

            if not source_is_held:
                if source is None:
                    return self._explore(goal, f"Source object '{spec.source_object}' was not found.")
                if not self._ready_to_interact(source):
                    plan = self._navigate(source, f"Navigate to {source.name} before pickup.")
                    plan.goal = goal
                    return plan
                return Plan(
                    goal=goal,
                    actions=[Action(ActionKind.PICKUP.value, {"objectId": source.object_id}, f"Pick up {source.name}.")],
                    rationale="Acquire the source object before placement.",
                )

            if target is None:
                return self._explore(goal, f"Target receptacle '{spec.put_target}' was not found.")
            if not self._ready_to_interact(target):
                plan = self._navigate(target, f"Navigate to {target.name} before placement.")
                plan.goal = goal
                return plan
            return Plan(
                goal=goal,
                actions=[Action(ActionKind.PUT.value, {"objectId": target.object_id}, f"Place held object on/in {target.name}.")],
                rationale="Source is held; complete the placement subgoal.",
            )

        if spec.pickup_object:
            obj = self._nearest(spec.pickup_object)
            if obj is None:
                return self._explore(goal, f"Pickup object '{spec.pickup_object}' was not found.")
            if obj.is_picked_up:
                return Plan(goal=goal, actions=[], rationale="Pickup already verified.")
            if not self._ready_to_interact(obj):
                plan = self._navigate(obj, f"Navigate to {obj.name} before pickup.")
                plan.goal = goal
                return plan
            return Plan(
                goal=goal,
                actions=[Action(ActionKind.PICKUP.value, {"objectId": obj.object_id}, f"Pick up {obj.name}.")],
            )

        if spec.open_object:
            obj = self._nearest(spec.open_object)
            if obj is None:
                return self._explore(goal, f"Open target '{spec.open_object}' was not found.")
            if obj.is_open is True:
                return Plan(goal=goal, actions=[], rationale="Object is already open.")
            if not self._ready_to_interact(obj):
                plan = self._navigate(obj, f"Navigate to {obj.name} before opening it.")
                plan.goal = goal
                return plan
            return Plan(goal=goal, actions=[Action(ActionKind.OPEN.value, {"objectId": obj.object_id}, f"Open {obj.name}.")])

        if spec.close_object:
            obj = self._nearest(spec.close_object)
            if obj is None:
                return self._explore(goal, f"Close target '{spec.close_object}' was not found.")
            if obj.is_open is False:
                return Plan(goal=goal, actions=[], rationale="Object is already closed.")
            if not self._ready_to_interact(obj):
                plan = self._navigate(obj, f"Navigate to {obj.name} before closing it.")
                plan.goal = goal
                return plan
            return Plan(goal=goal, actions=[Action(ActionKind.CLOSE.value, {"objectId": obj.object_id}, f"Close {obj.name}.")])

        return self._explore(goal, "Goal is outside the deterministic task grammar.")

    def _explore(self, goal: str, rationale: str) -> Plan:
        return Plan(
            goal=goal,
            actions=[Action(ActionKind.ROTATE_RIGHT.value, {}, "Sweep the scene for relevant objects.")],
            rationale=rationale,
        )


class LLMPlanner:
    def __init__(self, client: SyncLLMClient):
        self.client = client

    def plan(self, goal: str, world: WorldState, recent_memory: list[dict]) -> Plan:
        user = json.dumps(
            {
                "goal": goal,
                "world_state": world.compact(),
                "recent_memory": recent_memory,
            },
            ensure_ascii=False,
            indent=2,
        )
        raw = self.client.chat(SYSTEM_PROMPT, user, temperature=0.1)
        data = _extract_json(raw)
        actions: list[Action] = []
        allowed = {kind.value for kind in ActionKind}
        object_actions = {
            ActionKind.NAVIGATE_TO.value,
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }
        for row in data.get("actions", []):
            kind = str(row["kind"])
            if kind not in allowed:
                raise ValueError(f"LLM planner returned unsupported action kind: {kind}")
            params = dict(row.get("params") or {})
            if kind in object_actions:
                object_id = params.get("objectId")
                if not object_id or object_id not in world.objects:
                    raise ValueError(f"LLM planner returned invalid objectId for {kind}: {object_id}")
            actions.append(Action(kind=kind, params=params, reason=str(row.get("reason", ""))))
        return Plan(goal=goal, actions=actions, rationale=str(data.get("rationale", "")))


class HybridPlanner:
    def __init__(self, llm: Optional[LLMPlanner] = None, interaction_distance: float = 1.25):
        self.llm = llm
        self.interaction_distance = interaction_distance

    def plan(self, goal: str, world: WorldState, recent_memory: list[dict]) -> Plan:
        if self.llm is not None:
            try:
                return self.llm.plan(goal, world, recent_memory)
            except Exception as exc:
                recent_memory.append({"event": "llm_planner_fallback", "error": f"{type(exc).__name__}: {exc}"})
        return RulePlanner(world, interaction_distance=self.interaction_distance).plan(goal)
