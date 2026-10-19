from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .executor import ActionExecutor
from .grounding import GroundedAction, GroundingError, SemanticGrounder
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .models import Action, ActionKind, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter
from .vision import capture_frame_png, embodiment_snapshot


EMBODIED_SYSTEM_PROMPT = """
You are embodied in a simulated 3D environment. You can see through a camera
when an image is attached and you can control your body through the actions below.

When no user task is active, there is no required goal. You are free to decide
what to do. You may move, stop, look around, inspect something you perceive,
interact with something you perceive, revisit a place, or remain where you are.
There is no preferred object, destination, exploration strategy, or required
behavior.

When a user task is active, treat the exact user task as your current objective.
Do not invent extra objectives. You may report task_complete=true when you judge
an unstructured user task complete. Structured manipulation tasks are also
verified independently by the simulator.

The controller does NOT give you an object catalog. For object-directed actions,
write a short semantic reference based on your own perception, for example
"the object near the left side of my view". A hidden motor-grounding layer will translate your
reference into a simulator ID if a matching object is actually visible. Never
invent simulator IDs.

Available actions:
- MoveAhead, MoveBack, MoveLeft, MoveRight
- RotateLeft, RotateRight, LookUp, LookDown, Pass
- Approach: move near a perceived object; requires target text
- PickupObject: requires target text
- PutObject: place the currently held object into/on a perceived receptacle; requires target text
- OpenObject, CloseObject: require target text

Return ONLY one JSON object:
{
  "say": "optional brief outward status, not hidden reasoning",
  "pace": "slow|normal|fast",
  "task_complete": false,
  "actions": [
    {"kind": "MoveAhead", "target": null, "params": {}}
  ]
}

Return between 1 and MAX_BURST actions. Use only the capabilities listed above.
Do not expose private chain-of-thought; keep `say` brief and operational.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Embodied response was not JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


@dataclass(frozen=True)
class SemanticAction:
    kind: str
    target: str | None
    params: dict[str, Any]


@dataclass
class EmbodiedDecision:
    actions: list[Action]
    semantic_actions: list[SemanticAction]
    say: str = ""
    pace: str = "normal"
    task_complete: bool = False
    used_vision: bool = False
    fallback: bool = False


@dataclass
class RoamResult:
    decisions: int
    primitive_steps: int
    reason: str


class EmbodiedPlanner:
    """Neutral autonomy/task planner. Simulator object metadata never enters its prompt."""

    def __init__(
        self,
        client: SyncLLMClient | None,
        grounder: SemanticGrounder,
        allow_interactions: bool = True,
        max_burst_actions: int = 6,
    ):
        self.client = client
        self.grounder = grounder
        self.allow_interactions = allow_interactions
        self.max_burst_actions = max(1, min(12, int(max_burst_actions)))

    @property
    def system_prompt(self) -> str:
        prompt = EMBODIED_SYSTEM_PROMPT.replace("MAX_BURST", str(self.max_burst_actions))
        if not self.allow_interactions:
            prompt += "\nObject interaction actions are disabled for this run; use only movement/look/Pass/Approach."
        return prompt

    @staticmethod
    def _recent_for_llm(recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Only information the LLM itself produced plus generic execution success is returned.
        # Grounded object IDs/names and the simulator catalog are intentionally absent.
        out: list[dict[str, Any]] = []
        for item in recent[-8:]:
            out.append({
                "say": str(item.get("say", ""))[:120],
                "semantic_actions": item.get("semantic_actions", []),
                "success": item.get("success"),
                "pace": item.get("pace", "normal"),
            })
        return out

    def _payload(self, world: WorldState, mode: str, task: str | None, recent: list[dict[str, Any]]) -> str:
        payload = {
            "mode": mode,
            "user_task": task if mode == "task" else None,
            "embodiment": embodiment_snapshot(world),
            "recent_self_context": self._recent_for_llm(recent),
            "camera_attached": False,  # changed to true in the image call's text below
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _call(self, user_payload: str, image_bytes: bytes | None, correction: str | None = None) -> tuple[str, bool]:
        if self.client is None:
            raise RuntimeError("No LLM client configured.")
        user = user_payload
        if image_bytes is not None:
            # This flag is text-only and contains no semantic scene information.
            user = user.replace('"camera_attached":false', '"camera_attached":true')
        if correction:
            user += (
                "\nCORRECTION: The previous command could not be executed: "
                + correction
                + " Choose another valid action using only your own perception."
            )
        kwargs = {
            "temperature": 0.6,
            "max_tokens": 420,
            "response_format": {"type": "json_object"},
        }
        if image_bytes is not None:
            return self.client.chat_with_image(self.system_prompt, user, image_bytes, **kwargs), True
        return self.client.chat(self.system_prompt, user, **kwargs), False

    def _validate_and_ground(self, data: dict[str, Any], world: WorldState) -> EmbodiedDecision:
        say = str(data.get("say", ""))[:300]
        pace = str(data.get("pace", "normal")).lower()
        if pace not in {"slow", "normal", "fast"}:
            pace = "normal"
        task_complete = bool(data.get("task_complete", False))
        rows = data.get("actions")
        if not isinstance(rows, list) or not rows:
            raise ValueError("At least one embodied action is required.")
        rows = rows[: self.max_burst_actions]

        movement = {
            ActionKind.MOVE_AHEAD.value,
            ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value,
            ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value,
            ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value,
            ActionKind.LOOK_DOWN.value,
            ActionKind.WAIT.value,
            "Approach",
        }
        interactions = {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }
        allowed = movement | (interactions if self.allow_interactions else set())

        grounded: list[Action] = []
        semantic: list[SemanticAction] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each action must be an object.")
            kind = str(row.get("kind", ""))
            if kind not in allowed:
                raise ValueError(f"Unsupported embodied action: {kind}")
            target = row.get("target")
            if target is not None:
                target = str(target).strip() or None
            params = dict(row.get("params") or {})
            # Never allow the model to smuggle raw simulator IDs through params.
            params.pop("objectId", None)
            sem = SemanticAction(kind=kind, target=target, params=params)
            result = self.grounder.ground(kind, target, params, world, reason=say)
            semantic.append(sem)
            grounded.append(result.action)

        return EmbodiedDecision(
            actions=grounded,
            semantic_actions=semantic,
            say=say,
            pace=pace,
            task_complete=task_complete,
        )

    def decide(
        self,
        world: WorldState,
        mode: str,
        task: str | None,
        recent: list[dict[str, Any]],
        image_bytes: bytes | None,
    ) -> EmbodiedDecision:
        if self.client is None:
            return EmbodiedDecision(
                actions=[Action(ActionKind.WAIT.value, {}, "LLM unavailable; remain idle.")],
                semantic_actions=[SemanticAction(ActionKind.WAIT.value, None, {})],
                say="LLM unavailable; idling.",
                fallback=True,
            )

        payload = self._payload(world, mode, task, recent)
        first_error: Exception | None = None
        try:
            raw, used_vision = self._call(payload, image_bytes)
            decision = self._validate_and_ground(_extract_json(raw), world)
            decision.used_vision = used_vision
            return decision
        except (ValueError, GroundingError, json.JSONDecodeError) as exc:
            first_error = exc
            try:
                # The correction contains only the model's own failed semantic phrase and a
                # generic failure; it does not reveal hidden simulator alternatives.
                raw, used_vision = self._call(payload, image_bytes, correction=str(exc))
                decision = self._validate_and_ground(_extract_json(raw), world)
                decision.used_vision = used_vision
                return decision
            except Exception as retry_exc:
                first_error = retry_exc
        except Exception as exc:
            first_error = exc

        return EmbodiedDecision(
            actions=[Action(ActionKind.WAIT.value, {}, "LLM command failed; remain idle.")],
            semantic_actions=[SemanticAction(ActionKind.WAIT.value, None, {})],
            say=f"LLM command failed: {type(first_error).__name__}: {first_error}",
            pace="normal",
            fallback=True,
        )


class EmbodiedLoop:
    """One persistent body/controller used by both free-roam and interactive task sessions."""

    def __init__(
        self,
        thor: ThorAdapter,
        planner: EmbodiedPlanner,
        executor: ActionExecutor,
        memory: EpisodicMemory,
        recovery: RecoveryManager,
        use_vision: bool = True,
        vision_every: int = 1,
        print_decisions: bool = True,
        llm_controls_pace: bool = True,
    ):
        self.thor = thor
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.recovery = recovery
        self.use_vision = use_vision
        self.vision_every = max(1, int(vision_every))
        self.print_decisions = print_decisions
        self.llm_controls_pace = llm_controls_pace
        self.base_delay = float(thor.config.step_delay_seconds)
        self.recent: list[dict[str, Any]] = []
        self.decision_index = 0

    def _pace_delay(self, pace: str) -> float:
        if not self.llm_controls_pace:
            return self.base_delay
        factor = {"slow": 2.0, "normal": 1.0, "fast": 0.5}.get(pace, 1.0)
        return max(0.0, self.base_delay * factor)

    def step(self, mode: str = "autonomous", task: str | None = None) -> tuple[EmbodiedDecision, dict[str, Any]]:
        self.decision_index += 1
        world = self.thor.observe()
        image = None
        if self.use_vision and (self.decision_index == 1 or (self.decision_index - 1) % self.vision_every == 0):
            image = capture_frame_png(self.thor)

        decision = self.planner.decide(world, mode, task, self.recent, image)
        self.thor.config.step_delay_seconds = self._pace_delay(decision.pace)

        if self.print_decisions:
            semantic_summary = " -> ".join(
                a.kind + (f'("{a.target}")' if a.target else "") for a in decision.semantic_actions
            )
            labels = []
            if decision.used_vision:
                labels.append("vision")
            if decision.fallback:
                labels.append("fallback")
            suffix = (" " + " ".join(labels)) if labels else ""
            task_label = "task" if mode == "task" else "auto"
            print(
                f"[{task_label} {self.decision_index:03d} | step {self.executor.total_steps:04d}] "
                f"pace={decision.pace} {semantic_summary}{suffix}: {decision.say}",
                flush=True,
            )

        successes: list[bool] = []
        generic_errors: list[str] = []
        primitive_steps = 0
        for action in decision.actions:
            if self.executor.total_steps >= self.executor.max_steps:
                break
            outcome = self.executor.execute(action)
            primitive_steps += outcome.primitive_steps
            successes.append(outcome.success)
            if outcome.success:
                continue

            # Do not return hidden target metadata to the LLM. Recovery may use it internally.
            generic_errors.append("action_failed")
            recovery = self.recovery.classify(action, outcome, self.thor.observe())
            self.memory.add(
                "embodied_recovery_internal",
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
            break

        record = {
            "mode": mode,
            "task": task,
            "say": decision.say,
            "pace": decision.pace,
            "semantic_actions": [
                {"kind": a.kind, "target": a.target, "params": a.params} for a in decision.semantic_actions
            ],
            "success": bool(successes) and all(successes),
            "generic_errors": generic_errors,
            "primitive_steps": primitive_steps,
            "used_vision": decision.used_vision,
            "fallback": decision.fallback,
            "task_complete_claim": decision.task_complete,
        }
        self.recent.append(record)
        self.recent = self.recent[-16:]
        self.memory.add("embodied_decision", record)
        return decision, record


class FreeRoamAgent:
    def __init__(self, loop: EmbodiedLoop, max_decisions: int = 100):
        self.loop = loop
        self.max_decisions = max_decisions

    def run(self) -> RoamResult:
        for _ in range(self.max_decisions):
            if self.loop.executor.total_steps >= self.loop.executor.max_steps:
                return RoamResult(self.loop.decision_index, self.loop.executor.total_steps, "Primitive-action budget exhausted.")
            self.loop.step(mode="autonomous", task=None)
        return RoamResult(self.loop.decision_index, self.loop.executor.total_steps, "Decision budget exhausted.")


# Compatibility alias for code that imported the old planner name.
RoamPlanner = EmbodiedPlanner
