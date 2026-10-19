from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .executor import ActionExecutor
from .grounding import GroundingError, SemanticGrounder
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .models import Action, ActionKind, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter
from .vision import capture_frame_png, embodiment_snapshot, pose_delta


EMBODIED_SYSTEM_PROMPT = """
You are embodied in a simulated 3D environment. You can see through a camera
when an image is attached and can control the body through the interface below.

When no user task is active, there is no required goal or preferred behavior.
You may move, look, interact with something you perceive, remain still, or do
nothing this cycle. When a user task is active, the exact user task is your
current objective; do not invent additional objectives.

The controller does NOT give you an object catalog. Object names/IDs, hidden
objects, and simulator detections are not listed for you. If you independently
notice or refer to something, you may put that phrase in `focus` or use it as the
target of a semantic action. A hidden motor-grounding layer only translates your
own reference when it can match your current perception or your own previously
grounded focus.

`semantic_focus`, when present in body state, exists only because YOU previously
referred to that thing. Its fields report neutral physical facts and currently
possible body actions for that already-selected focus. They are capabilities,
not recommendations. `last_body_feedback.recent_progress`, when present, is only
a factual summary of recent pose change and repeated action signatures.

Primitive body actions (NO target):
- MoveAhead, MoveBack, MoveLeft, MoveRight
- RotateLeft, RotateRight, LookUp, LookDown
- Observe or Wait (remain still for this cycle)

Semantic actions (target text REQUIRED):
- Approach: move near something you independently identify
- Face: turn the body toward something you independently identify
- PickupObject
- PutObject
- OpenObject
- CloseObject

Important interface rules:
- Primitive movement/look/Observe/Wait actions do not take a target. If you want
  movement relative to something, use Approach or Face.
- `focus` is optional attention only; setting it does not cause movement.
- `actions` may be empty. An empty list means Observe/do nothing this cycle.
- You may report task_complete=true for an unstructured user task you genuinely
  judge complete. Structured manipulation tasks are independently verified.
- Do not expose private chain-of-thought. Keep `say` brief and operational.

Return ONLY one JSON object:
{
  "say": "optional brief outward status",
  "focus": null,
  "pace": "slow|normal|fast",
  "task_complete": false,
  "actions": [
    {"kind": "MoveAhead", "target": null, "params": {}}
  ]
}

Return at most MAX_BURST actions.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse model output conservatively, with neutral mechanical repair for common local-LLM drift."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # First accept any valid JSON object embedded in surrounding text. raw_decode
    # avoids the old first-'{'/last-'}' failure when a model emits extra material.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    # A small syntax repair for trailing commas before ]/}. This only repairs
    # serialization, never changes the requested behavior.
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    # If the model emits an obvious one-line action instead of JSON, translate
    # that stated intent into the same schema. No action is invented.
    compact = " ".join(raw.split())
    action_match = re.fullmatch(
        r"(?i)(MoveAhead|MoveBack|MoveLeft|MoveRight|RotateLeft|RotateRight|LookUp|LookDown|"
        r"Observe|Wait|Pass|Approach|Face|PickupObject|PutObject|OpenObject|CloseObject)"
        r"(?:\s+(?:the\s+)?(.+?))?[.!]?",
        compact,
    )
    if action_match:
        kind = action_match.group(1)
        target = (action_match.group(2) or "").strip() or None
        primitive = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Observe", "Wait", "Pass"}
        return {
            "say": compact[:300],
            "focus": target,
            "pace": "normal",
            "task_complete": False,
            "actions": [{"kind": kind, "target": None if kind in primitive else target, "params": {}}],
        }

    # Plain descriptive/no-op text is a valid outward report with an empty
    # physical choice. This prevents phrases such as "No object ... recommended"
    # from becoming an artificial LLM failure.
    if compact and not any(token in compact for token in ("{", "[", "}")):
        return {
            "say": compact[:300],
            "focus": None,
            "pace": "normal",
            "task_complete": False,
            "actions": [],
        }

    raise ValueError(f"Embodied response was not recoverable JSON: {compact[:300]}")


@dataclass(frozen=True)
class SemanticAction:
    kind: str
    target: str | None
    params: dict[str, Any]


@dataclass
class EmbodiedDecision:
    actions: list[Action]
    semantic_actions: list[SemanticAction]
    physical_semantic_actions: list[SemanticAction] = None
    spoken: list[str] = None
    say: str = ""
    attention: str | None = None
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
    """Neutral autonomy/task planner. Simulator object catalogs never enter its prompt."""

    def __init__(
        self,
        client: SyncLLMClient | None,
        grounder: SemanticGrounder,
        allow_interactions: bool = True,
        max_burst_actions: int = 6,
        social_enabled: bool = False,
        agent_label: str | None = None,
    ):
        self.client = client
        self.grounder = grounder
        self.allow_interactions = allow_interactions
        self.max_burst_actions = max(1, min(12, int(max_burst_actions)))
        self.social_enabled = bool(social_enabled)
        self.agent_label = agent_label

    @property
    def system_prompt(self) -> str:
        prompt = EMBODIED_SYSTEM_PROMPT.replace("MAX_BURST", str(self.max_burst_actions))
        if not self.allow_interactions:
            prompt += "\nDirect manipulation is disabled for this run; semantic Approach/Face remain available."
        if self.social_enabled:
            prompt += """

SOCIAL CAPABILITY:
Other embodied agents may be present in the same physical scene. AI2-THOR renders
them as simple colored capsule-like bodies when they are visible in RGB. The
`peer_camera_cone_sensor` may report that another embodied body's center lies
inside your camera cone, with neutral distance/bearing geometry. That sensor does not prove the body is unoccluded and does not identify ordinary scene objects; use
the attached RGB image for appearance. You can optionally use Speak to utter
words aloud. Speech is delivered only to embodied agents within hearing range.
You are not required to speak, respond, cooperate, approach anyone, or form any
particular relationship. Heard speech, when present, is sensory input only. You
never receive another agent's private camera, memory, task, semantic focus, or
hidden reasoning.

Social action (no target):
- Speak, with params.text containing the words you choose to say aloud.
"""
            if self.agent_label:
                prompt += f"\nYour session label is {self.agent_label}. The label is routing identity, not a role or personality.\n"
        return prompt

    @staticmethod
    def _recent_for_llm(recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in recent[-6:]:
            out.append({
                "say": str(item.get("say", ""))[:120],
                "attention": item.get("attention"),
                "semantic_actions": item.get("semantic_actions", []),
                "success": item.get("success"),
                "pace": item.get("pace", "normal"),
            })
        return out

    def _payload(
        self,
        world: WorldState,
        mode: str,
        task: str | None,
        recent: list[dict[str, Any]],
        body_feedback: dict[str, Any] | None,
        social_context: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "mode": mode,
            "user_task": task if mode == "task" else None,
            "embodiment": embodiment_snapshot(world),
            "last_body_feedback": body_feedback,
            "semantic_focus": self.grounder.focus_snapshot(world),
            "recent_self_context": self._recent_for_llm(recent),
            "camera_attached": False,
            "social": social_context if self.social_enabled else None,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _call(self, user_payload: str, image_bytes: bytes | None, correction: str | None = None) -> tuple[str, bool]:
        if self.client is None:
            raise RuntimeError("No LLM client configured.")
        user = user_payload
        if image_bytes is not None:
            user = user.replace('"camera_attached":false', '"camera_attached":true')
        if correction:
            user += (
                "\nINTERFACE RESULT: The previous command could not be executed for this mechanical reason: "
                + correction
                + " Choose any valid response you want. No object or behavior is being recommended."
            )
        kwargs = {
            "temperature": 0.6,
            "max_tokens": 420,
            "response_format": {"type": "json_object"},
        }
        if image_bytes is not None:
            return self.client.chat_with_image(self.system_prompt, user, image_bytes, **kwargs), True
        return self.client.chat(self.system_prompt, user, **kwargs), False

    @staticmethod
    def _normalize_kind(kind: str) -> str:
        aliases = {
            "Observe": "Observe",
            "Wait": "Observe",
            "Pass": "Observe",
        }
        return aliases.get(kind, kind)

    def _validate_and_ground(self, data: dict[str, Any], world: WorldState) -> EmbodiedDecision:
        say = str(data.get("say", ""))[:300]
        attention = data.get("focus")
        if attention is not None:
            attention = str(attention).strip() or None
        pace = str(data.get("pace", "normal")).lower()
        if pace not in {"slow", "normal", "fast"}:
            pace = "normal"
        task_complete = bool(data.get("task_complete", False))

        rows = data.get("actions", [])
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise ValueError("`actions` must be a JSON list.")
        rows = rows[: self.max_burst_actions]

        # Empty physical choice is valid autonomy: normalize to Observe/Pass.
        if not rows:
            rows = [{"kind": "Observe", "target": None, "params": {}}]

        primitive = {
            ActionKind.MOVE_AHEAD.value,
            ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value,
            ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value,
            ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value,
            ActionKind.LOOK_DOWN.value,
            "Observe",
        }
        semantic_motion = {"Approach", "Face"}
        interactions = {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }
        allowed = primitive | semantic_motion | (interactions if self.allow_interactions else set())
        if self.social_enabled:
            allowed.add("Speak")

        # Preserve model-selected attention even when it was expressed as a legacy
        # primitive target such as MoveAhead("microwave"). It becomes focus only;
        # it does not secretly steer the primitive action.
        legacy_attention: str | None = None
        grounded: list[Action] = []
        semantic: list[SemanticAction] = []
        physical_semantic: list[SemanticAction] = []
        spoken: list[str] = []

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each action must be a JSON object.")
            kind = self._normalize_kind(str(row.get("kind", "")))
            if kind not in allowed:
                raise ValueError(f"Unsupported embodied action: {kind}")
            target = row.get("target")
            if target is not None:
                target = str(target).strip() or None
            params = dict(row.get("params") or {})
            params.pop("objectId", None)

            if kind == "Speak":
                text = str(params.get("text") or row.get("text") or "").strip()
                text = " ".join(text.split())[:500]
                if not text:
                    raise ValueError("Speak requires non-empty params.text.")
                semantic.append(SemanticAction("Speak", None, {"text": text}))
                spoken.append(text)
                continue

            if kind in primitive:
                if target and not legacy_attention:
                    legacy_attention = target
                semantic_kind = "Observe" if kind == "Observe" else kind
                grounded_kind = ActionKind.WAIT.value if kind == "Observe" else kind
                result = self.grounder.ground(grounded_kind, None, params, world, reason=say)
                sem = SemanticAction(semantic_kind, None, params)
                semantic.append(sem)
                physical_semantic.append(sem)
                grounded.append(result.action)
                continue

            if not target:
                raise ValueError(f"{kind} requires target text based on your own perception or existing semantic focus.")
            result = self.grounder.ground(kind, target, params, world, reason=say)
            sem = SemanticAction(kind, target, params)
            semantic.append(sem)
            physical_semantic.append(sem)
            grounded.append(result.action)

        effective_attention = attention or legacy_attention
        if effective_attention:
            self.grounder.try_focus(effective_attention, world)

        return EmbodiedDecision(
            actions=grounded,
            semantic_actions=semantic,
            physical_semantic_actions=physical_semantic,
            spoken=spoken,
            say=say,
            attention=effective_attention,
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
        body_feedback: dict[str, Any] | None = None,
        social_context: dict[str, Any] | None = None,
    ) -> EmbodiedDecision:
        if self.client is None:
            return EmbodiedDecision(
                actions=[Action(ActionKind.WAIT.value, {}, "LLM unavailable; remain idle.")],
                semantic_actions=[SemanticAction("Observe", None, {})],
                physical_semantic_actions=[SemanticAction("Observe", None, {})],
                spoken=[],
                say="LLM unavailable; idling.",
                fallback=True,
            )

        payload = self._payload(world, mode, task, recent, body_feedback, social_context=social_context)
        first_error: Exception | None = None
        try:
            raw, used_vision = self._call(payload, image_bytes)
            decision = self._validate_and_ground(_extract_json(raw), world)
            decision.used_vision = used_vision
            return decision
        except (ValueError, GroundingError, json.JSONDecodeError) as exc:
            first_error = exc
            try:
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
            semantic_actions=[SemanticAction("Observe", None, {})],
            physical_semantic_actions=[SemanticAction("Observe", None, {})],
            spoken=[],
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
        self.last_body_feedback: dict[str, Any] | None = None

    def _pace_delay(self, pace: str) -> float:
        if not self.llm_controls_pace:
            return self.base_delay
        factor = {"slow": 2.0, "normal": 1.0, "fast": 0.5}.get(pace, 1.0)
        return max(0.0, self.base_delay * factor)

    def _recent_progress_summary(self, current_feedback: dict[str, Any], current_decision: EmbodiedDecision) -> dict[str, Any]:
        """Neutral proprioceptive summary over recent decisions; it does not prescribe a response."""
        rows = list(self.recent[-7:])
        distances: list[float] = []
        yaw_changes: list[float] = []
        signatures: list[str] = []
        for row in rows:
            feedback = row.get("body_feedback") or {}
            distances.append(abs(float(feedback.get("distance_moved", 0.0) or 0.0)))
            yaw_changes.append(abs(float(feedback.get("yaw_change_degrees", 0.0) or 0.0)))
            acts = row.get("semantic_actions") or []
            signatures.append("->".join(str(a.get("kind", "?")) for a in acts) or "Observe")

        distances.append(abs(float(current_feedback.get("distance_moved", 0.0) or 0.0)))
        yaw_changes.append(abs(float(current_feedback.get("yaw_change_degrees", 0.0) or 0.0)))
        signatures.append("->".join(a.kind for a in current_decision.semantic_actions) or "Observe")

        repeat_count = 1
        if signatures:
            last = signatures[-1]
            for sig in reversed(signatures[:-1]):
                if sig != last:
                    break
                repeat_count += 1
        total_distance = round(sum(distances), 3)
        total_yaw = round(sum(yaw_changes), 1)
        return {
            "window_decisions": len(distances),
            "total_distance_moved_m": total_distance,
            "total_abs_yaw_change_degrees": total_yaw,
            "decisions_with_translation": sum(1 for value in distances if value >= 0.02),
            "repeated_action_signature": signatures[-1] if signatures else "Observe",
            "repeated_action_count": repeat_count,
            "stagnant": bool(len(distances) >= 4 and total_distance < 0.05 and total_yaw < 5.0),
        }

    def step(
        self,
        mode: str = "autonomous",
        task: str | None = None,
        social_context: dict[str, Any] | None = None,
        print_prefix: str | None = None,
    ) -> tuple[EmbodiedDecision, dict[str, Any]]:
        self.decision_index += 1
        world_before_decision = self.thor.observe()
        image = None
        if self.use_vision and (self.decision_index == 1 or (self.decision_index - 1) % self.vision_every == 0):
            image = capture_frame_png(self.thor)

        decision = self.planner.decide(
            world_before_decision,
            mode,
            task,
            self.recent,
            image,
            body_feedback=self.last_body_feedback,
            social_context=social_context,
        )
        self.thor.config.step_delay_seconds = self._pace_delay(decision.pace)

        if self.print_decisions:
            semantic_summary = " -> ".join(
                a.kind + (f'("{a.target}")' if a.target else "") for a in decision.semantic_actions
            )
            labels = []
            if decision.attention:
                labels.append(f'focus="{decision.attention}"')
            if decision.used_vision:
                labels.append("vision")
            if decision.fallback:
                labels.append("fallback")
            suffix = (" " + " ".join(labels)) if labels else ""
            task_label = "task" if mode == "task" else "auto"
            task_label = print_prefix or task_label
            print(
                f"[{task_label} {self.decision_index:03d} | step {self.executor.total_steps:04d}] "
                f"pace={decision.pace} {semantic_summary}{suffix}: {decision.say}",
                flush=True,
            )

        successes: list[bool] = []
        generic_errors: list[str] = []
        primitive_steps = 0
        action_feedback: list[dict[str, Any]] = []

        physical_semantics = decision.physical_semantic_actions or []
        for semantic_action, action in zip(physical_semantics, decision.actions):
            if self.executor.total_steps >= self.executor.max_steps:
                break
            action_before = self.thor.observe()
            outcome = self.executor.execute(action)
            action_after = self.thor.observe()
            delta = pose_delta(action_before, action_after)
            primitive_steps += outcome.primitive_steps
            successes.append(outcome.success)
            action_feedback.append({
                "kind": semantic_action.kind,
                "success": bool(outcome.success),
                "primitive_steps": int(outcome.primitive_steps),
                **delta,
            })
            if outcome.success:
                continue

            generic_errors.append("action_failed")
            recovery = self.recovery.classify(action, outcome, action_after)
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
            action_feedback[-1]["recovery_occurred"] = bool(recovery.actions)
            break

        world_after_decision = self.thor.observe()
        decision_delta = pose_delta(world_before_decision, world_after_decision)
        self.last_body_feedback = {
            "actions": action_feedback,
            "decision_primitive_steps": primitive_steps,
            "decision_success": all(successes) if successes else True,
            **decision_delta,
        }
        self.last_body_feedback["recent_progress"] = self._recent_progress_summary(
            self.last_body_feedback, decision
        )

        record = {
            "mode": mode,
            "task": task,
            "say": decision.say,
            "spoken": list(decision.spoken or []),
            "attention": decision.attention,
            "pace": decision.pace,
            "semantic_actions": [
                {"kind": a.kind, "target": a.target, "params": a.params} for a in decision.semantic_actions
            ],
            "success": all(successes) if successes else True,
            "generic_errors": generic_errors,
            "primitive_steps": primitive_steps,
            "body_feedback": self.last_body_feedback,
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


RoamPlanner = EmbodiedPlanner
