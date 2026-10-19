from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .executor import ActionExecutor
from .embodied_skills import EmbodiedSkillLibrary
from .grounding import GroundingError, SemanticGrounder
from .llm_client import SyncLLMClient
from .memory import EpisodicMemory
from .models import Action, ActionKind, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter
from .task_contracts import (
    contract_from_context,
    next_required_interaction,
    semantic_reference_matches,
)
from .vision import capture_frame_png, embodiment_snapshot, pose_delta


EMBODIED_SYSTEM_PROMPT = """
You are embodied in a simulated 3D environment. You can see through a camera
when an image is attached and can control the body through the interface below.

When no user task is active, there is no required goal or preferred behavior.
You may move, look, interact with something you perceive, remain still, or do
nothing this cycle. When a user task is active, the exact user task is your
current objective; do not invent additional user objectives.

The controller does NOT give you an object catalog. Object names/IDs, hidden
objects, and simulator detections are not listed for you. If you independently
notice or refer to something, you may put that phrase in `focus` or use it as the
target of a semantic action. A hidden motor-grounding layer only translates your
own reference when it can match current perception or your own grounded focus.

Keep perception, intention, and outward status separate:
- `observation`: a SHORT report about the CURRENT attached camera/body state.
  Describe the present frame, not earlier reports. It is replaced next cycle and
  is not replayed to you as narrative history.
- `intention`: an OPTIONAL concise goal/interest that YOU choose for yourself.
  It may persist across turns if you still endorse it. Use null when you have no
  current self-chosen intention.
- `say`: OPTIONAL brief status for the human observer. It is not speech to other
  agents. Use Speak when you choose to communicate aloud with another embodied agent. In social mode you may also use TextPeer to send an explicit distance-independent text message.

`semantic_focus`, when present, exists only because YOU previously referred to
that thing. Its fields report neutral physical facts and currently possible body
actions for that already-selected focus. They are capabilities, not
recommendations. `last_body_feedback.recent_progress`, when present, is only a
factual summary of recent pose change and repeated action signatures.

`long_term_memory`, when present, contains compact records of your own prior
actions/tasks/speech plus coarse summaries of explicitly requested training. It
is memory, not a new task or preference. Detailed simulator route maps, hidden
object IDs, and old camera-description prose are not included.

Primitive body actions (NO target):
- MoveAhead, MoveBack, MoveLeft, MoveRight
- RotateLeft, RotateRight, LookUp, LookDown
- Observe or Wait (remain still for this cycle)

Semantic actions (target text REQUIRED):
- Approach: move near something you independently identify
- Face: turn the body toward something you independently identify
- PickupObject
- PutObject (place the currently held object onto/in a perceived receptacle)
- OpenObject
- CloseObject
- ToggleObjectOn
- ToggleObjectOff
- DropObject (no target; release the currently held object)

Important interface rules:
- Primitive movement/look/Observe/Wait actions do not take a target. If you want
  movement relative to something, use Approach or Face.
- `focus` is optional attention only; setting it does not cause movement.
- `actions` may be empty. An empty list means Observe/do nothing this cycle.
- `task_complete` is only a completion claim. The runtime independently verifies
  recognized text, manipulation, and connected-room exit tasks and may ignore an
  unsupported claim instead of ending the user task.
- Do not expose private chain-of-thought. Keep observation/intention/say concise.
- When a user task explicitly asks you to manipulate something (open, pick up, place, toggle, or close), do not repeatedly Approach after the target is already within reach. Use the requested semantic interaction and let simulator feedback verify success.
- When a user task explicitly requires texting a peer, use TextPeer. `say` is only
  observer status and does not send a message to another agent. Message wording is
  still your choice.
- `task_mechanical_contract`, when present, is derived only from the current user
  task. It states the requested capability and completion evidence; it is not an
  autonomous preference and contains no hidden object IDs or coordinates.
- A visible Door may be a real openable navigation boundary in custom scenes. Opening it changes only physical access; it does not create any obligation to enter or explore.

Return ONLY one JSON object:
{
  "observation": "brief current observation",
  "intention": null,
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
        r"Observe|Wait|Pass|Approach|Face|PickupObject|PutObject|OpenObject|CloseObject|ToggleObjectOn|ToggleObjectOff|DropObject)"
        r"(?:\s+(?:the\s+)?(.+?))?[.!]?",
        compact,
    )
    if action_match:
        kind = action_match.group(1)
        target = (action_match.group(2) or "").strip() or None
        primitive = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Observe", "Wait", "Pass", "DropObject"}
        return {
            "observation": "",
            "intention": None,
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
            "observation": compact[:300],
            "intention": None,
            "say": "",
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
    texted: list[dict[str, str]] = None
    observation: str = ""
    intention: str | None = None
    say: str = ""
    attention: str | None = None
    pace: str = "normal"
    task_complete: bool = False
    used_vision: bool = False
    fallback: bool = False
    motor_handoff: bool = False


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
        skill_library: EmbodiedSkillLibrary | None = None,
        skill_min_attempts: int = 5,
        skill_min_success_rate: float = 0.65,
    ):
        self.client = client
        self.grounder = grounder
        self.allow_interactions = allow_interactions
        self.max_burst_actions = max(1, min(12, int(max_burst_actions)))
        self.social_enabled = bool(social_enabled)
        self.agent_label = agent_label
        self.skill_library = skill_library
        self.skill_min_attempts = max(1, int(skill_min_attempts))
        self.skill_min_success_rate = max(0.0, min(1.0, float(skill_min_success_rate)))

    def _skill_ready(self, skill: str, object_type: str | None = None) -> bool:
        if self.skill_library is None:
            return False
        return self.skill_library.ready(
            skill, object_type,
            min_attempts=self.skill_min_attempts,
            min_success_rate=self.skill_min_success_rate,
        )

    @staticmethod
    def _preparatory_only(decision: EmbodiedDecision) -> bool:
        allowed = {
            "Observe", "Approach", "Face",
            ActionKind.MOVE_AHEAD.value, ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value, ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value, ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value, ActionKind.LOOK_DOWN.value,
            ActionKind.WAIT.value,
        }
        return all(
            row.kind in allowed
            for row in (decision.semantic_actions or [])
            if row.kind not in {"Speak", "TextPeer"}
        )

    @staticmethod
    def _install_motor_handoff(decision: EmbodiedDecision, sem: SemanticAction, action: Action) -> EmbodiedDecision:
        social_rows = [row for row in (decision.semantic_actions or []) if row.kind in {"Speak", "TextPeer"}]
        decision.semantic_actions = social_rows + [sem]
        decision.physical_semantic_actions = [sem]
        decision.actions = [action]
        decision.motor_handoff = True
        return decision

    @property
    def system_prompt(self) -> str:
        prompt = EMBODIED_SYSTEM_PROMPT.replace("MAX_BURST", str(self.max_burst_actions))
        if not self.allow_interactions:
            prompt += "\nDirect manipulation is disabled for this run; semantic Approach/Face remain available."
        if self.social_enabled:
            prompt += """

SOCIAL EMBODIMENT FACTS:
You are one independently embodied agent in a shared physical scene. The
`peer_identity` field names another independently embodied agent like you. That
peer has its own camera, memory, choices, task state, body, and ability to speak.
It is not an ordinary scene object. Its private observations, intentions, memory,
tasks, semantic focus, and hidden reasoning are never shared with you unless it
chooses to communicate information through speech.

The `peer_camera_cone_sensor` may identify that peer's body as lying inside your
camera cone and provides neutral distance/bearing geometry. With the v0.6 custom
Unity player the peer is rendered as a humanoid robot; with a stock player it may
still look simpler. The geometry sensor does not prove the body is unoccluded;
use the attached RGB image for appearance.

Knowing another agent exists does not impose a social goal. You are not required to speak, respond, cooperate, approach, follow, trust, avoid, or form any particular relationship. Those choices remain yours.

Social actions:
- Speak, with params.text containing the words you choose to say aloud. Speech is range-limited.
- TextPeer, with target set to the peer label and params.text containing the message. Text is distance-independent and delivered only because you explicitly chose to send it.
"""
            if self.agent_label:
                prompt += f"\nYou are Agent {self.agent_label}. This is a stable identity label, not a role, personality, or social instruction.\n"
        return prompt

    @staticmethod
    def _recent_for_llm(recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact mechanical history only; never replay prior observation/status prose."""
        out: list[dict[str, Any]] = []
        for item in recent[-4:]:
            out.append({
                "intention": item.get("intention"),
                "attention": item.get("attention"),
                "semantic_actions": [
                    {"kind": row.get("kind"), "target": row.get("target")}
                    for row in (item.get("semantic_actions") or [])[:4]
                ],
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
        current_intention: str | None = None,
        long_term_memory: list[dict[str, Any]] | None = None,
    ) -> str:
        contract = contract_from_context(
            task if mode == "task" else None,
            social_context=social_context,
            self_label=self.agent_label,
        )
        payload = {
            "mode": mode,
            "user_task": task if mode == "task" else None,
            "task_mechanical_contract": contract.for_llm() if mode == "task" else None,
            "embodiment": embodiment_snapshot(world),
            "last_body_feedback": body_feedback,
            "current_self_chosen_intention": current_intention,
            "semantic_focus": self.grounder.focus_snapshot(world),
            "recent_self_context": self._recent_for_llm(recent),
            "long_term_memory": list(long_term_memory or []),
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
            compact_error = " ".join(str(correction).split())[:220]
            user += "\nMECHANICAL_FEEDBACK=" + json.dumps({"previous_action_executable": False, "reason": compact_error})
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
            "Pickup": "PickupObject",
            "PickUp": "PickupObject",
            "Grab": "PickupObject",
            "Place": "PutObject",
            "Put": "PutObject",
            "Drop": "DropObject",
            "TurnOn": "ToggleObjectOn",
            "TurnOff": "ToggleObjectOff",
            "Text": "TextPeer",
        }
        return aliases.get(kind, kind)

    def _validate_and_ground(self, data: dict[str, Any], world: WorldState) -> EmbodiedDecision:
        observation = " ".join(str(data.get("observation", "")).split())[:360]
        intention_raw = data.get("intention")
        intention = None if intention_raw is None else " ".join(str(intention_raw).split())[:180] or None
        say = " ".join(str(data.get("say", "")).split())[:220]
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
            ActionKind.TOGGLE_ON.value,
            ActionKind.TOGGLE_OFF.value,
            "DropObject",
        }
        allowed = primitive | semantic_motion | (interactions if self.allow_interactions else set())
        if self.social_enabled:
            allowed.update({"Speak", "TextPeer"})

        # Preserve model-selected attention even when it was expressed as a legacy
        # primitive target such as MoveAhead("microwave"). It becomes focus only;
        # it does not secretly steer the primitive action.
        legacy_attention: str | None = None
        grounded: list[Action] = []
        semantic: list[SemanticAction] = []
        physical_semantic: list[SemanticAction] = []
        spoken: list[str] = []
        texted: list[dict[str, str]] = []

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

            if kind == "TextPeer":
                recipient = str(target or params.get("peer") or "").strip()
                text = str(params.get("text") or row.get("text") or "").strip()
                text = " ".join(text.split())[:800]
                if not recipient:
                    raise ValueError("TextPeer requires target set to the peer label.")
                if not text:
                    raise ValueError("TextPeer requires non-empty params.text.")
                semantic.append(SemanticAction("TextPeer", recipient, {"text": text}))
                texted.append({"recipient": recipient, "text": text})
                continue

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

            if kind == "DropObject":
                result = self.grounder.ground(kind, None, params, world, reason=say)
                sem = SemanticAction(kind, None, params)
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
            texted=texted,
            observation=observation,
            intention=intention,
            say=say,
            attention=effective_attention,
            pace=pace,
            task_complete=task_complete,
        )

    def _apply_task_motor_handoff(
        self,
        decision: EmbodiedDecision,
        world: WorldState,
        task: str | None,
        social_context: dict[str, Any] | None,
    ) -> EmbodiedDecision:
        """Hand explicit task mechanics to verified staged body skills.

        v0.10.3 no longer decides approach readiness from object-center distance.
        If the requested interaction is not currently possible, a trained approach
        skill resolves and physically reaches a GetInteractablePoses body pose.
        """
        if not task:
            return decision
        contract = contract_from_context(task, social_context=social_context, self_label=self.agent_label)
        preparation = self._preparatory_only(decision)

        if contract.kind == "ExitRegion":
            if not preparation or "discovery" not in str(world.scene or "").lower():
                return decision

            # Prefer a door explicitly selected by the semantic model when that
            # selection is physically grounded.  If the model has only supplied
            # the user-authorized region outcome (for example, "Leave the
            # kitchen"), the motor layer may privately choose an exit doorway.
            # Object IDs/coordinates remain below the semantic boundary.
            snapshot = self.grounder.focus_snapshot(world)
            focused_obj = self.grounder.focused_object(world)
            reference = None if not snapshot else snapshot.get("semantic_reference")
            chosen_door = bool(
                reference and focused_obj is not None and focused_obj.openable
                and "door" in f"{reference} {focused_obj.object_type} {focused_obj.name}".lower()
            )

            if not chosen_door:
                doors = [
                    obj for obj in world.objects.values()
                    if str(obj.object_type).lower() == "door"
                    and obj.openable
                    and obj.position is not None
                ]
                if not doors:
                    return decision

                def private_distance(obj):
                    if obj.distance is not None:
                        return float(obj.distance)
                    dx = float(obj.position.get("x", 0.0)) - float(world.pose.position.get("x", 0.0))
                    dz = float(obj.position.get("z", 0.0)) - float(world.pose.position.get("z", 0.0))
                    return (dx * dx + dz * dz) ** 0.5

                # Pure motor/topology choice: prefer the cheapest nearby doorway.
                # Stable object-id tie breaking avoids stochastic route churn.
                doors.sort(key=lambda obj: (private_distance(obj), str(obj.object_id)))
                focused_obj = doors[0]
                reference = "exit doorway"
                snapshot = None

            if focused_obj is None:
                return decision

            distance = None if focused_obj.distance is None else float(focused_obj.distance)
            if distance is None and focused_obj.position is not None:
                dx = float(focused_obj.position.get("x", 0.0)) - float(world.pose.position.get("x", 0.0))
                dz = float(focused_obj.position.get("z", 0.0)) - float(world.pose.position.get("z", 0.0))
                distance = (dx * dx + dz * dz) ** 0.5
            reached = distance is not None and distance <= self.grounder.interaction_distance + 0.15
            directly_actionable = bool(reached and focused_obj.visible)

            if focused_obj.is_open is True:
                if directly_actionable and self._skill_ready("door_cross", focused_obj.object_type):
                    return self._install_motor_handoff(
                        decision,
                        SemanticAction("CrossThreshold", str(reference), {}),
                        Action(
                            ActionKind.CROSS_THRESHOLD.value,
                            {"objectId": focused_obj.object_id, "_skill": "door_cross"},
                            "trained body skill: cross private exit doorway for explicit ExitRegion task",
                        ),
                    )
                if self._skill_ready("door_approach", focused_obj.object_type):
                    return self._install_motor_handoff(
                        decision,
                        SemanticAction("Approach", str(reference), {}),
                        Action(
                            ActionKind.NAVIGATE_TO.value,
                            {
                                "objectId": focused_obj.object_id,
                                "interactionDistance": self.grounder.interaction_distance,
                                "_skill": "door_approach",
                            },
                            "trained body skill: reach private open exit doorway",
                        ),
                    )
                return decision

            if directly_actionable:
                action = Action(
                    ActionKind.OPEN.value,
                    {"objectId": focused_obj.object_id, "forceAction": False, "_skill": "door_open"},
                    "explicit ExitRegion task: open private exit doorway",
                )
                return self._install_motor_handoff(
                    decision,
                    SemanticAction("OpenObject", str(reference), {}),
                    action,
                )

            if self._skill_ready("door_approach", focused_obj.object_type):
                return self._install_motor_handoff(
                    decision,
                    SemanticAction("Approach", str(reference), {}),
                    Action(
                        ActionKind.NAVIGATE_TO.value,
                        {
                            "objectId": focused_obj.object_id,
                            "interactionDistance": self.grounder.interaction_distance,
                            "_skill": "door_approach",
                            "_interactionKind": ActionKind.OPEN.value,
                        },
                        "trained body skill: resolve and reach private exit-door interaction pose",
                    ),
                )
            return decision

        required = next_required_interaction(contract, world)
        if required is None:
            return decision
        required_kind, required_target = required
        if any(
            row.kind == required_kind and (required_target is None or semantic_reference_matches(row.target, required_target))
            for row in (decision.semantic_actions or [])
        ):
            return decision

        if required_kind == "DropObject":
            if contract.kind != "DropObject" or world.held_object_id is None or not preparation:
                return decision
            result = self.grounder.ground("DropObject", None, {}, world, reason="explicit user-task motor handoff")
            return self._install_motor_handoff(decision, SemanticAction("DropObject", None, {}), result.action)

        snapshot = self.grounder.focus_snapshot(world)
        if not snapshot or not required_target:
            return decision
        if not semantic_reference_matches(snapshot.get("semantic_reference"), required_target):
            if decision.attention and semantic_reference_matches(decision.attention, required_target):
                self.grounder.try_focus(decision.attention, world)
                snapshot = self.grounder.focus_snapshot(world)
            if not snapshot or not semantic_reference_matches(snapshot.get("semantic_reference"), required_target):
                return decision

        focused_obj = self.grounder.focused_object(world)
        if focused_obj is None or not preparation:
            return decision
        possible = set(snapshot.get("currently_possible_actions") or [])
        approach_skill = {
            ActionKind.PICKUP.value: "pickup_approach",
            ActionKind.PUT.value: "place_approach",
            ActionKind.TOGGLE_ON.value: "toggle_approach",
            ActionKind.TOGGLE_OFF.value: "toggle_approach",
        }.get(required_kind)
        interaction_skill = {
            ActionKind.PICKUP.value: "pickup_grasp",
            ActionKind.PUT.value: "place_release",
            ActionKind.TOGGLE_ON.value: "toggle_interaction",
            ActionKind.TOGGLE_OFF.value: "toggle_interaction",
        }.get(required_kind)

        if required_kind not in possible and approach_skill and self._skill_ready(approach_skill, focused_obj.object_type):
            return self._install_motor_handoff(
                decision,
                SemanticAction("Approach", required_target, {}),
                Action(ActionKind.NAVIGATE_TO.value, {"objectId": focused_obj.object_id, "interactionDistance": self.thor.config.interaction_distance, "_skill": approach_skill, "_interactionKind": required_kind}, "trained body skill: resolve and reach explicit task interaction pose"),
            )
        if required_kind not in possible:
            return decision
        result = self.grounder.ground(required_kind, required_target, {}, world, reason="explicit user-task motor handoff")
        params = dict(result.action.params)
        if interaction_skill:
            params["_skill"] = interaction_skill
        action = Action(result.action.kind, params, result.action.reason)
        return self._install_motor_handoff(decision, SemanticAction(required_kind, required_target, {}), action)

    def _validate_task_interaction(
        self,
        decision: EmbodiedDecision,
        world: WorldState,
        task: str | None,
        social_context: dict[str, Any] | None,
    ) -> None:
        if not task:
            return
        contract = contract_from_context(task, social_context=social_context, self_label=self.agent_label)
        required = next_required_interaction(contract, world)
        if required is None:
            return
        expected_kind, expected_target = required
        direct = [
            row for row in (decision.semantic_actions or [])
            if row.kind in {
                ActionKind.PICKUP.value, ActionKind.PUT.value, ActionKind.OPEN.value,
                ActionKind.CLOSE.value, ActionKind.TOGGLE_ON.value, ActionKind.TOGGLE_OFF.value,
                "DropObject",
            }
        ]
        if not direct:
            return
        for row in direct:
            kind_ok = row.kind == expected_kind
            target_ok = expected_target is None or semantic_reference_matches(row.target, expected_target)
            if kind_ok and target_ok:
                return
        target_note = f" target '{expected_target}'" if expected_target else ""
        raise ValueError(
            f"The active user task's next verified interaction is {expected_kind}{target_note}. "
            "A different direct manipulation would not satisfy that task; choose preparation movement or the requested interaction."
        )

    def _validate_task_channel(
        self,
        decision: EmbodiedDecision,
        task: str | None,
        social_context: dict[str, Any] | None,
    ) -> None:
        if not task:
            return
        contract = contract_from_context(task, social_context=social_context, self_label=self.agent_label)
        if not contract.is_text:
            return
        expected = (contract.recipient or "").strip().lower()
        delivered = [row for row in (decision.texted or []) if str(row.get("text") or "").strip()]
        if expected:
            delivered = [row for row in delivered if str(row.get("recipient") or "").strip().lower() == expected]
        if not delivered:
            target_note = f" to {contract.recipient}" if contract.recipient else " to the peer"
            raise ValueError(
                "The active user task explicitly requires a TextPeer action" + target_note +
                ". `say`/observation does not deliver a peer message. Choose the message wording yourself and emit TextPeer."
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
        current_intention: str | None = None,
        long_term_memory: list[dict[str, Any]] | None = None,
    ) -> EmbodiedDecision:
        if self.client is None:
            return EmbodiedDecision(
                actions=[Action(ActionKind.WAIT.value, {}, "LLM unavailable; remain idle.")],
                semantic_actions=[SemanticAction("Observe", None, {})],
                physical_semantic_actions=[SemanticAction("Observe", None, {})],
                spoken=[],
                texted=[],
                observation="",
                intention=None,
                say="LLM unavailable; idling.",
                fallback=True,
            )

        payload = self._payload(
            world,
            mode,
            task,
            recent,
            body_feedback,
            social_context=social_context,
            current_intention=current_intention,
            long_term_memory=long_term_memory,
        )
        first_error: Exception | None = None
        try:
            raw, used_vision = self._call(payload, image_bytes)
            decision = self._validate_and_ground(_extract_json(raw), world)
            decision = self._apply_task_motor_handoff(decision, world, task, social_context)
            self._validate_task_interaction(decision, world, task, social_context)
            self._validate_task_channel(decision, task, social_context)
            decision.used_vision = used_vision
            return decision
        except (ValueError, GroundingError, json.JSONDecodeError) as exc:
            first_error = exc
            try:
                raw, used_vision = self._call(payload, image_bytes, correction=str(exc))
                decision = self._validate_and_ground(_extract_json(raw), world)
                decision = self._apply_task_motor_handoff(decision, world, task, social_context)
                self._validate_task_interaction(decision, world, task, social_context)
                self._validate_task_channel(decision, task, social_context)
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
            texted=[],
            observation="",
            intention=None,
            say=f"LLM command failed: {type(first_error).__name__}: {str(first_error)[:180]}",
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
        skill_library: EmbodiedSkillLibrary | None = None,
        skill_min_attempts: int = 5,
        skill_min_success_rate: float = 0.65,
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
        self.current_intention: str | None = None
        self.skill_library = skill_library
        self.skill_min_attempts = max(1, int(skill_min_attempts))
        self.skill_min_success_rate = max(0.0, min(1.0, float(skill_min_success_rate)))

    def _skill_ready(self, skill: str, object_type: str | None = None) -> bool:
        if self.skill_library is None:
            return False
        return self.skill_library.ready(
            skill,
            object_type,
            min_attempts=self.skill_min_attempts,
            min_success_rate=self.skill_min_success_rate,
        )

    def _record_runtime_skill(self, action: Action, before: WorldState, success: bool, primitive_steps: int, *, recovery: bool = False, recovery_success: bool = False, recovery_exhausted: bool = False) -> None:
        if self.skill_library is None:
            return
        object_id = action.params.get("objectId")
        obj = before.objects.get(object_id) if object_id else None
        object_type = None if obj is None else obj.object_type
        skill = action.params.get("_skill")
        if not skill:
            if action.kind == ActionKind.PICKUP.value:
                skill = "pickup_grasp"
            elif action.kind == ActionKind.PUT.value:
                skill = "place_release"
            elif action.kind in {ActionKind.TOGGLE_ON.value, ActionKind.TOGGLE_OFF.value}:
                skill = "toggle_interaction"
            elif action.kind == ActionKind.CROSS_THRESHOLD.value:
                skill = "door_cross"
            elif action.kind == ActionKind.OPEN.value and str(object_type or "").lower() == "door":
                skill = "door_open"
        if not skill:
            return
        self.skill_library.record(str(skill), success=bool(success), primitive_steps=int(primitive_steps), object_type=object_type, recovery=bool(recovery), recovery_success=bool(recovery_success), recovery_exhausted=bool(recovery_exhausted))
        if self.skill_library.path is not None:
            self.skill_library.save()

    @staticmethod
    def _preparatory_only(decision: EmbodiedDecision) -> bool:
        allowed = {
            "Observe", "Approach", "Face",
            ActionKind.MOVE_AHEAD.value, ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value, ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value, ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value, ActionKind.LOOK_DOWN.value,
            ActionKind.WAIT.value,
        }
        return all(
            row.kind in allowed
            for row in (decision.semantic_actions or [])
            if row.kind not in {"Speak", "TextPeer"}
        )

    @staticmethod
    def _install_motor_handoff(decision: EmbodiedDecision, sem: SemanticAction, action: Action) -> EmbodiedDecision:
        social_rows = [row for row in (decision.semantic_actions or []) if row.kind in {"Speak", "TextPeer"}]
        decision.semantic_actions = social_rows + [sem]
        decision.physical_semantic_actions = [sem]
        decision.actions = [action]
        decision.motor_handoff = True
        return decision

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
            current_intention=self.current_intention,
            long_term_memory=self.memory.safe_context(limit=12),
        )
        self.current_intention = decision.intention
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
            if decision.motor_handoff:
                labels.append("motor-handoff")
            suffix = (" " + " ".join(labels)) if labels else ""
            task_label = "task" if mode == "task" else "auto"
            task_label = print_prefix or task_label
            report = decision.observation or decision.say
            intention_text = f' intention="{decision.intention}"' if decision.intention else ""
            print(
                f"[{task_label} {self.decision_index:03d} | step {self.executor.total_steps:04d}] "
                f"pace={decision.pace} {semantic_summary}{suffix}{intention_text}: {report}",
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
            if self.skill_library is not None:
                for aux in (outcome.event_metadata.get("auxiliarySkills") or []):
                    skill = str(aux.get("skill") or "").strip()
                    if not skill:
                        continue
                    self.skill_library.record(
                        skill,
                        success=bool(aux.get("success")),
                        primitive_steps=int(aux.get("primitive_steps", 0)),
                        object_type=aux.get("object_type"),
                        recovery=bool(aux.get("recovery", False)),
                    )
                if outcome.event_metadata.get("auxiliarySkills") and self.skill_library.path is not None:
                    self.skill_library.save()
            action_feedback.append({
                "kind": semantic_action.kind,
                "success": bool(outcome.success),
                "primitive_steps": int(outcome.primitive_steps),
                **delta,
            })
            if outcome.success:
                internal_recovery = any(
                    bool(aux.get("recovery"))
                    for aux in (outcome.event_metadata.get("auxiliarySkills") or [])
                    if not str(aux.get("skill") or "").strip()
                )
                self._record_runtime_skill(action, action_before, True, outcome.primitive_steps, recovery=internal_recovery, recovery_success=internal_recovery)
                continue

            generic_errors.append("action_failed")
            internal_recovery = any(
                bool(aux.get("recovery"))
                for aux in (outcome.event_metadata.get("auxiliarySkills") or [])
                if not str(aux.get("skill") or "").strip()
            )
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
            self._record_runtime_skill(action, action_before, False, outcome.primitive_steps, recovery=internal_recovery, recovery_exhausted=internal_recovery)
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
            "observation": decision.observation,
            "intention": decision.intention,
            "say": decision.say,
            "spoken": list(decision.spoken or []),
            "texted": list(decision.texted or []),
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
            "motor_handoff": decision.motor_handoff,
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
