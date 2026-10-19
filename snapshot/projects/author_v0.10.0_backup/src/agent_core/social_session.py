from __future__ import annotations

from dataclasses import dataclass, field
import json
import queue
import threading
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping

from .roam import EmbodiedLoop
from .social import SpeechBus, TextBus, TextMessage, peer_presence_for
from .task_contracts import (
    TaskBaseline,
    capture_task_baseline,
    contract_satisfied,
    parse_task_contract,
)


@dataclass
class SocialControlState:
    tasks: dict[int, str | None] = field(default_factory=lambda: {0: None, 1: None})
    quit_requested: bool = False


class SocialSessionController:
    """Routes user commands to A, B, or both without changing autonomous preferences."""

    def __init__(self, labels: Mapping[int, str] | None = None):
        self.labels = dict(labels or {0: "A", 1: "B"})
        self.state = SocialControlState(tasks={agent_id: None for agent_id in self.labels})

    def _ids_for_prefix(self, prefix: str) -> list[int]:
        p = prefix.lower()
        if p in {"both", "all"}:
            return list(sorted(self.labels))
        for agent_id, label in self.labels.items():
            if p == str(label).lower() or p == str(agent_id):
                return [agent_id]
        return []

    def handle(self, line: str) -> str:
        text = line.strip()
        if not text:
            return ""
        low = text.lower()
        if low in {"/quit", "/exit", "quit", "exit"}:
            self.state.quit_requested = True
            return "Social session exit requested."
        if low == "/status":
            parts = []
            for agent_id in sorted(self.labels):
                task = self.state.tasks.get(agent_id)
                parts.append(f"{self.labels[agent_id]}={'task: ' + task if task else 'autonomous'}")
            return " | ".join(parts)
        if low == "/help":
            return "Commands: /a <task>, /b <task>, /both <task>, /a roam, /b roam, /both roam, /status, /quit. Plain text assigns the same task to both."

        if text.startswith("/"):
            head, _, body = text[1:].partition(" ")
            ids = self._ids_for_prefix(head)
            if not ids:
                return f"Unknown agent route '/{head}'. Use /a, /b, or /both."
            body = body.strip()
            if not body:
                return f"No task supplied for /{head}."
            if body.lower() in {"roam", "free", "cancel"}:
                for agent_id in ids:
                    self.state.tasks[agent_id] = None
                return f"Returned {', '.join(self.labels[i] for i in ids)} to autonomous mode."
            for agent_id in ids:
                self.state.tasks[agent_id] = body
            return f"Task assigned to {', '.join(self.labels[i] for i in ids)}: {body}"

        for agent_id in self.labels:
            self.state.tasks[agent_id] = text
        return f"Shared task assigned to all agents: {text}"


class SocialEmbodiedSession:
    """
    Two independent embodied LLM contexts alternating inside one shared Unity scene.

    The coordinator serializes inference. The only cross-agent information it moves is explicit communication: range-limited
    speech and deliberately emitted TextPeer messages.
    """

    def __init__(
        self,
        loops: Mapping[int, EmbodiedLoop],
        labels: Mapping[int, str],
        speech_bus: SpeechBus,
        text_bus: TextBus | None = None,
        max_rounds: int = 500,
        input_func: Callable[[str], str] = input,
        dashboard: Any | None = None,
        peer_sensor_enabled: bool = True,
        peer_camera_cone_margin_degrees: float = 3.0,
        max_texts_per_turn: int = 2,
        peer_text_every: int = 0,
        peer_text_fallback: bool = True,
        peer_text_max_chars: int = 500,
    ):
        self.loops = dict(loops)
        self.labels = dict(labels)
        self.speech_bus = speech_bus
        self.text_bus = text_bus or TextBus(labels)
        self.max_texts_per_turn = max(1, int(max_texts_per_turn))
        self.peer_text_every = max(0, int(peer_text_every))
        self.peer_text_fallback = bool(peer_text_fallback)
        self.peer_text_max_chars = max(80, int(peer_text_max_chars))
        self._peer_text_turns: dict[int, int] = {agent_id: 0 for agent_id in self.labels}
        self._recent_peer_texts: dict[int, list[str]] = {agent_id: [] for agent_id in self.labels}
        self._task_text_events: dict[int, list[TextMessage]] = {agent_id: [] for agent_id in self.labels}
        self._task_text_baseline: dict[int, int] = {agent_id: 0 for agent_id in self.labels}
        self._tracked_tasks: dict[int, str | None] = {agent_id: None for agent_id in self.labels}
        self._task_baselines: dict[int, TaskBaseline | None] = {agent_id: None for agent_id in self.labels}
        self._ignored_claims: set[tuple[int, str]] = set()
        self.max_rounds = int(max_rounds)
        self.controller = SocialSessionController(self.labels)
        self.input_func = input_func
        self.commands: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.dashboard = dashboard
        self.peer_sensor_enabled = bool(peer_sensor_enabled)
        self.peer_camera_cone_margin_degrees = float(peer_camera_cone_margin_degrees)

    def _read_commands(self) -> None:
        while not self.controller.state.quit_requested:
            try:
                line = self.input_func("")
            except (EOFError, KeyboardInterrupt):
                return
            self.commands.put(line)

    def _start_reader(self) -> None:
        self._reader = threading.Thread(target=self._read_commands, name="social-session-input", daemon=True)
        self._reader.start()

    def _drain_commands(self) -> None:
        while True:
            try:
                line = self.commands.get_nowait()
            except queue.Empty:
                return
            message = self.controller.handle(line)
            if message:
                print(f"[user-control] {message}", flush=True)
                if self.dashboard is not None:
                    self.dashboard.append_event("user-control", message)
                    for agent_id, task in self.controller.state.tasks.items():
                        self.dashboard.set_task_state(agent_id, task)
                for agent_id, task in self.controller.state.tasks.items():
                    if task != self._tracked_tasks.get(agent_id):
                        self._tracked_tasks[agent_id] = task
                        self._task_text_baseline[agent_id] = len(self._task_text_events.get(agent_id, []))
                        self._task_baselines[agent_id] = (
                            capture_task_baseline(self.loops[agent_id].thor.observe()) if task else None
                        )
                        self._ignored_claims = {
                            row for row in self._ignored_claims if row[0] != agent_id
                        }

    def _worlds(self):
        return {agent_id: loop.thor.observe() for agent_id, loop in self.loops.items()}

    def _complete_task_if_needed(self, agent_id: int, claimed: bool = False) -> None:
        task = self.controller.state.tasks.get(agent_id)
        if not task:
            return
        loop = self.loops[agent_id]
        label = self.labels[agent_id]
        peer_labels = [self.labels[i] for i in sorted(self.labels) if i != agent_id]
        contract = parse_task_contract(task, self_label=label, peer_labels=peer_labels)
        world_verified = bool(
            contract.is_world_verifiable
            and contract_satisfied(contract, loop.thor.observe(), self._task_baselines.get(agent_id))
        )

        text_verified = False
        if contract.is_text:
            baseline = self._task_text_baseline.get(agent_id, 0)
            recent_events = self._task_text_events.get(agent_id, [])[baseline:]
            if contract.recipient:
                wanted = contract.recipient.strip().lower()
                text_verified = any(event.recipient_label.strip().lower() == wanted for event in recent_events)
            else:
                text_verified = bool(recent_events)

        done = bool(world_verified or text_verified)
        if not done:
            if claimed and (agent_id, task) not in self._ignored_claims:
                self._ignored_claims.add((agent_id, task))
                print(f"[{label} task-claim ignored] {task} (no verifiable completion evidence yet)", flush=True)
                loop.memory.add(
                    "user_task_completion_claim_ignored",
                    {"task": task, "reason": "no verified simulator or delivered-text evidence"},
                )
                if self.dashboard is not None:
                    self.dashboard.append_event(
                        "task-claim-ignored",
                        "LLM completion claim ignored; waiting for verifiable evidence.",
                        agent_id=agent_id,
                    )
            return

        print(f"[{label} task-complete] {task}", flush=True)
        if self.dashboard is not None:
            self.dashboard.append_event("task-complete", task, agent_id=agent_id)
        loop.memory.add(
            "user_task_complete",
            {
                "task": task,
                "verified_by_simulator": bool(world_verified),
                "verified_text_delivery": bool(text_verified),
                "claimed_by_llm": bool(claimed),
            },
        )
        self.controller.state.tasks[agent_id] = None
        self._tracked_tasks[agent_id] = None
        self._task_baselines[agent_id] = None
        self._task_text_baseline[agent_id] = len(self._task_text_events.get(agent_id, []))
        self._ignored_claims = {row for row in self._ignored_claims if row[0] != agent_id}
        if self.dashboard is not None:
            self.dashboard.set_task_state(agent_id, None)
        print(f"[{label}] Returned to autonomous mode.", flush=True)

    def _other_agent_id(self, agent_id: int) -> int | None:
        for peer_id in sorted(self.labels):
            if peer_id != agent_id:
                return peer_id
        return None

    @staticmethod
    def _clean_peer_text(raw: str, max_chars: int) -> str:
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        # Be tolerant when a local model returns {"text":"..."} despite being asked
        # for plain text. We never expose hidden reasoning; only the explicit message.
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                text = str(data.get("text") or data.get("message") or "").strip()
            except Exception:
                pass
        text = " ".join(text.split())[:max_chars].strip()
        if text.strip().upper().replace(".", "") == "NO_MESSAGE":
            return ""
        return text

    @staticmethod
    def _peer_text_key(text: str) -> str:
        return " ".join(str(text or "").lower().replace('"', "").replace("'", "").split()).strip(" .!?")

    def _peer_text_is_redundant(self, agent_id: int, text: str) -> bool:
        candidate = self._peer_text_key(text)
        if not candidate:
            return True
        for prior in self._recent_peer_texts.get(agent_id, [])[-8:]:
            key = self._peer_text_key(prior)
            if not key:
                continue
            if candidate == key:
                return True
            if len(candidate) >= 24 and len(key) >= 24 and SequenceMatcher(None, candidate, key).ratio() >= 0.88:
                return True
        return False

    def _remember_peer_text(self, sender_id: int, recipient_id: int, text: str) -> None:
        for agent_id in (sender_id, recipient_id):
            rows = self._recent_peer_texts.setdefault(agent_id, [])
            rows.append(str(text))
            if len(rows) > 12:
                del rows[:-12]

    def _compose_peer_text(
        self,
        agent_id: int,
        recipient_id: int,
        *,
        task: str | None,
        decision: Any,
        received_texts: list[TextMessage],
        required_by_task: bool,
    ) -> tuple[str, str]:
        """Compose a public peer message from this agent's own state only.

        Returns (text, origin). origin is llm-peer-compose or status-relay-fallback.
        No peer-private camera, memory, task, focus, or reasoning is provided.
        """
        label = self.labels[agent_id]
        recipient_label = self.labels[recipient_id]
        loop = self.loops[agent_id]
        client = getattr(loop.planner, "client", None)
        payload = {
            "sender": label,
            "recipient": recipient_label,
            "active_user_task": task,
            "your_current_observation": getattr(decision, "observation", None),
            "your_current_intention": getattr(decision, "intention", None),
            "your_current_focus": getattr(decision, "attention", None),
            "peer_messages_received_this_turn": [
                {"from": event.sender_label, "text": event.text}
                for event in received_texts[-3:]
            ],
        }
        if client is not None:
            system = (
                f"You are Agent {label}, one independently embodied agent in a shared world. "
                f"Write one concise peer-to-peer text message to Agent {recipient_label}. "
                "Use only your own public observation, intention, active user task, and messages you actually received. "
                "Do not invent the peer's private state. Do not quote, echo, or paraphrase a peer message merely to acknowledge it. "
                "Contribute a new observation, a direct answer, a useful question, or a concrete coordination update. "
                "If you have nothing new or useful to communicate, return exactly NO_MESSAGE. "
                "Return only the message text, with no JSON and no analysis."
            )
            if required_by_task:
                system += " The user's active task explicitly requires peer communication, so the message should directly advance that task."
            try:
                raw = client.chat(
                    system,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    temperature=0.5,
                    max_tokens=120,
                )
                text = self._clean_peer_text(raw, self.peer_text_max_chars)
                if not text and str(raw or "").strip().upper().replace(".", "") == "NO_MESSAGE":
                    return "", "llm-no-message"
                if text:
                    if not required_by_task and self._peer_text_is_redundant(agent_id, text):
                        return "", "duplicate-suppressed"
                    return text, "llm-peer-compose"
            except Exception as exc:
                loop.memory.add("peer_text_compose_failure", {"error": f"{type(exc).__name__}: {str(exc)[:180]}"})

        if not self.peer_text_fallback:
            return "", "none"

        # Reliability fallback is visibly labeled and contains only this agent's public
        # task/observation. It is transport assistance, not hidden LLM speech.
        if required_by_task and task:
            fallback = f"[status relay] Communication was requested for this task: {task}"
        else:
            observation = " ".join(str(getattr(decision, "observation", "") or "").split())
            intention = " ".join(str(getattr(decision, "intention", "") or "").split())
            fallback = f"[status relay] {observation or intention or 'I am active in the shared scene.'}"
        fallback = fallback[: self.peer_text_max_chars]
        if not required_by_task and self._peer_text_is_redundant(agent_id, fallback):
            return "", "duplicate-suppressed"
        return fallback, "status-relay-fallback"

    def _deliver_peer_text(
        self,
        agent_id: int,
        recipient: str | int,
        body: str,
        *,
        origin: str,
        count_for_task: bool,
    ) -> TextMessage | None:
        loop = self.loops[agent_id]
        label = self.labels[agent_id]
        event = self.text_bus.send(agent_id, recipient, body)
        if event is None:
            loop.memory.add("text_failure", {"recipient": str(recipient), "reason": "invalid recipient or empty text", "origin": origin})
            print(f"[{label} text] Message not delivered: invalid peer target.", flush=True)
            return None
        self._remember_peer_text(agent_id, event.recipient_id, event.text)
        loop.memory.add("sent_text", {"to": event.recipient_label, "text": event.text, "origin": origin})
        if count_for_task:
            self._task_text_events.setdefault(agent_id, []).append(event)
        print(f"[{label} texts {event.recipient_label}] {event.text}", flush=True)
        if self.dashboard is not None:
            self.dashboard.append_event(
                "text",
                f"to {event.recipient_label}: {event.text}",
                agent_id=agent_id,
                peer_id=event.recipient_id,
            )
        return event

    def run(self) -> int:
        names = ", ".join(self.labels[i] for i in sorted(self.labels))
        print(f"Social embodied session started with agents: {names}.", flush=True)
        print("Each agent has its own camera, memory, focus, task state, and LLM context.", flush=True)
        print("Speech is explicit and range-limited; texting is explicit and distance-independent; private agent state is never copied between agents.", flush=True)
        if self.peer_text_every > 0:
            print(f"Visible peer-text cadence enabled: each agent gets a compose/send opportunity every {self.peer_text_every} turn(s).", flush=True)
        print("Commands: /a <task>  /b <task>  /both <task>  /a roam  /b roam  /both roam  /status  /quit", flush=True)

        if self.dashboard is not None:
            self.dashboard.start()
            self.dashboard.refresh_frames(self.loops)
            self.dashboard.append_event("session", f"Social embodied session started with agents: {names}.")
            print(f"Observer dashboard: {self.dashboard.url}", flush=True)
            print("Dashboard is read-only; it receives reports after decisions and never feeds state back to either agent.", flush=True)

        self._start_reader()
        try:
            for _round in range(1, self.max_rounds + 1):
                for agent_id in sorted(self.loops):
                    self._drain_commands()
                    if self.controller.state.quit_requested:
                        return 0
                    loop = self.loops[agent_id]
                    label = self.labels[agent_id]
                    if loop.executor.total_steps >= loop.executor.max_steps:
                        print(f"[{label}] Primitive-action budget exhausted; this body will remain idle.", flush=True)
                        if self.dashboard is not None:
                            self.dashboard.append_event("budget", "Primitive-action budget exhausted; body idle.", agent_id=agent_id)
                            self.dashboard.refresh_frames(self.loops)
                        continue

                    task = self.controller.state.tasks.get(agent_id)
                    if task:
                        self._complete_task_if_needed(agent_id, claimed=False)
                        task = self.controller.state.tasks.get(agent_id)

                    heard = self.speech_bus.consume(agent_id)
                    if heard:
                        for event in heard:
                            loop.memory.add("heard_speech", event.for_llm())
                            print(f"[{label} hears {event.speaker_label}] {event.text}", flush=True)
                            if self.dashboard is not None:
                                self.dashboard.append_event(
                                    "speech",
                                    f"heard {event.speaker_label}: {event.text}",
                                    agent_id=agent_id,
                                    peer_id=event.speaker_id,
                                )

                    texts = self.text_bus.consume(agent_id)
                    if texts:
                        for event in texts:
                            loop.memory.add("received_text", event.for_llm())
                            print(f"[{label} text from {event.sender_label}] {event.text}", flush=True)
                            if self.dashboard is not None:
                                self.dashboard.append_event("text", f"from {event.sender_label}: {event.text}", agent_id=agent_id, peer_id=event.sender_id)

                    worlds_before = self._worlds()
                    peer_presence = peer_presence_for(
                        agent_id,
                        worlds_before,
                        self.labels,
                        field_of_view_degrees=float(loop.thor.config.field_of_view),
                        speech_range_m=self.speech_bus.speech_range_m,
                        camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                    )
                    model_peer_sensor = peer_presence if self.peer_sensor_enabled else []
                    known_peers = [
                        {
                            "label": self.labels[peer_id],
                            "relation": "another independently embodied agent like you",
                            "private_state_shared": False,
                        }
                        for peer_id in sorted(self.labels)
                        if peer_id != agent_id
                    ]
                    social_context = {
                        "self_identity": {
                            "label": label,
                            "kind": "independently embodied agent",
                        },
                        "peer_identity": known_peers,
                        "heard_since_last_turn": [event.for_llm() for event in heard],
                        "texts_since_last_turn": [event.for_llm() for event in texts],
                        "text_channel_available": True,
                        "speech_range_m": self.speech_bus.speech_range_m,
                        # Identity-aware geometry sensor. It identifies only the known peer body;
                        # it never supplies ordinary scene-object names or private peer state.
                        "peer_camera_cone_sensor": [item.for_llm() for item in model_peer_sensor],
                    }
                    mode = "task" if task else "autonomous"
                    decision, record = loop.step(
                        mode=mode,
                        task=task,
                        social_context=social_context,
                        print_prefix=f"{label}-{mode}",
                    )

                    if self.dashboard is not None:
                        worlds_after = self._worlds()
                        dashboard_peer_presence = peer_presence_for(
                            agent_id,
                            worlds_after,
                            self.labels,
                            field_of_view_degrees=float(loop.thor.config.field_of_view),
                            speech_range_m=self.speech_bus.speech_range_m,
                            camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                        )
                        # Keep peer geometry for BOTH observer panels synchronized with
                        # the same shared-world state after every embodied action.
                        for observer_id, observer_loop in self.loops.items():
                            current_presence = peer_presence_for(
                                observer_id,
                                worlds_after,
                                self.labels,
                                field_of_view_degrees=float(observer_loop.thor.config.field_of_view),
                                speech_range_m=self.speech_bus.speech_range_m,
                                camera_cone_margin_degrees=self.peer_camera_cone_margin_degrees,
                            )
                            self.dashboard.set_peer_presence(observer_id, current_presence)
                        self.dashboard.update_agent(
                            agent_id,
                            loop,
                            mode=mode,
                            task=task,
                            decision=decision,
                            record=record,
                            heard=heard,
                            peer_presence=dashboard_peer_presence,
                        )
                        # Refresh both cameras after every embodied turn so the two
                        # side-by-side panels represent the same current shared world.
                        self.dashboard.refresh_frames(self.loops)

                    for utterance in decision.spoken or []:
                        loop.memory.add("spoken_speech", {"text": utterance})
                        delivered = self.speech_bus.speak(agent_id, utterance, self._worlds())
                        print(f"[{label} says] {utterance}", flush=True)
                        if self.dashboard is not None:
                            self.dashboard.append_event("speech", f"said: {utterance}", agent_id=agent_id)
                        if not delivered:
                            print(f"[{label} speech] No other agent was within hearing range.", flush=True)
                            if self.dashboard is not None:
                                self.dashboard.append_event(
                                    "speech",
                                    "No other agent was within hearing range.",
                                    agent_id=agent_id,
                                )

                    sent_this_turn = False
                    for text_request in (decision.texted or [])[: self.max_texts_per_turn]:
                        recipient = text_request.get("recipient", "")
                        body = text_request.get("text", "")
                        event = self._deliver_peer_text(
                            agent_id, recipient, body, origin="model-action", count_for_task=True
                        )
                        sent_this_turn = sent_this_turn or event is not None

                    # v0.9.3: if a user explicitly asked for texting but the normal
                    # embodied JSON failed to emit TextPeer, use a dedicated, simple
                    # LLM composition call and then the same auditable TextBus.
                    contract = parse_task_contract(
                        task,
                        self_label=label,
                        peer_labels=[self.labels[i] for i in sorted(self.labels) if i != agent_id],
                    ) if task else None
                    if contract is not None and contract.is_text and not sent_this_turn:
                        peer_id = self.text_bus.recipient_id(contract.recipient or "", sender_id=agent_id)
                        if peer_id is None:
                            peer_id = self._other_agent_id(agent_id)
                        if peer_id is not None:
                            body, origin = self._compose_peer_text(
                                agent_id, peer_id, task=task, decision=decision,
                                received_texts=texts, required_by_task=True,
                            )
                            if body:
                                event = self._deliver_peer_text(
                                    agent_id, peer_id, body, origin=origin, count_for_task=True
                                )
                                sent_this_turn = sent_this_turn or event is not None

                    # Optional periodic public peer channel. This is explicitly
                    # configured, visible, and carries only a message composed from
                    # the sender's own public state. The receiving LLM gets it on its
                    # next turn through texts_since_last_turn.
                    self._peer_text_turns[agent_id] = self._peer_text_turns.get(agent_id, 0) + 1
                    if (
                        self.peer_text_every > 0
                        and not sent_this_turn
                        and self._peer_text_turns[agent_id] % self.peer_text_every == 0
                    ):
                        peer_id = self._other_agent_id(agent_id)
                        if peer_id is not None:
                            body, origin = self._compose_peer_text(
                                agent_id, peer_id, task=task, decision=decision,
                                received_texts=texts, required_by_task=False,
                            )
                            if body:
                                self._deliver_peer_text(
                                    agent_id, peer_id, body, origin=origin, count_for_task=False
                                )

                    if task:
                        self._complete_task_if_needed(agent_id, claimed=decision.task_complete)

            print("[social-session] Round budget exhausted.", flush=True)
            if self.dashboard is not None:
                self.dashboard.append_event("session", "Round budget exhausted.")
            return 0
        finally:
            if self.dashboard is not None:
                self.dashboard.stop()
